"""Serve recorded tool outputs during replay (`--tools stub`).

Stubbing isolates the change under test: if you edit a prompt, you want to know
whether the *agent* still behaves, not whether a live API returned something
new today. We wrap `_run` / `_arun` on every BaseTool subclass. The wrapper is a
no-op unless a StubStore is active in the current context, so agent code is
never modified and normal runs are unaffected.
"""

from __future__ import annotations

import functools
import json
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from langchain_core.tools import BaseTool

from agentreg.recorder import _active as _active_recorder
from agentreg.schema import ToolStub

_store: ContextVar["StubStore | None"] = ContextVar("agentreg_stubs", default=None)
_patched: set[type] = set()
_patch_lock = threading.Lock()
_RUNTIME_KWARGS = {"run_manager", "config", "callbacks"}


class StubStore:
    def __init__(self, stubs: list[ToolStub]) -> None:
        self._stubs = stubs
        self._lock = threading.Lock()
        self.hits: list[tuple[str, str]] = []
        self.misses: list[str] = []

    def lookup(self, tool: BaseTool, args: dict[str, Any]) -> ToolStub | None:
        key = _key(tool, args)
        for stub in self._stubs:
            if stub.name == tool.name and _key(tool, stub.args) == key:
                with self._lock:
                    self.hits.append((tool.name, key))
                return stub
        with self._lock:
            self.misses.append(f"{tool.name}({json.dumps(args, sort_keys=True, default=str)})")
        return None


def _key(tool: BaseTool, args: dict[str, Any]) -> str:
    """Normalise args so that omitted defaults and explicit defaults compare equal."""
    defaults = {
        name: spec["default"]
        for name, spec in (tool.args or {}).items()
        if isinstance(spec, dict) and "default" in spec
    }
    kept = {k: v for k, v in args.items() if not (k in defaults and defaults[k] == v)}
    return json.dumps(kept, sort_keys=True, default=str)


def _call_args(args: tuple, kwargs: dict) -> dict[str, Any]:
    clean = {k: v for k, v in kwargs.items() if k not in _RUNTIME_KWARGS}
    if args and not clean:
        clean = {"input": args[0]} if len(args) == 1 else {"args": list(args)}
    return clean


def _stub_result(tool: BaseTool, stub: ToolStub, kwargs: dict) -> Any:
    run_manager = kwargs.get("run_manager")
    recorder = _active_recorder.get()
    if run_manager is not None and recorder is not None:
        recorder.mark_stubbed(run_manager.run_id)
    if tool.response_format == "content_and_artifact":
        return stub.output, stub.artifact
    return stub.output


def _wrap_sync(original):
    @functools.wraps(original)  # keeps the signature LangChain inspects for run_manager/config
    def _run(self, *args, **kwargs):
        store = _store.get()
        if store is not None:
            stub = store.lookup(self, _call_args(args, kwargs))
            if stub is not None:
                return _stub_result(self, stub, kwargs)
        return original(self, *args, **kwargs)  # no store, or a miss: run the real tool

    return _run


def _wrap_async(original):
    @functools.wraps(original)
    async def _arun(self, *args, **kwargs):
        store = _store.get()
        if store is not None:
            stub = store.lookup(self, _call_args(args, kwargs))
            if stub is not None:
                return _stub_result(self, stub, kwargs)
        return await original(self, *args, **kwargs)

    return _arun


def _subclasses(cls: type) -> set[type]:
    out: set[type] = set()
    for sub in cls.__subclasses__():
        out.add(sub)
        out |= _subclasses(sub)
    return out


def install() -> None:
    """Wrap every currently-defined tool class once. Safe to call repeatedly."""
    with _patch_lock:
        for cls in _subclasses(BaseTool):
            if cls in _patched:
                continue
            if "_run" in cls.__dict__:
                cls._run = _wrap_sync(cls.__dict__["_run"])
            if "_arun" in cls.__dict__:
                cls._arun = _wrap_async(cls.__dict__["_arun"])
            _patched.add(cls)


@contextmanager
def stubbing(stubs: list[ToolStub]) -> Iterator[StubStore]:
    install()
    store = StubStore(stubs)
    token = _store.set(store)
    try:
        yield store
    finally:
        _store.reset(token)
