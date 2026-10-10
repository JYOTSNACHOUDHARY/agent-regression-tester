"""Record agent runs as Traces.

The recorder is a LangChain callback handler. It is attached through a
configure hook bound to a context variable, so every LangChain / LangGraph
component that runs inside `recording(...)` reports to it - the agent under
test does not have to pass callbacks around or change any code.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from langchain_core.outputs import LLMResult
from langchain_core.tracers.context import register_configure_hook

from agentreg import pricing
from agentreg.schema import LLMCall, ToolCall, Trace

_active: ContextVar["Recorder | None"] = ContextVar("agentreg_recorder", default=None)
register_configure_hook(_active, inheritable=True)


class Recorder(BaseCallbackHandler):
    """Collects LLM and tool steps. Thread-safe: LangGraph runs tools in parallel."""

    raise_error = False

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._pending: dict[UUID, tuple[int, float, Any]] = {}
        self._steps: list[tuple[int, LLMCall | ToolCall]] = []

    # ------------------------------------------------------------------ helpers

    def _start(self, run_id: UUID, step: Any) -> None:
        with self._lock:
            self._seq += 1
            self._pending[run_id] = (self._seq, time.perf_counter(), step)

    def _finish(self, run_id: UUID) -> tuple[Any, float] | None:
        with self._lock:
            entry = self._pending.pop(run_id, None)
        if entry is None:
            return None
        seq, started, step = entry
        step.latency_s = round(time.perf_counter() - started, 4)
        with self._lock:
            self._steps.append((seq, step))
        return step, step.latency_s

    def mark_stubbed(self, run_id: UUID) -> None:
        with self._lock:
            entry = self._pending.get(run_id)
        if entry is not None and isinstance(entry[2], ToolCall):
            entry[2].stubbed = True

    @property
    def steps(self) -> list[LLMCall | ToolCall]:
        with self._lock:
            return [s for _, s in sorted(self._steps, key=lambda x: x[0])]

    # ------------------------------------------------------------------ LLM events

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        self._start(run_id, LLMCall(model=_model_name(kwargs, metadata)))

    def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, **kwargs):
        self._start(run_id, LLMCall(model=_model_name(kwargs, metadata)))

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs):
        with self._lock:
            entry = self._pending.get(run_id)
        if entry is None:
            return
        call: LLMCall = entry[2]
        llm_output = response.llm_output or {}
        call.model = llm_output.get("model_name") or llm_output.get("model") or call.model
        inp, out = _usage(response)
        call.input_tokens, call.output_tokens = inp, out
        call.requested_tools = _requested_tools(response)
        call.cost_usd = pricing.cost(call.model, inp, out)
        self._finish(run_id)

    def on_llm_error(self, error, *, run_id, **kwargs):
        with self._lock:
            entry = self._pending.get(run_id)
        if entry is not None:
            entry[2].error = f"{type(error).__name__}: {error}"
            self._finish(run_id)

    # ------------------------------------------------------------------ tool events

    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs):
        name = (serialized or {}).get("name") or kwargs.get("name") or "unknown_tool"
        args = dict(inputs) if isinstance(inputs, dict) else {"input": input_str}
        self._start(run_id, ToolCall(name=name, args=args))

    def on_tool_end(self, output, *, run_id, **kwargs):
        with self._lock:
            entry = self._pending.get(run_id)
        if entry is None:
            return
        call: ToolCall = entry[2]
        if isinstance(output, ToolMessage):
            call.output = _text(output.content)
            call.artifact = _jsonable(output.artifact)
            if output.status == "error":
                call.error = call.output
        else:
            call.output = _text(output)
        self._finish(run_id)

    def on_tool_error(self, error, *, run_id, **kwargs):
        with self._lock:
            entry = self._pending.get(run_id)
        if entry is not None:
            entry[2].error = f"{type(error).__name__}: {error}"
            self._finish(run_id)


@contextmanager
def recording() -> Iterator[Recorder]:
    """Attach a fresh Recorder to every LangChain run inside the block."""
    recorder = Recorder()
    token = _active.set(recorder)
    try:
        yield recorder
    finally:
        _active.reset(token)


def record_run(target_name: str, run: Any, payload: Any, case_input: Any) -> Trace:
    """Execute `run(payload)` under a recorder and return the resulting Trace.

    `run` returns the agent's final output. Exceptions are captured in the trace
    rather than raised, because a crashing agent is itself a regression signal.
    """
    trace = Trace(id=uuid.uuid4().hex[:12], target=target_name, input=case_input)
    started = time.perf_counter()
    with recording() as recorder:
        try:
            trace.output = _jsonable(run(payload))
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            trace.error = f"{type(exc).__name__}: {exc}"
    trace.latency_s = round(time.perf_counter() - started, 4)
    trace.steps = recorder.steps
    return trace


# ---------------------------------------------------------------------- utilities


def _model_name(kwargs: dict, metadata: dict | None) -> str | None:
    params = kwargs.get("invocation_params") or {}
    return (
        params.get("model")
        or params.get("model_name")
        or (metadata or {}).get("ls_model_name")
    )


def _usage(response: LLMResult) -> tuple[int, int]:
    inp = out = 0
    for gens in response.generations:
        for gen in gens:
            usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
            if usage:
                inp += usage.get("input_tokens", 0)
                out += usage.get("output_tokens", 0)
    if inp or out:
        return inp, out
    token_usage = (response.llm_output or {}).get("token_usage") or {}
    return token_usage.get("prompt_tokens", 0), token_usage.get("completion_tokens", 0)


def _requested_tools(response: LLMResult) -> list[str]:
    names: list[str] = []
    for gens in response.generations:
        for gen in gens:
            for tc in getattr(getattr(gen, "message", None), "tool_calls", None) or []:
                names.append(tc["name"])
    return names


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):  # content blocks
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in value)
    return str(value)


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of agent outputs and artifacts to JSON-safe data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "content"):
        return _text(value.content)
    return str(value)
