"""Turn a 'module:attribute' target into a function: case input -> final output."""

from __future__ import annotations

import importlib
import re
import sys
from typing import Any, Callable

from agentreg.config import Config

_TOKEN = re.compile(r"\.?([A-Za-z_]\w*)|\[(-?\d+)\]")


def load_target(cfg: Config) -> Callable[[Any], Any]:
    for entry in reversed(cfg.pythonpath):
        p = str(cfg.path(entry))
        if p not in sys.path:
            sys.path.insert(0, p)

    module_name, _, attr = cfg.target.partition(":")
    if not attr:
        raise ValueError(f"target must look like 'module:attribute', got {cfg.target!r}")
    obj = getattr(importlib.import_module(module_name), attr)

    if hasattr(obj, "invoke"):  # LangChain Runnable / compiled LangGraph graph
        call = obj.invoke
    elif callable(obj):
        call = obj
    else:
        raise TypeError(f"{cfg.target} is neither a Runnable nor a callable")

    def run(case_input: Any) -> Any:
        payload = case_input
        if cfg.input_key and not isinstance(case_input, dict):
            payload = {cfg.input_key: case_input}
        return extract(call(payload), cfg.output_path)

    return run


def extract(value: Any, path: str | None) -> Any:
    """Follow a path like 'messages[-1].content' through dicts, lists and attributes."""
    if not path:
        return value
    pos = 0
    for match in _TOKEN.finditer(path):
        if match.start() != pos:
            raise ValueError(f"Bad output_path {path!r} near {path[pos:]!r}")
        pos = match.end()
        key, index = match.groups()
        if index is not None:
            value = value[int(index)]
        elif isinstance(value, dict):
            value = value[key]
        else:
            value = getattr(value, key)
    if pos != len(path):
        raise ValueError(f"Bad output_path {path!r} near {path[pos:]!r}")
    return value
