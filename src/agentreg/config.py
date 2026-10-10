"""agentreg.yaml - where the agent lives and how strict the suite is."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from agentreg import pricing

DEFAULT_CONFIG = "agentreg.yaml"


class JudgeConfig(BaseModel):
    enabled: bool = True
    model: str = "openai:gpt-4.1-mini"


class Config(BaseModel):
    target: str = Field(description="Import path of the agent, 'module:attribute'.")
    pythonpath: list[str] = Field(default_factory=list)
    input_key: str | None = Field(
        default=None, description="Wrap a non-dict case input as {input_key: input}."
    )
    output_path: str | None = Field(
        default=None, description="Where the final answer lives, e.g. 'messages[-1].content'."
    )
    cases_dir: str = "cases"
    traces_dir: str = "traces"
    reports_dir: str = "reports"
    budget_tolerance: float = Field(1.5, description="Allowed growth of tokens, cost and steps.")
    latency_tolerance: float = Field(3.0, description="Latency is noisy, so allow more headroom.")
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    pricing: dict[str, list[float]] = Field(default_factory=dict)

    root: Path = Field(default=Path("."), exclude=True)

    def path(self, value: str) -> Path:
        return (self.root / value).resolve()


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No config at {path}. Create one - see README 'Configuration'.")
    cfg = Config.model_validate(yaml.safe_load(path.read_text()) or {})
    cfg.root = path.resolve().parent
    pricing.set_prices(cfg.pricing)
    return cfg
