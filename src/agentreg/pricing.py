"""Token prices used to turn usage into cost.

Prices are USD per 1M tokens (input, output). Providers change prices, so
treat these as defaults and override them in agentreg.yaml under `pricing:`.
"""

from __future__ import annotations

DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}

_prices: dict[str, tuple[float, float]] = dict(DEFAULT_PRICES)


def set_prices(overrides: dict[str, tuple[float, float] | list[float]]) -> None:
    for model, (inp, out) in overrides.items():
        _prices[model] = (float(inp), float(out))


def lookup(model: str | None) -> tuple[float, float] | None:
    """Find prices by longest matching prefix, so dated model ids resolve too."""
    if not model:
        return None
    name = model.split(":", 1)[-1].lower()
    matches = [k for k in _prices if name.startswith(k)]
    return _prices[max(matches, key=len)] if matches else None


def cost(model: str | None, input_tokens: int, output_tokens: int) -> float | None:
    prices = lookup(model)
    if prices is None:
        return None
    return round((input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000, 6)
