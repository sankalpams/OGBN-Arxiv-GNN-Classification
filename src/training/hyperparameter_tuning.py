from __future__ import annotations

import itertools
import pandas as pd


def parameter_grid(grid: dict[str, list]) -> list[dict]:
    """Create explicit trial configurations for transparent, small-scale tuning."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def results_frame(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results).sort_values("validation_accuracy", ascending=False).reset_index(drop=True)
