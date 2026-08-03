"""Shared synthetic fixtures for public tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcc_survival.constants import TARGET_NAME
from hcc_survival.schemas import FEATURE_NAMES, FEATURE_SPECS


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return deterministic synthetic HCC-shaped features and one-year survival labels."""

    random = np.random.default_rng(7)
    frame = pd.DataFrame(index=range(60))
    for spec in FEATURE_SPECS:
        if spec.categories is None:
            frame[spec.name] = random.normal(size=len(frame))
        else:
            frame[spec.name] = random.choice(spec.categories, size=len(frame))
    frame.loc[0, "Age"] = np.nan
    target = pd.Series(
        (random.random(len(frame)) > 0.45).astype(int),
        name=TARGET_NAME,
    )
    return frame.loc[:, FEATURE_NAMES], target
