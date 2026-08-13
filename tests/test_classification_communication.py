from __future__ import annotations

import numpy as np
import pytest

from solana_sniper_reverse_engineering.modeling import metrics


def test_rare_class_metrics_include_enrichment_without_replacing_raw_values() -> None:
    y = np.array([1, 0, 0, 0, 0], dtype=np.uint8)
    score = np.array([0.9, 0.8, 0.2, 0.1, 0.0])
    result = metrics(y, score, threshold=0.85)
    assert result["prevalence"] == pytest.approx(0.2)
    assert result["precision"] == pytest.approx(1.0)
    assert result["precision_lift_over_prevalence"] == pytest.approx(5.0)
    assert result["pr_auc_lift_over_prevalence"] == pytest.approx(
        result["pr_auc"] / result["prevalence"]
    )
