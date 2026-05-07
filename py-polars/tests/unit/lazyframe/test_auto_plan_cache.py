from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal


def test_auto_plan_cache_keeps_collect_transparent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLARS_AUTO_PLAN_CACHE", "1")
    dim = pl.DataFrame({"k": [1, 2], "label": ["one", "two"]})

    def run(facts: pl.DataFrame) -> pl.DataFrame:
        return (
            facts.lazy()
            .join(dim.lazy(), on="k", how="left")
            .with_columns((pl.col("v") * 2).alias("v2"))
            .filter(pl.col("v2") > 10)
            .sort("k")
            .collect()
        )

    out1 = run(pl.DataFrame({"k": [1, 2], "v": [3, 8]}))
    out2 = run(pl.DataFrame({"k": [1, 2], "v": [9, 4]}))

    assert_frame_equal(
        out1,
        pl.DataFrame({"k": [2], "v": [8], "label": ["two"], "v2": [16]}),
    )
    assert_frame_equal(
        out2,
        pl.DataFrame({"k": [1], "v": [9], "label": ["one"], "v2": [18]}),
    )
