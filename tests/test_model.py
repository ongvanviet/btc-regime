# -*- coding: utf-8 -*-
"""Test mô hình trên dữ liệu tổng hợp — không gọi mạng."""
import json

import numpy as np
import pandas as pd
import pytest

from app import model


@pytest.fixture(scope="module")
def close():
    # Random walk ~3000 ngày, seed cố định để test tái lập được
    rng = np.random.default_rng(42)
    n = 3000
    logret = rng.normal(loc=0.0008, scale=0.035, size=n)
    prices = 1000 * np.exp(np.cumsum(logret))
    idx = pd.date_range("2015-01-01", periods=n, freq="D")
    return pd.Series(prices, index=idx, name="close")


@pytest.fixture(scope="module")
def csv_path(close, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "btc.csv"
    close.rename("Close").rename_axis("Date").reset_index().to_csv(path, index=False)
    return str(path)


def test_build_features_no_nan(close):
    feats = model.build_features(close)
    assert not feats.isna().any().any()
    assert len(feats) > 2000  # mất tối đa ~400 ngày đầu do rolling window
    for col in ["long", "mid", "short", "score", "mayer", "vol_ratio"]:
        assert col in feats.columns


def test_classify_valid_states(close):
    feats = model.build_features(close)
    state = model.classify(feats["score"])
    assert set(state.unique()) <= {2, 1, -1, -2}
    assert len(state) == len(feats)


def test_alloc_matches_table(close):
    feats = model.build_features(close)
    state = model.classify(feats["score"])
    alloc = model.alloc_for(state)
    for code, pct in model.ALLOC.items():
        mask = state == code
        if mask.any():
            assert (alloc[mask] == pct).all()


def test_compute_keys_and_json(close, csv_path, monkeypatch):
    monkeypatch.setenv("BTC_CSV", csv_path)
    result = model.compute()

    expected = {"as_of", "price", "state", "score", "metrics", "regime_since",
                "regime_days", "backtest", "inflections", "history"}
    assert expected <= set(result.keys())
    assert result["state"]["code"] in (2, 1, -1, -2)
    assert result["state"]["alloc"] == model.ALLOC[result["state"]["code"]]
    for side in ("vector", "hold"):
        assert {"cagr", "max_drawdown", "sharpe", "multiple"} <= set(
            result["backtest"][side])

    # JSON-serializable, không lẫn kiểu numpy
    json.dumps(result)


def test_equity_starts_at_one(close, csv_path, monkeypatch):
    monkeypatch.setenv("BTC_CSV", csv_path)
    result = model.compute()
    hist = result["history"]
    assert hist["equity_strat"][0] == 1.0
    assert hist["equity_hold"][0] == 1.0
    assert len(hist["dates"]) == len(hist["close"]) == len(hist["state"]) \
        == len(hist["score"]) == len(hist["equity_strat"]) == len(hist["equity_hold"])
