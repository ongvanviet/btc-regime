# -*- coding: utf-8 -*-
"""Test mô hình trên dữ liệu tổng hợp — không gọi mạng."""
import json

import numpy as np
import pandas as pd
import pytest

from app import model
from app.model import Params


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
def extras(close):
    """MVRV giả (giá / SMA365) và DXY giả chỉ có ngày làm việc."""
    mvrv = (close / close.rolling(365).mean()).dropna().rename("mvrv")
    rng = np.random.default_rng(7)
    bdays = pd.bdate_range(close.index[0], close.index[-1])
    dxy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.003, len(bdays)))),
                    index=bdays, name="dxy")
    return {"mvrv": mvrv, "dxy": dxy}


@pytest.fixture(scope="module")
def csv_path(close, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "btc.csv"
    close.rename("Close").rename_axis("Date").reset_index().to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------
def test_build_features_no_nan(close):
    feats = model.build_features(close)
    core = ["long", "mid", "short", "score", "score_raw", "mayer", "vol_ratio"]
    assert not feats[core].isna().any().any()
    assert len(feats) > 2000  # mất tối đa ~400 ngày đầu do rolling window
    # Không có extras → onchain/macro toàn NaN nhưng score vẫn đủ
    assert feats["onchain"].isna().all() and feats["macro"].isna().all()


def test_score_scale_is_pm3(close, extras):
    for ex in ({}, extras):
        feats = model.build_features(close, ex)
        assert feats["score_raw"].abs().max() <= 3.0 + 1e-9
        assert feats["score"].abs().max() <= 3.0 + 1e-9


def test_extras_change_score_but_keep_scale(close, extras):
    a = model.build_features(close)
    b = model.build_features(close, extras)
    common = a.index.intersection(b.index)
    assert not np.allclose(a.loc[common, "score"], b.loc[common, "score"])
    assert b.loc[common, "onchain"].notna().mean() > 0.8
    assert b.loc[common, "macro"].notna().mean() > 0.9


def test_missing_extra_component_is_renormalized(close, extras):
    """Bỏ w_onchain/w_macro = 0 phải cho đúng kết quả như không có extras."""
    p = Params(w_onchain=0, w_macro=0)
    a = model.build_features(close, None, p)
    b = model.build_features(close, extras, p)
    pd.testing.assert_series_equal(a["score"], b["score"].loc[a.index])


def test_short_component_is_continuous(close):
    feats = model.build_features(close, None, Params(roc7_full=0.10))
    # với ROC7 nhỏ, |short| phải nhỏ hơn 1 (không còn nhảy ±0.5 như sign)
    small = feats[feats["roc7"].abs() < 0.01]
    assert (small["short"].abs() < 0.2).all()


# ---------------------------------------------------------------------------
# classify: hysteresis + xác nhận
# ---------------------------------------------------------------------------
def _states(scores, **kw):
    idx = pd.date_range("2020-01-01", periods=len(scores), freq="D")
    return model.classify(pd.Series(scores, index=idx), Params(**kw)).tolist()


def test_classify_valid_states(close):
    feats = model.build_features(close)
    state = model.classify(feats["score"])
    assert set(state.unique()) <= {2, 1, -1, -2}
    assert len(state) == len(feats)


def test_hysteresis_no_flip_inside_buffer():
    # Dao động 1.7 ↔ 1.9 quanh ENTER_STRONG=1.8: vào Strong rồi không rơi ra
    # vì ngưỡng ra là 1.8 - 0.4 = 1.4
    s = _states([0.5, 1.9, 1.9, 1.7, 1.9, 1.7, 1.7, 1.7], confirm_days=1)
    assert s == [1, 2, 2, 2, 2, 2, 2, 2]


def test_confirm_days_blocks_one_day_spike():
    # Một ngày vượt ngưỡng không đủ; hai ngày liên tiếp mới đổi
    s = _states([0.5, 1.9, 0.5, 0.5, 1.9, 1.9, 0.5], confirm_days=2)
    assert s == [1, 1, 1, 1, 1, 2, 2]


def test_confirm_streak_resets_on_different_target():
    # 1 → đòi lên 2 (1 ngày) → đòi xuống -1 (1 ngày): không chuỗi nào đủ 2 ngày
    s = _states([0.5, 1.9, -0.9, 1.9, -0.9], confirm_days=2)
    assert s == [1, 1, 1, 1, 1]


def test_strong_off_path():
    s = _states([-0.5, -1.9, -1.9, -1.0, -1.0, 0.8, 0.8], confirm_days=2)
    assert s == [-1, -1, -2, -2, -1, -1, 1]


def test_smoothing_reduces_changes(close):
    p_raw = Params(smooth_days=1, confirm_days=1)
    p_smooth = Params(smooth_days=7, confirm_days=2)
    n_raw = model.regime_stats(model.classify(model.build_features(close, None, p_raw)["score"], p_raw))
    n_sm = model.regime_stats(model.classify(model.build_features(close, None, p_smooth)["score"], p_smooth))
    assert n_sm["state_changes"] < n_raw["state_changes"]
    assert n_sm["median_regime_days"] > n_raw["median_regime_days"]


# ---------------------------------------------------------------------------
# alloc + backtest
# ---------------------------------------------------------------------------
def test_alloc_matches_table(close):
    feats = model.build_features(close)
    state = model.classify(feats["score"])
    alloc = model.alloc_for(state)
    for code, pct in model.ALLOC.items():
        mask = state == code
        if mask.any():
            assert (alloc[mask] == pct).all()


def test_backtest_no_lookahead(close):
    """Đổi phân bổ ngày t chỉ ảnh hưởng lợi nhuận từ ngày t+1."""
    idx = close.index[:10]
    c = pd.Series([100, 110, 121, 121, 121, 121, 121, 121, 121, 121], index=idx, dtype=float)
    alloc = pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], index=idx, dtype=float)
    bt = model.backtest(c, alloc, fee=0.0)
    eq = bt["_equity_strat"]
    # Ngày 1 tăng 10% nhưng vị thế áp dụng là của ngày 0 (=0) → không lời
    assert eq.iloc[1] == pytest.approx(1.0)
    # Ngày 2 tăng 10% với vị thế ngày 1 (=1) → +10%
    assert eq.iloc[2] == pytest.approx(1.1)


def test_backtest_yearly_and_fees(close):
    feats = model.build_features(close)
    state = model.classify(feats["score"])
    bt = model.backtest(feats["close"], model.alloc_for(state))
    years = [r["year"] for r in bt["yearly"]]
    assert years == sorted(set(feats.index.year))
    assert bt["total_fees"] >= 0


def test_regime_stats_shares_sum_to_one(close):
    feats = model.build_features(close)
    stats = model.regime_stats(model.classify(feats["score"]))
    assert sum(stats["time_in_state"].values()) == pytest.approx(1.0, abs=1e-3)
    assert stats["state_changes"] >= 0


# ---------------------------------------------------------------------------
# params_from_dict
# ---------------------------------------------------------------------------
def test_params_from_dict_bounds():
    p = model.params_from_dict({"buffer": "0.6", "smooth_days": "3", "bogus": 1})
    assert p.buffer == 0.6 and p.smooth_days == 3 and p.confirm_days == model.DEFAULT_PARAMS.confirm_days
    with pytest.raises(ValueError):
        model.params_from_dict({"smooth_days": 999})


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def test_compute_keys_and_json(close, extras):
    result = model.compute(close, extras, {"source": "test"})

    expected = {"as_of", "price", "state", "score", "weights", "metrics", "regime_since",
                "regime_days", "backtest", "inflections", "intraday", "params", "meta",
                "history"}
    assert expected <= set(result.keys())
    assert result["state"]["code"] in (2, 1, -1, -2)
    assert result["state"]["alloc"] == model.ALLOC[result["state"]["code"]]
    for side in ("vector", "hold"):
        assert {"cagr", "max_drawdown", "sharpe", "multiple"} <= set(result["backtest"][side])
    assert {"yearly", "total_fees", "regime"} <= set(result["backtest"])
    assert result["meta"]["source"] == "test"
    assert result["metrics"]["mvrv"] is not None
    # Dữ liệu tổng hợp kết thúc năm 2023 → không có nến hôm nay → không intraday
    assert result["intraday"] is None
    json.dumps(result)   # JSON-serializable, không lẫn kiểu numpy


def test_compute_csv_env(csv_path, monkeypatch):
    """Đường BTC_CSV (offline) qua lớp dữ liệu, không gọi mạng."""
    monkeypatch.setenv("BTC_CSV", csv_path)
    monkeypatch.setenv("DISABLE_EXTRAS", "1")
    result = model.compute()
    assert result["meta"]["source"] == "csv"
    assert result["score"]["onchain"] is None


def test_intraday_split(close):
    """Nến hôm nay (UTC) phải bị tách khỏi trạng thái chính thức."""
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    idx = pd.date_range(end=today, periods=len(close), freq="D")
    c = pd.Series(close.to_numpy(), index=idx, name="close")
    result = model.compute(c)
    assert result["as_of"] == str((today - pd.Timedelta(days=1)).date())
    assert result["intraday"] is not None
    assert result["intraday"]["date"] == str(today.date())
    assert result["history"]["dates"][-1] == result["as_of"]


def test_equity_starts_at_one(close):
    result = model.compute(close)
    hist = result["history"]
    assert hist["equity_strat"][0] == 1.0
    assert hist["equity_hold"][0] == 1.0
    assert hist["dd_strat"][0] == 0.0
    n = len(hist["dates"])
    for k in ("close", "state", "score", "equity_strat", "equity_hold", "dd_strat", "dd_hold"):
        assert len(hist[k]) == n
