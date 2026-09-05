# -*- coding: utf-8 -*-
"""Test lớp dữ liệu bền vững và cảnh báo — không gọi mạng."""
import json

import numpy as np
import pandas as pd
import pytest

from app import alerts, data


def _series(start, n, base=100.0):
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.Series(base + np.arange(n, dtype=float), index=idx, name="close")


# ---------------------------------------------------------------------------
# data: ghi/đọc, merge, incremental
# ---------------------------------------------------------------------------
def test_write_read_roundtrip(tmp_path):
    s = _series("2020-01-01", 50)
    path = tmp_path / "p.csv"
    data.write_series(s, path)
    back = data.read_series(path)
    pd.testing.assert_series_equal(s, back, check_freq=False, check_names=False)
    assert back.index.name is None or back.index.name == "Date"


def test_merge_new_overrides_old():
    old = _series("2020-01-01", 10, base=0)
    new = _series("2020-01-08", 5, base=1000)
    m = data.merge(old, new)
    assert len(m) == 12
    assert m.loc["2020-01-08"] == 1000.0
    assert m.loc["2020-01-01"] == 0.0
    assert data.merge(None, None).empty


def test_seed_file_exists_and_is_long_enough():
    assert data.SEED_PATH.exists()
    s = data.read_series(data.SEED_PATH)
    assert len(s) > data.MIN_ROWS
    assert s.index.is_monotonic_increasing and not s.index.duplicated().any()


def test_load_prices_incremental_only_fetches_tail(tmp_path, monkeypatch):
    """Có file trên đĩa → chỉ tải từ (ngày cuối − REFETCH_DAYS); nến hôm nay
    được trả về nhưng không ghi xuống đĩa."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("BTC_CSV", raising=False)
    today = data.today_utc()
    stored = _series(today - pd.Timedelta(days=800), 790)   # kết thúc 10 ngày trước
    data.write_series(stored, tmp_path / "prices.csv")

    asked = {}

    def fake_fetch(start):
        asked["start"] = start
        return _series(start, (today - start).days + 1, base=5000)   # gồm cả hôm nay

    monkeypatch.setattr(data, "PRICE_SOURCES", [("fake", fake_fetch)])
    close, meta = data.load_prices()

    assert asked["start"] == stored.index[-1] - pd.Timedelta(days=data.REFETCH_DAYS)
    assert close.index[-1] == today
    assert meta["source"] == "fake" and meta["stale"] is False and meta["has_intraday"]
    on_disk = data.read_series(tmp_path / "prices.csv")
    assert on_disk.index[-1] == today - pd.Timedelta(days=1)   # không lưu nến hôm nay
    assert on_disk.loc[stored.index[-1]] == pytest.approx(5000 + 5)  # ngày trùng bị đè


def test_load_prices_falls_back_to_seed_when_all_sources_fail(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("BTC_CSV", raising=False)

    def boom(start):
        raise RuntimeError("mạng hỏng")
    monkeypatch.setattr(data, "PRICE_SOURCES", [("a", boom), ("b", boom)])
    close, meta = data.load_prices()
    assert len(close) > data.MIN_ROWS
    assert meta["origin"] == "seed" and meta["stale"] is True
    assert len(meta["errors"]) == 2
    assert not (tmp_path / "prices.csv").exists()   # không ghi khi không có gì mới


def test_load_extra_uses_cache_on_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    s = _series("2021-01-01", 20).rename("mvrv")
    data.write_series(s, tmp_path / "mvrv.csv", val_col="Value")

    def boom(start):
        raise RuntimeError("api lỗi")
    out = data.load_extra("mvrv", boom, incremental=True)
    assert out is not None and len(out) == 20
    assert data.load_extra("dxy", lambda: (_ for _ in ()).throw(RuntimeError()), incremental=False) is None


# ---------------------------------------------------------------------------
# alerts
# ---------------------------------------------------------------------------
def _result(code, name, risk):
    return {"as_of": "2026-01-02", "price": 50000.0,
            "state": {"code": code, "name": name, "risk": risk,
                      "alloc": {2: 1, 1: .6, -1: .25, -2: 0}[code]},
            "score": {"total": 1.0, "long": 1.0, "mid": 0.5, "short": 0.2}}


def test_should_notify_rules():
    on = {"code": 1, "risk": "Risk-On"}
    assert alerts.should_notify(None, on) is False
    assert alerts.should_notify({"code": 2, "risk": "Risk-On"}, on, "risk") is False
    assert alerts.should_notify({"code": 2, "risk": "Risk-On"}, on, "state") is True
    assert alerts.should_notify({"code": -1, "risk": "Risk-Off"}, on) is True


def test_maybe_notify_sends_once_per_change(tmp_path, monkeypatch):
    path = tmp_path / "last_state.json"
    sent = []
    monkeypatch.setattr(alerts, "send_telegram", lambda text: sent.append(text) or True)
    monkeypatch.setattr(alerts, "send_email", lambda s, b: False)
    monkeypatch.setenv("ALERT_ENABLED", "1")

    r1 = alerts.maybe_notify(_result(1, "Mild-On", "Risk-On"), path=path)
    assert r1["sent"] is False and json.loads(path.read_text(encoding="utf-8"))["code"] == 1

    r2 = alerts.maybe_notify(_result(-1, "Mild-Off", "Risk-Off"), path=path)
    assert r2["sent"] is True and r2["channels"] == ["telegram"]
    assert "Mild-Off" in sent[0] and "Risk-Off" in sent[0]

    r3 = alerts.maybe_notify(_result(-1, "Mild-Off", "Risk-Off"), path=path)
    assert r3["sent"] is False and len(sent) == 1


def test_maybe_notify_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("ALERT_ENABLED", raising=False)
    r = alerts.maybe_notify(_result(1, "Mild-On", "Risk-On"), path=tmp_path / "s.json")
    assert r["sent"] is False and not (tmp_path / "s.json").exists()
