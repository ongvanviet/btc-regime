# -*- coding: utf-8 -*-
"""Test API bằng TestClient — thay lớp dữ liệu bằng chuỗi giả, không gọi mạng."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def fake_close():
    rng = np.random.default_rng(1)
    n = 2000
    prices = 1000 * np.exp(np.cumsum(rng.normal(0.0008, 0.035, n)))
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    return pd.Series(prices, index=idx, name="close")


@pytest.fixture
def reset_cache():
    """Đưa cache về trạng thái ban đầu trước và sau mỗi test."""
    def _reset():
        with main._lock:
            main._cache.update({"data": None, "inputs": None, "ts": 0.0, "error": None,
                                "error_ts": 0.0, "refreshing": False})
        main._backtest_cached.cache_clear()
    _reset()
    yield
    _reset()


@pytest.fixture
def client(fake_close, reset_cache, monkeypatch):
    """Client với load_all giả và refresh chạy đồng bộ để test tất định."""
    monkeypatch.setattr(main.data, "load_all",
                        lambda: (fake_close, {}, {"source": "fake"}))
    monkeypatch.setenv("ALERT_ENABLED", "0")

    class _SyncThread:
        def __init__(self, target, daemon=True):
            self.target = target
        def start(self):
            self.target()
    monkeypatch.setattr(main.threading, "Thread", _SyncThread)
    with TestClient(main.app) as c:
        yield c


def test_503_before_data(reset_cache, monkeypatch):
    """Chưa có cache và refresh chưa xong → 503 kèm Retry-After."""
    monkeypatch.setattr(main.threading, "Thread",
                        lambda target, daemon=True: type("T", (), {"start": lambda self: None})())
    with TestClient(main.app) as c:
        r = c.get("/api/read")
        assert r.status_code == 503
        assert r.headers["Retry-After"] == "5"
        h = c.get("/health")
        assert h.status_code == 200 and h.json()["cached"] is False


def test_read_excludes_history_and_full_includes(client):
    r = client.get("/api/read")
    assert r.status_code == 200
    body = r.json()
    assert "history" not in body
    assert body["meta"]["source"] == "fake"
    assert r.headers["Cache-Control"].startswith("public, max-age=")

    f = client.get("/api/full")
    assert "history" in f.json()
    h = client.get("/api/history")
    assert set(h.json()["history"]) >= {"dates", "close", "state", "dd_strat"}


def test_gzip_when_accepted(client):
    r = client.get("/api/full", headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("Content-Encoding") == "gzip"


def test_history_csv(client):
    r = client.get("/api/history.csv")
    assert r.status_code == 200
    assert r.headers["Content-Type"].startswith("text/csv")
    lines = r.text.strip().splitlines()
    assert lines[0] == "date,close,state,state_name,score,equity_strat,equity_hold"
    assert len(lines) > 100


def test_backtest_endpoint_and_bounds(client):
    r = client.get("/api/backtest", params={"smooth_days": 1, "confirm_days": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["result"]["params"]["smooth_days"] == 1
    assert {"vector", "hold", "regime"} <= set(body["result"])
    assert body["baseline_params"]["smooth_days"] == main.model.DEFAULT_PARAMS.smooth_days
    # Không làm mượt → đổi trạng thái nhiều hơn mặc định
    assert body["result"]["regime"]["state_changes"] >= body["baseline"]["regime"]["state_changes"]

    bad = client.get("/api/backtest", params={"smooth_days": 999})
    assert bad.status_code == 400


def test_health_ok_and_stale(client):
    h = client.get("/health")
    assert h.status_code == 200 and h.json()["ok"] is True
    with main._lock:
        main._cache["ts"] = 0.0     # giả lập dữ liệu quá cũ
    h2 = client.get("/health")
    assert h2.status_code == 503 and h2.json()["ok"] is False


def test_refresh_keeps_old_data_on_error(client, monkeypatch):
    """Nguồn lỗi ở lần refresh sau → giữ cache cũ, ghi nhận error."""
    def boom():
        raise RuntimeError("nguồn hỏng")
    monkeypatch.setattr(main.data, "load_all", boom)
    main._refresh()
    r = client.get("/api/read")
    assert r.status_code == 200
    h = client.get("/health").json()
    assert "nguồn hỏng" in h["error"]


def test_only_one_refresh_at_a_time(reset_cache, monkeypatch, fake_close):
    calls = []

    def slow_load():
        calls.append(1)
        return fake_close, {}, {"source": "fake"}
    monkeypatch.setattr(main.data, "load_all", slow_load)
    with main._lock:
        main._cache["refreshing"] = True    # giả lập đang có thread refresh
    main._refresh()                          # phải thoát ngay, không tải
    assert calls == []
