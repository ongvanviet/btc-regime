# -*- coding: utf-8 -*-
"""FastAPI server cho RiskonBTC: cache trong bộ nhớ, refresh nền duy nhất,
gzip, cache header, endpoint so sánh tham số và hook cảnh báo đổi trạng thái."""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app import alerts, data, model

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("riskonbtc")

CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))
STALE_AFTER_SECONDS = int(os.environ.get("STALE_AFTER_SECONDS", str(24 * 3600)))
API_MAX_AGE = int(os.environ.get("API_MAX_AGE", "300"))   # Cache-Control cho /api/*
STATIC_DIR = Path(__file__).parent / "static"

# Cache đơn giản trong bộ nhớ, bảo vệ bằng lock vì tính toán chạy ở thread nền.
# `inputs` giữ (close, extras) để endpoint /api/backtest tái sử dụng không tải lại.
_cache: dict = {"data": None, "inputs": None, "ts": 0.0, "error": None,
                "error_ts": 0.0, "refreshing": False}
_lock = threading.Lock()


def _refresh() -> None:
    """Tính lại mô hình và cập nhật cache; lưu lỗi nếu thất bại.

    Chỉ một thread refresh chạy tại một thời điểm (cờ `refreshing`) để nhiều
    request đến cùng lúc sau cold start không tạo ra hàng loạt lần tải dữ liệu.
    """
    with _lock:
        if _cache["refreshing"]:
            return
        _cache["refreshing"] = True
    t0 = time.time()
    try:
        close, extras, meta = data.load_all()
        result = model.compute(close, extras, meta)
        with _lock:
            _cache["data"] = result
            _cache["inputs"] = (close, extras)
            _cache["ts"] = time.time()
            _cache["error"] = None
        _backtest_cached.cache_clear()
        log.info("Refresh xong sau %.1fs — trạng thái %s (%s), nguồn %s",
                 time.time() - t0, result["state"]["name"], result["as_of"],
                 meta.get("source"))
        try:
            alerts.maybe_notify(result)
        except Exception as e:   # cảnh báo lỗi không được làm hỏng dữ liệu
            log.warning("Gửi cảnh báo thất bại: %s", e)
    except Exception as e:  # giữ cache cũ nếu có, chỉ ghi nhận lỗi
        log.exception("Refresh thất bại")
        with _lock:
            _cache["error"] = f"{type(e).__name__}: {e}"
            _cache["error_ts"] = time.time()
    finally:
        with _lock:
            _cache["refreshing"] = False


def _get_data() -> Optional[dict]:
    """Trả dữ liệu cache; nếu hết TTL thì kích hoạt tính lại ở nền (một lần)."""
    with _lock:
        cached = _cache["data"]
        expired = time.time() - _cache["ts"] > CACHE_TTL_SECONDS
        busy = _cache["refreshing"]
    if (cached is None or expired) and not busy:
        threading.Thread(target=_refresh, daemon=True).start()
    return cached


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tính lần đầu ở nền để request đầu tiên không bị treo
    threading.Thread(target=_refresh, daemon=True).start()
    yield


app = FastAPI(title="RiskonBTC", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _unavailable() -> JSONResponse:
    with _lock:
        err = _cache["error"]
    detail = "Dữ liệu đang được tính lần đầu. Vui lòng thử lại sau vài giây."
    if err:
        detail = "Nguồn dữ liệu giá đang lỗi: " + err
    return JSONResponse(status_code=503, content={"detail": detail},
                        headers={"Cache-Control": "no-store", "Retry-After": "5"})


def _json(payload: dict) -> JSONResponse:
    return JSONResponse(payload, headers={
        "Cache-Control": f"public, max-age={API_MAX_AGE}, stale-while-revalidate=600"})


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html",
                        headers={"Cache-Control": "no-cache"})


@app.get("/api/read")
def api_read():
    cached = _get_data()
    if cached is None:
        return _unavailable()
    return _json({k: v for k, v in cached.items() if k != "history"})


@app.get("/api/history")
def api_history():
    cached = _get_data()
    if cached is None:
        return _unavailable()
    return _json({"history": cached["history"]})


@app.get("/api/full")
def api_full():
    cached = _get_data()
    if cached is None:
        return _unavailable()
    return _json(cached)


@app.get("/api/history.csv")
def api_history_csv():
    """Xuất lịch sử trạng thái dạng CSV để tải về."""
    cached = _get_data()
    if cached is None:
        return _unavailable()
    h = cached["history"]
    lines = ["date,close,state,state_name,score,equity_strat,equity_hold"]
    for i, d in enumerate(h["dates"]):
        lines.append(f'{d},{h["close"][i]},{h["state"][i]},'
                     f'{model.STATE_INFO[h["state"][i]]["name"]},{h["score"][i]},'
                     f'{h["equity_strat"][i]},{h["equity_hold"][i]}')
    return Response("\n".join(lines) + "\n", media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=riskonbtc_history.csv",
                             "Cache-Control": f"public, max-age={API_MAX_AGE}"})


@lru_cache(maxsize=64)
def _backtest_cached(key: tuple) -> dict:
    """Backtest theo bộ tham số; cache theo key = tuple đã sắp xếp. Bị xoá khi refresh."""
    p = model.params_from_dict(dict(key))
    with _lock:
        inputs = _cache["inputs"]
    close, extras = inputs
    settled = close[close.index < data.today_utc()]
    if len(settled) < 400:
        settled = close
    return model.run_backtest(settled, extras, p)


@app.get("/api/backtest")
def api_backtest(
    enter_strong: Optional[float] = Query(None), enter_mild: Optional[float] = Query(None),
    buffer: Optional[float] = Query(None), smooth_days: Optional[int] = Query(None),
    confirm_days: Optional[int] = Query(None),
    w_long: Optional[float] = Query(None), w_mid: Optional[float] = Query(None),
    w_short: Optional[float] = Query(None), w_onchain: Optional[float] = Query(None),
    w_macro: Optional[float] = Query(None),
):
    """So sánh tham số: chạy lại mô hình với tham số truyền qua query string."""
    cached = _get_data()
    if cached is None:
        return _unavailable()
    raw = {"enter_strong": enter_strong, "enter_mild": enter_mild, "buffer": buffer,
           "smooth_days": smooth_days, "confirm_days": confirm_days,
           "w_long": w_long, "w_mid": w_mid, "w_short": w_short,
           "w_onchain": w_onchain, "w_macro": w_macro}
    key = tuple(sorted((k, v) for k, v in raw.items() if v is not None))
    try:
        result = _backtest_cached(key)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    return _json({"result": result, "baseline": cached["backtest"],
                  "baseline_params": cached["params"]})


@app.get("/health")
def health():
    """ok=false khi chưa từng có dữ liệu mà đã lỗi, hoặc dữ liệu cũ quá 24 giờ."""
    with _lock:
        has = _cache["data"] is not None
        age = time.time() - _cache["ts"] if has else None
        err = _cache["error"]
        refreshing = _cache["refreshing"]
    ok = True
    if has and age is not None and age > STALE_AFTER_SECONDS:
        ok = False
    if not has and err and not refreshing:
        ok = False
    body = {"ok": ok, "cached": has, "age_seconds": int(age) if age is not None else None,
            "refreshing": refreshing, "error": err}
    return JSONResponse(body, status_code=200 if ok else 503,
                        headers={"Cache-Control": "no-store"})
