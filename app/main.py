# -*- coding: utf-8 -*-
"""FastAPI server cho btc-regime: cache trong bộ nhớ + tính nền khi khởi động."""
from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app import model

CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))
STATIC_DIR = Path(__file__).parent / "static"

# Cache đơn giản trong bộ nhớ, bảo vệ bằng lock vì tính toán chạy ở thread nền
_cache: dict = {"data": None, "ts": 0.0, "error": None}
_lock = threading.Lock()


def _refresh() -> None:
    """Tính lại mô hình và cập nhật cache; lưu lỗi nếu thất bại."""
    try:
        data = model.compute()
        with _lock:
            _cache["data"] = data
            _cache["ts"] = time.time()
            _cache["error"] = None
    except Exception as e:  # giữ cache cũ nếu có, chỉ ghi nhận lỗi
        with _lock:
            _cache["error"] = f"{type(e).__name__}: {e}"


def _get_data() -> dict | None:
    """Trả dữ liệu cache; nếu hết TTL thì kích hoạt tính lại ở nền."""
    with _lock:
        data = _cache["data"]
        expired = time.time() - _cache["ts"] > CACHE_TTL_SECONDS
    if data is None or expired:
        # Không chặn request: tính ở thread nền, request hiện tại dùng cache cũ
        threading.Thread(target=_refresh, daemon=True).start()
    return data


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tính lần đầu ở nền để request đầu tiên không bị treo
    threading.Thread(target=_refresh, daemon=True).start()
    yield


app = FastAPI(title="RiskonBTC", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_UNAVAILABLE = JSONResponse(
    status_code=503,
    content={"detail": "Dữ liệu đang được tính lần đầu hoặc nguồn giá đang lỗi. "
                       "Vui lòng thử lại sau vài giây."},
)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/read")
def api_read():
    data = _get_data()
    if data is None:
        return _UNAVAILABLE
    return {k: v for k, v in data.items() if k != "history"}


@app.get("/api/history")
def api_history():
    data = _get_data()
    if data is None:
        return _UNAVAILABLE
    return {"history": data["history"]}


@app.get("/api/full")
def api_full():
    data = _get_data()
    if data is None:
        return _UNAVAILABLE
    return data


@app.get("/health")
def health():
    with _lock:
        return {"ok": True, "cached": _cache["data"] is not None,
                "error": _cache["error"]}
