# -*- coding: utf-8 -*-
"""
Lớp dữ liệu bền vững cho RiskonBTC.

Nguyên tắc:
- Giá BTC được lưu xuống đĩa (DATA_DIR/prices.csv). Mỗi lần refresh chỉ tải
  phần còn thiếu (vài ngày gần nhất) thay vì toàn bộ lịch sử.
- Lần chạy đầu không có file → dùng file hạt giống app/seed/btc_seed.csv đi kèm
  repo, nên app luôn khởi động được kể cả khi mọi API đều lỗi.
- Chỉ lưu các nến ĐÃ ĐÓNG (ngày < hôm nay UTC). Nến hôm nay được ghép vào
  chuỗi trả về để tính "xem trước intraday" nhưng không ghi xuống đĩa.
- Dữ liệu phụ (MVRV từ Coin Metrics, DXY từ FRED) cũng được lưu và cập nhật
  tăng dần; thiếu thì mô hình tự bỏ thành phần đó.

Thứ tự nguồn giá: BTC_CSV (offline) → Binance → Coinbase → Coin Metrics → yfinance.
(CoinGecko và CryptoCompare đã yêu cầu API key nên không còn dùng.)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("riskonbtc.data")

ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = Path(__file__).resolve().parent / "seed" / "btc_seed.csv"
FIRST_DATE = "2015-01-01"
REFETCH_DAYS = 5          # tải lại vài ngày cuối để sửa nến từng lưu dở
MIN_ROWS = 600            # đủ cho SMA200 + z-score 365 ngày


def data_dir() -> Path:
    d = Path(os.environ.get("DATA_DIR", ROOT / "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=2, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "riskonbtc/1.0 (+https://github.com)"
    return s


def today_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()


# ---------------------------------------------------------------------------
# ĐỌC / GHI CSV
# ---------------------------------------------------------------------------
def read_series(path: Path, date_col: str = "Date", val_col: str = "Close") -> pd.Series:
    """Đọc CSV 2 cột thành Series index ngày, tăng dần, không trùng."""
    df = pd.read_csv(path)
    idx = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.normalize()
    s = pd.Series(df[val_col].astype(float).to_numpy(), index=idx, name="close")
    s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
    return s


def write_series(s: pd.Series, path: Path, val_col: str = "Close") -> None:
    tmp = path.with_suffix(".tmp")
    s.rename(val_col).rename_axis("Date").to_frame().to_csv(
        tmp, index=True, date_format="%Y-%m-%d")
    os.replace(tmp, path)   # ghi nguyên tử


def merge(old: Optional[pd.Series], new: Optional[pd.Series]) -> pd.Series:
    """Ghép hai chuỗi; giá trị mới đè giá trị cũ ở ngày trùng."""
    parts = [x for x in (old, new) if x is not None and len(x) > 0]
    if not parts:
        return pd.Series(dtype=float, name="close")
    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = "close"
    return s


# ---------------------------------------------------------------------------
# NGUỒN GIÁ
# ---------------------------------------------------------------------------
def fetch_binance(start: pd.Timestamp) -> pd.Series:
    """Nến ngày BTCUSDT, phân trang 1000 nến/lần (có từ 2017-08-17)."""
    s = _session()
    url = "https://api.binance.com/api/v3/klines"
    t = int(start.timestamp() * 1000)
    rows: list[tuple[int, float]] = []
    while True:
        r = s.get(url, params={"symbol": "BTCUSDT", "interval": "1d",
                               "startTime": t, "limit": 1000}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend((k[0], float(k[4])) for k in batch)
        if len(batch) < 1000:
            break
        t = batch[-1][0] + 1
    return _series_from_rows(rows, unit="ms")


def fetch_coinbase(start: pd.Timestamp) -> pd.Series:
    """Nến ngày BTC-USD từ Coinbase Exchange, tối đa 300 nến/lần."""
    s = _session()
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
    rows: list[tuple[int, float]] = []
    cur = start
    end_all = today_utc() + pd.Timedelta(days=1)
    while cur < end_all:
        nxt = min(cur + pd.Timedelta(days=299), end_all)
        r = s.get(url, params={"granularity": 86400,
                               "start": cur.strftime("%Y-%m-%dT00:00:00Z"),
                               "end": nxt.strftime("%Y-%m-%dT00:00:00Z")}, timeout=30)
        r.raise_for_status()
        rows.extend((k[0], float(k[4])) for k in r.json())   # [time, low, high, open, close, vol]
        cur = nxt
    return _series_from_rows(rows, unit="s")


def fetch_coinmetrics_price(start: pd.Timestamp) -> pd.Series:
    """PriceUSD (reference rate cuối ngày) từ Coin Metrics community API."""
    return _coinmetrics("PriceUSD", start)


def fetch_yfinance(start: pd.Timestamp) -> pd.Series:
    import yfinance as yf
    df = yf.download("BTC-USD", start=start.strftime("%Y-%m-%d"), interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance trả về rỗng (thường do rate-limit)")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    s = close.dropna().astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = "close"
    return s


PRICE_SOURCES: list[tuple[str, Callable[[pd.Timestamp], pd.Series]]] = [
    ("binance", fetch_binance),
    ("coinbase", fetch_coinbase),
    ("coinmetrics", fetch_coinmetrics_price),
    ("yfinance", fetch_yfinance),
]


def _series_from_rows(rows: list, unit: str) -> pd.Series:
    if not rows:
        return pd.Series(dtype=float, name="close")
    idx = pd.to_datetime([t for t, _ in rows], unit=unit).normalize()
    s = pd.Series([c for _, c in rows], index=idx, name="close", dtype=float)
    return s[~s.index.duplicated(keep="last")].sort_index()


def _coinmetrics(metric: str, start: pd.Timestamp) -> pd.Series:
    s = _session()
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {"assets": "btc", "metrics": metric, "frequency": "1d",
              "start_time": start.strftime("%Y-%m-%d"), "page_size": 10000}
    out: list[tuple[str, float]] = []
    while True:
        r = s.get(url, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        out.extend((d["time"][:10], float(d[metric])) for d in j.get("data", [])
                   if d.get(metric) not in (None, ""))
        nxt = j.get("next_page_token")
        if not nxt:
            break
        params["next_page_token"] = nxt
    if not out:
        return pd.Series(dtype=float, name="close")
    idx = pd.to_datetime([t for t, _ in out]).normalize()
    ser = pd.Series([v for _, v in out], index=idx, name="close", dtype=float)
    return ser[~ser.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------------------
# DỮ LIỆU PHỤ: MVRV, DXY
# ---------------------------------------------------------------------------
def fetch_mvrv(start: pd.Timestamp) -> pd.Series:
    return _coinmetrics("CapMVRVCur", start).rename("mvrv")


def fetch_dxy() -> pd.Series:
    """Chỉ số USD trade-weighted (DTWEXBGS) từ FRED, CSV không cần key."""
    s = _session()
    r = s.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
              params={"id": "DTWEXBGS"}, timeout=30)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    df.columns = ["Date", "Value"]
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")   # FRED ghi "." cho ngày nghỉ
    df = df.dropna()
    idx = pd.to_datetime(df["Date"]).dt.normalize()
    ser = pd.Series(df["Value"].to_numpy(), index=idx, name="dxy", dtype=float)
    return ser[~ser.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------------------
# TẢI TỔNG HỢP
# ---------------------------------------------------------------------------
def load_prices() -> tuple[pd.Series, dict]:
    """Trả (close, meta). close có thể chứa nến hôm nay (chưa đóng)."""
    csv_path = os.environ.get("BTC_CSV")
    if csv_path:
        s = read_series(csv_path)
        return s, {"source": "csv", "fetched_at": _now_iso(), "stale": False,
                   "rows": len(s), "errors": []}

    today = today_utc()
    prices_file = data_dir() / "prices.csv"
    stored: Optional[pd.Series] = None
    origin = "none"
    if prices_file.exists():
        try:
            stored, origin = read_series(prices_file), "disk"
        except Exception as e:
            log.warning("Không đọc được %s: %s", prices_file, e)
    if stored is None and SEED_PATH.exists():
        stored, origin = read_series(SEED_PATH), "seed"

    if stored is not None and len(stored) > 0:
        start = stored.index[-1] - pd.Timedelta(days=REFETCH_DAYS)
    else:
        start = pd.Timestamp(FIRST_DATE)

    fresh, source, errors = _fetch_first_ok(start)
    combined = merge(stored, fresh)
    if len(combined) < MIN_ROWS:
        raise RuntimeError("Không đủ dữ liệu giá (%d dòng). Lỗi nguồn: %s"
                           % (len(combined), " | ".join(errors) or "không có"))

    settled = combined[combined.index < today]
    if fresh is not None and len(fresh) > 0:
        try:
            write_series(settled, prices_file)
        except Exception as e:
            log.warning("Không ghi được %s: %s", prices_file, e)

    stale = fresh is None or len(fresh) == 0 or settled.index[-1] < today - pd.Timedelta(days=2)
    meta = {
        "source": source or origin,
        "origin": origin,
        "fetched_at": _now_iso(),
        "last_settled": str(settled.index[-1].date()),
        "has_intraday": bool((combined.index >= today).any()),
        "rows": int(len(combined)),
        "stale": bool(stale),
        "errors": errors,
    }
    log.info("Giá BTC: nguồn=%s, dòng=%d, chốt=%s, stale=%s",
             meta["source"], meta["rows"], meta["last_settled"], stale)
    return combined, meta


def _fetch_first_ok(start: pd.Timestamp) -> tuple[Optional[pd.Series], Optional[str], list[str]]:
    """Thử lần lượt các nguồn, trả nguồn đầu tiên có dữ liệu."""
    errors: list[str] = []
    full = start <= pd.Timestamp(FIRST_DATE)
    sources = PRICE_SOURCES
    if full:
        # Tải toàn bộ lịch sử: Coin Metrics có từ 2015, ghép Binance từ 2017 lên trên
        try:
            base = fetch_coinmetrics_price(start)
        except Exception as e:
            base = None
            errors.append(f"coinmetrics: {type(e).__name__}: {e}")
        for name, fn in sources:
            if name == "coinmetrics":
                continue
            try:
                s = fn(max(start, pd.Timestamp("2017-08-17")) if name == "binance" else start)
                if len(s) > 0:
                    return merge(base, s), (f"coinmetrics+{name}" if base is not None else name), errors
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__}: {e}")
        if base is not None and len(base) > 0:
            return base, "coinmetrics", errors
        return None, None, errors

    for name, fn in sources:
        try:
            s = fn(start)
            if len(s) > 0:
                return s, name, errors
            errors.append(f"{name}: rỗng")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    return None, None, errors


def load_extra(name: str, fetch: Callable[..., pd.Series], incremental: bool) -> Optional[pd.Series]:
    """Tải + lưu một chuỗi phụ; lỗi thì dùng bản trên đĩa, không có thì None."""
    path = data_dir() / f"{name}.csv"
    stored: Optional[pd.Series] = None
    if path.exists():
        try:
            stored = read_series(path, val_col="Value").rename(name)
        except Exception as e:
            log.warning("Không đọc được %s: %s", path, e)
    try:
        if incremental:
            start = (stored.index[-1] - pd.Timedelta(days=REFETCH_DAYS)) if stored is not None \
                else pd.Timestamp(FIRST_DATE)
            fresh = fetch(start)
        else:
            fresh = fetch()
        combined = merge(stored, fresh).rename(name)
        if len(fresh) > 0:
            write_series(combined, path, val_col="Value")
        return combined
    except Exception as e:
        log.warning("Không tải được %s: %s — dùng bản cache", name, e)
        return stored


def load_all() -> tuple[pd.Series, dict, dict]:
    """Trả (close, extras, meta) cho model.compute()."""
    close, meta = load_prices()
    extras: dict = {}
    if os.environ.get("DISABLE_EXTRAS", "").lower() not in ("1", "true"):
        mvrv = load_extra("mvrv", fetch_mvrv, incremental=True)
        dxy = load_extra("dxy", fetch_dxy, incremental=False)
        if mvrv is not None and len(mvrv) > 0:
            extras["mvrv"] = mvrv
        if dxy is not None and len(dxy) > 0:
            extras["dxy"] = dxy
    meta["extras"] = {k: str(v.index[-1].date()) for k, v in extras.items()}
    return close, extras, meta


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
