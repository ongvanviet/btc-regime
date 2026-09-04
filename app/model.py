# -*- coding: utf-8 -*-
"""
Mô hình risk regime cho Bitcoin — tái tạo theo mô tả công khai,
KHÔNG dùng thuật toán gốc của bất kỳ sản phẩm thương mại nào.

Toàn bộ tham số chỉnh được nằm ngay dưới đây.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# THAM SỐ MÔ HÌNH (chỉnh ở đây)
# ---------------------------------------------------------------------------
W_LONG = 1.2          # trọng số khung dài hạn
W_MID = 1.0           # trọng số khung trung hạn
W_SHORT = 0.8         # trọng số khung ngắn hạn

ENTER_STRONG = 1.8    # ngưỡng vào trạng thái Strong
ENTER_MILD = 0.3      # ngưỡng vào trạng thái Mild
BUFFER = 0.4          # đệm hysteresis chống đảo trạng thái liên tục

# Phân bổ BTC theo mã trạng thái
ALLOC = {2: 1.00, 1: 0.60, -1: 0.25, -2: 0.00}

FEE = 0.001           # phí 0.1% trên mỗi đơn vị thay đổi tỷ lệ phân bổ

HISTORY_YEARS = 4     # số năm lịch sử trả về cho frontend
N_INFLECTIONS = 15    # số điểm đảo chiều Risk-On/Off gần nhất

# Thông tin hiển thị cho từng trạng thái
STATE_INFO = {
    2:  {"name": "Strong-On",  "label": "Euphoria",     "risk": "Risk-On"},
    1:  {"name": "Mild-On",    "label": "Accumulation", "risk": "Risk-On"},
    -1: {"name": "Mild-Off",   "label": "Caution",      "risk": "Risk-Off"},
    -2: {"name": "Strong-Off", "label": "Capitulation", "risk": "Risk-Off"},
}


# ---------------------------------------------------------------------------
# TẢI DỮ LIỆU GIÁ
# ---------------------------------------------------------------------------
def load_prices() -> pd.Series:
    """Trả về chuỗi giá đóng cửa ngày (index DatetimeIndex tăng dần).

    Thứ tự ưu tiên: BTC_CSV (offline/test) → yfinance → CoinGecko → Binance.
    (CoinGecko free đôi khi chỉ trả 365 ngày — không đủ cho rolling window,
    nên cần tối thiểu MIN_ROWS dòng; Binance là lưới an toàn cuối cùng.)
    """
    csv_path = os.environ.get("BTC_CSV")
    if csv_path:
        return _load_csv(csv_path)

    MIN_ROWS = 600  # cần đủ cho SMA200 + z-score 365 ngày sau dropna
    errors = []
    for loader in (_load_yfinance, _load_coingecko, _load_binance):
        try:
            s = loader()
            if len(s) >= MIN_ROWS:
                return s
            errors.append(f"{loader.__name__}: chỉ có {len(s)} dòng")
        except Exception as e:
            errors.append(f"{loader.__name__}: {type(e).__name__}: {e}")

    raise RuntimeError("Mọi nguồn dữ liệu giá đều lỗi — " + " | ".join(errors))


def _load_csv(path: str) -> pd.Series:
    # CSV cần 2 cột: Date, Close
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    s = df.set_index("Date")["Close"].astype(float).sort_index()
    s.name = "close"
    return s


def _load_yfinance() -> pd.Series:
    import yfinance as yf

    df = yf.download("BTC-USD", start="2015-01-01", interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance trả về rỗng")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance mới trả MultiIndex cột
        close = close.iloc[:, 0]
    s = close.dropna().astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = "close"
    return s


def _load_coingecko() -> pd.Series:
    import requests

    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    r = requests.get(url, params={"vs_currency": "usd", "days": "max",
                                  "interval": "daily"}, timeout=30)
    r.raise_for_status()
    prices = r.json()["prices"]  # [[ms, price], ...]
    idx = pd.to_datetime([p[0] for p in prices], unit="ms").normalize()
    s = pd.Series([float(p[1]) for p in prices], index=idx, name="close")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _load_binance() -> pd.Series:
    """Nến ngày BTCUSDT từ Binance, phân trang 1000 nến/lần (từ 2017-08)."""
    import requests

    url = "https://api.binance.com/api/v3/klines"
    start = int(pd.Timestamp("2017-08-01").timestamp() * 1000)
    rows: list[tuple[int, float]] = []
    while True:
        r = requests.get(url, params={"symbol": "BTCUSDT", "interval": "1d",
                                      "startTime": start, "limit": 1000},
                         timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend((k[0], float(k[4])) for k in batch)  # k[4] = close
        if len(batch) < 1000:
            break
        start = batch[-1][0] + 1

    idx = pd.to_datetime([t for t, _ in rows], unit="ms").normalize()
    s = pd.Series([c for _, c in rows], index=idx, name="close")
    return s[~s.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------------------
# TÍNH ĐẶC TRƯNG 3 KHUNG THỜI GIAN
# ---------------------------------------------------------------------------
def build_features(close: pd.Series) -> pd.DataFrame:
    """Tính long/mid/short/score cùng các chỉ số phụ; dropna() cuối cùng."""
    df = pd.DataFrame({"close": close.astype(float)})

    # --- Khung dài hạn: cấu trúc xu hướng ---
    sma200 = df["close"].rolling(200).mean()
    df["mayer"] = df["close"] / sma200
    sma200_slope = sma200.pct_change(20)
    direction = np.where(df["mayer"] > 1, 1.0, -1.0)
    # slope cùng dấu với (mayer − 1) → xu hướng "thuận", hệ số 1.0; ngược lại 0.5
    aligned = np.sign(sma200_slope) == np.sign(df["mayer"] - 1)
    df["long"] = direction * np.where(aligned, 1.0, 0.5)

    # --- Khung trung hạn: momentum ---
    roc30 = df["close"].pct_change(30)
    mu = roc30.rolling(365).mean()
    sd = roc30.rolling(365).std()
    df["roc30_z"] = (roc30 - mu) / sd
    sma50 = df["close"].rolling(50).mean()
    golden = np.where(sma50 > sma200, 1.0, -1.0)
    df["mid"] = df["roc30_z"].clip(-1, 1) * 0.6 + golden * 0.4

    # --- Khung ngắn hạn: sốc biến động ---
    logret = np.log(df["close"] / df["close"].shift(1))
    vol7 = logret.rolling(7).std() * np.sqrt(365)
    vol90 = logret.rolling(90).std() * np.sqrt(365)
    df["vol_ratio"] = vol7 / vol90
    df["roc7"] = df["close"].pct_change(7)
    shock = (df["vol_ratio"] - 1).clip(0, 1)
    df["short"] = np.sign(df["roc7"]) * (0.5 + 0.5 * shock)

    # --- Tổng hợp ---
    df["score"] = W_LONG * df["long"] + W_MID * df["mid"] + W_SHORT * df["short"]

    return df.dropna()


# ---------------------------------------------------------------------------
# PHÂN LOẠI TRẠNG THÁI CÓ HYSTERESIS
# ---------------------------------------------------------------------------
def classify(score: pd.Series) -> pd.Series:
    """Duyệt tuần tự theo ngày, trả chuỗi mã trạng thái {2, 1, -1, -2}."""
    values = score.to_numpy()
    states = np.empty(len(values), dtype=int)

    state = 1 if values[0] > 0 else -1
    for i, sc in enumerate(values):
        if state == 2:
            if sc < ENTER_STRONG - BUFFER:
                state = 1
        elif state == 1:
            if sc > ENTER_STRONG:
                state = 2
            elif sc < -ENTER_MILD - BUFFER:
                state = -1
        elif state == -1:
            if sc > ENTER_MILD + BUFFER:
                state = 1
            elif sc < -ENTER_STRONG:
                state = -2
        else:  # state == -2
            if sc > -ENTER_STRONG + BUFFER:
                state = -1
        states[i] = state

    return pd.Series(states, index=score.index, name="state")


def alloc_for(state: pd.Series) -> pd.Series:
    """Ánh xạ mã trạng thái → tỷ lệ nắm giữ BTC."""
    return state.map(ALLOC).rename("alloc")


# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------
def backtest(close: pd.Series, alloc: pd.Series) -> dict:
    """Tín hiệu ngày t áp cho lợi nhuận ngày t+1; phí trên thay đổi phân bổ."""
    ret = close.pct_change().fillna(0.0)
    pos = alloc.shift(1).fillna(alloc.iloc[0])          # không nhìn trước
    fee_cost = pos.diff().abs().fillna(0.0) * FEE
    strat_ret = pos * ret - fee_cost

    equity_strat = (1 + strat_ret).cumprod()
    equity_hold = (1 + ret).cumprod()

    return {
        "from": str(close.index[0].date()),
        "vector": _perf_stats(equity_strat, strat_ret),
        "hold": _perf_stats(equity_hold, ret),
        "_equity_strat": equity_strat,   # nội bộ, không đưa vào JSON
        "_equity_hold": equity_hold,
    }


def _perf_stats(equity: pd.Series, ret: pd.Series) -> dict:
    n_years = len(equity) / 365.0
    multiple = float(equity.iloc[-1] / equity.iloc[0])
    cagr = multiple ** (1 / n_years) - 1 if n_years > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    sd = ret.std()
    sharpe = float(ret.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0
    return {
        "cagr": round(float(cagr), 4),
        "max_drawdown": round(float(drawdown.min()), 4),
        "sharpe": round(sharpe, 2),
        "multiple": round(multiple, 2),
    }


# ---------------------------------------------------------------------------
# KẾT QUẢ TỔNG HỢP CHO API
# ---------------------------------------------------------------------------
def compute(close: Optional[pd.Series] = None) -> dict:
    """Tính toàn bộ mô hình và trả về dict JSON-serializable."""
    if close is None:
        close = load_prices()

    feats = build_features(close)
    state = classify(feats["score"])
    alloc = alloc_for(state)
    bt = backtest(feats["close"], alloc)

    last = feats.iloc[-1]
    code = int(state.iloc[-1])
    info = STATE_INFO[code]

    # Ngày bắt đầu trạng thái hiện tại
    changed = state != state.shift(1)
    since_idx = state.index[changed][-1] if changed.any() else state.index[0]

    # Các điểm đảo chiều Risk-On/Risk-Off (đổi dấu trạng thái)
    risk_sign = np.sign(state)
    flips = risk_sign != risk_sign.shift(1)
    flips.iloc[0] = False
    inflections = []
    for ts in state.index[flips][::-1][:N_INFLECTIONS]:
        c = int(state.loc[ts])
        inflections.append({
            "date": str(ts.date()),
            "risk": STATE_INFO[c]["risk"],
            "state": STATE_INFO[c]["name"],
            "price": round(float(feats.loc[ts, "close"]), 2),
        })

    # Lịch sử 4 năm gần nhất, equity chuẩn hoá về 1 tại đầu kỳ
    cutoff = feats.index[-1] - pd.DateOffset(years=HISTORY_YEARS)
    hist_idx = feats.index[feats.index >= cutoff]
    eq_s = bt["_equity_strat"].loc[hist_idx]
    eq_h = bt["_equity_hold"].loc[hist_idx]
    history = {
        "dates": [str(d.date()) for d in hist_idx],
        "close": [round(float(v), 2) for v in feats.loc[hist_idx, "close"]],
        "state": [int(v) for v in state.loc[hist_idx]],
        "score": [round(float(v), 3) for v in feats.loc[hist_idx, "score"]],
        "equity_strat": [round(float(v / eq_s.iloc[0]), 4) for v in eq_s],
        "equity_hold": [round(float(v / eq_h.iloc[0]), 4) for v in eq_h],
    }

    return {
        "as_of": str(feats.index[-1].date()),
        "price": round(float(last["close"]), 2),
        "state": {
            "code": code,
            "name": info["name"],
            "label": info["label"],
            "risk": info["risk"],
            "alloc": ALLOC[code],
        },
        "score": {
            "total": round(float(last["score"]), 3),
            "long": round(float(last["long"]), 3),
            "mid": round(float(last["mid"]), 3),
            "short": round(float(last["short"]), 3),
        },
        "metrics": {
            "mayer": round(float(last["mayer"]), 3),
            "vol_ratio": round(float(last["vol_ratio"]), 3),
            "roc30_z": round(float(last["roc30_z"]), 3),
            "roc7": round(float(last["roc7"]), 4),
        },
        "regime_since": str(since_idx.date()),
        "regime_days": int((feats.index[-1] - since_idx).days),
        "backtest": {
            "from": bt["from"],
            "vector": bt["vector"],
            "hold": bt["hold"],
        },
        "inflections": inflections,
        "history": history,
    }
