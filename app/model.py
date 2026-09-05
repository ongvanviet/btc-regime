# -*- coding: utf-8 -*-
"""
Mô hình risk regime cho Bitcoin — tái tạo theo mô tả công khai,
KHÔNG dùng thuật toán gốc của bất kỳ sản phẩm thương mại nào.

Cấu trúc:
  Params          — toàn bộ tham số chỉnh được (mặc định = DEFAULT_PARAMS)
  build_features  — tính 5 thành phần: long / mid / short / onchain / macro
  classify        — phân loại 4 trạng thái có hysteresis + xác nhận N ngày
  backtest        — tín hiệu ngày t áp cho lợi nhuận ngày t+1, có phí
  compute         — gom tất cả thành dict JSON cho API
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# THAM SỐ MÔ HÌNH
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    # Trọng số 5 thành phần. Điểm tổng được chuẩn hoá về thang ±3 bất kể
    # tổng trọng số, nên thêm/bớt thành phần không làm lệch ngưỡng.
    w_long: float = 1.2       # dài hạn: Mayer multiple + độ dốc SMA200
    w_mid: float = 1.0        # trung hạn: z-score ROC30 + golden cross
    w_short: float = 0.8      # ngắn hạn: ROC7 nhân sốc biến động
    w_onchain: float = 0.6    # on-chain: z-score MVRV (Coin Metrics)
    w_macro: float = 0.4      # vĩ mô: sức mạnh USD (DXY từ FRED), dấu âm

    enter_strong: float = 1.8  # ngưỡng vào Strong
    enter_mild: float = 0.3    # ngưỡng vào Mild
    buffer: float = 0.4        # đệm hysteresis

    smooth_days: int = 7       # EMA làm mượt điểm tổng (1 = không làm mượt)
    confirm_days: int = 2      # số ngày liên tiếp vượt ngưỡng mới đổi trạng thái

    roc7_full: float = 0.10    # |ROC7| đạt 10% coi là tín hiệu ngắn hạn đủ mạnh
    fee: float = 0.001         # phí 0,1% trên mỗi đơn vị thay đổi phân bổ

    def weights(self) -> dict[str, float]:
        return {"long": self.w_long, "mid": self.w_mid, "short": self.w_short,
                "onchain": self.w_onchain, "macro": self.w_macro}


DEFAULT_PARAMS = Params()

# Giới hạn hợp lệ cho endpoint /api/backtest (chống giá trị vô lý)
PARAM_BOUNDS = {
    "enter_strong": (0.5, 3.0), "enter_mild": (0.0, 2.0), "buffer": (0.0, 1.5),
    "smooth_days": (1, 30), "confirm_days": (1, 15),
    "w_long": (0.0, 3.0), "w_mid": (0.0, 3.0), "w_short": (0.0, 3.0),
    "w_onchain": (0.0, 3.0), "w_macro": (0.0, 3.0),
}

SCORE_SCALE = 3.0     # điểm tổng nằm trong [-3, +3]
ALLOC = {2: 1.00, 1: 0.60, -1: 0.25, -2: 0.00}   # phân bổ BTC theo trạng thái
HISTORY_YEARS = 4     # số năm lịch sử trả về cho frontend
N_INFLECTIONS = 15    # số điểm đảo chiều Risk-On/Off gần nhất

STATE_INFO = {
    2:  {"name": "Strong-On",  "label": "Expansion",    "risk": "Risk-On"},
    1:  {"name": "Mild-On",    "label": "Accumulation", "risk": "Risk-On"},
    -1: {"name": "Mild-Off",   "label": "Caution",      "risk": "Risk-Off"},
    -2: {"name": "Strong-Off", "label": "Capitulation", "risk": "Risk-Off"},
}


def params_from_dict(d: dict) -> Params:
    """Tạo Params từ dict (ví dụ query string), kiểm tra giới hạn."""
    clean = {}
    for k, v in d.items():
        if k not in PARAM_BOUNDS or v is None:
            continue
        lo, hi = PARAM_BOUNDS[k]
        val = int(v) if isinstance(lo, int) else float(v)
        if not (lo <= val <= hi):
            raise ValueError(f"{k} phải nằm trong [{lo}, {hi}]")
        clean[k] = val
    return replace(DEFAULT_PARAMS, **clean)


# ---------------------------------------------------------------------------
# TÍNH ĐẶC TRƯNG
# ---------------------------------------------------------------------------
def _zscore(s: pd.Series, window: int = 365) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - mu) / sd.replace(0.0, np.nan)


def build_features(close: pd.Series, extras: Optional[dict] = None,
                   p: Params = DEFAULT_PARAMS) -> pd.DataFrame:
    """Tính các thành phần và điểm tổng.

    extras: dict tuỳ chọn {"mvrv": Series, "dxy": Series} (index ngày).
    Thành phần thiếu dữ liệu sẽ bị loại khỏi tổng và trọng số được chuẩn hoá
    lại, nên điểm luôn ở thang ±3. Cột `score_raw` là điểm trước khi làm mượt.
    """
    extras = extras or {}
    df = pd.DataFrame({"close": close.astype(float)})

    # --- Dài hạn: cấu trúc xu hướng ---
    sma200 = df["close"].rolling(200).mean()
    df["mayer"] = df["close"] / sma200
    sma200_slope = sma200.pct_change(20)
    direction = np.where(df["mayer"] > 1, 1.0, -1.0)
    aligned = np.sign(sma200_slope) == np.sign(df["mayer"] - 1)
    df["long"] = direction * np.where(aligned, 1.0, 0.5)

    # --- Trung hạn: momentum ---
    roc30 = df["close"].pct_change(30)
    df["roc30_z"] = _zscore(roc30)
    sma50 = df["close"].rolling(50).mean()
    golden = np.where(sma50 > sma200, 1.0, -1.0)
    df["mid"] = df["roc30_z"].clip(-1, 1) * 0.6 + golden * 0.4

    # --- Ngắn hạn: ROC7 liên tục (thay cho sign) nhân sốc biến động ---
    logret = np.log(df["close"] / df["close"].shift(1))
    vol7 = logret.rolling(7).std() * np.sqrt(365)
    vol90 = logret.rolling(90).std() * np.sqrt(365)
    df["vol_ratio"] = vol7 / vol90
    df["roc7"] = df["close"].pct_change(7)
    shock = (df["vol_ratio"] - 1).clip(0, 1)
    df["short"] = (df["roc7"] / p.roc7_full).clip(-1, 1) * (0.5 + 0.5 * shock)

    # Các cột bắt buộc phải đủ dữ liệu
    core_cols = ["mayer", "roc30_z", "vol_ratio", "roc7", "long", "mid", "short"]
    df = df.dropna(subset=core_cols)

    # --- On-chain: MVRV (tuỳ chọn) ---
    mvrv = extras.get("mvrv")
    if mvrv is not None and len(mvrv) > 0:
        m = mvrv.reindex(close.index).ffill(limit=7)   # cho phép trễ vài ngày
        z = _zscore(m)
        oc = (z / 2).clip(-1, 1)
        oc = oc.where(m <= 3.0, oc - 0.5).clip(-1, 1)   # phạt khi quá nóng
        df["mvrv"] = m.reindex(df.index)
        df["onchain"] = oc.reindex(df.index)
    else:
        df["mvrv"] = np.nan
        df["onchain"] = np.nan

    # --- Vĩ mô: USD mạnh → bất lợi cho BTC (tuỳ chọn) ---
    dxy = extras.get("dxy")
    if dxy is not None and len(dxy) > 0:
        d = dxy.reindex(close.index).ffill(limit=15)    # DXY chỉ có ngày làm việc, FRED trễ ~1 tuần
        z = _zscore(d.pct_change(20))
        df["dxy"] = d.reindex(df.index)
        df["macro"] = (-(z / 2).clip(-1, 1)).reindex(df.index)
    else:
        df["dxy"] = np.nan
        df["macro"] = np.nan

    # --- Tổng hợp: trung bình có trọng số trên các thành phần có dữ liệu ---
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for col, w in p.weights().items():
        if w <= 0:
            continue
        avail = df[col].notna()
        num = num + df[col].fillna(0.0) * w
        den = den + avail.astype(float) * w
    df["score_raw"] = SCORE_SCALE * num / den.replace(0.0, np.nan)
    if p.smooth_days > 1:
        df["score"] = df["score_raw"].ewm(span=p.smooth_days, adjust=False).mean()
    else:
        df["score"] = df["score_raw"]

    return df.dropna(subset=["score"])


# ---------------------------------------------------------------------------
# PHÂN LOẠI TRẠNG THÁI: HYSTERESIS + XÁC NHẬN N NGÀY
# ---------------------------------------------------------------------------
def classify(score: pd.Series, p: Params = DEFAULT_PARAMS) -> pd.Series:
    """Duyệt tuần tự theo ngày, trả chuỗi mã trạng thái {2, 1, -1, -2}.

    Một chuyển trạng thái chỉ xảy ra khi điều kiện vượt ngưỡng đúng
    `confirm_days` ngày liên tiếp; đảm bảo không nhìn trước.
    """
    values = score.to_numpy()
    states = np.empty(len(values), dtype=int)
    state = 1 if values[0] > 0 else -1
    pending: Optional[int] = None   # trạng thái đang chờ xác nhận
    streak = 0

    for i, sc in enumerate(values):
        target = _target_state(state, sc, p)
        if target == state:
            pending, streak = None, 0
        else:
            if target == pending:
                streak += 1
            else:
                pending, streak = target, 1
            if streak >= p.confirm_days:
                state = target
                pending, streak = None, 0
        states[i] = state

    return pd.Series(states, index=score.index, name="state")


def _target_state(state: int, sc: float, p: Params) -> int:
    """Trạng thái mà điểm `sc` đang muốn chuyển sang, từ trạng thái hiện tại."""
    if state == 2:
        return 1 if sc < p.enter_strong - p.buffer else 2
    if state == 1:
        if sc > p.enter_strong:
            return 2
        return -1 if sc < -p.enter_mild - p.buffer else 1
    if state == -1:
        if sc > p.enter_mild + p.buffer:
            return 1
        return -2 if sc < -p.enter_strong else -1
    return -1 if sc > -p.enter_strong + p.buffer else -2


def alloc_for(state: pd.Series) -> pd.Series:
    """Ánh xạ mã trạng thái → tỷ lệ nắm giữ BTC."""
    return state.map(ALLOC).rename("alloc")


# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------
def backtest(close: pd.Series, alloc: pd.Series, fee: float = DEFAULT_PARAMS.fee) -> dict:
    """Tín hiệu ngày t áp cho lợi nhuận ngày t+1; phí trên thay đổi phân bổ."""
    ret = close.pct_change().fillna(0.0)
    pos = alloc.shift(1).fillna(alloc.iloc[0])          # không nhìn trước
    fee_cost = pos.diff().abs().fillna(0.0) * fee
    strat_ret = pos * ret - fee_cost

    equity_strat = (1 + strat_ret).cumprod()
    equity_hold = (1 + ret).cumprod()

    yearly = pd.DataFrame({"strat": strat_ret, "hold": ret})
    yearly = (1 + yearly).groupby(yearly.index.year).prod() - 1

    return {
        "from": str(close.index[0].date()),
        "vector": _perf_stats(equity_strat, strat_ret),
        "hold": _perf_stats(equity_hold, ret),
        "yearly": [{"year": int(y), "strat": round(float(r.strat), 4),
                    "hold": round(float(r.hold), 4)} for y, r in yearly.iterrows()],
        "total_fees": round(float(fee_cost.sum()), 4),
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


def regime_stats(state: pd.Series) -> dict:
    """Thống kê hành vi trạng thái: số lần đổi, thời gian ở mỗi trạng thái."""
    n_years = max(len(state) / 365.0, 1e-9)
    changes = int((state != state.shift(1)).sum() - 1)
    flips = int((np.sign(state) != np.sign(state).shift(1)).sum() - 1)
    runs = (state != state.shift(1)).cumsum()
    lens = state.groupby(runs).size()
    share = (state.value_counts(normalize=True)).to_dict()
    return {
        "state_changes": changes,
        "changes_per_year": round(changes / n_years, 1),
        "risk_flips": flips,
        "flips_per_year": round(flips / n_years, 1),
        "median_regime_days": int(lens.median()),
        "time_in_state": {str(c): round(float(share.get(c, 0.0)), 4) for c in (2, 1, -1, -2)},
    }


def run_backtest(close: pd.Series, extras: Optional[dict], p: Params) -> dict:
    """Backtest gọn cho endpoint so sánh tham số (không kèm history)."""
    feats = build_features(close, extras, p)
    state = classify(feats["score"], p)
    bt = backtest(feats["close"], alloc_for(state), p.fee)
    return {
        "params": asdict(p),
        "from": bt["from"],
        "vector": bt["vector"],
        "hold": bt["hold"],
        "total_fees": bt["total_fees"],
        "regime": regime_stats(state),
    }


# ---------------------------------------------------------------------------
# KẾT QUẢ TỔNG HỢP CHO API
# ---------------------------------------------------------------------------
def _settled_cutoff() -> pd.Timestamp:
    """Ngày UTC hôm nay: nến của ngày này chưa đóng nên chưa được 'chốt'."""
    return pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()


def compute(close: Optional[pd.Series] = None, extras: Optional[dict] = None,
            meta: Optional[dict] = None, p: Params = DEFAULT_PARAMS) -> dict:
    """Tính toàn bộ mô hình và trả về dict JSON-serializable.

    - Trạng thái chính thức chỉ dùng các nến ĐÃ ĐÓNG (ngày < hôm nay UTC).
    - Nếu chuỗi giá có nến hôm nay (đang chạy), tính thêm `intraday` để xem trước.
    """
    if close is None:
        from app import data as _data
        close, extras, meta = _data.load_all()
    extras = extras or {}
    meta = dict(meta or {})

    today = _settled_cutoff()
    settled = close[close.index < today]
    intraday_close = close[close.index >= today]
    if len(settled) < 400:            # chuỗi quá ngắn (test/CSV cũ): dùng tất cả
        settled, intraday_close = close, close.iloc[0:0]

    feats = build_features(settled, extras, p)
    state = classify(feats["score"], p)
    alloc = alloc_for(state)
    bt = backtest(feats["close"], alloc, p.fee)

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
    eq_s_n = eq_s / eq_s.iloc[0]
    eq_h_n = eq_h / eq_h.iloc[0]
    history = {
        "dates": [str(d.date()) for d in hist_idx],
        "close": _round_list(feats.loc[hist_idx, "close"], 2),
        "state": [int(v) for v in state.loc[hist_idx]],
        "score": _round_list(feats.loc[hist_idx, "score"], 3),
        "equity_strat": _round_list(eq_s_n, 4),
        "equity_hold": _round_list(eq_h_n, 4),
        "dd_strat": _round_list(eq_s_n / eq_s_n.cummax() - 1, 4),
        "dd_hold": _round_list(eq_h_n / eq_h_n.cummax() - 1, 4),
    }

    # Xem trước intraday: chạy lại trên chuỗi có nến hôm nay
    intraday = None
    if len(intraday_close) > 0:
        feats_i = build_features(close, extras, p)
        state_i = classify(feats_i["score"], p)
        li = feats_i.iloc[-1]
        ci = int(state_i.iloc[-1])
        intraday = {
            "date": str(feats_i.index[-1].date()),
            "price": round(float(li["close"]), 2),
            "score": round(float(li["score"]), 3),
            "state": {"code": ci, "name": STATE_INFO[ci]["name"],
                      "label": STATE_INFO[ci]["label"], "risk": STATE_INFO[ci]["risk"]},
            "differs": ci != code,
        }

    components = {k: _nan_round(last[k], 3) for k in ("long", "mid", "short", "onchain", "macro")}
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
        "score": {"total": round(float(last["score"]), 3),
                  "raw": round(float(last["score_raw"]), 3), **components},
        "weights": p.weights(),
        "metrics": {
            "mayer": round(float(last["mayer"]), 3),
            "vol_ratio": round(float(last["vol_ratio"]), 3),
            "roc30_z": round(float(last["roc30_z"]), 3),
            "roc7": round(float(last["roc7"]), 4),
            "mvrv": _nan_round(last["mvrv"], 3),
            "dxy": _nan_round(last["dxy"], 2),
        },
        "regime_since": str(since_idx.date()),
        "regime_days": int((feats.index[-1] - since_idx).days),
        "backtest": {
            "from": bt["from"],
            "vector": bt["vector"],
            "hold": bt["hold"],
            "yearly": bt["yearly"],
            "total_fees": bt["total_fees"],
            "regime": regime_stats(state),
        },
        "inflections": inflections,
        "intraday": intraday,
        "params": asdict(p),
        "meta": meta,
        "history": history,
    }


def _round_list(s: pd.Series, nd: int) -> list:
    return [round(float(v), nd) for v in s]


def _nan_round(v, nd: int):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)
