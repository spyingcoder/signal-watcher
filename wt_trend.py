"""
WT Trend System — Python port of the Pine v5 strategy "WT Trend ST".

Mirrors the Pine logic bar-for-bar:
  entry  : ASGMA bullish AND close > VWAP AND close > Supertrend AND ADX > min
  stop   : close - clamp(high - lastSwingLow, min*ATR, max*ATR)   [fixed version]
  trail  : ratchets to bestHigh - trailDistance on every new high
  sizing : risk% of equity / stopDistance, capped by leverage and max lot
  costs  : commission % of notional each way + slippage in ticks

Long-only by default, matching enableSellSignals = false.
"""

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────
#  Pine-equivalent indicator primitives
# ────────────────────────────────────────────────────────────────

def rma(series: pd.Series, length: int) -> pd.Series:
    """Pine ta.rma — Wilder smoothing (EMA with alpha = 1/length)."""
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    return rma(true_range(df), length)


def alma(series: pd.Series, length: int, offset: float, sigma: float) -> pd.Series:
    """Pine ta.alma."""
    m = offset * (length - 1)
    s = length / sigma
    i = np.arange(length)
    w = np.exp(-((i - m) ** 2) / (2 * s * s))
    w = w / w.sum()
    # Pine indexes series[length-1-i] with weight w[i], i.e. oldest bar gets w[0]
    vals = series.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    if len(vals) >= length:
        windows = np.lib.stride_tricks.sliding_window_view(vals, length)
        out[length - 1:] = windows @ w
    return pd.Series(out, index=series.index)


def supertrend(df: pd.DataFrame, factor: float, atr_period: int):
    """Pine ta.supertrend. Returns (value, direction) where direction 1 = down."""
    a = atr(df, atr_period).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    close = df["close"].to_numpy()
    n = len(close)

    upper = hl2 + factor * a
    lower = hl2 - factor * a
    st = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)

    for i in range(n):
        if i == 0:
            direction[i] = 1
            st[i] = upper[i]
            continue
        prev_lower = lower[i - 1]
        prev_upper = upper[i - 1]
        if not (lower[i] > prev_lower or close[i - 1] < prev_lower):
            lower[i] = prev_lower
        if not (upper[i] < prev_upper or close[i - 1] > prev_upper):
            upper[i] = prev_upper

        if np.isnan(a[i - 1]):
            direction[i] = 1
        elif st[i - 1] == prev_upper:
            direction[i] = -1 if close[i] > upper[i] else 1
        else:
            direction[i] = 1 if close[i] < lower[i] else -1
        st[i] = lower[i] if direction[i] == -1 else upper[i]

    return pd.Series(st, index=df.index), pd.Series(direction, index=df.index)


def adx(df: pd.DataFrame, di_length: int, adx_smoothing: int) -> pd.Series:
    """Pine ta.dmi — returns the ADX line only."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    trur = rma(true_range(df), di_length)
    plus = 100 * rma(pd.Series(plus_dm, index=df.index), di_length) / trur
    minus = 100 * rma(pd.Series(minus_dm, index=df.index), di_length) / trur
    total = plus + minus
    dx = (plus - minus).abs() / total.replace(0, 1)
    return 100 * rma(dx, adx_smoothing)


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Pine ta.vwap(hlc3) — cumulative, resets each UTC day."""
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.normalize()
    pv = (hlc3 * df["volume"]).groupby(day).cumsum()
    v = df["volume"].groupby(day).cumsum()
    return pv / v.replace(0, np.nan)


def asgma_state(df: pd.DataFrame, p) -> pd.Series:
    """ASGMA bullish flag: ALMA-smoothed %change vs Gaussian adaptive MA of itself."""
    src = df[p["asgma_source"]]
    pct = src.diff(p["asgma_smoothing"]) / src * 100
    spc = alma(pct, p["asgma_lookback"], 0.85, 7)

    if p["gaussian_adaptive"]:
        sigma = df["close"].rolling(p["gaussian_vol_period"]).std(ddof=0)
    else:
        sigma = pd.Series(p["gaussian_fixed_sigma"], index=df.index)

    L = p["gaussian_length"]
    spc_v = spc.to_numpy(dtype=float)
    sig_v = sigma.to_numpy(dtype=float)
    n = len(spc_v)

    # envelope[i] = highest(spc, i+1) + lowest(spc, i+1), for i = 0..L-1
    env = np.full((n, L), np.nan)
    for i in range(L):
        w = i + 1
        env[:, i] = (pd.Series(spc_v).rolling(w).max()
                     + pd.Series(spc_v).rolling(w).min()).to_numpy()

    idx = np.arange(L)
    gma = np.full(n, np.nan)
    for t in range(n):
        s = sig_v[t]
        if np.isnan(s) or s == 0 or np.isnan(env[t]).any():
            continue
        w = np.exp(-np.power((idx - (L - 1)) / (2 * s), 2) / 2)
        gma[t] = (env[t] @ w / w.sum()) / 2

    gma = ema(pd.Series(gma, index=df.index), 7)
    return spc >= gma


def swing_levels(df: pd.DataFrame, lookback: int):
    """Pine swing detection: bar[t-lookback] is a pivot vs +/- lookback neighbours."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(high)
    last_high = np.full(n, np.nan)
    last_low = np.full(n, np.nan)

    lh = np.nan
    ll = np.nan
    for t in range(n):
        c = t - lookback
        if c - lookback >= 0 and c + lookback <= t:
            seg_h = high[c - lookback: c + lookback + 1]
            seg_l = low[c - lookback: c + lookback + 1]
            if high[c] > np.delete(seg_h, lookback).max():
                lh = high[c]
            if low[c] < np.delete(seg_l, lookback).min():
                ll = low[c]
        last_high[t] = lh
        last_low[t] = ll

    return (pd.Series(last_high, index=df.index),
            pd.Series(last_low, index=df.index))


# ────────────────────────────────────────────────────────────────
#  Backtest engine
# ────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    # signals
    enable_buy=True,
    enable_sell=False,
    # chop filter
    enable_chop_filter=True,
    adx_length=14,
    adx_minimum=20.0,
    require_adx_rising=False,
    # entry filter toggles (for ablation)
    use_asgma=True,
    use_vwap=True,
    use_supertrend=True,
    # ASGMA
    asgma_source="close",
    asgma_smoothing=1,
    asgma_lookback=25,
    gaussian_length=14,
    gaussian_adaptive=True,
    gaussian_vol_period=20,
    gaussian_fixed_sigma=1.0,
    # supertrend
    st_atr_period=21,
    st_multiplier=2.0,
    # vwap
    vwap_fallback_length=20,
    # stop placement
    swing_lookback=5,
    trail_atr_length=14,
    trail_min_atr=1.5,
    trail_max_atr=4.0,
    # risk
    risk_percent=0.5,
    max_leverage=5.0,
    max_lot=0.35,
    # costs
    initial_capital=10000.0,
    commission_percent=0.02,
    slippage_ticks=2,
    mintick=0.01,
    swap_annual_percent=20.0,   # NYS: -20% annualised, both directions
    swap_rollover_hour=21,      # UTC hour the swap is charged
    swap_triple_weekday=2,      # Wednesday = 2 (Mon=0); charged 3x
    # exit mode: "trail" | "fixed"
    exit_mode="trail",
    reward_multiple=2.0,
)


def prepare(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    d = df.copy()
    d["atr"] = atr(d, p["trail_atr_length"])
    st, _ = supertrend(d, p["st_multiplier"], p["st_atr_period"])
    d["supertrend"] = st
    vw = session_vwap(d)
    d["vwap"] = vw.fillna(ema((d["high"] + d["low"] + d["close"]) / 3,
                              p["vwap_fallback_length"]))
    d["adx"] = adx(d, p["adx_length"], p["adx_length"])
    d["asgma_bull"] = asgma_state(d, p)
    lh, ll = swing_levels(d, p["swing_lookback"])
    d["last_swing_high"] = lh
    d["last_swing_low"] = ll
    return d


def swap_nights(entry_time, exit_time, hour: int, triple_weekday: int) -> float:
    """Weighted count of rollovers crossed, Wednesday counted triple."""
    if exit_time <= entry_time:
        return 0.0
    first = entry_time.normalize() + pd.Timedelta(hours=hour)
    if first <= entry_time:
        first += pd.Timedelta(days=1)
    n = 0.0
    t = first
    while t <= exit_time:
        n += 3.0 if t.weekday() == triple_weekday else 1.0
        t += pd.Timedelta(days=1)
    return n


def backtest(df: pd.DataFrame, params: dict = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    d = prepare(df, p)

    slip = p["slippage_ticks"] * p["mintick"]
    comm = p["commission_percent"] / 100.0

    equity = p["initial_capital"]
    trades = []
    equity_curve = []

    in_pos = False
    entry_price = qty = trail_dist = trail_level = best_high = target = np.nan
    entry_time = None
    active_dir = 0

    o = d["open"].to_numpy()
    h = d["high"].to_numpy()
    l = d["low"].to_numpy()
    c = d["close"].to_numpy()
    a = d["atr"].to_numpy()
    stv = d["supertrend"].to_numpy()
    vw = d["vwap"].to_numpy()
    ax = d["adx"].to_numpy()
    bull = d["asgma_bull"].to_numpy()
    sl_low = d["last_swing_low"].to_numpy()
    times = d.index

    for i in range(len(d)):
        # ── manage an open position on this bar ──
        if in_pos:
            # Pine: the stop active on this bar is the level computed on the
            # PREVIOUS bar. Check first, ratchet after.
            hit_stop = l[i] <= trail_level
            hit_tp = p["exit_mode"] == "fixed" and h[i] >= target

            if hit_stop or hit_tp:
                exit_price = (trail_level - slip) if hit_stop else target
                pnl = (exit_price - entry_price) * qty
                pnl -= comm * exit_price * qty
                nights = swap_nights(entry_time, times[i],
                                     p["swap_rollover_hour"],
                                     p["swap_triple_weekday"])
                swap = (p["swap_annual_percent"] / 100 / 365) * entry_price * qty * nights
                pnl -= swap
                equity += pnl
                trades.append(dict(
                    entry_time=entry_time, exit_time=times[i],
                    entry=entry_price, exit=exit_price, qty=qty,
                    pnl=pnl, swap=swap, nights=nights, equity=equity,
                    reason="stop" if hit_stop else "tp",
                ))
                in_pos = False
                # Pine does NOT reset activeDirection on exit: the next long
                # can only fire after a sell condition flips it to -1.
            else:
                if h[i] > best_high:
                    best_high = h[i]
                    if p["exit_mode"] == "trail":
                        nxt = best_high - trail_dist
                        if nxt > trail_level:
                            trail_level = nxt

        # ── signal on confirmed bar close ──
        trending = (not p["enable_chop_filter"]) or (ax[i] > p["adx_minimum"])
        f_asgma_up = bull[i] if p["use_asgma"] else True
        f_vwap_up = (c[i] > vw[i]) if p["use_vwap"] else True
        f_st_up = (c[i] > stv[i]) if p["use_supertrend"] else True
        buy_cond = f_asgma_up and f_vwap_up and f_st_up and trending

        f_asgma_dn = (not bull[i]) if p["use_asgma"] else True
        f_vwap_dn = (c[i] < vw[i]) if p["use_vwap"] else True
        f_st_dn = (c[i] < stv[i]) if p["use_supertrend"] else True
        sell_cond = f_asgma_dn and f_vwap_dn and f_st_dn and trending

        if buy_cond and active_dir != 1:
            active_dir = 1
            if (p["enable_buy"] and not in_pos and not np.isnan(sl_low[i])
                    and not np.isnan(a[i])):
                raw = h[i] - sl_low[i]
                trail_dist = min(max(raw, a[i] * p["trail_min_atr"]),
                                 a[i] * p["trail_max_atr"])
                entry_price = c[i] + slip
                trail_level = c[i] - trail_dist          # THE FIX: close, not high
                risk_per_unit = entry_price - trail_level
                if risk_per_unit > 0:
                    size = (equity * p["risk_percent"] / 100) / risk_per_unit
                    size = min(size, (equity * p["max_leverage"]) / c[i],
                               p["max_lot"])
                    if size > 0:
                        qty = size
                        best_high = h[i]
                        target = entry_price + risk_per_unit * p["reward_multiple"]
                        equity -= comm * entry_price * qty
                        entry_time = times[i]
                        in_pos = True
        elif sell_cond:
            active_dir = -1

        equity_curve.append(equity)

    t = pd.DataFrame(trades)
    res = dict(trades=t, equity_curve=pd.Series(equity_curve, index=d.index))

    if len(t) == 0:
        res.update(pf=np.nan, net=0.0, n=0, win_rate=np.nan, max_dd=np.nan)
        return res

    gross_profit = t.loc[t.pnl > 0, "pnl"].sum()
    gross_loss = -t.loc[t.pnl < 0, "pnl"].sum()
    curve = res["equity_curve"]
    dd = (curve.cummax() - curve) / curve.cummax()

    res.update(
        pf=gross_profit / gross_loss if gross_loss > 0 else np.inf,
        net=t.pnl.sum(),
        n=len(t),
        win_rate=(t.pnl > 0).mean(),
        max_dd=dd.max(),
    )
    return res


def summary(res: dict, label: str = "") -> str:
    if res["n"] == 0:
        return f"{label:<22} no trades"
    return (f"{label:<22} PF {res['pf']:>6.3f}   "
            f"net {res['net']:>+10,.2f}   "
            f"trades {res['n']:>4}   "
            f"win {res['win_rate']*100:>5.1f}%   "
            f"maxDD {res['max_dd']*100:>5.2f}%")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "btcusd_1h.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")

    # validation window — same span as the TradingView run
    tv = df.loc["2026-01-01":"2026-09-23"]
    print(summary(backtest(tv), "TV window 2026"))
    print(summary(backtest(df), "full 2018-2026"))