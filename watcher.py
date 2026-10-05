"""
Signal watcher — BTC/ETH on 1h and 4h, both directions, alerts to Telegram.

Only alerts on a NEWLY CLOSED bar, and remembers what it has already sent
so a restart doesn't re-fire old signals.

Run it on a schedule (cron / launchd) every 5 minutes.
"""

import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from wt_trend import DEFAULTS, prepare

# ── config ────────────────────────────────────────────────────────────
TOKEN = os.environ.get("TG_TOKEN", "")
CHAT_ID = os.environ.get("TG_CHAT", "")
STATE_FILE = "watcher_state.json"

SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
INTERVALS = ["1h", "4h"]

# the only configuration the backtest validated
PARAMS = {"enable_chop_filter": False, "max_lot": 1e9, "risk_percent": 1.0}
EQUITY = float(os.environ.get("ACCOUNT_EQUITY", "10009"))

KLINES = "https://api.binance.com/api/v3/klines"


def fetch(symbol: str, interval: str, limit: int = 400) -> pd.DataFrame:
    r = requests.get(KLINES, params={"symbol": symbol, "interval": interval,
                                     "limit": limit}, timeout=20)
    r.raise_for_status()
    df = pd.DataFrame(r.json()).iloc[:, :6]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df.astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    return df.iloc[:-1]          # drop the still-forming bar


def signal(df: pd.DataFrame, p: dict):
    """Return (direction, entry, stop, lot, risk) for the last closed bar, or None."""
    d = prepare(df, p)
    i = len(d) - 1
    j = i - 1
    if j < 1:
        return None

    def state(k):
        c = d["close"].iloc[k]
        up = (d["asgma_bull"].iloc[k] and c > d["vwap"].iloc[k]
              and c > d["supertrend"].iloc[k])
        dn = ((not d["asgma_bull"].iloc[k]) and c < d["vwap"].iloc[k]
              and c < d["supertrend"].iloc[k])
        return 1 if up else (-1 if dn else 0)

    now, prev = state(i), state(j)
    if now == 0 or now == prev:
        return None                          # no flip on this bar

    c = d["close"].iloc[i]
    atr = d["atr"].iloc[i]
    swing = (d["last_swing_low"].iloc[i] if now == 1
             else d["last_swing_high"].iloc[i])
    if np.isnan(atr) or np.isnan(swing):
        return None

    raw = (d["high"].iloc[i] - swing) if now == 1 else (swing - d["low"].iloc[i])
    dist = min(max(raw, atr * p["trail_min_atr"]), atr * p["trail_max_atr"])
    stop = c - dist * now
    rpu = abs(c - stop)
    if rpu <= 0:
        return None

    lot = (EQUITY * p["risk_percent"] / 100) / rpu
    lot = min(lot, (EQUITY * p["max_leverage"]) / c)
    return dict(dir=now, entry=c, stop=stop, lot=lot, risk=rpu * lot,
                bar=d.index[i])


def send(text: str):
    if not TOKEN or not CHAT_ID:
        print("NO TELEGRAM CONFIG —", text)
        return
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                  json={"chat_id": CHAT_ID, "text": text,
                        "parse_mode": "HTML"}, timeout=20)


def load_state():
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {}


def main():
    p = {**DEFAULTS, **PARAMS}
    state = load_state()
    fired = []

    for name, sym in SYMBOLS.items():
        for iv in INTERVALS:
            key = f"{name}_{iv}"
            try:
                df = fetch(sym, iv)
                s = signal(df, p)
            except Exception as e:
                print(f"{key}: {type(e).__name__}: {e}")
                continue
            if s is None:
                continue
            stamp = s["bar"].isoformat()
            if state.get(key) == stamp:
                continue                      # already alerted for this bar
            state[key] = stamp

            side = "BUY" if s["dir"] == 1 else "SELL"
            tradeable = (iv == "4h" and s["dir"] == 1)
            tag = "" if tradeable else "\n<i>not in the tested set — info only</i>"
            msg = (f"<b>{iv} {name} — {side}</b>\n"
                   f"Entry {s['entry']:,.2f}\n"
                   f"Stop  {s['stop']:,.2f}\n"
                   f"Lot   {s['lot']:.3f}\n"
                   f"Risk  ${s['risk']:,.2f}\n"
                   f"Bar   {stamp} UTC{tag}")
            send(msg)
            with open("signals.log", "a") as lg:
                lg.write(f"{stamp}\t{iv}\t{name}\t{side}\t"
                         f"{s['entry']:.2f}\t{s['stop']:.2f}\t"
                         f"{s['lot']:.4f}\t{s['risk']:.2f}\n")
            fired.append(key)

    json.dump(state, open(STATE_FILE, "w"))
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC — "
          f"{'alerted: ' + ', '.join(fired) if fired else 'no new signals'}")


if __name__ == "__main__":
    main()
