"""
Signal watcher — BTC/ETH on 1h and 4h, both directions, alerts to Telegram.

Also reads commands from the Telegram chat on each run:
    /bal                  show both balances
    /bal005 10029.51      set account 242005 balance
    /bal006 9724.10       set account 242006 balance
    /status               show what the watcher is tracking

State lives in two JSON files the workflow commits back to the repo:
    watcher_state.json    last-alerted bar per symbol/interval, Telegram offset
    balances.json         per-account equity

Only alerts on a newly closed bar, so a restart never re-fires old signals.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from wt_trend import DEFAULTS, prepare

TOKEN = os.environ.get("TG_TOKEN", "")
CHAT_ID = os.environ.get("TG_CHAT", "")

STATE_FILE = "watcher_state.json"
BAL_FILE = "balances.json"
LOG_FILE = "signals.log"

SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
INTERVALS = ["1h", "4h"]
PARAMS = {"enable_chop_filter": False, "max_lot": 1e9, "risk_percent": 1.0}

DEFAULT_BALANCES = {"242005": 10000.0, "242006": 10000.0}
API = f"https://api.telegram.org/bot{TOKEN}"
KLINES = "https://api.binance.com/api/v3/klines"


def load(path, fallback):
    try:
        return json.load(open(path))
    except Exception:
        return dict(fallback)


def save(path, obj):
    json.dump(obj, open(path, "w"), indent=2)


def send(text: str):
    if not TOKEN or not CHAT_ID:
        print("NO TELEGRAM CONFIG -", text)
        return
    try:
        requests.post(f"{API}/sendMessage",
                      json={"chat_id": CHAT_ID, "text": text,
                            "parse_mode": "HTML"}, timeout=20)
    except Exception as e:
        print("send failed:", e)


def handle_commands(state, balances):
    """Poll for new messages and act on commands. True if balances changed."""
    if not TOKEN:
        return False
    offset = state.get("tg_offset", 0)
    try:
        r = requests.get(f"{API}/getUpdates",
                         params={"offset": offset + 1, "timeout": 0}, timeout=20)
        updates = r.json().get("result", [])
    except Exception as e:
        print("getUpdates failed:", e)
        return False

    changed = False
    for u in updates:
        state["tg_offset"] = u["update_id"]
        msg = u.get("message") or u.get("edited_message") or {}
        text = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            continue

        parts = text.split()
        cmd = parts[0].lower().lstrip("/").split("@")[0]

        if cmd == "bal" and len(parts) == 1:
            send("<b>Balances</b>\n"
                 + "\n".join(f"{k}: ${v:,.2f}" for k, v in sorted(balances.items())))

        elif cmd in ("bal005", "bal006") and len(parts) == 2:
            acct = "242005" if cmd.endswith("005") else "242006"
            try:
                val = float(parts[1].replace(",", ""))
            except ValueError:
                send(f"Could not read a number from: {parts[1]}")
                continue
            if not (1000 <= val <= 1_000_000):
                send(f"${val:,.2f} looks wrong - ignored.")
                continue
            old = balances.get(acct, 0.0)
            balances[acct] = val
            changed = True
            send(f"{acct}: ${old:,.2f} -> <b>${val:,.2f}</b>")

        elif cmd == "status":
            lines = ["<b>Watching</b>"]
            for name in SYMBOLS:
                for iv in INTERVALS:
                    k = f"{name}_{iv}"
                    lines.append(f"{iv} {name}: last {state.get(k, 'none')}")
            lines.append("")
            lines += [f"{k}: ${v:,.2f}" for k, v in sorted(balances.items())]
            send("\n".join(lines))

        else:
            send("Commands:\n/bal\n/bal005 10029.51\n/bal006 9724.10\n/status")

    return changed


def fetch(symbol: str, interval: str, limit: int = 400) -> pd.DataFrame:
    r = requests.get(KLINES, params={"symbol": symbol, "interval": interval,
                                     "limit": limit}, timeout=20)
    r.raise_for_status()
    df = pd.DataFrame(r.json()).iloc[:, :6]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df.astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.set_index("timestamp").sort_index().iloc[:-1]


def signal(df: pd.DataFrame, p: dict):
    d = prepare(df, p)
    i = len(d) - 1
    if i < 2:
        return None

    def state_at(k):
        c = d["close"].iloc[k]
        if d["asgma_bull"].iloc[k] and c > d["vwap"].iloc[k] and c > d["supertrend"].iloc[k]:
            return 1
        if (not d["asgma_bull"].iloc[k]) and c < d["vwap"].iloc[k] and c < d["supertrend"].iloc[k]:
            return -1
        return 0

    now, prev = state_at(i), state_at(i - 1)
    if now == 0 or now == prev:
        return None

    c = d["close"].iloc[i]
    atr = d["atr"].iloc[i]
    swing = d["last_swing_low"].iloc[i] if now == 1 else d["last_swing_high"].iloc[i]
    if np.isnan(atr) or np.isnan(swing):
        return None

    raw = (d["high"].iloc[i] - swing) if now == 1 else (swing - d["low"].iloc[i])
    dist = min(max(raw, atr * p["trail_min_atr"]), atr * p["trail_max_atr"])
    stop = c - dist * now
    rpu = abs(c - stop)
    if rpu <= 0:
        return None

    return dict(dir=now, entry=c, stop=stop, rpu=rpu, bar=d.index[i])


def main():
    p = {**DEFAULTS, **PARAMS}
    state = load(STATE_FILE, {})
    balances = load(BAL_FILE, DEFAULT_BALANCES)

    handle_commands(state, balances)

    fired = []
    for name, sym in SYMBOLS.items():
        for iv in INTERVALS:
            key = f"{name}_{iv}"
            try:
                s = signal(fetch(sym, iv), p)
            except Exception as e:
                print(f"{key}: {type(e).__name__}: {e}")
                continue
            if s is None:
                continue
            stamp = s["bar"].isoformat()
            if state.get(key) == stamp:
                continue
            state[key] = stamp

            side = "BUY" if s["dir"] == 1 else "SELL"
            tradeable = (iv == "4h" and s["dir"] == 1)

            lots = []
            for acct, bal in sorted(balances.items()):
                risk = bal * p["risk_percent"] / 100
                lot = min(risk / s["rpu"], (bal * p["max_leverage"]) / s["entry"])
                lots.append(f"{acct}:  {lot:.3f} lot  (${risk:,.2f})")

            tag = "" if tradeable else "\n<i>outside the tested set - info only</i>"
            send(f"<b>{iv} {name} - {side}</b>\n"
                 f"Entry {s['entry']:,.2f}\n"
                 f"Stop  {s['stop']:,.2f}\n"
                 f"Dist  {s['rpu']:,.2f}\n\n"
                 + "\n".join(lots)
                 + f"\n\nBar {stamp} UTC{tag}")

            with open(LOG_FILE, "a") as lg:
                lg.write(f"{stamp}\t{iv}\t{name}\t{side}\t"
                         f"{s['entry']:.2f}\t{s['stop']:.2f}\t{s['rpu']:.2f}\n")
            fired.append(key)

    save(STATE_FILE, state)
    save(BAL_FILE, balances)
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC - "
          f"{'alerted: ' + ', '.join(fired) if fired else 'no new signals'}")


if __name__ == "__main__":
    main()
