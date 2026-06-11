#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI/Technical Signal Bot  ->  แจ้งเตือนเข้า Telegram
- ดึงราคาจริงจาก yfinance (US เช่น NVDA / หุ้นไทยใส่ .BK เช่น PTT.BK)
- คำนวณสัญญาณจาก SMA20/50, RSI14, MACD(12/26/9), ATR14
- ส่งเฉพาะ "สัญญาณใหม่ที่แรงพอ" (กันสแปมด้วย state.json)
- คุณเปิดแอป Dime! แล้วกดส่งคำสั่งเอง
"""

import os, json, math, datetime as dt
import requests
import yfinance as yf
import pandas as pd

# ============================================================
# 1) ตั้งค่าตรงนี้ได้เลย
# ============================================================
WATCHLIST = [
    # หุ้น US (ใส่ ticker ตรงๆ)
    "NVDA", "AAPL", "MSFT", "TSLA", "AMD",
    # หุ้นไทย (ต้องลงท้าย .BK)
    "PTT.BK", "KBANK.BK", "ADVANC.BK", "CPALL.BK",
]

CONF_THRESHOLD = 62          # ส่งเตือนเมื่อความมั่นใจ >= ค่านี้ (%)
RISK_PER_TRADE = 1.0         # % ของพอร์ตต่อไม้ (ใช้คำนวณจำนวนหุ้นแนะนำ)
PORTFOLIO_SIZE = 100000      # ขนาดพอร์ต (บาท) สำหรับคำนวณจำนวนหุ้นแนะนำ
SUGGEST_OPTIONS = True       # แนะนำไอเดีย options สำหรับหุ้น US (Dime! รองรับ)
STATE_FILE = "state.json"

# ความลับ -> ตั้งเป็น GitHub Secrets (อย่าใส่ค่าจริงในไฟล์)
TG_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT  = os.environ.get("TELEGRAM_CHAT_ID", "")
DRY_RUN  = os.environ.get("DRY_RUN", "") == "1"   # 1 = พิมพ์อย่างเดียว ไม่ส่งจริง


# ============================================================
# 2) อินดิเคเตอร์
# ============================================================
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series):
    e12 = close.ewm(span=12, adjust=False).mean()
    e26 = close.ewm(span=26, adjust=False).mean()
    line = e12 - e26
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal, line - signal


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def clip(x, lo, hi):
    return max(lo, min(hi, x))


# ============================================================
# 3) เครื่องคิดสัญญาณ
# ============================================================
def compute_signal(ticker: str):
    try:
        df = yf.download(ticker, period="8mo", interval="1d",
                         progress=False, auto_adjust=True)
    except Exception as e:
        print(f"[WARN] โหลด {ticker} ไม่ได้: {e}")
        return None
    if df is None or len(df) < 60:
        print(f"[WARN] {ticker}: ข้อมูลไม่พอ")
        return None
    # yfinance บางเวอร์ชันคืน MultiIndex แม้ ticker เดียว
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].dropna()
    price = float(close.iloc[-1])
    s20 = float(close.rolling(20).mean().iloc[-1])
    s50 = float(close.rolling(50).mean().iloc[-1])
    r = float(rsi(close).iloc[-1])
    _, _, hist = macd(close)
    h = float(hist.iloc[-1])
    a = float(atr(df).iloc[-1])
    mom5 = price / float(close.iloc[-6]) - 1 if len(close) > 6 else 0.0

    score = 0.0
    score += 22 if price > s50 else -22
    score += 14 if s20 > s50 else -14
    if r < 32:   score += 18
    elif r > 70: score -= 18
    else:        score += (50 - r) * 0.25
    score += clip((h / price) * 9000, -22, 22)
    score += clip(mom5 * 400, -10, 10)
    score = clip(score, -100, 100)

    action = "BUY" if score >= 20 else "SELL" if score <= -20 else "HOLD"
    conf = round(min(94, 42 + abs(score) * 0.55))

    dist = max(a * 1.5, price * 0.03)
    if action == "BUY":
        target, stop = price + dist * 1.7, price - dist
    elif action == "SELL":
        target, stop = price - dist * 1.7, price + dist
    else:
        target, stop = price + dist, price - dist

    qty = 0
    per_share = abs(price - stop) or price * 0.01
    qty = max(1, int((PORTFOLIO_SIZE * RISK_PER_TRADE / 100) / per_share))

    option = None
    if SUGGEST_OPTIONS and action != "HOLD" and conf >= 58 and not ticker.endswith(".BK"):
        is_call = action == "BUY"
        strike = round_strike(price * (1.02 if is_call else 0.98))
        option = {"type": "CALL" if is_call else "PUT", "strike": strike}

    return dict(ticker=ticker, price=price, action=action, conf=conf,
                target=target, stop=stop, rsi=r, qty=qty, option=option)


def round_strike(p):
    if p < 25:  return round(p)
    if p < 100: return round(p / 2.5) * 2.5
    if p < 300: return round(p / 5) * 5
    return round(p / 10) * 10


# ============================================================
# 4) Telegram
# ============================================================
def fmt(n, d=2):
    return f"{n:,.{d}f}"


def build_message(s):
    is_buy = s["action"] == "BUY"
    head = "🟢 <b>สัญญาณซื้อ</b>" if is_buy else "🔴 <b>สัญญาณขาย/ระวัง</b>"
    market = "🇹🇭 หุ้นไทย" if s["ticker"].endswith(".BK") else "🇺🇸 หุ้น US"
    tv = s["ticker"].replace(".BK", "")
    lines = [
        f"{head}  <b>{s['ticker']}</b>  ({market})",
        f"ราคาปัจจุบัน: <b>{fmt(s['price'])}</b>  •  ความมั่นใจ {s['conf']}%",
        "",
        f"🎯 เป้าหมาย: <b>{fmt(s['target'])}</b>",
        f"🛑 ตัดขาดทุน: <b>{fmt(s['stop'])}</b>",
        f"📦 จำนวนแนะนำ: ~{s['qty']} หุ้น  (เสี่ยง {RISK_PER_TRADE}%/ไม้)",
        f"📊 RSI(14): {fmt(s['rsi'],0)}",
    ]
    if s["option"]:
        o = s["option"]
        lines.append(f"⚡ ไอเดีย Options: <b>{o['type']}</b> strike ~{fmt(o['strike'],1)} (Dime! options)")
    lines += [
        "",
        f'📲 เปิดแอป <b>Dime!</b> แล้วกดส่งคำสั่งเอง',
        f'📈 กราฟ: https://www.tradingview.com/symbols/{tv}/',
        "",
        "<i>ข้อมูลดีเลย์ ~15 นาที • เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน</i>",
    ]
    return "\n".join(lines)


def send_telegram(text):
    if DRY_RUN or not TG_TOKEN or not TG_CHAT:
        print("---- (DRY RUN / ยังไม่ได้ตั้ง token) ----")
        print(text)
        print("-----------------------------------------")
        return
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    r = requests.post(url, data={
        "chat_id": TG_CHAT, "text": text,
        "parse_mode": "HTML", "disable_web_page_preview": "true",
    }, timeout=20)
    if r.status_code != 200:
        print(f"[ERROR] Telegram: {r.status_code} {r.text}")


# ============================================================
# 5) state กันแจ้งซ้ำ
# ============================================================
def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ============================================================
# 6) main
# ============================================================
def main():
    print(f"== รันบอท {dt.datetime.utcnow().isoformat()}Z ==")
    state = load_state()
    sent = 0
    for t in WATCHLIST:
        s = compute_signal(t)
        if not s:
            continue
        strong = s["action"] in ("BUY", "SELL") and s["conf"] >= CONF_THRESHOLD
        print(f"{t:10s} {s['action']:4s} conf={s['conf']:>3}  price={s['price']:.2f}")
        if strong and state.get(t) != s["action"]:
            send_telegram(build_message(s))
            sent += 1
        # จำสถานะล่าสุดไว้เสมอ (รวม HOLD) เพื่อรู้ว่าเปลี่ยนทิศเมื่อไหร่
        state[t] = s["action"]
    save_state(state)
    print(f"ส่งแจ้งเตือน {sent} รายการ")


if __name__ == "__main__":
    main()
