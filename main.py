import os, sys, json, time
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

TOKEN = os.environ.get("TELEGRAM_TOKEN")
ROOMS = {
    "EMA7": os.environ.get("CHAT_EMA7"),
    "EMA10": os.environ.get("CHAT_EMA10"),
    "EMA20": os.environ.get("CHAT_EMA20"),
    "EMA50": os.environ.get("CHAT_EMA50"),
    "EMA120": os.environ.get("CHAT_EMA120"),
    "EMA200": os.environ.get("CHAT_EMA200"),
    "거가중100": os.environ.get("CHAT_VWAP100"),
}
CROSS_ROOMS = {"골크": os.environ.get("CHAT_GC"), "데크": os.environ.get("CHAT_DC")}
ALIGN_ROOMS = {"정배열": os.environ.get("CHAT_BULL"), "역배열": os.environ.get("CHAT_BEAR")} 
_ONE = os.environ.get("CHAT_EMA7")
ROOMS = {k: _ONE for k in ROOMS}
CROSS_ROOMS = {k: _ONE for k in CROSS_ROOMS}
ALIGN_ROOMS = {k: _ONE for k in ALIGN_ROOMS}
ALIGN = [7, 20, 50, 120, 200]   # 정배열 = 7>20>50>120>200
PAIRS = [(7, 20), (7, 50), (20, 50), (50, 200)]   # 골크/데크를 볼 이평 짝
GAP = 0.003      # 두 이평 차이가 가격의 0.3% 안이면 '근접'
CROSS_SURGE = 1.5  # 골크/데크 근접은 거래대금 평균의 1.5배 이상
BASE = "https://www.okx.com"
BAR = "15m"      # 봉 종류
TOP_N = 100      # 거래대금 상위 몇 개 코인을 볼지
NEAR = 0.005     # 선 근처 = 위아래 0.5%
SURGE = 2.0      # 거래대금이 직전 20봉 평균의 몇 배 이상이면 알림
SHOW = 120       # 차트에 보여줄 봉 개수
RR = 1.5         # 익절 거리 = 손절 거리의 몇 배
MIN_TP = 0.01   # 익절이 진입가의 0.8%보다 작으면 알림 안 보냄
STATE = "state.json"

names = {f.name for f in font_manager.fontManager.ttflist}
for n in ["NanumGothic", "Malgun Gothic", "AppleGothic"]:
    if n in names:
        plt.rcParams["font.family"] = n
        break
plt.rcParams["axes.unicode_minus"] = False


def get_candles(inst):
    r = requests.get(BASE + "/api/v5/market/candles",
                     params={"instId": inst, "bar": BAR, "limit": 300},
                     timeout=10).json()["data"]
    df = pd.DataFrame(r).iloc[:, :9]
    df.columns = ["ts", "o", "h", "l", "c", "vol", "volc", "turn", "confirm"]
    for k in df.columns:
        df[k] = df[k].astype(float)
    df = df[df["confirm"] == 1].sort_values("ts").reset_index(drop=True)
    if df["turn"].sum() == 0:
        df["turn"] = df["volc"] * df["c"]
    return df


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def add_indicators(df):
    for n in (7, 10, 20, 50, 120, 200):
        df[f"ema{n}"] = ema(df["c"], n)
    tp = (df["h"] + df["l"] + df["c"]) / 3
    df["vwap100"] = (tp * df["volc"]).rolling(100).sum() / df["volc"].rolling(100).sum()
    for n in (6, 12, 24):
        df[f"rsi{n}"] = rsi(df["c"], n)
    df["macd"] = ema(df["c"], 12) - ema(df["c"], 26)
    df["sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["hist"] = df["macd"] - df["sig"]
    pc = df["c"].shift(1)
    tr = pd.concat([df["h"] - df["l"], (df["h"] - pc).abs(), (df["l"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


def plan(df, col):
    last = df.iloc[-1]
    entry, line, atr = last["c"], last[col], last["atr"]
    if entry >= line:
        side = "롱"
        sl = min(line, entry) - 0.5 * atr
        tp = entry + RR * (entry - sl)
    else:
        side = "숏"
        sl = max(line, entry) + 0.5 * atr
        tp = entry - RR * (sl - entry)
    return side, entry, sl, tp


def cross_check(df, f, s_):
    a, b = df[f"ema{f}"], df[f"ema{s_}"]
    g = (a - b).iloc[-4:].values
    price = df["c"].iloc[-1]
    if abs(g[-1]) / price > GAP:
        return None
    if g[-1] < 0 and g[-2] < 0 and g[-1] > g[-2] > g[-3]:
        return "골크"
    if g[-1] > 0 and g[-2] > 0 and g[-1] < g[-2] < g[-3]:
        return "데크"
    return None


def align_check(df):
    last = df.iloc[-1]
    v = [last[f"ema{n}"] for n in ALIGN]
    up = [v[i] > v[i + 1] for i in range(len(v) - 1)]
    for kind, ok in (("정배열", up), ("역배열", [not u for u in up])):
        bad = [i for i, o in enumerate(ok) if not o]
        if len(bad) != 1:
            continue
        f, s_ = ALIGN[bad[0]], ALIGN[bad[0] + 1]
        want = "골크" if kind == "정배열" else "데크"
        if cross_check(df, f, s_) == want:
            return kind, f, s_
    return None


def cross_plan(df, kind, s_):
    last = df.iloc[-1]
    entry, slow, atr = last["c"], last[f"ema{s_}"], last["atr"]
    if kind == "골크":
        sl = min(df["l"].iloc[-5:].min(), slow) - 0.3 * atr
        sl = min(sl, entry - 0.5 * atr)
        return "롱", entry, sl, entry + RR * (entry - sl)
    sl = max(df["h"].iloc[-5:].max(), slow) + 0.3 * atr
    sl = max(sl, entry + 0.5 * atr)
    return "숏", entry, sl, entry - RR * (sl - entry)


def fmt(v):
    if v >= 100:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:.4f}"
    return f"{v:.7f}"


LINES = [("EMA7", "ema7", "#d81b60"), ("EMA10", "ema10", "#29b6f6"),
         ("EMA20", "ema20", "#f9a825"), ("EMA50", "ema50", "#2e7d32"),
         ("EMA120", "ema120", "#757575"), ("EMA200", "ema200", "#e53935"),
         ("거가중100", "vwap100", "#6a1b9a")]


def check(df):
    if len(df) < 210:
        return None
    last = df.iloc[-1]
    near = []
    for label, col, _ in LINES:
        v = last[col]
        if pd.notna(v) and abs(last["c"] - v) / v <= NEAR:
            near.append(label)
    base = df["turn"].iloc[-21:-1].mean()
    ratio = last["turn"] / base if base > 0 else 0
    if near and ratio >= SURGE:
        return near, ratio
    return None


def draw(df, title, path, only, pl):
    d = df.iloc[-SHOW:].reset_index(drop=True)
    x = np.arange(len(d))
    fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True,
                           gridspec_kw={"height_ratios": [5, 1.5, 1.5, 1.5]})
    p = ax[0]
    for i, r in d.iterrows():
        col = "#2e9e44" if r["c"] >= r["o"] else "#d62728"
        p.vlines(i, r["l"], r["h"], color=col, lw=0.8)
        p.bar(i, abs(r["c"] - r["o"]), bottom=min(r["o"], r["c"]), color=col, width=0.7)
    for label, c, color in LINES:
        if label in only:
            p.plot(x, d[c], color=color, lw=2.2, label=label)
    side, entry, sl, tp = pl
    for v, name, color in ((tp, "익절", "#2e7d32"), (entry, "진입", "#000000"), (sl, "손절", "#c62828")):
        p.axhline(v, color=color, ls="--", lw=1.2)
        p.annotate(f" {name} {fmt(v)}", xy=(1.0, v), xycoords=("axes fraction", "data"),
                   color=color, fontsize=9, va="center", ha="left")
    p.legend(loc="upper left", fontsize=10)
    p.set_title(title, fontsize=14, color="#2e7d32" if side == "롱" else "#c62828")
    p.grid(alpha=0.25)
    for n, color in ((6, "#8e44ad"), (12, "#1e88e5"), (24, "#fb8c00")):
        ax[1].plot(x, d[f"rsi{n}"], color=color, lw=1, label=f"RSI{n}")
    ax[1].axhline(70, color="gray", ls="--", lw=0.6)
    ax[1].axhline(30, color="gray", ls="--", lw=0.6)
    ax[1].legend(loc="upper left", fontsize=7, ncol=3)
    ax[1].grid(alpha=0.25)
    ax[2].bar(x, d["hist"], color=np.where(d["hist"] >= 0, "#2e9e44", "#d62728"), width=0.7)
    ax[2].plot(x, d["macd"], color="#1e88e5", lw=1, label="MACD")
    ax[2].plot(x, d["sig"], color="#fb8c00", lw=1, label="Signal")
    ax[2].legend(loc="upper left", fontsize=7, ncol=2)
    ax[2].grid(alpha=0.25)
    ax[3].bar(x, d["turn"], color=np.where(d["c"] >= d["o"], "#2e9e44", "#d62728"), width=0.7)
    ax[3].set_ylabel("거래대금(USDT)", fontsize=8)
    ax[3].grid(alpha=0.25)
    ticks = list(range(0, len(d), 20))
    t = pd.to_datetime(d["ts"], unit="ms") + pd.Timedelta(hours=9)
    ax[3].set_xticks(ticks)
    ax[3].set_xticklabels([t[i].strftime("%m-%d %H:%M") for i in ticks], fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def send(path, caption, chat):
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto",
                  data={"chat_id": chat, "caption": caption},
                  files={"photo": open(path, "rb")}, timeout=30)


def do_cross(df, inst, coin, ts, state):
    if len(df) < 210:
        return
    base = df["turn"].iloc[-21:-1].mean()
    ratio = df["turn"].iloc[-1] / base if base > 0 else 0
    if ratio < CROSS_SURGE:
        return
    for f, s_ in PAIRS:
        kind = cross_check(df, f, s_)
        chat = CROSS_ROOMS.get(kind) if kind else None
        key = f"{inst}|{kind}|{f}-{s_}"
        if not chat or state.get(key) == ts:
            continue
        pl = cross_plan(df, kind, s_)
        side, entry, sl, tp = pl
        if abs(tp / entry - 1) < MIN_TP:
            continue
        names = [f"EMA{f}", f"EMA{s_}"]
        draw(df, f"{coin} {kind} 임박 · EMA{f}×EMA{s_} · {side}진입", "chart.png", names, pl)
        cap = (f"{kind} 임박 · {coin} · EMA{f}×EMA{s_} → {side}진입\n"
               f"진입가: {fmt(entry)}\n"
               f"손절가: {fmt(sl)} ({(sl / entry - 1) * 100:+.2f}%)\n"
               f"익절가: {fmt(tp)} ({(tp / entry - 1) * 100:+.2f}%)\n"
               f"거래대금: 평균의 {ratio:.1f}배 (15분봉)")
        send("chart.png", cap, chat)
        state[key] = ts


def do_align(df, inst, coin, ts, state):
    if len(df) < 210:
        return
    base = df["turn"].iloc[-21:-1].mean()
    ratio = df["turn"].iloc[-1] / base if base > 0 else 0
    if ratio < CROSS_SURGE:
        return
    res = align_check(df)
    if not res:
        return
    kind, f, s_ = res
    chat = ALIGN_ROOMS.get(kind)
    key = f"{inst}|{kind}"
    if not chat or state.get(key) == ts:
        return
    pl = cross_plan(df, "골크" if kind == "정배열" else "데크", s_)
    side, entry, sl, tp = pl
    if abs(tp / entry - 1) < MIN_TP:
        return
    names = [f"EMA{n}" for n in ALIGN]
    draw(df, f"{coin} {kind} 임박 · {side}진입", "chart.png", names, pl)
    cap = (f"{kind} 임박 · {coin} · 마지막 남은 EMA{f}×EMA{s_} → {side}진입\n"
           f"진입가: {fmt(entry)}\n"
           f"손절가: {fmt(sl)} ({(sl / entry - 1) * 100:+.2f}%)\n"
           f"익절가: {fmt(tp)} ({(tp / entry - 1) * 100:+.2f}%)\n"
           f"거래대금: 평균의 {ratio:.1f}배 (15분봉)")
    send("chart.png", cap, chat)
    state[key] = ts


def main():
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    tk = requests.get(BASE + "/api/v5/market/tickers", params={"instType": "SWAP"},
                      timeout=10).json()["data"]
    tk = [t for t in tk if t["instId"].endswith("-USDT-SWAP")]
    tk.sort(key=lambda t: num(t.get("volCcy24h")) * num(t.get("last")), reverse=True)
    for t in tk[:TOP_N]:
        inst = t["instId"]
        try:
            df = add_indicators(get_candles(inst))
            res = check(df)
            time.sleep(0.15)
            ts = int(df["ts"].iloc[-1])
            coin = inst.replace("-USDT-SWAP", "")
            do_cross(df, inst, coin, ts, state)
            do_align(df, inst, coin, ts, state)
            if not res:
                continue
            near, ratio = res
            for label in near:
                key = inst + "|" + label
                chat = ROOMS.get(label)
                if not chat or state.get(key) == ts:
                    continue
                col = [c for l, c, _ in LINES if l == label][0]
                pl = plan(df, col)
                side, entry, sl, tp = pl
                if abs(tp / entry - 1) < MIN_TP:
                    continue
                draw(df, f"{coin} {side}진입 · {label} 근처", "chart.png", [label], pl)
                cap = (f"{side}진입 · {coin} · {label} 근처\n"
                       f"진입가: {fmt(entry)}\n"
                       f"손절가: {fmt(sl)} ({(sl / entry - 1) * 100:+.2f}%)\n"
                       f"익절가: {fmt(tp)} ({(tp / entry - 1) * 100:+.2f}%)\n"
                       f"거래대금: 평균의 {ratio:.1f}배 (15분봉)")
                send("chart.png", cap, chat)
                state[key] = ts
        except Exception as e:
            print(inst, "error:", e)
    json.dump(state, open(STATE, "w"))


if __name__ == "__main__":
    main()
