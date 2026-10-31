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
BASE = "https://www.okx.com"
BAR = "15m"      # 봉 종류
TOP_N = 100      # 거래대금 상위 몇 개 코인을 볼지
NEAR = 0.005     # 선 근처 = 위아래 0.5%
SURGE = 2.0      # 거래대금이 직전 20봉 평균의 몇 배 이상이면 알림
SHOW = 120       # 차트에 보여줄 봉 개수
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
    return df


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


def draw(df, title, path, only):
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
        if label == only:
            p.plot(x, d[c], color=color, lw=2.2, label=label)
    p.legend(loc="upper left", fontsize=10)
    p.set_title(title, fontsize=13)
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
    fig.savefig(path, dpi=110)
    plt.close(fig)


def send(path, caption, chat):
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto",
                  data={"chat_id": chat, "caption": caption},
                  files={"photo": open(path, "rb")}, timeout=30)


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
            if not res:
                continue
            ts = int(df["ts"].iloc[-1])
            near, ratio = res
            coin = inst.replace("-USDT-SWAP", "")
            for label in near:
                key = inst + "|" + label
                chat = ROOMS.get(label)
                if not chat or state.get(key) == ts:
                    continue
                draw(df, f"{coin} 15분봉 · {label} 근처", "chart.png", label)
                send("chart.png", f"{coin} · {label} 근처 · 거래대금 평균의 {ratio:.1f}배 (15분봉)", chat)
                state[key] = ts
        except Exception as e:
            print(inst, "error:", e)
    json.dump(state, open(STATE, "w"))


if __name__ == "__main__":
    main()
