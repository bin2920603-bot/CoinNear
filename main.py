import os, json, time
from datetime import datetime, timedelta, timezone
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT = os.environ.get("CHAT_EMA7")   # 모든 알림을 이 방 하나로
BASE = "https://www.okx.com"
BARS = [("1D", "일봉"), ("4H", "4시간봉"), ("1H", "1시간봉"), ("15m", "15분봉")]
TOP_N = 100        # 거래대금 상위 몇 개 코인을 볼지
ALIGN = [7, 20, 50, 120, 200]                  # 정배열 = 7>20>50>120>200
PAIRS = [(7, 20), (7, 50), (20, 50), (50, 200)]  # 골크/데크를 볼 이평 짝
GAP = 0.003        # 두 이평 차이가 가격의 0.3% 안이면 '근접'
CROSS_SURGE = 1.5  # 골크/데크/배열 알림: 거래대금 평균의 1.5배 이상
SURGE = 2.0        # 선 튕김 알림: 거래대금 평균의 2배 이상
TOUCH = 0.001      # 꼬리가 선의 0.1% 안까지 오면 '닿았다'
SHOW = 120         # 차트에 보여줄 봉 개수
RR = 1.5           # 익절 거리 = 손절 거리의 몇 배
MIN_TP = 0.01      # 익절이 진입가의 1%보다 작으면 알림 안 보냄
MAX_SL = 0.05      # 손절이 진입가에서 5%보다 멀면 알림 안 보냄
MAX_WAIT = 96      # 이 봉 수 안에 익절/손절 둘 다 안 닿으면 '시간초과'
REPORT_HOUR = 9    # 매일 이 시각(한국시간) 이후 첫 실행 때 성적표 보냄
REPORT_DAYS = 7    # 성적표는 최근 며칠치
STATE = "state.json"
KST = timezone(timedelta(hours=9))

names = {f.name for f in font_manager.fontManager.ttflist}
for n in ["NanumGothic", "Malgun Gothic", "AppleGothic"]:
    if n in names:
        plt.rcParams["font.family"] = n
        break
plt.rcParams["axes.unicode_minus"] = False

LINES = [("EMA7", "ema7", "#d81b60"), ("EMA10", "ema10", "#29b6f6"),
         ("EMA20", "ema20", "#f9a825"), ("EMA50", "ema50", "#2e7d32"),
         ("EMA120", "ema120", "#757575"), ("EMA200", "ema200", "#e53935"),
         ("거가중100", "vwap100", "#6a1b9a")]
BAR_NAME = dict(BARS)


# ---------------- 데이터 ----------------
def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def get_candles(inst, bar):
    r = requests.get(BASE + "/api/v5/market/candles",
                     params={"instId": inst, "bar": bar, "limit": 300},
                     timeout=10).json()["data"]
    df = pd.DataFrame(r).iloc[:, :9]
    df.columns = ["ts", "o", "h", "l", "c", "vol", "volc", "turn", "confirm"]
    for k in df.columns:
        df[k] = df[k].astype(float)
    df = df[df["confirm"] == 1].sort_values("ts").reset_index(drop=True)
    if df["turn"].sum() == 0:
        df["turn"] = df["volc"] * df["c"]
    return df


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


def surge_ratio(df):
    base = df["turn"].iloc[-21:-1].mean()
    return df["turn"].iloc[-1] / base if base > 0 else 0


# ---------------- 1. 큰 흐름(추세) ----------------
def trend_of(df):
    if df is None or len(df) < 60:
        return None
    last = df.iloc[-1]
    if last["c"] > last["ema50"] and last["ema20"] > last["ema50"]:
        return "up"
    if last["c"] < last["ema50"] and last["ema20"] < last["ema50"]:
        return "down"
    return None


def arrow(t):
    return {"up": "↑", "down": "↓"}.get(t, "↔")


# ---------------- 2. 선에서 튕겼나 ----------------
def bounce_check(df, want):
    """마감된 마지막 봉이 선을 꼬리로 찍고 반대편에서 마감했는지"""
    last = df.iloc[-1]
    hits = []
    for label, col, _ in LINES:
        v = last[col]
        if pd.isna(v) or v <= 0:
            continue
        if want == "롱" and last["l"] <= v * (1 + TOUCH) and last["c"] > v and last["c"] > last["o"]:
            hits.append((label, col))
        if want == "숏" and last["h"] >= v * (1 - TOUCH) and last["c"] < v and last["c"] < last["o"]:
            hits.append((label, col))
    return hits


def limit_sl(pl):
    """손절이 너무 멀면 익절=진입으로 바꿔서 알림이 안 나가게 함"""
    side, entry, sl, tp = pl
    if abs(sl / entry - 1) > MAX_SL:
        return side, entry, entry, entry
    return pl


def bounce_plan(df, col, side):
    last = df.iloc[-1]
    entry, line, atr = last["c"], last[col], last["atr"]
    if side == "롱":
        sl = min(last["l"], line) - 0.3 * atr
        return limit_sl((side, entry, sl, entry + RR * (entry - sl)))
    sl = max(last["h"], line) + 0.3 * atr
    return limit_sl((side, entry, sl, entry - RR * (sl - entry)))


# ---------------- 골크/데크, 정배열/역배열 ----------------
def cross_check(df, f, s_):
    g = (df[f"ema{f}"] - df[f"ema{s_}"]).iloc[-4:].values
    if abs(g[-1]) / df["c"].iloc[-1] > GAP:
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
        return limit_sl(("롱", entry, sl, entry + RR * (entry - sl)))
    sl = max(df["h"].iloc[-5:].max(), slow) + 0.3 * atr
    sl = max(sl, entry + 0.5 * atr)
    return limit_sl(("숏", entry, sl, entry - RR * (sl - entry)))


def candle_ok(df, side):
    last = df.iloc[-1]
    return (side == "롱" and last["c"] >= last["o"]) or (side == "숏" and last["c"] <= last["o"])


# ---------------- 그림 · 보내기 ----------------
def fmt(v):
    if v >= 100:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:.4f}"
    return f"{v:.7f}"


def um(v):
    if v >= 1e8:
        return f"{v / 1e8:,.1f}억"
    if v >= 1e4:
        return f"{v / 1e4:,.1f}만"
    return f"{v:,.0f}"


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
    ax[3].set_ylabel("거래대금($)", fontsize=8)
    ax[3].grid(alpha=0.25)
    ticks = list(range(0, len(d), 20))
    t = pd.to_datetime(d["ts"], unit="ms") + pd.Timedelta(hours=9)
    ax[3].set_xticks(ticks)
    ax[3].set_xticklabels([t[i].strftime("%m-%d %H:%M") for i in ticks], fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def send(path, caption):
    if not TOKEN or not CHAT:
        return
    with open(path, "rb") as fh:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto",
                      data={"chat_id": CHAT, "caption": caption[:1000]},
                      files={"photo": fh}, timeout=30)


def alert(df, state, key, ts, pl, kind, coin, inst, bar, title, head, lines, ratio, t1d, t4h, d24):
    """조건 맞으면 차트 보내고 성적표 기록에 올림"""
    if state.get(key) == ts:
        return
    side, entry, sl, tp = pl
    if abs(tp / entry - 1) < MIN_TP:
        return
    draw(df, title, "chart.png", lines, pl)
    cap = (f"{head}\n"
           f"진입가: {fmt(entry)}\n"
           f"손절가: {fmt(sl)} ({(sl / entry - 1) * 100:+.2f}%)\n"
           f"익절가: {fmt(tp)} ({(tp / entry - 1) * 100:+.2f}%)\n"
           f"큰 흐름: 일봉{arrow(t1d)} · 4시간{arrow(t4h)}\n"
           f"거래대금: 평균의 {ratio:.1f}배 ({BAR_NAME[bar]})\n"
           f"봉 거래대금: ${um(df['turn'].iloc[-1])}\n"
           f"24시간 거래대금: ${um(d24)}")
    send("chart.png", cap)
    state[key] = ts
    state.setdefault("open", []).append({
        "inst": inst, "bar": bar, "kind": kind, "side": side,
        "entry": entry, "sl": sl, "tp": tp, "ts": ts})


# ---------------- 3. 성적표 ----------------
def judge(t, df):
    rows = df[df["ts"] > t["ts"]]
    for _, r in rows.iterrows():
        if t["side"] == "롱":
            hit_sl, hit_tp = r["l"] <= t["sl"], r["h"] >= t["tp"]
        else:
            hit_sl, hit_tp = r["h"] >= t["sl"], r["l"] <= t["tp"]
        if hit_sl:                      # 한 봉에서 둘 다 닿으면 손절로 침(보수적으로)
            return "손절", int(r["ts"])
        if hit_tp:
            return "익절", int(r["ts"])
    if len(rows) >= MAX_WAIT:
        return "시간초과", int(rows["ts"].iloc[-1])
    return None, None


def update_scores(state, cache):
    still = []
    for t in state.get("open", []):
        k = (t["inst"], t["bar"])
        try:
            if k not in cache:
                cache[k] = get_candles(*k)
                time.sleep(0.15)
            res, when = judge(t, cache[k])
        except Exception as e:
            print("judge error:", t["inst"], t["bar"], e)
            res = None
        if res:
            state.setdefault("done", []).append(
                {"bar": t["bar"], "kind": t["kind"], "side": t["side"], "res": res, "ts": when})
        else:
            still.append(t)
    state["open"] = still
    state["done"] = state.get("done", [])[-3000:]


def rate_table(rows, field, order):
    out = []
    for name in order:
        sub = [r for r in rows if r[field] == name]
        w = sum(r["res"] == "익절" for r in sub)
        l = sum(r["res"] == "손절" for r in sub)
        if w + l:
            out.append((name, w / (w + l) * 100, w, l))
    return out


def daily_report(state):
    now = datetime.now(KST)
    today = now.strftime("%Y-%m-%d")
    if now.hour < REPORT_HOUR or state.get("report_day") == today:
        return
    since = (now - timedelta(days=REPORT_DAYS)).timestamp() * 1000
    rows = [r for r in state.get("done", []) if r["ts"] >= since]
    w = sum(r["res"] == "익절" for r in rows)
    l = sum(r["res"] == "손절" for r in rows)
    if w + l == 0:
        return
    state["report_day"] = today
    by_bar = rate_table(rows, "bar", [b for b, _ in BARS])
    by_bar = [(BAR_NAME[n], p, a, b) for n, p, a, b in by_bar]
    kinds = []
    for r in rows:
        if r["kind"] not in kinds:
            kinds.append(r["kind"])
    by_kind = sorted(rate_table(rows, "kind", kinds), key=lambda x: x[1])

    fig, ax = plt.subplots(2, 1, figsize=(8, 3 + 0.45 * (len(by_bar) + len(by_kind))),
                           gridspec_kw={"height_ratios": [max(len(by_bar), 1), max(len(by_kind), 1)]})
    for a, data, ttl in ((ax[0], by_bar, "봉 종류별 승률"), (ax[1], by_kind, "알림 종류별 승률")):
        lab = [d[0] for d in data]
        pct = [d[1] for d in data]
        a.barh(lab, pct, color=["#2e9e44" if p >= 50 else "#d62728" for p in pct])
        for i, d in enumerate(data):
            a.text(min(d[1], 88) + 1, i, f"{d[1]:.0f}% ({d[2]}승 {d[3]}패)", va="center", fontsize=9)
        a.axvline(40, color="gray", ls="--", lw=0.8)   # 손익비 1.5면 승률 40%가 본전
        a.set_xlim(0, 100)
        a.set_title(ttl, fontsize=12)
        a.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig("report.png", dpi=110)
    plt.close(fig)
    send("report.png", f"📊 최근 {REPORT_DAYS}일 승률 {w / (w + l) * 100:.0f}% (익절 {w} · 손절 {l}) · 점선 40% 넘으면 이득")


# ---------------- 실행 ----------------
def main():
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    tk = requests.get(BASE + "/api/v5/market/tickers", params={"instType": "SWAP"},
                      timeout=10).json()["data"]
    tk = [t for t in tk if t["instId"].endswith("-USDT-SWAP")]
    vol24 = {t["instId"]: num(t.get("volCcy24h")) * num(t.get("last")) for t in tk}
    tk.sort(key=lambda t: vol24[t["instId"]], reverse=True)
    cache = {}
    for t in tk[:TOP_N]:
        inst = t["instId"]
        coin = inst.replace("-USDT-SWAP", "")
        dfs = {}
        for bar, _ in BARS:
            try:
                raw = get_candles(inst, bar)
                cache[(inst, bar)] = raw
                dfs[bar] = add_indicators(raw.copy())
            except Exception as e:
                print(inst, bar, "error:", e)
            time.sleep(0.15)
        t1d, t4h = trend_of(dfs.get("1D")), trend_of(dfs.get("4H"))
        if t1d is None or t1d != t4h:
            continue                     # 큰 흐름이 엇갈리면 이 코인은 안 봄
        want = "롱" if t1d == "up" else "숏"
        for bar, bn in BARS:
            df = dfs.get(bar)
            if df is None or len(df) < 210:
                continue
            try:
                ts = int(df["ts"].iloc[-1])
                tag = inst + "@" + bar
                ratio = surge_ratio(df)
                d24 = vol24.get(inst, 0)
                # 선 튕김
                hits = bounce_check(df, want) if ratio >= SURGE else []
                if hits:                 # 선 여러 개를 동시에 찍어도 알림은 1개
                    last = df.iloc[-1]
                    tip = last["l"] if want == "롱" else last["h"]
                    label, col = min(hits, key=lambda h: abs(last[h[1]] - tip))
                    both = " · ".join(h[0] for h in hits)
                    pl = bounce_plan(df, col, want)
                    alert(df, state, f"{tag}|튕김", ts, pl, label, coin, inst, bar,
                          f"[{bn}] {coin} {want}진입 · {both} 튕김",
                          f"[{bn}] {want}진입 · {coin} · {both} 튕김",
                          [h[0] for h in hits], ratio, t1d, t4h, d24)
                if ratio < CROSS_SURGE:
                    continue
                # 골크/데크
                for f, s_ in PAIRS:
                    kind = cross_check(df, f, s_)
                    if not kind or (kind == "골크") != (want == "롱") or not candle_ok(df, want):
                        continue
                    pl = cross_plan(df, kind, s_)
                    alert(df, state, f"{tag}|{kind}|{f}-{s_}", ts, pl, kind, coin, inst, bar,
                          f"[{bn}] {coin} {kind} 임박 · EMA{f}×EMA{s_} · {want}진입",
                          f"[{bn}] {kind} 임박 · {coin} · EMA{f}×EMA{s_} → {want}진입",
                          [f"EMA{f}", f"EMA{s_}"], ratio, t1d, t4h, d24)
                # 정배열/역배열
                res = align_check(df)
                if res:
                    kind, f, s_ = res
                    if (kind == "정배열") == (want == "롱") and candle_ok(df, want):
                        pl = cross_plan(df, "골크" if kind == "정배열" else "데크", s_)
                        alert(df, state, f"{tag}|{kind}", ts, pl, kind, coin, inst, bar,
                              f"[{bn}] {coin} {kind} 임박 · {want}진입",
                              f"[{bn}] {kind} 임박 · {coin} · 마지막 남은 EMA{f}×EMA{s_} → {want}진입",
                              [f"EMA{n}" for n in ALIGN], ratio, t1d, t4h, d24)
            except Exception as e:
                print(inst, bar, "error:", e)
    update_scores(state, cache)
    daily_report(state)
    json.dump(state, open(STATE, "w"))


if __name__ == "__main__":
    main()
