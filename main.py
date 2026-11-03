import os, json, time
from concurrent.futures import ThreadPoolExecutor
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
SURGE = 2.0        # 선 근처 알림: 거래대금 평균의 2배 이상
NEAR = 0.005       # 선 근처 = 위아래 0.5%
TOUCH = 0.001      # 꼬리가 선의 0.1% 안까지 오면 '닿았다'(튕김)
SHOW = 120         # 차트에 보여줄 봉 개수
RR = 1.5           # 익절 거리 = 손절 거리의 몇 배
MIN_TP = 0.01      # 익절이 진입가의 1%보다 작으면 알림 안 보냄
MAX_SL = 0.05      # 손절이 진입가에서 5%보다 멀면 알림 안 보냄
MAX_WAIT = 96      # 이 봉 수 안에 익절/손절 둘 다 안 닿으면 '시간초과'
REPORT_HOUR = 9    # 매일 이 시각(한국시간) 이후 첫 실행 때 성적표 보냄
REPORT_DAYS = 7    # 성적표는 최근 며칠치
# ---- 새로 넣은 설정 ----
ATR_MULT = 1.5       # 손절은 최소 ATR(평소 흔들림)의 1.5배 이상 떨어뜨림
HOT = 0.15           # 24시간에 이미 15% 넘게 오른 코인은 롱 안 봄(15% 넘게 빠진 코인은 숏 안 봄)
FUND_SHORT = -0.0002 # 펀딩비가 -0.02% 이하(숏이 너무 몰림)면 숏 안 봄
WALL_FRAC = 0.5      # 진입~익절 거리의 절반 안에 큰 선(벽)이 있으면 대기로 보냄
WAIT_HOURS = 8       # 벽 때문에 대기 중인 코인은 몇 시간까지 기다릴지
COOLDOWN_MIN = 60    # 같은 코인·같은 방향은 이 시간(분) 안에 한 번만 알림(겹치면 한 알림에 합침)
WORKERS = 4          # 코인 봉 데이터를 동시에 몇 개씩 받을지(속도용. OKX 요청 제한 때문에 너무 크게 하지 않음)
SLOPE_BARS = 5       # 선의 방향을 볼 봉 개수
SLOPE_ATR = 0.3      # 그 사이 선이 ATR의 0.3배 이상 움직여야 '방향 있음'
SLOPE_LINES = ("EMA10", "거가중100")   # 방향을 확인할 선 (승률이 낮았던 것)
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
WALL_LINES = [("EMA50", "ema50"), ("EMA120", "ema120"), ("EMA200", "ema200"), ("거가중100", "vwap100")]
BAR_NAME = dict(BARS)
LIVE = {"1D": 86400000, "4H": 14400000}   # 이 봉들은 끝나기 전에도 5분마다 봄(봉 길이 ms)


# ---------------- 데이터 ----------------
def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def get_candles(inst, bar, live=False):
    r = []
    for _ in range(3):                    # 요청이 몰려 빈 답이 오면 잠깐 쉬고 다시 시도
        r = requests.get(BASE + "/api/v5/market/candles",
                         params={"instId": inst, "bar": bar, "limit": 300},
                         timeout=10).json()["data"]
        if r:
            break
        time.sleep(0.6)
    df = pd.DataFrame(r).iloc[:, :9]
    df.columns = ["ts", "o", "h", "l", "c", "vol", "volc", "turn", "confirm"]
    for k in df.columns:
        df[k] = df[k].astype(float)
    df = df.sort_values("ts").reset_index(drop=True)
    if not live:
        df = df[df["confirm"] == 1].reset_index(drop=True)
    if df["turn"].sum() == 0:
        df["turn"] = df["volc"] * df["c"]
    return df


def get_funding(inst):
    """펀딩비. 못 가져오면 0(거르지 않음)"""
    try:
        r = requests.get(BASE + "/api/v5/public/funding-rate",
                         params={"instId": inst}, timeout=10).json()["data"]
        return num(r[0]["fundingRate"])
    except Exception:
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


def surge_ratio(df, period=None):
    """마지막 봉 거래대금이 직전 20봉 평균의 몇 배인지.
    아직 안 끝난 봉(period 있음)은 지금 속도로 봉 끝까지 갔을 때 거래대금으로 계산"""
    base = df["turn"].iloc[-21:-1].mean()
    turn = df["turn"].iloc[-1]
    if period:
        frac = (time.time() * 1000 - df["ts"].iloc[-1]) / period
        turn = turn / min(max(frac, 0.25), 1)
    return turn / base if base > 0 else 0


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


def market_ok(side, btc):
    """비트코인 일봉이 내리는 중이면 롱 쉼, 오르는 중이면 숏 쉼. 애매하면 둘 다 받음"""
    return not ((side == "롱" and btc == "down") or (side == "숏" and btc == "up"))


# ---------------- 2. 선 근처 (위·아래 둘 다) ----------------
def line_dir_ok(df, col, want):
    """선이 진입 방향으로 기울어 있는지 (롱=오르는 선, 숏=내리는 선)"""
    if len(df) <= SLOPE_BARS:
        return False
    now, before, atr = df[col].iloc[-1], df[col].iloc[-1 - SLOPE_BARS], df["atr"].iloc[-1]
    if pd.isna(now) or pd.isna(before) or pd.isna(atr):
        return False
    move = now - before
    return move >= SLOPE_ATR * atr if want == "롱" else move <= -SLOPE_ATR * atr


def near_check(df, want):
    """마감된 마지막 봉의 종가가 선 위아래 0.5% 안에 있고, 봉 색이 방향과 맞는지"""
    last = df.iloc[-1]
    if not candle_ok(df, want):
        return []
    hits = []
    for label, col, _ in LINES:
        v = last[col]
        if (pd.notna(v) and v > 0 and abs(last["c"] - v) / v <= NEAR
                and (label not in SLOPE_LINES or line_dir_ok(df, col, want))):
            hits.append((label, col))
    return hits


def bounce_check(df, want):
    """마감된 마지막 봉이 꼬리로 선을 찍고 반대편에서 마감했는지"""
    last = df.iloc[-1]
    if not candle_ok(df, want):
        return []
    hits = []
    for label, col, _ in LINES:
        v = last[col]
        if pd.isna(v) or v <= 0:
            continue
        if label in SLOPE_LINES and not line_dir_ok(df, col, want):
            continue
        if want == "롱" and last["l"] <= v * (1 + TOUCH) and last["c"] > v:
            hits.append((label, col))
        if want == "숏" and last["h"] >= v * (1 - TOUCH) and last["c"] < v:
            hits.append((label, col))
    return hits


def line_signal(df, want):
    """튕김이 있으면 튕김, 없으면 근처. 둘 다 없으면 None"""
    last = df.iloc[-1]
    hits, kind = bounce_check(df, want), "튕김"
    if not hits:
        hits, kind = near_check(df, want), "근처"
    if not hits:
        return None
    label, col = min(hits, key=lambda h: abs(last[h[1]] - last["c"]))
    return kind, label, col, " · ".join(h[0] for h in hits), [h[0] for h in hits]


def limit_sl(pl):
    """손절이 너무 멀면 익절=진입으로 바꿔서 알림이 안 나가게 함"""
    side, entry, sl, tp = pl
    if abs(sl / entry - 1) > MAX_SL:
        return side, entry, entry, entry
    return pl


def near_plan(df, col, side):
    """손절은 꼬리·선 아래(위)에서 조금 더 떨어뜨리되, 최소 ATR의 1.5배는 확보"""
    last = df.iloc[-1]
    entry, line, atr = last["c"], last[col], last["atr"]
    if side == "롱":
        sl = min(last["l"], line) - 0.3 * atr
        sl = min(sl, entry - ATR_MULT * atr)
        return limit_sl((side, entry, sl, entry + RR * (entry - sl)))
    sl = max(last["h"], line) + 0.3 * atr
    sl = max(sl, entry + ATR_MULT * atr)
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
        sl = min(sl, entry - ATR_MULT * atr)
        return limit_sl(("롱", entry, sl, entry + RR * (entry - sl)))
    sl = max(df["h"].iloc[-5:].max(), slow) + 0.3 * atr
    sl = max(sl, entry + ATR_MULT * atr)
    return limit_sl(("숏", entry, sl, entry - RR * (sl - entry)))


def candle_ok(df, side):
    last = df.iloc[-1]
    return (side == "롱" and last["c"] >= last["o"]) or (side == "숏" and last["c"] <= last["o"])


# ---------------- 벽(바로 앞의 큰 선) ----------------
def wall_check(df, pl, skip):
    """진입가에서 익절 방향으로 절반 거리 안에 큰 선(벽)이 있으면 (이름, 컬럼, 거리) 돌려줌.
    알림을 만든 선 자신(skip)은 벽으로 치지 않음"""
    side, entry, sl, tp = pl
    last = df.iloc[-1]
    reach = abs(tp - entry) * WALL_FRAC
    best = None
    for label, col in WALL_LINES:
        if label in skip:
            continue
        v = last[col]
        if pd.isna(v) or v <= 0:
            continue
        d = (v - entry) if side == "롱" else (entry - v)
        if 0 < d <= reach and (best is None or d < best[2]):
            best = (label, col, d)
    return best


def add_wait(state, inst, bar, side, label, col, ts, t1d, t4h):
    w = state.setdefault("wait", [])
    for x in w:
        if x["inst"] == inst and x["bar"] == bar and x["side"] == side and x["col"] == col:
            return
    w.append({"inst": inst, "bar": bar, "side": side, "label": label, "col": col,
              "ts": ts, "made": int(time.time() * 1000), "t1d": t1d, "t4h": t4h})


def queue(cands, state, inst, c, skip, t1d, t4h):
    """후보를 모음. 바로 앞에 벽이 있으면 알림 대신 대기 목록으로 보냄"""
    if state.get(c["key"]) == c["ts"]:
        return
    wall = wall_check(c["df"], c["pl"], skip)
    if wall:
        add_wait(state, inst, c["bar"], c["pl"][0], wall[0], wall[1], c["ts"], t1d, t4h)
        return
    cands.append(c)


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


def emit(state, inst, cands, t1d, t4h, d24, btc):
    """한 코인에 모인 신호들을 알림 하나로 합쳐서 보냄. 큰 봉 신호를 대표로 씀"""
    cands = [c for c in cands if abs(c["pl"][3] / c["pl"][1] - 1) >= MIN_TP]
    if not cands:
        return
    order = {b: i for i, (b, _) in enumerate(BARS)}
    cands.sort(key=lambda c: order[c["bar"]])
    p = cands[0]
    side, entry, sl, tp = p["pl"]
    draw(p["df"], p["title"], "chart.png", p["lines"], p["pl"])
    extra = ""
    if len(cands) > 1:
        listed = " / ".join(f"{BAR_NAME[c['bar']]} {c['kind']}" for c in cands[:6])
        extra = f"겹침: 신호 {len(cands)}개 ({listed})\n"
    cap = (f"{p['head']}\n"
           f"진입가: {fmt(entry)}\n"
           f"손절가: {fmt(sl)} ({(sl / entry - 1) * 100:+.2f}%)\n"
           f"익절가: {fmt(tp)} ({(tp / entry - 1) * 100:+.2f}%)\n"
           f"큰 흐름: 일봉{arrow(t1d)} · 4시간{arrow(t4h)}\n"
           f"시장(비트코인 일봉): {arrow(btc)}\n"
           f"{extra}"
           f"거래대금: 평균의 {p['ratio']:.1f}배 ({BAR_NAME[p['bar']]})\n"
           f"봉 거래대금: ${um(p['df']['turn'].iloc[-1])}\n"
           f"24시간 거래대금: ${um(d24)}")
    send("chart.png", cap)
    for c in cands:
        state[c["key"]] = c["ts"]
    state.setdefault("open", []).append({
        "inst": inst, "bar": p["bar"], "kind": p["kind"], "side": side,
        "entry": entry, "sl": sl, "tp": tp, "ts": p["ts"]})
    state.setdefault("coin_last", {})[inst + "|" + side] = int(time.time() * 1000)


def check_waits(state, cache, chg, vol24, btc):
    """벽 때문에 대기 중인 코인이 벽 위(아래)에서 마감하면 '벽 돌파'로 다시 알림"""
    now = int(time.time() * 1000)
    keep = []
    for w in state.get("wait", []):
        if now - w["made"] > WAIT_HOURS * 3600 * 1000:
            continue
        try:
            inst, bar, side = w["inst"], w["bar"], w["side"]
            raw = cache.get((inst, bar))
            if raw is None:
                raw = get_candles(inst, bar)
                cache[(inst, bar)] = raw
                time.sleep(0.15)
            if len(raw) < 210:
                keep.append(w)
                continue
            df = add_indicators(raw.copy())
            last = df.iloc[-1]
            v = last[w["col"]]
            broke = (pd.notna(v) and last["ts"] >= w["ts"] and candle_ok(df, side)
                     and ((side == "롱" and last["c"] > v) or (side == "숏" and last["c"] < v)))
            if not broke:
                keep.append(w)
                continue
            c24 = chg.get(inst, 0)
            if (not market_ok(side, btc) or (side == "롱" and c24 >= HOT)
                    or (side == "숏" and c24 <= -HOT)):
                continue                      # 돌파는 했지만 시장·과열 조건이 안 맞으면 버림
            pl = near_plan(df, w["col"], side)
            coin = inst.replace("-USDT-SWAP", "")
            bn = BAR_NAME[bar]
            verb = "위로" if side == "롱" else "아래로"
            c = {"key": f"{inst}@{bar}|벽|{w['col']}", "ts": int(last["ts"]), "bar": bar,
                 "kind": "벽돌파", "pl": pl, "df": df, "lines": [w["label"]],
                 "ratio": surge_ratio(df),
                 "title": f"[{bn}] {coin} 벽 돌파 · {w['label']} {verb} 마감 · {side}진입",
                 "head": f"[{bn}] 벽 돌파 · {coin} · {w['label']} {verb} 마감 → {side}진입"}
            if state.get(c["key"]) != c["ts"]:
                emit(state, inst, [c], w.get("t1d"), w.get("t4h"), vol24.get(inst, 0), btc)
        except Exception as e:
            print("wait error:", w.get("inst"), w.get("bar"), e)
            keep.append(w)
    state["wait"] = keep


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
def fetch_coin(inst):
    """한 코인의 봉 4종을 받아서 (마감봉 원본, 지표 붙인 마감봉, 진행 중 포함 봉)으로 돌려줌"""
    raws, dfs, lives = {}, {}, {}
    for bar, _ in BARS:
        try:
            full = get_candles(inst, bar, live=True)
            raw = full[full["confirm"] == 1].reset_index(drop=True)
            raws[bar] = raw
            dfs[bar] = add_indicators(raw.copy())
            if bar in LIVE and full["confirm"].iloc[-1] == 0:
                lives[bar] = add_indicators(full.copy())   # 진행 중인 봉 포함
        except Exception as e:
            print(inst, bar, "error:", e)
        time.sleep(0.15)
    return raws, dfs, lives


def main():
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    now_ms = int(time.time() * 1000)
    state["coin_last"] = {k: v for k, v in state.get("coin_last", {}).items()
                          if now_ms - v < 86400000}
    # 시장 방향: 비트코인 일봉(진행 중인 봉 포함)
    btc = None
    try:
        btc = trend_of(add_indicators(get_candles("BTC-USDT-SWAP", "1D", live=True)))
    except Exception as e:
        print("btc error:", e)
    tk = requests.get(BASE + "/api/v5/market/tickers", params={"instType": "SWAP"},
                      timeout=10).json()["data"]
    tk = [t for t in tk if t["instId"].endswith("-USDT-SWAP")]
    vol24 = {t["instId"]: num(t.get("volCcy24h")) * num(t.get("last")) for t in tk}
    chg = {}
    for t in tk:
        o = num(t.get("open24h"))
        chg[t["instId"]] = num(t.get("last")) / o - 1 if o > 0 else 0
    tk.sort(key=lambda t: vol24[t["instId"]], reverse=True)
    cache = {}
    top = [t["instId"] for t in tk[:TOP_N]]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:     # 봉 받기만 동시에 함(속도용)
        fetched = list(ex.map(fetch_coin, top))
    for inst, (raws, dfs, lives) in zip(top, fetched):
        coin = inst.replace("-USDT-SWAP", "")
        for bar, raw in raws.items():
            cache[(inst, bar)] = raw
        t1d, t4h = trend_of(dfs.get("1D")), trend_of(dfs.get("4H"))
        if t1d is None or t1d != t4h:
            continue                     # 큰 흐름이 엇갈리면 이 코인은 안 봄
        want = "롱" if t1d == "up" else "숏"
        if not market_ok(want, btc):
            continue                     # 시장(비트코인)이 반대로 가는 날은 쉼
        if (want == "롱" and chg.get(inst, 0) >= HOT) or (want == "숏" and chg.get(inst, 0) <= -HOT):
            continue                     # 이미 너무 많이 움직인 코인은 안 봄
        if want == "숏" and get_funding(inst) <= FUND_SHORT:
            continue                     # 숏이 너무 몰린 코인은 숏 안 봄
        cands = []
        for bar, bn in BARS:
            df = lives.get(bar, dfs.get(bar))   # 4시간·일봉은 진행 중인 봉으로 봄
            if df is None or len(df) < 210:
                continue
            try:
                ts = int(df["ts"].iloc[-1])
                tag = inst + "@" + bar
                ratio = surge_ratio(df, LIVE.get(bar) if bar in lives else None)
                # 선 튕김 / 선 근처 (튕김이 먼저, 봉 하나에 후보 1개)
                sig = line_signal(df, want) if ratio >= SURGE else None
                if sig:
                    kind, label, col, both, names_ = sig
                    pl = near_plan(df, col, want)
                    queue(cands, state, inst,
                          {"key": f"{tag}|선", "ts": ts, "bar": bar, "kind": f"{label} {kind}",
                           "pl": pl, "df": df, "lines": names_, "ratio": ratio,
                           "title": f"[{bn}] {coin} {want}진입 · {both} {kind}",
                           "head": f"[{bn}] {want}진입 · {coin} · {both} {kind}"},
                          names_, t1d, t4h)
                if ratio < CROSS_SURGE:
                    continue
                # 골크/데크
                for f, s_ in PAIRS:
                    kind = cross_check(df, f, s_)
                    if not kind or (kind == "골크") != (want == "롱") or not candle_ok(df, want):
                        continue
                    if kind == "골크" and not line_dir_ok(df, f"ema{s_}", want):
                        continue                 # 골크는 느린 선이 오르는 중일 때만
                    pl = cross_plan(df, kind, s_)
                    queue(cands, state, inst,
                          {"key": f"{tag}|{kind}|{f}-{s_}", "ts": ts, "bar": bar, "kind": kind,
                           "pl": pl, "df": df, "lines": [f"EMA{f}", f"EMA{s_}"], "ratio": ratio,
                           "title": f"[{bn}] {coin} {kind} 임박 · EMA{f}×EMA{s_} · {want}진입",
                           "head": f"[{bn}] {kind} 임박 · {coin} · EMA{f}×EMA{s_} → {want}진입"},
                          [f"EMA{f}", f"EMA{s_}"], t1d, t4h)
                # 정배열/역배열
                res = align_check(df)
                if res:
                    kind, f, s_ = res
                    if (kind == "정배열") == (want == "롱") and candle_ok(df, want):
                        pl = cross_plan(df, "골크" if kind == "정배열" else "데크", s_)
                        queue(cands, state, inst,
                              {"key": f"{tag}|{kind}", "ts": ts, "bar": bar, "kind": kind,
                               "pl": pl, "df": df, "lines": [f"EMA{n}" for n in ALIGN], "ratio": ratio,
                               "title": f"[{bn}] {coin} {kind} 임박 · {want}진입",
                               "head": f"[{bn}] {kind} 임박 · {coin} · 마지막 남은 EMA{f}×EMA{s_} → {want}진입"},
                              [f"EMA{n}" for n in ALIGN], t1d, t4h)
            except Exception as e:
                print(inst, bar, "error:", e)
        if cands:
            ck = inst + "|" + want
            if now_ms - state["coin_last"].get(ck, 0) < COOLDOWN_MIN * 60000:
                for c in cands:          # 방금 알린 코인이면 새 알림 대신 합친 것으로 침
                    state[c["key"]] = c["ts"]
            else:
                try:
                    emit(state, inst, cands, t1d, t4h, vol24.get(inst, 0), btc)
                except Exception as e:
                    print(inst, "emit error:", e)
    check_waits(state, cache, chg, vol24, btc)
    update_scores(state, cache)
    daily_report(state)
    json.dump(state, open(STATE, "w"))


if __name__ == "__main__":
    main()
