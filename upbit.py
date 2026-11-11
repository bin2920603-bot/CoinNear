# 업비트 원화마켓 근접알림 (main.py의 계산·차트를 그대로 씀)
# 업비트는 현물이라 '롱'(사기)만 알림
import os, json, time
import requests
import pandas as pd
import matplotlib.axes
import main as m

UP = "https://api.upbit.com/v1"
TOP_N = 60          # 거래대금 상위 몇 개 코인을 볼지
STATE = "upbit_state.json"
UNITS = {"1D": "days", "4H": "minutes/240", "1H": "minutes/60", "15m": "minutes/15"}

# 차트 아래 칸 이름을 원화로
_ylabel = matplotlib.axes.Axes.set_ylabel
matplotlib.axes.Axes.set_ylabel = lambda self, s, *a, **k: _ylabel(self, s.replace("($)", "(원)"), *a, **k)

# 업비트는 따로 만든 알림방(CHAT_UPBIT)으로 보냄
m.CHAT = os.environ.get("CHAT_UPBIT")
m.DAY_OPEN_KST = 9    # 업비트 일봉은 한국시간 오전 9시에 시작·마감


# (지지·저항 박스·구름·▲▼·하이킨아시는 이제 main.py 카드가 직접 그림)


def candles(market, bar, live=False):
    rows, to = [], None
    for _ in range(2):                       # 업비트는 한 번에 200개라 두 번 받음
        p = {"market": market, "count": 200}
        if to:
            p["to"] = to
        r = requests.get(f"{UP}/candles/{UNITS[bar]}", params=p, timeout=10).json()
        if not isinstance(r, list) or not r:
            break
        rows += r
        to = r[-1]["candle_date_time_utc"].replace("T", " ")
        time.sleep(0.12)
    df = pd.DataFrame({
        "ts": [pd.Timestamp(x["candle_date_time_utc"], tz="UTC").timestamp() * 1000 for x in rows],
        "o": [x["opening_price"] for x in rows], "h": [x["high_price"] for x in rows],
        "l": [x["low_price"] for x in rows], "c": [x["trade_price"] for x in rows],
        "volc": [x["candle_acc_trade_volume"] for x in rows],
        "turn": [x["candle_acc_trade_price"] for x in rows]}).astype(float)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    if live:
        return df                                # 진행 중인 봉까지 포함
    return df.iloc[:-1].reset_index(drop=True)   # 맨 끝(아직 안 끝난 봉) 버림


m.get_candles = candles


def won(v):
    return m.um(v) + "원"


def alert(df, state, key, ts, pl, kind, coin, market, bar, title, head, lines, ratio, t1d, t4h, d24, dfs):
    if state.get(key) == ts:
        return
    side, entry, sl, tp = pl
    if abs(tp / entry - 1) < m.MIN_TP:
        return
    cards = dict(dfs)
    cards[bar] = df                          # 신호가 난 봉은 진행 중인 봉까지 포함해서 그림
    m.draw_card(cards, {bar: lines}, title, pl, bar, "chart.png")   # 일봉·4시간·1시간·15분 4칸 카드
    cap = (f"{head}\n"
           f"진입가: {m.fmt(entry)}원\n"
           f"손절가: {m.fmt(sl)}원 ({(sl / entry - 1) * 100:+.2f}%)\n"
           f"익절가: {m.fmt(tp)}원 ({(tp / entry - 1) * 100:+.2f}%)\n"
           f"큰 흐름: 일봉{m.arrow(t1d)} · 4시간{m.arrow(t4h)}\n"
           f"거래대금: 평균의 {ratio:.1f}배 ({m.BAR_NAME[bar]})\n"
           f"봉 거래대금: {won(df['turn'].iloc[-1])}\n"
           f"24시간 거래대금: {won(d24)}")
    m.send("chart.png", cap)
    state[key] = ts
    state.setdefault("open", []).append({
        "inst": market, "bar": bar, "kind": kind, "side": side,
        "entry": entry, "sl": sl, "tp": tp, "ts": ts})


def main():
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    mk = [x["market"] for x in requests.get(f"{UP}/market/all", timeout=10).json()
          if x["market"].startswith("KRW-")]
    tk = requests.get(f"{UP}/ticker", params={"markets": ",".join(mk)}, timeout=10).json()
    vol24 = {t["market"]: m.num(t.get("acc_trade_price_24h")) for t in tk}
    top = sorted(vol24, key=vol24.get, reverse=True)[:TOP_N]
    cache = {}
    for market in top:
        coin = market.replace("KRW-", "")
        dfs, lives = {}, {}
        for bar, _ in m.BARS:
            try:
                full = candles(market, bar, live=True)
                raw = full.iloc[:-1].reset_index(drop=True)
                cache[(market, bar)] = raw
                dfs[bar] = m.add_indicators(raw.copy())
                if bar in m.LIVE:
                    lives[bar] = m.add_indicators(full.copy())   # 진행 중인 봉 포함
            except Exception as e:
                print(market, bar, "error:", e)
        t1d, t4h = m.trend_of(dfs.get("1D")), m.trend_of(dfs.get("4H"))
        if not (t1d == "up" and t4h == "up"):
            continue                       # 현물은 둘 다 오를 때만 (롱만)
        want = "롱"
        for bar, bn in m.BARS:
            df = lives.get(bar, dfs.get(bar))   # 4시간·일봉은 진행 중인 봉으로 봄
            if df is None or len(df) < 210:
                continue
            try:
                ts = int(df["ts"].iloc[-1])
                tag = market + "@" + bar
                ratio = m.surge_ratio(df, m.LIVE.get(bar) if bar in lives else None)
                d24 = vol24.get(market, 0)
                sig = m.line_signal(df, want) if ratio >= m.SURGE else None
                if sig:                          # 튕김이 먼저, 봉 하나에 알림 1개
                    kind, label, col, both, names_ = sig
                    pl = m.near_plan(df, col, want)
                    alert(df, state, f"{tag}|선", ts, pl, f"{label} {kind}", coin, market, bar,
                          f"[{bn}] {coin} 매수 · {both} {kind}",
                          f"[{bn}] 매수 · {coin} · {both} {kind}",
                          names_, ratio, t1d, t4h, d24, dfs)
                if ratio < m.CROSS_SURGE or not m.candle_ok(df, want):
                    continue
                for f, s_ in m.PAIRS:
                    if m.cross_check(df, f, s_) != "골크":
                        continue
                    pl = m.cross_plan(df, "골크", s_)
                    alert(df, state, f"{tag}|골크|{f}-{s_}", ts, pl, "골크", coin, market, bar,
                          f"[{bn}] {coin} 골크 임박 · EMA{f}×EMA{s_} · 매수",
                          f"[{bn}] 골크 임박 · {coin} · EMA{f}×EMA{s_} → 매수",
                          [f"EMA{f}", f"EMA{s_}"], ratio, t1d, t4h, d24, dfs)
                res = m.align_check(df)
                if res and res[0] == "정배열":
                    _, f, s_ = res
                    pl = m.cross_plan(df, "골크", s_)
                    alert(df, state, f"{tag}|정배열", ts, pl, "정배열", coin, market, bar,
                          f"[{bn}] {coin} 정배열 임박 · 매수",
                          f"[{bn}] 정배열 임박 · {coin} · 마지막 남은 EMA{f}×EMA{s_} → 매수",
                          [f"EMA{n}" for n in m.ALIGN], ratio, t1d, t4h, d24, dfs)
            except Exception as e:
                print(market, bar, "error:", e)
    m.update_scores(state, cache)
    m.daily_report(state)
    json.dump(state, open(STATE, "w"))


if __name__ == "__main__":
    main()
