#!/usr/bin/env python3
"""
Quantum Edge — Core Signal Backtest & Parameter Sweep
=====================================================
Replicates the weighted scoring signal system from the QE PineScript indicator.
Sweeps key parameters to find conservative long-term timing settings.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════
# INDICATOR CALCULATIONS
# ═══════════════════════════════════════════

def calc_sma(series, period):
    return series.rolling(window=period, min_periods=period).mean()


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = calc_atr(high, low, close, period)
    plus_di = 100 * calc_ema(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * calc_ema(minus_dm, period) / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = calc_ema(dx.fillna(0), period)
    return adx, plus_di, minus_di


def calc_linreg(series, period):
    """Linear regression value (equivalent to ta.linreg(source, length, 0))."""
    def _lr(x):
        n = len(x)
        if n < 2:
            return np.nan
        xi = np.arange(n, dtype=float)
        sx = xi.sum()
        sy = x.sum()
        sxy = (xi * x).sum()
        sx2 = (xi * xi).sum()
        denom = n * sx2 - sx * sx
        if abs(denom) < 1e-10:
            return x[-1]
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        return intercept + slope * (n - 1)
    return series.rolling(window=period, min_periods=period).apply(_lr, raw=True)


def calc_stochastic(close, high, low, period=14):
    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (close - lowest_low) / denom
    return k.fillna(50)


# ═══════════════════════════════════════════
# QUANTUM EDGE SIGNAL ENGINE
# ═══════════════════════════════════════════

class QuantumEdgeStrategy:

    def __init__(self,
                 # Signal thresholds
                 buy_threshold=60.0, sell_threshold=60.0,
                 elite_buy_threshold=85.0, elite_sell_threshold=85.0,
                 # Filter weights
                 squeeze_weight=25.0, rsi_weight=20.0,
                 trend_weight=20.0, volume_weight=15.0,
                 adx_weight=10.0, stoch_weight=10.0,
                 # Technical parameters
                 bb_len=20, bb_mult=2.0,
                 kc_len=20, kc_mult=1.5,
                 rsi_len=14, rsi_buy_level=35, rsi_sell_level=65,
                 ema_length=50,
                 adx_length=14, adx_threshold=18,
                 volume_mult=1.7,
                 stoch_len=14, stoch_buy=25, stoch_sell=75,
                 squeeze_min_duration=3,
                 # Signal controls
                 confirmation_bars=2, signal_cooldown=5,
                 # Costs
                 commission_pct=0.1, slippage_pct=0.05):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.elite_buy_threshold = elite_buy_threshold
        self.elite_sell_threshold = elite_sell_threshold
        self.squeeze_weight = squeeze_weight
        self.rsi_weight = rsi_weight
        self.trend_weight = trend_weight
        self.volume_weight = volume_weight
        self.adx_weight = adx_weight
        self.stoch_weight = stoch_weight
        self.bb_len = bb_len
        self.bb_mult = bb_mult
        self.kc_len = kc_len
        self.kc_mult = kc_mult
        self.rsi_len = rsi_len
        self.rsi_buy_level = rsi_buy_level
        self.rsi_sell_level = rsi_sell_level
        self.ema_length = ema_length
        self.adx_length = adx_length
        self.adx_threshold = adx_threshold
        self.volume_mult = volume_mult
        self.stoch_len = stoch_len
        self.stoch_buy = stoch_buy
        self.stoch_sell = stoch_sell
        self.squeeze_min_duration = squeeze_min_duration
        self.confirmation_bars = confirmation_bars
        self.signal_cooldown = signal_cooldown
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def compute_signals(self, df):
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume'].fillna(0)
        n = len(df)

        # ── Bollinger Bands ──
        bb_basis = calc_sma(close, self.bb_len)
        bb_std = close.rolling(self.bb_len).std()
        bb_upper = bb_basis + self.bb_mult * bb_std
        bb_lower = bb_basis - self.bb_mult * bb_std

        # ── Keltner Channels ──
        kc_basis = calc_sma(close, self.kc_len)
        tr = pd.concat([high - low,
                        (high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()], axis=1).max(axis=1)
        kc_range = calc_sma(tr, self.kc_len)
        kc_upper = kc_basis + kc_range * self.kc_mult
        kc_lower = kc_basis - kc_range * self.kc_mult

        # ── Squeeze Detection ──
        sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        sqz_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)

        # Squeeze duration
        sqz_count = np.zeros(n)
        for i in range(1, n):
            if sqz_on.iloc[i]:
                sqz_count[i] = sqz_count[i-1] + 1
            else:
                sqz_count[i] = 0

        # Squeeze release
        sqz_release = np.zeros(n, dtype=bool)
        for i in range(1, n):
            sqz_release[i] = (sqz_on.iloc[i-1] and sqz_off.iloc[i]
                              and sqz_count[i-1] >= self.squeeze_min_duration)

        # ── Squeeze Momentum (linreg) ──
        highest_h = high.rolling(self.kc_len).max()
        lowest_l = low.rolling(self.kc_len).min()
        avg1 = (highest_h + lowest_l) / 2
        avg2 = calc_sma(close, self.kc_len)
        sqz_source = close - (avg1 + avg2) / 2
        sqz_mom = calc_linreg(sqz_source, self.kc_len)

        mom_up = sqz_mom > 0
        mom_down = sqz_mom < 0

        # ── RSI ──
        rsi = calc_rsi(close, self.rsi_len)
        rsi_buy = rsi < self.rsi_buy_level
        rsi_sell = rsi > self.rsi_sell_level

        # ── EMA Trend ──
        ema = calc_ema(close, self.ema_length)
        above_ema = close > ema * 0.98  # 2% tolerance for buy
        below_ema = close < ema * 1.02  # 2% tolerance for sell

        # ── Volume Spike ──
        avg_vol = calc_sma(volume, 20)
        vol_spike = volume > avg_vol * self.volume_mult

        # ── ADX ──
        adx, _, _ = calc_adx(high, low, close, self.adx_length)
        strong_trend = adx > self.adx_threshold

        # ── Stochastic ──
        stoch = calc_stochastic(close, high, low, self.stoch_len)
        stoch_buy = stoch < self.stoch_buy
        stoch_sell = stoch > self.stoch_sell

        # ═══════════════════════════════════════
        # WEIGHTED SCORING
        # ═══════════════════════════════════════
        core_total = (self.squeeze_weight + self.rsi_weight +
                      self.volume_weight + self.trend_weight + self.adx_weight)
        elite_total = core_total + self.stoch_weight

        sqz_release_s = pd.Series(sqz_release, index=df.index)
        sqz_off_s = sqz_off

        # Core buy score components
        squeeze_buy = ((sqz_release_s & mom_down) | (sqz_off_s & mom_down))
        core_buy_score = (
            squeeze_buy.astype(float) * self.squeeze_weight +
            rsi_buy.astype(float) * self.rsi_weight +
            vol_spike.astype(float) * self.volume_weight +
            above_ema.astype(float) * self.trend_weight +
            strong_trend.astype(float) * self.adx_weight
        )
        core_buy_pct = (core_buy_score / core_total * 100).fillna(0)

        # Core sell score components
        squeeze_sell = ((sqz_release_s & mom_up) | (sqz_off_s & mom_up))
        core_sell_score = (
            squeeze_sell.astype(float) * self.squeeze_weight +
            rsi_sell.astype(float) * self.rsi_weight +
            vol_spike.astype(float) * self.volume_weight +
            below_ema.astype(float) * self.trend_weight +
            strong_trend.astype(float) * self.adx_weight
        )
        core_sell_pct = (core_sell_score / core_total * 100).fillna(0)

        # Elite scores (add stochastic)
        elite_buy_score = core_buy_score + stoch_buy.astype(float) * self.stoch_weight
        elite_buy_pct = (elite_buy_score / elite_total * 100).fillna(0)
        elite_sell_score = core_sell_score + stoch_sell.astype(float) * self.stoch_weight
        elite_sell_pct = (elite_sell_score / elite_total * 100).fillna(0)

        # ═══════════════════════════════════════
        # SIGNAL GENERATION WITH CONFIRMATION + COOLDOWN
        # ═══════════════════════════════════════
        buy_raw = core_buy_pct >= self.buy_threshold
        sell_raw = core_sell_pct >= self.sell_threshold
        elite_buy_raw = elite_buy_pct >= self.elite_buy_threshold
        elite_sell_raw = elite_sell_pct >= self.elite_sell_threshold

        # Confirmation: signal must persist for N bars
        buy_conf = buy_raw.copy()
        sell_conf = sell_raw.copy()
        elite_buy_conf = elite_buy_raw.copy()
        elite_sell_conf = elite_sell_raw.copy()
        if self.confirmation_bars > 1:
            for lag in range(1, self.confirmation_bars):
                buy_conf = buy_conf & buy_raw.shift(lag).fillna(False)
                sell_conf = sell_conf & sell_raw.shift(lag).fillna(False)
                elite_buy_conf = elite_buy_conf & elite_buy_raw.shift(lag).fillna(False)
                elite_sell_conf = elite_sell_conf & elite_sell_raw.shift(lag).fillna(False)

        # Apply cooldown
        buy_arr = buy_conf.values.astype(bool)
        sell_arr = sell_conf.values.astype(bool)
        eb_arr = elite_buy_conf.values.astype(bool)
        es_arr = elite_sell_conf.values.astype(bool)
        close_arr = close.values

        sig_out = np.zeros(n, dtype=np.int8)
        last_buy_bar = -999
        last_sell_bar = -999

        for i in range(n):
            buy_ok = buy_arr[i] and (i - last_buy_bar >= self.signal_cooldown)
            sell_ok = sell_arr[i] and (i - last_sell_bar >= self.signal_cooldown)

            # Elite overrides standard
            if eb_arr[i] and (i - last_buy_bar >= self.signal_cooldown):
                buy_ok = True
            if es_arr[i] and (i - last_sell_bar >= self.signal_cooldown):
                sell_ok = True

            if buy_ok and not sell_ok:
                sig_out[i] = 1
                last_buy_bar = i
            elif sell_ok and not buy_ok:
                sig_out[i] = -1
                last_sell_bar = i

        signals = pd.Series(sig_out, index=df.index)
        return signals, core_buy_pct, core_sell_pct


# ═══════════════════════════════════════════
# BACKTESTING ENGINE
# ═══════════════════════════════════════════

def backtest(df, signals, commission_pct=0.1, slippage_pct=0.05):
    c = df['Close'].values
    n = len(df)
    sig_arr = signals.values
    cost_pct = (commission_pct + slippage_pct) / 100

    equity = 100000.0
    peak_equity = equity
    position = 0
    entry_price = 0.0
    trades = []
    max_dd = 0.0

    for i in range(1, n):
        if position == 1:
            equity *= c[i] / c[i-1] if c[i-1] > 0 else 1.0

        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd > max_dd:
            max_dd = dd

        sig = sig_arr[i]
        if sig == 1 and position == 0:
            equity -= equity * cost_pct
            entry_price = c[i]
            position = 1
        elif sig == -1 and position == 1:
            equity -= equity * cost_pct
            trades.append((c[i] - entry_price) / entry_price * 100)
            position = 0

    trade_arr = np.array(trades) if trades else np.array([])
    total = len(trade_arr)
    if total > 0:
        wins = int((trade_arr >= 0).sum())
        losses = total - wins
        win_rate = wins / total * 100
        pf_win = float(trade_arr[trade_arr >= 0].sum()) if wins > 0 else 0
        pf_loss = abs(float(trade_arr[trade_arr < 0].sum())) if losses > 0 else 0
        profit_factor = pf_win / pf_loss if pf_loss > 0 else 999
    else:
        wins = losses = 0
        win_rate = 0
        profit_factor = 0

    strat_return = (equity / 100000 - 1) * 100

    # Forward hit rates (13-bar)
    buy_locs = np.where(sig_arr == 1)[0]
    sell_locs = np.where(sig_arr == -1)[0]
    valid_buy = buy_locs[buy_locs + 13 < n]
    valid_sell = sell_locs[sell_locs + 13 < n]
    buy_hit = int((c[valid_buy + 13] > c[valid_buy]).sum()) / len(valid_buy) * 100 if len(valid_buy) > 0 else 0
    sell_hit = int((c[valid_sell + 13] < c[valid_sell]).sum()) / len(valid_sell) * 100 if len(valid_sell) > 0 else 0

    return {
        'trades': total, 'wins': wins, 'losses': losses,
        'win_rate': win_rate, 'profit_factor': profit_factor,
        'max_dd': max_dd, 'return_pct': strat_return,
        'buy_hit_13': buy_hit, 'sell_hit_13': sell_hit,
        'buy_signals': len(buy_locs), 'sell_signals': len(sell_locs),
    }


# ═══════════════════════════════════════════
# PARAMETER SWEEP
# ═══════════════════════════════════════════

def sweep(df, name):
    print(f"\n{'='*75}")
    print(f"  QUANTUM EDGE PARAMETER SWEEP — {name}")
    print(f"{'='*75}")

    # 1) Baseline with defaults
    print(f"\n  ── BASELINE (Default Settings) ──")
    strat = QuantumEdgeStrategy()
    sigs, _, _ = strat.compute_signals(df)
    res = backtest(df, sigs)
    print(f"  Trades: {res['trades']:>3}  WR: {res['win_rate']:>5.1f}%  PF: {res['profit_factor']:>5.2f}  "
          f"Return: {res['return_pct']:>9.1f}%  DD: {res['max_dd']:>5.1f}%  "
          f"BuyHit: {res['buy_hit_13']:>5.1f}%  SellHit: {res['sell_hit_13']:>5.1f}%")

    # 2) Buy/Sell Threshold Sweep
    print(f"\n  ── BUY/SELL THRESHOLD SWEEP ──")
    print(f"  {'Thresh':>6}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'Return%':>10}  {'DD%':>6}  {'BuyHit%':>8}  {'SellHit%':>9}")
    best_hr = {'val': 0, 'thresh': 0}
    for thresh in [40, 50, 55, 60, 65, 70, 75, 80, 85, 90]:
        s = QuantumEdgeStrategy(buy_threshold=thresh, sell_threshold=thresh)
        sigs, _, _ = s.compute_signals(df)
        r = backtest(df, sigs)
        marker = ""
        if r['trades'] > 0 and r['buy_hit_13'] > best_hr['val']:
            best_hr = {'val': r['buy_hit_13'], 'thresh': thresh}
        if r['trades'] > 0 and r['buy_hit_13'] >= 60:
            marker = " ◄"
        print(f"  {thresh:>6}  {r['trades']:>6}  {r['win_rate']:>5.1f}%  {r['profit_factor']:>5.2f}  "
              f"{r['return_pct']:>9.1f}%  {r['max_dd']:>5.1f}%  {r['buy_hit_13']:>7.1f}%  {r['sell_hit_13']:>8.1f}%{marker}")

    # 3) Cooldown Sweep
    print(f"\n  ── SIGNAL COOLDOWN SWEEP (thresh={best_hr['thresh']}) ──")
    print(f"  {'Cool':>4}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'Return%':>10}  {'DD%':>6}  {'BuyHit%':>8}")
    bt = best_hr['thresh'] if best_hr['thresh'] > 0 else 60
    for cool in [3, 5, 8, 10, 15, 20, 25, 30]:
        s = QuantumEdgeStrategy(buy_threshold=bt, sell_threshold=bt, signal_cooldown=cool)
        sigs, _, _ = s.compute_signals(df)
        r = backtest(df, sigs)
        print(f"  {cool:>4}  {r['trades']:>6}  {r['win_rate']:>5.1f}%  {r['profit_factor']:>5.2f}  "
              f"{r['return_pct']:>9.1f}%  {r['max_dd']:>5.1f}%  {r['buy_hit_13']:>7.1f}%")

    # 4) RSI Level Sweep
    print(f"\n  ── RSI LEVEL SWEEP (thresh={bt}, cool=5) ──")
    print(f"  {'RSI_B':>5}  {'RSI_S':>5}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'Return%':>10}  {'BuyHit%':>8}")
    for rsi_b, rsi_s in [(25, 75), (30, 70), (35, 65), (40, 60), (45, 55)]:
        s = QuantumEdgeStrategy(buy_threshold=bt, sell_threshold=bt,
                                rsi_buy_level=rsi_b, rsi_sell_level=rsi_s)
        sigs, _, _ = s.compute_signals(df)
        r = backtest(df, sigs)
        print(f"  {rsi_b:>5}  {rsi_s:>5}  {r['trades']:>6}  {r['win_rate']:>5.1f}%  {r['profit_factor']:>5.2f}  "
              f"{r['return_pct']:>9.1f}%  {r['buy_hit_13']:>7.1f}%")

    # 5) EMA Length Sweep
    print(f"\n  ── EMA LENGTH SWEEP (thresh={bt}) ──")
    print(f"  {'EMA':>4}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'Return%':>10}  {'BuyHit%':>8}")
    for ema_l in [20, 50, 100, 150, 200]:
        s = QuantumEdgeStrategy(buy_threshold=bt, sell_threshold=bt, ema_length=ema_l)
        sigs, _, _ = s.compute_signals(df)
        r = backtest(df, sigs)
        print(f"  {ema_l:>4}  {r['trades']:>6}  {r['win_rate']:>5.1f}%  {r['profit_factor']:>5.2f}  "
              f"{r['return_pct']:>9.1f}%  {r['buy_hit_13']:>7.1f}%")

    # 6) Weight Tuning — heavier squeeze + trend
    print(f"\n  ── WEIGHT CONFIGURATION SWEEP (thresh={bt}) ──")
    print(f"  {'Config':>20}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'Return%':>10}  {'BuyHit%':>8}")
    configs = [
        ("Default 25/20/20/15/10", 25, 20, 20, 15, 10),
        ("Heavy Squeeze 40/15/15/15/15", 40, 15, 15, 15, 15),
        ("Heavy Trend 20/15/35/15/15", 20, 15, 35, 15, 15),
        ("Squeeze+RSI 30/30/15/15/10", 30, 30, 15, 15, 10),
        ("Squeeze+Trend 30/15/30/15/10", 30, 15, 30, 15, 10),
        ("Equal 20/20/20/20/20", 20, 20, 20, 20, 20),
        ("RSI+Trend 15/30/30/15/10", 15, 30, 30, 15, 10),
    ]
    for label, sw, rw, tw, vw, aw in configs:
        s = QuantumEdgeStrategy(buy_threshold=bt, sell_threshold=bt,
                                squeeze_weight=sw, rsi_weight=rw,
                                trend_weight=tw, volume_weight=vw, adx_weight=aw)
        sigs, _, _ = s.compute_signals(df)
        r = backtest(df, sigs)
        print(f"  {label:>20}  {r['trades']:>6}  {r['win_rate']:>5.1f}%  {r['profit_factor']:>5.2f}  "
              f"{r['return_pct']:>9.1f}%  {r['buy_hit_13']:>7.1f}%")

    # 7) Confirmation Bars Sweep
    print(f"\n  ── CONFIRMATION BARS SWEEP (thresh={bt}) ──")
    print(f"  {'Conf':>4}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'Return%':>10}  {'BuyHit%':>8}")
    for cb in [1, 2, 3]:
        s = QuantumEdgeStrategy(buy_threshold=bt, sell_threshold=bt, confirmation_bars=cb)
        sigs, _, _ = s.compute_signals(df)
        r = backtest(df, sigs)
        print(f"  {cb:>4}  {r['trades']:>6}  {r['win_rate']:>5.1f}%  {r['profit_factor']:>5.2f}  "
              f"{r['return_pct']:>9.1f}%  {r['buy_hit_13']:>7.1f}%")

    # 8) Combined Best Conservative
    print(f"\n  ── COMBINED CONSERVATIVE CONFIGURATIONS ──")
    print(f"  {'Label':>30}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'Return%':>10}  {'DD%':>6}  {'BuyHit%':>8}")
    combos = [
        ("Default", dict()),
        ("Conservative A (T70,C10)", dict(buy_threshold=70, sell_threshold=70, signal_cooldown=10)),
        ("Conservative B (T75,C10)", dict(buy_threshold=75, sell_threshold=75, signal_cooldown=10)),
        ("Conservative C (T70,C15)", dict(buy_threshold=70, sell_threshold=70, signal_cooldown=15)),
        ("Ultra Cons (T80,C20)", dict(buy_threshold=80, sell_threshold=80, signal_cooldown=20)),
        ("RSI Tight (T70,RSI30/70,C10)", dict(buy_threshold=70, sell_threshold=70, rsi_buy_level=30, rsi_sell_level=70, signal_cooldown=10)),
        ("Long EMA (T70,EMA200,C10)", dict(buy_threshold=70, sell_threshold=70, ema_length=200, signal_cooldown=10)),
        ("Squeeze Heavy (T70,Sq40,C10)", dict(buy_threshold=70, sell_threshold=70, squeeze_weight=40, rsi_weight=15, trend_weight=15, volume_weight=15, adx_weight=15, signal_cooldown=10)),
        ("High Conf (T70,Conf3,C10)", dict(buy_threshold=70, sell_threshold=70, confirmation_bars=3, signal_cooldown=10)),
        ("Best Mix (T70,RSI30/70,EMA200,C10)", dict(buy_threshold=70, sell_threshold=70, rsi_buy_level=30, rsi_sell_level=70, ema_length=200, signal_cooldown=10)),
    ]
    for label, kwargs in combos:
        s = QuantumEdgeStrategy(**kwargs)
        sigs, _, _ = s.compute_signals(df)
        r = backtest(df, sigs)
        marker = " ★" if r['trades'] >= 5 and r['buy_hit_13'] >= 60 and r['win_rate'] >= 50 else ""
        print(f"  {label:>30}  {r['trades']:>6}  {r['win_rate']:>5.1f}%  {r['profit_factor']:>5.2f}  "
              f"{r['return_pct']:>9.1f}%  {r['max_dd']:>5.1f}%  {r['buy_hit_13']:>7.1f}%{marker}")


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def load_data(filepath, name):
    try:
        df = pd.read_csv(filepath, parse_dates=['Date'])
        df = df.sort_values('Date').reset_index(drop=True).set_index('Date')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns and col.lower() in df.columns:
                df = df.rename(columns={col.lower(): col})
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        df = df.dropna(subset=['Close'])
        print(f"  {name}: {len(df)} bars ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
        return df
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    print("="*75)
    print("  QUANTUM EDGE — PARAMETER OPTIMIZATION FOR CONSERVATIVE TIMING")
    print("="*75)
    print("\nLoading data...")

    btc = load_data('/home/user/Rockstar-Indicator/data/BTC-USD_daily.csv', 'BTC-USD')
    spy = load_data('/home/user/Rockstar-Indicator/data/SPY_daily.csv', 'SPY')

    if btc is not None and len(btc) >= 300:
        sweep(btc, 'BTC-USD')
    if spy is not None and len(spy) >= 300:
        sweep(spy, 'SPY')

    print(f"\n{'='*75}")
    print("  SWEEP COMPLETE — Look for ★ markers on best conservative configs")
    print(f"{'='*75}")


if __name__ == '__main__':
    main()
