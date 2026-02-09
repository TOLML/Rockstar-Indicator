#!/usr/bin/env python3
"""
Quantum Edge v2 — Event-Based High-Confidence Backtest
=======================================================
Redesigned signal system:
  - Event-based: triggers on TRANSITIONS, not states
  - Buy = meaningful pullback + reversal confirmation
  - Sell = extended peak + exhaustion confirmation
  - Independent signals (multiple buys/sells in a row OK)
  - Measures forward accuracy at multiple horizons
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════
# INDICATOR CALCULATIONS
# ═══════════════════════════════════════════

def calc_sma(s, p):
    return s.rolling(window=p, min_periods=p).mean()

def calc_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0)
    l = -d.where(d < 0, 0.0)
    ag = g.ewm(alpha=1/p, min_periods=p).mean()
    al = l.ewm(alpha=1/p, min_periods=p).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_atr(high, low, close, p=14):
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, min_periods=p).mean()

def calc_stoch(close, high, low, p=14, smooth_k=3):
    ll = low.rolling(p).min()
    hh = high.rolling(p).max()
    k = (100 * (close - ll) / (hh - ll).replace(0, np.nan)).fillna(50)
    d = k.rolling(smooth_k).mean()
    return k, d

def calc_linreg(series, period):
    def _lr(x):
        n = len(x)
        if n < 2:
            return np.nan
        xi = np.arange(n, dtype=float)
        sx, sy = xi.sum(), x.sum()
        sxy, sx2 = (xi * x).sum(), (xi * xi).sum()
        denom = n * sx2 - sx * sx
        if abs(denom) < 1e-10:
            return x[-1]
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        return intercept + slope * (n - 1)
    return series.rolling(window=period, min_periods=period).apply(_lr, raw=True)

def calc_obv(close, volume):
    sign = np.sign(close.diff()).fillna(0)
    return (sign * volume).cumsum()


# ═══════════════════════════════════════════
# EVENT-BASED SIGNAL ENGINE
# ═══════════════════════════════════════════

class QEv2:
    """
    Event-based system: detects TRANSITIONS not states.

    Buy = pullback event + reversal confirmation + context
    Sell = extension event + exhaustion confirmation + context

    Tier system (all tiers must pass):
      Tier 1: Entry event (required trigger)
      Tier 2: Confirmation (at least N of M)
      Tier 3: Context (at least 1)
    """

    def __init__(self,
                 # Pullback detection
                 pullback_pct=5.0,       # min % drop from recent high for buy
                 pullback_lookback=20,   # bars to look back for recent high
                 # Extension detection
                 extension_pct=8.0,      # min % above SMA for sell
                 # RSI
                 rsi_len=14,
                 rsi_oversold=30,        # event: RSI was below this
                 rsi_recovery=35,        # event: RSI crossed above this
                 rsi_overbought=70,      # event: RSI was above this
                 rsi_exhaustion=65,      # event: RSI crossed below this
                 # Stochastic
                 stoch_len=14,
                 stoch_oversold=20,      # event threshold
                 stoch_overbought=80,
                 # Squeeze
                 bb_len=20, bb_mult=2.0, kc_len=20, kc_mult=1.5,
                 squeeze_min_bars=3,
                 # Moving averages
                 sma_slow=200,
                 ema_fast=50,
                 # Volume
                 vol_mult=1.3,
                 # Confirmation thresholds
                 buy_confirm_min=2,      # min confirmations (out of 4)
                 sell_confirm_min=2,     # min confirmations (out of 4)
                 # Cooldown
                 buy_cooldown=10,
                 sell_cooldown=15,
                 # Adaptivity (for crypto vs stocks)
                 atr_normalize=True):    # auto-adapt pullback to volatility
        self.pullback_pct = pullback_pct
        self.pullback_lookback = pullback_lookback
        self.extension_pct = extension_pct
        self.rsi_len = rsi_len
        self.rsi_oversold = rsi_oversold
        self.rsi_recovery = rsi_recovery
        self.rsi_overbought = rsi_overbought
        self.rsi_exhaustion = rsi_exhaustion
        self.stoch_len = stoch_len
        self.stoch_oversold = stoch_oversold
        self.stoch_overbought = stoch_overbought
        self.bb_len = bb_len
        self.bb_mult = bb_mult
        self.kc_len = kc_len
        self.kc_mult = kc_mult
        self.squeeze_min_bars = squeeze_min_bars
        self.sma_slow = sma_slow
        self.ema_fast = ema_fast
        self.vol_mult = vol_mult
        self.buy_confirm_min = buy_confirm_min
        self.sell_confirm_min = sell_confirm_min
        self.buy_cooldown = buy_cooldown
        self.sell_cooldown = sell_cooldown
        self.atr_normalize = atr_normalize

    def compute(self, df):
        c = df['Close']
        h = df['High']
        lo = df['Low']
        v = df['Volume'].fillna(0)
        n = len(df)

        # ── Core indicators ──
        rsi = calc_rsi(c, self.rsi_len)
        stoch_k, stoch_d = calc_stoch(c, h, lo, self.stoch_len)
        sma_slow = calc_sma(c, self.sma_slow)
        ema_fast = calc_ema(c, self.ema_fast)
        atr = calc_atr(h, lo, c, 14)
        obv = calc_obv(c, v)
        obv_ma = calc_sma(obv, 20)
        avg_vol = calc_sma(v, 20)

        # ── Bollinger + Keltner → Squeeze ──
        bb_basis = calc_sma(c, self.bb_len)
        bb_std = c.rolling(self.bb_len).std()
        bb_upper = bb_basis + self.bb_mult * bb_std
        bb_lower = bb_basis - self.bb_mult * bb_std
        kc_basis = calc_sma(c, self.kc_len)
        tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
        kc_range = calc_sma(tr, self.kc_len)
        kc_upper = kc_basis + kc_range * self.kc_mult
        kc_lower = kc_basis - kc_range * self.kc_mult

        sqz_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        sqz_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)

        sqz_count = np.zeros(n)
        for i in range(1, n):
            sqz_count[i] = sqz_count[i-1] + 1 if sqz_on.iloc[i] else 0
        sqz_release = pd.Series(np.zeros(n, dtype=bool), index=df.index)
        for i in range(1, n):
            sqz_release.iloc[i] = sqz_on.iloc[i-1] and sqz_off.iloc[i] and sqz_count[i-1] >= self.squeeze_min_bars

        # Squeeze momentum
        hh_kc = h.rolling(self.kc_len).max()
        ll_kc = lo.rolling(self.kc_len).min()
        sqz_src = c - ((hh_kc + ll_kc) / 2 + calc_sma(c, self.kc_len)) / 2
        sqz_mom = calc_linreg(sqz_src, self.kc_len)
        mom_increasing = sqz_mom > sqz_mom.shift(1)

        # ── Derived signals ──
        recent_high = h.rolling(self.pullback_lookback).max()
        drawdown_from_high = (recent_high - c) / recent_high * 100

        # Auto-adapt pullback threshold to asset volatility
        if self.atr_normalize:
            avg_atr_pct = (atr / c * 100).rolling(50).mean().fillna(self.pullback_pct)
            adaptive_pullback = (avg_atr_pct * self.pullback_pct / 2).clip(lower=2.0, upper=25.0)
        else:
            adaptive_pullback = pd.Series(self.pullback_pct, index=df.index)

        pct_from_sma = ((c - sma_slow) / sma_slow * 100).fillna(0)

        # ═══════════════════════════════════════
        # BUY EVENT DETECTION
        # ═══════════════════════════════════════

        # Tier 1: TRIGGER EVENT (at least one must fire)
        # A) Meaningful pullback: price dropped >= X% from recent high
        pullback_event = drawdown_from_high >= adaptive_pullback

        # B) SMA200 bounce: price was below SMA200, now crossing back above
        sma_bounce = (c.shift(1) < sma_slow.shift(1)) & (c >= sma_slow)

        # C) Squeeze release with upward momentum
        squeeze_release_up = sqz_release & mom_increasing

        buy_trigger = pullback_event | sma_bounce | squeeze_release_up

        # Tier 2: CONFIRMATION (at least N of 4)
        # Must confirm the pullback is reversing, not continuing
        rsi_was_oversold = rsi.rolling(5).min() < self.rsi_oversold
        rsi_recovering = (rsi > self.rsi_recovery) & rsi_was_oversold

        stoch_reversal = (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1)) & (stoch_k < 50)

        momentum_turn = mom_increasing & (sqz_mom < 0)

        vol_confirm = v > avg_vol * self.vol_mult

        buy_confirmations = (rsi_recovering.astype(int) +
                            stoch_reversal.astype(int) +
                            momentum_turn.astype(int) +
                            vol_confirm.astype(int))

        # Tier 3: CONTEXT (at least one)
        trend_ok = c > sma_slow * 0.90  # long-term trend not broken
        obv_bullish = obv > obv_ma       # accumulation
        near_ema = c < ema_fast * 1.02   # near/below fast EMA (not extended)

        buy_context = trend_ok | obv_bullish | near_ema

        # Combine all tiers
        buy_raw = buy_trigger & (buy_confirmations >= self.buy_confirm_min) & buy_context

        # Strong buy: trigger + 3+ confirmations + multiple context
        strong_buy_raw = buy_trigger & (buy_confirmations >= 3) & (trend_ok & (obv_bullish | near_ema))

        # ═══════════════════════════════════════
        # SELL EVENT DETECTION
        # ═══════════════════════════════════════

        # Tier 1: TRIGGER EVENT
        # A) Price significantly extended above SMA200
        if self.atr_normalize:
            adaptive_extension = (avg_atr_pct * self.extension_pct / 2).clip(lower=3.0, upper=30.0)
        else:
            adaptive_extension = pd.Series(self.extension_pct, index=df.index)
        extended_event = pct_from_sma > adaptive_extension

        # B) RSI was overbought and crossing back down
        rsi_was_overbought = rsi.rolling(5).max() > self.rsi_overbought
        rsi_exhausting = (rsi < self.rsi_exhaustion) & rsi_was_overbought

        sell_trigger = extended_event | rsi_exhausting

        # Tier 2: CONFIRMATION (at least N of 4)
        rsi_declining = (rsi < rsi.shift(1)) & (rsi.shift(1) < rsi.shift(2)) & (rsi > 50)  # 2 bars declining

        stoch_cross_down = (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1)) & (stoch_k > 50)

        momentum_fade = (sqz_mom < sqz_mom.shift(1)) & (sqz_mom > 0)  # losing steam from positive

        vol_dist = v > avg_vol * (self.vol_mult * 1.3)  # higher threshold for sells

        sell_confirmations = (rsi_declining.astype(int) +
                             stoch_cross_down.astype(int) +
                             momentum_fade.astype(int) +
                             vol_dist.astype(int))

        # Tier 3: CONTEXT
        above_sma = c > sma_slow  # must be above SMA to sell
        obv_bearish = obv < obv_ma

        sell_context = above_sma

        sell_raw = sell_trigger & (sell_confirmations >= self.sell_confirm_min) & sell_context

        strong_sell_raw = sell_trigger & (sell_confirmations >= 3) & sell_context & obv_bearish

        # ═══════════════════════════════════════
        # COOLDOWN
        # ═══════════════════════════════════════
        buy_a = buy_raw.values.astype(bool)
        sbuy_a = strong_buy_raw.values.astype(bool)
        sell_a = sell_raw.values.astype(bool)
        ssell_a = strong_sell_raw.values.astype(bool)

        final_buy = np.zeros(n, dtype=np.int8)
        final_sell = np.zeros(n, dtype=np.int8)
        last_buy = -999
        last_sell = -999

        for i in range(n):
            if buy_a[i] and (i - last_buy >= self.buy_cooldown):
                final_buy[i] = 2 if sbuy_a[i] else 1
                last_buy = i
            if sell_a[i] and (i - last_sell >= self.sell_cooldown):
                final_sell[i] = 2 if ssell_a[i] else 1
                last_sell = i

        return (pd.Series(final_buy, index=df.index),
                pd.Series(final_sell, index=df.index),
                buy_confirmations, sell_confirmations)


# ═══════════════════════════════════════════
# ACCURACY MEASUREMENT
# ═══════════════════════════════════════════

def measure_accuracy(df, buys, sells):
    c = df['Close'].values
    lo = df['Low'].values
    hi = df['High'].values
    n = len(c)
    horizons = [5, 13, 26, 52]

    results = {}

    for label, sig, direction in [('BUY', buys, 1), ('SELL', sells, -1)]:
        locs = np.where(sig.values > 0)[0]
        strong_locs = np.where(sig.values >= 2)[0]
        results[f'{label}_total'] = len(locs)
        results[f'{label}_strong'] = len(strong_locs)

        for h in horizons:
            valid = locs[locs + h < n]
            if len(valid) == 0:
                results[f'{label}_acc_{h}'] = 0.0
                results[f'{label}_ret_{h}'] = 0.0
                continue
            fwd = (c[valid + h] - c[valid]) / c[valid] * 100
            if direction == 1:
                results[f'{label}_acc_{h}'] = float((fwd > 0).sum()) / len(valid) * 100
            else:
                results[f'{label}_acc_{h}'] = float((fwd < 0).sum()) / len(valid) * 100
            results[f'{label}_ret_{h}'] = float(fwd.mean()) * direction

        for h in horizons:
            valid = strong_locs[strong_locs + h < n]
            if len(valid) == 0:
                results[f'{label}_strong_acc_{h}'] = 0.0
                continue
            fwd = (c[valid + h] - c[valid]) / c[valid] * 100
            if direction == 1:
                results[f'{label}_strong_acc_{h}'] = float((fwd > 0).sum()) / len(valid) * 100
            else:
                results[f'{label}_strong_acc_{h}'] = float((fwd < 0).sum()) / len(valid) * 100

        # Sell: also measure "did price dip at least 3% within N bars?"
        if direction == -1:
            for h in horizons:
                valid = locs[locs + h < n]
                if len(valid) == 0:
                    results[f'{label}_dip3_{h}'] = 0.0
                    continue
                dipped = np.zeros(len(valid), dtype=bool)
                for j, idx in enumerate(valid):
                    end = min(idx + h, n - 1)
                    min_price = lo[idx+1:end+1].min() if idx+1 <= end else c[idx]
                    dipped[j] = (c[idx] - min_price) / c[idx] * 100 >= 3.0
                results[f'{label}_dip3_{h}'] = float(dipped.sum()) / len(valid) * 100

    return results


def print_result(label, r):
    line = f"  {label:>30}  B:{r['BUY_total']:>3}({r['BUY_strong']:>2})"
    line += f"  5d:{r['BUY_acc_5']:>4.0f}%  13d:{r['BUY_acc_13']:>4.0f}%  26d:{r['BUY_acc_26']:>4.0f}%  52d:{r['BUY_acc_52']:>4.0f}%"
    line += f"  | S:{r['SELL_total']:>3}({r['SELL_strong']:>2})"
    if r['SELL_total'] > 0:
        line += f"  dip:{r.get('SELL_dip3_26',0):>3.0f}%"
    # Stars
    if r['BUY_total'] >= 3 and r['BUY_acc_13'] >= 65:
        line += " ★"
    if r['BUY_total'] >= 3 and r['BUY_acc_26'] >= 70:
        line += "★"
    print(line)


# ═══════════════════════════════════════════
# PARAMETER SWEEP
# ═══════════════════════════════════════════

def sweep(df, name):
    print(f"\n{'='*95}")
    print(f"  QUANTUM EDGE v2 — EVENT-BASED SIGNALS — {name}")
    print(f"  Bars: {len(df)}  Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"{'='*95}")

    # ── 1. Pullback threshold ──
    print(f"\n  ── PULLBACK THRESHOLD (adaptive ATR-normalized) ──")
    print(f"  {'Config':>30}  {'B(str)':>8}  {'5d':>6}  {'13d':>7}  {'26d':>7}  {'52d':>7}  {'S(str)':>8}  {'dip3%':>6}")
    for pb in [3, 5, 7, 10, 15]:
        s = QEv2(pullback_pct=pb)
        buys, sells, _, _ = s.compute(df)
        r = measure_accuracy(df, buys, sells)
        print_result(f"Pullback={pb}%", r)

    # ── 2. Confirmation threshold ──
    print(f"\n  ── BUY CONFIRMATION THRESHOLD (out of 4) ──")
    for bc in [1, 2, 3]:
        s = QEv2(buy_confirm_min=bc)
        buys, sells, _, _ = s.compute(df)
        r = measure_accuracy(df, buys, sells)
        print_result(f"BuyConfirm>={bc}", r)

    # ── 3. RSI levels ──
    print(f"\n  ── RSI OVERSOLD LEVEL ──")
    for os_lvl, rec_lvl in [(25, 30), (30, 35), (35, 40), (40, 45)]:
        s = QEv2(rsi_oversold=os_lvl, rsi_recovery=rec_lvl)
        buys, sells, _, _ = s.compute(df)
        r = measure_accuracy(df, buys, sells)
        print_result(f"RSI_OS={os_lvl}, Rec={rec_lvl}", r)

    # ── 4. SMA length ──
    print(f"\n  ── SMA SLOW LENGTH ──")
    for sma_l in [100, 150, 200, 250]:
        s = QEv2(sma_slow=sma_l)
        buys, sells, _, _ = s.compute(df)
        r = measure_accuracy(df, buys, sells)
        print_result(f"SMA={sma_l}", r)

    # ── 5. Cooldown ──
    print(f"\n  ── COOLDOWN ──")
    for bc, sc in [(5, 10), (10, 15), (10, 20), (15, 25), (20, 30)]:
        s = QEv2(buy_cooldown=bc, sell_cooldown=sc)
        buys, sells, _, _ = s.compute(df)
        r = measure_accuracy(df, buys, sells)
        print_result(f"BuyCool={bc}, SellCool={sc}", r)

    # ── 6. Extension for sells ──
    print(f"\n  ── SELL EXTENSION THRESHOLD ──")
    for ext in [5, 8, 10, 15, 20]:
        s = QEv2(extension_pct=ext)
        buys, sells, _, _ = s.compute(df)
        r = measure_accuracy(df, buys, sells)
        print_result(f"SellExtension={ext}%", r)

    # ── 7. Sell confirmation ──
    print(f"\n  ── SELL CONFIRMATION THRESHOLD ──")
    for sc in [1, 2, 3]:
        s = QEv2(sell_confirm_min=sc)
        buys, sells, _, _ = s.compute(df)
        r = measure_accuracy(df, buys, sells)
        print_result(f"SellConfirm>={sc}", r)

    # ── 8. Combined configs ──
    print(f"\n  ── COMBINED CONFIGURATIONS ──")
    print(f"  {'Config':>30}  {'B(str)':>8}  {'5d':>6}  {'13d':>7}  {'26d':>7}  {'52d':>7}  {'S(str)':>8}  {'dip3%':>6}")
    combos = [
        ("Default", {}),
        ("Tight Confirm", dict(buy_confirm_min=3, sell_confirm_min=3)),
        ("Deep Pullback", dict(pullback_pct=10, buy_confirm_min=2)),
        ("Loose Pullback", dict(pullback_pct=3, buy_confirm_min=2)),
        ("Max Quality", dict(pullback_pct=7, buy_confirm_min=3, sell_confirm_min=3, buy_cooldown=15, sell_cooldown=25)),
        ("Frequent Buys", dict(pullback_pct=3, buy_confirm_min=1, buy_cooldown=5)),
        ("RSI Tight", dict(rsi_oversold=25, rsi_recovery=30, rsi_overbought=75, rsi_exhaustion=70)),
        ("Long SMA", dict(sma_slow=250, pullback_pct=5)),
        ("Short SMA", dict(sma_slow=150, pullback_pct=5)),
        ("High Conviction", dict(pullback_pct=7, buy_confirm_min=3, sell_confirm_min=2, buy_cooldown=20, sell_cooldown=20)),
        ("Balanced", dict(pullback_pct=5, buy_confirm_min=2, sell_confirm_min=2, buy_cooldown=10, sell_cooldown=15)),
    ]
    for label, kwargs in combos:
        s = QEv2(**kwargs)
        buys, sells, _, _ = s.compute(df)
        r = measure_accuracy(df, buys, sells)
        print_result(label, r)

    # ── 9. Best config detailed breakdown ──
    print(f"\n  ── DETAILED ACCURACY (Default config) ──")
    s = QEv2()
    buys, sells, _, _ = s.compute(df)
    r = measure_accuracy(df, buys, sells)
    print(f"  BUY  All ({r['BUY_total']:>3}):   5d={r['BUY_acc_5']:.0f}%  13d={r['BUY_acc_13']:.0f}%  26d={r['BUY_acc_26']:.0f}%  52d={r['BUY_acc_52']:.0f}%  |  Avg ret: 5d={r['BUY_ret_5']:>+.2f}%  13d={r['BUY_ret_13']:>+.2f}%  26d={r['BUY_ret_26']:>+.2f}%  52d={r['BUY_ret_52']:>+.2f}%")
    print(f"  BUY  Str ({r['BUY_strong']:>3}):   5d={r.get('BUY_strong_acc_5',0):.0f}%  13d={r.get('BUY_strong_acc_13',0):.0f}%  26d={r.get('BUY_strong_acc_26',0):.0f}%  52d={r.get('BUY_strong_acc_52',0):.0f}%")
    print(f"  SELL All ({r['SELL_total']:>3}):   5d={r['SELL_acc_5']:.0f}%  13d={r['SELL_acc_13']:.0f}%  26d={r['SELL_acc_26']:.0f}%  52d={r['SELL_acc_52']:.0f}%  |  Dip>=3%: 13d={r.get('SELL_dip3_13',0):.0f}%  26d={r.get('SELL_dip3_26',0):.0f}%  52d={r.get('SELL_dip3_52',0):.0f}%")
    print(f"  SELL Str ({r['SELL_strong']:>3}):   5d={r.get('SELL_strong_acc_5',0):.0f}%  13d={r.get('SELL_strong_acc_13',0):.0f}%  26d={r.get('SELL_strong_acc_26',0):.0f}%  52d={r.get('SELL_strong_acc_52',0):.0f}%")

    # Show signal dates for inspection
    buy_dates = df.index[buys.values > 0]
    sell_dates = df.index[sells.values > 0]
    print(f"\n  Recent Buy signals: {[d.strftime('%Y-%m-%d') for d in buy_dates[-8:]]}")
    print(f"  Recent Sell signals: {[d.strftime('%Y-%m-%d') for d in sell_dates[-5:]]}")


def main():
    print("="*95)
    print("  QUANTUM EDGE v2 — EVENT-BASED HIGH-CONFIDENCE BACKTEST")
    print("  Independent buy/sell — buy-focused — rare high-conviction sells")
    print("="*95)
    print("\nLoading data...")

    datasets = [
        ('/home/user/Rockstar-Indicator/data/BTC-USD_daily.csv', 'BTC-USD'),
        ('/home/user/Rockstar-Indicator/data/SPY_daily.csv', 'MGK (SPY proxy)'),
    ]

    for path, name in datasets:
        df = load_data(path, name)
        if df is not None and len(df) >= 300:
            sweep(df, name)

    print(f"\n  NOTE: Colonial Coal (CAD.V) — no public data source found.")
    print(f"  Export from TradingView → Save as data/CADV_daily.csv to include.")

    print(f"\n{'='*95}")
    print("  ★ = Buy 13d accuracy >= 65%   ★★ = Buy 26d accuracy >= 70%")
    print(f"{'='*95}")


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
        print(f"  {name}: ERROR — {e}")
        return None


if __name__ == '__main__':
    main()
