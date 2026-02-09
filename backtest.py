#!/usr/bin/env python3
"""
Adaptive Momentum v4.1 — Hierarchical Multi-Layer Backtest
===========================================================
3-Layer architecture:
  L1: Trend Gate (200-day SMA, Faber 2007)
  L2: Momentum Confirmation (multi-period ROC + RSI>50, Jegadeesh-Titman 1993)
  L3: Conviction Scoring (OBV, CMF, ATR regime, BB squeeze, Supertrend)

Backtests against BTC-USD, Colonial Coal proxy, MGK/SPY using available data.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════
# INDICATOR CALCULATIONS
# ═══════════════════════════════════════════

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_sma(series, period):
    return series.rolling(window=period, min_periods=period).mean()


def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()


def calc_obv(close, volume):
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()


def calc_cmf(high, low, close, volume, period=21):
    mf_mult = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mf_mult = mf_mult.fillna(0)
    mf_vol = mf_mult * volume
    return calc_sma(mf_vol, period) / calc_sma(volume, period).replace(0, np.nan)


def calc_supertrend(high, low, close, period=14, multiplier=3.0):
    atr = calc_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = pd.Series(np.nan, index=close.index)
    direction = pd.Series(1, index=close.index)  # 1=bullish, -1=bearish

    for i in range(1, len(close)):
        if pd.isna(atr.iloc[i]):
            continue

        # Adjust bands
        if not pd.isna(lower_band.iloc[i-1]):
            if lower_band.iloc[i] < lower_band.iloc[i-1] and close.iloc[i-1] > lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i-1]
        if not pd.isna(upper_band.iloc[i-1]):
            if upper_band.iloc[i] > upper_band.iloc[i-1] and close.iloc[i-1] < upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i-1]

        if pd.isna(supertrend.iloc[i-1]):
            direction.iloc[i] = 1
            supertrend.iloc[i] = lower_band.iloc[i]
        elif supertrend.iloc[i-1] == upper_band.iloc[i-1]:
            if close.iloc[i] > upper_band.iloc[i]:
                direction.iloc[i] = 1
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                direction.iloc[i] = -1
                supertrend.iloc[i] = upper_band.iloc[i]
        else:
            if close.iloc[i] < lower_band.iloc[i]:
                direction.iloc[i] = -1
                supertrend.iloc[i] = upper_band.iloc[i]
            else:
                direction.iloc[i] = 1
                supertrend.iloc[i] = lower_band.iloc[i]

    return supertrend, direction


def calc_percentrank(series, period=100):
    return series.rolling(window=period, min_periods=period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


# ═══════════════════════════════════════════
# STRATEGY ENGINE — 3-LAYER HIERARCHICAL
# ═══════════════════════════════════════════

class HierarchicalMomentumStrategy:

    SENSITIVITY = {
        1: (1.6, 2.0),
        2: (1.3, 1.5),
        3: (1.15, 1.2),
        4: (1.0, 1.0),
        5: (0.85, 0.8),
        6: (0.72, 0.6),
        7: (0.6, 0.4),
    }

    # Base score smoothing periods (before volatility adjustment)
    # Low vol assets (SPY) get +3, moderate +1, high vol (BTC) +0
    SMOOTH_PERIOD = {1: 1, 2: 2, 3: 1, 4: 2, 5: 3, 6: 3, 7: 4}

    # Zone hysteresis spread: 0 at low sensitivity (already well-filtered),
    # 0.5 at moderate-high sensitivity (prevents rapid zone flipping)
    HYST_SPREAD = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.5, 5: 0.5, 6: 0.5, 7: 0.5}

    def __init__(self, sensitivity=4, st_period=14, st_mult=3.0,
                 use_drawdown_ctrl=True, max_drawdown_pct=15.0,
                 min_hold_bars=5, commission_pct=0.1, slippage_pct=0.05):
        self.sensitivity = sensitivity
        thresh_mult, cool_mult = self.SENSITIVITY[sensitivity]
        self.thresh_mult = thresh_mult
        self.base_cooldown = 12  # Increased from 8 to reduce whipsaw
        self.cooldown = max(3, round(self.base_cooldown * cool_mult))
        self.st_period = st_period
        self.st_mult = st_mult
        self.use_drawdown_ctrl = use_drawdown_ctrl
        self.max_drawdown_pct = max_drawdown_pct
        self.min_hold_bars = min_hold_bars  # Minimum bars before Supertrend can exit
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.score_smooth_period = self.SMOOTH_PERIOD[sensitivity]

        # Zone thresholds (scaled by sensitivity)
        self.zone_strong_buy = min(8.0 * thresh_mult, 9.5)
        self.zone_accumulate = min(6.0 * thresh_mult, self.zone_strong_buy - 0.5)
        self.zone_hold = min(4.0 * thresh_mult, self.zone_accumulate - 0.5)
        self.zone_distribute = min(2.0 * thresh_mult, self.zone_hold - 0.5)

        # Zone hysteresis: entry thresholds are standard, exit thresholds
        # are more lenient (prevents rapid zone flipping)
        hyst = self.HYST_SPREAD[sensitivity]
        self.zone_strong_buy_exit = self.zone_strong_buy - hyst
        self.zone_accumulate_exit = self.zone_accumulate - hyst
        self.zone_distribute_exit = self.zone_distribute + hyst

    def compute_signals(self, df):
        close = df['Close'].copy()
        high = df['High'].copy()
        low = df['Low'].copy()
        volume = df['Volume'].copy().fillna(0)
        n = len(df)

        # ── LAYER 1: TREND GATE ──
        sma200 = calc_sma(close, 200)
        trend_gate = close > sma200

        # ── LAYER 2: MOMENTUM ──
        roc252 = close.pct_change(252) * 100
        roc126 = close.pct_change(126) * 100
        roc63 = close.pct_change(63) * 100
        roc21 = close.pct_change(21) * 100
        avg_roc = roc21.fillna(0) * 0.1 + roc63.fillna(0) * 0.2 + roc126.fillna(0) * 0.3 + roc252.fillna(0) * 0.4

        rsi = calc_rsi(close, 14)
        rsi_above_50 = rsi > 50
        rsi_extreme_low = rsi < 30

        mom_roc_score = np.where(avg_roc > 0, 1.0, -1.0)
        mom_rsi_score = np.where(rsi_above_50, 1.0, -1.0)
        momentum_score = pd.Series(mom_roc_score * 0.6 + mom_rsi_score * 0.4, index=df.index)
        momentum_confirmed = (avg_roc > 0) & rsi_above_50

        # ── LAYER 3: CONVICTION ──
        # OBV
        obv = calc_obv(close, volume)
        obv_ema20 = calc_ema(obv, 20)
        obv_ema5 = calc_ema(obv, 5)
        obv_slope = obv_ema5 > obv_ema20
        obv_score = pd.Series(np.where(obv_slope, 1.0, np.where(obv_ema5 < obv_ema20, -0.5, 0.0)), index=df.index)

        # CMF
        cmf = calc_cmf(high, low, close, volume, 21)
        cmf_score = pd.Series(
            np.where(cmf > 0.05, 1.0, np.where(cmf > 0, 0.5, np.where(cmf > -0.05, -0.25, -1.0))),
            index=df.index
        )

        # Volume
        vol_sma50 = calc_sma(volume, 50)
        vol_ratio = volume / vol_sma50.replace(0, np.nan)
        vol_spike = vol_ratio >= 2.0
        vol_confirm = pd.Series(
            np.where(vol_spike & (close > close.shift(1)), 0.5,
                     np.where(vol_spike & (close < close.shift(1)), -0.5, 0.0)),
            index=df.index
        )

        # Volume gate (auto-detect low liquidity)
        vol_sma10 = calc_sma(volume, 10)
        low_volume = (vol_sma10 < 0.3 * vol_sma50) | (volume < 1000)
        volume_adequate = ~low_volume

        # ATR regime
        atr = calc_atr(high, low, close, 14)
        atr_pct = atr / close.replace(0, np.nan) * 100
        atr_percentile = calc_percentrank(atr_pct, 100)
        vol_regime_low = atr_percentile < 30
        vol_regime_high = atr_percentile > 70
        atr_score = pd.Series(
            np.where(vol_regime_low, 0.5, np.where(vol_regime_high, -0.5, 0.0)),
            index=df.index
        )

        # BB Squeeze
        bb_basis = calc_sma(close, 20)
        bb_std = close.rolling(20).std()
        bb_upper = bb_basis + 2.0 * bb_std
        bb_lower = bb_basis - 2.0 * bb_std
        bb_width = bb_upper - bb_lower
        bb_width_pctile = calc_percentrank(bb_width, 100)

        kc_range = calc_atr(high, low, close, 20)
        kc_upper = bb_basis + kc_range * 1.5
        kc_lower = bb_basis - kc_range * 1.5
        squeeze_confirmed = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        squeeze_release = squeeze_confirmed.shift(1).fillna(False) & ~squeeze_confirmed

        sqz_highest = high.rolling(20).max()
        sqz_lowest = low.rolling(20).min()
        sqz_midline = (sqz_highest + sqz_lowest) / 2
        deviation = close - (sqz_midline + bb_basis) / 2

        # Simple momentum proxy for squeeze direction
        squeeze_score = pd.Series(0.0, index=df.index)
        squeeze_score = squeeze_score.where(~(squeeze_release & (deviation > 0)), 1.0)
        squeeze_score = squeeze_score.where(~(squeeze_release & (deviation < 0) & (squeeze_score == 0)), -1.0)
        squeeze_score = squeeze_score.where(~(squeeze_confirmed & (squeeze_score == 0)), 0.25)

        # Supertrend — auto-scale multiplier based on ATR percentile
        # Low-vol assets (SPY) get wider multiplier to avoid whipsaw
        # High-vol assets (BTC) get tighter multiplier for faster reaction
        median_atr_pct = atr_pct.rolling(252, min_periods=50).median()
        # Scale: if median ATR% < 1.0 (low vol like SPY), widen to 4-5x
        # If median ATR% > 3.0 (high vol like BTC), keep at base 3x
        st_mult_auto = pd.Series(self.st_mult, index=df.index)
        for i in range(n):
            m = median_atr_pct.iloc[i] if not pd.isna(median_atr_pct.iloc[i]) else 2.0
            if m < 0.8:
                st_mult_auto.iloc[i] = self.st_mult + 2.0  # Very low vol: +2
            elif m < 1.5:
                st_mult_auto.iloc[i] = self.st_mult + 1.0  # Low vol: +1
            elif m > 4.0:
                st_mult_auto.iloc[i] = max(2.0, self.st_mult - 0.5)  # Very high vol: tighter

        # Use the median auto-scaled multiplier for Supertrend
        effective_st_mult = st_mult_auto.median()
        st_value, st_direction = calc_supertrend(high, low, close, self.st_period, effective_st_mult)
        st_bullish = st_direction > 0
        st_score = pd.Series(np.where(st_bullish, 0.5, -0.5), index=df.index)

        # ═══════════════════════════════════════
        # COMPOSITE SCORE (0–10)
        # ═══════════════════════════════════════
        composite = pd.Series(0.0, index=df.index)

        for i in range(n):
            if pd.isna(sma200.iloc[i]):
                composite.iloc[i] = 5.0
                continue

            if trend_gate.iloc[i]:
                base = 5.0
                mom = momentum_score.iloc[i] * 1.5
                obv_c = obv_score.iloc[i] * 0.6
                cmf_c = cmf_score.iloc[i] * 0.4
                vol_c = vol_confirm.iloc[i] * 0.3
                atr_c = atr_score.iloc[i] * 0.3
                sqz_c = squeeze_score.iloc[i] * 0.5
                st_c = st_score.iloc[i] * 0.4
                raw = base + mom + obv_c + cmf_c + vol_c + atr_c + sqz_c + st_c

                # Contrarian RSI extreme
                if rsi_extreme_low.iloc[i]:
                    raw = max(raw, 7.0)

                composite.iloc[i] = max(0.0, min(10.0, raw))
            else:
                base = 2.0
                mom = momentum_score.iloc[i] * 0.8
                st_c = st_score.iloc[i] * 0.4
                sqz_c = squeeze_score.iloc[i] * 0.3
                raw = base + mom + st_c + sqz_c
                composite.iloc[i] = max(0.0, min(10.0, raw))

        # ═══════════════════════════════════════
        # COMPOSITE SCORE SMOOTHING
        # ═══════════════════════════════════════
        # Volatility-adaptive smoothing: low-vol assets need more smoothing
        # (more noise in score), high-vol assets need less (signals are clearer)
        stable_atr_pct = median_atr_pct.dropna()
        if len(stable_atr_pct) > 0:
            med_atr = stable_atr_pct.iloc[-1]
        else:
            med_atr = atr_pct.median()

        smooth_adjust = 3 if med_atr < 1.5 else (1 if med_atr < 2.5 else 0)
        effective_smooth = self.score_smooth_period + smooth_adjust

        if effective_smooth > 1:
            composite = calc_ema(composite, effective_smooth).clip(0.0, 10.0)

        # ═══════════════════════════════════════
        # SIGNAL GENERATION
        # ═══════════════════════════════════════
        signals = pd.Series(0, index=df.index)
        bars_since = 999
        last_dir = 0
        bars_in_trade = 0  # Track how long we've been in current trade

        peak_equity = 1.0
        equity = 1.0
        in_position = False
        prev_zone = 'hold'  # Zone hysteresis state

        for i in range(1, n):
            bars_since += 1
            if in_position:
                bars_in_trade += 1
            cooldown_met = bars_since >= self.cooldown
            score = composite.iloc[i]

            # Zone classification with hysteresis
            # When in bullish zone, use lower exit thresholds (sticky bullish)
            # When in neutral/bearish zone, use standard entry thresholds
            if prev_zone in ('strong_buy', 'accumulate'):
                is_strong_buy = score >= self.zone_strong_buy_exit
                is_accumulate = score >= self.zone_accumulate_exit and not is_strong_buy
                is_distribute = score < self.zone_hold and score >= self.zone_distribute_exit
                is_strong_sell = score < self.zone_distribute_exit
            else:
                is_strong_buy = score >= self.zone_strong_buy
                is_accumulate = score >= self.zone_accumulate and not is_strong_buy
                is_distribute = score >= self.zone_distribute and score < self.zone_hold
                is_strong_sell = score < self.zone_distribute

            # Update zone state
            if is_strong_buy:
                prev_zone = 'strong_buy'
            elif is_accumulate:
                prev_zone = 'accumulate'
            elif is_strong_sell:
                prev_zone = 'strong_sell'
            elif is_distribute:
                prev_zone = 'distribute'
            else:
                prev_zone = 'hold'

            buy_sig = (is_strong_buy or is_accumulate) and cooldown_met and last_dir != 1
            sell_sig = False

            # Score-based sell: only exit when BOTH score drops AND trend gate fails
            # This prevents premature exits when score dips but trend is intact
            trend_failed = not trend_gate.iloc[i] if not pd.isna(trend_gate.iloc[i]) else False
            if (is_strong_sell and trend_failed) and last_dir != -1:
                sell_sig = True

            # Supertrend trailing stop — only after minimum hold period
            st_exit = (st_direction.iloc[i] < 0 and last_dir == 1 and
                       bars_in_trade >= self.min_hold_bars)

            # Volume gate
            if not volume_adequate.iloc[i]:
                buy_sig = False

            # Prevent simultaneous
            buy_sig = buy_sig and not sell_sig
            sell_sig = (sell_sig or st_exit) and not buy_sig

            # High vol regime gate
            if vol_regime_high.iloc[i] and buy_sig and not is_strong_buy:
                buy_sig = False

            # Drawdown control
            if in_position:
                ret = close.iloc[i] / close.iloc[i-1] if close.iloc[i-1] > 0 else 1.0
                equity *= ret
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity * 100
                if self.use_drawdown_ctrl and dd >= self.max_drawdown_pct:
                    sell_sig = True
                    buy_sig = False

            if buy_sig:
                signals.iloc[i] = 1
                last_dir = 1
                bars_since = 0
                bars_in_trade = 0
                if not in_position:
                    in_position = True
                    equity = 1.0
                    peak_equity = 1.0
            elif sell_sig:
                signals.iloc[i] = -1
                last_dir = -1
                bars_since = 0
                bars_in_trade = 0
                in_position = False

        return signals, composite, {
            'trend_gate': trend_gate,
            'momentum_confirmed': momentum_confirmed,
            'composite': composite,
            'vol_regime_high': vol_regime_high,
            'vol_regime_low': vol_regime_low,
            'sma200': sma200,
        }


# ═══════════════════════════════════════════
# BACKTESTING ENGINE
# ═══════════════════════════════════════════

def backtest(df, signals, commission_pct=0.1, slippage_pct=0.05):
    close = df['Close'].copy()
    n = len(df)

    equity = 100000.0
    peak_equity = equity
    position = 0
    entry_price = 0.0
    trades = []
    equity_curve = [equity]
    max_dd = 0.0

    for i in range(1, n):
        if position == 1:
            daily_return = close.iloc[i] / close.iloc[i-1] if close.iloc[i-1] > 0 else 1.0
            equity *= daily_return

        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        max_dd = max(max_dd, dd)

        sig = signals.iloc[i]
        if sig == 1 and position == 0:
            cost = equity * (commission_pct + slippage_pct) / 100
            equity -= cost
            entry_price = close.iloc[i]
            position = 1
        elif sig == -1 and position == 1:
            cost = equity * (commission_pct + slippage_pct) / 100
            equity -= cost
            exit_price = close.iloc[i]
            trade_pct = (exit_price - entry_price) / entry_price * 100
            trades.append({'pct_return': trade_pct})
            position = 0

        equity_curve.append(equity)

    # Also compute buy-and-hold with same costs
    bh_equity = 100000.0
    bh_cost = bh_equity * (commission_pct + slippage_pct) / 100
    bh_equity -= bh_cost
    bh_equity *= (close.iloc[-1] / close.iloc[0])
    bh_cost2 = bh_equity * (commission_pct + slippage_pct) / 100
    bh_equity -= bh_cost2
    bh_return = (bh_equity / 100000 - 1) * 100

    # Faber 10-month SMA benchmark
    sma200 = calc_sma(close, 200)
    faber_equity = 100000.0
    faber_position = 0
    faber_trades = 0
    for i in range(1, n):
        if faber_position == 1:
            faber_equity *= close.iloc[i] / close.iloc[i-1] if close.iloc[i-1] > 0 else 1.0
        if not pd.isna(sma200.iloc[i]):
            if close.iloc[i] > sma200.iloc[i] and faber_position == 0:
                faber_cost = faber_equity * (commission_pct + slippage_pct) / 100
                faber_equity -= faber_cost
                faber_position = 1
                faber_trades += 1
            elif close.iloc[i] < sma200.iloc[i] and faber_position == 1:
                faber_cost = faber_equity * (commission_pct + slippage_pct) / 100
                faber_equity -= faber_cost
                faber_position = 0
                faber_trades += 1
    faber_return = (faber_equity / 100000 - 1) * 100

    # Compute metrics
    total_trades = len(trades)
    wins = [t for t in trades if t['pct_return'] >= 0]
    losses_list = [t for t in trades if t['pct_return'] < 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_win = np.mean([t['pct_return'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pct_return'] for t in losses_list]) if losses_list else 0
    gross_profit = sum(t['pct_return'] for t in wins)
    gross_loss = abs(sum(t['pct_return'] for t in losses_list))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
    strat_return = (equity / 100000 - 1) * 100

    # Forward returns
    fwd_returns = {}
    buy_indices = signals[signals == 1].index
    sell_indices = signals[signals == -1].index

    for period in [4, 8, 13, 26]:
        buy_fwds, sell_fwds = [], []
        for idx in buy_indices:
            loc = df.index.get_loc(idx)
            if loc + period < n:
                buy_fwds.append((close.iloc[loc + period] - close.iloc[loc]) / close.iloc[loc] * 100)
        for idx in sell_indices:
            loc = df.index.get_loc(idx)
            if loc + period < n:
                sell_fwds.append((close.iloc[loc + period] - close.iloc[loc]) / close.iloc[loc] * 100)
        fwd_returns[period] = {
            'buy_avg': np.mean(buy_fwds) if buy_fwds else None, 'buy_n': len(buy_fwds),
            'sell_avg': np.mean(sell_fwds) if sell_fwds else None, 'sell_n': len(sell_fwds),
        }

    # Hit rates
    buy_correct_13 = sum(1 for idx in buy_indices
                         if df.index.get_loc(idx) + 13 < n and
                         close.iloc[df.index.get_loc(idx) + 13] > close.iloc[df.index.get_loc(idx)])
    buy_total_13 = sum(1 for idx in buy_indices if df.index.get_loc(idx) + 13 < n)
    sell_correct_13 = sum(1 for idx in sell_indices
                          if df.index.get_loc(idx) + 13 < n and
                          close.iloc[df.index.get_loc(idx) + 13] < close.iloc[df.index.get_loc(idx)])
    sell_total_13 = sum(1 for idx in sell_indices if df.index.get_loc(idx) + 13 < n)

    return {
        'total_trades': total_trades, 'wins': len(wins), 'losses': len(losses_list),
        'win_rate': win_rate, 'avg_win_pct': avg_win, 'avg_loss_pct': avg_loss,
        'profit_factor': profit_factor, 'max_drawdown': max_dd,
        'net_return_pct': strat_return, 'buy_hold_return_pct': bh_return,
        'faber_sma_return_pct': faber_return, 'faber_trades': faber_trades,
        'equity_curve': pd.Series(equity_curve),
        'forward_returns': fwd_returns,
        'buy_hit_rate_13': buy_correct_13 / buy_total_13 * 100 if buy_total_13 > 0 else None,
        'sell_hit_rate_13': sell_correct_13 / sell_total_13 * 100 if sell_total_13 > 0 else None,
        'buy_signals': len(buy_indices), 'sell_signals': len(sell_indices),
        'final_equity': equity,
    }


def print_results(ticker, results, sensitivity):
    print(f"\n{'='*65}")
    print(f"  {ticker} — Sensitivity {sensitivity}")
    print(f"{'='*65}")
    print(f"  Total Trades:     {results['total_trades']}")
    print(f"  Wins / Losses:    {results['wins']}W / {results['losses']}L")
    print(f"  Win Rate:         {results['win_rate']:.1f}%")
    print(f"  Profit Factor:    {results['profit_factor']:.2f}")
    print(f"  Avg Win %:        {results['avg_win_pct']:.2f}%")
    print(f"  Avg Loss %:       {results['avg_loss_pct']:.2f}%")
    print(f"  Max Drawdown:     {results['max_drawdown']:.2f}%")
    print(f"  ─────────────────────────────────────")
    print(f"  Strategy Return:  {results['net_return_pct']:>10.2f}%  (${results['final_equity']:,.0f})")
    print(f"  Buy & Hold:       {results['buy_hold_return_pct']:>10.2f}%")
    print(f"  Faber SMA200:     {results['faber_sma_return_pct']:>10.2f}%  ({results['faber_trades']} trades)")
    print(f"  ─────────────────────────────────────")
    if results['buy_hit_rate_13'] is not None:
        print(f"  Buy Hit Rate(13): {results['buy_hit_rate_13']:.1f}%")
    if results['sell_hit_rate_13'] is not None:
        print(f"  Sell Hit Rate(13):{results['sell_hit_rate_13']:.1f}%")
    print(f"  Buy Signals: {results['buy_signals']}  |  Sell Signals: {results['sell_signals']}")
    print(f"\n  Forward Returns (avg % after signal):")
    for period, data in results['forward_returns'].items():
        buy_str = f"{data['buy_avg']:+.2f}% (n={data['buy_n']})" if data['buy_avg'] is not None else "N/A"
        sell_str = f"{data['sell_avg']:+.2f}% (n={data['sell_n']})" if data['sell_avg'] is not None else "N/A"
        print(f"    {period:>2}-bar:  Buy={buy_str}  |  Sell={sell_str}")


def plot_results(all_results, filename):
    fig, axes = plt.subplots(len(all_results), 1, figsize=(14, 5 * len(all_results)))
    if len(all_results) == 1:
        axes = [axes]

    for ax, (ticker, res) in zip(axes, all_results.items()):
        eq = res['equity_curve']
        ax.plot(eq.values, linewidth=1.5, color='#58A6FF', label=f'Strategy ({res["net_return_pct"]:.1f}%)')
        ax.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax.set_title(f'{ticker} — Equity Curve (v4.1 Hierarchical)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Equity ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nEquity curves saved to {filename}")


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def load_data(filepath, name):
    try:
        df = pd.read_csv(filepath, parse_dates=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df = df.set_index('Date')
        # Ensure standard column names
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns:
                # Try lowercase
                if col.lower() in df.columns:
                    df = df.rename(columns={col.lower(): col})
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        df = df.dropna(subset=['Close'])
        print(f"  {name}: {len(df)} bars ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
        return df
    except Exception as e:
        print(f"  ERROR loading {name}: {e}")
        return None


def main():
    sensitivity = 4

    print("="*65)
    print("  ADAPTIVE MOMENTUM v4.1 — HIERARCHICAL MULTI-LAYER BACKTEST")
    print("  L1: Trend Gate (200-SMA)  |  L2: Momentum (ROC+RSI)")
    print("  L3: Conviction (OBV, CMF, ATR, Squeeze, Supertrend)")
    print(f"  Sensitivity: {sensitivity} (Standard)")
    print("="*65)

    print("\nLoading data...")
    datasets = {}

    btc = load_data('/home/user/Rockstar-Indicator/data/BTC-USD_daily.csv', 'BTC-USD')
    if btc is not None and len(btc) >= 300:
        datasets['BTC-USD'] = btc

    spy = load_data('/home/user/Rockstar-Indicator/data/SPY_daily.csv', 'SPY (MGK proxy)')
    if spy is not None and len(spy) >= 300:
        datasets['SPY (MGK proxy)'] = spy

    # Try Colonial Coal proxy
    import os
    cad_path = '/home/user/Rockstar-Indicator/data/CAD_V_daily.csv'
    if os.path.exists(cad_path):
        cad = load_data(cad_path, 'Colonial Coal (CAD.V)')
        if cad is not None and len(cad) >= 200:
            datasets['Colonial Coal (CAD.V)'] = cad

    if not datasets:
        print("No data loaded. Exiting.")
        return

    # Run backtests
    strategy = HierarchicalMomentumStrategy(sensitivity=sensitivity)
    all_results = {}

    for ticker, df in datasets.items():
        print(f"\nBacktesting {ticker}...")
        signals, composite, factors = strategy.compute_signals(df)
        results = backtest(df, signals, strategy.commission_pct, strategy.slippage_pct)
        all_results[ticker] = results
        print_results(ticker, results, sensitivity)

    # Sensitivity sweep on BTC
    if 'BTC-USD' in datasets:
        print(f"\n{'='*65}")
        print("  SENSITIVITY SWEEP — BTC-USD")
        print(f"{'='*65}")
        print(f"  {'Sens':>4}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'Return%':>10}  {'MaxDD%':>7}  {'vs BH':>8}  {'vs Faber':>9}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*7}  {'─'*8}  {'─'*9}")
        for sens in [1, 2, 3, 4, 5, 6, 7]:
            strat = HierarchicalMomentumStrategy(sensitivity=sens)
            sigs, _, _ = strat.compute_signals(datasets['BTC-USD'])
            res = backtest(datasets['BTC-USD'], sigs)
            vs_bh = res['net_return_pct'] - res['buy_hold_return_pct']
            vs_faber = res['net_return_pct'] - res['faber_sma_return_pct']
            print(f"  {sens:>4}  {res['total_trades']:>6}  {res['win_rate']:>5.1f}%  {res['profit_factor']:>5.2f}  {res['net_return_pct']:>9.2f}%  {res['max_drawdown']:>6.2f}%  {vs_bh:>+7.1f}%  {vs_faber:>+8.1f}%")

    # Sensitivity sweep on SPY
    if 'SPY (MGK proxy)' in datasets:
        print(f"\n{'='*65}")
        print("  SENSITIVITY SWEEP — SPY (MGK proxy)")
        print(f"{'='*65}")
        print(f"  {'Sens':>4}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'Return%':>10}  {'MaxDD%':>7}  {'vs BH':>8}  {'vs Faber':>9}")
        print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*7}  {'─'*8}  {'─'*9}")
        for sens in [1, 2, 3, 4, 5, 6, 7]:
            strat = HierarchicalMomentumStrategy(sensitivity=sens)
            sigs, _, _ = strat.compute_signals(datasets['SPY (MGK proxy)'])
            res = backtest(datasets['SPY (MGK proxy)'], sigs)
            vs_bh = res['net_return_pct'] - res['buy_hold_return_pct']
            vs_faber = res['net_return_pct'] - res['faber_sma_return_pct']
            print(f"  {sens:>4}  {res['total_trades']:>6}  {res['win_rate']:>5.1f}%  {res['profit_factor']:>5.2f}  {res['net_return_pct']:>9.2f}%  {res['max_drawdown']:>6.2f}%  {vs_bh:>+7.1f}%  {vs_faber:>+8.1f}%")

    plot_results(all_results, '/home/user/Rockstar-Indicator/backtest_results.png')
    print("\nBacktest complete.")


if __name__ == '__main__':
    main()
