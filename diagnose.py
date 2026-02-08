#!/usr/bin/env python3
"""
Diagnostic analysis of v4.0 backtest trades.
Identifies where the strategy loses money and why.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import from backtest.py
from backtest import (
    HierarchicalMomentumStrategy, calc_sma, calc_rsi, calc_ema,
    calc_obv, calc_cmf, calc_atr, calc_supertrend, calc_percentrank,
    load_data, backtest
)


def detailed_trade_analysis(df, signals, composite, factors, label):
    """Analyze individual trades to find patterns in wins vs losses."""
    close = df['Close']
    n = len(df)

    trades = []
    entry_idx = None
    entry_price = None

    for i in range(n):
        sig = signals.iloc[i]
        if sig == 1 and entry_idx is None:
            entry_idx = i
            entry_price = close.iloc[i]
        elif sig == -1 and entry_idx is not None:
            exit_price = close.iloc[i]
            pct = (exit_price - entry_price) / entry_price * 100
            duration = i - entry_idx
            entry_score = composite.iloc[entry_idx]
            exit_score = composite.iloc[i]

            # Check conditions at entry
            trend_at_entry = factors['trend_gate'].iloc[entry_idx]
            mom_at_entry = factors['momentum_confirmed'].iloc[entry_idx]
            high_vol_at_entry = factors['vol_regime_high'].iloc[entry_idx] if not pd.isna(factors['vol_regime_high'].iloc[entry_idx]) else False

            # Max adverse excursion (worst drawdown during trade)
            if i > entry_idx:
                trade_prices = close.iloc[entry_idx:i+1]
                mae = (trade_prices.min() - entry_price) / entry_price * 100
                mfe = (trade_prices.max() - entry_price) / entry_price * 100
            else:
                mae = 0
                mfe = 0

            trades.append({
                'entry_idx': entry_idx, 'exit_idx': i,
                'entry_price': entry_price, 'exit_price': exit_price,
                'pct_return': pct, 'duration': duration,
                'entry_score': entry_score, 'exit_score': exit_score,
                'trend_pass': trend_at_entry, 'mom_confirmed': mom_at_entry,
                'high_vol': high_vol_at_entry,
                'mae': mae, 'mfe': mfe,
                'win': pct >= 0,
            })
            entry_idx = None
            entry_price = None

    if not trades:
        print(f"\n  No trades for {label}")
        return

    tdf = pd.DataFrame(trades)
    wins = tdf[tdf['win']]
    losses = tdf[~tdf['win']]

    print(f"\n{'='*70}")
    print(f"  TRADE ANALYSIS: {label}")
    print(f"{'='*70}")

    print(f"\n  Total: {len(tdf)} trades  |  {len(wins)}W / {len(losses)}L  |  WR: {len(wins)/len(tdf)*100:.1f}%")

    print(f"\n  DURATION ANALYSIS:")
    print(f"    Avg duration:     {tdf['duration'].mean():.1f} bars")
    print(f"    Win avg duration: {wins['duration'].mean():.1f} bars" if len(wins) > 0 else "")
    print(f"    Loss avg duration:{losses['duration'].mean():.1f} bars" if len(losses) > 0 else "")
    print(f"    Median duration:  {tdf['duration'].median():.0f} bars")

    print(f"\n  RETURN ANALYSIS:")
    print(f"    Avg return:       {tdf['pct_return'].mean():+.2f}%")
    print(f"    Avg win:          {wins['pct_return'].mean():+.2f}%" if len(wins) > 0 else "")
    print(f"    Avg loss:         {losses['pct_return'].mean():+.2f}%" if len(losses) > 0 else "")
    print(f"    Median return:    {tdf['pct_return'].median():+.2f}%")
    print(f"    Best trade:       {tdf['pct_return'].max():+.2f}%")
    print(f"    Worst trade:      {tdf['pct_return'].min():+.2f}%")

    print(f"\n  MAX ADVERSE/FAVORABLE EXCURSION:")
    print(f"    Avg MAE (drawdown during trade): {tdf['mae'].mean():+.2f}%")
    print(f"    Avg MFE (peak gain during trade): {tdf['mfe'].mean():+.2f}%")
    if len(losses) > 0:
        print(f"    Loss MAE avg:    {losses['mae'].mean():+.2f}%")
        print(f"    Loss MFE avg:    {losses['mfe'].mean():+.2f}%  <- gave back this much profit")

    # Short trades (< 5 bars) analysis
    short_trades = tdf[tdf['duration'] < 5]
    if len(short_trades) > 0:
        print(f"\n  SHORT TRADES (< 5 bars): {len(short_trades)} ({len(short_trades)/len(tdf)*100:.0f}% of all)")
        print(f"    Avg return:       {short_trades['pct_return'].mean():+.2f}%")
        print(f"    Win rate:         {short_trades['win'].mean()*100:.1f}%")

    # Whipsaw detection: re-entry within 10 bars of exit
    whipsaws = 0
    for i in range(1, len(tdf)):
        if tdf.iloc[i]['entry_idx'] - tdf.iloc[i-1]['exit_idx'] < 10:
            whipsaws += 1
    print(f"\n  WHIPSAW (re-entry <10 bars after exit): {whipsaws} ({whipsaws/max(len(tdf)-1,1)*100:.0f}%)")

    # Score analysis
    print(f"\n  ENTRY SCORE ANALYSIS:")
    print(f"    Avg entry score (wins):   {wins['entry_score'].mean():.2f}" if len(wins) > 0 else "")
    print(f"    Avg entry score (losses): {losses['entry_score'].mean():.2f}" if len(losses) > 0 else "")

    # Duration buckets
    print(f"\n  PERFORMANCE BY DURATION:")
    for lo, hi, label_d in [(1,5,'1-5 bars'), (5,15,'5-15 bars'), (15,50,'15-50 bars'), (50,500,'50+ bars')]:
        bucket = tdf[(tdf['duration'] >= lo) & (tdf['duration'] < hi)]
        if len(bucket) > 0:
            wr = bucket['win'].mean() * 100
            avg_r = bucket['pct_return'].mean()
            print(f"    {label_d:>12}: {len(bucket):>3} trades, WR={wr:.0f}%, avg={avg_r:+.2f}%")

    # Supertrend exit analysis
    st_exits = tdf[tdf['exit_score'] > 3.0]  # Exited while score was still OK-ish
    if len(st_exits) > 0:
        print(f"\n  EXITS WITH SCORE > 3.0 (potential premature exits): {len(st_exits)}")
        print(f"    Avg return:       {st_exits['pct_return'].mean():+.2f}%")
        print(f"    Win rate:         {st_exits['win'].mean()*100:.1f}%")

    return tdf


def main():
    print("="*70)
    print("  DIAGNOSTIC TRADE ANALYSIS — v4.0 Hierarchical Strategy")
    print("="*70)

    print("\nLoading data...")
    btc = load_data('/home/user/Rockstar-Indicator/data/BTC-USD_daily.csv', 'BTC-USD')
    spy = load_data('/home/user/Rockstar-Indicator/data/SPY_daily.csv', 'SPY')

    # Analyze at different sensitivities
    for sens in [3, 4, 5]:
        strategy = HierarchicalMomentumStrategy(sensitivity=sens)

        if btc is not None:
            signals, composite, factors = strategy.compute_signals(btc)
            detailed_trade_analysis(btc, signals, composite, factors, f"BTC-USD Sens={sens}")

        if spy is not None:
            signals, composite, factors = strategy.compute_signals(spy)
            detailed_trade_analysis(spy, signals, composite, factors, f"SPY Sens={sens}")

    # Test parameter variations
    print(f"\n\n{'='*70}")
    print("  PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*70}")

    if btc is not None:
        print(f"\n  BTC-USD — Varying Supertrend Multiplier (Sens=5):")
        print(f"  {'STMult':>6}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'Return%':>10}  {'MaxDD%':>7}")
        for st_mult in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
            strat = HierarchicalMomentumStrategy(sensitivity=5, st_mult=st_mult)
            sigs, _, _ = strat.compute_signals(btc)
            res = backtest(btc, sigs)
            print(f"  {st_mult:>6.1f}  {res['total_trades']:>6}  {res['win_rate']:>5.1f}%  {res['profit_factor']:>5.2f}  {res['net_return_pct']:>9.2f}%  {res['max_drawdown']:>6.2f}%")

        print(f"\n  BTC-USD — Varying Drawdown Limit (Sens=5):")
        print(f"  {'MaxDD':>6}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'Return%':>10}  {'MaxDD%':>7}")
        for max_dd in [8, 10, 12, 15, 20, 25, 30]:
            strat = HierarchicalMomentumStrategy(sensitivity=5, max_drawdown_pct=max_dd)
            sigs, _, _ = strat.compute_signals(btc)
            res = backtest(btc, sigs)
            print(f"  {max_dd:>5}%  {res['total_trades']:>6}  {res['win_rate']:>5.1f}%  {res['profit_factor']:>5.2f}  {res['net_return_pct']:>9.2f}%  {res['max_drawdown']:>6.2f}%")

        print(f"\n  BTC-USD — Varying Drawdown Limit (Sens=5, No Drawdown Control):")
        strat = HierarchicalMomentumStrategy(sensitivity=5, use_drawdown_ctrl=False)
        sigs, _, _ = strat.compute_signals(btc)
        res = backtest(btc, sigs)
        print(f"  {'OFF':>6}  {res['total_trades']:>6}  {res['win_rate']:>5.1f}%  {res['profit_factor']:>5.02f}  {res['net_return_pct']:>9.2f}%  {res['max_drawdown']:>6.2f}%")

    if spy is not None:
        print(f"\n  SPY — Varying Supertrend Multiplier (Sens=3):")
        print(f"  {'STMult':>6}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'Return%':>10}  {'MaxDD%':>7}")
        for st_mult in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
            strat = HierarchicalMomentumStrategy(sensitivity=3, st_mult=st_mult)
            sigs, _, _ = strat.compute_signals(spy)
            res = backtest(spy, sigs)
            print(f"  {st_mult:>6.1f}  {res['total_trades']:>6}  {res['win_rate']:>5.1f}%  {res['profit_factor']:>5.02f}  {res['net_return_pct']:>9.2f}%  {res['max_drawdown']:>6.2f}%")

        print(f"\n  SPY — Varying Drawdown Limit (Sens=3):")
        print(f"  {'MaxDD':>6}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  {'Return%':>10}  {'MaxDD%':>7}")
        for max_dd in [8, 10, 12, 15, 20, 25, 30]:
            strat = HierarchicalMomentumStrategy(sensitivity=3, max_drawdown_pct=max_dd)
            sigs, _, _ = strat.compute_signals(spy)
            res = backtest(spy, sigs)
            print(f"  {max_dd:>5}%  {res['total_trades']:>6}  {res['win_rate']:>5.1f}%  {res['profit_factor']:>5.02f}  {res['net_return_pct']:>9.2f}%  {res['max_drawdown']:>6.2f}%")

        print(f"\n  SPY — No Drawdown Control (Sens=3):")
        strat = HierarchicalMomentumStrategy(sensitivity=3, use_drawdown_ctrl=False)
        sigs, _, _ = strat.compute_signals(spy)
        res = backtest(spy, sigs)
        print(f"  {'OFF':>6}  {res['total_trades']:>6}  {res['win_rate']:>5.1f}%  {res['profit_factor']:>5.02f}  {res['net_return_pct']:>9.2f}%  {res['max_drawdown']:>6.2f}%")


if __name__ == '__main__':
    main()
