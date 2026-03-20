# signals.py — leading indicator stack and 1-day GEX-conditioned probability
import os, re, time, math, datetime as dt, json, tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from fredapi import Fred
from scipy import stats as scipy_stats
from scipy.stats import norm as scipy_norm
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from config import GammaState, GammaRegime
from utils import _to_1d, zscore, current_pct_rank, resample_ffill, kelly, _bs_iv_from_price

def compute_leading_stack(
    y2, y3m, y10, y30, s_2s10s, vix, m2, claims,
    copx, gld, hyg, lqd, dxy, spy, qqq, iwm,
    net_liq, net_liq_4w, walcl, bs_13w, idx,
    # NEW signals
    tips_10y=None, bank_reserves=None, bank_credit=None,
    ism_no=None, gdp_quarterly=None, mmmf=None,
) -> Dict:
    """
    Signal stack reorganised by forecast horizon per the architecture critique.

    Signals are grouped into three horizon buckets and NEVER blended across
    buckets in the probability composite.  Each bucket feeds the corresponding
    sub-model (tactical / short_term / medium_term).

    HORIZON_MAP — signal → horizon:
        Tactical   (1–5 days):    GEX regime (via gex_state), VIX term structure,
                                  DXY 5D momentum
        Short-term (1–4 weeks):   HYG/LQD, small-cap leadership,
                                  net liquidity 4W impulse, ISM NO momentum
        Medium-term (1–3 months): Curve phase (not just level), copper/gold 13W,
                                  real credit impulse, real rate regime,
                                  reserve adequacy, M2 acceleration

    Percentile ranking uses current_pct_rank() (trailing 252-day window,
    current observation only — no full time-series construction).

    Magnitude information is preserved alongside the percentile rank
    where economically meaningful (e.g. real rate level vs threshold).
    """
    R = {}

    # ════════════════════════════════════════════════════════════════
    # HORIZON 1 — TACTICAL (1–5 days)
    # ════════════════════════════════════════════════════════════════

    # T1. VIX term structure: VIX / 63D realised vol ratio
    # High premium → fear priced in → bearish next-week signal (inverted)
    spy_a   = _to_1d(spy).reindex(idx).ffill()
    vix_a   = _to_1d(vix).reindex(idx).ffill()
    spy_ret = spy_a.pct_change()
    rvol    = spy_ret.rolling(63, min_periods=20).std() * np.sqrt(252) * 100
    vts     = vix_a / rvol.replace(0, np.nan)
    R["vix_ts_pct"]    = current_pct_rank(-vts, 63)   # 63-day window for tactical
    R["vix_ts_level"]  = float(vts.dropna().iloc[-1]) if vts.dropna().size else np.nan

    # T2. DXY 5D momentum (short window for tactical)
    dxy_a  = _to_1d(dxy).reindex(idx).ffill()
    dxy_5d = dxy_a.pct_change(5) * 100
    R["dxy_5d_pct"]    = current_pct_rank(-dxy_5d, 63)   # inverted: DXY up = risk-off

    # ════════════════════════════════════════════════════════════════
    # HORIZON 2 — SHORT-TERM (1–4 weeks)
    # ════════════════════════════════════════════════════════════════

    # S1. HYG/LQD ratio 1M momentum — credit spread compression/widening
    hyg_a = _to_1d(hyg).reindex(idx).ffill()
    lqd_a = _to_1d(lqd).reindex(idx).ffill()
    if hyg_a.dropna().size > 21 and lqd_a.dropna().size > 21:
        hl_mom = (hyg_a / lqd_a.replace(0, np.nan)).pct_change(21) * 100
        R["hyg_lqd_pct"] = current_pct_rank(hl_mom, 252)
    else:
        R["hyg_lqd_pct"] = 50.0

    # S2. Small-cap vs large-cap leadership (IWM/SPY) 3-week momentum
    iwm_a = _to_1d(iwm).reindex(idx).ffill()
    if iwm_a.dropna().size > 21 and spy_a.dropna().size > 21:
        rs_mom = (iwm_a / spy_a.replace(0, np.nan)).pct_change(21) * 100
        R["smallcap_pct"] = current_pct_rank(rs_mom, 252)
    else:
        R["smallcap_pct"] = 50.0

    # S3. Net liquidity 4-week impulse
    R["liq_impulse_4w_pct"]   = current_pct_rank(net_liq_4w, 252)
    R["liq_impulse_4w_level"] = float(net_liq_4w.dropna().iloc[-1]) if net_liq_4w.dropna().size else 0.0

    # S4. ISM Manufacturing New Orders momentum (if available)
    if ism_no is not None and len(ism_no.dropna()) > 6:
        ism_no_r = resample_ffill(ism_no, idx)
        ism_mom  = ism_no_r.diff(2)   # 2-month change (monthly series)
        # Classify: above/below 50 AND rising/falling → 4 quadrants
        ism_latest    = float(ism_no_r.dropna().iloc[-1]) if ism_no_r.dropna().size else 50.0
        ism_mom_latest = float(ism_mom.dropna().iloc[-1]) if ism_mom.dropna().size else 0.0
        # Bull signal: below 50 but rising (contraction decelerating) OR above 50 rising
        ism_bull = float(ism_latest + ism_mom_latest * 3)   # momentum-boosted score
        R["ism_no_pct"]    = current_pct_rank(ism_no_r + ism_mom * 2, 252)
        R["ism_level"]     = ism_latest
        R["ism_momentum"]  = ism_mom_latest
        # Quadrant classification
        if ism_latest > 50 and ism_mom_latest > 0:
            R["ism_quadrant"] = "Expansion Accelerating"
        elif ism_latest > 50 and ism_mom_latest <= 0:
            R["ism_quadrant"] = "Expansion Decelerating"
        elif ism_latest <= 50 and ism_mom_latest < 0:
            R["ism_quadrant"] = "Contraction Accelerating"
        else:
            R["ism_quadrant"] = "Contraction Decelerating"  # the buy signal
    else:
        R["ism_no_pct"]   = 50.0
        R["ism_level"]    = np.nan
        R["ism_momentum"] = np.nan
        R["ism_quadrant"] = "Unknown"

    # ════════════════════════════════════════════════════════════════
    # HORIZON 3 — MEDIUM-TERM (1–3 months)
    # ════════════════════════════════════════════════════════════════

    # M1. Yield curve PHASE — not just level or simple acceleration
    # Bull steepening (both rates falling, long faster) vs
    # Bear steepening (short falling, long stable/rising) are OPPOSITE signals
    y10_20d = y10.diff(20); y2_20d = y2.diff(20)
    bull_steepen = ((y10_20d < 0) & (y2_20d < y10_20d)).astype(float)  # both down, long faster
    bear_steepen = ((y2_20d  < 0) & (y10_20d >= 0)).astype(float)      # front end down, long sticky
    bear_flatten = ((y2_20d  > 0) & (y10_20d > y2_20d)).astype(float)  # bear flatten (late tighten)
    bull_flatten = ((y10_20d < 0) & (y2_20d  >= 0)).astype(float)      # long-end rally (risk-off)

    # Encode phase as numeric: bull_steepen=+2, bear_steepen=+1, neutral=0,
    # bull_flatten=-1 (safe haven bid), bear_flatten=-2 (policy error risk)
    curve_phase_raw = 2*bull_steepen + 1*bear_steepen - 1*bull_flatten - 2*bear_flatten
    R["curve_phase_pct"]  = current_pct_rank(curve_phase_raw, 252)
    R["curve_phase_label"] = (
        "Bull Steepen"  if bool(bull_steepen.iloc[-1])  else
        "Bear Steepen"  if bool(bear_steepen.iloc[-1])  else
        "Bull Flatten"  if bool(bull_flatten.iloc[-1])  else
        "Bear Flatten"  if bool(bear_flatten.iloc[-1])  else
        "Parallel Shift"
    )
    R["curve_raw"]  = float(s_2s10s.dropna().iloc[-1]) if s_2s10s.dropna().size else 0.0
    R["curve_inverted"] = R["curve_raw"] < 0

    # M2. Copper/Gold ratio 13-week momentum (medium-term window)
    copx_a = _to_1d(copx).reindex(idx).ffill()
    gld_a  = _to_1d(gld).reindex(idx).ffill()
    if copx_a.dropna().size > 91 and gld_a.dropna().size > 91:
        cg_mom = (copx_a / gld_a.replace(0, np.nan)).pct_change(91) * 100
        R["copper_gold_pct"]   = current_pct_rank(cg_mom, 252)
        R["copper_gold_13w"]   = float(cg_mom.dropna().iloc[-1]) if cg_mom.dropna().size else 0.0
    else:
        R["copper_gold_pct"]  = 50.0
        R["copper_gold_13w"]  = 0.0

    # M3. REAL credit impulse — change in bank credit flow / GDP
    # Biggs et al (2010): this leads equity returns by ~6 months
    if bank_credit is not None and gdp_quarterly is not None        and len(bank_credit.dropna()) > 91 and len(gdp_quarterly.dropna()) > 4:
        bc_r   = resample_ffill(bank_credit, idx)
        gdp_r  = resample_ffill(gdp_quarterly, idx).ffill()  # quarterly → daily ffill
        gdp_r  = gdp_r.replace(0, np.nan)
        credit_flow    = bc_r.diff(91)             # new credit created (quarterly flow)
        credit_impulse = credit_flow.diff(91) / gdp_r  # change in flow / GDP
        R["credit_impulse_pct"]   = current_pct_rank(credit_impulse, 252)
        R["credit_impulse_level"] = float(credit_impulse.dropna().iloc[-1]) if credit_impulse.dropna().size else 0.0
    else:
        # Fallback: M2 acceleration proxy (labelled clearly as inferior)
        ci     = m2.diff(91) / m2.shift(91) * 100
        ci_roc = ci.diff(63).dropna()
        R["credit_impulse_pct"]   = current_pct_rank(ci_roc, 252)
        R["credit_impulse_level"] = float(ci_roc.dropna().iloc[-1]) if ci_roc.dropna().size else 0.0
    R["credit_impulse_source"] = "TOTBKCR/GDP" if (bank_credit is not None and len(bank_credit.dropna()) > 91) else "M2 proxy"

    # M4. Real rate regime (DFII10: 10Y TIPS yield)
    # Financial repression (<0%) vs neutral (0-1.5%) vs compression (>1.5%) vs danger (>2.5%)
    if tips_10y is not None and len(tips_10y.dropna()) > 20:
        tips_r = resample_ffill(tips_10y, idx)
        real_rate_now = float(tips_r.dropna().iloc[-1]) if tips_r.dropna().size else np.nan
        if np.isfinite(real_rate_now):
            if real_rate_now < 0:
                rr_regime = "Repression"
            elif real_rate_now < 1.5:
                rr_regime = "Neutral"
            elif real_rate_now < 2.5:
                rr_regime = "Compression"
            else:
                rr_regime = "Danger"
            # Momentum matters too: rising real rates = valuation headwind
            rr_mom = tips_r.diff(63)   # 3M change
            R["real_rate_regime"]  = rr_regime
            R["real_rate_level"]   = real_rate_now
            R["real_rate_mom_pct"] = current_pct_rank(-rr_mom, 252)  # inverted: rising = bearish
            # Bull signal: level below 1.5% AND falling (multiple expansion supported)
            R["real_rate_pct"] = float(np.clip(
                (1.5 - real_rate_now) / 3.5 * 50 + 50 +   # level component
                current_pct_rank(-rr_mom, 252) * 0.2,       # momentum component
                5, 95
            ))
        else:
            R["real_rate_regime"] = "Unknown"; R["real_rate_level"] = np.nan
            R["real_rate_pct"]    = 50.0;      R["real_rate_mom_pct"] = 50.0
    else:
        R["real_rate_regime"] = "Unavailable"; R["real_rate_level"] = np.nan
        R["real_rate_pct"]    = 50.0;          R["real_rate_mom_pct"] = 50.0

    # M5. Reserve adequacy — bank reserves at Fed
    # Below ~$3T threshold = repo stress risk (Sep 2019 analogue)
    if bank_reserves is not None and len(bank_reserves.dropna()) > 20:
        res_r = resample_ffill(bank_reserves, idx)
        res_now_b = float(res_r.dropna().iloc[-1]) if res_r.dropna().size else np.nan  # billions
        # Historical mean for normalisation (or use hard threshold)
        res_mean  = float(res_r.rolling(504, min_periods=50).mean().dropna().iloc[-1])                     if res_r.rolling(504).mean().dropna().size else 3000.0
        adequacy_ratio = res_now_b / max(res_mean, 1.0)
        R["reserve_adequacy_ratio"] = adequacy_ratio
        R["reserve_level_bn"]       = res_now_b
        # Regime classification
        if adequacy_ratio >= 0.90:
            R["reserve_regime"] = "Ample"
        elif adequacy_ratio >= 0.75:
            R["reserve_regime"] = "Watch"
        else:
            R["reserve_regime"] = "Alert"   # repo stress risk zone
        R["reserve_pct"] = float(np.clip(adequacy_ratio * 50 + 25, 5, 95))
    else:
        R["reserve_adequacy_ratio"] = np.nan
        R["reserve_regime"]         = "Unavailable"
        R["reserve_level_bn"]       = np.nan
        R["reserve_pct"]            = 50.0

    # M6. M2 YoY growth (keep — medium-term liquidity backdrop)
    m2_yoy = (m2 / m2.shift(365) - 1) * 100
    R["m2_yoy"]     = float(m2_yoy.dropna().iloc[-1]) if m2_yoy.dropna().size else np.nan
    R["m2_yoy_pct"] = current_pct_rank(m2_yoy, 252)

    # M7. Net liquidity 13-week impulse (medium-term window)
    liq_13w = net_liq.diff(91)
    R["liq_impulse_13w_pct"]   = current_pct_rank(liq_13w, 252)
    R["liq_impulse_13w_level"] = float(liq_13w.dropna().iloc[-1]) if liq_13w.dropna().size else 0.0

    return R

# ============================================================
# PROBABILISTIC COMPOSITE
# ============================================================

# ============================================================
# 1-DAY PROBABILITY MODEL
# ============================================================
def compute_1d_prob(
    gex_state: GammaState,
    spot: float,
    vix_level: float,
    vix_series: pd.Series,      # full VIX history for percentile
    spy_series: pd.Series,      # SPY history for realised vol + momentum
    hyg_series: pd.Series,      # HYG for intraday credit proxy
    lqd_series: pd.Series,
    dxy_series: pd.Series,      # UUP for dollar intraday
    s_2s10s: pd.Series,         # curve raw bp
    net_liq_4w: pd.Series,      # 4W liquidity impulse
    nfci_z: pd.Series,          # NFCI z-score series
    fear_score: float,
    session: Dict,
    idx: pd.DatetimeIndex,
    sahm_rule=None,
    hy_spread=None,
) -> Dict:
    """
    1-Day directional probability — built around GEX mechanics.

    Signal architecture: 6 factors, each genuinely 1-day relevant.
    No signal from the 5-day or longer stack is included here because
    they answer a different question.

    ── FACTOR 1: GEX Regime & Proximity ──────────────────────────
    The single most important 1-day structural signal.
    Positive gamma → dealers buy dips, sell rallies → mean-reversion bias.
    Negative gamma → dealers amplify moves → trend/momentum bias.
    Flip proximity → binary outcome risk, compress toward 50.

    ── FACTOR 2: VIX Term Structure (intraday horizon) ──────────
    VIX / 5-day realised vol (not 63-day as in tactical bucket).
    The very near-term vol premium tells you whether fear is building
    TODAY, not over the next week.  High premium = options market
    pricing in near-term event risk.

    ── FACTOR 3: Intraday Momentum (5D SPY return) ──────────────
    Empirically: the strongest single-day predictor for next-day
    direction in the short run is the last 5 days of price momentum
    (Jegadeesh & Titman short-horizon reversion vs. momentum depends
    on GEX regime — positive GEX inverts it, negative GEX extends it).
    This is why we need GEX to interpret momentum correctly.

    ── FACTOR 4: Credit / Dollar Intraday Microstructure ─────────
    HYG/LQD intraday change and DXY intraday change are the two
    fastest-moving cross-asset signals.  Both lead equity by minutes
    to hours on a daily basis.  1-day change, not 21-day.

    ── FACTOR 5: Curve Inversion Status ─────────────────────────
    A simple binary: is the curve inverted?  On any given day, an
    inverted curve means carry is negative (short-term funding costs
    more than long-term), which creates a persistent structural headwind
    for risk assets regardless of regime.

    ── FACTOR 6: Session Context Multiplier ─────────────────────
    A 1-day probability is only meaningful if you can act on it.
    Outside prime-time (10:30-12:00 ET), liquidity is lower and
    intraday signals are noisier.  The session multiplier compresses
    the signal toward 50 during thin periods.

    ── GEX DIRECTION INTERPRETATION ─────────────────────────────
    This is the key conceptual point that separates this model from
    naive approaches:

    GEX does NOT give a direction by itself.  But it tells you HOW
    to interpret the other signals:

    In POSITIVE gamma:
      - Momentum signals should be FADED (dealers suppress extremes)
      - Credit/dollar signals are PRIMARY direction indicators
      - Base rate is mean-reversion to gamma flip

    In NEGATIVE gamma:
      - Momentum signals should be FOLLOWED (dealers amplify)
      - Credit/dollar signals CONFIRM the direction
      - Base rate is continuation

    In NEUTRAL (near flip):
      - No GEX signal — all signals equal weight
      - Compress toward 50, widen uncertainty

    ── PERFORMANCE EXPECTATION ──────────────────────────────────
    Realistic 1-day AUC: 0.52–0.55.  This is consistent with the
    academic literature on daily equity direction.  Anyone claiming
    >0.58 on 1-day equity direction is overfitting.  The value of
    this model is not a high AUC — it is conditional positioning:
    knowing WHEN to trade (session + GEX regime) and HOW MUCH to
    risk (Kelly with honest uncertainty).
    """
    # ── Factor 1: GEX regime ─────────────────────────────────────────────
    regime   = gex_state.regime
    dist_pct = abs(gex_state.distance_to_flip_pct)
    stability = gex_state.regime_stability

    # Base directional bias from GEX: positive = mean-reversion (neutral 50),
    # negative = amplification (still 50 directionally — GEX ≠ direction)
    # What GEX gives us is CONFIDENCE in the other signals, not direction itself
    gex_signal_confidence = float(np.clip(stability, 0.3, 1.0))

    # Flip proximity penalty — when within 0.5% of flip, all signals lose meaning
    flip_proximity_penalty = float(np.clip(1.0 - max(0, (0.75 - dist_pct) / 0.75), 0.4, 1.0))

    # ── Factor 2: VIX term structure (5D realised vol — intraday window) ──
    spy_a   = _to_1d(spy_series).reindex(idx).ffill()
    rvol_5d = spy_a.pct_change().rolling(5, min_periods=3).std() * np.sqrt(252) * 100
    vts_1d  = vix_level / float(rvol_5d.dropna().iloc[-1]) if rvol_5d.dropna().size and float(rvol_5d.dropna().iloc[-1]) > 0 else 1.0

    # High VIX/RVol ratio = fear premium = near-term bearish signal (inverted)
    # Ratio > 1.3: market pricing in more fear than recent realised → bearish 1D
    # Ratio < 0.8: complacency relative to realised vol → slightly bullish 1D
    if vts_1d > 1.5:   vts_score = 30.0   # high fear premium
    elif vts_1d > 1.2: vts_score = 42.0
    elif vts_1d > 0.9: vts_score = 50.0   # neutral zone
    elif vts_1d > 0.7: vts_score = 58.0
    else:              vts_score = 65.0   # complacency / vol crush

    # ── Factor 3: 5D SPY momentum (regime-conditioned) ───────────────────
    spy_5d_ret = float(spy_a.pct_change(5).dropna().iloc[-1]) if spy_a.pct_change(5).dropna().size else 0.0
    spy_1d_ret = float(spy_a.pct_change(1).dropna().iloc[-1]) if spy_a.pct_change(1).dropna().size else 0.0

    # Key insight: momentum interpretation depends on GEX regime
    if regime in (GammaRegime.STRONG_POSITIVE, GammaRegime.POSITIVE):
        # Positive gamma → FADE momentum (dealers mean-revert)
        # Recent strength = resistance ahead; recent weakness = support ahead
        if spy_5d_ret > 0.015:   mom_score = 38.0   # overbought into GEX resistance
        elif spy_5d_ret > 0.005: mom_score = 46.0
        elif spy_5d_ret > -0.005: mom_score = 52.0
        elif spy_5d_ret > -0.015: mom_score = 57.0
        else:                     mom_score = 63.0   # oversold, GEX support below
    elif regime in (GammaRegime.NEGATIVE, GammaRegime.STRONG_NEGATIVE):
        # Negative gamma → FOLLOW momentum (dealers amplify)
        if spy_5d_ret > 0.015:   mom_score = 62.0   # momentum continuation
        elif spy_5d_ret > 0.005: mom_score = 56.0
        elif spy_5d_ret > -0.005: mom_score = 50.0
        elif spy_5d_ret > -0.015: mom_score = 44.0
        else:                     mom_score = 36.0   # continuation downside
    else:
        # Neutral / near flip — equal weight, slight short-term reversion
        mom_score = 50.0 - spy_5d_ret * 200   # mild reversion
        mom_score = float(np.clip(mom_score, 35, 65))

    # ── Factor 4: Credit & dollar microstructure (1-day change) ─────────
    hyg_a   = _to_1d(hyg_series).reindex(idx).ffill()
    lqd_a   = _to_1d(lqd_series).reindex(idx).ffill()
    dxy_a   = _to_1d(dxy_series).reindex(idx).ffill()

    hyg_1d  = float(hyg_a.pct_change(1).dropna().iloc[-1]) if hyg_a.pct_change(1).dropna().size else 0.0
    lqd_1d  = float(lqd_a.pct_change(1).dropna().iloc[-1]) if lqd_a.pct_change(1).dropna().size else 0.0
    dxy_1d  = float(dxy_a.pct_change(1).dropna().iloc[-1]) if dxy_a.pct_change(1).dropna().size else 0.0

    # HYG/LQD spread tightening (HYG up relative to LQD) = risk-on = bullish
    credit_1d_signal = hyg_1d - lqd_1d * 0.5   # HYG weighted more
    # Dollar weakening = risk-on = bullish (inverted)
    dollar_signal    = -dxy_1d

    # Combine: both pointing same direction = stronger signal
    # Threshold-based scoring (same approach as vts/mom/curve — avoids clipping)
    # credit_1d_signal: HYG daily change minus half LQD daily change
    # dollar_signal: negative of DXY daily change (dollar up = bearish)
    if credit_1d_signal >  0.003:  credit_score = 65.0   # strong risk-on
    elif credit_1d_signal >  0.001: credit_score = 57.0
    elif credit_1d_signal > -0.001: credit_score = 50.0   # neutral
    elif credit_1d_signal > -0.003: credit_score = 43.0
    else:                           credit_score = 35.0   # strong risk-off

    if dollar_signal >  0.002:   dollar_score = 60.0   # dollar weak = bullish
    elif dollar_signal >  0.0005: dollar_score = 54.0
    elif dollar_signal > -0.0005: dollar_score = 50.0
    elif dollar_signal > -0.002:  dollar_score = 46.0
    else:                         dollar_score = 40.0   # dollar strong = bearish

    micro_score = float(credit_score * 0.6 + dollar_score * 0.4)

    # ── Factor 5: Curve inversion (structural 1D headwind/tailwind) ───────
    curve_bp = float(s_2s10s.dropna().iloc[-1]) if s_2s10s.dropna().size else 0.0
    if curve_bp < -50:    curve_score = 38.0   # deeply inverted — persistent headwind
    elif curve_bp < -10:  curve_score = 45.0   # inverted — mild headwind
    elif curve_bp < 10:   curve_score = 50.0   # near flat — neutral
    elif curve_bp < 50:   curve_score = 54.0   # normal — mild tailwind
    else:                 curve_score = 57.0   # steep — tailwind (carry positive)

    # ── Factor 6: Net liquidity 1-day direction ─────────────────────────
    # The 4W impulse is a slow signal; for 1D we use its sign + recent acceleration
    liq_4w_level = float(net_liq_4w.dropna().iloc[-1]) if net_liq_4w.dropna().size else 0.0
    liq_accel_1d  = float(net_liq_4w.diff(1).dropna().iloc[-1]) if net_liq_4w.diff(1).dropna().size else 0.0
    liq_score = float(np.clip(52.0 + np.sign(liq_4w_level) * 5 + np.sign(liq_accel_1d) * 3, 30, 70))

    # ── Combine factors with GEX-regime-aware weights ────────────────────
    #
    # In POSITIVE GAMMA:   credit/dollar microstructure matters most (what's
    #                       actually flowing TODAY), momentum is faded
    # In NEGATIVE GAMMA:   momentum matters most (dealers amplifying),
    #                       credit confirms, VIX TS is critical
    # In NEUTRAL:          equal weights, all compressed toward 50
    #
    if regime in (GammaRegime.STRONG_POSITIVE, GammaRegime.POSITIVE):
        # Positive gamma: mean-reversion dominant, credit confirms, momentum faded
        w = {"vts": 0.20, "mom": 0.15, "micro": 0.25, "curve": 0.15, "liq": 0.25}
    elif regime in (GammaRegime.NEGATIVE, GammaRegime.STRONG_NEGATIVE):
        # Negative gamma: momentum matters most, VIX term structure critical
        w = {"vts": 0.25, "mom": 0.30, "micro": 0.20, "curve": 0.10, "liq": 0.15}
    else:
        # Neutral: balanced weights
        w = {"vts": 0.22, "mom": 0.20, "micro": 0.22, "curve": 0.16, "liq": 0.20}

    raw_1d = (
        w["vts"]   * vts_score   +
        w["mom"]   * mom_score   +
        w["micro"] * micro_score +
        w["curve"] * curve_score +
        w["liq"]   * liq_score
    )

    # ── GEX confidence scaling ─────────────────────────────────────────
    # Compress toward 50 when GEX is uncertain (near flip) or stability low
    # This is the correct way to incorporate GEX into direction: as a
    # confidence scalar on the other signals, not as a direction vote
    scaled = 50.0 + (raw_1d - 50.0) * gex_signal_confidence * flip_proximity_penalty

    # ── Session context ─────────────────────────────────────────────────────────
    # session_mult controls TRADING SIZE not the probability model.
    # When session_mult=0 (Globex 6pm-9:30am), the old code did:
    #   scaled = 50 + (scaled-50) * 0 = exactly 50 for 15+ hours every day.
    # Fix: leave the probability unaffected by session. 
    # The session_valid flag in the output tells the UI whether to act on it.
    session_mult = session.get("size_mult", 0.5)

    # ── Fear overlay (coincident risk-off condition) ──────────────────
    # High fear compresses upside potential modestly — not a hard cap
    if fear_score > 70:
        scaled = 50.0 + (scaled - 50.0) * 0.75   # compress deviation by 25% in high fear
    elif fear_score > 55:
        scaled = 50.0 + (scaled - 50.0) * 0.90   # compress deviation by 10% in elevated fear

    # Sahm Rule recession gate
    if sahm_rule is not None:
        try:
            sahm_val = float(sahm_rule.dropna().iloc[-1])
            if sahm_val >= 0.50:   scaled = min(scaled, 55.0)
            elif sahm_val >= 0.30: scaled = min(scaled, 65.0)
        except: pass
    # HY spread stress gate
    if hy_spread is not None:
        try:
            hy_val = float(hy_spread.dropna().iloc[-1])
            if hy_val > 600:   scaled = 50.0 + (scaled - 50.0) * 0.50
            elif hy_val > 400: scaled = 50.0 + (scaled - 50.0) * 0.75
        except: pass
    prob_1d = float(np.clip(scaled, 10, 90))

    # ── Uncertainty band (wider than longer horizons — more noise) ─────
    # 1-day direction has ~±15pp structural uncertainty
    base_unc_1d = 15.0
    # Widen further near gamma flip (regime is unstable)
    flip_extra  = max(0, (0.75 - dist_pct) / 0.75) * 8.0
    # Widen in high VIX (vol-of-vol is high)
    vix_extra   = max(0, (vix_level - 20) / 30) * 5.0
    unc_1d = round(base_unc_1d + flip_extra + vix_extra, 1)

    lo_1d = float(np.clip(prob_1d - unc_1d, 5, 95))
    hi_1d = float(np.clip(prob_1d + unc_1d, 5, 95))

    # ── Kelly for 1D (conservative — shortest horizon, most noise) ────
    kelly_1d = kelly(prob_1d, payoff=1.0) * 0.35   # 35% Kelly (more conservative than 50%)

    # ── Narrative ─────────────────────────────────────────────────────
    regime_interp = {
        GammaRegime.STRONG_POSITIVE: "Strong pos. gamma: dealers PIN price, fade extremes",
        GammaRegime.POSITIVE:        "Pos. gamma: mean-reversion bias, fade momentum",
        GammaRegime.NEUTRAL:         "Near gamma flip: binary risk, reduce size",
        GammaRegime.NEGATIVE:        "Neg. gamma: moves amplified, follow momentum",
        GammaRegime.STRONG_NEGATIVE: "Strong neg. gamma: cascades possible, trend continuation",
    }

    # Dominant signal
    scores = {"VIX TS": vts_score, "Momentum": mom_score,
              "Credit/FX": micro_score, "Curve": curve_score, "Liquidity": liq_score}
    dominant = max(scores, key=lambda k: abs(scores[k] - 50))
    dominant_dir = "bullish" if scores[dominant] > 50 else "bearish"

    return {
        "prob_1d":        prob_1d,
        "lo_1d":          lo_1d,
        "hi_1d":          hi_1d,
        "unc_1d":         unc_1d,
        "kelly_1d":       kelly_1d,
        # Component scores (0-100, 50=neutral)
        "score_vts":      vts_score,
        "score_mom":      mom_score,
        "score_micro":    micro_score,
        "score_curve":    curve_score,
        "score_liq":      liq_score,
        "vts_ratio":      vts_1d,
        "spy_5d_ret":     spy_5d_ret,
        "credit_1d":      credit_1d_signal,
        "dollar_1d":      dollar_signal,
        "dominant_signal": dominant,
        "dominant_dir":    dominant_dir,
        "regime_interp":   regime_interp.get(regime, ""),
        "gex_confidence":  gex_signal_confidence,
        "flip_proximity":  flip_proximity_penalty,
        "session_valid":   session_mult >= 0.5,
        "sahm_triggered":  (float(sahm_rule.dropna().iloc[-1]) >= 0.50) if (sahm_rule is not None and hasattr(sahm_rule,"dropna") and sahm_rule.dropna().size) else False,
        "hy_spread_level": float(hy_spread.dropna().iloc[-1]) if (hy_spread is not None and hasattr(hy_spread,"dropna") and hy_spread.dropna().size) else 0.0,
        "_note": (
            f"1D model: GEX-regime-conditioned. "
            f"Realistic AUC 0.52-0.55. "
            f"Range: {lo_1d:.0f}-{hi_1d:.0f}%."
        ),
    }


