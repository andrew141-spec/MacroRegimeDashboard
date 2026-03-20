# probability.py — probability composite, session context, setups, regime
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
from config import GammaState, GammaRegime, SetupScore
from utils import _to_1d, kelly, current_pct_rank

def compute_prob_composite(leading, fear_score, geo_shock, regime_change_p,
                            gex_state: GammaState,
                            nfci_coincident: float = 50.0,
                            liq_dir_coincident: float = 50.0) -> Dict:
    """
    Three-horizon probability composite — separating signals by forecast window.

    ARCHITECTURE:
      Tactical bucket  (1–5d):   VIX term structure, DXY 5D momentum
      Short-term bucket (1–4w):  HYG/LQD, small-cap, liq impulse 4W, ISM NO
      Medium-term bucket (1–3m): Curve phase, Cu/Au 13W, credit impulse,
                                 real rate regime, reserve adequacy, M2 YoY,
                                 liq impulse 13W

    Signals are averaged WITHIN each bucket before blending across buckets.
    This eliminates intra-bucket double-counting (the core critique).

    GEX feeds execution context only (not direction). It contributes a
    volatility regime label and a size scalar, not a directional probability.

    Uncertainty band is explicit and regime-aware.
    """

    # ── Tactical bucket ───────────────────────────────────────────────────
    tactical_signals = [
        leading.get("vix_ts_pct",      50.0),
        leading.get("dxy_5d_pct",      50.0),
    ]
    tactical_prob = float(np.mean(tactical_signals))

    # ── Short-term bucket ────────────────────────────────────────────────
    short_signals = [
        leading.get("hyg_lqd_pct",          50.0),
        leading.get("smallcap_pct",          50.0),
        leading.get("liq_impulse_4w_pct",    50.0),
        leading.get("ism_no_pct",            50.0),
    ]
    short_prob = float(np.mean(short_signals))

    # ── Medium-term bucket ───────────────────────────────────────────────
    medium_signals = [
        leading.get("curve_phase_pct",       50.0),
        leading.get("copper_gold_pct",        50.0),
        leading.get("credit_impulse_pct",     50.0),
        leading.get("real_rate_pct",          50.0),
        leading.get("reserve_pct",            50.0),
        leading.get("m2_yoy_pct",             50.0),
        leading.get("liq_impulse_13w_pct",    50.0),
    ]
    medium_prob = float(np.mean(medium_signals))

    # ── Coincident conditions (3 orthogonal signals) ────────────────────
    # Each measures a genuinely distinct underlying factor:
    #   fear_score   → VIX-based equity stress (volatility surface)
    #   nfci_signal  → NFCI-based financial conditions (credit/funding)
    #   liq_dir      → current net liquidity direction (Fed plumbing flow)
    #
    # Explicitly excluded from this bucket because they share underlying
    # data with the signals above:
    #   three_puts_score  — contains y10_20 (yield curve, already in short bucket)
    #   liq_anxiety       — contains policy_mistrust (yield curve again)
    #   exhaustion        — contains policy_mistrust (yield curve a 3rd time)
    #   market_index      — contains growth_z (2s10s z-score, yield curve again)
    #
    # The yield curve appeared in all 5 elements of the old bucket.
    # Effective signal count was ~2. Now it is 3 with minimal overlap.
    # nfci_coincident and liq_dir_coincident are pre-computed in render_dashboard
    # and passed as arguments — they reference local series (nfci, net_liq_4w, idx)
    # that are not in scope inside this standalone function.
    coincident_conditions = float(np.mean([
        100 - fear_score,       # VIX / equity stress
        nfci_coincident,        # NFCI financial conditions (credit, funding)
        liq_dir_coincident,     # net liquidity direction: 70=expanding, 30=draining
    ]))

    # ── Cross-bucket blend ────────────────────────────────────────────────
    # Weights reflect empirical lead times — medium-term leads most,
    # tactical is highest-frequency noise.
    # Coincident conditions are a reality check, not a leading indicator.
    BUCKET_W = {
        "tactical":    0.10,
        "short_term":  0.30,
        "medium_term": 0.40,
        "coincident":  0.20,
    }
    raw_composite = (
        BUCKET_W["tactical"]   * tactical_prob  +
        BUCKET_W["short_term"] * short_prob     +
        BUCKET_W["medium_term"]* medium_prob    +
        BUCKET_W["coincident"] * coincident_conditions
    )

    # ── GEX volatility regime adjustment ─────────────────────────────────
    # GEX does NOT vote on direction — it adjusts for volatility regime.
    # Positive gamma → mean-reversion expected → compress toward 50 slightly
    # Negative gamma → amplification expected → expand away from 50 slightly
    gex_adjustment = 0.0
    if gex_state.regime in (GammaRegime.STRONG_POSITIVE, GammaRegime.POSITIVE):
        # Dampen extremes in positive gamma (moves are suppressed)
        gex_adjustment = (50.0 - raw_composite) * 0.08 * gex_state.regime_stability
    elif gex_state.regime in (GammaRegime.NEGATIVE, GammaRegime.STRONG_NEGATIVE):
        # Amplify signal in negative gamma (moves are amplified)
        gex_adjustment = (raw_composite - 50.0) * 0.08 * gex_state.regime_stability

    raw = float(np.clip(raw_composite + gex_adjustment, 5, 95))

    # ── Geo drag ──────────────────────────────────────────────────────────
    geo_drag = float(geo_shock / 100 * 15)   # max 15pt (was 25pt — geo scoring is noisy)

    # ── Regime uncertainty compression ───────────────────────────────────
    unc        = regime_change_p / 100
    compressed = 50.0 + (raw - 50.0) * (1.0 - unc * 0.45) - geo_drag
    bull_prob  = float(np.clip(compressed, 5, 95))

    # ── Honest uncertainty band ───────────────────────────────────────────
    # Effective independent groups: ~3-4 (not 14 signals)
    # Base uncertainty ±10pp + regime + geo widening
    base_unc     = 10.0
    regime_extra = unc * 5.0
    geo_extra    = (geo_shock / 100) * 3.0
    unc_band     = round(base_unc + regime_extra + geo_extra, 1)
    bull_lo      = float(np.clip(bull_prob - unc_band, 5, 95))
    bull_hi      = float(np.clip(bull_prob + unc_band, 5, 95))

    # ── Per-horizon Kelly fractions ───────────────────────────────────────
    # Each horizon paired with its natural payoff assumption:
    #   Tactical (5D):    mean-reversion trades, typical R:R ~1.3 (Setup 5 pin)
    #   Short-term (21D): directional with stop, typical R:R ~2.0 (Setup 1/2)
    #   Medium-term (63D): swing position, typical R:R ~2.5 (Setup 3/4)
    # Half-Kelly throughout for position sizing discipline.
    kelly_5d  = kelly(tactical_prob,  payoff=1.3) * 0.5
    kelly_21d = kelly(short_prob,     payoff=2.0) * 0.5
    kelly_63d = kelly(medium_prob,    payoff=2.5) * 0.5

    return {
        # ── Per-horizon probabilities (the primary outputs) ───────────────
        "prob_5d":       tactical_prob,
        "prob_21d":      short_prob,
        "prob_63d":      medium_prob,
        "coincident":    coincident_conditions,
        # ── Uncertainty-bounded composite (for display only) ─────────────
        "bull_prob":     bull_prob,
        "bull_rounded":  round(bull_prob / 5) * 5,
        "bull_lo":       bull_lo,
        "bull_hi":       bull_hi,
        "uncertainty":   unc_band,
        "bear_prob":     100 - bull_prob,
        # ── Per-horizon Kelly fractions ───────────────────────────────────
        "kelly_5d":      kelly_5d,
        "kelly_21d":     kelly_21d,
        "kelly_63d":     kelly_63d,
        "half_kelly":    kelly_21d,   # kept for legacy UI references
        # ── Forward-looking vs coincident (renamed from leading/lagging) ──
        "fwd_prob":      float(np.mean([short_prob, medium_prob])),
        "coincident_prob": coincident_conditions,
        # Legacy aliases so WIM / other UI code doesn't break
        "leading_prob":  float(np.mean([short_prob, medium_prob])),
        "lagging_prob":  coincident_conditions,
        "tactical_prob": tactical_prob,
        "short_prob":    short_prob,
        "medium_prob":   medium_prob,
        "group_scores": {
            "tactical":   tactical_prob,
            "short_term": short_prob,
            "medium":     medium_prob,
            "coincident": coincident_conditions,
        },
        # ── Adjustments ───────────────────────────────────────────────────
        "gex_adjustment": gex_adjustment,
        "geo_drag":       geo_drag,
        # ── Cross-horizon divergence ──────────────────────────────────────
        "asymmetry":     abs(bull_prob - 50) / 50 * 100,
        "divergent":     abs(tactical_prob - medium_prob) > 20,
        "divergence_gap":abs(tactical_prob - medium_prob),
        "_precision_note": (
            f"3-horizon model: 5D={tactical_prob:.0f}% | "
            f"21D={short_prob:.0f}% | 63D={medium_prob:.0f}%. "
            f"Composite range: {bull_lo:.0f}–{bull_hi:.0f}%."
        ),
    }

# ============================================================
# ECONOMIC CALENDAR
# ============================================================

# FOMC meeting dates (decision day = second day of 2-day meeting)
# Updated through end of 2026. Add new dates as Fed publishes schedule.
_FOMC_DATES = {
    # 2024
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12",
    "2024-07-31","2024-09-18","2024-11-07","2024-12-18",
    # 2025
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18",
    "2025-07-30","2025-09-17","2025-10-29","2025-12-17",
    # 2026
    "2026-01-28","2026-03-18","2026-04-29","2026-06-17",
    "2026-07-29","2026-09-16","2026-10-28","2026-12-16",
}

# CPI release dates (BLS releases ~13th of following month)
# These are approximate — BLS announces exact dates ~1 year ahead.
_CPI_DATES = {
    # 2025
    "2025-01-15","2025-02-12","2025-03-12","2025-04-10",
    "2025-05-13","2025-06-11","2025-07-11","2025-08-13",
    "2025-09-10","2025-10-15","2025-11-13","2025-12-10",
    # 2026
    "2026-01-14","2026-02-11","2026-03-11","2026-04-09",
    "2026-05-13","2026-06-10","2026-07-09","2026-08-12",
    "2026-09-09","2026-10-14","2026-11-12","2026-12-09",
}

# NFP / Jobs report (BLS, first Friday of each month)
_NFP_DATES = {
    # 2025
    "2025-01-10","2025-02-07","2025-03-07","2025-04-04",
    "2025-05-02","2025-06-06","2025-07-03","2025-08-01",
    "2025-09-05","2025-10-03","2025-11-07","2025-12-05",
    # 2026
    "2026-01-09","2026-02-06","2026-03-06","2026-04-03",
    "2026-05-08","2026-06-05","2026-07-02","2026-08-07",
    "2026-09-04","2026-10-02","2026-11-06","2026-12-04",
}

# PCE / Core PCE (BEA, ~last business day of following month)
_PCE_DATES = {
    # 2025
    "2025-01-31","2025-02-28","2025-03-28","2025-04-30",
    "2025-05-30","2025-06-27","2025-07-31","2025-08-29",
    "2025-09-26","2025-10-31","2025-11-26","2025-12-19",
    # 2026
    "2026-01-30","2026-02-27","2026-03-27","2026-04-30",
    "2026-05-29","2026-06-26","2026-07-31","2026-08-28",
    "2026-09-25","2026-10-30","2026-11-25","2026-12-18",
}

# GDP advance estimate (BEA, ~last Wednesday of month following quarter end)
_GDP_DATES = {
    "2025-01-29","2025-04-30","2025-07-30","2025-10-29",
    "2026-01-28","2026-04-29","2026-07-29","2026-10-28",
}

_ALL_DATA_DAYS = _FOMC_DATES | _CPI_DATES | _NFP_DATES | _PCE_DATES | _GDP_DATES


def get_calendar_context(date: dt.date = None) -> Dict:
    """
    Return what economic events are scheduled today (or on the given date).
    Used to set is_data_day and event_type in session context.
    """
    d = date or dt.date.today()
    d_str = d.isoformat()

    events = []
    if d_str in _FOMC_DATES: events.append("FOMC")
    if d_str in _CPI_DATES:  events.append("CPI")
    if d_str in _NFP_DATES:  events.append("NFP")
    if d_str in _PCE_DATES:  events.append("PCE")
    if d_str in _GDP_DATES:  events.append("GDP")

    is_data_day = len(events) > 0
    is_fomc     = "FOMC" in events
    # FOMC days get extra penalty: entire session is event-driven
    size_penalty = 0.5 if is_fomc else (0.75 if is_data_day else 1.0)

    return {
        "is_data_day":  is_data_day,
        "is_fomc":      is_fomc,
        "events":       events,
        "event_label":  " + ".join(events) if events else "",
        "size_penalty": size_penalty,   # multiply onto session size_mult
    }


# ============================================================
# SESSION CONTEXT
# ============================================================
def get_session_context() -> Dict:
    now_et = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=4)
    h, m = now_et.hour, now_et.minute
    total_mins = h * 60 + m
    weekday = now_et.weekday()

    sessions = [
        (0,   9*60+30,  "Globex",      "Very Low",    0.0,  "Monitor only — do NOT trade gamma setups"),
        (9*60+30, 9*60+45, "RTH Open",  "Very High",  0.0,  "Opening rotation — do NOT trade (noise > signal)"),
        (9*60+45, 10*60+30,"IB Forming","High",        0.5,  "First cautious entries possible if confirmation is strong"),
        (10*60+30,12*60,   "Morning",   "High/Stable", 1.0,  "PRIME TIME — deploy all setups at full regime-adjusted size"),
        (12*60,   14*60,   "Midday",    "Reduced",     0.35, "Reduce size 50%+ — thin book, false signals common"),
        (14*60,   15*60,   "Afternoon", "Increasing",  0.65, "Reassess 0DTE gamma — selective entries only"),
        (15*60,   16*60,   "Close/MOC", "Very High",   0.25, "MOC flow overrides gamma — pin trades only after 15:30"),
        (16*60,   24*60,   "Post-RTH",  "Low",         0.0,  "No gamma setups — monitor only"),
    ]
    for start_m, end_m, name, liq, mult, note in sessions:
        if start_m <= total_mins < end_m:
            is_opex_friday = (weekday == 4)
            cal = get_calendar_context(now_et.date())
            is_data_day   = cal["is_data_day"]
            # Reduce size mult on high-impact data days
            effective_mult = mult * cal["size_penalty"]
            event_note = f" · ⚠ {cal['event_label']} TODAY" if is_data_day else ""
            return {
                "window": name, "liquidity": liq,
                "size_mult": effective_mult,
                "size_mult_base": mult,
                "note": note + event_note,
                "time_et": now_et.strftime("%H:%M ET"),
                "is_opex_friday": is_opex_friday,
                "is_data_day": is_data_day,
                "is_fomc": cal["is_fomc"],
                "events": cal["events"],
                "event_label": cal["event_label"],
                "prime_time": name == "Morning",
            }
    cal = get_calendar_context()
    return {"window": "Unknown", "liquidity": "Unknown", "size_mult": 0.5,
            "size_mult_base": 0.5, "note": "Cannot determine session",
            "time_et": now_et.strftime("%H:%M ET"),
            "is_opex_friday": False, "is_data_day": cal["is_data_day"],
            "is_fomc": cal["is_fomc"], "events": cal["events"],
            "event_label": cal["event_label"], "prime_time": False}

# ============================================================
# TRADE SETUP EVALUATOR (5 setups from note 07)
# ============================================================
SETUPS = {
    1: ("Gamma Support Bounce",    "Positive regime. Price at GEX support from above. Dealers must buy.", 0.55, 1.0),
    2: ("Gamma Resistance Fade",   "Positive regime. Price at GEX resistance from below. Dealers must sell.", 0.52, 1.0),
    3: ("Gamma Flip Breakout",     "Price crossing gamma flip. Regime transition — amplified directional move.", 0.45, 0.75),
    4: ("Exhaustion Reversal",     "Extended negative gamma move. Volume climax + delta divergence = reversal.", 0.40, 0.50),
    5: ("0DTE Gamma Pin",          "Late session. Dominant 0DTE strike = gravitational pin. Fade deviations.", 0.65, 1.0),
}

def evaluate_setups(gex_state: GammaState, session: Dict, spot: float,
                    fear_score: float, vix_level: float) -> List[Dict]:
    results = []
    regime = gex_state.regime
    flip = gex_state.gamma_flip
    dist_pct = gex_state.distance_to_flip_pct
    stability = gex_state.regime_stability
    size_mult = session["size_mult"]
    vol_adj = 0.5 if vix_level > 35 else (0.75 if vix_level > 25 else 1.0)

    near_flip = abs(dist_pct) < 0.75
    at_support = any(abs(spot - s) / spot < 0.003 for s in gex_state.key_support)
    at_resistance = any(abs(spot - r) / spot < 0.003 for r in gex_state.key_resistance)
    late_session = session["window"] in ("Afternoon", "Close/MOC")
    prime_time = session["prime_time"]

    for setup_num, (name, desc, base_hr, base_size) in SETUPS.items():
        active = False
        score = SetupScore()
        note = ""

        if setup_num == 1:
            active = (regime in (GammaRegime.STRONG_POSITIVE, GammaRegime.POSITIVE) and at_support)
            score.gamma_alignment = 0.9 if regime == GammaRegime.STRONG_POSITIVE else 0.75
            score.orderflow_confirmation = 0.5  # Needs real-time confirmation
            score.tpo_context = 0.6
            score.level_freshness = 0.9
            score.event_risk = 0.8 if not session["is_data_day"] else 0.2
            note = "WAIT for absorption on footprint + delta flip before entry"

        elif setup_num == 2:
            active = (regime in (GammaRegime.STRONG_POSITIVE, GammaRegime.POSITIVE) and at_resistance)
            score.gamma_alignment = 0.85 if regime == GammaRegime.STRONG_POSITIVE else 0.70
            score.orderflow_confirmation = 0.5
            score.tpo_context = 0.6
            score.level_freshness = 0.85
            score.event_risk = 0.8 if not session["is_data_day"] else 0.2
            note = "Marginally lower hit rate than Setup 1 due to structural long bias"

        elif setup_num == 3:
            active = near_flip and session["window"] in ("IB Forming","Morning","Afternoon")
            score.gamma_alignment = 0.8 if near_flip else 0.3
            score.orderflow_confirmation = 0.4  # Need initiative selling confirmation
            score.tpo_context = 0.5
            score.level_freshness = 0.9
            score.event_risk = 0.7 if not session["is_data_day"] else 0.15
            note = f"Watch for initiative break through flip ({flip:.0f}). No absorption = regime change."

        elif setup_num == 4:
            neg_regime = regime in (GammaRegime.NEGATIVE, GammaRegime.STRONG_NEGATIVE)
            active = neg_regime and abs(dist_pct) > 2.0
            score.gamma_alignment = 0.7 if neg_regime else 0.2
            score.orderflow_confirmation = 0.3  # Needs exhaustion evidence
            score.tpo_context = 0.5
            score.level_freshness = 0.8
            score.event_risk = 0.7 if not session["is_data_day"] else 0.2
            note = "LOWEST hit rate. 50% size only. Wait for volume climax + delta divergence. 3:1 R:R minimum."

        elif setup_num == 5:
            # Note 09: pin trades most potent 10:30AM-12PM on OpEx, and after 14:00 any day
            morning_opex = session["is_opex_friday"] and session["window"] == "Morning"
            active = (late_session or morning_opex) and len(gex_state.key_resistance) > 0
            score.gamma_alignment = 0.9 if morning_opex else (0.8 if late_session else 0.2)
            score.orderflow_confirmation = 0.5
            score.tpo_context = 0.6
            score.level_freshness = 0.95
            score.event_risk = 0.9 if not session["is_data_day"] else 0.3
            note = "Only valid after 14:00 ET or OpEx morning 10:30-12:00. Pin strengthens as expiry approaches. Exit before 15:50."

        effective_size = base_size * size_mult * vol_adj
        results.append({
            "setup": setup_num, "name": name, "desc": desc,
            "active": active, "score": score, "est_hit_rate": base_hr,
            "effective_size": effective_size, "note": note,
            "regime_ok": active,
        })
    return results

# ============================================================
# FAILURE MODE CHECKER
# ============================================================
FAILURE_MODES = {
    1: ("Stale GEX Data",          "GEX data may not reflect intraday OI changes. No reaction at levels = dead level."),
    2: ("Gamma Level Crowding",     "Level published by multiple services. Front-running and stop hunts likely."),
    3: ("Regime Transition Mid-Trade","Price approaching gamma flip. Mean-reversion thesis invalidated if flip breaks."),
    4: ("Exogenous Shock Override", "Macro event flow > dealer hedging. GEX levels become irrelevant."),
    5: ("Footprint Spoofing",       "Single large absorption print may be manufactured. Require 3+ bars of confirmation."),
    6: ("Correlation Regime Break", "Cross-asset drive (bonds/FX) overriding options-mechanical flow."),
    7: ("OpEx Regime Decay",        "Expiring options removing gamma. Post-OpEx vacuum — levels weakening."),
    8: ("Delta-Hedging Timing Lag", "Dealers hedge in bursts, not continuously. Level is a zone, not a precise price."),
}

def check_failure_modes(gex_state: GammaState, session: Dict,
                        vix_level: float, is_data_day=False) -> List[Tuple[int, str, str, bool]]:
    """Returns list of (mode_id, name, desc, triggered)"""
    triggered = []
    dist = abs(gex_state.distance_to_flip_pct)

    fm1 = gex_state.data_source == "yfinance" and session["window"] in ("Afternoon","Close/MOC")

    # FM2: Crowding heuristic — note 10 §2. We can't detect real-time options flow,
    # but we can flag when the gamma flip level is a round number (high crowd visibility)
    # and regime stability is high (level has been "sitting there" long enough to be
    # widely published). Round numbers ending in 0 or 5 at the nearest integer are
    # most likely to be crowded per note 10 §2.
    flip = gex_state.gamma_flip
    flip_is_round = (flip > 0) and (round(flip) % 5 == 0)
    fm2 = flip_is_round and gex_state.regime_stability > 0.6

    fm3 = dist < 0.5
    fm4 = is_data_day or vix_level > 28
    fm6 = vix_level > 25
    fm7 = session["is_opex_friday"]
    fm8 = True  # Always relevant — dealers hedge in bursts, not continuously

    active = {1: fm1, 2: fm2, 3: fm3, 4: fm4, 5: False, 6: fm6, 7: fm7, 8: fm8}
    return [(k, FAILURE_MODES[k][0], FAILURE_MODES[k][1], active.get(k, False))
            for k in FAILURE_MODES]

# ============================================================
# REGIME TRANSITION PROBABILITY
# ============================================================
def classify_macro_regime_abs(core_cpi_yoy: float, curve_raw_bp: float,
                               gdp_qoq: Optional[float] = None) -> str:
    """
    Classify macro regime using ABSOLUTE economic thresholds, not z-scores.

    Z-score classification (old approach) was wrong because:
    - The boundary at z=0 shifts with the sample window
    - In a sustained bear market, all readings look "normal" relative to recent history
    - Small oscillations around z=0 cause spurious regime flips

    Absolute thresholds:
    - Inflation: CPI YoY vs 3.0% (elevated) and 2.0% (target) thresholds
    - Growth proxy: yield curve level vs 0bp (inversion), not vs its mean
    """
    high_inflation = core_cpi_yoy > 3.0
    inv_curve      = curve_raw_bp < 0

    if not high_inflation and not inv_curve:
        return "Goldilocks"
    elif not high_inflation and inv_curve:
        return "Deflation/Recession"
    elif high_inflation and not inv_curve:
        return "Overheating"
    else:
        return "Stagflation"


def regime_transition_prob(
    macro_regime: str,
    core_cpi_yoy: pd.Series,   # raw CPI series, not z-scored
    curve_raw: pd.Series,      # 2s10s raw bp, not z-scored
    window: int = 63
) -> Dict:
    """
    Estimate P(regime change in next 20 days) using absolute thresholds
    and a simple empirical Markov persistence measure.

    Improvements over the old z-score approach:
    1. Absolute thresholds (CPI>3%, curve<0) — not regime-window-relative
    2. Regime changes only counted when the absolute classification flips,
       not when a z-score oscillates around zero
    3. Serial correlation accounted for via persistence penalty
    4. Reports days in current regime and days since last change
    """
    cpi_s   = _to_1d(core_cpi_yoy).dropna()
    curve_s = _to_1d(curve_raw).dropna()

    if len(cpi_s) < 21 or len(curve_s) < 21:
        return {"current": macro_regime, "p_change_20d": 30.0, "persistence": 0,
                "last_change_days": None}

    # Align on common index
    common = cpi_s.index.intersection(curve_s.index)
    if len(common) < 21:
        return {"current": macro_regime, "p_change_20d": 30.0, "persistence": 0,
                "last_change_days": None}

    cpi_c   = cpi_s.reindex(common)
    curve_c = curve_s.reindex(common)

    # Classify regime at each date using absolute thresholds
    regimes = pd.Series([
        classify_macro_regime_abs(float(cpi_c.iloc[i]), float(curve_c.iloc[i]))
        for i in range(len(common))
    ], index=common)

    current       = regimes.iloc[-1]
    regime_changes = (regimes != regimes.shift(1)).iloc[1:]  # True on change days

    # How long in current regime?
    persistence = 0
    for i in range(len(regimes) - 1, -1, -1):
        if regimes.iloc[i] == current:
            persistence += 1
        else:
            break

    # Days since last regime change
    change_indices = regime_changes[regime_changes].index
    last_change_days = int((regimes.index[-1] - change_indices[-1]).days)                        if len(change_indices) > 0 else None

    # Rolling change rate over trailing window
    recent      = regime_changes.iloc[-window:]
    n_changes   = int(recent.sum())
    change_rate = n_changes / max(window, 1)

    # P(at least one change in next 20 days)
    # Using geometric: P = 1 - (1 - rate)^20, but with serial correlation correction
    # Macro regimes are highly autocorrelated — penalise based on persistence
    # Longer persistence → lower effective rate (momentum)
    momentum_discount = max(0.4, 1.0 - 0.015 * persistence)   # decays to 40% floor
    adj_rate   = change_rate * momentum_discount
    p_change   = float(np.clip((1 - (1 - adj_rate) ** 20) * 100, 0, 88))

    return {
        "current":          current,
        "p_change_20d":     p_change,
        "persistence":      persistence,
        "last_change_days": last_change_days,
        "n_changes_window": n_changes,
    }

# ============================================================
# DRIVER ALERTS
# ============================================================


def driver_alerts(prev: dict, now: dict) -> list:
    """Compare prev and now state dicts, return list of alert strings."""
    alerts = []
    thresholds = {
        "Fear": 8, "Three Puts": 8, "Liquidity Anxiety": 10,
        "Exhaustion": 10, "Market Index": 12, "Bull Prob": 8,
    }
    for k, thr in thresholds.items():
        if k in prev and k in now:
            import numpy as np
            if np.isfinite(prev.get(k, float("nan"))) and np.isfinite(now.get(k, float("nan"))):
                d = now[k] - prev[k]
                if abs(d) >= thr:
                    alerts.append(f"{'↑' if d > 0 else '↓'} {k}: {d:+.1f} (thr {thr:.0f})")
    for k in ["Risk Regime","Macro Regime","Bubble","Stealth QE","Section","Overall","GEX Regime"]:
        if k in prev and k in now and prev[k] != now[k]:
            alerts.append(f"⚡ STATE CHANGE → {k}: {prev[k]} ▶ {now[k]}")
    return alerts[:12]
