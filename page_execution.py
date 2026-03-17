# pages/execution.py — render_execution_page
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
from config import GammaState, GammaRegime, FeedItem, SetupScore
from utils import _to_1d, zscore, resample_ffill, yf_close, kelly, current_pct_rank
from config import _get_secret
from ui_components import pill, pbar, sec_hdr, plotly_dark, regime_chip, autorefresh_js
from gex_engine import build_gamma_state, compute_gex_from_chain
from schwab_api import get_schwab_client, schwab_get_spot, schwab_get_options_chain, schwab_run_auth_flow, schwab_complete_auth, _get_supabase, SCHWAB_AVAILABLE, SUPABASE_AVAILABLE
from data_loaders import load_macro, get_gex_from_yfinance, get_fwd_pe
from intel_monitor import load_feeds, geo_shock_score, score_relevance, categorise_items, category_shock_score, _all_feeds_flat, INTEL_CATEGORIES
from signals import compute_leading_stack, compute_1d_prob
from probability import compute_prob_composite, get_session_context, evaluate_setups, check_failure_modes, classify_macro_regime_abs, regime_transition_prob, driver_alerts

def render_execution_page():
    """Execution checklist and failure modes."""
    st.markdown("## 📋 Execution Framework")

    session = get_session_context()
    chain_df, spot, source = get_gex_from_yfinance("SPY")
    gs = build_gamma_state(chain_df, spot, source) if chain_df is not None else GammaState()
    vix_df = yf.Ticker("^VIX").history(period="1d")
    vix_level = float(vix_df["Close"].iloc[-1]) if len(vix_df) > 0 else 20.0

    # Pre-trade checklist (from note 07)
    st.markdown("### Pre-Trade Checklist")
    st.markdown("<div class='small'>Run this before every trade. All boxes must be checked or the trade does not exist.</div>", unsafe_allow_html=True)

    items = [
        ("Gamma level identified (current, not yesterday's)", gs.data_source != "unavailable"),
        ("Gamma regime confirmed above/below flip", gs.gamma_flip != 0),
        (f"Session context favorable (current: {session['window']}, mult: {session['size_mult']:.2f}x)",
         session["size_mult"] >= 0.5),
        ("Orderflow confirmation present (absorption/initiative/exhaustion on footprint)", None),  # Manual
        ("Setup maps to one of the 5 defined setups (no guesses)", None),
        ("Invalidation (stop) defined at structural level", None),
        (f"Risk within limits (VIX: {vix_level:.0f} → {'REDUCE 50%' if vix_level>30 else 'REDUCE 25%' if vix_level>22 else 'Normal'})",
         vix_level <= 22),
        ("R:R acceptable (≥2:1 for Setups 1-3, ≥3:1 for Setup 4, ≥1.3:1 for Setup 5)", None),
        (f"No disqualifying macro event within 2h (data day: {session['is_data_day']})",
         not session["is_data_day"]),
        (f"Not in opening 15min or post-15:30 (current: {session['window']})",
         session["window"] not in ("RTH Open", "Close/MOC")),
    ]

    for label, status in items:
        if status is True: icon, cls = "✅", "check-ok"
        elif status is False: icon, cls = "❌", "check-fail"
        else: icon, cls = "☐", "check-warn"
        st.markdown(f"<div class='check-row'><span class='{cls}'>{icon}</span><span>{label}</span></div>",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Failure Mode Detector")
    failure_modes = check_failure_modes(gs, session, vix_level, session["is_data_day"])
    n_triggered = sum(1 for fm in failure_modes if fm[3])
    if n_triggered == 0:
        st.success("No failure modes currently triggered. Framework conditions appear nominal.")
    else:
        st.warning(f"{n_triggered} failure mode(s) active — review before trading.")

    for fid, fname, fdesc, triggered in failure_modes:
        c_card = "alert-card" if triggered else "panel"
        icon = "⚠" if triggered else "✓"
        c_text = "#ef4444" if triggered else "var(--muted)"
        with st.expander(f"{icon} FM{fid}: {fname}", expanded=triggered):
            st.markdown(f"<div class='small'>{fdesc}</div>", unsafe_allow_html=True)
            st.markdown(f"**Status:** {'🔴 TRIGGERED' if triggered else '🟢 Clear'}")

    st.markdown("---")
    st.markdown("### Session Structure Guide")
    session_data = [
        ("Globex (overnight)", "Very Low", "0.0x", "Monitor only — no gamma setups"),
        ("RTH Open (09:30-09:45)", "Very High/Chaotic", "0.0x", "DO NOT TRADE — opening rotation"),
        ("IB Forming (09:45-10:30)", "High", "0.5x", "First cautious entries if strong confirmation"),
        ("Morning (10:30-12:00)", "High/Stable", "1.0x", "PRIME TIME — deploy all setups"),
        ("Midday (12:00-14:00)", "Reduced", "0.35x", "Reduce 50%+ — false signals common"),
        ("Afternoon (14:00-15:00)", "Increasing", "0.65x", "Reassess 0DTE, selective entries"),
        ("Close/MOC (15:00-16:00)", "Very High", "0.25x", "MOC overrides gamma — pin trades only"),
    ]
    df_sess = pd.DataFrame(session_data, columns=["Session","Liquidity","Size Mult","Action"])
    st.dataframe(df_sess, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### Gamma Regime → Setup Matrix")
    matrix_data = [
        ("Strong Positive (>2%)","Balanced/D-profile","1, 2 (bounce/fade)","Breakout trades"),
        ("Positive (0.5-2%)","IB near GEX levels","1, 2","Forcing breakouts"),
        ("Neutral (±0.5%)","Narrow IB","3 (watch for flip)","Full-size at extremes"),
        ("Negative (-0.5 to -2%)","Trending","3 continuation, 4 if exhaustion","Counter-trend fades"),
        ("Strong Negative (>-2%)","Extreme displacement ≥2ATR","4 (exhaustion reversal)","Any continuation"),
        ("High 0DTE, single strike","Late session, near pin","5 (gamma pin)","Directional bets"),
    ]
    df_matrix = pd.DataFrame(matrix_data, columns=["Gamma Regime","Market Structure","Applicable Setups","Avoid"])
    st.dataframe(df_matrix, hide_index=True, use_container_width=True)


