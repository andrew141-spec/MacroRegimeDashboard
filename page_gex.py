# pages/gex.py — render_gex_engine + render_setups_page
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
from config import GammaState, GammaRegime, FeedItem, SetupScore, CSS
from utils import _to_1d, zscore, resample_ffill, yf_close, kelly, current_pct_rank
from config import _get_secret
from ui_components import pill, pbar, sec_hdr, plotly_dark, regime_chip, autorefresh_js
from gex_engine import build_gamma_state, compute_gex_from_chain
from schwab_api import get_schwab_client, schwab_get_spot, schwab_get_options_chain, schwab_run_auth_flow, schwab_complete_auth, _get_supabase, SCHWAB_AVAILABLE, SUPABASE_AVAILABLE
from data_loaders import load_macro, get_gex_from_yfinance, get_fwd_pe
from intel_monitor import load_feeds, geo_shock_score, score_relevance, categorise_items, category_shock_score, _all_feeds_flat, INTEL_CATEGORIES
from signals import compute_leading_stack, compute_1d_prob
from probability import compute_prob_composite, get_session_context, evaluate_setups, check_failure_modes, classify_macro_regime_abs, regime_transition_prob, driver_alerts

def render_gex_engine():
    """Deep-dive GEX analysis page."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## ⚡ GEX Engine")
    st.markdown("Live gamma exposure computation and regime analysis.")

    col_s, col_m = st.columns([1, 3])
    with col_s:
        symbol = st.text_input("Options Symbol", "SPY")
        use_schwab = st.toggle("Use Schwab/TOS (requires API credentials)", False)

    with col_m:
        if use_schwab:
            schwab_client = get_schwab_client()
            if schwab_client is not None:
                st.info("Schwab API connected · fetching live chain (real per-strike IV + OI)")
                with st.spinner(f"Fetching {symbol} options chain from Schwab…"):
                    spot_live = schwab_get_spot(schwab_client, symbol)
                    spot      = spot_live or spot or 580.0
                    st.caption(f"Spot: {spot:.2f}")

                cache_key     = f"schwab_chain_{symbol}_{round(spot / 5) * 5}"
                cached        = st.session_state.get("_schwab_chain_cache_key")
                force_refetch = st.session_state.pop("_schwab_force_refetch", False)

                if cached != cache_key or force_refetch:
                    chain_df = schwab_get_options_chain(schwab_client, symbol, spot)
                    chain_err = st.session_state.get("_schwab_chain_error")
                    if chain_err:
                        st.error(f"Chain error: {chain_err}")
                        chain_df, spot, source = get_gex_from_yfinance(symbol)
                    elif chain_df is not None:
                        st.session_state["_schwab_chain_df"]        = chain_df
                        st.session_state["_schwab_chain_cache_key"] = cache_key
                        st.success(f"✅ {len(chain_df)} strike/expiry rows · real IV + OI from Schwab")
                        source = "Schwab API"
                    else:
                        st.warning("Empty chain — falling back to yfinance")
                        chain_df, spot, source = get_gex_from_yfinance(symbol)
                else:
                    chain_df = st.session_state.get("_schwab_chain_df")
                    source   = "Schwab API (cached)"
                    st.caption("Using cached chain. Click ⟳ to force refresh.")

                if st.button("⟳ Force chain refresh", key="schwab_chain_refresh"):
                    st.session_state["_schwab_force_refetch"] = True
                    st.rerun()
            else:
                st.warning(
                    "Schwab not connected — go to the **Schwab/TOS** tab to authorise. "
                    "Falling back to yfinance."
                )
                chain_df, spot, source = get_gex_from_yfinance(symbol)
        else:
            chain_df, spot, source = get_gex_from_yfinance(symbol)

    if chain_df is None or len(chain_df) == 0:
        st.error("No options data available. Check symbol or data source.")
        return

    gs = build_gamma_state(chain_df, spot, source)
    gex_chain = compute_gex_from_chain(chain_df, spot)

    # Header metrics
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Spot", f"{spot:.2f}")
    m2.metric("Gamma Flip", f"{gs.gamma_flip:.2f}" if gs.gamma_flip else "N/A",
              f"{gs.distance_to_flip_pct:+.2f}%")
    m3.metric("Net GEX", f"{gs.total_gex/1e6:.1f}M")
    m4.metric("Regime Stability", f"{gs.regime_stability:.2f}")

    st.markdown(f"**Regime:** {regime_chip(gs.regime)}", unsafe_allow_html=True)
    st.markdown(f"**Source:** {source} · **As of:** {gs.timestamp}")

    # GEX chart
    if gs.gex_by_strike:
        strikes = sorted(gs.gex_by_strike.keys())
        near = [s for s in strikes if spot * 0.90 < s < spot * 1.10]
        if near:
            vals = [gs.gex_by_strike[s] for s in near]
            colors_gex = ["#10b981" if v > 0 else "#ef4444" for v in vals]
            fig_full = go.Figure(go.Bar(x=near, y=[v/1e6 for v in vals],
                                        marker_color=colors_gex, opacity=0.85, name="GEX ($M)"))
            if gs.gamma_flip:
                fig_full.add_vline(x=gs.gamma_flip, line_dash="dash", line_color="#f59e0b",
                                   annotation_text=f"FLIP {gs.gamma_flip:.0f}", annotation_font_size=11)
            fig_full.add_vline(x=spot, line_dash="dot", line_color="rgba(255,255,255,0.6)",
                               annotation_text="SPOT", annotation_font_size=11)
            # Mark support and resistance
            for s in gs.key_support[:3]:
                fig_full.add_vrect(x0=s-0.5, x1=s+0.5, fillcolor="rgba(16,185,129,0.20)", layer="below")
            for r in gs.key_resistance[:3]:
                fig_full.add_vrect(x0=r-0.5, x1=r+0.5, fillcolor="rgba(239,68,68,0.20)", layer="below")
            st.plotly_chart(plotly_dark(fig_full, "Net GEX by Strike ($M) — Green=Resistance/Damping · Red=Support/Amplifying", 380), use_container_width=True)

    # Key levels table
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🟢 GEX Resistance Walls (Dealers Sell)")
        if gs.key_resistance:
            res_data = [(r, gs.gex_by_strike.get(r, 0)) for r in gs.key_resistance]
            df_r = pd.DataFrame(res_data, columns=["Strike", "Net GEX ($)"]).sort_values("Strike")
            df_r["Net GEX ($M)"] = df_r["Net GEX ($)"] / 1e6
            df_r["Distance %"] = ((df_r["Strike"] - spot) / spot * 100).round(2)
            st.dataframe(df_r[["Strike","Net GEX ($M)","Distance %"]].round(2), hide_index=True)
    with c2:
        st.markdown("### 🔴 GEX Support Walls (Dealer Selling = Amplifying)")
        if gs.key_support:
            sup_data = [(s, gs.gex_by_strike.get(s, 0)) for s in gs.key_support]
            df_s = pd.DataFrame(sup_data, columns=["Strike", "Net GEX ($)"]).sort_values("Strike", ascending=False)
            df_s["Net GEX ($M)"] = df_s["Net GEX ($)"] / 1e6
            df_s["Distance %"] = ((df_s["Strike"] - spot) / spot * 100).round(2)
            st.dataframe(df_s[["Strike","Net GEX ($M)","Distance %"]].round(2), hide_index=True)

    st.markdown("---")
    st.markdown("""
**Reading the GEX Chart:**
- **Green bars (positive GEX)**: Dealers long gamma here. They BUY as price falls → mechanical support. They SELL as price rises → mechanical resistance. Mean-reversion is the expectation.
- **Red bars (negative GEX)**: Dealers short gamma here. They SELL as price falls → amplifying. They BUY as price rises → amplifying. Trend continuation is the expectation.
- **Gamma flip**: The price where net GEX crosses zero. Above = positive gamma regime (stabilizing). Below = negative gamma regime (destabilizing). The regime determines which trade setups apply.
- **Stability score**: How far spot is from the nearest regime boundary. Low stability = regime change imminent.
""")


def render_setups_page():
    st.markdown(CSS, unsafe_allow_html=True)
    """Trade setups reference page with live context."""
    st.markdown("## 🎯 Trade Setups — Live Context")
    st.markdown("Five operationally-defined setups from the GEX×Orderflow framework. Each requires: **WHERE** (GEX) + **WHY** (AMT) + **WHEN** (Orderflow).")

    chain_df, spot, source = get_gex_from_yfinance("SPY")
    gs = build_gamma_state(chain_df, spot, source) if chain_df is not None else GammaState()
    session = get_session_context()
    vix_df = yf.Ticker("^VIX").history(period="1d")
    vix_level = float(vix_df["Close"].iloc[-1]) if len(vix_df) > 0 else 20.0
    fear_est = float(np.clip((vix_level - 15) / 25 * 100, 0, 100))

    setups = evaluate_setups(gs, session, spot, fear_est, vix_level)

    # Session header
    sm = session["size_mult"]
    st.markdown(f"""<div class='warn-card'>
      <b>Session:</b> {session['window']} · <b>Liquidity:</b> {session['liquidity']} ·
      <b>Size Multiplier:</b> {sm:.2f}x ·
      <b>GEX Regime:</b> {gs.regime.value} · <b>Flip:</b> {gs.gamma_flip:.1f} · <b>Dist:</b> {gs.distance_to_flip_pct:+.2f}%
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    for s in setups:
        score = s["score"]
        active = s["active"]
        tradeable = score.tradeable
        c_border = "#10b981" if (active and tradeable) else ("#f59e0b" if active else "rgba(255,255,255,0.09)")
        bg = "rgba(16,185,129,0.06)" if (active and tradeable) else ("rgba(245,158,11,0.04)" if active else "rgba(255,255,255,0.02)")

        with st.expander(f"{'✅' if (active and tradeable) else '⚪'} Setup {s['setup']}: {s['name']}", expanded=active and tradeable):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Composite Score", f"{score.composite:.2f}", help="≥0.65 + both core ≥0.5 = tradeable")
                st.metric("Est. Hit Rate", f"{s['est_hit_rate']*100:.0f}%")
                st.metric("Effective Size", f"{s['effective_size']*100:.0f}% of standard",
                          help=f"Base {s['est_hit_rate']:.0f} × session {sm:.2f}x × vol adj")
            with col2:
                st.markdown("**Score Components**")
                for label, val in [
                    ("Gamma Alignment", score.gamma_alignment),
                    ("Orderflow Confirm", score.orderflow_confirmation),
                    ("TPO Context", score.tpo_context),
                    ("Level Freshness", score.level_freshness),
                    ("Event Risk", score.event_risk),
                ]:
                    c = "#10b981" if val >= 0.7 else ("#f59e0b" if val >= 0.5 else "#ef4444")
                    st.markdown(f"<div style='display:flex;justify-content:space-between;'>"
                                f"<span class='small'>{label}</span>"
                                f"<span style='font-family:var(--mono);font-size:11px;color:{c};'>{val:.2f}</span>"
                                f"</div>", unsafe_allow_html=True)
            with col3:
                st.markdown("**Current Status**")
                status = "🟢 ACTIVE" if (active and tradeable) else ("🟡 Watching" if active else "⚫ Not Active")
                st.markdown(f"**{status}**")
                st.markdown(f"<div class='small'>{s['desc']}</div>", unsafe_allow_html=True)
                if s["note"]:
                    st.markdown(f"<div class='warn-card' style='margin-top:6px;font-size:10px;'>{s['note']}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Position Sizing Table")
    sizing_data = {
        "Setup": ["1: Gamma Bounce", "2: Gamma Fade", "3: Flip Breakout", "4: Exhaustion Rev.", "5: 0DTE Pin"],
        "Base Size": ["100%", "100%", "75%", "50%", "100%"],
        "Session Mult": [f"{session['size_mult']:.2f}x"] * 5,
        "Vol Adj (VIX)": ["×0.75 if >25, ×0.50 if >35"] * 5,
        "Min R:R": ["2:1", "2:1", "2:1", "3:1", "1.3:1"],
        "Est. Hit Rate": ["~55%", "~52%", "~45%", "~40%", "~65%"],
        "Note": [
            "Core setup. Mechanical dealer buy.",
            "Slightly lower due to long bias.",
            "Binary — place stops above flip.",
            "Tightest stops. R:R must compensate.",
            "Valid only after 14:00 ET."
        ]
    }
    st.dataframe(pd.DataFrame(sizing_data), hide_index=True, use_container_width=True)

    st.markdown("### Kelly Fraction Reference")
    kelly_data = []
    for name, p, r_r in [("Setup 1 Bounce",0.55,2.5),("Setup 2 Fade",0.52,2.2),
                         ("Setup 3 Breakout",0.45,3.0),("Setup 4 Reversal",0.40,4.0),("Setup 5 Pin",0.65,1.5)]:
        f_star = (p * r_r - (1-p)) / r_r
        kelly_data.append({"Setup": name, "p(win)": f"{p:.0%}", "R:R": f"{r_r:.1f}",
                           "Full Kelly": f"{f_star:.0%}", "Half Kelly": f"{f_star/2:.0%}",
                           "Rec. Risk%": f"{min(f_star/2*100, 15):.0f}%"})
    st.dataframe(pd.DataFrame(kelly_data), hide_index=True, use_container_width=True)


