# pages/guide.py — render_guide + router
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

def render_probability_page():
    """Probability engine deep-dive: signal overview bar chart + rolling history."""
    st.markdown("## 📊 Probability Engine")
    st.markdown("Per-signal percentile ranks and rolling history for all leading indicators.")

    start = st.sidebar.date_input("Start", value=dt.date.today() - dt.timedelta(days=730), key="pb_start")
    end   = st.sidebar.date_input("End",   value=dt.date.today(), key="pb_end")
    idx   = pd.date_range(start, end, freq="D")

    with st.spinner("Loading data..."):
        raw = load_macro(start.isoformat(), end.isoformat())
        def r(k): return resample_ffill(raw.get(k, pd.Series(dtype=float)), idx)
        y2=r("DGS2"); y3m=r("DGS3MO"); y10=r("DGS10"); y30=r("DGS30")
        m2=r("M2SL"); walcl=r("WALCL"); tga=r("WTREGEN"); rrp=r("RRPONTSYD")
        copx=r("COPX"); gld=r("GLD"); hyg=r("HYG"); lqd=r("LQD")
        dxy=r("UUP"); spy=r("SPY"); vix=r("VIX"); qqq=r("QQQ"); iwm=r("IWM")
        claims=r("ICSA")
        tips_10y=r("DFII10"); bank_reserves=r("WRBWFRBL")
        bank_credit=r("TOTBKCR")
        ism_no_raw = raw.get("AMTMNO", pd.Series(dtype=float))
        ism_no = ism_no_raw if len(ism_no_raw.dropna()) > 4 else None
        gdp_quarterly=r("GDPC1")
        mmmf=r("WRMFSL")

    net_liq    = (walcl - tga - rrp) / 1000.0
    net_liq_4w = net_liq.diff(28)
    bs_13w     = walcl.diff(91) / 1000.0
    s_2s10s    = (y10 - y2) * 100

    leading = compute_leading_stack(
        y2, y3m, y10, y30, s_2s10s, vix, m2, claims,
        copx, gld, hyg, lqd, dxy, spy, qqq, iwm,
        net_liq, net_liq_4w, walcl, bs_13w, idx,
        tips_10y=tips_10y, bank_reserves=bank_reserves,
        bank_credit=bank_credit, ism_no=ism_no,
        gdp_quarterly=gdp_quarterly, mmmf=mmmf,
    )

    # Signal labels matching actual keys — organised by horizon bucket
    SIGNAL_LABELS = {
        # Tactical
        "vix_ts_pct":           "VIX Term Structure [5D]",
        "dxy_5d_pct":           "DXY 5D Momentum [5D]",
        # Short-term
        "hyg_lqd_pct":          "HYG/LQD Ratio [21D]",
        "smallcap_pct":         "Small-Cap Leadership [21D]",
        "liq_impulse_4w_pct":   "Net Liquidity 4W [21D]",
        "ism_no_pct":           "ISM New Orders [21D]",
        # Medium-term
        "curve_phase_pct":      "Curve Phase [63D]",
        "copper_gold_pct":      "Copper/Gold 13W [63D]",
        "credit_impulse_pct":   "Credit Impulse [63D]",
        "real_rate_pct":        "Real Rate Regime [63D]",
        "reserve_pct":          "Reserve Adequacy [63D]",
        "m2_yoy_pct":           "M2 YoY Growth [63D]",
        "liq_impulse_13w_pct":  "Net Liquidity 13W [63D]",
    }

    # Only show keys that have data (no silent 50.0 defaults)
    pcts = [(label, leading.get(key)) for key, label in SIGNAL_LABELS.items()]
    pcts = [(label, val) for label, val in pcts if val is not None]

    if not pcts:
        st.warning("No signal data available yet. Check data sources.")
        return

    # Horizon separator colours
    def _bar_colour(v):
        if v > 65:   return "#10b981"
        elif v > 55: return "#34d399"
        elif v > 45: return "#f59e0b"
        elif v > 35: return "#f97316"
        else:        return "#ef4444"

    colours = [_bar_colour(v) for _, v in pcts]

    fig_p = go.Figure(go.Bar(
        x=[v for _, v in pcts],
        y=[l for l, _ in pcts],
        orientation="h",
        marker_color=colours,
        text=[f"{v:.0f}th" for _, v in pcts],
        textposition="outside",
    ))
    fig_p.add_vline(x=50, line_dash="dot", line_color="rgba(255,255,255,0.25)")
    fig_p.add_vline(x=80, line_dash="dot", line_color="rgba(16,185,129,0.30)")
    fig_p.add_vline(x=20, line_dash="dot", line_color="rgba(239,68,68,0.30)")
    fig_p.update_layout(xaxis=dict(range=[0, 115]))
    st.plotly_chart(
        plotly_dark(fig_p, "Leading Signals — Historical Percentile (50=Neutral · 80=Top Quintile)", 520),
        use_container_width=True
    )

    # Additional info cards
    col_a, col_b, col_c = st.columns(3)
    rr = leading.get("real_rate_regime", "N/A")
    rl = leading.get("real_rate_level", float("nan"))
    col_a.metric("Real Rate Regime", rr,
                 f"{rl:.2f}%" if not (rl != rl) else "N/A")

    res_regime = leading.get("reserve_regime", "N/A")
    res_bn     = leading.get("reserve_level_bn", float("nan"))
    col_b.metric("Reserve Adequacy", res_regime,
                 f"${res_bn:,.0f}B" if not (res_bn != res_bn) else "N/A")

    ism_q = leading.get("ism_quadrant", "Unknown")
    ism_l = leading.get("ism_level", float("nan"))
    col_c.metric("ISM New Orders", ism_q,
                 f"{ism_l:.1f}" if not (ism_l != ism_l) else "N/A")

    st.markdown("---")

    # Rolling history for selected signal
    st.markdown("### Rolling Percentile History")
    spy_r = spy.reindex(idx).ffill()
    series_map = {
        "VIX Term Structure [5D]":    vix.reindex(idx).ffill() / (spy_r.pct_change().rolling(63,min_periods=20).std()*np.sqrt(252)*100).replace(0,np.nan),
        "DXY 5D Momentum [5D]":       -dxy.reindex(idx).ffill().pct_change(5)*100,
        "HYG/LQD Ratio [21D]":        hyg.reindex(idx).ffill()/lqd.reindex(idx).ffill().replace(0,np.nan),
        "Small-Cap Leadership [21D]":  iwm.reindex(idx).ffill()/spy_r.replace(0,np.nan),
        "Net Liquidity 4W [21D]":      net_liq_4w,
        "ISM New Orders [21D]":        resample_ffill(raw.get("AMTMNO", pd.Series(dtype=float)), idx),
        "Curve Phase [63D]":           s_2s10s,
        "Copper/Gold 13W [63D]":       copx.reindex(idx).ffill()/gld.reindex(idx).ffill().replace(0,np.nan),
        "Credit Impulse [63D]":        m2.diff(91)/m2.shift(91)*100,
        "Real Rate Regime [63D]":      r("DFII10"),
        "Reserve Adequacy [63D]":      r("WRBWFRBL"),
        "M2 YoY Growth [63D]":         (m2/m2.shift(365)-1)*100,
        "Net Liquidity 13W [63D]":     net_liq.diff(91),
    }

    available = [l for l, _ in pcts if l in series_map]
    if available:
        choice = st.selectbox("Signal", available)
        s_plot = series_map.get(choice, s_2s10s)
        pct_s  = rolling_pct(s_plot.dropna(), 252)

        fig1 = go.Figure(go.Scatter(x=s_plot.index, y=s_plot.values,
                                     mode="lines", line=dict(color="#3b82f6", width=1.5)))
        st.plotly_chart(plotly_dark(fig1, choice, 240), use_container_width=True)

        fig2 = go.Figure(go.Scatter(x=pct_s.index, y=pct_s.values,
                                     mode="lines", line=dict(color="#8b5cf6"),
                                     fill="tozeroy", fillcolor="rgba(139,92,246,0.08)"))
        fig2.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.2)")
        fig2.add_hline(y=80, line_dash="dot", line_color="rgba(16,185,129,0.3)")
        fig2.add_hline(y=20, line_dash="dot", line_color="rgba(239,68,68,0.3)")
        st.plotly_chart(
            plotly_dark(fig2, f"{choice} — Rolling 252D Percentile Rank", 200),
            use_container_width=True
        )

        st.caption(
            "Percentile rank uses trailing 252-day window. "
            "Current value only — not a full time-series construction. "
            "Each horizon bucket feeds the corresponding sub-model."
        )


def render_guide():
    st.markdown("## 📖 Guide — Unified Framework")
    st.markdown("""
### Architecture: Three Pillars

| Pillar | Answers | Source |
|---|---|---|
| **GEX (Gamma Exposure)** | WHERE — mechanical support/resistance from dealer hedging | yfinance options (or IBKR live) |
| **AMT / Macro** | WHY — regime context, whether auction accepts or rejects the level | FRED + yfinance price history |
| **Orderflow** | WHEN — real-time confirmation to enter | Footprint (external: Bookmap, Sierra Chart, MotiveWave) |

All three must align. Missing any one degrades the edge significantly.

---

### GEX Regime Guide

| Regime | Dist to Flip | Implication | Best Setups |
|---|---|---|---|
| Strong Positive (>+2%) | Far above | Dealers buy dips, sell rips. High mean-reversion. | Setup 1, 2 |
| Positive (+0.5–2%) | Above | Same but less extreme. | Setup 1, 2 |
| Neutral (±0.5%) | Near flip | Regime uncertain. Reduce size. | Watch only |
| Negative (−0.5 to −2%) | Below | Dealers amplify moves. Trend-following. | Setup 3 continuation |
| Strong Negative (<−2%) | Far below | Strong amplification. Extended moves. | Setup 3, 4 |

---

### Probability Engine

The Bull Probability is a **Bayesian blend** of:
- **Leading Stack (55% weight)**: 9 forward-looking signals converted to rolling percentile ranks
- **Lagging Stack (45% weight)**: Fear / Three Puts / Liq Anxiety / Exhaustion / Market Index
- **GEX Overlay**: Positive regime adds +5pt × stability; Negative subtracts −5pt × stability
- **Geo Drag**: Headline-weighted geopolitical risk score reduces bull probability
- **Regime Uncertainty**: High P(regime change) compresses probabilities toward 50%

---

### IBKR Integration

Connect to TWS or IB Gateway:
1. Start TWS/Gateway on your machine
2. Enable API: Configure → API → Settings → Enable ActiveX and Socket Clients
3. Add 127.0.0.1 to trusted IPs
4. Use the Schwab/TOS tab to authorise via OAuth2
5. Navigate to the IBKR tab in this dashboard

Features: Live options chain for GEX, Portfolio positions, Account P&L, Real-time spot price.

---

### The 8 Failure Modes (from Note 10)

| # | Mode | Key Signal | Response |
|---|---|---|---|
| 1 | Stale GEX Data | No mechanical response at level | Require 3 tests before trusting level |
| 2 | Gamma Crowding | Level in multiple public services | Reduce exp. hold % by 30-50% |
| 3 | Regime Transition | Approaching gamma flip | Monitor flip crossing; exit if reclaimed |
| 4 | Exogenous Shock | VIX spike, cross-asset correlation | Exit at market. GEX levels irrelevant. |
| 5 | Footprint Spoofing | Single large print, DOM then pulls | Require 3+ bars of confirmed absorption |
| 6 | Correlation Regime Break | ES following bonds/FX, not options | Downweight GEX, follow macro driver |
| 7 | OpEx Decay | Expiry removes gamma | Post-OpEx: treat levels as weaker |
| 8 | Hedging Timing Lag | Level is a zone, not a price | Use 4-8 tick buffer on all stops |

---

### Kelly Criterion (Educational)

At 55% win rate and 2.5:1 R:R:
```
f* = (0.55 × 2.5 − 0.45) / 2.5 = 0.37
Half-Kelly = 0.185 → ~18% of risk capital
```
Standard practice: use 25-50% of Kelly fraction to account for estimation error.
*Not investment advice.*
""")


# ============================================================
# ROUTER
# ============================================================
if page == "Dashboard":        render_dashboard()
elif page == "GEX Engine":     render_gex_engine()
elif page == "Trade Setups":   render_setups_page()
elif page == "Execution":      render_execution_page()
elif page == "Probability Engine": render_probability_page()
elif page == "Schwab/TOS":     render_schwab_page()
elif page == "Guide":          render_guide()
