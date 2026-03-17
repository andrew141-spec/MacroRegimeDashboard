# page_gex.py — GEX Engine + Trade Setups with VEX/CEX/Key Nodes/Module 3 setups
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
from scipy.stats import norm as scipy_norm
from config import GammaState, GammaRegime, FeedItem, SetupScore, CSS
from utils import _to_1d, zscore, resample_ffill, yf_close, kelly, current_pct_rank
from config import _get_secret
from ui_components import pill, pbar, sec_hdr, plotly_dark, regime_chip, autorefresh_js, colored, gauge
from gex_engine import (build_gamma_state, compute_gex_from_chain, find_gamma_flip,
                        classify_gex_regime, compute_dealer_greeks, DealerGreeks)
from schwab_api import (get_schwab_client, schwab_get_spot, schwab_get_options_chain,
                        SCHWAB_AVAILABLE)
from data_loaders import get_gex_from_yfinance
from probability import get_session_context, evaluate_setups

# ── Colors ────────────────────────────────────────────────────────────────────
_C_POS   = "#10b981"   # green  — positive / bullish
_C_NEG   = "#ef4444"   # red    — negative / bearish
_C_FLIP  = "#f59e0b"   # yellow — flip zone
_C_VEX   = "#8b5cf6"   # purple — vanna
_C_CEX   = "#06b6d4"   # teal   — charm
_C_MUTED = "rgba(255,255,255,0.52)"

def _greek_bar_chart(by_strike: dict, spot: float, title: str,
                     pos_color: str, neg_color: str,
                     flip_level: float = None, height=340) -> go.Figure:
    strikes = sorted(by_strike.keys())
    near    = [s for s in strikes if spot * 0.90 < s < spot * 1.10]
    vals    = [by_strike[s] / 1e6 for s in near]
    colors  = [pos_color if v > 0 else neg_color for v in vals]
    fig = go.Figure(go.Bar(x=near, y=vals, marker_color=colors, opacity=0.85, name=title))
    fig.add_vline(x=spot, line_dash="dot", line_color="rgba(255,255,255,0.6)",
                  annotation_text="SPOT", annotation_font_size=10)
    if flip_level:
        fig.add_vline(x=flip_level, line_dash="dash", line_color=_C_FLIP,
                      annotation_text=f"FLIP {flip_level:.0f}", annotation_font_size=10)
    return plotly_dark(fig, title, height)


def _key_nodes_table(nodes: List[Tuple[float, float]], spot: float, label: str):
    if not nodes:
        st.caption("No key nodes computed.")
        return
    rows = []
    for strike, val in nodes:
        dist_pct = (strike - spot) / spot * 100
        direction = "Above" if strike > spot else "Below"
        rows.append({
            "Strike":    f"${strike:.1f}",
            "Exposure":  f"${val/1e6:.1f}M",
            "Abs Size":  f"${abs(val)/1e6:.1f}M",
            "Dist":      f"{dist_pct:+.2f}%",
            "Direction": direction,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)


def _module3_setups(dg: DealerGreeks, gs: GammaState, spot: float,
                    vix_level: float, session: dict) -> List[dict]:
    """
    Evaluate Module 3 setups from the heatmap framework.
    Returns list of dicts with: name, active, conditions_met, conditions_missing, action, note
    """
    setups = []
    regime = gs.regime
    dist   = gs.distance_to_flip_pct
    flip   = gs.gamma_flip

    # Helpers
    ntm_gex = sum(v for k, v in dg.gex_by_strike.items() if abs(k - spot) / spot < 0.02)
    ntm_vex = sum(v for k, v in dg.vex_by_strike.items() if abs(k - spot) / spot < 0.02)
    ntm_cex = sum(v for k, v in dg.cex_by_strike.items() if abs(k - spot) / spot < 0.02)

    # Largest nodes above and below
    above_nodes = [(k, v) for k, v in dg.gex_by_strike.items() if k > spot]
    below_nodes = [(k, v) for k, v in dg.gex_by_strike.items() if k < spot]
    largest_above = max(above_nodes, key=lambda x: abs(x[1])) if above_nodes else (spot, 0)
    largest_below = min(below_nodes, key=lambda x: abs(x[1])) if below_nodes else (spot, 0)

    # Consecutive same-sign strikes near spot
    near_strikes = sorted([k for k in dg.gex_by_strike if abs(k - spot) / spot < 0.05])
    pos_run = sum(1 for k in near_strikes if dg.gex_by_strike.get(k, 0) > 0)
    neg_run = sum(1 for k in near_strikes if dg.gex_by_strike.get(k, 0) < 0)

    is_late = session["window"] in ("Afternoon", "Close/MOC")
    is_prime = session["window"] == "Morning"

    # ── 1. Gamma Pin and Magnet Drift ─────────────────────────────────────
    cond1_met = []
    cond1_mis = []
    if regime in (GammaRegime.STRONG_POSITIVE, GammaRegime.POSITIVE):
        cond1_met.append("Positive gamma regime ✓")
    else:
        cond1_mis.append("Need positive gamma regime")
    if pos_run >= 3:
        cond1_met.append(f"{pos_run} consecutive positive strikes near spot ✓")
    else:
        cond1_mis.append(f"Need 3+ consecutive positive strikes (have {pos_run})")
    if abs(largest_above[1]) > 50e6 or abs(largest_below[1]) > 50e6:
        cond1_met.append("Large key nodes define range ✓")
    else:
        cond1_mis.append("Need large key nodes above/below ($50M+)")

    setups.append({
        "name": "Gamma Pin & Magnet Drift",
        "icon": "📌",
        "active": len(cond1_met) >= 2,
        "conditions_met": cond1_met,
        "conditions_missing": cond1_mis,
        "action": f"Fade moves away from key node ${dg.key_nodes_gex[0][0]:.0f} if present. Use outer positive nodes as range edges.",
        "note": "Avoid chasing breakouts unless outer node rapidly shrinks (magnet weakening).",
        "module": "Module 3 §1",
    })

    # ── 2. Node Break — Regime Shift ──────────────────────────────────────
    near_flip = abs(dist) < 1.5
    large_node_nearby = any(abs(k - spot) / spot < 0.015 and abs(v) > 100e6
                            for k, v in dg.gex_by_strike.items())
    cond2_met = []
    cond2_mis = []
    if near_flip:
        cond2_met.append(f"Near gamma flip ({dist:+.2f}%) ✓")
    else:
        cond2_mis.append(f"Need price closer to flip (currently {dist:+.2f}%)")
    if large_node_nearby:
        cond2_met.append("Large node within 1.5% of spot ✓")
    else:
        cond2_mis.append("Need large node ($100M+) within 1.5% of spot")
    if regime in (GammaRegime.POSITIVE, GammaRegime.NEUTRAL):
        cond2_met.append("Regime can transition ✓")
    else:
        cond2_mis.append("Best when transitioning from positive → neutral/negative")

    setups.append({
        "name": "Node Break — Regime Shift",
        "icon": "💥",
        "active": len(cond2_met) >= 2,
        "conditions_met": cond2_met,
        "conditions_missing": cond2_mis,
        "action": "Watch for failed rejection + fast decrease in node value. Enter direction of break, target next largest node.",
        "note": "Vol expands the moment the former wall is gone. Expect vacuum to next node.",
        "module": "Module 3 §2",
    })

    # ── 3. Negative Gamma Amplification ───────────────────────────────────
    cond3_met = []
    cond3_mis = []
    if regime in (GammaRegime.NEGATIVE, GammaRegime.STRONG_NEGATIVE):
        cond3_met.append("Negative gamma regime ✓")
    else:
        cond3_mis.append("Need negative gamma regime")
    if neg_run >= 3:
        cond3_met.append(f"{neg_run} consecutive negative strikes near spot ✓")
    else:
        cond3_mis.append(f"Need 3+ consecutive negative strikes (have {neg_run})")
    if ntm_gex < -100e6:
        cond3_met.append("Strong negative GEX cluster near spot ✓")
    else:
        cond3_mis.append("Need strong negative GEX cluster ($-100M+)")

    setups.append({
        "name": "Negative Gamma Amplification",
        "icon": "🌊",
        "active": len(cond3_met) >= 2,
        "conditions_met": cond3_met,
        "conditions_missing": cond3_mis,
        "action": "Trade breakouts away from flip zone. Down moves → delta selling. Up moves → delta buying. Follow the direction.",
        "note": "Gap-and-go and sharp moves expected. Do NOT fade — dealers are chasing.",
        "module": "Module 3 §3",
    })

    # ── 4. Vanna Squeeze ──────────────────────────────────────────────────
    pos_vanna_ntm = ntm_vex > 50e6
    iv_elevated   = vix_level > 22
    cond4_met = []
    cond4_mis = []
    if pos_vanna_ntm:
        cond4_met.append(f"Positive vanna near spot (${ntm_vex/1e6:.0f}M) ✓")
    else:
        cond4_mis.append("Need positive vanna band near spot ($50M+)")
    if iv_elevated:
        cond4_met.append(f"IV elevated (VIX {vix_level:.1f}) — compression potential ✓")
    else:
        cond4_mis.append(f"IV not elevated enough (VIX {vix_level:.1f}, need >22)")
    if regime in (GammaRegime.POSITIVE, GammaRegime.NEUTRAL):
        cond4_met.append("Positive gamma supportive ✓")
    else:
        cond4_mis.append("Positive/neutral gamma regime preferred")

    setups.append({
        "name": "Vanna Squeeze",
        "icon": "🚀",
        "active": len(cond4_met) >= 2,
        "conditions_met": cond4_met,
        "conditions_missing": cond4_mis,
        "action": "Look for consolidation with positive vanna band. Once IV compresses, look for upside break targeting next largest node.",
        "note": "Rising spot + falling IV both reduce put deltas — dealers amplify upside. Post-event (earnings/FOMC) ideal.",
        "module": "Module 3 §4",
    })

    # ── 5. Vanna Rug / Vol Shock ──────────────────────────────────────────
    neg_vanna_above = sum(v for k, v in dg.vex_by_strike.items() if k > spot and v < 0)
    iv_compressed   = vix_level < 18
    cond5_met = []
    cond5_mis = []
    if neg_vanna_above < -50e6:
        cond5_met.append(f"Strong negative vanna above spot (${neg_vanna_above/1e6:.0f}M) ✓")
    else:
        cond5_mis.append("Need negative vanna ceiling above spot ($-50M+)")
    if iv_compressed:
        cond5_met.append(f"IV compressed (VIX {vix_level:.1f}) — shock potential ✓")
    else:
        cond5_mis.append(f"IV not compressed (VIX {vix_level:.1f}, need <18 for maximum effect)")

    setups.append({
        "name": "Vanna Rug / Vol Shock",
        "icon": "🧨",
        "active": len(cond5_met) >= 2,
        "conditions_met": cond5_met,
        "conditions_missing": cond5_mis,
        "action": "Watch for catalysts (earnings, macro). IV spike forces dealers short options to sell underlying — amplifies downside. Short bias.",
        "note": "Vanna ceiling at ${:.0f}. IV jump reverses dealer delta hedging direction aggressively.".format(
            max((k for k, v in dg.vex_by_strike.items() if k > spot and v < 0), default=spot)),
        "module": "Module 3 §5",
    })

    # ── 6. Charm Drift (EOD) ──────────────────────────────────────────────
    strong_charm = abs(ntm_cex) > 20e6
    cond6_met = []
    cond6_mis = []
    if strong_charm:
        direction_str = "bullish (dealers buy)" if ntm_cex > 0 else "bearish (dealers sell)"
        cond6_met.append(f"Strong charm near spot (${ntm_cex/1e6:.0f}M — {direction_str}) ✓")
    else:
        cond6_mis.append("Need strong charm near spot ($20M+ abs)")
    if is_late:
        cond6_met.append("Late session — charm effect strongest ✓")
    else:
        cond6_mis.append("Charm drift strongest afternoon/EOD (currently earlier session)")

    setups.append({
        "name": "Charm Drift (EOD)",
        "icon": "⏰",
        "active": len(cond6_met) >= 2,
        "conditions_met": cond6_met,
        "conditions_missing": cond6_mis,
        "action": f"Favor {'long' if ntm_cex > 0 else 'short'} bias into EOD. Charm forces dealers to {'buy' if ntm_cex > 0 else 'sell'} as time passes — strongest 14:00–16:00 ET.",
        "note": "Even if price/IV unchanged, time decay shifts dealer deltas. Use GEX magnets as targets.",
        "module": "Module 3 §6",
    })

    # ── 7. Charm-Vanna Alignment ──────────────────────────────────────────
    cond7_met = []
    cond7_mis = []
    if dg.vanna_charm_aligned:
        cond7_met.append(f"Vanna and Charm both {dg.vanna_direction} ✓")
        cond7_met.append("High-odds drift window ✓")
    else:
        cond7_mis.append(f"Vanna ({dg.vanna_direction}) and Charm ({dg.charm_direction}) not aligned")
    if abs(ntm_vex) > 30e6 and abs(ntm_cex) > 15e6:
        cond7_met.append("Both exposures have meaningful magnitude ✓")
    else:
        cond7_mis.append("Need both VEX ($30M+) and CEX ($15M+) near spot")

    setups.append({
        "name": "Charm-Vanna Alignment",
        "icon": "⚡",
        "active": dg.vanna_charm_aligned and len(cond7_met) >= 2,
        "conditions_met": cond7_met,
        "conditions_missing": cond7_mis,
        "action": f"Both second-order flows lean {dg.vanna_direction}. Favor directional swing with the flow. Use GEX magnets as take-profit levels.",
        "note": "Treat as high-odds drift window. Time decay + IV compression both push the same direction.",
        "module": "Module 3 §7",
    })

    # ── 8. Cross-Expiry Overlap (Macro Magnet) ────────────────────────────
    # Find strikes with large exposure in multiple expirations
    multi_exp = {}
    for _, row in (compute_gex_from_chain(
                        pd.DataFrame({"strike": [], "expiry_T": [], "iv": [], "call_oi": [], "put_oi": []}),
                        spot) if False else pd.DataFrame()).iterrows():
        pass  # placeholder — detect from chain
    # Use key node overlap as proxy
    gex_nodes  = {k for k, v in dg.key_nodes_gex}
    vex_nodes  = {k for k, v in dg.key_nodes_vex}
    cex_nodes  = {k for k, v in dg.key_nodes_cex}
    stacked    = gex_nodes & (vex_nodes | cex_nodes)

    cond8_met = []
    cond8_mis = []
    if stacked:
        cond8_met.append(f"Stacked nodes at: {', '.join(f'${k:.0f}' for k in sorted(stacked))} ✓")
    else:
        cond8_mis.append("No strike appears as key node in multiple Greeks")

    setups.append({
        "name": "Cross-Expiry / Stacked Node (Macro Magnet)",
        "icon": "🧲",
        "active": bool(stacked),
        "conditions_met": cond8_met,
        "conditions_missing": cond8_mis,
        "action": f"Treat stacked strikes as macro magnets: {', '.join(f'${k:.0f}' for k in sorted(stacked)) or 'none detected'}. Expect strong reactions on first touch.",
        "note": "Real regime change if they decisively break. Multiple dealer books adjusting simultaneously.",
        "module": "Module 3 §9",
    })

    return setups


def render_gex_engine():
    """Deep-dive GEX analysis page with GEX / VEX / CEX tabs."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## ⚡ GEX Engine — Dealer Greeks")

    col_s, col_m = st.columns([1, 3])
    with col_s:
        symbol    = st.text_input("Options Symbol", "SPY", key="gex_symbol_input")
        use_schwab = st.toggle("Use Schwab/TOS (live IV)", False, key="gex_use_schwab")

    # ── Data fetch ────────────────────────────────────────────────────────
    chain_df, spot, source = None, 580.0, "unknown"
    with col_m:
        if use_schwab:
            client = get_schwab_client()
            if client:
                st.info("Schwab connected — fetching live chain")
                spot_live = schwab_get_spot(client, symbol)
                spot      = spot_live or spot
                chain_df  = schwab_get_options_chain(client, symbol, spot)
                source    = "Schwab API (live IV)"
                if chain_df is None:
                    st.warning("Schwab chain empty — falling back to yfinance")
                    chain_df, spot, source = get_gex_from_yfinance(symbol)
            else:
                st.warning("Schwab not connected — using yfinance. Go to **Schwab/TOS** tab to authorise.")
                chain_df, spot, source = get_gex_from_yfinance(symbol)
        else:
            chain_df, spot, source = get_gex_from_yfinance(symbol)

    if chain_df is None or len(chain_df) == 0:
        st.error(f"No options data available. Source returned: `{source}`")
        with st.expander("🔧 Debug Info", expanded=True):
            st.write(f"**Symbol:** {symbol}")
            st.write(f"**chain_df:** {'None' if chain_df is None else f'Empty DataFrame ({len(chain_df)} rows)'}")
            st.write(f"**spot:** {spot}")
            st.write(f"**source:** {source}")
            try:
                import yfinance as _yf
                _t = _yf.Ticker(symbol)
                _exps = _t.options
                st.write(f"**ticker.options:** {_exps[:5] if _exps else 'EMPTY LIST'}")
                _h = _t.history(period="5d")
                st.write("**ticker.history (last 2 rows):**")
                st.dataframe(_h.tail(2))
                _fi = _t.fast_info
                st.write(f"**fast_info.last_price:** {_fi.last_price}")
            except Exception as _e:
                st.write(f"**Direct yfinance error:** {type(_e).__name__}: {_e}")
        return

    # ── Compute all Greeks ────────────────────────────────────────────────
    gs  = build_gamma_state(chain_df, spot, source)
    dg  = compute_dealer_greeks(chain_df, spot, source)
    session   = get_session_context()
    vix_df    = yf.Ticker("^VIX").history(period="1d")
    vix_level = float(vix_df["Close"].iloc[-1]) if len(vix_df) > 0 else 20.0

    # ── Header ────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Spot",          f"{spot:.2f}")
    m2.metric("Gamma Flip",    f"{gs.gamma_flip:.2f}" if gs.gamma_flip else "N/A",
                               f"{gs.distance_to_flip_pct:+.2f}%")
    m3.metric("Net GEX",       f"${gs.total_gex/1e6:.1f}M")
    m4.metric("Vanna Dir",     dg.vanna_direction.upper())
    m5.metric("Charm Dir",     dg.charm_direction.upper())

    st.markdown(f"**Regime:** {regime_chip(gs.regime)} &nbsp; **Source:** {source} &nbsp; **As of:** {gs.timestamp}",
                unsafe_allow_html=True)

    va_c = _C_POS if dg.vanna_charm_aligned else _C_MUTED
    if dg.vanna_charm_aligned:
        st.markdown(f"""<div class='warn-card' style='background:rgba(16,185,129,0.07);border-color:rgba(16,185,129,0.30);margin:6px 0;'>
          ⚡ <b>CHARM-VANNA ALIGNED</b> — Both pointing <b style='color:{_C_POS};'>{dg.vanna_direction.upper()}</b> · High-odds drift window active
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Tabs: GEX / VEX / CEX / Key Nodes / OTM Anchors ─────────────────
    tab_gex, tab_vex, tab_cex, tab_nodes, tab_otm = st.tabs(
        ["📊 GEX (Gamma)", "🌀 VEX (Vanna)", "⏱ CEX (Charm)", "🎯 Key Nodes", "🔭 OTM Anchors"]
    )

    with tab_gex:
        st.markdown(f"{sec_hdr('GAMMA EXPOSURE — Reaction to Price')}", unsafe_allow_html=True)
        st.caption("Green = positive gamma (dealers stabilize). Red = negative gamma (dealers amplify). Yellow line = gamma flip.")
        fig_gex = _greek_bar_chart(dg.gex_by_strike, spot,
                                   "Net GEX by Strike ($M)", _C_POS, _C_NEG, gs.gamma_flip)
        st.plotly_chart(fig_gex, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🟢 GEX Resistance Walls** (Dealers Sell Into Rallies)")
            res_data = [(r, dg.gex_by_strike.get(r, 0)) for r, _ in
                        sorted([(k, v) for k, v in dg.gex_by_strike.items() if v > 0 and k > spot],
                               key=lambda x: -x[1])[:5]]
            if res_data:
                df_r = pd.DataFrame(res_data, columns=["Strike", "GEX ($)"])
                df_r["GEX ($M)"]   = (df_r["GEX ($)"] / 1e6).round(1)
                df_r["Dist %"]     = ((df_r["Strike"] - spot) / spot * 100).round(2)
                st.dataframe(df_r[["Strike","GEX ($M)","Dist %"]], hide_index=True)
        with c2:
            st.markdown("**🔴 GEX Support Walls** (Dealers Amplify Falls)")
            sup_data = [(s, dg.gex_by_strike.get(s, 0)) for s, _ in
                        sorted([(k, v) for k, v in dg.gex_by_strike.items() if v < 0 and k < spot],
                               key=lambda x: x[1])[:5]]
            if sup_data:
                df_s = pd.DataFrame(sup_data, columns=["Strike", "GEX ($)"])
                df_s["GEX ($M)"]   = (df_s["GEX ($)"] / 1e6).round(1)
                df_s["Dist %"]     = ((df_s["Strike"] - spot) / spot * 100).round(2)
                st.dataframe(df_s[["Strike","GEX ($M)","Dist %"]], hide_index=True)

        st.markdown("""
**GEX = Reaction to Price**
- **Green (positive)**: Long gamma dealers buy dips, sell rips → mean reversion, range compression
- **Red (negative)**: Short gamma dealers amplify → trend days, gap-and-go, cascades
- **Flip zone**: Where regime transitions from stabilizing to destabilizing
""")

    with tab_vex:
        st.markdown(f"{sec_hdr('VANNA EXPOSURE — Reaction to IV Changes')}", unsafe_allow_html=True)
        st.caption("Positive vanna: falling IV → dealers buy underlying (bullish). Negative vanna: falling IV → dealers sell (bearish).")
        if "source" not in source.lower() or True:
            if "yfinance" in source.lower():
                st.info("📋 VEX computed from EOD IV. With Schwab/TOS connected, this uses live per-strike IV for real-time vanna.")
        fig_vex = _greek_bar_chart(dg.vex_by_strike, spot,
                                   "Net VEX by Strike ($M)", _C_VEX, _C_NEG, gs.gamma_flip)
        st.plotly_chart(fig_vex, use_container_width=True)

        v1, v2 = st.columns(2)
        with v1:
            st.markdown(f"**Net Vanna near spot:** ${sum(v for k,v in dg.vex_by_strike.items() if abs(k-spot)/spot < 0.02)/1e6:.1f}M")
            st.markdown(f"**Direction:** {dg.vanna_direction.upper()}")
        with v2:
            st.markdown("**Interpretation:**")
            if dg.vanna_direction == "bullish":
                st.success("Positive vanna: If IV compresses → dealers forced to BUY. Bullish drift on calm tape.")
            elif dg.vanna_direction == "bearish":
                st.error("Negative vanna: If IV compresses → dealers forced to SELL. Bearish drift on calm tape.")
            else:
                st.info("Neutral vanna — no strong IV-driven directional pressure.")

        st.markdown("""
**VEX = Reaction to Volatility**
- **Positive (purple)**: IV falling → dealers buy underlying → upward pressure
- **Negative (red)**: IV falling → dealers sell underlying → downward pressure
- **Vanna Squeeze**: High positive vanna + IV likely to compress = powerful rally setup
- **Vanna Rug**: Negative vanna ceiling + IV spike = amplified selloff
""")

    with tab_cex:
        st.markdown(f"{sec_hdr('CHARM EXPOSURE — Reaction to Time Decay')}", unsafe_allow_html=True)
        st.caption("Positive charm: time passing → dealers buy underlying (upward drift). Negative charm: dealers sell (downward drift).")
        if "yfinance" in source.lower():
            st.info("📋 CEX computed from EOD IV/OI. With Schwab/TOS, charm reflects live 0DTE positioning for intraday drift signals.")
        fig_cex = _greek_bar_chart(dg.cex_by_strike, spot,
                                   "Net CEX by Strike ($M)", _C_CEX, _C_NEG, gs.gamma_flip)
        st.plotly_chart(fig_cex, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Net Charm near spot:** ${sum(v for k,v in dg.cex_by_strike.items() if abs(k-spot)/spot < 0.02)/1e6:.1f}M")
            st.markdown(f"**Direction:** {dg.charm_direction.upper()}")
        with c2:
            st.markdown("**Interpretation:**")
            if dg.charm_direction == "bullish":
                st.success("Positive charm: As 0DTE expires, dealers forced to BUY. EOD upward drift expected.")
            elif dg.charm_direction == "bearish":
                st.error("Negative charm: As 0DTE expires, dealers forced to SELL. EOD downward drift expected.")
            else:
                st.info("Neutral charm — no strong time-decay-driven directional pressure.")

        st.markdown("""
**CEX = Reaction to Time**
- **Positive (teal)**: Time passing increases dealer delta → they must buy → upward drift into EOD
- **Negative (red)**: Time passing decreases dealer delta → they must sell → downward drift into EOD
- **Strongest effect**: 0DTE options, afternoon session (14:00–16:00 ET)
- **Charm Drift**: Even with flat price and IV, charm creates directional drift
""")

    with tab_nodes:
        st.markdown(f"{sec_hdr('KEY NODES — Largest Absolute Exposure')}", unsafe_allow_html=True)
        st.caption("Per Module 2: the most important factor is ABSOLUTE VALUE, not color. Larger node = more dealer flow = stronger magnet.")

        n1, n2, n3 = st.columns(3)
        with n1:
            st.markdown("**🎯 GEX Key Nodes**")
            _key_nodes_table(dg.key_nodes_gex, spot, "GEX")
        with n2:
            st.markdown("**🌀 VEX Key Nodes**")
            _key_nodes_table(dg.key_nodes_vex, spot, "VEX")
        with n3:
            st.markdown("**⏱ CEX Key Nodes**")
            _key_nodes_table(dg.key_nodes_cex, spot, "CEX")

        # Stacked nodes
        gex_set = {k for k, v in dg.key_nodes_gex}
        vex_set = {k for k, v in dg.key_nodes_vex}
        cex_set = {k for k, v in dg.key_nodes_cex}
        stacked = gex_set & (vex_set | cex_set)

        if stacked:
            st.markdown(f"""<div class='warn-card' style='background:rgba(139,92,246,0.08);border-color:rgba(139,92,246,0.30);margin-top:10px;'>
              🧲 <b>STACKED / MACRO MAGNET NODES:</b> {', '.join(f'<b>${k:.0f}</b>' for k in sorted(stacked))}
              <span class='small'> — appear as key node in multiple Greeks. Expect strong reactions. Real regime change if decisively broken.</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
**Key Node Principles (Module 2):**
- **Key Node Rejection**: Price pushed away early in NY AM session — use as bounce/fade entry
- **Intraday Pin**: Price pinned between key node and flip zone into EOD — expect tight range
- **Concept of Magnets**: Closer price gets to high-value node → stronger pull. Watch for node shrinking as price approaches (wall crumbling)
- **Node Decrease**: After rejection, node often loses value → probability of re-test weakens. Next node becomes target.
""")

    with tab_otm:
        st.markdown(f"{sec_hdr('OTM ANCHORS — Weekly Bias & Directional Targets')}", unsafe_allow_html=True)
        st.caption("Large OTM nodes represent structural positions. Growing downside node = bearish weekly bias. Vanishing upside node = downside confirmation.")

        if dg.otm_anchors:
            rows = []
            for strike, gex_val in dg.otm_anchors:
                dist_pct = (strike - spot) / spot * 100
                side = "Above (resistance)" if strike > spot else "Below (support)"
                bias = "Bullish target" if strike > spot and gex_val > 0 else \
                       "Bearish target" if strike < spot and gex_val < 0 else \
                       "Breakout level" if strike > spot and gex_val < 0 else "Reversal zone"
                rows.append({
                    "Strike":    f"${strike:.1f}",
                    "GEX ($M)":  f"${gex_val/1e6:.1f}M",
                    "Abs Size":  f"${abs(gex_val)/1e6:.1f}M",
                    "Dist":      f"{dist_pct:+.2f}%",
                    "Side":      side,
                    "Weekly Bias": bias,
                })
            df_otm = pd.DataFrame(rows)
            st.dataframe(df_otm, hide_index=True, use_container_width=True)

            # Simple weekly bias summary
            otm_above = [(k, v) for k, v in dg.otm_anchors if k > spot]
            otm_below = [(k, v) for k, v in dg.otm_anchors if k < spot]
            above_gex  = sum(v for _, v in otm_above)
            below_gex  = sum(v for _, v in otm_below)

            if above_gex > abs(below_gex) * 1.3:
                bias_str = "🟢 BULLISH WEEKLY BIAS — Larger positive OTM nodes above. Upside anchors dominant."
                bias_color = "rgba(16,185,129,0.08)"
                bias_border = "rgba(16,185,129,0.30)"
            elif abs(below_gex) > above_gex * 1.3:
                bias_str = "🔴 BEARISH WEEKLY BIAS — Larger negative OTM nodes below. Downside anchors dominant."
                bias_color = "rgba(239,68,68,0.08)"
                bias_border = "rgba(239,68,68,0.30)"
            else:
                bias_str = "⚪ NEUTRAL WEEKLY BIAS — OTM nodes roughly balanced."
                bias_color = "rgba(255,255,255,0.04)"
                bias_border = "rgba(255,255,255,0.12)"

            st.markdown(f"""<div class='panel' style='background:{bias_color};border-color:{bias_border};margin-top:10px;'>
              {bias_str}
            </div>""", unsafe_allow_html=True)
        else:
            st.info("No significant OTM anchors detected beyond 3% from spot.")

        st.markdown("""
**OTM Anchor Principles (Module 2 §4):**
- **Growing downside node** = bearish swing bias confirmed
- **Vanishing upside node** = upside wall crumbling = downside confirmation
- **Large persistent OTM node** = likely weekly termination target for swings
- **Sudden disappearance** of large OTM node = higher-timeframe shift in dealer exposure → reassess bias
""")


def render_setups_page():
    """Trade setups — original 5 setups + Module 3 dealer flow setups."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## 🎯 Trade Setups — Live Context")

    chain_df, spot, source = get_gex_from_yfinance("SPY")
    gs = build_gamma_state(chain_df, spot, source) if chain_df is not None else GammaState()
    dg = compute_dealer_greeks(chain_df, spot, source) if chain_df is not None else DealerGreeks()
    session   = get_session_context()
    vix_df    = yf.Ticker("^VIX").history(period="1d")
    vix_level = float(vix_df["Close"].iloc[-1]) if len(vix_df) > 0 else 20.0
    fear_est  = float(np.clip((vix_level - 15) / 25 * 100, 0, 100))

    setups_orig = evaluate_setups(gs, session, spot, fear_est, vix_level)
    setups_m3   = _module3_setups(dg, gs, spot, vix_level, session)

    # Session header
    sm = session["size_mult"]
    st.markdown(f"""<div class='warn-card'>
      <b>Session:</b> {session['window']} · <b>Liquidity:</b> {session['liquidity']} ·
      <b>Size Mult:</b> {sm:.2f}x · <b>Regime:</b> {gs.regime.value} ·
      <b>Flip:</b> {gs.gamma_flip:.1f if gs.gamma_flip else 'N/A'} ·
      <b>Vanna:</b> {dg.vanna_direction.upper()} · <b>Charm:</b> {dg.charm_direction.upper()}
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    tab_orig, tab_m3, tab_sizing = st.tabs(
        ["📐 Core GEX Setups (5)", "🌊 Dealer Flow Setups (Module 3)", "📏 Sizing Reference"]
    )

    # ── Tab 1: Original 5 setups ──────────────────────────────────────────
    with tab_orig:
        st.markdown(f"{sec_hdr('CORE GEX × ORDERFLOW SETUPS')}", unsafe_allow_html=True)
        st.caption("WHERE (GEX) + WHY (AMT) + WHEN (Orderflow). All three required.")

        for s in setups_orig:
            score    = s["score"]
            active   = s["active"]
            tradeable = score.tradeable
            with st.expander(
                f"{'✅' if (active and tradeable) else ('🟡' if active else '⚫')} "
                f"Setup {s['setup']}: {s['name']}",
                expanded=active and tradeable
            ):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Composite Score", f"{score.composite:.2f}")
                    st.metric("Est. Hit Rate",   f"{s['est_hit_rate']*100:.0f}%")
                    st.metric("Effective Size",  f"{s['effective_size']*100:.0f}%")
                with col2:
                    st.markdown("**Score Components**")
                    for label, val in [
                        ("Gamma Alignment",    score.gamma_alignment),
                        ("Orderflow Confirm",  score.orderflow_confirmation),
                        ("TPO Context",        score.tpo_context),
                        ("Level Freshness",    score.level_freshness),
                        ("Event Risk",         score.event_risk),
                    ]:
                        c = _C_POS if val >= 0.7 else (_C_FLIP if val >= 0.5 else _C_NEG)
                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;'>"
                            f"<span class='small'>{label}</span>"
                            f"<span style='font-family:var(--mono);font-size:11px;color:{c};'>{val:.2f}</span>"
                            f"</div>", unsafe_allow_html=True)
                with col3:
                    status = "🟢 ACTIVE" if (active and tradeable) else ("🟡 Watching" if active else "⚫ Not Active")
                    st.markdown(f"**{status}**")
                    st.markdown(f"<div class='small'>{s['desc']}</div>", unsafe_allow_html=True)
                    if s["note"]:
                        st.markdown(f"<div class='warn-card' style='margin-top:6px;font-size:10px;'>{s['note']}</div>",
                                    unsafe_allow_html=True)

    # ── Tab 2: Module 3 Dealer Flow Setups ───────────────────────────────
    with tab_m3:
        st.markdown(f"{sec_hdr('DEALER FLOW SETUPS — MODULE 3')}", unsafe_allow_html=True)
        st.caption("Based on GEX + VEX + CEX heatmap framework. Active alert = conditions met. Missing = what to watch for.")

        active_count = sum(1 for s in setups_m3 if s["active"])
        if active_count > 0:
            st.markdown(f"""<div class='warn-card' style='background:rgba(16,185,129,0.07);border-color:rgba(16,185,129,0.30);margin-bottom:10px;'>
              ✅ <b>{active_count} dealer flow setup{'s' if active_count > 1 else ''} currently active</b>
            </div>""", unsafe_allow_html=True)

        for s in setups_m3:
            active = s["active"]
            with st.expander(
                f"{s['icon']} {'✅ ACTIVE' if active else '⚪'} — {s['name']} ({s['module']})",
                expanded=active
            ):
                col1, col2 = st.columns([1, 1])
                with col1:
                    if s["conditions_met"]:
                        st.markdown("**✅ Conditions Met:**")
                        for c in s["conditions_met"]:
                            st.markdown(f"<div style='font-size:11px;color:{_C_POS};'>✓ {c}</div>",
                                        unsafe_allow_html=True)
                    if s["conditions_missing"]:
                        st.markdown("**⏳ Still Watching For:**")
                        for c in s["conditions_missing"]:
                            st.markdown(f"<div style='font-size:11px;color:{_C_MUTED};'>○ {c}</div>",
                                        unsafe_allow_html=True)
                with col2:
                    st.markdown("**📋 Action:**")
                    st.markdown(f"<div class='small' style='color:var(--text);'>{s['action']}</div>",
                                unsafe_allow_html=True)
                    if s["note"]:
                        st.markdown(f"<div class='warn-card' style='margin-top:8px;font-size:10px;'>{s['note']}</div>",
                                    unsafe_allow_html=True)

    # ── Tab 3: Sizing reference ───────────────────────────────────────────
    with tab_sizing:
        st.markdown("### Position Sizing — Core Setups")
        sizing_data = {
            "Setup":         ["1: Gamma Bounce", "2: Gamma Fade", "3: Flip Breakout",
                               "4: Exhaustion Rev.", "5: 0DTE Pin"],
            "Base Size":     ["100%","100%","75%","50%","100%"],
            "Session Mult":  [f"{sm:.2f}x"] * 5,
            "Vol Adj":       ["×0.5 if VIX>35, ×0.75 if VIX>25"] * 5,
            "Min R:R":       ["2:1","2:1","2:1","3:1","1.3:1"],
            "Est. Hit Rate": ["~55%","~52%","~45%","~40%","~65%"],
        }
        st.dataframe(pd.DataFrame(sizing_data), hide_index=True, use_container_width=True)

        st.markdown("### Kelly Fractions")
        kelly_data = []
        for name, p, r_r in [("Setup 1 Bounce", 0.55, 2.5), ("Setup 2 Fade", 0.52, 2.2),
                               ("Setup 3 Breakout", 0.45, 3.0), ("Setup 4 Reversal", 0.40, 4.0),
                               ("Setup 5 Pin", 0.65, 1.5)]:
            f_star = (p * r_r - (1 - p)) / r_r
            kelly_data.append({
                "Setup": name, "p(win)": f"{p:.0%}", "R:R": f"{r_r:.1f}",
                "Full Kelly": f"{f_star:.0%}", "Half Kelly": f"{f_star/2:.0%}",
                "Rec. Risk%": f"{min(f_star/2*100, 15):.0f}%",
            })
        st.dataframe(pd.DataFrame(kelly_data), hide_index=True, use_container_width=True)
