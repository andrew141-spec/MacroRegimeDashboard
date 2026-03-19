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

def _days_to_exp(label: str) -> int:
    """Convert 'Mar 18' style label back to days from today.""",
    try:
        exp = dt.datetime.strptime(label + f" {dt.date.today().year}", "%b %d %Y").date()
        return (exp - dt.date.today()).days
    except Exception:
        return 999  # unknown = treat as far-dated


def _make_heatmap(chain_df: pd.DataFrame, spot: float,
                  greek: str = "net_gex",
                  title: str = "GEX Heatmap",
                  height: int = 580) -> go.Figure:
    """
    Strike × Expiry heatmap.
    Uses NUMERIC y-axis (actual strike prices) so shapes/lines work correctly.
    Spot is always centred by filtering ±pct% symmetrically.
    """
    from gex_engine import compute_gex_from_chain

    if chain_df is None or len(chain_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False,
                           xref="paper", yref="paper", font=dict(color="white", size=14))
        return fig

    gex_chain = compute_gex_from_chain(chain_df, spot)
    greek_col = {"net_gex":"net_gex","net_vex":"net_vex","net_cex":"net_cex"}.get(greek,"net_gex")

    # ── Expiry labels ─────────────────────────────────────────────────────
    gex_chain["exp_label"] = (gex_chain["expiry_T"] * 365).round(0).astype(int).apply(
        lambda d: (dt.date.today() + dt.timedelta(days=int(d))).strftime("%Y-%m-%d")
    )

    # ── Pivot: numeric strike index ───────────────────────────────────────
    pivot = (gex_chain
             .groupby(["strike", "exp_label"])[greek_col]
             .sum()
             .unstack(fill_value=0)
             / 1e6)
    pivot.columns = pivot.columns.get_level_values(0) if pivot.columns.nlevels > 1 else pivot.columns

    # ── Filters ───────────────────────────────────────────────────────────
    # Use absolute strike bounds set by the UI controls (saved in session_state)
    # Fall back to ±6% of spot if not set
    default_lo = round(spot * 0.94 / 5) * 5   # round to nearest $5
    default_hi = round(spot * 1.06 / 5) * 5
    strike_lo = getattr(_make_heatmap, "_strike_lo", default_lo)
    strike_hi = getattr(_make_heatmap, "_strike_hi", default_hi)
    max_dte   = getattr(_make_heatmap, "_max_dte",    30)

    # Also store pct for the spot-centering range in yaxis
    pct = max(abs(strike_hi - spot), abs(spot - strike_lo)) / spot

    pivot = pivot[(pivot.index >= strike_lo) & (pivot.index <= strike_hi)]

    def _dte(col):
        try:    return (dt.date.fromisoformat(col) - dt.date.today()).days
        except: return 999

    keep = [c for c in pivot.columns if _dte(c) <= max_dte]
    if keep:
        pivot = pivot[sorted(keep)]

    pivot = pivot.loc[(pivot.abs() >= 0.5).any(axis=1)]
    if pivot.empty:
        fig = go.Figure()
        fig.add_annotation(text="No near-term data in range", x=0.5, y=0.5,
                           showarrow=False, xref="paper", yref="paper",
                           font=dict(color="white", size=13))
        return fig

    # ── TOTAL column ──────────────────────────────────────────────────────
    pivot["TOTAL"] = pivot.sum(axis=1)

    # Strikes ascending for the matrix (Heatmap y goes bottom→top by default)
    # We will flip with yaxis.autorange="reversed" so highest strike is at top
    pivot = pivot.sort_index(ascending=True)

    strikes     = list(pivot.index)          # numeric: [580, 585, 590 ...]
    display_x   = [c if c == "TOTAL" else
                   dt.date.fromisoformat(c).strftime("%b %-d")
                   if _dte(c) < 999 else c
                   for c in pivot.columns]
    z_vals      = pivot.values.tolist()
    n_rows, n_cols = len(strikes), len(pivot.columns)

    # ── Cell text ─────────────────────────────────────────────────────────
    thresh = 0.5
    def _cell(v):
        if abs(v) < thresh: return ""
        if abs(v) >= 1000:  return f"${v/1000:.1f}B"
        return f"${v:.1f}M"
    text_vals = [[_cell(v) for v in row] for row in z_vals]

    # ── Colour ────────────────────────────────────────────────────────────
    flat = [abs(v) for row in z_vals for v in row if abs(v) >= thresh]
    zmax = float(np.percentile(flat, 97)) if flat else 100.0
    colorscale = [
        [0.00, "#7f1d1d"], [0.30, "#ef4444"],
        [0.47, "#1c1c1c"], [0.50, "#111111"], [0.53, "#1c1c1c"],
        [0.70, "#10b981"], [1.00, "#064e3b"],
    ]

    # ── Figure ────────────────────────────────────────────────────────────
    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=display_x,
        y=strikes,          # NUMERIC — enables correct shapes and range
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=max(7, min(10, int(380/max(n_cols,1)))),
                      color="rgba(255,255,255,0.90)", family="monospace"),
        colorscale=colorscale,
        zmid=0, zmin=-zmax, zmax=zmax,
        showscale=True,
        colorbar=dict(tickfont=dict(size=9, color="rgba(255,255,255,0.6)"),
                      thickness=14, len=0.88, tickformat="$.0s",
                      bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)"),
        xgap=0, ygap=0,
        hovertemplate="Strike: $%{y:.0f}<br>Expiry: %{x}<br>%{z:.1f}M<extra></extra>",
    ))

    # ── Spot line — horizontal, numeric y ────────────────────────────────
    # Nearest strike to spot
    nearest = min(strikes, key=lambda s: abs(s - spot))
    # Draw a cyan line at the spot strike value on the numeric y axis
    fig.add_shape(
        type="line",
        x0=-0.5, x1=n_cols - 0.5,
        y0=nearest, y1=nearest,
        xref="x", yref="y",
        line=dict(color="#06b6d4", width=2),
    )
    # Spot label on the left
    fig.add_annotation(
        x=-0.5, y=nearest,
        text=f"▶ ${spot:.2f}",
        showarrow=False,
        xref="x", yref="y",
        xanchor="right",
        font=dict(color="#06b6d4", size=10, family="monospace"),
    )

    # ── TOTAL separator ───────────────────────────────────────────────────
    fig.add_shape(
        type="line",
        x0=n_cols - 1.5, x1=n_cols - 1.5,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="rgba(255,255,255,0.20)", width=1, dash="dot"),
    )

    # ── Y-axis tick labels: "$630" format, custom per strike ─────────────
    tickvals = strikes
    ticktext = [f"${s:.0f}" for s in strikes]

    fig.update_layout(
        title=dict(
            text=f"{title} — {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Current Price: ${spot:.2f}",
            font=dict(size=11, color="rgba(255,255,255,0.65)"), x=0.5,
        ),
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        font=dict(color="rgba(255,255,255,0.80)", family="monospace"),
        height=height,
        margin=dict(l=60, r=70, t=40, b=50),
        xaxis=dict(side="bottom", tickfont=dict(size=9, color="rgba(255,255,255,0.65)"),
                   tickangle=-25, showgrid=False, fixedrange=False),
        yaxis=dict(
            tickvals=tickvals,
            ticktext=ticktext,
            tickfont=dict(size=9, color="rgba(255,255,255,0.75)"),
            showgrid=False, fixedrange=False,
            # Tight range: from lowest to highest ACTUAL strike in pivot
            # This eliminates empty rows at top/bottom
            range=[min(strikes) - 0.5, max(strikes) + 0.5],
            autorange=False,
        ),
    )
    return fig

    return fig

    return fig




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



def _levels(dg, gs, spot, side="long"):
    """
    Compute concrete entry, stop, and target levels from GEX data.
    side: "long" or "short"
    Returns dict with entry, stop, t1, t2, rr1, rr2
    """
    gex  = dg.gex_by_strike
    flip = gs.gamma_flip if gs.gamma_flip else spot

    # Key positive nodes (call walls / support) — sorted by proximity to spot
    pos_above = sorted([(k,v) for k,v in gex.items() if v > 0 and k > spot], key=lambda x: x[0])
    pos_below = sorted([(k,v) for k,v in gex.items() if v > 0 and k < spot], key=lambda x: -x[0])
    neg_above = sorted([(k,v) for k,v in gex.items() if v < 0 and k > spot], key=lambda x: x[0])
    neg_below = sorted([(k,v) for k,v in gex.items() if v < 0 and k < spot], key=lambda x: -x[0])

    # Biggest nodes (by abs value) above and below
    biggest_above = max([(k,v) for k,v in gex.items() if k > spot], key=lambda x:abs(x[1]), default=(spot*1.01, 0))
    biggest_below = min([(k,v) for k,v in gex.items() if k < spot], key=lambda x:abs(x[1]), default=(spot*0.99, 0))

    # Nearest positive node below (support) and nearest positive node above (resistance)
    nearest_support    = pos_below[0][0]  if pos_below    else spot * 0.995
    nearest_resistance = pos_above[0][0]  if pos_above    else spot * 1.005
    nearest_neg_below  = neg_below[0][0]  if neg_below    else spot * 0.99
    nearest_neg_above  = neg_above[0][0]  if neg_above    else spot * 1.01

    # Second targets (next significant node beyond first)
    sec_support    = pos_below[1][0] if len(pos_below) > 1 else nearest_support    * 0.995
    sec_resistance = pos_above[1][0] if len(pos_above) > 1 else nearest_resistance * 1.005

    if side == "long":
        entry = nearest_support          # buy at/near support node
        stop  = round(entry * 0.997, 2)  # 0.3% below entry (just below the node)
        t1    = nearest_resistance       # first positive node above
        t2    = sec_resistance           # second positive node above
        rr1   = round((t1 - entry) / (entry - stop), 1) if entry > stop else 0
        rr2   = round((t2 - entry) / (entry - stop), 1) if entry > stop else 0
    else:
        entry = nearest_resistance       # sell at/near resistance node
        stop  = round(entry * 1.003, 2)  # 0.3% above entry
        t1    = nearest_support          # first positive node below
        t2    = sec_support              # second positive node below
        rr1   = round((entry - t1) / (stop - entry), 1) if stop > entry else 0
        rr2   = round((entry - t2) / (stop - entry), 1) if stop > entry else 0

    return {
        "entry": round(entry, 2),
        "stop":  round(stop,  2),
        "t1":    round(t1,    2),
        "t2":    round(t2,    2),
        "rr1":   rr1,
        "rr2":   rr2,
        "flip":  round(flip,  2),
    }


def _module3_setups(dg: "DealerGreeks", gs: "GammaState", spot: float,
                    vix_level: float, session: dict) -> list:
    """
    Evaluate Module 3 setups with precise entry/stop/target levels.
    Each setup checks real conditions, then computes exact price levels.
    """
    setups = []
    regime = gs.regime
    dist   = gs.distance_to_flip_pct
    flip   = gs.gamma_flip or spot

    # Near-the-money aggregates (within 2% of spot)
    ntm_gex = sum(v for k, v in dg.gex_by_strike.items() if abs(k-spot)/spot < 0.02)
    ntm_vex = sum(v for k, v in dg.vex_by_strike.items() if abs(k-spot)/spot < 0.02)
    ntm_cex = sum(v for k, v in dg.cex_by_strike.items() if abs(k-spot)/spot < 0.02)

    # Consecutive same-sign strikes in 5% band
    near = sorted([k for k in dg.gex_by_strike if abs(k-spot)/spot < 0.05])
    pos_run = sum(1 for k in near if dg.gex_by_strike.get(k, 0) > 0)
    neg_run = sum(1 for k in near if dg.gex_by_strike.get(k, 0) < 0)

    # Largest node above/below
    gex = dg.gex_by_strike
    above = [(k,v) for k,v in gex.items() if k > spot]
    below = [(k,v) for k,v in gex.items() if k < spot]
    largest_above = max(above, key=lambda x:abs(x[1])) if above else (spot*1.01, 0)
    largest_below = min(below, key=lambda x:abs(x[1])) if below else (spot*0.99, 0)

    pos_nodes_above = sorted([(k,v) for k,v in gex.items() if v > 0 and k > spot], key=lambda x:x[0])
    pos_nodes_below = sorted([(k,v) for k,v in gex.items() if v > 0 and k < spot], key=lambda x:-x[0])
    neg_nodes_above = sorted([(k,v) for k,v in gex.items() if v < 0 and k > spot], key=lambda x:x[0])
    neg_nodes_below = sorted([(k,v) for k,v in gex.items() if v < 0 and k < spot], key=lambda x:-x[0])

    is_late  = session["window"] in ("Afternoon", "Close/MOC")
    is_prime = session["window"] == "Morning"
    near_flip = abs(dist) < 1.5

    def _setup(name, icon, met, mis, action, entry_text, stop_text,
               t1_text, t2_text, rr, note, module, active=None):
        a = (len(met) >= 2) if active is None else active
        setups.append({
            "name": name, "icon": icon, "active": a,
            "conditions_met": met, "conditions_missing": mis,
            "action": action,
            "entry": entry_text, "stop": stop_text,
            "t1": t1_text, "t2": t2_text, "rr": rr,
            "note": note, "module": module,
        })

    # ── 1. Gamma Pin & Magnet Drift ───────────────────────────────────────
    # Conditions: positive regime + consecutive positive strikes + large nodes defining range
    m, x = [], []
    (m if regime in (GammaRegime.STRONG_POSITIVE, GammaRegime.POSITIVE) else x).append("✓ Positive gamma regime")
    (m if pos_run >= 3 else x).append(f"{'✓' if pos_run>=3 else '○'} {pos_run}/3 consecutive positive strikes near spot")
    (m if (abs(largest_above[1]) > 50e6 or abs(largest_below[1]) > 50e6) else x).append(
        f"{'✓' if abs(largest_above[1])>50e6 or abs(largest_below[1])>50e6 else '○'} Large nodes defining range ($50M+)")

    sup  = pos_nodes_below[0][0] if pos_nodes_below else spot * 0.997
    res  = pos_nodes_above[0][0] if pos_nodes_above else spot * 1.003
    sup2 = pos_nodes_below[1][0] if len(pos_nodes_below) > 1 else sup * 0.997
    res2 = pos_nodes_above[1][0] if len(pos_nodes_above) > 1 else res * 1.003
    stop_l = round(sup * 0.9975, 2)
    stop_s = round(res * 1.0025, 2)
    rr_l = round((res - sup) / (sup - stop_l), 1) if sup > stop_l else 0
    rr_s = round((res - sup) / (stop_s - res), 1) if stop_s > res else 0

    _setup("Gamma Pin & Magnet Drift", "📌", m, x,
           f"Range trade between ${sup:.2f} support and ${res:.2f} resistance. "
           f"Dealers mechanically buy dips to ${sup:.2f} and sell rips to ${res:.2f}.",
           f"LONG: ${sup:.2f} (support node) | SHORT: ${res:.2f} (resistance node)",
           f"LONG stop: ${stop_l:.2f} (below node) | SHORT stop: ${stop_s:.2f} (above node)",
           f"T1 (long): ${res:.2f} | T1 (short): ${sup:.2f}",
           f"T2 (long): ${res2:.2f} | T2 (short): ${sup2:.2f}",
           f"~{rr_l:.1f}:1 long / ~{rr_s:.1f}:1 short",
           "Fade moves away from the node. Wait for price to reach the level — do not anticipate.",
           "§1")

    # ── 2. Node Break — Regime Shift ─────────────────────────────────────
    # Conditions: near gamma flip + large node within 1.5% of spot
    m, x = [], []
    (m if near_flip else x).append(f"{'✓' if near_flip else '○'} Near gamma flip (dist: {dist:+.2f}%, need <1.5%)")
    large_nearby = [(k,v) for k,v in gex.items() if abs(k-spot)/spot < 0.015 and abs(v) > 100e6]
    (m if large_nearby else x).append(f"{'✓' if large_nearby else '○'} Large node ($100M+) within 1.5% of spot")
    (m if regime in (GammaRegime.POSITIVE, GammaRegime.NEUTRAL) else x).append(
        f"{'✓' if regime in (GammaRegime.POSITIVE,GammaRegime.NEUTRAL) else '○'} Transitional regime")

    # Entry is a break of the flip level
    break_long_entry  = round(flip * 1.001, 2)   # just above flip
    break_short_entry = round(flip * 0.999, 2)   # just below flip
    break_stop_l      = round(flip * 0.998, 2)   # invalidation: flip reclaimed
    break_stop_s      = round(flip * 1.002, 2)
    next_pos_above    = pos_nodes_above[0][0] if pos_nodes_above else flip * 1.01
    next_neg_below    = neg_nodes_below[0][0] if neg_nodes_below else flip * 0.99
    rr_break_l        = round((next_pos_above - break_long_entry)  / (break_long_entry  - break_stop_l), 1) if break_long_entry > break_stop_l else 0
    rr_break_s        = round((break_short_entry - next_neg_below) / (break_stop_s - break_short_entry), 1) if break_stop_s > break_short_entry else 0

    _setup("Node Break — Regime Shift", "💥", m, x,
           f"Enter on confirmed break through flip at ${flip:.2f}. "
           f"Failed rejection + fast node decrease = signal. Target next node on breakout side.",
           f"LONG break: ${break_long_entry:.2f} (flip + buffer) | SHORT break: ${break_short_entry:.2f}",
           f"LONG stop: ${break_stop_l:.2f} (flip reclaimed = invalid) | SHORT stop: ${break_stop_s:.2f}",
           f"T1 (long): ${next_pos_above:.2f} | T1 (short): ${next_neg_below:.2f}",
           f"T2: next major node beyond T1",
           f"~{rr_break_l:.1f}:1 long / ~{rr_break_s:.1f}:1 short",
           "Enter ONLY on a confirmed close through flip — not a wick. Flip reclaimed = exit immediately.",
           "§2")

    # ── 3. Negative Gamma Amplification ──────────────────────────────────
    m, x = [], []
    (m if regime in (GammaRegime.NEGATIVE, GammaRegime.STRONG_NEGATIVE) else x).append(
        f"{'✓' if regime in (GammaRegime.NEGATIVE,GammaRegime.STRONG_NEGATIVE) else '○'} Negative gamma regime")
    (m if neg_run >= 3 else x).append(f"{'✓' if neg_run>=3 else '○'} {neg_run}/3 consecutive negative strikes near spot")
    (m if ntm_gex < -100e6 else x).append(
        f"{'✓' if ntm_gex < -100e6 else '○'} Strong negative GEX cluster (${ntm_gex/1e6:.0f}M, need -$100M)")

    # In neg gamma: follow breakouts. Entry = current direction continuation.
    # Next neg node in direction of move = target (dealers amplify through them)
    mom_dir = "long" if spot > flip else "short"
    if mom_dir == "long":
        ng_entry = spot      # enter now (momentum)
        ng_stop  = round(flip * 0.999, 2)   # flip is invalidation
        ng_t1    = neg_nodes_above[0][0] if neg_nodes_above else spot * 1.01
        ng_t2    = neg_nodes_above[1][0] if len(neg_nodes_above) > 1 else ng_t1 * 1.01
        ng_rr    = round((ng_t1 - ng_entry) / (ng_entry - ng_stop), 1) if ng_entry > ng_stop else 0
    else:
        ng_entry = spot
        ng_stop  = round(flip * 1.001, 2)
        ng_t1    = neg_nodes_below[0][0] if neg_nodes_below else spot * 0.99
        ng_t2    = neg_nodes_below[1][0] if len(neg_nodes_below) > 1 else ng_t1 * 0.99
        ng_rr    = round((ng_entry - ng_t1) / (ng_stop - ng_entry), 1) if ng_stop > ng_entry else 0

    _setup("Negative Gamma Amplification", "🌊", m, x,
           f"Follow breakouts — dealers amplify moves. Momentum: {'LONG' if mom_dir=='long' else 'SHORT'} bias "
           f"(spot {'above' if mom_dir=='long' else 'below'} flip ${flip:.2f}). Do NOT fade.",
           f"{'LONG' if mom_dir=='long' else 'SHORT'} @ ${ng_entry:.2f} (market / pullback entry)",
           f"Stop: ${ng_stop:.2f} (flip ${flip:.2f} = regime change, exit immediately)",
           f"T1: ${ng_t1:.2f} (next negative node — dealers amplify through it)",
           f"T2: ${ng_t2:.2f}",
           f"~{ng_rr:.1f}:1",
           "Gap-and-go expected. Do NOT fade. Trail stop to entry once T1 hit.",
           "§3")

    # ── 4. Vanna Squeeze ─────────────────────────────────────────────────
    m, x = [], []
    (m if ntm_vex > 50e6 else x).append(f"{'✓' if ntm_vex>50e6 else '○'} Positive vanna near spot (${ntm_vex/1e6:.0f}M, need $50M+)")
    (m if vix_level > 22 else x).append(f"{'✓' if vix_level>22 else '○'} IV elevated (VIX {vix_level:.1f}, need >22)")
    (m if regime in (GammaRegime.POSITIVE, GammaRegime.NEUTRAL) else x).append(
        f"{'✓' if regime in (GammaRegime.POSITIVE,GammaRegime.NEUTRAL) else '○'} Positive/neutral gamma")

    # Vanna squeeze: buy near largest positive GEX below, target positive nodes above
    vs_entry = pos_nodes_below[0][0] if pos_nodes_below else spot * 0.998
    vs_stop  = round(vs_entry * 0.997, 2)
    vs_t1    = pos_nodes_above[0][0] if pos_nodes_above else spot * 1.01
    vs_t2    = pos_nodes_above[1][0] if len(pos_nodes_above) > 1 else vs_t1 * 1.01
    vs_rr    = round((vs_t1 - vs_entry) / (vs_entry - vs_stop), 1) if vs_entry > vs_stop else 0

    _setup("Vanna Squeeze", "🚀", m, x,
           f"IV drop → call deltas rise → dealers BUY → self-reinforcing rally. "
           f"Accumulate near ${vs_entry:.2f} (positive GEX node). "
           f"Feedback loop accelerates as IV compresses further.",
           f"LONG @ ${vs_entry:.2f} (largest positive GEX node below spot)",
           f"Stop: ${vs_stop:.2f} (node broken = squeeze thesis invalid)",
           f"T1: ${vs_t1:.2f} (next positive node above — take 50% here)",
           f"T2: ${vs_t2:.2f} (trail remainder)",
           f"~{vs_rr:.1f}:1",
           "Most powerful post-FOMC/earnings when IV crushes. Patience at entry — wait for node hold.",
           "§4")

    # ── 5. Vanna Rug / Vol Shock ─────────────────────────────────────────
    neg_vex_ntm = sum(v for k,v in dg.vex_by_strike.items() if abs(k-spot)/spot < 0.03 and v < 0)
    m, x = [], []
    (m if neg_vex_ntm < -30e6 else x).append(f"{'✓' if neg_vex_ntm < -30e6 else '○'} Negative vanna near spot (${neg_vex_ntm/1e6:.0f}M, need -$30M+)")
    (m if vix_level < 18 else x).append(f"{'✓' if vix_level<18 else '○'} IV compressed (VIX {vix_level:.1f}, need <18)")
    (m if (session.get('is_data_day') or session.get('is_opex_friday')) else x).append(
        f"{'✓' if session.get('is_data_day') or session.get('is_opex_friday') else '○'} Catalyst present (data day/OpEx)")

    # Vanna rug: short bias before catalyst, stop above nearest resistance
    vr_entry = neg_nodes_above[0][0] if neg_nodes_above else spot * 1.002
    vr_stop  = round(vr_entry * 1.003, 2)
    vr_t1    = neg_nodes_below[0][0] if neg_nodes_below else spot * 0.99
    vr_t2    = neg_nodes_below[1][0] if len(neg_nodes_below) > 1 else vr_t1 * 0.99
    vr_rr    = round((vr_entry - vr_t1) / (vr_stop - vr_entry), 1) if vr_stop > vr_entry else 0

    _setup("Vanna Rug / Vol Shock", "🧨", m, x,
           f"IV spike → dealers SELL → cascade. Short bias BEFORE catalyst. "
           f"Negative vanna + compressed IV = most dangerous combo. "
           f"Do not hold through event — position BEFORE.",
           f"SHORT @ ${vr_entry:.2f} (negative node, pre-catalyst)",
           f"Stop: ${vr_stop:.2f} (above node = thesis wrong, exit fast)",
           f"T1: ${vr_t1:.2f} (first negative node below — cascades through it)",
           f"T2: ${vr_t2:.2f} (next node — trail stop to entry after T1)",
           f"~{vr_rr:.1f}:1",
           "Must be positioned BEFORE the catalyst. IV spike + negative vanna = self-reinforcing. Cover into panic, not after.",
           "§5")

    # ── 6. Charm Drift (EOD) ─────────────────────────────────────────────
    m, x = [], []
    strong_charm = abs(ntm_cex) > 20e6
    charm_long = ntm_cex > 0
    (m if strong_charm else x).append(f"{'✓' if strong_charm else '○'} Strong charm near spot (${ntm_cex/1e6:.0f}M, need ±$20M+)")
    (m if is_late else x).append(f"{'✓' if is_late else '○'} Late session — charm strongest after 2pm (currently: {session['window']})")

    cd_entry = spot
    if charm_long:
        cd_stop = pos_nodes_below[0][0] * 0.998 if pos_nodes_below else spot * 0.997
        cd_t1   = pos_nodes_above[0][0] if pos_nodes_above else spot * 1.005
        cd_t2   = pos_nodes_above[1][0] if len(pos_nodes_above) > 1 else cd_t1 * 1.003
    else:
        cd_stop = pos_nodes_above[0][0] * 1.002 if pos_nodes_above else spot * 1.003
        cd_t1   = pos_nodes_below[0][0] if pos_nodes_below else spot * 0.995
        cd_t2   = pos_nodes_below[1][0] if len(pos_nodes_below) > 1 else cd_t1 * 0.997
    cd_rr = round(abs(cd_t1 - cd_entry) / abs(cd_entry - cd_stop), 1) if cd_entry != cd_stop else 0

    _setup("Charm Drift (EOD)", "⏰", m, x,
           f"As 0DTE options decay, dealers {'buy' if charm_long else 'sell'} to rebalance delta. "
           f"EOD {'upward' if charm_long else 'downward'} drift expected. "
           f"Strongest 14:00–16:00 ET. Exit by 15:50.",
           f"{'LONG' if charm_long else 'SHORT'} @ ${cd_entry:.2f} (current spot, enter after 2pm)",
           f"Stop: ${cd_stop:.2f} ({'below support' if charm_long else 'above resistance'})",
           f"T1: ${cd_t1:.2f} (nearest GEX {'resistance' if charm_long else 'support'} node)",
           f"T2: ${cd_t2:.2f} | Hard exit: 15:50 ET regardless",
           f"~{cd_rr:.1f}:1",
           "Exit before 15:50 ET — MOC flow overrides charm after that. Reduce size 50% if VIX > 25.",
           "§6")

    # ── 7. Charm-Vanna Alignment ─────────────────────────────────────────
    m, x = [], []
    (m if dg.vanna_charm_aligned else x).append(
        f"{'✓' if dg.vanna_charm_aligned else '○'} Vanna ({dg.vanna_direction}) + Charm ({dg.charm_direction}) aligned")
    both_size = abs(ntm_vex) > 30e6 and abs(ntm_cex) > 15e6
    (m if both_size else x).append(f"{'✓' if both_size else '○'} VEX (${ntm_vex/1e6:.0f}M need $30M+) and CEX (${ntm_cex/1e6:.0f}M need $15M+)")

    align_long = dg.vanna_direction == "bullish"
    if align_long:
        al_entry = pos_nodes_below[0][0] if pos_nodes_below else spot * 0.998
        al_stop  = round(al_entry * 0.997, 2)
        al_t1    = pos_nodes_above[0][0] if pos_nodes_above else spot * 1.005
        al_t2    = pos_nodes_above[1][0] if len(pos_nodes_above) > 1 else al_t1 * 1.005
    else:
        al_entry = pos_nodes_above[0][0] if pos_nodes_above else spot * 1.002
        al_stop  = round(al_entry * 1.003, 2)
        al_t1    = pos_nodes_below[0][0] if pos_nodes_below else spot * 0.995
        al_t2    = pos_nodes_below[1][0] if len(pos_nodes_below) > 1 else al_t1 * 0.995
    al_rr = round(abs(al_t1 - al_entry) / abs(al_entry - al_stop), 1) if al_entry != al_stop else 0

    _setup("Charm-Vanna Alignment", "⚡", m, x,
           f"Both second-order flows lean {dg.vanna_direction}. "
           f"High-conviction directional window — two mechanical flows compounding.",
           f"{'LONG' if align_long else 'SHORT'} @ ${al_entry:.2f}",
           f"Stop: ${al_stop:.2f}",
           f"T1: ${al_t1:.2f}",
           f"T2: ${al_t2:.2f}",
           f"~{al_rr:.1f}:1",
           "Highest-conviction window. Scale in — 50% at entry, add 50% on first pullback to entry zone.",
           "§7",
           active=dg.vanna_charm_aligned and len(m) >= 2)

    return setups




def render_gex_engine():
    """Deep-dive GEX analysis page with GEX / VEX / CEX tabs."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## ⚡ GEX Engine — Dealer Greeks")

    col_s, col_m = st.columns([1, 3])
    with col_s:
        symbol    = st.text_input("Options Symbol", "QQQ", key="gex_symbol_input").strip().upper()
        use_schwab = st.toggle("Use Schwab/TOS (live IV)", False, key="gex_use_schwab")

    # ── Data fetch ────────────────────────────────────────────────────────
    chain_df, spot, source = None, 0.0, "unknown"
    with col_m:
        if use_schwab:
            client = get_schwab_client()
            if client:
                st.info("Schwab connected — fetching live chain")
                # Always pass spot=None so schwab_get_options_chain fetches
                # the price from the API response itself — avoids using a
                # stale spot from a previously loaded symbol (e.g. SPY→QQQ)
                chain_df = schwab_get_options_chain(client, symbol, spot=None)
                source   = "Schwab API (live IV)"
                if chain_df is not None and len(chain_df) > 0:
                    # Get spot from a live quote (after chain is confirmed working)
                    spot_live = schwab_get_spot(client, symbol)
                    if spot_live and spot_live > 0:
                        spot = spot_live
                    else:
                        # Fall back to midpoint of chain strikes
                        spot = float(chain_df["strike"].median())
                else:
                    err = st.session_state.get("_schwab_chain_error", "unknown error")
                    st.warning(f"Schwab chain empty — {err} — falling back to yfinance")
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
    # ── Sync heatmap config from session_state (works regardless of which tab is active) ──
    if "gex_strikes_each_side" not in st.session_state:
        st.session_state["gex_strikes_each_side"] = 20
    if "gex_hm_dte"            not in st.session_state:
        st.session_state["gex_hm_dte"]            = 30
    if "gex_hm_height"         not in st.session_state:
        st.session_state["gex_hm_height"]         = 1000
    _n = int(st.session_state["gex_strikes_each_side"])
    _make_heatmap._strike_lo = float(spot - _n)
    _make_heatmap._strike_hi = float(spot + _n)
    _make_heatmap._max_dte   = int(st.session_state["gex_hm_dte"])

    tab_gex, tab_vex, tab_cex, tab_nodes, tab_otm = st.tabs(
        ["📊 GEX (Gamma)", "🌀 VEX (Vanna)", "⏱ CEX (Charm)", "🎯 Key Nodes", "🔭 OTM Anchors"]
    )

    with tab_gex:
        st.markdown(f"{sec_hdr('GAMMA EXPOSURE — Reaction to Price')}", unsafe_allow_html=True)

        view_mode = st.radio("View", ["Heatmap", "Bar Chart"], horizontal=True, key="gex_view_mode")

        if view_mode == "Heatmap":
            st.caption("Strike × Expiry matrix · Green = positive GEX (dealers stabilize) · Red = negative GEX (dealers amplify) · → = spot price · TOTAL = net across all expiries")

            # ── Persistent controls (survive tab switches via session_state keys) ──
            if "gex_strikes_each_side" not in st.session_state:
                st.session_state["gex_strikes_each_side"] = 20
            if "gex_hm_dte"            not in st.session_state:
                st.session_state["gex_hm_dte"]            = 30
            if "gex_hm_height"         not in st.session_state:
                st.session_state["gex_hm_height"]         = 1000

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                strikes_each_side = st.number_input(
                    "Strikes each side of spot",
                    min_value=5, max_value=100, step=5, format="%d",
                    key="gex_strikes_each_side",
                    help=f"Shows this many $1 strikes above AND below ${spot:.0f}. "
                         f"e.g. 20 → ${spot-20:.0f} to ${spot+20:.0f}",
                )
            with sc2:
                max_dte = st.number_input(
                    "Max DTE (days)", min_value=0, max_value=365,
                    step=7, format="%d",
                    key="gex_hm_dte",
                )
            with sc3:
                hm_height = st.number_input(
                    "Height (px)", min_value=300, max_value=2000,
                    step=50, format="%d",
                    key="gex_hm_height",
                )

            # Compute symmetric bounds centred on spot
            # Round to nearest $1 so they align with actual option strikes
            strike_lo = float(spot - int(strikes_each_side))
            strike_hi = float(spot + int(strikes_each_side))

            _make_heatmap._strike_lo = strike_lo
            _make_heatmap._strike_hi = strike_hi
            _make_heatmap._max_dte   = int(max_dte)

            # Show the computed range as info text
            st.caption(
                f"Showing strikes ${strike_lo:.0f} – ${strike_hi:.0f} "
                f"({int(strikes_each_side)} each side of spot ${spot:.2f}) · "
                f"Max DTE: {int(max_dte)}d · Height: {int(hm_height)}px"
            )

            fig_gex = _make_heatmap(chain_df, spot, "net_gex",
                                    f"{symbol} GEX", int(hm_height))
            st.plotly_chart(fig_gex, use_container_width=True)
        else:
            st.caption("Green = positive gamma (dealers stabilize). Red = negative gamma (dealers amplify). Yellow line = gamma flip.")
            # Use same strike range setting as heatmap
            if "gex_strikes_each_side" not in st.session_state:
                st.session_state["gex_strikes_each_side"] = 20
            if "gex_hm_height" not in st.session_state:
                st.session_state["gex_hm_height"] = 1000
            n_side   = int(st.session_state["gex_strikes_each_side"])
            bar_lo   = spot - n_side
            bar_hi   = spot + n_side
            filtered = {k: v for k, v in dg.gex_by_strike.items() if bar_lo <= k <= bar_hi}
            fig_gex = _greek_bar_chart(filtered, spot,
                                       "Net GEX by Strike ($M)", _C_POS, _C_NEG, gs.gamma_flip,
                                       height=int(st.session_state["gex_hm_height"]))
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
        if "yfinance" in source.lower():
            st.info("📋 VEX computed from EOD IV. With Schwab/TOS connected, this uses live per-strike IV for real-time vanna.")

        view_mode_vex = st.radio("View", ["Heatmap", "Bar Chart"], horizontal=True, key="vex_view_mode")
        if view_mode_vex == "Heatmap":
            st.caption("Strike × Expiry matrix · Purple/green = positive vanna · Red = negative vanna · TOTAL = net across all expiries")
            fig_vex = _make_heatmap(chain_df, spot, "net_vex", f"{symbol} VEX", int(st.session_state.get("gex_hm_height", 1000)))
            st.plotly_chart(fig_vex, use_container_width=True)
        else:
            fig_vex = _greek_bar_chart(dg.vex_by_strike, spot,
                                       "Net VEX by Strike ($M)", _C_VEX, _C_NEG, gs.gamma_flip)
            st.plotly_chart(fig_vex, use_container_width=True)

        v1, v2 = st.columns(2)
        ntm_vex_val = sum(v for k,v in dg.vex_by_strike.items() if abs(k-spot)/spot < 0.02)
        with v1:
            st.markdown(f"**Net Vanna near spot:** ${ntm_vex_val/1e6:.1f}M")
            st.markdown(f"**Vanna Sign:** {dg.vanna_sign.upper()}")
        with v2:
            st.markdown("**Interpretation:**")
            if dg.vanna_sign == "positive":
                st.markdown("""
**Positive Vanna near spot:**
- 📈 **IV rises** → dealers forced to **BUY** underlying (bullish pressure)
- 📉 **IV falls** → dealers forced to **SELL** underlying (bearish pressure)
- → **Vanna Squeeze**: IV elevated + likely to compress → dealer selling phase → then upside break
""")
            elif dg.vanna_sign == "negative":
                st.markdown("""
**Negative Vanna near spot:**
- 📈 **IV rises** → dealers forced to **SELL** underlying (bearish — amplifies selloff)
- 📉 **IV falls** → dealers forced to **BUY** underlying (bullish on vol crush)
- → **Vanna Rug risk**: catalyst spikes IV → dealers sell → cascading selloff
""")
            else:
                st.info("Neutral vanna — no strong IV-driven directional pressure.")

        st.markdown("""
**VEX = Reaction to Volatility**
- **Positive vanna**: IV rises → dealers BUY · IV falls → dealers SELL
- **Negative vanna**: IV rises → dealers SELL · IV falls → dealers BUY
- **Vanna Squeeze**: Positive vanna + IV elevated → IV compresses → dealers forced to buy → self-reinforcing rally
- **Vanna Rug**: Negative vanna near spot + IV compressed + catalyst → IV spikes → dealers forced to sell → cascading selloff
""")

    with tab_cex:
        st.markdown(f"{sec_hdr('CHARM EXPOSURE — Reaction to Time Decay')}", unsafe_allow_html=True)
        st.caption("Positive charm: time passing → dealers buy underlying (upward drift). Negative charm: dealers sell (downward drift).")
        if "yfinance" in source.lower():
            st.info("📋 CEX computed from EOD IV/OI. With Schwab/TOS, charm reflects live 0DTE positioning for intraday drift signals.")

        view_mode_cex = st.radio("View", ["Heatmap", "Bar Chart"], horizontal=True, key="cex_view_mode")
        if view_mode_cex == "Heatmap":
            st.caption("Strike × Expiry matrix · Teal/green = positive charm (dealers buy as time passes) · Red = negative · TOTAL = net")
            fig_cex = _make_heatmap(chain_df, spot, "net_cex", f"{symbol} CEX", int(st.session_state.get("gex_hm_height", 1000)))
            st.plotly_chart(fig_cex, use_container_width=True)
        else:
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
    """Trade Setups — full live context with symbol picker, all setups, entry/stop/target levels."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## 🎯 Trade Setups — Live Context")

    # ── Symbol + data ─────────────────────────────────────────────────────
    col_sym, col_sch, col_info = st.columns([1, 1, 3])
    with col_sym:
        symbol = st.text_input("Symbol", "QQQ", key="setups_symbol").strip().upper()
    with col_sch:
        use_schwab = st.toggle("Schwab (live IV)", False, key="setups_schwab")

    chain_df, spot, source = None, 580.0, "unknown"
    if use_schwab:
        client = get_schwab_client()
        if client:
            chain_df = schwab_get_options_chain(client, symbol, spot=None)
            source   = "Schwab API (live IV)"
            if chain_df is not None and len(chain_df) > 0:
                spot_live = schwab_get_spot(client, symbol)
                spot = spot_live if (spot_live and spot_live > 0) else float(chain_df["strike"].median())
            else:
                chain_df, spot, source = get_gex_from_yfinance(symbol)
        else:
            chain_df, spot, source = get_gex_from_yfinance(symbol)
    else:
        chain_df, spot, source = get_gex_from_yfinance(symbol)

    if chain_df is None or len(chain_df) == 0:
        st.error(f"No options data for {symbol}. Try refreshing or switching symbol.")
        return

    gs        = build_gamma_state(chain_df, spot, source)
    dg        = compute_dealer_greeks(chain_df, spot, source)
    session   = get_session_context()
    vix_df    = yf.Ticker("^VIX").history(period="1d")
    vix_level = float(vix_df["Close"].iloc[-1]) if len(vix_df) > 0 else 20.0
    fear_est  = float(np.clip((vix_level - 15) / 25 * 100, 0, 100))

    setups_orig = evaluate_setups(gs, session, spot, fear_est, vix_level)
    setups_m3   = _module3_setups(dg, gs, spot, vix_level, session)

    active_orig = sum(1 for s in setups_orig if s["active"] and s["score"].tradeable)
    active_m3   = sum(1 for s in setups_m3   if s["active"])
    total_active = active_orig + active_m3

    # ── Session / regime header bar ───────────────────────────────────────
    sm    = session["size_mult"]
    sm_c  = "#10b981" if sm >= 0.9 else ("#f59e0b" if sm >= 0.5 else ("#f97316" if sm > 0 else "#ef4444"))
    ev    = session.get("event_label", "")
    ev_html = f"<span style='color:#f59e0b;font-weight:700;'>⚠ {ev}</span> · " if ev else ""
    flip_str = f"${gs.gamma_flip:.2f}" if gs.gamma_flip else "N/A"

    st.markdown(f"""
<div style='display:flex;gap:12px;flex-wrap:wrap;align-items:center;
            padding:10px 14px;background:rgba(255,255,255,0.03);
            border:1px solid rgba(255,255,255,0.08);border-radius:10px;
            margin-bottom:10px;font-family:monospace;font-size:11px;'>
  {ev_html}
  <span>Session: <b style='color:{sm_c};'>{session['window']}</b></span>
  <span style='color:rgba(255,255,255,0.4);'>·</span>
  <span>Size: <b style='color:{sm_c};'>{sm:.2f}×</b></span>
  <span style='color:rgba(255,255,255,0.4);'>·</span>
  <span>Spot: <b>${spot:.2f}</b></span>
  <span style='color:rgba(255,255,255,0.4);'>·</span>
  <span>Flip: <b style='color:#f59e0b;'>{flip_str}</b></span>
  <span style='color:rgba(255,255,255,0.4);'>·</span>
  <span>Regime: <b>{gs.regime.value}</b></span>
  <span style='color:rgba(255,255,255,0.4);'>·</span>
  <span>Vanna: <b style='color:{"#10b981" if dg.vanna_direction=="bullish" else "#ef4444"};'>{dg.vanna_direction.upper()}</b></span>
  <span style='color:rgba(255,255,255,0.4);'>·</span>
  <span>Charm: <b style='color:{"#10b981" if dg.charm_direction=="bullish" else "#ef4444"};'>{dg.charm_direction.upper()}</b></span>
  <span style='color:rgba(255,255,255,0.4);'>·</span>
  <span>VIX: <b>{vix_level:.1f}</b></span>
</div>""", unsafe_allow_html=True)

    if total_active > 0:
        st.markdown(f"""<div style='padding:8px 14px;background:rgba(16,185,129,0.08);
            border:1px solid rgba(16,185,129,0.30);border-radius:8px;margin-bottom:10px;
            font-family:monospace;font-size:11px;'>
            ✅ <b>{total_active} setup{"s" if total_active>1 else ""} currently active</b>
            — {active_orig} core GEX · {active_m3} dealer flow
        </div>""", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_m3, tab_orig, tab_sizing = st.tabs([
        "🌊 Dealer Flow Setups", "📐 Core GEX Setups", "📏 Sizing Reference"
    ])

    # ── Tab 1: Module 3 Dealer Flow Setups (PRIMARY) ─────────────────────
    with tab_m3:
        st.markdown(f"{sec_hdr('DEALER FLOW SETUPS — ENTRY / STOP / TARGET')}", unsafe_allow_html=True)
        st.caption(
            "Levels derived from live GEX nodes. Entry = nearest relevant node. "
            "Stop = node break = thesis invalid. Targets = next significant nodes."
        )

        def _lcard(label, val, color):
            return (f"<div style='display:flex;justify-content:space-between;align-items:center;"
                    f"padding:5px 10px;margin:2px 0;background:rgba(255,255,255,0.04);"
                    f"border-radius:6px;border-left:3px solid {color};'>"
                    f"<span style='font-size:10px;color:rgba(255,255,255,0.50);"
                    f"font-family:monospace;letter-spacing:0.5px;'>{label}</span>"
                    f"<span style='font-family:monospace;font-size:12px;"
                    f"font-weight:700;color:{color};'>{val}</span></div>")

        for s in setups_m3:
            active = s["active"]
            border_color = "rgba(16,185,129,0.35)" if active else "rgba(255,255,255,0.08)"
            bg_color     = "rgba(16,185,129,0.05)"  if active else "rgba(0,0,0,0)"
            with st.expander(
                f"{s['icon']} {'✅ ACTIVE' if active else '⚪'} — {s['name']} ({s['module']})",
                expanded=active
            ):
                cond_col, lvl_col = st.columns([1, 1.1])

                with cond_col:
                    st.markdown("**Conditions**")
                    for c in s.get("conditions_met", []):
                        st.markdown(
                            f"<div style='font-size:11px;color:#10b981;padding:2px 0;'>{c}</div>",
                            unsafe_allow_html=True)
                    for c in s.get("conditions_missing", []):
                        st.markdown(
                            f"<div style='font-size:11px;color:rgba(255,255,255,0.35);padding:2px 0;'>{c}</div>",
                            unsafe_allow_html=True)
                    if s.get("action"):
                        st.markdown(
                            f"<div style='margin-top:8px;font-size:11px;"
                            f"color:rgba(255,255,255,0.70);line-height:1.4;'>"
                            f"{s['action']}</div>",
                            unsafe_allow_html=True)

                with lvl_col:
                    st.markdown("**Price Levels**")
                    html = ""
                    if s.get("entry"): html += _lcard("ENTRY",  s["entry"], "#06b6d4")
                    if s.get("stop"):  html += _lcard("STOP",   s["stop"],  "#ef4444")
                    if s.get("t1"):    html += _lcard("T1",     s["t1"],    "#10b981")
                    if s.get("t2"):    html += _lcard("T2",     s["t2"],    "#34d399")
                    if s.get("rr"):    html += _lcard("R:R",    s["rr"],    "#f59e0b")
                    st.markdown(html, unsafe_allow_html=True)

                if s.get("note"):
                    st.markdown(
                        f"<div style='margin-top:8px;padding:7px 10px;"
                        f"background:rgba(245,158,11,0.07);"
                        f"border-left:2px solid rgba(245,158,11,0.40);"
                        f"border-radius:4px;font-size:10px;"
                        f"color:rgba(255,255,255,0.60);'>⚠ {s['note']}</div>",
                        unsafe_allow_html=True)

    # ── Tab 2: Core GEX Setups (original 5) ──────────────────────────────
    with tab_orig:
        st.markdown(f"{sec_hdr('CORE GEX × ORDERFLOW SETUPS')}", unsafe_allow_html=True)
        st.caption("WHERE (GEX) + WHY (AMT) + WHEN (Orderflow). All three required.")

        for s in setups_orig:
            score     = s["score"]
            active    = s["active"]
            tradeable = score.tradeable
            icon = "✅" if (active and tradeable) else ("🟡" if active else "⚫")
            with st.expander(
                f"{icon} Setup {s['setup']}: {s['name']}",
                expanded=(active and tradeable)
            ):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Composite Score", f"{score.composite:.2f}")
                    st.metric("Est. Hit Rate",   f"{s['est_hit_rate']*100:.0f}%")
                    st.metric("Effective Size",  f"{s['effective_size']*100:.0f}%")
                with c2:
                    st.markdown("**Score Components**")
                    for label, val in [
                        ("Gamma Alignment",   score.gamma_alignment),
                        ("Orderflow Confirm", score.orderflow_confirmation),
                        ("TPO Context",       score.tpo_context),
                        ("Level Freshness",   score.level_freshness),
                        ("Event Risk",        score.event_risk),
                    ]:
                        c = "#10b981" if val >= 0.7 else ("#f59e0b" if val >= 0.5 else "#ef4444")
                        st.markdown(
                            f"<div style='display:flex;justify-content:space-between;'>"
                            f"<span class='small'>{label}</span>"
                            f"<span style='font-family:monospace;font-size:11px;color:{c};'>{val:.2f}</span>"
                            f"</div>", unsafe_allow_html=True)
                with c3:
                    status = "🟢 ACTIVE" if (active and tradeable) else ("🟡 Watching" if active else "⚫ Not Active")
                    st.markdown(f"**{status}**")
                    st.markdown(f"<div class='small'>{s['desc']}</div>", unsafe_allow_html=True)
                    if s.get("note"):
                        st.markdown(
                            f"<div class='warn-card' style='margin-top:6px;font-size:10px;'>{s['note']}</div>",
                            unsafe_allow_html=True)

    # ── Tab 3: Sizing reference ───────────────────────────────────────────
    with tab_sizing:
        st.markdown("### Position Sizing — Core Setups")
        st.dataframe(pd.DataFrame({
            "Setup":        ["1: Gamma Bounce","2: Gamma Fade","3: Flip Breakout","4: Exhaustion Rev.","5: 0DTE Pin"],
            "Base Size":    ["100%","100%","75%","50%","100%"],
            "Session Mult": [f"{sm:.2f}×"] * 5,
            "Vol Adj":      ["×0.5 if VIX>35, ×0.75 if VIX>25"] * 5,
            "Min R:R":      ["2:1","2:1","2:1","3:1","1.3:1"],
            "Est. Hit Rate":["~55%","~52%","~45%","~40%","~65%"],
        }), hide_index=True, use_container_width=True)

        st.markdown("### Kelly Fractions")
        kelly_data = []
        for name, p, r_r in [
            ("Setup 1 Bounce", 0.55, 2.5), ("Setup 2 Fade", 0.52, 2.2),
            ("Setup 3 Breakout", 0.45, 3.0), ("Setup 4 Reversal", 0.40, 4.0),
            ("Setup 5 Pin", 0.65, 1.5)
        ]:
            f_star = (p * r_r - (1 - p)) / r_r
            kelly_data.append({
                "Setup": name, "p(win)": f"{p:.0%}", "R:R": f"{r_r:.1f}",
                "Full Kelly": f"{f_star:.0%}", "Half Kelly": f"{f_star/2:.0%}",
                "Rec. Risk%": f"{min(f_star/2*100,15):.0f}%",
            })
        st.dataframe(pd.DataFrame(kelly_data), hide_index=True, use_container_width=True)
