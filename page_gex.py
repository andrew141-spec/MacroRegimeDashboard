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
from config import GammaState, GammaRegime, FeedItem, SetupScore, CSS, REGIME_OPERATIONAL_LABEL
from utils import _to_1d, zscore, resample_ffill, yf_close, kelly, current_pct_rank
from config import _get_secret
from ui_components import pill, pbar, sec_hdr, plotly_dark, regime_chip, autorefresh_js, colored, gauge
from data_loaders import get_gex_from_yfinance
from gex_engine import (build_gamma_state, compute_gex_from_chain, find_gamma_flip,
                        nearest_expiry_chain, compute_cumulative_gex_profile,
                        classify_gex_regime, compute_dealer_greeks, DealerGreeks,
                        compute_gwas, compute_gex_term_structure, compute_flow_imbalance,
                        gex_zero_crossing)
from schwab_api import (get_schwab_client, schwab_get_spot, schwab_get_options_chain,
                        SCHWAB_AVAILABLE)
# OI source priority: Schwab/TOS (live IV + volume) → yfinance fallback (OI only)
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



@st.cache_data(ttl=300)
def _fetch_rvol_surface(symbol: str = "SPY") -> dict:
    """
    Fetch VIX term structure (VIX / VIX3M / VIX6M) and compute realized vol.
    All tickers are free on yfinance — no Schwab needed for this.

    Returns:
      vix1m, vix3m, vix6m      — current IV term structure levels
      slope_1_3                 — VIX3M - VIX (positive = normal contango)
      slope_3_6                 — VIX6M - VIX3M
      rvol_5d                   — 5-day realized vol (annualized)
      iv_rv_spread              — VIX - RVol (positive = IV premium over realized)
      term_structure_regime     — "contango" / "backwardation" / "flat"
    """
    out = {}
    try:
        import yfinance as _yf
        # VIX term structure — all three free on yfinance
        for label, ticker in [("vix1m", "^VIX"), ("vix3m", "^VIX3M"), ("vix6m", "^VIX6M")]:
            try:
                h = _yf.Ticker(ticker).history(period="2d")
                out[label] = float(h["Close"].iloc[-1]) if not h.empty else None
            except Exception:
                out[label] = None

        # 5-day realized vol for the underlying
        try:
            h = _yf.Ticker(symbol).history(period="10d")
            if len(h) >= 5:
                log_rets = np.log(h["Close"] / h["Close"].shift(1)).dropna()
                out["rvol_5d"] = float(log_rets.tail(5).std() * np.sqrt(252) * 100)
            else:
                out["rvol_5d"] = None
        except Exception:
            out["rvol_5d"] = None

        # Derived metrics
        v1, v3, v6 = out.get("vix1m"), out.get("vix3m"), out.get("vix6m")
        out["slope_1_3"] = round(v3 - v1, 2) if (v1 and v3) else None
        out["slope_3_6"] = round(v6 - v3, 2) if (v3 and v6) else None
        rv = out.get("rvol_5d")
        out["iv_rv_spread"] = round(v1 - rv, 2) if (v1 and rv) else None

        if v1 and v3:
            if v3 > v1 + 0.5:
                out["term_structure_regime"] = "contango"   # normal — back IV > front
            elif v1 > v3 + 0.5:
                out["term_structure_regime"] = "backwardation"  # stress — near IV spike
            else:
                out["term_structure_regime"] = "flat"
        else:
            out["term_structure_regime"] = "unknown"

    except Exception as e:
        out["error"] = str(e)
    return out


def _make_heatmap(chain_df: pd.DataFrame, spot: float,
                  greek: str = "net_gex",
                  title: str = "GEX Heatmap",
                  height: int = 580,
                  use_volume: bool = False) -> go.Figure:
    """
    Strike × Expiry heatmap.
    Uses NUMERIC y-axis (actual strike prices) so shapes/lines work correctly.

    use_volume: if True and net_vol_gex column exists, use volume-based GEX
                instead of OI-based GEX (only applies when greek="net_gex").
    """
    from gex_engine import compute_gex_from_chain

    if chain_df is None or len(chain_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False,
                           xref="paper", yref="paper", font=dict(color="white", size=14))
        return fig

    gex_chain = compute_gex_from_chain(chain_df, spot)

    # Volume vs OI routing for GEX column
    _vol_gex_available = (
        "net_vol_gex" in gex_chain.columns and
        (gex_chain["net_vol_gex"].abs() > 0).any()
    )
    if greek == "net_gex" and use_volume and _vol_gex_available:
        greek_col  = "net_vol_gex"
        _flow_note = " [VOLUME]"
    else:
        greek_col  = {"net_gex": "net_gex", "net_vex": "net_vex", "net_cex": "net_cex"}.get(greek, "net_gex")
        _flow_note = " [OI]" if greek == "net_gex" else ""

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
    strike_lo = getattr(_make_heatmap, "_strike_lo", None)
    strike_hi = getattr(_make_heatmap, "_strike_hi", None)
    max_dte   = getattr(_make_heatmap, "_max_dte",    30)

    # If controls haven't set bounds yet, or spot is 0, derive from actual data
    if not strike_lo or not strike_hi or spot <= 0:
        all_strikes = pivot.index
        if len(all_strikes) > 0:
            mid = float(all_strikes[len(all_strikes)//2])  # median strike
            spot_ref = spot if spot > 0 else mid
            strike_lo = spot_ref - 20
            strike_hi = spot_ref + 20
        else:
            strike_lo, strike_hi = 0, 99999  # show everything if no data

    pct = max(abs(strike_hi - spot), abs(spot - strike_lo)) / max(spot, 1)
    pivot = pivot[(pivot.index >= strike_lo) & (pivot.index <= strike_hi)]

    def _dte(col):
        try:    return (dt.date.fromisoformat(col) - dt.date.today()).days
        except: return 999

    keep = [c for c in pivot.columns if _dte(c) <= max_dte]
    if keep:
        pivot = pivot[sorted(keep)]

    pivot = pivot.loc[(pivot.abs() >= 0.5).any(axis=1)]
    if pivot.empty:
        # Lower threshold and try again before giving up
        pivot_retry = (gex_chain
                       .groupby(["strike", "exp_label"])[greek_col]
                       .sum()
                       .unstack(fill_value=0)
                       / 1e6)
        pivot_retry.columns = pivot_retry.columns.get_level_values(0) if pivot_retry.columns.nlevels > 1 else pivot_retry.columns
        pivot_retry = pivot_retry[(pivot_retry.index >= strike_lo) & (pivot_retry.index <= strike_hi)]
        keep2 = [c for c in pivot_retry.columns if _dte(c) <= max_dte]
        if keep2:
            pivot_retry = pivot_retry[sorted(keep2)]
        pivot_retry = pivot_retry.loc[(pivot_retry.abs() > 0).any(axis=1)]
        if not pivot_retry.empty:
            pivot = pivot_retry  # use lower-threshold data
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

    # ── Colour — per-Greek colorscale ─────────────────────────────────────
    flat = [abs(v) for row in z_vals for v in row if abs(v) >= thresh]
    zmax = float(np.percentile(flat, 97)) if flat else 100.0

    if greek_col == "net_vex":
        # VEX (vanna): purple positive, red negative
        colorscale = [
            [0.00, "#4c1d95"], [0.30, "#7c3aed"],   # deep purple → violet (negative vanna)
            [0.47, "#1c1c1c"], [0.50, "#111111"], [0.53, "#1c1c1c"],
            [0.70, "#8b5cf6"], [1.00, "#c4b5fd"],   # medium → light purple (positive vanna)
        ]
    elif greek_col == "net_cex":
        # CEX (charm): teal positive, orange negative
        colorscale = [
            [0.00, "#7c2d12"], [0.30, "#f97316"],   # deep orange → orange (negative charm)
            [0.47, "#1c1c1c"], [0.50, "#111111"], [0.53, "#1c1c1c"],
            [0.70, "#06b6d4"], [1.00, "#0e7490"],   # teal → deep teal (positive charm)
        ]
    else:
        # GEX: red negative, green positive
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

    # ── Spot line — exact price on numeric y axis ─────────────────────────
    fig.add_shape(
        type="line",
        x0=-0.5, x1=n_cols - 0.5,
        y0=spot, y1=spot,
        xref="x", yref="y",
        line=dict(color="#06b6d4", width=2),
    )
    fig.add_annotation(
        x=-0.5, y=spot,
        text=f"▶ ${spot:.2f}",
        showarrow=False,
        xref="x", yref="y",
        xanchor="right",
        font=dict(color="#06b6d4", size=10, family="monospace"),
    )

    # ── Flip line — exact price, only for GEX heatmap ────────────────────
    _hm_flip = getattr(_make_heatmap, "_flip_level", None)
    if greek_col == "net_gex" and _hm_flip and np.isfinite(_hm_flip):
        y_min, y_max = min(strikes), max(strikes)
        if y_min <= _hm_flip <= y_max:
            fig.add_shape(
                type="line",
                x0=-0.5, x1=n_cols - 0.5,
                y0=_hm_flip, y1=_hm_flip,
                xref="x", yref="y",
                line=dict(color=_C_FLIP, width=2, dash="dash"),
            )
            fig.add_annotation(
                x=-0.5, y=_hm_flip,
                text=f"⚡ FLIP ${_hm_flip:.2f}",
                showarrow=False,
                xref="x", yref="y",
                xanchor="right",
                font=dict(color=_C_FLIP, size=10, family="monospace"),
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
            text=f"{title}{_flow_note} — {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Current Price: ${spot:.2f}",
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





def _cumulative_gex_chart(chain_df: pd.DataFrame, spot: float,
                           flip: float, max_dte: int = 45,
                           height: int = 320) -> go.Figure:
    """
    Cumulative GEX profile chart.
    Shows how total dealer gamma exposure accumulates from lowest to highest strike.
    Zero crossing = gamma flip. Slope = regime sensitivity at each level.
    """
    _C_POS  = "#10b981"
    _C_NEG  = "#ef4444"
    _C_FLIP = "#f59e0b"

    if chain_df is None or len(chain_df) == 0:
        fig = go.Figure()
        plotly_dark(fig, "Cumulative GEX Profile (no data)", height)
        return fig

    try:
        prof = compute_cumulative_gex_profile(chain_df, spot, max_dte)
        if prof.empty:
            raise ValueError("Empty profile")

        # Auto-scale
        max_abs = prof["cum_gex"].abs().max() / 1e6
        if max_abs >= 5000:
            scale, unit = 1e9, "$B"
        else:
            scale, unit = 1e6, "$M"

        pos_mask = prof["cum_gex"] >= 0
        neg_mask = ~pos_mask

        fig = go.Figure()

        # Filled area: positive (green)
        if pos_mask.any():
            fig.add_trace(go.Scatter(
                x=prof.loc[pos_mask, "strike"],
                y=prof.loc[pos_mask, "cum_gex"] / scale,
                fill="tozeroy",
                fillcolor="rgba(16,185,129,0.18)",
                line=dict(color=_C_POS, width=1.5),
                name="Positive (dealers pin)",
                showlegend=True,
            ))
        # Filled area: negative (red)
        if neg_mask.any():
            fig.add_trace(go.Scatter(
                x=prof.loc[neg_mask, "strike"],
                y=prof.loc[neg_mask, "cum_gex"] / scale,
                fill="tozeroy",
                fillcolor="rgba(239,68,68,0.18)",
                line=dict(color=_C_NEG, width=1.5),
                name="Negative (dealers amplify)",
                showlegend=True,
            ))

        # Zero line
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.30)", line_width=1)

        # Spot line
        fig.add_vline(x=spot, line_dash="dot",
                      line_color="rgba(255,255,255,0.70)", line_width=1.5,
                      annotation_text=f"SPOT ${spot:.0f}",
                      annotation_font_size=10,
                      annotation_position="top right")

        # Flip line
        x_range_lo = prof["strike"].min()
        x_range_hi = prof["strike"].max()
        if flip and x_range_lo < flip < x_range_hi:
            fig.add_vline(x=flip, line_dash="dash",
                          line_color=_C_FLIP, line_width=1.5,
                          annotation_text=f"FLIP ${flip:.0f}",
                          annotation_font_size=10,
                          annotation_font_color=_C_FLIP,
                          annotation_position="top left")

        # Min/max annotations
        min_idx = prof["cum_gex"].idxmin()
        max_idx = prof["cum_gex"].idxmax()
        for idx, label, col in [(min_idx, "MAX NEG GEX", _C_NEG),
                                 (max_idx, "MAX PIN",  _C_POS)]:
            fig.add_annotation(
                x=float(prof.loc[idx, "strike"]),
                y=float(prof.loc[idx, "cum_gex"] / scale),
                text=f"{label}<br>${prof.loc[idx,'strike']:.0f}",
                showarrow=True, arrowhead=2, arrowcolor=col,
                font=dict(size=9, color=col),
                bgcolor="rgba(0,0,0,0.6)", bordercolor=col,
            )

        fig.update_layout(
            xaxis_title="Strike",
            yaxis_title=f"Cumulative Net GEX ({unit})",
            showlegend=True,
            legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
        )
        plotly_dark(fig, f"Cumulative GEX Profile — ≤{max_dte}DTE", height)
        return fig

    except Exception as exc:
        fig = go.Figure()
        plotly_dark(fig, f"Cumulative GEX Profile (error: {exc})", height)
        return fig

def _greek_bar_chart(by_strike: dict, spot: float, title: str,
                     pos_color: str, neg_color: str,
                     flip_level: float = None, height=340) -> go.Figure:
    """Horizontal bar chart matching GEXBot layout: strike on Y-axis, GEX on X-axis."""
    strikes = sorted(by_strike.keys(), reverse=True)   # high strike at top (GEXBot style)
    near    = [s for s in strikes if spot * 0.90 < s < spot * 1.10]
    if not near:
        near = strikes

    # Auto-scale: $B if any value exceeds $5B, else $M
    raw_vals = [by_strike[s] / 1e6 for s in near]
    max_abs  = max((abs(v) for v in raw_vals), default=1)
    if max_abs >= 5000:
        scale, unit = 1000.0, "$B"
    else:
        scale, unit = 1.0, "$M"
    vals   = [v / scale for v in raw_vals]
    colors = [pos_color if v > 0 else neg_color for v in vals]

    # Strike labels — format as "$587"
    y_labels = [f"${s:.0f}" for s in near]

    fig = go.Figure(go.Bar(
        x=vals,
        y=y_labels,
        orientation="h",          # horizontal — matches GEXBot
        marker_color=colors,
        marker_line_width=0,
        opacity=0.88,
        name=title,
    ))

    # Zero line (vertical for horizontal bars)
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.35)", line_width=1)

    # Spot and flip lines — interpolated position on categorical axis
    # near is sorted HIGH→LOW; paper y: 0=bottom, 1=top
    hi_s = near[0] if near else spot
    lo_s = near[-1] if near else spot

    def _cat_y(target_price: float) -> float:
        if hi_s == lo_s: return 0.5
        frac = (target_price - lo_s) / (hi_s - lo_s)
        return float(np.clip(frac, 0.0, 1.0))

    spot_y = _cat_y(spot)
    fig.add_shape(type="line", xref="paper", yref="paper",
                  x0=0, x1=1, y0=spot_y, y1=spot_y,
                  line=dict(dash="dot", color="rgba(255,255,255,0.70)", width=1.5))
    fig.add_annotation(xref="paper", yref="paper",
                       x=1.01, y=spot_y,
                       text=f"SPOT ${spot:.2f}",
                       showarrow=False, xanchor="left",
                       font=dict(size=10, color="rgba(255,255,255,0.85)"))

    # Flip line
    if flip_level and np.isfinite(flip_level) and abs(flip_level - spot) / max(spot, 1) < 0.20:
        flip_y = _cat_y(flip_level)
        fig.add_shape(type="line", xref="paper", yref="paper",
                      x0=0, x1=1, y0=flip_y, y1=flip_y,
                      line=dict(dash="dash", color=_C_FLIP, width=2.0))
        fig.add_annotation(xref="paper", yref="paper",
                           x=1.01, y=flip_y,
                           text=f"⚡ FLIP ${flip_level:.2f}",
                           showarrow=False, xanchor="left",
                           font=dict(size=10, color=_C_FLIP))

    # Y-axis: show all strike labels, tight range
    fig.update_layout(
        xaxis_title=f"Net GEX ({unit})",
        yaxis=dict(
            title="Strike",
            categoryorder="array",
            categoryarray=y_labels,   # maintains high-to-low order
            tickfont=dict(size=10),
        ),
        bargap=0.12,
        showlegend=False,
    )
    return plotly_dark(fig, title, height)


def _two_sided_gex_chart(chain_df: pd.DataFrame, spot: float,
                          flip_level: float = None,
                          use_volume: bool = True,
                          max_dte: int = 1,
                          height: int = 420,
                          strike_range_pct: float = 0.05) -> go.Figure:
    """
    Two-sided GEX profile (Doc 2 standard):
      - Calls (OTM: K >= spot) → positive bars on the RIGHT
      - Puts  (OTM: K <= spot) → negative bars on the LEFT
      - Volume-based when available, OI fallback otherwise
      - OTM-only filtering (removes ITM distortion)
      - Strikes within ±strike_range_pct of spot

    This matches the GEXBot / SpotGamma intraday 0DTE flow chart.
    """
    if chain_df is None or len(chain_df) == 0:
        fig = go.Figure()
        plotly_dark(fig, "0DTE GEX Profile (no data)", height)
        return fig

    from gex_engine import compute_gex_from_chain

    # ── Filter: 0DTE (or nearest expiry ≤ max_dte) ────────────────────────
    near = chain_df[chain_df["expiry_T"] <= max_dte / 365.0].copy()
    if near.empty:
        # fallback: just the nearest expiry
        min_T = chain_df["expiry_T"].min()
        near  = chain_df[chain_df["expiry_T"] <= min_T + 1 / 365.0].copy()
    actual_dte = int(round(near["expiry_T"].min() * 365)) if not near.empty else 0
    dte_label  = "0DTE" if actual_dte <= 1 else f"{actual_dte}DTE"

    gex_chain = compute_gex_from_chain(near, spot)

    # ── Decide OI vs volume ───────────────────────────────────────────────
    has_vol = ("call_volume" in gex_chain.columns and
               (gex_chain["call_volume"].fillna(0) > 0).any() and
               "put_volume" in gex_chain.columns and
               (gex_chain["put_volume"].fillna(0) > 0).any())
    using_volume = use_volume and has_vol
    flow_label   = "VOLUME" if using_volume else "OI"

    if using_volume:
        call_gex_col = "call_vol_gex"
        put_gex_col  = "put_vol_gex"
    else:
        call_gex_col = "call_gex"
        put_gex_col  = "put_gex"

    # Aggregate across expirations (if multiple) by strike
    agg = (gex_chain.groupby("strike")
                    .agg(**{
                        "call_gex_val": (call_gex_col, "sum"),
                        "put_gex_val":  (put_gex_col,  "sum"),
                    })
                    .reset_index())

    # ── OTM filter: calls K≥spot, puts K≤spot (Doc 2 §4 & §5) ────────────
    call_side = agg[agg["strike"] >= spot].copy()
    put_side  = agg[agg["strike"] <= spot].copy()

    # ── Strike range filter: ±strike_range_pct of spot ────────────────────
    lo = spot * (1 - strike_range_pct)
    hi = spot * (1 + strike_range_pct)
    call_side = call_side[call_side["strike"] <= hi]
    put_side  = put_side[put_side["strike"]  >= lo]

    # Merge into single strike set for unified Y-axis
    all_strikes = sorted(set(call_side["strike"]).union(set(put_side["strike"])), reverse=True)
    if not all_strikes:
        fig = go.Figure()
        plotly_dark(fig, f"NET {dte_label} GEX ({flow_label}) — no strikes in range", height)
        return fig

    call_map = dict(zip(call_side["strike"], call_side["call_gex_val"]))
    put_map  = dict(zip(put_side["strike"],  put_side["put_gex_val"]))

    # Per-strike values ($M): calls positive, puts negative
    call_vals = [call_map.get(s, 0.0) / 1e6 for s in all_strikes]
    put_vals  = [put_map.get(s,  0.0) / 1e6 for s in all_strikes]
    y_labels  = [f"${s:.0f}" for s in all_strikes]

    # ── Auto-scale: $B if any bar exceeds $1B ─────────────────────────────
    max_abs = max((abs(v) for v in call_vals + put_vals), default=1.0)
    if max_abs >= 1000:
        call_vals = [v / 1000 for v in call_vals]
        put_vals  = [v / 1000 for v in put_vals]
        unit = "$B"
    else:
        unit = "$M"

    fig = go.Figure()

    # Call bars (positive, right side) — green
    fig.add_trace(go.Bar(
        x=call_vals,
        y=y_labels,
        orientation="h",
        name=f"Call GEX ({flow_label})",
        marker_color=[_C_POS if v > 0 else _C_NEG for v in call_vals],
        marker_line_width=0,
        opacity=0.88,
    ))

    # Put bars (negative, left side) — red
    fig.add_trace(go.Bar(
        x=put_vals,
        y=y_labels,
        orientation="h",
        name=f"Put GEX ({flow_label})",
        marker_color=[_C_NEG if v < 0 else _C_POS for v in put_vals],
        marker_line_width=0,
        opacity=0.88,
    ))

    # Zero line
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.40)", line_width=1.5)

    # Spot and flip lines.
    # Categorical Y-axis maps strike labels to integer positions 0..n-1.
    # all_strikes is sorted HIGH→LOW, so index 0 = top of chart.
    # Plotly paper coords: 0=bottom, 1=top → invert the index fraction.
    n_cats = len(all_strikes)

    def _cat_y(target_price: float) -> float:
        """
        Interpolated paper-fraction for target_price on the categorical axis.
        Handles prices that fall between two listed strikes correctly.
        """
        if n_cats <= 1:
            return 0.5
        # all_strikes sorted high→low
        hi_s = all_strikes[0]
        lo_s = all_strikes[-1]
        if hi_s == lo_s:
            return 0.5
        # Linear interpolation within [lo_s, hi_s]
        # price at hi_s → paper y = 1.0, price at lo_s → paper y = 0.0
        frac = (target_price - lo_s) / (hi_s - lo_s)
        return float(np.clip(frac, 0.0, 1.0))

    spot_y = _cat_y(spot)
    fig.add_shape(type="line", xref="paper", yref="paper",
                  x0=0, x1=1, y0=spot_y, y1=spot_y,
                  line=dict(dash="dot", color="rgba(255,255,255,0.75)", width=1.5))
    fig.add_annotation(xref="paper", yref="paper",
                       x=1.01, y=spot_y, text=f"SPOT ${spot:.2f}",
                       showarrow=False, xanchor="left",
                       font=dict(size=10, color="rgba(255,255,255,0.85)"))

    # Flip line — exact interpolated position
    if flip_level and np.isfinite(flip_level):
        flip_y = _cat_y(flip_level)
        fig.add_shape(type="line", xref="paper", yref="paper",
                      x0=0, x1=1, y0=flip_y, y1=flip_y,
                      line=dict(dash="dash", color=_C_FLIP, width=2.0))
        fig.add_annotation(xref="paper", yref="paper",
                           x=1.01, y=flip_y, text=f"⚡ FLIP ${flip_level:.2f}",
                           showarrow=False, xanchor="left",
                           font=dict(size=10, color=_C_FLIP, family="monospace"))

    fig.update_layout(
        barmode="overlay",
        xaxis_title=f"GEX ({unit}, per 1% S move × 0.01)",
        yaxis=dict(
            title="Strike",
            categoryorder="array",
            categoryarray=y_labels,
            tickfont=dict(size=10),
        ),
        bargap=0.10,
        showlegend=True,
        legend=dict(orientation="h", y=1.04, x=0, font=dict(size=10)),
    )
    source_note = "VOLUME (intraday flow)" if using_volume else "OI (positional)"
    return plotly_dark(fig, f"NET {dte_label} GEX — {source_note} — OTM Only", height)


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

    # ── Initialise ALL persistent settings once — never overwrite existing values ──
    # These survive navigation between pages because session_state persists for the
    # entire browser session. We only set defaults when the key is absent.
    _DEFAULTS = {
        "gex_symbol_input":     "",
        # gex_use_schwab removed — Schwab is always primary
        "gex_view_mode":        "Heatmap",
        "gex_auto_refresh":     True,
        "gex_refresh_interval": "2m",
        "gex_strikes_each_side": 20,
        "gex_hm_dte":           30,
        "gex_hm_height":        1000,
    }
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar refresh controls ──────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ GEX Refresh")
    auto_refresh = st.sidebar.toggle("Auto refresh", key="gex_auto_refresh")

    _INTERVALS = {"30s": 30, "1m": 60, "2m": 120, "5m": 300, "10m": 600, "15m": 900, "30m": 1800}
    refresh_label = st.sidebar.selectbox(
        "Refresh interval", options=list(_INTERVALS.keys()),
        key="gex_refresh_interval", disabled=not auto_refresh,
    )
    refresh_sec = _INTERVALS[refresh_label]

    if st.sidebar.button("🔄 Refresh now", use_container_width=True, key="gex_manual_refresh"):
        st.cache_data.clear()
        st.session_state["_autorefresh_last"] = __import__("time").time()
        st.rerun()

    if auto_refresh:
        st.sidebar.caption(f"🟢 Auto-refreshing every {refresh_label}")
        st.sidebar.caption(
            "ℹ️ Schwab: live IV + spot on every cycle. "
            "OI updates once/day (OCC). "
            "Bar heights refresh each morning when OCC publishes."
        )
    else:
        st.sidebar.caption("⚫ Auto-refresh off")

    autorefresh_js(refresh_sec, auto_refresh)
    st.sidebar.markdown("---")

    col_s, col_m = st.columns([1, 3])
    with col_s:
        # No 'value=' arg — Streamlit reads from session_state[key] automatically
        symbol = st.text_input("Options Symbol", key="gex_symbol_input", placeholder="e.g. SPY").strip().upper()

    if not symbol:
        st.info("Enter an options symbol above to load GEX data.")
        return

    # ── Data fetch — Schwab/TOS only (yfinance removed as OI fallback) ──────
    # Use a per-symbol spot key so switching tickers never bleeds the wrong price
    _spot_key = f"_last_known_spot_{symbol}" if symbol else "_last_known_spot"
    chain_df, spot, source = None, float(st.session_state.get(_spot_key, 500.0)), "unknown"
    with col_m:
        client = get_schwab_client()
        if client:
            chain_df = schwab_get_options_chain(client, symbol, spot=None)
            source   = "Schwab API (live IV)"
            if chain_df is not None and len(chain_df) > 0:
                spot_live = schwab_get_spot(client, symbol)
                if spot_live and spot_live > 0:
                    spot = spot_live
                else:
                    last_known = float(st.session_state.get(_spot_key, 0))
                    spot = last_known if last_known > 0 else float(chain_df["strike"].median())
                st.success(f"Schwab connected — live IV · {symbol} · ${spot:.2f}")
            else:
                err = st.session_state.get("_schwab_chain_error", "unknown error")
                st.warning(
                    f"⚠️ Schwab returned empty chain for **{symbol}** ({err}). "
                    "Falling back to yfinance OI data."
                )
                source = "Schwab API (empty chain)"
        if chain_df is None or len(chain_df) == 0:
            # yfinance fallback — less data than Schwab but better than a dead end
            chain_df, spot_yf, source = get_gex_from_yfinance(symbol)
            if chain_df is not None and len(chain_df) > 0:
                spot = spot_yf or spot
                if not client:
                    st.info(f"📊 Using yfinance OI data for **{symbol}** — connect Schwab/TOS for live IV and volume.")
            else:
                if not client:
                    st.warning(
                        "🔌 **Schwab/TOS not connected** and yfinance returned no data. "
                        "Go to the **Schwab/TOS** tab to authorise, or check your connection."
                    )

    # Save last known good spot keyed per symbol so switching tickers never bleeds price
    if spot and spot > 0:
        st.session_state[_spot_key] = spot

    if chain_df is None or len(chain_df) == 0:
        st.error(f"No options data available for **{symbol}**. Source: `{source}`")
        with st.expander("🔧 Debug Info", expanded=True):
            st.write(f"**Symbol:** {symbol}")
            st.write(f"**chain_df:** {'None' if chain_df is None else f'Empty DataFrame ({len(chain_df)} rows)'}")
            st.write(f"**spot:** {spot}")
            st.write(f"**source:** {source}")
            st.write("**OI sources tried:** Schwab API → yfinance fallback")
            schwab_err = st.session_state.get("_schwab_chain_error", "none")
            schwab_dbg = st.session_state.get("_schwab_chain_debug", "none")
            st.write(f"**Schwab chain error:** {schwab_err}")
            st.write(f"**Schwab chain debug:** {schwab_dbg}")
        return

    # ── Compute all Greeks — cached so widget interactions don't recompute ──
    # st.cache_data caches by (chain hash, spot, source, max_dte).
    # The cache is cleared on manual refresh or auto-refresh cycle.
    _max_dte = int(st.session_state.get('gex_hm_dte', 45))

    @st.cache_data(ttl=120, show_spinner=False)
    def _cached_gamma_state(_chain_json, _spot, _source, _max_dte):
        import pandas as _pd, io as _io
        _df = _pd.read_json(_io.StringIO(_chain_json))
        return build_gamma_state(_df, _spot, _source, max_dte=_max_dte)

    @st.cache_data(ttl=120, show_spinner=False)
    def _cached_dealer_greeks(_chain_json, _spot, _source, _max_dte):
        import pandas as _pd, io as _io
        _df = _pd.read_json(_io.StringIO(_chain_json))
        return compute_dealer_greeks(_df, _spot, _source, max_dte=_max_dte)

    @st.cache_data(ttl=120, show_spinner=False)
    def _cached_analytics(_chain_json, _spot):
        import pandas as _pd, io as _io
        _df = _pd.read_json(_io.StringIO(_chain_json))
        return (compute_gwas(_df, _spot),
                compute_gex_term_structure(_df, _spot),
                compute_flow_imbalance(_df, _spot))

    try:
        _chain_json = chain_df.to_json()
        gs   = _cached_gamma_state(_chain_json, spot, source, _max_dte)
        dg   = _cached_dealer_greeks(_chain_json, spot, source, _max_dte)
        gwas, term_str, flow = _cached_analytics(_chain_json, spot)
    except Exception:
        # Fallback: compute directly if serialization fails
        gs   = build_gamma_state(chain_df, spot, source, max_dte=_max_dte)
        dg   = compute_dealer_greeks(chain_df, spot, source, max_dte=_max_dte)
        gwas     = compute_gwas(chain_df, spot)
        term_str = compute_gex_term_structure(chain_df, spot)
        flow     = compute_flow_imbalance(chain_df, spot)

    session   = get_session_context()
    vix_df    = yf.Ticker("^VIX").history(period="1d")
    vix_level = float(vix_df["Close"].iloc[-1]) if len(vix_df) > 0 else 20.0
    rvol      = _fetch_rvol_surface(symbol)

    # ── Header ────────────────────────────────────────────────────────────
    _flip_vol = (("call_volume" in chain_df.columns and chain_df["call_volume"].fillna(0).sum() > 0) and
                 ("put_volume"  in chain_df.columns and chain_df["put_volume"].fillna(0).sum()  > 0))
    _flip_src = "vol-flow" if _flip_vol else "OI"

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Spot",       f"{spot:.2f}")
    m2.metric(f"Gamma Flip ({_flip_src})",
              f"{gs.gamma_flip:.2f}" if gs.gamma_flip else "N/A",
              f"{gs.distance_to_flip_pct:+.2f}%")
    m3.metric("Net GEX",   f"${gs.total_gex/1e6:.1f}M")
    m4.metric("Vanna Dir",  dg.vanna_direction.upper())
    m5.metric("Charm Dir",  dg.charm_direction.upper())

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

    tab_gex, tab_vex, tab_cex, tab_nodes, tab_otm, tab_rvol, tab_flow, tab_gwas, tab_durability = st.tabs(
        ["📊 GEX (Gamma)", "🌀 VEX (Vanna)", "⏱ CEX (Charm)", "🎯 Key Nodes", "🔭 OTM Anchors",
         "📈 Vol Surface", "💸 Flow Imbalance", "🧲 GWAS", "⏳ GEX Duration"]
    )

    with tab_gex:
        st.markdown(f"{sec_hdr('GAMMA EXPOSURE — Reaction to Price')}", unsafe_allow_html=True)

        view_mode = st.radio("View", ["Heatmap", "Bar Chart"], horizontal=True, key="gex_view_mode")

        if view_mode == "Heatmap":
            st.caption("Strike × Expiry matrix · Green = positive GEX (dealers stabilize) · Red = negative GEX (dealers amplify) · → = spot price · TOTAL = net across all expiries")

            sc1, sc2, sc3, sc4 = st.columns(4)
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
            with sc4:
                hm_use_volume = st.toggle(
                    "Use Volume (flow)",
                    value=False,
                    key="gex_hm_use_volume",
                    help="ON = volume-weighted intraday flow. OFF = OI-based structural positioning.",
                )

            strike_lo = float(spot - int(strikes_each_side))
            strike_hi = float(spot + int(strikes_each_side))

            _make_heatmap._strike_lo = strike_lo
            _make_heatmap._strike_hi = strike_hi
            _make_heatmap._max_dte   = int(max_dte)
            _make_heatmap._flip_level = float(gs.gamma_flip) if gs.gamma_flip else float("nan")

            src_label = "VOLUME (intraday flow)" if hm_use_volume else "OI (structural)"
            st.caption(
                f"Showing strikes ${strike_lo:.0f} – ${strike_hi:.0f} "
                f"({int(strikes_each_side)} each side of spot ${spot:.2f}) · "
                f"Max DTE: {int(max_dte)}d · Source: {src_label}"
            )

            fig_gex = _make_heatmap(chain_df, spot, "net_gex",
                                    f"{symbol} GEX", int(hm_height),
                                    use_volume=hm_use_volume)
            st.plotly_chart(fig_gex, use_container_width=True, key="gex_chart_heatmap")
        else:
            # ── Use EXACT same chain slice as heatmap ─────────────────────
            _hm_lo  = getattr(_make_heatmap, "_strike_lo", spot - int(st.session_state.get("gex_strikes_each_side", 20)))
            _hm_hi  = getattr(_make_heatmap, "_strike_hi", spot + int(st.session_state.get("gex_strikes_each_side", 20)))
            _hm_dte = int(getattr(_make_heatmap, "_max_dte", st.session_state.get("gex_hm_dte", 30)))

            # ── Sub-controls for bar chart ────────────────────────────────
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                use_vol_toggle = st.toggle(
                    "Use Volume (0DTE flow)", value=True, key="gex_bar_use_volume",
                    help="ON = volume-weighted intraday flow (real-time). OFF = OI-weighted positional inventory."
                )
            with bc2:
                range_pct = st.slider(
                    "Strike range (±% of spot)", min_value=1, max_value=15, value=5,
                    key="gex_bar_range_pct",
                    help="Controls how many strikes are shown around spot."
                ) / 100.0
            with bc3:
                near_dte = st.number_input(
                    "0DTE max DTE", min_value=0, max_value=7, value=1,
                    key="gex_bar_near_dte",
                    help="0=0DTE only, 1=today+tomorrow, up to 7 for weekly"
                )

            # ── PRIMARY: Two-sided 0DTE flow chart (Doc 2 standard) ──────
            st.markdown(
                "<div style='font-size:10px;color:rgba(255,255,255,0.5);margin:4px 0 2px;'>"
                "🔴🟢 NET 0DTE GEX — OTM calls (right) vs OTM puts (left) — volume = real-time dealer flow</div>",
                unsafe_allow_html=True
            )
            # Compute flip directly from the same chain slice each chart uses
            # so the line sits exactly at the red→green boundary in the bars
            _near_chain_0dte = chain_df[chain_df["expiry_T"] <= int(near_dte) / 365.0].copy()
            if _near_chain_0dte.empty:
                min_T = chain_df["expiry_T"].min()
                _near_chain_0dte = chain_df[chain_df["expiry_T"] <= min_T + 1/365.0].copy()
            _flip_0dte = gex_zero_crossing(_near_chain_0dte, spot, max_dte=int(near_dte))
            if _flip_0dte is None:
                _flip_0dte = gs.gamma_flip  # fallback to vol-trigger

            fig_twosided = _two_sided_gex_chart(
                chain_df, spot,
                flip_level=_flip_0dte,
                use_volume=use_vol_toggle,
                max_dte=int(near_dte),
                height=int(st.session_state["gex_hm_height"] // 2),
                strike_range_pct=range_pct,
            )
            st.plotly_chart(fig_twosided, use_container_width=True, key="gex_chart_bar_twosided")

            flow_src = "volume" if use_vol_toggle else "OI"
            flip_disp = f"${_flip_0dte:.2f}" if _flip_0dte else "N/A"
            st.caption(
                f"Calls OTM (K≥spot) positive · Puts OTM (K≤spot) negative · "
                f"{flow_src}-weighted · ≤{near_dte}DTE · ±{range_pct*100:.0f}% strike range · "
                f"Flip: {flip_disp} (net GEX = 0 crossing)"
            )

            st.markdown("---")

            # ── SECONDARY: All-expiry net GEX bar (structural view) ──────
            _bar_chain = chain_df[chain_df["expiry_T"] <= _hm_dte / 365.0].copy()
            if _bar_chain.empty:
                _bar_chain = chain_df.copy()

            _bar_gex = compute_gex_from_chain(_bar_chain, spot)
            _bar_agg = (_bar_gex.groupby("strike")["net_gex"].sum().reset_index())
            _bar_agg = _bar_agg[(_bar_agg["strike"] >= _hm_lo) & (_bar_agg["strike"] <= _hm_hi)]
            filtered = dict(zip(_bar_agg["strike"], _bar_agg["net_gex"]))

            # Flip for the all-expiry bar — computed from same chain slice
            _flip_bar = gex_zero_crossing(_bar_chain, spot, max_dte=_hm_dte)
            if _flip_bar is None:
                _flip_bar = gs.gamma_flip

            neg_frac = sum(1 for v in filtered.values() if v < 0) / max(len(filtered), 1)
            if neg_frac > 0.8:
                regime_note = f"⚠ {neg_frac*100:.0f}% of strikes negative — heavy put positioning."
            elif neg_frac < 0.2:
                regime_note = f"✓ {(1-neg_frac)*100:.0f}% of strikes positive — dealers PIN price."
            else:
                regime_note = f"Mixed: {(1-neg_frac)*100:.0f}% pos / {neg_frac*100:.0f}% neg."

            st.markdown(
                "<div style='font-size:10px;color:rgba(255,255,255,0.4);margin:8px 0 2px;'>"
                f"ALL-EXPIRY NET GEX (OI-based structural view) — ≤{_hm_dte}DTE · {regime_note}</div>",
                unsafe_allow_html=True
            )
            regime_str = REGIME_OPERATIONAL_LABEL.get(gs.regime, gs.regime.value)
            fig_gex = _greek_bar_chart(filtered, spot,
                                       f"Net GEX (OI) · {regime_str} · ≤{_hm_dte}DTE",
                                       _C_POS, _C_NEG, _flip_bar,
                                       height=int(st.session_state["gex_hm_height"] // 2))
            st.plotly_chart(fig_gex, use_container_width=True, key="gex_chart_bar")

            st.markdown(
                "<div style='font-size:10px;color:rgba(255,255,255,0.4);margin:8px 0 2px;'>"
                "CUMULATIVE GEX PROFILE — zero crossing = gamma flip</div>",
                unsafe_allow_html=True
            )
            fig_cum = _cumulative_gex_chart(
                _bar_chain, spot, _flip_bar,
                max_dte=_hm_dte,
                height=int(st.session_state["gex_hm_height"] // 2),
            )
            st.plotly_chart(fig_cum, use_container_width=True, key="gex_chart_cumulative")

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
- **Green (positive)**: Long gamma — dealers buy dips, sell rips → mean reversion, range compression
- **Red (negative)**: Short gamma — dealers amplify moves → trend days, gap-and-go, cascades
- **Gamma Flip**: Price where net GEX crosses zero. Spot **above** flip = positive gamma (dealers stabilize). Spot **below** flip = negative gamma (dealers amplify). Not the other way around.
""")

    with tab_vex:
        st.markdown(f"{sec_hdr('VANNA EXPOSURE (VEX) — Reaction to IV Changes')}", unsafe_allow_html=True)
        st.caption(
            "VEX = OI × Vanna × 100. Vanna = dDelta/dIV = -φ(d1)·d2/σ. "
            "Positive vanna: IV rises → dealers BUY. Negative: IV rises → dealers SELL."
        )
        if "not connected" in source.lower() or "empty chain" in source.lower():
            st.warning("⚠️ VEX requires a live Schwab/TOS connection for per-strike IV and OI data.")

        view_mode_vex = st.radio("View", ["Heatmap", "Bar Chart"], horizontal=True, key="vex_view_mode")
        if view_mode_vex == "Heatmap":
            st.caption("Strike × Expiry matrix · Purple/green = positive vanna · Red = negative vanna · TOTAL = net across all expiries")
            fig_vex = _make_heatmap(chain_df, spot, "net_vex", f"{symbol} VEX (Vanna)", int(st.session_state.get("gex_hm_height", 1000)))
            st.plotly_chart(fig_vex, use_container_width=True, key="gex_chart_vex_heatmap")
        else:
            fig_vex = _greek_bar_chart(dg.vex_by_strike, spot,
                                       "Net VEX (Vanna) by Strike", _C_VEX, _C_NEG, gs.gamma_flip)
            st.plotly_chart(fig_vex, use_container_width=True, key="gex_chart_vex_bar")

        ntm_vex_val = sum(v for k, v in dg.vex_by_strike.items() if abs(k - spot) / spot < 0.02)
        v1, v2 = st.columns(2)
        with v1:
            st.metric("Net VEX (Vanna) near spot", f"${ntm_vex_val / 1e6:.1f}M")
            st.caption("VEX = OI × vanna × 100  where  vanna = -φ(d1)·d2/σ")
        with v2:
            st.markdown("""
**VEX = Vanna Exposure**
- Large positive VEX → positive vanna dominant → IV drop → dealers BUY
- Large negative VEX → negative vanna dominant → IV spike → dealers SELL
""")

        ntm_vex_val = sum(v for k, v in dg.vex_by_strike.items() if abs(k - spot) / spot < 0.02)
        va1, va2 = st.columns(2)
        with va1:
            st.metric("Net VEX (Vanna) near spot", f"${ntm_vex_val / 1e6:.1f}M")
            st.markdown(f"**Vanna Sign:** {dg.vanna_sign.upper()}")
        with va2:
            if dg.vanna_sign == "positive":
                st.markdown("""
**Positive Vanna near spot:**
- 📈 **IV rises** → dealers forced to **BUY** (bullish)
- 📉 **IV falls** → dealers forced to **SELL** (bearish)
- → **Vanna Squeeze**: IV elevated + compress → dealer buying → upside break
""")
            elif dg.vanna_sign == "negative":
                st.markdown("""
**Negative Vanna near spot:**
- 📈 **IV rises** → dealers forced to **SELL** (bearish, amplifies selloff)
- 📉 **IV falls** → dealers forced to **BUY** (bullish on vol crush)
- → **Vanna Rug risk**: catalyst spikes IV → dealers sell → cascade
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
        if "not connected" in source.lower() or "empty chain" in source.lower():
            st.warning("⚠️ CEX requires a live Schwab/TOS connection for live 0DTE positioning and intraday drift signals.")

        view_mode_cex = st.radio("View", ["Heatmap", "Bar Chart"], horizontal=True, key="cex_view_mode")
        if view_mode_cex == "Heatmap":
            st.caption("Strike × Expiry matrix · Teal/green = positive charm (dealers buy as time passes) · Red = negative · TOTAL = net")
            fig_cex = _make_heatmap(chain_df, spot, "net_cex", f"{symbol} CEX", int(st.session_state.get("gex_hm_height", 1000)))
            st.plotly_chart(fig_cex, use_container_width=True, key="gex_chart_cex_heatmap")
        else:
            fig_cex = _greek_bar_chart(dg.cex_by_strike, spot,
                                       "Net CEX by Strike ($M)", _C_CEX, _C_NEG, gs.gamma_flip)
            st.plotly_chart(fig_cex, use_container_width=True, key="gex_chart_cex_bar")

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

    # ── Vol Surface Tab ───────────────────────────────────────────────────
    with tab_rvol:
        st.markdown(f"{sec_hdr('REALIZED vs IMPLIED VOL SURFACE')}", unsafe_allow_html=True)
        st.caption("VIX term structure slope reveals where the options market expects volatility to resolve. "
                   "Near-term IV crushed + elevated back end = different regime than uniform compression.")

        v1  = rvol.get("vix1m")
        v3  = rvol.get("vix3m")
        v6  = rvol.get("vix6m")
        rv5 = rvol.get("rvol_5d")
        s13 = rvol.get("slope_1_3")
        s36 = rvol.get("slope_3_6")
        ivr = rvol.get("iv_rv_spread")
        ts_regime = rvol.get("term_structure_regime", "unknown")

        rv1, rv2, rv3, rv4 = st.columns(4)
        rv1.metric("VIX (1M IV)",  f"{v1:.1f}" if v1 else "N/A")
        rv2.metric("VIX3M",        f"{v3:.1f}" if v3 else "N/A",
                   delta=f"slope: {s13:+.1f}" if s13 is not None else None)
        rv3.metric("VIX6M",        f"{v6:.1f}" if v6 else "N/A",
                   delta=f"slope: {s36:+.1f}" if s36 is not None else None)
        rv4.metric("5D RVol",      f"{rv5:.1f}%" if rv5 else "N/A",
                   delta=f"IV-RV: {ivr:+.1f}" if ivr is not None else None)

        # Term structure chart
        if v1 and v3 and v6:
            import plotly.graph_objects as _go
            fig_ts = _go.Figure()
            fig_ts.add_trace(_go.Scatter(
                x=["1M (VIX)", "3M (VIX3M)", "6M (VIX6M)"],
                y=[v1, v3, v6],
                mode="lines+markers+text",
                text=[f"{v1:.1f}", f"{v3:.1f}", f"{v6:.1f}"],
                textposition="top center",
                line=dict(color="#6366f1", width=2),
                marker=dict(size=10),
                name="IV Term Structure",
            ))
            if rv5:
                fig_ts.add_hline(y=rv5, line_dash="dash", line_color="#10b981",
                                 annotation_text=f"5D RVol: {rv5:.1f}%",
                                 annotation_position="bottom right")
            plotly_dark(fig_ts, title="VIX Term Structure vs Realized Vol", height=350)
            fig_ts.update_layout(yaxis_title="Volatility (%)")
            st.plotly_chart(fig_ts, use_container_width=True, key="gex_rvol_chart")

        # Regime interpretation
        if ts_regime == "backwardation":
            color, border = "rgba(239,68,68,0.08)", "rgba(239,68,68,0.30)"
            msg = ("🔴 **BACKWARDATION** — Near-term IV > 3M IV. "
                   "Options market pricing acute near-term stress. "
                   "This is NOT uniform vol compression — gamma risks are front-loaded. "
                   "GEX levels may shift rapidly as front-month OI dominates.")
        elif ts_regime == "contango":
            color, border = "rgba(16,185,129,0.08)", "rgba(16,185,129,0.30)"
            msg = ("🟢 **CONTANGO** — Back IV > Front IV. "
                   "Normal structure. Market pricing mean-reversion toward calmer conditions. "
                   "Near-term IV compression likely → positive vanna tailwind if sustained.")
        else:
            color, border = "rgba(255,255,255,0.04)", "rgba(255,255,255,0.12)"
            msg = "⚪ **FLAT** — IV roughly equal across term. No strong directional vol regime signal."

        st.markdown(f"<div class='panel' style='background:{color};border-color:{border};margin-top:10px;'>{msg}</div>",
                    unsafe_allow_html=True)

        if ivr is not None:
            if ivr > 5:
                st.markdown(f"**IV Premium:** {ivr:+.1f} pts above realized → options expensive, vol sellers favoured, mean-reversion environment.")
            elif ivr < -3:
                st.markdown(f"**IV Discount:** {ivr:+.1f} pts below realized → options cheap, realized vol outrunning implied → vol buyers favoured, trending environment.")
            else:
                st.markdown(f"**IV-RV Spread:** {ivr:+.1f} pts — near fair value. No strong structural vol edge.")

        st.markdown("""
**Reading the Term Structure:**
- **Steep contango** (VIX3M >> VIX): market expects current vol to subside → favours positive vanna setups, IV-compression trades
- **Backwardation** (VIX > VIX3M): acute front-end stress; GEX levels more volatile, dealer hedging accelerates near-term
- **Flat + IV >> RVol**: options overpriced → sell premium, fade vol spikes
- **Flat + RVol >> IV**: market underpricing actual moves → buy protection, reduce short-gamma exposure
""")

    # ── Flow Imbalance Tab ────────────────────────────────────────────────
    with tab_flow:
        st.markdown(f"{sec_hdr('OPTIONS FLOW IMBALANCE — Put/Call Dollar Premium')}", unsafe_allow_html=True)
        st.caption("Dollar premium volume = what's happening NOW. OI is yesterday's snapshot. "
                   "Heavy put dollar flow = active hedging / fear. Call dominance = speculation / complacency.")

        if flow:
            pv  = flow.get("put_dollar_vol", 0)
            cv  = flow.get("call_dollar_vol", 0)
            pcr = flow.get("pc_ratio", 1.0)
            fb  = flow.get("flow_bias", "neutral")
            pp  = flow.get("put_pct", 0.5)
            uv  = flow.get("using_volume", False)

            data_note = "📡 Using **live intraday volume** from Schwab" if uv else "📋 Using **OI as inventory proxy** — volume unavailable. This shows accumulated positioning, NOT today's flow."
            st.caption(data_note)

            f1, f2, f3 = st.columns(3)
            f1.metric("Put $ Premium",  f"${pv/1e6:.1f}M")
            f2.metric("Call $ Premium", f"${cv/1e6:.1f}M")
            f3.metric("P/C Ratio",      f"{pcr:.2f}",
                      delta="Fear" if pcr > 1.3 else ("Greed" if pcr < 0.77 else "Neutral"),
                      delta_color="inverse" if pcr > 1.3 else "normal")

            # Bar chart
            import plotly.graph_objects as _go
            _flow_or_oi = "Dollar Flow" if uv else "OI-Weighted Premium (Inventory)"
            fig_flow = _go.Figure()
            fig_flow.add_trace(_go.Bar(
                x=["Put Premium", "Call Premium"],
                y=[pv / 1e6, cv / 1e6],
                marker_color=[_C_NEG, _C_POS],
                text=[f"${pv/1e6:.1f}M", f"${cv/1e6:.1f}M"],
                textposition="auto",
            ))
            plotly_dark(fig_flow, title=f"{symbol} Options {_flow_or_oi} — P/C Ratio: {pcr:.2f}", height=320)
            fig_flow.update_layout(yaxis_title="Dollar Premium ($M)")
            st.plotly_chart(fig_flow, use_container_width=True, key="gex_flow_chart")

            # Gauge-style put% bar
            st.markdown(f"**Put share of total flow:** {pp*100:.1f}%")
            st.progress(float(pp))

            if fb == "bearish":
                st.markdown(f"""<div class='warn-card' style='background:rgba(239,68,68,0.08);border-color:rgba(239,68,68,0.30);'>
                  🔴 <b>BEARISH FLOW</b> — P/C ratio {pcr:.2f} → Put premium heavily dominant.
                  Active hedging or directional put buying in progress. Confirms downside pressure.
                </div>""", unsafe_allow_html=True)
            elif fb == "bullish":
                st.markdown(f"""<div class='warn-card' style='background:rgba(16,185,129,0.07);border-color:rgba(16,185,129,0.30);'>
                  🟢 <b>BULLISH FLOW</b> — P/C ratio {pcr:.2f} → Call premium dominant.
                  Speculative call buying or put selling. Complacency signal — watch for vol expansion.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"⚪ **NEUTRAL FLOW** — P/C ratio {pcr:.2f}. Balanced put/call dollar premium. No strong directional flow signal.")

        st.markdown("""
**Why Dollar Premium > OI for Real-Time Reads:**
- OI only updates once/day (OCC settlement). It tells you yesterday's positioning.
- Volume × premium = what traders are actually paying right now.
- A $5 put with 10,000 contracts = $5M premium. A $0.10 OTM call with 100,000 contracts = $1M. OI count misleads; dollar weight clarifies.
- P/C ratio > 1.3: elevated fear / active hedging → watch for mean-reversion once hedges expire
- P/C ratio < 0.8: speculative call buying → vol expansion risk, don't fade too aggressively
""")

    # ── GWAS Tab ─────────────────────────────────────────────────────────
    with tab_gwas:
        st.markdown(f"{sec_hdr('GAMMA-WEIGHTED AVERAGE STRIKE (GWAS)')}", unsafe_allow_html=True)
        st.caption("Probabilistic gravity centres for dealer hedging — where price gets 'pinned' in positive gamma. "
                   "More informative than discrete walls which imply false precision.")

        if gwas:
            ga = gwas.get("gwas_above")
            gb = gwas.get("gwas_below")
            gn = gwas.get("gwas_net")
            ma = gwas.get("total_gex_above", 0)
            mb = gwas.get("total_gex_below", 0)

            g1, g2, g3 = st.columns(3)
            g1.metric("GWAS Above Spot",
                      f"${ga:.2f}" if ga else "N/A",
                      delta=f"{(ga - spot):.2f} ({(ga-spot)/spot*100:+.2f}%)" if ga else None)
            g2.metric("GWAS Below Spot",
                      f"${gb:.2f}" if gb else "N/A",
                      delta=f"{(gb - spot):.2f} ({(gb-spot)/spot*100:+.2f}%)" if gb else None,
                      delta_color="inverse")
            g3.metric("Net Positive GEX Centre",
                      f"${gn:.2f}" if gn else "N/A",
                      delta=f"{(gn - spot):+.2f}" if gn else None)

            # Visual: spot vs GWAS levels on a mini chart
            import plotly.graph_objects as _go
            fig_gwas = _go.Figure()

            # Magnitude bars
            if ga:
                fig_gwas.add_trace(_go.Bar(x=[ga], y=[ma / 1e6], name="GWAS Above",
                                           marker_color=_C_POS, width=0.5,
                                           text=[f"${ga:.1f}"], textposition="auto"))
            if gb:
                fig_gwas.add_trace(_go.Bar(x=[gb], y=[mb / 1e6], name="GWAS Below",
                                           marker_color=_C_NEG, width=0.5,
                                           text=[f"${gb:.1f}"], textposition="auto"))
            fig_gwas.add_vline(x=spot, line_dash="dash", line_color="white",
                               annotation_text=f"Spot ${spot:.2f}", annotation_position="top")
            if gn:
                fig_gwas.add_vline(x=gn, line_dash="dot", line_color="#6366f1",
                                   annotation_text=f"Net GEX Ctr ${gn:.2f}", annotation_position="bottom")
            plotly_dark(fig_gwas, title="Gamma Gravity Centres (GWAS)", height=320)
            fig_gwas.update_layout(xaxis_title="Strike", yaxis_title="Weighted GEX ($M)", barmode="overlay")
            st.plotly_chart(fig_gwas, use_container_width=True, key="gex_gwas_chart")

            # Interpretation
            if ga and gb:
                range_pct = (ga - gb) / spot * 100
                st.markdown(f"**Implied pin range:** ${gb:.2f} – ${ga:.2f} ({range_pct:.1f}% wide)")
                if range_pct < 1.5:
                    st.markdown("🧲 **Tight pin zone** — gamma gravity concentrated. Expect intraday mean-reversion inside this band.")
                elif range_pct > 4.0:
                    st.markdown("📐 **Wide gravity range** — diffuse gamma support. Less pinning force; larger intraday swings possible.")
                else:
                    st.markdown("📊 **Moderate pin zone** — standard gamma gravity. Use GWAS levels as soft S/R, not hard walls.")

        st.markdown("""
**GWAS vs Discrete Walls:**
- Classic GEX walls say "resistance at $5475" — implying a binary level. Reality is a diffuse zone.
- GWAS is the gamma-weighted *centre of mass* of dealer hedging above/below spot.
- Price doesn't bounce off a wall — it gravitates toward and gets absorbed by the weighted centre.
- In strong positive gamma: spot orbits the net GEX centre intraday.
- GWAS above > GWAS below in magnitude = more upside hedging flow → upside gravity stronger.
""")

    # ── GEX Duration / Term Structure Tab ────────────────────────────────
    with tab_durability:
        st.markdown(f"{sec_hdr('GEX TERM STRUCTURE — Regime Durability')}", unsafe_allow_html=True)
        st.caption("A positive gamma regime driven by 2DTE options is gone by Friday. "
                   "Monthly gamma provides structural support. This distinction is critical for position sizing.")

        if term_str:
            g07  = term_str.get("gex_0_7dte", 0)
            g845 = term_str.get("gex_8_45dte", 0)
            frag = term_str.get("fragility_ratio", 0.5)
            dur  = term_str.get("durability", "mixed")
            buckets = term_str.get("dte_buckets", [])

            d1, d2, d3 = st.columns(3)
            d1.metric("0–7 DTE GEX",   f"${g07/1e6:.1f}M",
                      delta="Weekly / 0DTE", delta_color="off")
            d2.metric("8–45 DTE GEX",  f"${g845/1e6:.1f}M",
                      delta="Monthly", delta_color="off")
            d3.metric("Weekly Fragility", f"{frag*100:.0f}%",
                      delta=dur.upper(),
                      delta_color="inverse" if dur == "fragile" else "normal")

            # DTE bucket bar chart
            if buckets:
                import plotly.graph_objects as _go
                labels = [b[0] for b in buckets]
                values = [b[1] / 1e6 for b in buckets]
                colors = [_C_POS if v > 0 else _C_NEG for v in values]
                fig_dur = _go.Figure(_go.Bar(
                    x=labels, y=values,
                    marker_color=colors,
                    text=[f"${v:.1f}M" for v in values],
                    textposition="auto",
                ))
                plotly_dark(fig_dur, title="Net GEX by Expiration Bucket", height=320)
                fig_dur.update_layout(yaxis_title="Net GEX ($M)")
                st.plotly_chart(fig_dur, use_container_width=True, key="gex_duration_chart")

            # Durability signal
            if dur == "fragile":
                st.markdown(f"""<div class='warn-card' style='background:rgba(239,68,68,0.08);border-color:rgba(239,68,68,0.30);'>
                  ⚠️ <b>FRAGILE GAMMA REGIME</b> — {frag*100:.0f}% of net GEX expires within 7 days.
                  Current gamma environment is NOT durable. Regime can flip by end of week as weeklies expire.
                  <b>Position sizing implication:</b> treat current gamma walls as short-lived. Do not size for a multi-day pin.
                </div>""", unsafe_allow_html=True)
            elif dur == "durable":
                st.markdown(f"""<div class='warn-card' style='background:rgba(16,185,129,0.07);border-color:rgba(16,185,129,0.30);'>
                  ✅ <b>DURABLE GAMMA REGIME</b> — Only {frag*100:.0f}% of net GEX in weeklies.
                  Monthly options dominate. Current gamma structure is structurally supported through expiration.
                  <b>Position sizing implication:</b> gamma walls and flip levels are durable across sessions. Higher conviction for range trades.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"⚪ **MIXED DURABILITY** — {frag*100:.0f}% weekly. Moderate regime confidence. Re-assess after weekly expiry.")

        st.markdown("""
**Why GEX Duration Matters:**
- 45 DTE filter captures *magnitude* of GEX but hides *when* it expires.
- Weekly-concentrated positive gamma regime: flip levels meaningful Mon–Wed, then degrade Thu/Fri as gamma melts.
- Monthly-concentrated regime: gamma walls persist across the expiry cycle; S/R levels are durable for swing trades.
- High fragility + end-of-week approaching: reduce position size, widen stops, expect regime shift.
- Low fragility: gamma structure survives expiry; key nodes remain valid for 2–4 weeks.
""")



def render_setups_page():
    """Trade Setups — full live context with symbol picker, all setups, entry/stop/target levels."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## 🎯 Trade Setups — Live Context")

    # ── Symbol + data ─────────────────────────────────────────────────────
    # Persist settings across page navigation
    if "setups_symbol" not in st.session_state:
        st.session_state["setups_symbol"] = ""
    if "setups_schwab" not in st.session_state:
        st.session_state["setups_schwab"] = False

    col_sym, col_info = st.columns([1, 3])
    with col_sym:
        symbol = st.text_input("Symbol", key="setups_symbol", placeholder="e.g. SPY").strip().upper()

    if not symbol:
        st.info("Enter a symbol above to load trade setups.")
        return

    chain_df, spot, source = None, float(st.session_state.get(f"_last_known_spot_{symbol}" if symbol else "_last_known_spot", 500.0)), "unknown"
    client = get_schwab_client()
    if client:
        chain_df = schwab_get_options_chain(client, symbol, spot=None)
        source   = "Schwab API (live IV)"
        if chain_df is not None and len(chain_df) > 0:
            spot_live = schwab_get_spot(client, symbol)
            spot = spot_live if (spot_live and spot_live > 0) else float(chain_df["strike"].median())
        else:
            err = st.session_state.get("_schwab_chain_error", "unknown error")
            st.warning(f"⚠️ Schwab empty chain for **{symbol}** ({err}). Falling back to yfinance.")
            source = "Schwab API (empty chain)"
    if chain_df is None or len(chain_df) == 0:
        chain_df, spot_yf, source = get_gex_from_yfinance(symbol)
        if chain_df is not None and len(chain_df) > 0:
            spot = spot_yf or spot
            if not client:
                st.info(f"📊 yfinance OI data for **{symbol}** — connect Schwab/TOS for live IV.")
        else:
            if not client:
                st.warning("🔌 **Schwab/TOS not connected** and yfinance returned no data.")
            else:
                st.error(f"⚠️ No options data for **{symbol}** from Schwab or yfinance.")
            return

    gs        = build_gamma_state(chain_df, spot, source, max_dte=int(st.session_state.get('gex_hm_dte', 45)))
    dg        = compute_dealer_greeks(chain_df, spot, source, max_dte=int(st.session_state.get('gex_hm_dte', 45)))
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
