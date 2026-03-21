# page_thesis.py — Daily Thesis Briefing (full 13-section report)
import math, datetime as dt
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import skew as _scipy_skew, kurtosis as _scipy_kurt

from config import GammaState, GammaRegime, CSS, REGIME_OPERATIONAL_LABEL
from utils import _to_1d, zscore, resample_ffill, current_pct_rank
from ui_components import plotly_dark
from data_loaders import load_macro, get_gex_from_yfinance
from gex_engine import (build_gamma_state, compute_gwas, compute_gex_from_chain, compute_dealer_greeks,
                         compute_gex_term_structure, compute_flow_imbalance)
from schwab_api import get_schwab_client, schwab_get_spot, schwab_get_options_chain
from signals import compute_leading_stack, compute_1d_prob
from probability import (compute_prob_composite, get_session_context,
                          classify_macro_regime_abs, regime_transition_prob)
from intel_monitor import (load_feeds, geo_shock_score, categorise_items,
                            category_shock_score, _all_feeds_flat, INTEL_CATEGORIES)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _sl(s, d=float("nan")):
    try:
        v = s.dropna()
        return float(v.iloc[-1]) if len(v) else d
    except Exception:
        return d

@st.cache_data(ttl=60)
def _quotes() -> Dict:
    out = {}
    pairs = [("SPX","^GSPC"),("NDX","^NDX"),("VIX","^VIX"),("VIX3M","^VIX3M"),
             ("VVIX","^VVIX"),("VXN","^VXN"),("DXY","DX-Y.NYB"),("GLD","GLD"),("TLT","TLT"),
             ("TNX","^TNX"),("IRX","^IRX"),("SPY","SPY"),("QQQ","QQQ"),
             ("HYG","HYG"),("ES","ES=F"),("NQ","NQ=F")]
    for k, sym in pairs:
        try:
            h = yf.Ticker(sym).history(period="5d")
            if not h.empty:
                out[k+"_last"] = float(h["Close"].iloc[-1])
                out[k+"_prev"] = float(h["Close"].iloc[-2]) if len(h)>1 else out[k+"_last"]
                out[k+"_pct"]  = (out[k+"_last"]/out[k+"_prev"]-1)*100
        except Exception:
            pass
    return out

def _vrp_full(vix: float, spy: pd.Series, idx) -> Dict:
    sa   = _to_1d(spy).reindex(idx).ffill()
    rets = sa.pct_change()
    rv21 = rets.rolling(21, min_periods=10).std() * np.sqrt(252) * 100
    vrp  = vix - rv21
    val  = _sl(vrp); z = _sl(zscore(vrp)); pct = current_pct_rank(vrp, 252)
    rv21v = _sl(rv21)
    reg  = ("RICH" if val>2 else "CHEAP" if val<-1 else "FAIR") if np.isfinite(val) else "N/A"
    return {"val":val,"z":z,"pct":float(pct),"regime":reg,"rv21":rv21v,"spread":val}

def _vts(q: Dict) -> Dict:
    v=q.get("VIX_last",float("nan")); v3=q.get("VIX3M_last",float("nan"))
    ratio = v/v3 if (v3 and v3>0) else float("nan")
    carry = (v3-v)/v*100 if (v and v>0 and np.isfinite(v3)) else float("nan")
    shape = ("BACKWARDATION" if (np.isfinite(ratio) and ratio>1.05)
             else "CONTANGO" if (np.isfinite(ratio) and ratio<0.95) else "MIXED")
    return {"ratio":ratio,"carry":carry,"shape":shape}

def _tail(q: Dict) -> Dict:
    vvix=q.get("VVIX_last",float("nan")); vix=q.get("VIX_last",20.0)
    ratio = vvix/vix if (np.isfinite(vvix) and vix>0) else float("nan")
    reg   = ("ELEVATED" if (np.isfinite(ratio) and ratio>5.5)
             else "MODERATE" if (np.isfinite(ratio) and ratio>4.5) else "LOW")
    return {"vvix":vvix,"ratio":ratio,"regime":reg}

def _retdist(spy: pd.Series, idx, spot: float) -> Dict:
    sa   = _to_1d(spy).reindex(idx).ffill()
    rets = sa.pct_change().dropna()
    if len(rets)<30: return {}
    ds = float(rets.rolling(21,min_periods=10).std().iloc[-1])*100
    sk = float(_scipy_skew(rets.tail(252)))
    ku = float(_scipy_kurt(rets.tail(252), fisher=True))
    return {"daily_sigma":ds,"skew":sk,"kurtosis":ku}

def _merton_calibrate(vix: float, vvix: float,
                       vts_shape: str, vrp_val: float) -> dict:
    """
    Calibrate Merton jump-diffusion parameters to current vol regime.

    Base calibration from Broadie-Chernov-Johannes (2007) SPX fitted values:
      lam ≈ 0.59/yr, mj ≈ -0.026, sj ≈ 0.047 (long-run average)

    Adjustments:
      lam (jump intensity): scaled by VVIX/VIX ratio. High vol-of-vol = more
           frequent jumps expected. VVIX/VIX > 5.5 → crisis regime.
      mj  (mean jump size): scaled by VTS shape. Backwardation means the
           options market is pricing larger near-term downside jumps.
      sj  (jump size std):  scales with VIX level — higher IV = wider jump distribution.

    All parameters documented so users can see the calibration logic.
    """
    # Base parameters (Broadie-Chernov-Johannes 2007, rescaled for 1/252 dt)
    lam_base = 0.59   # jumps per year
    mj_base  = -0.026 # mean log jump size
    sj_base  = 0.047  # std dev of log jump size

    # lam scales with VVIX/VIX (vol-of-vol / vol ratio = tail risk premium)
    vvix_vix_ratio = (vvix / vix) if (vix > 0 and np.isfinite(vvix) and vvix > 0) else 4.8
    # Empirical range: ~3.5 (calm) to ~7.0 (crisis)
    # Normalise so ratio=4.8 (historical median) → multiplier=1.0
    lam_mult = np.clip(vvix_vix_ratio / 4.8, 0.4, 2.5)
    lam = lam_base * lam_mult

    # mj scales with VTS regime:
    #   Contango (normal): smaller jump fear → mj closer to base
    #   Backwardation (stress): market pricing larger near-term down jumps
    vts_adj = {"BACKWARDATION": 1.40, "MIXED": 1.10, "CONTANGO": 0.80, "N/A": 1.0}
    mj = mj_base * vts_adj.get(vts_shape, 1.0)

    # sj scales with VIX level — empirically wider distribution in high-vol regimes
    # VIX 15 → baseline, VIX 30 → +50% wider, VIX 45 → +100% wider
    sj_mult = np.clip(vix / 15.0, 0.7, 2.5)
    sj = sj_base * sj_mult

    # Drift adjustment: subtract jump compensation term so paths are martingales
    # E[e^j] = exp(mj + 0.5*sj^2), so mu = r - lambda*(exp(mj+0.5*sj^2)-1)
    jump_comp = lam * (np.exp(mj + 0.5 * sj**2) - 1)

    return {
        "lam": round(lam, 3),
        "mj":  round(mj, 4),
        "sj":  round(sj, 4),
        "jump_comp": jump_comp,
        "lam_mult": round(lam_mult, 2),
        "vts_adj_used": vts_shape,
        "sj_mult": round(sj_mult, 2),
    }

def _forward_prob_fig(
    spx: float, vix: float,
    ndx: float, vxn: float,
    spy: pd.Series, qqq: pd.Series,
    idx, days: int = 5, n: int = 6000,
) -> go.Figure:
    """
    Dual SPX / NDX 1-week forward probability heatmap — image-2 style.
    Two rows: SPX (top) and NDX (bottom).
    Each row: probability density heatmap (left) + terminal distribution (right).
    Bottom: scenario comparison chart + summary table.
    """
    from plotly.subplots import make_subplots
    from scipy.stats import norm as _norm

    rng = np.random.default_rng()

    def _rv20(series):
        s = _to_1d(series).reindex(idx).ffill()
        return float(s.pct_change().rolling(20, min_periods=10).std().dropna().iloc[-1] * np.sqrt(252) * 100)

    rv_spx = _rv20(spy)
    rv_ndx = _rv20(qqq)

    def _sim_paths(spot, ann_vol, days, n):
        """GBM paths, risk-neutral drift."""
        sig  = ann_vol / 100.0
        dt   = 1 / 252
        drift = -0.5 * sig**2
        paths = np.zeros((n, days + 1))
        paths[:, 0] = spot
        for t in range(1, days + 1):
            z = rng.standard_normal(n)
            paths[:, t] = paths[:, t-1] * np.exp(drift * dt + sig * np.sqrt(dt) * z)
        return paths

    def _density_grid(paths, spot, days, n_levels=60):
        """Return Z[levels, days], level_vals, pct_changes for heatmap."""
        lo = spot * 0.93
        hi = spot * 1.07
        lvls = np.linspace(lo, hi, n_levels)
        bkt  = (hi - lo) / n_levels
        Z    = np.zeros((n_levels, days))
        for d in range(days):
            f = paths[:, d + 1]
            for i, lv in enumerate(lvls):
                Z[i, d] = np.mean((f >= lv) & (f < lv + bkt)) * 100
        pcts = (lvls / spot - 1) * 100
        return Z, lvls, pcts

    spx_paths = _sim_paths(spx, vix,   days, n)
    ndx_paths = _sim_paths(ndx, vxn,   days, n)

    Z_spx, lvls_spx, pcts_spx = _density_grid(spx_paths, spx, days)
    Z_ndx, lvls_ndx, pcts_ndx = _density_grid(ndx_paths, ndx, days)

    day_labels  = [f"D{i+1}" for i in range(days)]
    wk_days     = ["Mon", "Tue", "Wed", "Thu", "Fri"][:days]
    x_labels    = [f"0{i+1}\n{wk_days[i]}" if i < len(wk_days) else f"D{i+1}" for i in range(days)]

    # Terminal distributions (Day 5)
    term_spx = spx_paths[:, -1]
    term_ndx = ndx_paths[:, -1]

    def _prob_row(paths, spot):
        t = paths[:, -1]
        return {
            "P(down>3%)": round(np.mean(t < spot * 0.97) * 100, 1),
            "P(down>2%)": round(np.mean(t < spot * 0.98) * 100, 1),
            "P(down>1%)": round(np.mean(t < spot * 0.99) * 100, 1),
            "P(flat±1%)": round(np.mean(np.abs(t / spot - 1) < 0.01) * 100, 1),
            "P(up>1%)":   round(np.mean(t > spot * 1.01) * 100, 1),
            "P(up>2%)":   round(np.mean(t > spot * 1.02) * 100, 1),
            "P(up>3%)":   round(np.mean(t > spot * 1.03) * 100, 1),
            "p5":  round(float(np.percentile(t, 5)),  2),
            "p95": round(float(np.percentile(t, 95)), 2),
            "p1":  round(float(np.percentile(t, 1)),  2),
            "median": round(float(np.median(t)), 2),
            "expected": round(float(np.mean(t)), 2),
        }

    pr_spx = _prob_row(spx_paths, spx)
    pr_ndx = _prob_row(ndx_paths, ndx)

    # ── Layout ────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=5, cols=2,
        row_heights=[0.26, 0.26, 0.20, 0.14, 0.14],
        column_widths=[0.72, 0.28],
        specs=[
            [{"type": "heatmap"}, {"type": "bar"}],
            [{"type": "heatmap"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "table"},   {"type": "table"}],
            [{"type": "table"},   {"type": "table"}],
        ],
        vertical_spacing=0.04,
        horizontal_spacing=0.04,
        subplot_titles=[
            f"SPX Probability Density  |  Spot: {spx:,.2f}  |  VIX: {vix:.1f}%  |  RV(20): {rv_spx:.1f}%",
            "SPX Terminal",
            f"NDX Probability Density  |  Spot: {ndx:,.2f}  |  VXN: {vxn:.1f}%  |  RV(20): {rv_ndx:.1f}%",
            "NDX Terminal",
            "SPX Scenario Comparison: Median + IQR + 90% CI",
            "NDX Scenario Comparison: Median + IQR + 90% CI",
            "SPX — 1-Week Forward Probability Summary", "",
            "NDX — 1-Week Forward Probability Summary", "",
        ],
    )

    # ── Row 1: SPX heatmap + terminal ─────────────────────────────────────
    fig.add_trace(go.Heatmap(
        z=Z_spx, x=x_labels, y=[f"{p:+.1f}%" for p in pcts_spx],
        colorscale="RdYlGn", reversescale=False,
        showscale=False, zmin=0,
    ), row=1, col=1)

    # Spot line on heatmap
    spot_idx_spx = int(np.argmin(np.abs(lvls_spx - spx)))
    fig.add_hline(y=f"{pcts_spx[spot_idx_spx]:+.1f}%",
                  line_color="white", line_width=1.2, line_dash="solid", row=1, col=1)

    # Percentile fan lines on heatmap (as scatter on secondary y approach)
    for pct_val, label, color in [
        (5, "-2σ", "rgba(239,68,68,0.7)"),
        (25, "-1σ", "rgba(245,158,11,0.7)"),
        (75, "+1σ", "rgba(245,158,11,0.7)"),
        (95, "+2σ", "rgba(239,68,68,0.7)"),
    ]:
        fan_y = []
        for d in range(days):
            p = spx_paths[:, d+1]
            val = np.percentile(p, pct_val)
            pct_change = (val / spx - 1) * 100
            fan_y.append(f"{pct_change:+.1f}%")

    # Terminal distribution (SPX)
    hist_vals, bin_edges = np.histogram(term_spx, bins=40, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig.add_trace(go.Bar(
        x=hist_vals, y=[f"{(v/spx-1)*100:+.1f}%" for v in bin_centers],
        orientation="h", marker_color="#06b6d4", opacity=0.7,
        name="SPX terminal", showlegend=False,
    ), row=1, col=2)

    # ── Row 2: NDX heatmap + terminal ─────────────────────────────────────
    fig.add_trace(go.Heatmap(
        z=Z_ndx, x=x_labels, y=[f"{p:+.1f}%" for p in pcts_ndx],
        colorscale="RdYlGn", reversescale=False,
        showscale=False, zmin=0,
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=hist_vals, y=[f"{(v/ndx-1)*100:+.1f}%" for v in
                        (bin_edges[:-1] + bin_edges[1:]) / 2 / spx * ndx],
        orientation="h", marker_color="#06b6d4", opacity=0.7,
        name="NDX terminal", showlegend=False,
    ), row=2, col=2)

    # ── Row 3: Scenario comparison (SPX left, NDX right) ──────────────────
    t_axis = np.arange(1, days + 1)
    for paths, spot_v, row, col in [(spx_paths, spx, 3, 1), (ndx_paths, ndx, 3, 2)]:
        p5  = [np.percentile(paths[:, d], 5)  for d in range(1, days+1)]
        p25 = [np.percentile(paths[:, d], 25) for d in range(1, days+1)]
        p50 = [np.percentile(paths[:, d], 50) for d in range(1, days+1)]
        p75 = [np.percentile(paths[:, d], 75) for d in range(1, days+1)]
        p95 = [np.percentile(paths[:, d], 95) for d in range(1, days+1)]

        fig.add_trace(go.Scatter(
            x=list(t_axis)+list(t_axis[::-1]), y=p5+p95[::-1],
            fill="toself", fillcolor="rgba(239,68,68,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="90% CI", showlegend=(col==1),
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=list(t_axis)+list(t_axis[::-1]), y=p25+p75[::-1],
            fill="toself", fillcolor="rgba(99,102,241,0.25)",
            line=dict(color="rgba(0,0,0,0)"), name="IQR", showlegend=(col==1),
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=t_axis, y=p50, line=dict(color="#10b981", width=2),
            name="Median", showlegend=(col==1),
        ), row=row, col=col)

    # ── Rows 4–5: Summary tables ──────────────────────────────────────────
    def _summary_table(pr, spot_v, sym, rv, vix_v):
        exp_chg = (pr["expected"] / spot_v - 1) * 100
        med_chg = (pr["median"]   / spot_v - 1) * 100
        rows = [
            (f"{sym} Spot",      f"{spot_v:,.2f}", ""),
            ("Expected (5d)",    f"{pr['expected']:,.2f} ({exp_chg:+.2f}%)", ""),
            ("Median (5d)",      f"{pr['median']:,.2f} ({med_chg:+.2f}%)", ""),
            ("", "", ""),
            ("P(down > 3%)",     f"{pr['P(down>3%)']}%",  "ALERT" if pr["P(down>3%)"] > 30 else ("WATCH" if pr["P(down>3%)"] > 20 else "")),
            ("P(down > 2%)",     f"{pr['P(down>2%)']}%",  ""),
            ("P(down > 1%)",     f"{pr['P(down>1%)']}%",  ""),
            ("P(flat ±1%)",      f"{pr['P(flat±1%)']}%",  ""),
            ("P(up > 1%)",       f"{pr['P(up>1%)']}%",    ""),
            ("P(up > 2%)",       f"{pr['P(up>2%)']}%",    ""),
            ("P(up > 3%)",       f"{pr['P(up>3%)']}%",    ""),
            ("", "", ""),
            ("5th pctile",       f"{pr['p5']:,.2f} ({(pr['p5']/spot_v-1)*100:+.2f}%)", ""),
            ("95th pctile",      f"{pr['p95']:,.2f} ({(pr['p95']/spot_v-1)*100:+.2f}%)", ""),
            ("1st pctile",       f"{pr['p1']:,.2f} ({(pr['p1']/spot_v-1)*100:+.2f}%)", "TAIL"),
            ("", "", ""),
            ("DV",               f"{vix_v:.1f}%", ""),
            ("RV(20)",           f"{rv:.1f}%", ""),
            ("RV(5d)",           f"{vix_v * 0.85:.1f}%", ""),
            ("Skew(60d)",        "n/a", ""),
        ]
        return rows

    for pr, spot_v, sym, rv, vix_v, row, col in [
        (pr_spx, spx, "SPX", rv_spx, vix,  4, 1),
        (pr_ndx, ndx, "NDX", rv_ndx, vxn,  4, 2),
    ]:
        rows = _summary_table(pr, spot_v, sym, rv, vix_v)
        metrics = [r[0] for r in rows]
        values  = [r[1] for r in rows]
        signals = [r[2] for r in rows]
        sig_colors = ["#ef4444" if s == "ALERT" else "#f59e0b" if s in ("WATCH","TAIL") else "rgba(0,0,0,0)" for s in signals]

        fig.add_trace(go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>Value</b>", "<b>Signal</b>"],
                fill_color="rgba(30,30,50,0.9)",
                font=dict(color=["#f59e0b","#f59e0b","#f59e0b"], size=10),
                align="left", height=20,
            ),
            cells=dict(
                values=[metrics, values, signals],
                fill_color=[
                    ["rgba(15,15,25,0.95)"] * len(metrics),
                    ["rgba(15,15,25,0.95)"] * len(values),
                    sig_colors,
                ],
                font=dict(
                    color=[
                        ["rgba(245,158,11,0.9)"] * len(metrics),
                        ["rgba(255,255,255,0.85)"] * len(values),
                        ["#ef4444" if s == "ALERT" else "#f59e0b" if s in ("WATCH","TAIL") else "rgba(0,0,0,0)" for s in signals],
                    ],
                    size=10, family="monospace",
                ),
                align="left", height=18,
            ),
        ), row=row, col=col)

    # ── Styling ────────────────────────────────────────────────────────────
    title_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    plotly_dark(fig, title=f"SPX / NDX  FORWARD PROBABILITY HEATMAP — 1-WEEK FORECAST — {title_date}", height=1200)
    fig.update_layout(
        margin=dict(t=80, b=20, l=50, r=20),
        legend=dict(orientation="h", y=0.52, x=0, font=dict(size=10)),
    )
    # Darken heatmap backgrounds
    for r in [1, 2]:
        fig.update_xaxes(showgrid=False, row=r, col=1)
        fig.update_yaxes(showgrid=False, row=r, col=1)
        fig.update_xaxes(showgrid=False, row=r, col=2)
        fig.update_yaxes(showgrid=False, row=r, col=2)

    return fig


def _merton_fig(spot: float, vix: float, vvix: float = float("nan"),
                vts_shape: str = "MIXED", vrp_val: float = 0.0,
                days=5, n=4000) -> go.Figure:
    rng = np.random.default_rng()   # unseeded local generator — varies each call
    dt_s = 1 / 252
    sig  = vix / 100
    params = _merton_calibrate(vix, vvix, vts_shape, vrp_val)
    lam, mj, sj = params["lam"], params["mj"], params["sj"]
    drift = -0.5 * sig**2 - params["jump_comp"]  # risk-neutral drift

    paths = np.zeros((n, days + 1)); paths[:, 0] = spot
    for t in range(1, days + 1):
        z  = rng.standard_normal(n)
        nj = rng.poisson(lam * dt_s, n)
        j  = rng.normal(mj, sj, n) * nj
        paths[:, t] = paths[:, t-1] * np.exp(drift * dt_s + sig * np.sqrt(dt_s) * z + j)
    lo=spot*0.93; hi=spot*1.07; lvls=np.linspace(lo,hi,24); bkt=(hi-lo)/24
    Z=np.zeros((24,days))
    for d in range(days):
        f=paths[:,d+1]
        for i,lv in enumerate(lvls):
            Z[i,d]=np.mean((f>=lv)&(f<lv+bkt))*100
    y_labels=[f"{lv:.0f}" for lv in lvls]
    fig=go.Figure(go.Heatmap(z=Z,x=[f"Day {i+1}" for i in range(days)],
        y=y_labels,colorscale="Viridis",showscale=True,
        colorbar=dict(title="Prob%",thickness=12)))
    # Categorical y-axis: add_hline fails. Use add_shape with paper coords instead.
    closest_idx=int(np.argmin(np.abs(lvls-spot)))
    y_paper=closest_idx/max(len(lvls)-1,1)
    fig.add_shape(type="line",xref="paper",yref="paper",
                  x0=0,x1=1,y0=y_paper,y1=y_paper,
                  line=dict(color="white",width=1.5,dash="dash"))
    fig.add_annotation(xref="paper",yref="paper",x=0.01,y=y_paper,
                       text=f"Spot {spot:.0f}",showarrow=False,
                       font=dict(color="white",size=10),
                       xanchor="left",yanchor="bottom")
    plotly_dark(fig,title="Probability Heatmap — Merton Jump-Diffusion (1-week)",height=420)
    fig.update_layout(xaxis_title="Trading Day Forward",yaxis_title="Price Level")
    return fig

def _ivsurf_from_chain(chain_df, spot: float, label: str,
                        vix: float, vts_shape: str) -> go.Figure:
    """
    Build IV surface from actual options chain data using scattered 2D interpolation.

    The chain has sparse discrete data (few expiry buckets × few strikes), so a
    simple pivot → fillna approach creates jagged spikes. Instead we:
      1. Extract (moneyness, DTE, IV) scatter points from the raw chain
      2. Interpolate onto a smooth dense grid using scipy.interpolate.griddata
         with cubic method (falls back to linear if too few points)
      3. Clip outliers and smooth with a 2D gaussian filter to remove edge artifacts

    If chain_df is unavailable, shows a clearly-labelled placeholder — never a
    fake parametric surface.
    """
    if chain_df is None or len(chain_df) == 0:
        fig = go.Figure()
        plotly_dark(fig, title=f"IV Surface — {label} (DATA UNAVAILABLE)", height=420)
        if label in ("NDX", "QQQ"):
            msg = (
                "<b>No NDX/QQQ options chain data available.</b><br>"
                "Set GEX Symbol to QQQ or NDX to load this surface.<br>"
                "When a different underlying is selected, NDX surface<br>"
                "is intentionally left blank to avoid mislabelled data."
            )
        else:
            msg = (
                f"<b>No {label} options chain data available.</b><br>"
                "Data loads automatically via yfinance (no login needed)<br>"
                "or from Schwab/TOS if connected. If this persists,<br>"
                "check your internet connection or try refreshing."
            )
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text=msg,
            showarrow=False, font=dict(color="rgba(255,255,255,0.6)", size=13),
            align="center",
        )
        return fig

    try:
        from scipy.interpolate import griddata
        from scipy.ndimage import gaussian_filter

        df = chain_df.copy()
        df["dte"] = (df["expiry_T"] * 365).clip(lower=1)
        df["moneyness"] = df["strike"] / spot

        # Filter to near-term tradeable range and valid IV
        # ── Change 1 & 2: tighter moneyness (0.93–1.15) and short DTE (≤10,
        #    fallback to ≤21 if fewer than 4 points survive the initial cut) ──
        df = df[(df["moneyness"] >= 0.93) & (df["moneyness"] <= 1.15)]
        _dte_limit = 10
        df = df[(df["dte"] >= 1) & (df["dte"] <= _dte_limit)]
        if len(df) < 4:
            # fallback: widen DTE window to 21 days
            _dte_limit = 21
            df = chain_df.copy()
            df["dte"] = (df["expiry_T"] * 365).clip(lower=1)
            df["moneyness"] = df["strike"] / spot
            df = df[(df["moneyness"] >= 0.93) & (df["moneyness"] <= 1.15)]
            df = df[(df["dte"] >= 1) & (df["dte"] <= _dte_limit)]

        df = df[(df["iv"] > 0.005) & (df["iv"] < 2.0)]   # tightened: >200% IV is bad data

        # Remove extreme outliers: IV > 3× median is almost certainly bad data
        iv_med = df["iv"].median()
        df = df[(df["iv"] >= iv_med * 0.25) & (df["iv"] < iv_med * 3.0)]

        if len(df) < 6:
            raise ValueError(f"Only {len(df)} valid chain points — insufficient for surface")

        # Compute ATM IV estimate from raw data BEFORE griddata (used for clipping)
        _atm_mask = df["moneyness"].between(0.98, 1.02)
        atm_iv_est = float(df[_atm_mask]["iv"].mean() * 100) if _atm_mask.sum() > 0 else vix
        if not np.isfinite(atm_iv_est) or atm_iv_est <= 0:
            atm_iv_est = vix
        # Hard clip bounds: IV must be in [2%, 120%] and within 3× ATM
        hard_lo = max(2.0,  atm_iv_est * 0.40)
        hard_hi = min(120.0, atm_iv_est * 3.0)

        # Scatter points in (DTE, moneyness) space
        pts_dte = df["dte"].values.astype(float)
        pts_m   = df["moneyness"].values.astype(float)
        pts_iv  = np.clip(df["iv"].values.astype(float) * 100, hard_lo, hard_hi)

        # ── Change 3: dense output grid capped at 10 DTE, 16 cells ──────
        dte_grid = np.linspace(max(pts_dte.min(), 1.0), min(pts_dte.max(), 10.0), 16)
        m_grid   = np.linspace(pts_m.min(),   pts_m.max(),   30)
        DTE, MON = np.meshgrid(dte_grid, m_grid)

        # Use linear interpolation — cubic produces wild extrapolated values
        # outside the convex hull of sparse option chain data points
        method = "linear" if len(df) >= 8 else "nearest"
        Z = griddata(
            points=np.column_stack([pts_dte, pts_m]),
            values=pts_iv,
            xi=np.column_stack([DTE.ravel(), MON.ravel()]),
            method=method,
            fill_value=np.nan,
        ).reshape(DTE.shape)

        # Fill any NaN edges with nearest-neighbour
        nan_mask = ~np.isfinite(Z)
        if nan_mask.any():
            Z_nn = griddata(
                points=np.column_stack([pts_dte, pts_m]),
                values=pts_iv,
                xi=np.column_stack([DTE.ravel(), MON.ravel()]),
                method="nearest",
            ).reshape(DTE.shape)
            Z[nan_mask] = Z_nn[nan_mask]

        # Hard clip BEFORE smoothing — prevents any interpolation artefact from
        # propagating into the smoothed surface (the main cause of the 400% spikes
        # and -300% troughs seen with cubic interpolation on short-dated chains)
        Z = np.clip(Z, hard_lo, hard_hi)

        # ── Change 8: sigma=1.2 Gaussian smooth + re-clip after ──────────
        Z = gaussian_filter(Z, sigma=1.2)
        Z = np.clip(Z, hard_lo, hard_hi)

        # ── Skew & term structure metrics from raw data ───────────────────
        def _avg(mask): return float(df[mask]["iv"].mean() * 100) if mask.sum() > 0 else float("nan")
        puts_25d  = _avg((df["moneyness"].between(0.93, 0.97)) & (df["dte"].between(15, 45)))
        calls_25d = _avg((df["moneyness"].between(1.03, 1.07)) & (df["dte"].between(15, 45)))
        skew_25d  = puts_25d - calls_25d
        front_atm = _avg((df["moneyness"].between(0.985, 1.015)) & (df["dte"] <= 10))
        back_atm  = _avg((df["moneyness"].between(0.985, 1.015)) & (df["dte"].between(25, 70)))
        skew_str  = f"25Δ Skew: {skew_25d:+.1f}pts" if (np.isfinite(puts_25d) and np.isfinite(calls_25d)) else "skew: n/a"
        ts_str    = f"TS: {front_atm:.1f}% → {back_atm:.1f}%" if (np.isfinite(front_atm) and np.isfinite(back_atm)) else "TS: n/a"
        regime_note = {"BACKWARDATION": "⚠️ Backwardation", "CONTANGO": "✓ Contango", "MIXED": "~ Mixed"}.get(vts_shape, "")
        n_points = len(df)

        fig = go.Figure(go.Surface(
            x=dte_grid,
            y=m_grid * 100,   # display as moneyness %
            z=Z,
            # ── Change 4: plasma-inspired colorscale ─────────────────────
            colorscale=[
                [0.00, "#0d0221"],
                [0.20, "#5c1a8a"],
                [0.45, "#c0392b"],
                [0.70, "#e67e22"],
                [0.85, "#f1c40f"],
                [1.00, "#ffffcc"],
            ],
            showscale=True,
            colorbar=dict(title="IV%", thickness=14, len=0.75),
            opacity=0.95,
            # ── Change 5: disable contour floor projection ───────────────
            contours=dict(
                z=dict(show=False),
            ),
            lighting=dict(ambient=0.7, diffuse=0.8, roughness=0.5, specular=0.3),
            lightposition=dict(x=100, y=200, z=0),
        ))

        # ── Change 7: updated title with spot + ATM IV ───────────────────
        plotly_dark(fig, title=f"SPX Implied Volatility Surface | Spot: {spot:,.2f} | ATM IV: {atm_iv_est:.1f}%", height=460)
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="DTE", backgroundcolor="rgba(0,0,0,0)",
                           gridcolor="rgba(255,255,255,0.08)", showbackground=True),
                yaxis=dict(title="Moneyness (%)", backgroundcolor="rgba(0,0,0,0)",
                           gridcolor="rgba(255,255,255,0.08)", showbackground=True),
                zaxis=dict(title="IV%", backgroundcolor="rgba(0,0,0,0)",
                           gridcolor="rgba(255,255,255,0.08)", showbackground=True),
                bgcolor="rgba(0,0,0,0)",
                # ── Change 6: updated camera angle ───────────────────────
                camera=dict(eye=dict(x=-1.8, y=-1.5, z=0.8)),
            ),
            margin=dict(l=0, r=0, t=50, b=0),
        )
        return fig

    except Exception as exc:
        fig = go.Figure()
        plotly_dark(fig, title=f"IV Surface — {label} (BUILD ERROR)", height=380)
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text=f"Surface build failed: {type(exc).__name__}: {exc}",
            showarrow=False, font=dict(color="rgba(255,200,0,0.8)", size=11),
            align="center",
        )
        return fig


def _ivrv_fig(vix_s: pd.Series, spy: pd.Series, idx) -> go.Figure:
    from plotly.subplots import make_subplots
    sa  = _to_1d(spy).reindex(idx).ffill()
    va  = _to_1d(vix_s).reindex(idx).ffill()
    rv5  = sa.pct_change().rolling(5,  min_periods=3).std()  * np.sqrt(252) * 100
    rv21 = sa.pct_change().rolling(21, min_periods=10).std() * np.sqrt(252) * 100
    rv63 = sa.pct_change().rolling(63, min_periods=30).std() * np.sqrt(252) * 100
    vrp  = (va - rv21).dropna()
    dates = va.dropna().index[-252:]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.04,
        subplot_titles=["", "Variance Risk Premium Spread (VIX - 21d RV)"],
    )

    # ── Top panel: VIX + RV lines + RV range band ────────────────────────
    rv5_d  = rv5.reindex(dates)
    rv63_d = rv63.reindex(dates)
    rv21_d = rv21.reindex(dates)
    va_d   = va.reindex(dates)

    # RV range band (5d–63d)
    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list(rv5_d.fillna(method="ffill")) + list(rv63_d.fillna(method="ffill")[::-1]),
        fill="toself", fillcolor="rgba(6,182,212,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="5d-63d RV range",
        showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=va_d, name="VIX (IV)",
        line=dict(color="#a855f7", width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=rv21_d, name="21d RV",
        line=dict(color="#06b6d4", width=2),
    ), row=1, col=1)

    # ── Bottom panel: VRP bars ────────────────────────────────────────────
    vrp_d = vrp.reindex(dates)
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in vrp_d.fillna(0)]
    fig.add_trace(go.Bar(
        x=dates, y=vrp_d,
        marker_color=colors, opacity=0.8,
        name="VRP", showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1, row=2, col=1)

    # Dummy traces for legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
        marker=dict(color="#10b981", size=8, symbol="square"),
        name="IV > RV (premium)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
        marker=dict(color="#ef4444", size=8, symbol="square"),
        name="IV < RV (discount)"), row=2, col=1)

    plotly_dark(fig, title="Implied vs Realized Volatility — SPX (1Y)", height=520)
    fig.update_layout(
        legend=dict(orientation="h", y=1.02, x=0),
        yaxis=dict(title="Volatility %", gridcolor="rgba(255,255,255,0.06)"),
        yaxis2=dict(title="VRP pts",     gridcolor="rgba(255,255,255,0.06)"),
        xaxis2=dict(title=""),
        margin=dict(t=60, b=20, l=60, r=20),
    )
    fig.update_annotations(font_size=11)
    return fig

def _rdist_fig(spy: pd.Series, idx, rd: Dict, spot: float = float("nan")) -> go.Figure:
    from plotly.subplots import make_subplots
    sa   = _to_1d(spy).reindex(idx).ffill()
    rets = sa.pct_change().dropna().tail(504) * 100   # 2Y ≈ 504 trading days
    mu   = rd.get("daily_mu",  float(rets.mean()))
    sig  = rd.get("daily_sigma", 1.0)
    skw  = rd.get("skew", 0.0)
    kurt = rd.get("kurtosis", 0.0)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        specs=[[{"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.06,
    )

    # ── Histogram ─────────────────────────────────────────────────────────
    fig.add_trace(go.Histogram(
        x=rets, nbinsx=80, name="Actual",
        marker_color="#6366f1", opacity=0.75, histnorm="probability density",
    ), row=1, col=1)

    xn  = np.linspace(float(rets.min()) - 0.5, float(rets.max()) + 0.5, 300)
    yn  = np.exp(-0.5 * ((xn - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
    fig.add_trace(go.Scatter(
        x=xn, y=yn, name=f"Normal(μ={mu:.3f}, σ={sig:.3f})",
        line=dict(color="#f59e0b", width=2),
    ), row=1, col=1)

    # Mean line
    fig.add_vline(x=mu, line_color="#06b6d4", line_width=1.5,
                  annotation_text=f"Mean: {mu:.3f}%", annotation_font_size=10,
                  row=1, col=1)

    # σ bands
    for m, col_s in [(1, "#10b981"), (2, "#f59e0b"), (3, "#ef4444")]:
        for s in [-1, 1]:
            fig.add_vline(x=s * m * sig + mu, line_dash="dash",
                          line_color=col_s, line_width=1, row=1, col=1)

    # ── Stats panel (right col — invisible axes, monospace annotations) ──
    # 1-day probability bands
    d1_lo68 = round(spot * (1 + (mu - sig)   / 100), 2) if np.isfinite(spot) else float("nan")
    d1_hi68 = round(spot * (1 + (mu + sig)   / 100), 2) if np.isfinite(spot) else float("nan")
    d1_lo95 = round(spot * (1 + (mu - 2*sig) / 100), 2) if np.isfinite(spot) else float("nan")
    d1_hi95 = round(spot * (1 + (mu + 2*sig) / 100), 2) if np.isfinite(spot) else float("nan")
    d1_lo90 = round(spot * (1 + (mu - 1.645*sig) / 100), 2) if np.isfinite(spot) else float("nan")
    d1_hi90 = round(spot * (1 + (mu + 1.645*sig) / 100), 2) if np.isfinite(spot) else float("nan")
    d1_lo99 = round(spot * (1 + (mu - 2.576*sig) / 100), 2) if np.isfinite(spot) else float("nan")
    d1_hi99 = round(spot * (1 + (mu + 2.576*sig) / 100), 2) if np.isfinite(spot) else float("nan")

    wk_sig = sig * np.sqrt(5)
    w1_lo68 = round(spot * (1 + (mu*5 - wk_sig)   / 100), 2) if np.isfinite(spot) else float("nan")
    w1_hi68 = round(spot * (1 + (mu*5 + wk_sig)   / 100), 2) if np.isfinite(spot) else float("nan")
    w1_lo95 = round(spot * (1 + (mu*5 - 2*wk_sig) / 100), 2) if np.isfinite(spot) else float("nan")
    w1_hi95 = round(spot * (1 + (mu*5 + 2*wk_sig) / 100), 2) if np.isfinite(spot) else float("nan")
    w1_lo90 = round(spot * (1 + (mu*5 - 1.645*wk_sig) / 100), 2) if np.isfinite(spot) else float("nan")
    w1_hi90 = round(spot * (1 + (mu*5 + 1.645*wk_sig) / 100), 2) if np.isfinite(spot) else float("nan")
    w1_lo99 = round(spot * (1 + (mu*5 - 2.576*wk_sig) / 100), 2) if np.isfinite(spot) else float("nan")
    w1_hi99 = round(spot * (1 + (mu*5 + 2.576*wk_sig) / 100), 2) if np.isfinite(spot) else float("nan")

    days_2sig = int((np.abs(rets) > 2 * sig).sum())
    pct_2sig  = round(days_2sig / len(rets) * 100, 1)
    fat_tails = "YES" if pct_2sig > 6.0 else "NO"

    def _fmt(v): return f"{v:,.2f}" if np.isfinite(v) else "N/A"

    panel_lines = [
        ("GAUSSIAN MEASUREMENTS", None, "#ffffff"),
        ("", None, None),
        (f"SPX Spot:        {_fmt(spot)}", None, "#ffffff"),
        (f"Daily μ:         {mu:+.4f}%", None, "#e2e8f0"),
        (f"Daily σ:         {sig:.4f}%", None, "#e2e8f0"),
        (f"Skewness:        {skw:.4f}", None, "#e2e8f0"),
        (f"Excess Kurtosis: {kurt:.4f}", None, "#e2e8f0"),
        ("", None, None),
        ("1-DAY PROBABILITY BANDS", None, "#ffffff"),
        ("", None, None),
        (f"  68% (1σ):  {_fmt(d1_lo68)} — {_fmt(d1_hi68)}", None, "#94a3b8"),
        (f"  90%:       {_fmt(d1_lo90)} — {_fmt(d1_hi90)}", None, "#94a3b8"),
        (f"  95% (2σ):  {_fmt(d1_lo95)} — {_fmt(d1_hi95)}", None, "#94a3b8"),
        (f"  99%:       {_fmt(d1_lo99)} — {_fmt(d1_hi99)}", None, "#94a3b8"),
        ("", None, None),
        ("1-WEEK PROBABILITY BANDS", None, "#ffffff"),
        ("", None, None),
        (f"  68% (1σ):  {_fmt(w1_lo68)} — {_fmt(w1_hi68)}", None, "#94a3b8"),
        (f"  90%:       {_fmt(w1_lo90)} — {_fmt(w1_hi90)}", None, "#94a3b8"),
        (f"  95% (2σ):  {_fmt(w1_lo95)} — {_fmt(w1_hi95)}", None, "#94a3b8"),
        (f"  99%:       {_fmt(w1_lo99)} — {_fmt(w1_hi99)}", None, "#94a3b8"),
        ("", None, None),
        ("TAIL ANALYSIS", None, "#ffffff"),
        ("", None, None),
        (f"  Days > 2σ:       {pct_2sig}%", None, "#94a3b8"),
        (f"  (Normal expect:  4.6%)", None, "#64748b"),
        (f"  Fat tails:       {fat_tails}", "#ef4444" if fat_tails == "YES" else "#10b981", None),
    ]

    y_step = 1.0 / max(len(panel_lines), 1)
    for i, (text, override_color, base_color) in enumerate(panel_lines):
        if not text:
            continue
        color = override_color or base_color or "#94a3b8"
        is_header = base_color == "#ffffff" and not text.startswith(" ")
        fig.add_annotation(
            xref="x2 domain", yref="y2 domain",
            x=0.02, y=1.0 - i * y_step,
            text=text,
            showarrow=False,
            font=dict(
                family="monospace",
                size=11 if is_header else 10,
                color=color,
            ),
            xanchor="left", yanchor="top",
        )

    # Hide right axes
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)
    # Add border box for stats panel
    fig.add_shape(type="rect", xref="x2 domain", yref="y2 domain",
                  x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="rgba(255,255,255,0.15)", width=1),
                  fillcolor="rgba(255,255,255,0.03)",
                  row=1, col=2)

    plotly_dark(fig, title=f"SPX Daily Return Distribution (2Y)", height=480)
    fig.update_layout(
        xaxis=dict(title="Daily Return %", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Density",        gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="v", x=0.36, y=0.99, font=dict(size=10)),
        margin=dict(t=60, b=40, l=60, r=20),
        bargap=0.02,
    )
    return fig

def _recession_stress(sahm: float, hy: float, s2s10_bp: float,
                      icsa_4w_chg: float, vix: float) -> dict:
    """
    Recession Stress Index — a composite indicator, NOT a calibrated probability.

    Uses a simple probit-style score combining:
      1. Sahm Rule (0–0.5+ scale, most direct recession onset signal)
      2. HY OAS (credit stress, historical recession threshold ~500bp)
      3. Yield curve (2s10s inversion depth and duration proxy)
      4. Initial jobless claims 4-week % change (ICSA — weekly, leads UNRATE)
      5. VIX level (financial conditions proxy)

    These factors are standardized and combined with weights derived from
    the empirical literature on recession leading indicators:
      - Sahm Rule: 35% (most reliable real-time signal, Sahm 2019)
      - HY OAS:    25% (credit leads equity and employment)
      - Yield curve:20% (classic Estrella-Mishkin probit)
      - ICSA 4W Δ: 12% (initial jobless claims — weekly frequency, genuinely
                        orthogonal to Sahm which uses monthly UNRATE averages)
      - VIX:        8% (financial conditions proxy)

    Output is mapped to [0, 99] via a logistic function to avoid the
    false precision of a "probability" label.

    Call this a Stress Index, not a recession probability.
    """
    from scipy.special import expit as _sigmoid

    # Each component → z-score relative to recession/non-recession historical ranges
    # Sahm: 0=clear, 0.3=warning, 0.5=triggered (historical recession onset)
    sahm_z = np.clip(sahm / 0.4, 0, 1)  # normalise: 0.4 = historical trigger midpoint

    # HY OAS: 300=normal, 500=stress, 800=crisis, 1200=systemic
    hy_z = np.clip((hy - 300) / 700, 0, 1)

    # Yield curve: inversion (negative 2s10s) is bearish
    # -150bp = deep inversion (strong recession signal), +50bp = normal
    curve_z = np.clip((-s2s10_bp - 0) / 150, 0, 1)  # 0 at flat, 1 at -150bp

    # ICSA 4-week % change: genuinely orthogonal to Sahm Rule.
    # Sahm uses monthly UNRATE 3M avg vs 12M min — a lagging stock measure.
    # ICSA is weekly flow data that leads payrolls and UNRATE by 4–8 weeks.
    # 0% = stable, +10% = deteriorating, +20% = recession-onset pace.
    # Normalise: 20% 4-week rise historically coincides with recession onset.
    icsa_z = np.clip(icsa_4w_chg / 0.20, 0, 1)

    # VIX: 15=normal, 25=elevated, 40=crisis
    vix_z = np.clip((vix - 15) / 35, 0, 1)

    # Weighted composite
    raw = (0.35 * sahm_z + 0.25 * hy_z + 0.20 * curve_z
           + 0.12 * icsa_z + 0.08 * vix_z)

    # Map to 0-99 via logistic (avoids false precision near 0 and 100)
    # Calibrate so raw=0 → ~5%, raw=0.5 → ~50%, raw=1.0 → ~95%
    score = float(_sigmoid((raw - 0.5) * 8) * 100)
    score = round(np.clip(score, 1, 99), 1)

    # Regime label based on Sahm + HY combination (hardest to fake)
    if sahm >= 0.50 or hy > 600:
        label = "ELEVATED"
    elif sahm >= 0.30 or hy > 420:
        label = "MODERATE"
    else:
        label = "LOW"

    return {
        "score": score,
        "label": label,
        "components": {
            "sahm_contribution":  round(0.35 * sahm_z  * 100, 1),
            "hy_contribution":    round(0.25 * hy_z    * 100, 1),
            "curve_contribution": round(0.20 * curve_z * 100, 1),
            "icsa_4w_contribution": round(0.12 * icsa_z * 100, 1),
            "vix_contribution":   round(0.08 * vix_z   * 100, 1),
        },
        "note": (
            "Recession Stress Index — NOT a calibrated probability. "
            "Use Cleveland Fed CFNAI model or Estrella-Mishkin probit for "
            "statistically grounded recession probabilities."
        ),
    }

def _composite(prob: dict, vrp_val: float) -> int:
    """
    Signed ±10 composite score built from four non-overlapping signal buckets.

    Inputs chosen to avoid double-counting:
      - tactical_prob  (5D):  VIX term structure + DXY momentum
      - short_prob    (21D):  HYG/LQD credit, small-cap leadership, liquidity impulse
      - medium_prob   (63D):  Yield curve phase, copper/gold, credit impulse, real rates
      - VRP:                  Variance risk premium — orthogonal to all three buckets above

    Deliberately excluded:
      - bull_prob: already blends fear (via coincident_conditions) and GEX adjustment
        internally inside compute_prob_composite. Using it here would double-count
        both signals.
      - fear_score: enters via coincident_conditions → bull_prob → already in medium/short
      - GEX regime: already applied as gex_adjustment inside compute_prob_composite

    Scoring: each bucket mapped from [0,100] → [-2.5, +2.5] points.
    VRP adds ±0.5 as a tiebreaker.
    Total range: ±10.5, clipped to ±10.
    """
    def _bucket_score(p: float) -> float:
        """Map 0-100 probability to [-2.5, +2.5] score, linear through 50."""
        return (p - 50.0) / 20.0  # 50→0, 70→+1, 30→-1, 90→+2, 10→-2

    tactical = prob.get("tactical_prob", 50.0)
    short    = prob.get("short_prob",    50.0)
    medium   = prob.get("medium_prob",   50.0)

    # Equal-weight the three horizon buckets (no cross-horizon blending bias)
    s = _bucket_score(tactical) + _bucket_score(short) + _bucket_score(medium)

    # VRP is genuinely orthogonal: options pricing premium above/below realised vol
    # Not in any signal bucket. Adds ±0.5 as a tiebreaker only.
    if np.isfinite(vrp_val):
        if vrp_val > 4:    s -= 0.5   # IV much > RV = fear premium = mild bearish
        elif vrp_val > 2:  s -= 0.25
        elif vrp_val < -2: s += 0.25  # IV below RV = complacency or trending
        elif vrp_val < -4: s += 0.5

    return int(np.clip(round(s), -10, 10))

def _verdict(c: int, gex: GammaRegime) -> Tuple[str,str,str]:
    neg=gex in (GammaRegime.NEGATIVE,GammaRegime.STRONG_NEGATIVE)
    pos=gex in (GammaRegime.POSITIVE,GammaRegime.STRONG_POSITIVE)
    if c<=-4: return "BEARISH","#ef4444","Multiple signals aligned bearish."
    if c<=-2:
        return (("BEARISH","#ef4444","Negative macro + negative gamma.") if neg
                else ("LEANING BEARISH","#f97316","Modest bearish lean."))
    if c<=1:
        if neg: return "CAUTIOUS","#f59e0b","Neutral signals but negative gamma — tail risk elevated."
        if pos: return "NEUTRAL / RANGE","#6366f1","Positive gamma → compression and pin."
        return "NEUTRAL","#94a3b8","No strong conviction."
    if c<=3: return "LEANING BULLISH","#10b981","Bullish lean. Credit and liquidity constructive."
    return "BULLISH","#10b981","Broad bullish alignment."

def _bands(spot: float, vix: float) -> Dict:
    dv=vix/100/np.sqrt(252); wv=vix/100/np.sqrt(52)
    return {k:round(v,2) for k,v in {
        "d1lo":spot*(1-dv),"d1hi":spot*(1+dv),
        "d2lo":spot*(1-2*dv),"d2hi":spot*(1+2*dv),
        "w1lo":spot*(1-wv),"w1hi":spot*(1+wv),
        "w2lo":spot*(1-2*wv),"w2hi":spot*(1+2*wv)}.items()}

def _news_cats(cat_intel: dict) -> List[Dict]:
    out=[]
    for k,items in cat_intel.items():
        if not items: continue
        m=INTEL_CATEGORIES.get(k,{})
        sh=category_shock_score(items)
        out.append({"label":m.get("label",k),"icon":m.get("icon","📰"),
                    "sentiment":round(-(sh-30)/70,4),"count":len(items)})
    return sorted(out,key=lambda x:abs(x["sentiment"]),reverse=True)

# ── HTML helpers ──────────────────────────────────────────────────────────────

def _card(body,bg="rgba(255,255,255,0.03)",border="rgba(255,255,255,0.10)"):
    return (f"<div style='background:{bg};border:1px solid {border};border-radius:12px;"
            f"padding:16px 20px;margin-bottom:14px;'>{body}</div>")

def _sh(n,t):
    return (f"<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);"
            f"letter-spacing:0.15em;text-transform:uppercase;margin-bottom:8px;'>{n}. {t}</div>")

def _kv(label,value,color="rgba(255,255,255,0.85)"):
    return (f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
            f"margin-bottom:3px;font-size:13px;'>"
            f"<span style='color:rgba(255,255,255,0.50);'>{label}</span>"
            f"<span style='font-family:monospace;font-weight:600;color:{color};'>{value}</span></div>")

def _tk(sym,price,pct):
    if not np.isfinite(price): return ""
    pc=("#10b981" if pct>=0 else "#ef4444") if np.isfinite(pct) else "#94a3b8"
    ps=(f"+{pct:.2f}%" if pct>=0 else f"{pct:.2f}%") if np.isfinite(pct) else ""
    return (f"<div style='text-align:center;'>"
            f"<div style='font-size:10px;color:rgba(255,255,255,0.4);'>{sym}</div>"
            f"<div style='font-family:monospace;font-size:14px;font-weight:700;'>{price:,.2f}</div>"
            f"<div style='font-size:11px;color:{pc};font-weight:600;'>{ps}</div></div>")

def _pill(text,color="#6366f1"):
    return (f"<span style='background:{color}22;color:{color};border:1px solid {color}44;"
            f"border-radius:6px;padding:2px 8px;font-size:11px;font-weight:600;"
            f"margin-right:4px;display:inline-block;'>{text}</span>")

def _sig(e,t):
    return f"<div style='font-size:12px;margin-bottom:3px;'>{e} {t}</div>"

def _gl(term,defn):
    return (f"<div style='margin-bottom:10px;'>"
            f"<div style='font-size:13px;font-weight:700;color:rgba(255,255,255,0.9);margin-bottom:2px;'>{term}</div>"
            f"<div style='font-size:12px;color:rgba(255,255,255,0.55);line-height:1.55;'>{defn}</div></div>")

# ── Main ──────────────────────────────────────────────────────────────────────


def _gex_histogram(chain_df, spot: float, flip: float,
                   max_dte: int = 45, height: int = 360) -> go.Figure:
    """
    GEX bar chart derived directly from the live options chain.
    Pulls net GEX per strike from compute_gex_from_chain (same engine as
    the GEX Engine page) so the bars are identical to what you see there.

    Green = positive GEX (dealers long gamma, stabilising).
    Red    = negative GEX (dealers short gamma, amplifying).
    Yellow dashed = gamma flip level.
    White dotted  = current spot.
    """
    _C_POS  = "#10b981"
    _C_NEG  = "#ef4444"
    _C_FLIP = "#f59e0b"

    if chain_df is None or len(chain_df) == 0:
        fig = go.Figure()
        plotly_dark(fig, title="GEX Histogram (no chain data)", height=height)
        return fig

    try:
        near_chain = chain_df[chain_df["expiry_T"] <= max_dte / 365.0].copy()
        if near_chain.empty:
            near_chain = chain_df.copy()

        gex_chain = compute_gex_from_chain(near_chain, spot)

        # Aggregate net GEX by strike across all expirations
        agg = (gex_chain.groupby("strike")["net_gex"]
                        .sum()
                        .reset_index()
                        .sort_values("strike"))

        # Filter to ±10% around spot for readability
        lo, hi = spot * 0.90, spot * 1.10
        agg = agg[(agg["strike"] >= lo) & (agg["strike"] <= hi)]

        if agg.empty:
            raise ValueError("No strikes in ±10% range")

        strikes = agg["strike"].tolist()
        vals_m  = (agg["net_gex"] / 1e6).tolist()   # $ millions

        # Auto-scale to $B if large
        max_abs = max(abs(v) for v in vals_m)
        if max_abs >= 5000:
            vals_display = [v / 1000 for v in vals_m]
            unit = "$B"
        else:
            vals_display = vals_m
            unit = "$M"

        colors = [_C_POS if v > 0 else _C_NEG for v in vals_display]

        fig = go.Figure(go.Bar(
            x=strikes,
            y=vals_display,
            marker_color=colors,
            marker_line_width=0,
            opacity=0.88,
            name="Net GEX",
        ))

        # Zero line
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.30)", line_width=1)

        # Spot line
        fig.add_vline(
            x=spot, line_dash="dot",
            line_color="rgba(255,255,255,0.75)", line_width=1.5,
            annotation_text=f"SPOT {spot:.0f}",
            annotation_font_size=10,
            annotation_font_color="rgba(255,255,255,0.85)",
            annotation_position="top right",
        )

        # Gamma flip line
        if flip and lo < flip < hi:
            fig.add_vline(
                x=flip, line_dash="dash",
                line_color=_C_FLIP, line_width=1.5,
                annotation_text=f"FLIP {flip:.0f}",
                annotation_font_size=10,
                annotation_font_color=_C_FLIP,
                annotation_position="top left",
            )

        # Centre x-axis on spot
        x_pad = max(abs(hi - spot), abs(spot - lo)) * 1.05
        fig.update_layout(
            xaxis=dict(range=[spot - x_pad, spot + x_pad], title="Strike"),
            yaxis_title=f"Net GEX ({unit})",
            bargap=0.12,
            showlegend=False,
        )
        plotly_dark(fig, title=f"GEX Histogram — {unit} per Strike (≤{max_dte} DTE)", height=height)
        return fig

    except Exception as exc:
        fig = go.Figure()
        plotly_dark(fig, title=f"GEX Histogram (error: {exc})", height=height)
        return fig


def render_thesis_page():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## 📋 Daily Thesis Briefing")
    st.markdown(f"*{dt.datetime.now().strftime('%A, %B %d, %Y — %H:%M ET')}*  |  Not financial advice.")

    st.sidebar.markdown("### Thesis Controls")
    start=st.sidebar.date_input("Data Start",dt.date.today()-dt.timedelta(days=730),key="th_start")
    end=st.sidebar.date_input("Data End",dt.date.today(),key="th_end")
    gex_sym=st.sidebar.text_input("GEX Symbol","SPY",key="th_gex").upper().strip()
    show_ua=st.sidebar.toggle("Show ES/NQ levels",True,key="th_ua")
    if st.sidebar.button("🔄 Refresh",use_container_width=True,key="th_ref"):
        st.cache_data.clear(); st.rerun()

    idx=pd.date_range(start,end,freq="D")

    with st.spinner("Loading macro data…"):
        raw=load_macro(start.isoformat(),end.isoformat())
        def r(k): return resample_ffill(raw.get(k,pd.Series(dtype=float)),idx)
        y3m=r("DGS3MO");y2=r("DGS2");y10=r("DGS10");y30=r("DGS30")
        cpi=r("CPIAUCSL");core=r("CPILFESL");unrate=r("UNRATE")
        walcl=r("WALCL");tga=r("WTREGEN");rrp=r("RRPONTSYD");m2=r("M2SL")
        nfci=resample_ffill(raw.get("NFCI",pd.Series(dtype=float)),idx).fillna(0)
        vix_s=r("VIX");spy=r("SPY");tlt=r("TLT");qqq=r("QQQ")
        copx=r("COPX");gld=r("GLD");hyg=r("HYG");lqd=r("LQD");dxy=r("UUP");iwm=r("IWM")
        tips=r("DFII10");bres=r("WRBWFRBL");bcred=r("TOTBKCR");gdp=r("GDPC1");mmmf=r("WRMFSL")
        ism_r=raw.get("AMTMNO",pd.Series(dtype=float))
        ism=ism_r if len(ism_r.dropna())>4 else None
        sahm_r=raw.get("SAHM_RULE",pd.Series(dtype=float))
        hy_r=raw.get("BAMLH0A0HYM2",pd.Series(dtype=float))
        sahm=resample_ffill(sahm_r,idx) if len(sahm_r.dropna())>0 else None
        hys=resample_ffill(hy_r,idx) if len(hy_r.dropna())>0 else None
        icsa_r=raw.get("ICSA",pd.Series(dtype=float))
        icsa=resample_ffill(icsa_r,idx) if len(icsa_r.dropna())>0 else None
    core_yoy=(core/core.shift(365)-1)*100
    cpi_yoy=(cpi/cpi.shift(365)-1)*100
    s2s10=(y10-y2)*100
    net_liq=(walcl-tga-rrp)/1000
    nl4w=net_liq.diff(28); bs13=walcl.diff(91)/1000
    cyl=_sl(core_yoy,2.5); crl=_sl(s2s10,0.0)
    macro_reg=classify_macro_regime_abs(cyl,crl)
    gz=zscore(s2s10.fillna(0)); iz=zscore(core_yoy.fillna(cyl))
    vz=zscore(vix_s.fillna(20)); nz=zscore(nfci.fillna(0))
    inv=(s2s10<0).astype(int); lt=(nl4w<0).astype(int)
    fear_raw=0.45*vz+0.35*nz+0.10*inv+0.10*lt
    fear=float(((fear_raw.iloc[-1]+2)/4).clip(0,1)*100)   # 0-100 for dashboard/signals compat
    fear_z=float(fear_raw.iloc[-1])                        # raw z-score for thesis display
    vl=_sl(vix_s,20.0); sahmv=_sl(sahm,0.0) if sahm is not None else 0.0
    # BAMLH0A0HYM2 from FRED is in percent (3.20 = 320bp). Multiply by 100
    # to convert to basis points before any comparisons or display.
    hyv=_sl(hys,3.0)*100 if hys is not None else 300.0
    ur=_sl(unrate,4.0); cyi=_sl(cpi_yoy,2.5); gzv=_sl(gz,0.0); izv=_sl(iz,0.0)
    nl4wv=_sl(nl4w,0.0)
    liq_lab="Expanding" if nl4wv>=0 else "Contracting"
    y10_20=y10.diff(20)
    # ICSA 4-week % change: weekly, leads UNRATE, orthogonal to Sahm Rule.
    # Replaces the old u3m (UNRATE 90-day diff) which was redundant with Sahm.
    if icsa is not None and icsa.dropna().size > 28:
        _icsa_clean = icsa.dropna()
        icsa_4w_chg = float(_icsa_clean.pct_change(28).dropna().iloc[-1])
    else:
        icsa_4w_chg = 0.0
    u3m = icsa_4w_chg  # keep u3m alias for fp formula below (unrate direction proxy)
    warsh=((y10.diff(20)<0)&(bs13<0)).astype(int)
    spydd=(spy/spy.rolling(126).max()-1).fillna(0)
    tp=float(np.clip(45+35*float(spydd.iloc[-1]<=-0.07)+20*(fear_z>1.0),0,100))
    fp=float(np.clip(55+25*float((y10_20.iloc[-1]<0)and(u3m>0))-10*float((cyl-3.0)>0)-15*float(warsh.iloc[-1]),0,100))
    tgadd=(tga.diff(28)<0).astype(int); rrpd=(rrp<50).astype(int)
    trp=float(np.clip(50+20*float(tgadd.iloc[-1])+15*float(rrpd.iloc[-1])+15*float(nl4w.iloc[-1]>=0),0,100))
    threeP=float(np.clip(0.35*trp+0.35*fp+0.30*tp,0,100))
    # Recession Stress Index — regime-calibrated stress score, NOT a calibrated probability
    # Using pre-quotes yield curve (crl) and pre-quotes VIX (vl); updated after quotes below
    _stress = _recession_stress(sahmv, hyv, crl, icsa_4w_chg, vl)
    rec     = _stress["score"]
    rec_lbl = _stress["label"]
    sr=_to_1d(spy).reindex(idx).ffill().pct_change().dropna()
    tr2=_to_1d(tlt).reindex(idx).ffill().pct_change().reindex(sr.index).dropna()
    stlc=round(float(sr.rolling(21).corr(tr2).dropna().iloc[-1]),3) if sr.dropna().size>21 else float("nan")
    # CPI MoM: use raw monthly FRED series (pct_change(1) = month-over-month).
    # Do NOT use the daily-ffilled `cpi` series — pct_change(21) on a ffilled
    # monthly series returns 0 on almost every day because the value is flat
    # between FRED releases. The raw series has one observation per month.
    _cpi_raw = raw.get("CPIAUCSL", pd.Series(dtype=float)).dropna()
    cpi_now = round(float(_cpi_raw.pct_change(1).dropna().iloc[-1]) * 100, 3) if len(_cpi_raw) > 1 else float("nan")
    rd=_retdist(spy,idx,0)

    with st.spinner("Fetching live quotes…"):
        q=_quotes()

    spx=q.get("SPX_last",_sl(spy)*10); ndx=q.get("NDX_last",_sl(qqq)*40)
    es=q.get("ES_last",spx); nq=q.get("NQ_last",ndx)
    dxyv=q.get("DXY_last",float("nan")); gldv=q.get("GLD_last",float("nan"))
    tnxv=q.get("TNX_last",float("nan"))
    tnxv=tnxv/10 if (np.isfinite(tnxv) and tnxv>20) else tnxv
    irxv=q.get("IRX_last",float("nan"))
    irxv=irxv/10 if (np.isfinite(irxv) and irxv>20) else irxv
    s2s10v=(tnxv-irxv)*100 if (np.isfinite(tnxv) and np.isfinite(irxv)) else crl
    vix_live=q.get("VIX_last",vl)
    vxn_live=q.get("VXN_last", vix_live * 1.10)  # VXN ~10% higher than VIX historically
    vrp=_vrp_full(vix_live,spy,idx)
    vts=_vts(q); tail=_tail(q)
    rd=_retdist(spy,idx,spx)
    b=_bands(spx,vix_live)

    gex_spot=spx if gex_sym in ("SPY","SPX") else ndx
    with st.spinner("Fetching options data…"):
        client=get_schwab_client(); chain_df=None
        if client:
            chain_df=schwab_get_options_chain(client,gex_sym,spot=None)
            if chain_df is not None and len(chain_df)>0:
                gex_spot=schwab_get_spot(client,gex_sym) or gex_spot
        if chain_df is None or len(chain_df)==0:
            chain_df,gex_spot,_=get_gex_from_yfinance(gex_sym)
        if chain_df is not None and gex_spot:
            gex_st=build_gamma_state(chain_df,float(gex_spot),"live",max_dte=45)
            gwas=compute_gwas(chain_df,float(gex_spot))
            tstr=compute_gex_term_structure(chain_df,float(gex_spot))
            fl=compute_flow_imbalance(chain_df,float(gex_spot))
            net_gex=gex_st.total_gex; gex_score=int(np.clip(net_gex/1e9*10,-50,50))
            # ATM IV from chain: strikes within ±1% of spot, averaged across expirations
            _atm_m = chain_df["strike"].between(float(gex_spot)*0.99, float(gex_spot)*1.01)
            _atm_v = chain_df[_atm_m]["iv"].dropna()
            atm_iv_chain = float(_atm_v.mean() * 100) if len(_atm_v) > 0 else None
        else:
            gex_st=GammaState(data_source="unavailable",timestamp=dt.datetime.now().strftime("%H:%M:%S"))
            gwas=tstr=fl={}; net_gex=gex_score=0; atm_iv_chain=None

    flip=gex_st.gamma_flip or gex_spot
    upper=gex_st.key_resistance[0] if gex_st.key_resistance else gex_spot*1.03
    lower=gex_st.key_support[0] if gex_st.key_support else gex_spot*0.97
    gex_reg=gex_st.regime.value if hasattr(gex_st.regime,"value") else str(gex_st.regime)
    gex_op_label=REGIME_OPERATIONAL_LABEL.get(gex_st.regime, gex_reg)  # e.g. DEALERS_SELL_RALLIES
    gex_rc=("#ef4444" if "NEGATIVE" in gex_reg.upper() or "SELL" in gex_op_label
            else "#10b981" if "POSITIVE" in gex_reg.upper() or "BUY" in gex_op_label else "#f59e0b")
    gwas_a=gwas.get("gwas_above") if gwas else None
    gwas_b=gwas.get("gwas_below") if gwas else None
    dur=tstr.get("durability","N/A").upper() if tstr else "N/A"
    frag=tstr.get("fragility_ratio",0.5)*100 if tstr else 50.0
    pcr=fl.get("pc_ratio",1.0) if fl else 1.0
    fb=fl.get("flow_bias","neutral") if fl else "neutral"
    _fl_using_vol = fl.get("using_volume", False) if fl else False
    _fl_bias_label = "Flow Bias" if _fl_using_vol else "OI Bias (inventory)"

    with st.spinner("Computing signals…"):
        leading=compute_leading_stack(
            y2,y3m,y10,y30,s2s10,vix_s,m2,pd.Series(dtype=float),
            copx,gld,hyg,lqd,dxy,spy,qqq,iwm,net_liq,nl4w,walcl,bs13,idx,
            tips_10y=tips,bank_reserves=bres,bank_credit=bcred,ism_no=ism,gdp_quarterly=gdp,mmmf=mmmf)
        meta=regime_transition_prob(macro_reg,core_yoy,s2s10)
        nc=float(current_pct_rank(-_to_1d(nfci).reindex(idx).ffill(),252))
        lc=float(50.0+np.sign(nl4wv)*20)
    with st.spinner("Loading feeds…"):
        try:
            rss=load_feeds(tuple(_all_feeds_flat().items()),60)
            geo,_=geo_shock_score(rss); cat_intel=categorise_items(rss)
        except Exception:
            geo=0.0; cat_intel={k:[] for k in INTEL_CATEGORIES}
    prob=compute_prob_composite(leading,fear,geo,meta["p_change_20d"],gex_st,nfci_coincident=nc,liq_dir_coincident=lc)
    p1d=compute_1d_prob(gex_state=gex_st,spot=float(gex_spot),vix_level=vix_live,
        vix_series=vix_s,spy_series=spy,hyg_series=hyg,lqd_series=lqd,dxy_series=dxy,
        s_2s10s=s2s10,net_liq_4w=nl4w,nfci_z=nz,fear_score=fear,
        session=get_session_context(),idx=idx,sahm_rule=sahm,hy_spread=hys)

    comp=_composite(prob,vrp["val"])
    vrd,vc,ve=_verdict(comp,gex_st.regime)
    news=_news_cats(cat_intel)
    ua="NQ" if gex_sym in ("QQQ","NDX") else "ES"
    um=40.0 if ua=="NQ" else 10.0
    def _ua(p): return f"{p*um:,.0f}"
    reg_col={"Goldilocks":"#10b981","Overheating":"#f59e0b","Stagflation":"#ef4444","Deflation":"#6366f1"}.get(macro_reg,"#94a3b8")
    fl2="ELEVATED" if fear_z>1.0 else "MODERATE" if fear_z>0.0 else "COMPLACENT" if fear_z<-1.0 else "LOW"
    fc="#ef4444" if fear_z>1.0 else "#f59e0b" if fear_z>0.0 else "#10b981"

    # ── 1. MARKET REGIME ──────────────────────────────────────────────────
    hdr=(f"<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px;'>"
         +_pill(f"Market Regime: {macro_reg}",reg_col)
         +_pill(f"Fear Level: {fl2} ({fear_z:+.2f}σ)",fc)
         +_pill(f"Liquidity: {liq_lab} (${abs(nl4wv):.0f}B)","#10b981" if nl4wv>=0 else "#ef4444")
         +_pill(f"Recession P(6m): {rec:.1f}%","#ef4444" if rec>50 else "#f59e0b" if rec>30 else "#10b981")
         +"</div>")
    strip=("<div style='display:grid;grid-template-columns:repeat(9,1fr);gap:6px;'>"
           +_tk("SPX",spx,q.get("SPX_pct",float("nan")))
           +_tk("NDX",ndx,q.get("NDX_pct",float("nan")))
           +_tk("VIX",vix_live,q.get("VIX_pct",float("nan")))
           +_tk("ES",es,q.get("ES_pct",float("nan")))
           +_tk("NQ",nq,q.get("NQ_pct",float("nan")))
           +_tk("DXY",dxyv,q.get("DXY_pct",float("nan")))
           +_tk("10Y",tnxv,float("nan"))
           +_tk("2s10s",s2s10v,float("nan"))
           +_tk("GLD",gldv,q.get("GLD_pct",float("nan")))
           +"</div>")
    st.markdown(_card(_sh(1,"MARKET REGIME")+hdr+strip),unsafe_allow_html=True)

    # ── 2. VOLATILITY REGIME ──────────────────────────────────────────────
    vc2=("#ef4444" if vrp["regime"]=="CHEAP" else "#10b981" if vrp["regime"]=="RICH" else "#94a3b8")
    tc=("#ef4444" if vts["shape"]=="BACKWARDATION" else "#10b981" if vts["shape"]=="CONTANGO" else "#f59e0b")
    trc=("#ef4444" if tail["regime"]=="ELEVATED" else "#f59e0b" if tail["regime"]=="MODERATE" else "#10b981")
    vol=(  _sh(2,"VOLATILITY REGIME")
         +"<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px 32px;'><div>"
         +"<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>VRP</div>"
         +_kv("Value",f"{vrp['val']:+.4f}" if np.isfinite(vrp['val']) else "N/A",vc2)
         +_kv("Z-Score",f"{vrp['z']:.2f}" if np.isfinite(vrp['z']) else "N/A")
         +_kv("Pct",f"{vrp['pct']:.0f}%" if np.isfinite(vrp['pct']) else "N/A")
         +_kv("Regime",vrp["regime"],vc2)
         +"<div style='margin-top:8px;font-size:10px;color:rgba(255,255,255,0.4);letter-spacing:0.1em;'>TERM STRUCTURE</div>"
         +_kv("Shape",vts["shape"],tc)
         +_kv("VIX/VIX3M",f"{vts['ratio']:.3f}" if np.isfinite(vts.get('ratio',float('nan'))) else "N/A")
         +_kv("Carry",f"{vts['carry']:.2f}%" if np.isfinite(vts.get('carry',float('nan'))) else "N/A")
         +"<div style='margin-top:8px;font-size:10px;color:rgba(255,255,255,0.4);letter-spacing:0.1em;'>TAIL RISK</div>"
         +_kv("VVIX",f"{tail['vvix']:.1f}" if np.isfinite(tail['vvix']) else "N/A")
         +_kv("VVIX/VIX",f"{tail['ratio']:.2f}" if np.isfinite(tail['ratio']) else "N/A",trc)
         +_kv("Regime",tail["regime"],trc)
         +"</div><div>"
         +"<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>IV vs RV</div>"
         +_kv("VIX (IV)",f"{vix_live:.2f}")
         +_kv("21D RV",f"{vrp['rv21']:.2f}%" if np.isfinite(vrp['rv21']) else "N/A")
         +_kv("VRP Spread",f"{vrp['spread']:+.2f}" if np.isfinite(vrp['spread']) else "N/A",vc2)
         +"<div style='margin-top:8px;font-size:10px;color:rgba(255,255,255,0.4);letter-spacing:0.1em;'>ATM IV</div>"
         +_kv("SPX ATM IV",
              f"{atm_iv_chain:.1f}%" if (atm_iv_chain and gex_sym in ("SPY","SPX"))
              else f"{vix_live:.1f}% (VIX proxy)")
         +_kv("NDX ATM IV",
              f"{atm_iv_chain:.1f}%" if (atm_iv_chain and gex_sym in ("QQQ","NDX"))
              else f"{q.get('VIX3M_last', vix_live * 0.92):.1f}% (VIX3M proxy)")
         +"</div></div>")
    st.markdown(_card(vol),unsafe_allow_html=True)

    # ── 3. GEX & DEALER POSITIONING ───────────────────────────────────────
    if "NEGATIVE" in gex_reg.upper():
        narr=f"Negative dealer flow ({gex_score:+d}): Dealers short gamma — amplifying moves. Expect trending/volatile behavior."
    elif "POSITIVE" in gex_reg.upper():
        narr=f"Positive dealer flow ({gex_score:+d}): Dealers long gamma — suppressing moves. Expect mean-reversion."
    else:
        narr=f"Neutral dealer positioning ({gex_score:+d}): Near gamma flip — binary risk, reduce size."

    # Section 3: show both ETF and SPX-equivalent levels
    _g3_mult = 10.0 if gex_sym in ("SPY",) else 40.0 if gex_sym in ("QQQ",) else 1.0
    _g3_label = "SPX" if gex_sym in ("SPY","SPX") else "NDX" if gex_sym in ("QQQ","NDX") else gex_sym
    gleft=("<div>"
           +f"<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>GEX LEVELS ({gex_sym} → {_g3_label})</div>"
           +_kv(f"{_g3_label} Spot",f"{spx:,.2f}","#fff")
           +_kv("GEX Flip",f"{flip*_g3_mult:,.2f}","#f59e0b")
           +_kv("GEX Upper",f"{upper*_g3_mult:,.2f}","#10b981")
           +_kv("GEX Lower",f"{lower*_g3_mult:,.2f}","#ef4444")
           +_kv("GWAS Above",f"{gwas_a*_g3_mult:,.2f}" if gwas_a else "N/A","#6366f1")
           +_kv("GWAS Below",f"{gwas_b*_g3_mult:,.2f}" if gwas_b else "N/A","#6366f1")
           +_kv("GEX Regime",gex_op_label,gex_rc)
           +"</div>")
    gright=("<div>"
            +"<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>OPTIONS FLOW</div>"
            +_kv("P/C Dollar Ratio",f"{pcr:.2f}","#ef4444" if pcr>1.3 else "#10b981" if pcr<0.8 else "#94a3b8")
            +_kv(_fl_bias_label,fb.upper(),"#ef4444" if fb=="bearish" else "#10b981" if fb=="bullish" else "#94a3b8")
            +_kv("GEX Duration",f"{dur} ({frag:.0f}% weekly)","#ef4444" if dur=="FRAGILE" else "#10b981")
            +_kv("Net GEX",f"${gex_st.total_gex/1e9:.1f}B" if (gex_st.total_gex and abs(gex_st.total_gex)>=1e9) else f"${gex_st.total_gex/1e6:.0f}M" if gex_st.total_gex else "N/A")
            +_kv("Dist to Flip",f"{gex_st.distance_to_flip_pct:+.2f}%"))
    if show_ua:
        gright+=("<div style='margin-top:8px;border-top:1px solid rgba(255,255,255,0.08);padding-top:6px;'>"
                 +f"<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>{ua} EQUIVALENT</div>"
                 +_kv(f"{ua} Spot",_ua(float(gex_spot)))
                 +_kv(f"{ua} Flip",_ua(flip),"#f59e0b")
                 +_kv(f"{ua} Upper",_ua(upper),"#10b981")
                 +_kv(f"{ua} Lower",_ua(lower),"#ef4444")
                 +"</div>")
    gright+="</div>"
    gex_body=(_sh(3,"GEX LEVELS & DEALER POSITIONING")
              +"<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px 32px;'>"
              +gleft+gright+"</div>"
              +f"<div style='margin-top:10px;border-top:1px solid rgba(255,255,255,0.08);padding-top:8px;"
              +f"font-size:12px;color:rgba(255,255,255,0.65);font-style:italic;'>"
              +f"<span style='color:{gex_rc};font-weight:700;'>{gex_op_label}</span> — {narr}</div>")
    st.markdown(_card(gex_body),unsafe_allow_html=True)

    # ── 3b. GEX HISTOGRAM ────────────────────────────────────────────────
    st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>GEX HISTOGRAM — Net Gamma Exposure per Strike</div>",unsafe_allow_html=True)
    st.plotly_chart(
        _gex_histogram(chain_df if chain_df is not None else None, float(gex_spot), float(flip)),
        use_container_width=True, key="th_gex_hist"
    )

    # ── 4. PROBABILITY HEATMAP ────────────────────────────────────────────
    st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>4. SPX / NDX FORWARD PROBABILITY HEATMAP — 1-WEEK FORECAST</div>",unsafe_allow_html=True)
    st.caption("GBM Monte Carlo (n=6,000) · VIX/VXN implied vol · Risk-neutral drift · 5 trading days")
    with st.spinner("Running simulations…"):
        st.plotly_chart(
            _forward_prob_fig(
                spx=spx, vix=vix_live,
                ndx=ndx, vxn=vxn_live,
                spy=spy, qqq=qqq, idx=idx,
            ),
            use_container_width=True, key="th_hm"
        )

    # ── 5 & 6. IV SURFACES ────────────────────────────────────────────────
    c5,c6=st.columns(2)
    with c5:
        st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>5. IMPLIED VOL SURFACE — SPX</div>",unsafe_allow_html=True)
        st.plotly_chart(_ivsurf_from_chain(chain_df if "chain_df" in dir() else None, float(gex_spot), "SPX", vix_live, vts.get("shape","MIXED")), use_container_width=True, key="th_ivs_spx")
    with c6:
        st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>6. IMPLIED VOL SURFACE — NDX</div>",unsafe_allow_html=True)
        # Only pass the chain if it's actually NDX/QQQ data — do NOT fake NDX
        # surface by scaling SPY/SPX chain with a kludge multiplier.
        _ndx_chain = (chain_df if ("chain_df" in dir() and chain_df is not None
                                   and gex_sym in ("QQQ", "NDX", "^NDX"))
                      else None)
        st.plotly_chart(_ivsurf_from_chain(_ndx_chain, float(gex_spot), "NDX", vix_live, vts.get("shape","MIXED")), use_container_width=True, key="th_ivs_ndx")

    # ── 7. IV vs RV ───────────────────────────────────────────────────────
    st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>7. IMPLIED vs REALIZED VOLATILITY</div>",unsafe_allow_html=True)
    st.plotly_chart(_ivrv_fig(vix_s,spy,idx),use_container_width=True,key="th_ivrv")

    # ── 8. RETURN DISTRIBUTION ────────────────────────────────────────────
    st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>8. RETURN DISTRIBUTION & PROBABILITY BANDS</div>",unsafe_allow_html=True)
    if rd:
        st.markdown(f"SPX Spot: `{spx:,.2f}` &nbsp;&nbsp; Daily σ: `{rd['daily_sigma']:.4f}%` &nbsp;&nbsp; Skew: `{rd['skew']:.4f}` &nbsp;&nbsp; Kurtosis: `{rd['kurtosis']:.4f}`")
        st.plotly_chart(_rdist_fig(spy,idx,rd,spot=spx),use_container_width=True,key="th_rd")

    # ── 9. MACRO REGIME & NEWS ────────────────────────────────────────────
    nr=""
    for cat in news[:6]:
        s=cat["sentiment"]; col=("#10b981" if s>0.01 else "#ef4444" if s<-0.01 else "#94a3b8")
        sign="+" if s>0.00005 else ("" if abs(s)<0.00005 else "")
        nr+=(f"<span style='font-size:11px;font-family:monospace;margin-right:10px;'>"
             f"<span style='color:rgba(255,255,255,0.5);'>{cat['icon']} {cat['label'].split('&')[0].strip().lower()}</span>: "
             f"<span style='color:{col};font-weight:600;'>{sign}{s:.4f}</span> "
             f"<span style='color:rgba(255,255,255,0.3);'>({cat['count']} articles)</span></span>")
    mac=(_sh(9,"MACRO REGIME & NEWS SENTIMENT")
         +f"<div style='font-size:16px;font-weight:700;color:{reg_col};margin-bottom:8px;'>{macro_reg}</div>"
         +"<div style='display:grid;grid-template-columns:1fr 1fr;gap:4px 32px;'><div>"
         +_kv("Growth Z",f"{gzv:+.2f}","#10b981" if gzv>0 else "#ef4444")
         +_kv("Inflation Z",f"{izv:+.2f}","#f59e0b" if izv>0.5 else "#94a3b8")
         +_kv("CPI YoY",f"{cyi:.2f}%")
         +_kv("CPI Nowcast",f"{cpi_now:+.3f}% MoM" if np.isfinite(cpi_now) else "N/A")
         +"</div><div>"
         +_kv("Unemployment",f"{ur:.1f}%")
         +_kv("HY OAS",f"{hyv:.0f}bp","#ef4444" if hyv>450 else "#f59e0b" if hyv>350 else "#94a3b8")
         +_kv("SPY-TLT Corr",f"{stlc:.3f}" if np.isfinite(stlc) else "N/A",
              "#ef4444" if (np.isfinite(stlc) and stlc>0.2) else "#94a3b8")
         +_kv("Sahm Rule",f"{sahmv:.3f}",
              "#ef4444" if sahmv>=0.5 else "#f59e0b" if sahmv>=0.3 else "#10b981")
         +"</div></div>"
         +f"<div style='margin-top:10px;border-top:1px solid rgba(255,255,255,0.08);padding-top:8px;'>"
         +"<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:5px;letter-spacing:0.1em;'>NEWS SENTIMENT</div>"
         +f"<div style='display:flex;flex-wrap:wrap;gap:4px;'>{nr}</div></div>")
    st.markdown(_card(mac),unsafe_allow_html=True)

    # ── 10. THESIS VERDICT ────────────────────────────────────────────────
    cc="#10b981" if comp>0 else "#ef4444" if comp<0 else "#94a3b8"
    vp=np.isfinite(vrp["val"]) and vrp["val"]>0
    vs2=f"{vrp['val']:+.4f}" if np.isfinite(vrp["val"]) else "N/A"
    sigs=(_sig("✅" if vp else "⚠️",f"VRP {'positive' if vp else 'negative'} ({vs2})")
          +_sig("⚠️" if fear_z>0.5 else "✅",f"Fear composite {fl2} ({fear_z:+.2f}σ)")
          +_sig("🔴" if rec>60 else "🟡" if rec>35 else "🟢",
                f"Recession risk {'elevated' if rec>60 else 'moderate' if rec>35 else 'low'} ({rec:.1f}%)"))
    # Convert GEX levels to SPX scale for display (GEX symbol may be SPY=1/10 SPX)
    _kls_mult = 10.0 if gex_sym in ("SPY","SPX") else 40.0 if gex_sym in ("QQQ","NDX") else 1.0
    _flip_disp  = flip  * _kls_mult if gex_sym in ("SPY","QQQ") else flip
    _upper_disp = upper * _kls_mult if gex_sym in ("SPY","QQQ") else upper
    _lower_disp = lower * _kls_mult if gex_sym in ("SPY","QQQ") else lower
    kls=(_kv("SPX Spot",f"{spx:,.2f}","#fff")
         +_kv("GEX Flip",f"{_flip_disp:,.2f}","#f59e0b")
         +_kv("GEX Upper",f"{_upper_disp:,.2f}","#10b981")
         +_kv("GEX Lower",f"{_lower_disp:,.2f}","#ef4444")
         +_kv("1σ Daily",f"{b['d1lo']:,.2f} — {b['d1hi']:,.2f}")
         +_kv("2σ Daily",f"{b['d2lo']:,.2f} — {b['d2hi']:,.2f}")
         +_kv("1σ Weekly",f"{b['w1lo']:,.2f} — {b['w1hi']:,.2f}")
         +_kv("2σ Weekly",f"{b['w2lo']:,.2f} — {b['w2hi']:,.2f}"))
    risks=[]
    if fear_z>1.0: risks.append(("⚠️","Elevated fear composite — potential for sharp moves"))
    if rec>50: risks.append(("⚠️",f"Recession probability at {rec:.1f}% — monitor labor data"))
    if np.isfinite(stlc) and stlc>0.2: risks.append(("⚠️","Positive stock-bond correlation — diversification impaired"))
    if "SELL" in gex_op_label or "NEGATIVE" in gex_reg.upper(): risks.append(("🔴","Negative gamma regime — dealer hedging amplifies moves. No fading."))
    if dur=="FRAGILE": risks.append(("⚠️",f"GEX regime fragile — {frag:.0f}% of gamma ≤7 DTE. Levels expire by Friday."))
    if leading.get("corr_regime") in ("STRESS","SYSTEMIC"): risks.append(("🔴",f"Cross-asset correlation: {leading.get('corr_regime')} — credit leading equity lower"))
    if vts["shape"]=="BACKWARDATION": risks.append(("⚠️","VIX backwardation — near-term stress priced above medium-term"))
    if not risks: risks.append(("✅","No major risk flags. Conditions broadly constructive."))
    rr="".join(_sig(e,t) for e,t in risks)
    vbd=(_sh(10,f"THESIS VERDICT: {vrd}")
         +f"<div style='display:flex;align-items:baseline;gap:16px;margin-bottom:8px;'>"
         +f"<div style='font-size:26px;font-weight:800;color:{vc};'>{vrd}</div>"
         +f"<div style='font-size:13px;color:rgba(255,255,255,0.5);'>Composite Score: <span style='font-family:monospace;font-weight:700;color:{cc};'>{comp:+d}</span> / ±10</div>"
         +f"<div style='font-size:12px;color:rgba(255,255,255,0.35);'>Date: {dt.date.today().strftime('%A, %B %d, %Y')}</div></div>"
         +f"<div style='font-size:12px;color:rgba(255,255,255,0.55);margin-bottom:10px;font-style:italic;'>{ve}</div>"
         +"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px 24px;'>"
         +"<div><div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>SIGNAL BREAKDOWN</div>"+sigs+"</div>"
         +"<div><div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>KEY LEVELS</div>"+kls+"</div>"
         +"<div><div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>RISK FACTORS</div>"+rr+"</div>"
         +"</div>")
    st.markdown(_card(vbd,bg=f"{vc}08",border=f"{vc}30"),unsafe_allow_html=True)

    # ── 11-13. GLOSSARIES ─────────────────────────────────────────────────
    g11,g12,g13=st.tabs(["📖 Glossary: Vol","📖 Glossary: Macro","📖 Glossary: Charts"])
    with g11:
        st.markdown(_card(
            _sh(11,"GLOSSARY — WHAT EVERYTHING MEANS")
            +_gl("VRP (Variance Risk Premium)",
                 "Difference between implied vol (VIX) and realized vol. Positive VRP = options expensive vs actual moves. "
                 "Traders sell vol when VRP is high. Negative VRP = market underpricing realized moves → buy protection.")
            +_gl("GEX (Gamma Exposure)",
                 "Total gamma held by options dealers. Positive GEX = dealers long gamma → buy dips, sell rips → suppresses vol. "
                 "Negative GEX = dealers short gamma → amplify moves. GEX Flip = strike where dealer gamma flips sign.")
            +_gl("VIX Term Structure",
                 "Curve of implied vol across expirations. Contango (VIX3M > VIX) = normal. "
                 "Backwardation (VIX > VIX3M) = near-term stress elevated, hedging demand high.")
            +_gl("Tail Risk (VVIX/VIX)",
                 "VVIX = vol of VIX (vol-of-vol). High ratio = market pricing sharp VIX spikes = crash insurance expensive. "
                 "Ratio > 5.5 = elevated. < 4.5 = low.")),unsafe_allow_html=True)
    with g12:
        st.markdown(_card(
            _sh(12,"GLOSSARY — MACRO & PROBABILITY")
            +_gl("σ (Sigma / Standard Deviation)",
                 "1σ ≈ 68% of expected moves, 2σ ≈ 95%, 3σ ≈ 99.7%. Daily σ of 1% → ±1% range 68% of the time.")
            +_gl("Skewness & Kurtosis",
                 "Skewness = asymmetry. Negative skew = more large down moves. "
                 "High kurtosis = fat tails (extreme moves more common than normal distribution predicts).")
            +_gl("Recession P(6m)",
                 "Blends Sahm Rule, HY OAS, and Three Puts backstop. Sahm: 3M avg unemployment up 0.5% above 12M low = recession onset.")
            +_gl("Net Liquidity",
                 "Fed Balance Sheet minus TGA minus RRP. Expanding = risk asset tailwind. Contracting = headwind.")),unsafe_allow_html=True)
    with g13:
        st.markdown(_card(
            _sh(13,"GLOSSARY — READING THE CHARTS")
            +_gl("IV Surface","3D plot: implied vol across strikes (moneyness) and DTE. Steep put skew = downside protection expensive.")
            +_gl("Probability Heatmap","GBM Monte Carlo (n=6,000). Shows probability of SPX/NDX reaching each price level over the next 5 trading days. VIX/VXN implied vol inputs. Includes terminal distribution, scenario bands, and tail probability summary.")
            +_gl("GEX Histogram","Gamma per strike. Green = positive (dealers stabilize). Red = negative (dealers amplify).")
            +_gl("IV vs RV Chart","VIX overlaid with 21D realized vol. VRP spread: green = IV premium (sell vol), red = IV discount (buy protection).")
            +_gl("Fear Composite","VIX z-score + NFCI + curve inversion + liquidity tightening. Reported as σ from mean. >+1.0σ = elevated fear. <−1.0σ = complacency.")),unsafe_allow_html=True)

    st.markdown(
        f"<div style='text-align:center;font-size:10px;color:rgba(255,255,255,0.2);margin-top:16px;'>"
        f"Data: FRED · yfinance · Schwab API | Generated {dt.datetime.now().strftime('%H:%M ET')} | Not financial advice</div>",
        unsafe_allow_html=True)
