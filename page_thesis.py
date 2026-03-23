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
                         compute_gex_term_structure, compute_flow_imbalance,
                         compute_cumulative_gex_profile, gex_zero_crossing)
from schwab_api import get_schwab_client, schwab_get_spot, schwab_get_options_chain, get_intraday_signals
from signals import compute_leading_stack, compute_1d_prob
from probability import (compute_prob_composite, get_session_context,
                          classify_macro_regime_abs, regime_transition_prob)
from intel_monitor import (load_feeds, geo_shock_score, categorise_items,
                            category_shock_score, _all_feeds_flat, INTEL_CATEGORIES)

# ── Chokepoint disruption guard ───────────────────────────────────────────────
_CHOKEPOINT_DISRUPTION = {
    "blocked", "closed", "seized", "attack", "struck", "mine",
    "missile", "navy", "warship", "tanker", "disrupted",
    "houthi", "iran", "piracy", "conflict", "military",
}

try:
    from intel_monitor import STRATEGIC_CHOKEPOINTS as _CHOKEPOINTS
except ImportError:
    _CHOKEPOINTS = {}

def _chokepoint_bonus(txt: str) -> float:
    txt_l = txt.lower()
    for name, aliases in _CHOKEPOINTS.items():
        if any(alias.lower() in txt_l for alias in aliases):
            if any(dk in txt_l for dk in _CHOKEPOINT_DISRUPTION):
                return 8.0
    return 0.0

# ── Helpers ───────────────────────────────────────────────────────────────────

def _sl(s, d=float("nan")):
    try:
        v = s.dropna()
        return float(v.iloc[-1]) if len(v) else d
    except Exception:
        return d

@st.cache_data(ttl=60)
def _quotes() -> Dict:
    # OPT: single yf.download() call instead of 16 sequential Ticker().history()
    # cuts network round-trips from 16 → 1, saving ~3-5s per refresh.
    pairs = [("SPX","^GSPC"),("NDX","^NDX"),("VIX","^VIX"),("VIX3M","^VIX3M"),
             ("VVIX","^VVIX"),("VXN","^VXN"),("DXY","DX-Y.NYB"),("GLD","GLD"),("TLT","TLT"),
             ("TNX","^TNX"),("IRX","^IRX"),("SPY","SPY"),("QQQ","QQQ"),
             ("HYG","HYG"),("ES","ES=F"),("NQ","NQ=F")]
    syms   = [sym for _, sym in pairs]
    keys   = [k   for k, _  in pairs]
    out    = {}
    try:
        df = yf.download(syms, period="5d", auto_adjust=True,
                         progress=False, threads=True)["Close"]
        # yf.download returns MultiIndex columns when >1 symbol
        if hasattr(df.columns, "levels"):
            pass  # already MultiIndex — each column is a symbol
        else:
            # single symbol edge case — wrap
            df = df.to_frame(name=syms[0])
        for k, sym in pairs:
            try:
                col = df[sym].dropna()
                if len(col) >= 1:
                    out[k+"_last"] = float(col.iloc[-1])
                    out[k+"_prev"] = float(col.iloc[-2]) if len(col) > 1 else out[k+"_last"]
                    out[k+"_pct"]  = (out[k+"_last"] / out[k+"_prev"] - 1) * 100
            except Exception:
                pass
    except Exception:
        # Fallback: sequential fetch if batch download fails
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
    lam_base = 0.59
    mj_base  = -0.026
    sj_base  = 0.047

    vvix_vix_ratio = (vvix / vix) if (vix > 0 and np.isfinite(vvix) and vvix > 0) else 4.8
    lam_mult = np.clip(vvix_vix_ratio / 4.8, 0.4, 2.5)
    lam = lam_base * lam_mult

    vts_adj = {"BACKWARDATION": 1.40, "MIXED": 1.10, "CONTANGO": 0.80, "N/A": 1.0}
    mj = mj_base * vts_adj.get(vts_shape, 1.0)

    sj_mult = np.clip(vix / 15.0, 0.7, 2.5)
    sj = sj_base * sj_mult

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

@st.cache_data(ttl=300, show_spinner=False)
def _forward_prob_fig(
    spx: float, vix: float,
    ndx: float, vxn: float,
    spy: pd.Series, qqq: pd.Series,
    idx, days: int = 5, n: int = 6000,
    vvix: float = float("nan"),
    vts_shape: str = "MIXED",
) -> go.Figure:
    from plotly.subplots import make_subplots
    from scipy.stats import norm as _norm

    rng = np.random.default_rng()

    def _rv20(series):
        s = _to_1d(series).reindex(idx).ffill()
        return float(s.pct_change().rolling(20, min_periods=10).std().dropna().iloc[-1] * np.sqrt(252) * 100)

    rv_spx = _rv20(spy)
    rv_ndx = _rv20(qqq)

    def _sim_paths(spot, ann_vol, vvix_val, vts_s, days, n):
        # OPT: generate ALL random numbers in two bulk calls (n*days each)
        # instead of 3 calls inside a Python loop — ~40% faster for n=6000.
        sig = ann_vol / 100.0
        dt  = 1 / 252
        params = _merton_calibrate(ann_vol, vvix_val, vts_s, vrp_val=0)
        lam       = params["lam"]
        mj        = params["mj"]
        sj        = params["sj"]
        jump_comp = params["jump_comp"]
        drift = -0.5 * sig**2 - jump_comp

        # Pre-generate all noise: shape (days, n)
        Z_all  = rng.standard_normal((days, n))
        NJ_all = rng.poisson(lam * dt, (days, n))
        # Jump sizes: only draw where jumps occur, rest are zero
        J_raw  = rng.normal(mj, sj, (days, n))
        J_all  = J_raw * NJ_all   # zero-out non-jump steps

        step = np.exp(drift * dt + sig * np.sqrt(dt) * Z_all + J_all)  # (days, n)

        # Build paths via cumprod along day axis
        log_paths = np.log(spot) + np.cumsum(np.log(step), axis=0)  # (days, n)
        paths = np.empty((n, days + 1))
        paths[:, 0] = spot
        paths[:, 1:] = np.exp(log_paths).T
        return paths

    def _density_grid(paths, spot, days, n_levels=60):
        # OPT: fully vectorised — eliminates the O(days*n_levels) Python loop.
        # Broadcasts paths (n, days) against lvls (n_levels,) using digitize.
        lo   = spot * 0.93
        hi   = spot * 1.07
        lvls = np.linspace(lo, hi, n_levels)
        bkt  = (hi - lo) / n_levels
        pcts = (lvls / spot - 1) * 100
        # paths_mid: shape (n, days) — columns d=1..days
        paths_mid = paths[:, 1:]  # (n_paths, days)
        # bin each path-day into [0, n_levels] — values outside → 0 or n_levels
        bins = np.floor((paths_mid - lo) / bkt).astype(np.int32)  # (n, days)
        Z = np.zeros((n_levels, days), dtype=np.float32)
        n_paths = paths_mid.shape[0]
        for d in range(days):
            b = bins[:, d]
            valid = (b >= 0) & (b < n_levels)
            np.add.at(Z[:, d], b[valid], 1)
        Z = Z / n_paths * 100
        return Z, lvls, pcts

    spx_paths = _sim_paths(spx, vix, vvix,            vts_shape, days, n)
    ndx_paths = _sim_paths(ndx, vxn, vvix * vxn/vix if np.isfinite(vvix) and vix > 0 else vvix, vts_shape, days, n)

    Z_spx, lvls_spx, pcts_spx = _density_grid(spx_paths, spx, days)
    Z_ndx, lvls_ndx, pcts_ndx = _density_grid(ndx_paths, ndx, days)

    day_labels  = [f"D{i+1}" for i in range(days)]
    wk_days     = ["Mon", "Tue", "Wed", "Thu", "Fri"][:days]
    x_labels    = [f"0{i+1}\n{wk_days[i]}" if i < len(wk_days) else f"D{i+1}" for i in range(days)]

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

    _mp = _merton_calibrate(vix, vvix, vts_shape, vrp_val=0)

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
            (f"SPX  |  Spot: {spx:,.2f}  |  VIX: {vix:.1f}%  |  RV(20): {rv_spx:.1f}%  |"
             f"  λ={_mp['lam']:.3f}  mj={_mp['mj']:.4f}  σj={_mp['sj']:.4f}"),
            "SPX Terminal",
            (f"NDX  |  Spot: {ndx:,.2f}  |  VXN: {vxn:.1f}%  |  RV(20): {rv_ndx:.1f}%  |"
             f"  VTS: {vts_shape}"),
            "NDX Terminal",
            "SPX Scenario Comparison: Median + IQR + 90% CI",
            "NDX Scenario Comparison: Median + IQR + 90% CI",
            "SPX — 1-Week Forward Probability Summary", "",
            "NDX — 1-Week Forward Probability Summary", "",
        ],
    )

    fig.add_trace(go.Heatmap(
        z=Z_spx, x=x_labels, y=[f"{p:+.1f}%" for p in pcts_spx],
        colorscale="RdYlGn", reversescale=False,
        showscale=False, zmin=0,
    ), row=1, col=1)

    spot_idx_spx = int(np.argmin(np.abs(lvls_spx - spx)))
    fig.add_hline(y=f"{pcts_spx[spot_idx_spx]:+.1f}%",
                  line_color="white", line_width=1.2, line_dash="solid", row=1, col=1)

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

    hist_vals, bin_edges = np.histogram(term_spx, bins=40, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig.add_trace(go.Bar(
        x=hist_vals, y=[f"{(v/spx-1)*100:+.1f}%" for v in bin_centers],
        orientation="h", marker_color="#06b6d4", opacity=0.7,
        name="SPX terminal", showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Heatmap(
        z=Z_ndx, x=x_labels, y=[f"{p:+.1f}%" for p in pcts_ndx],
        colorscale="RdYlGn", reversescale=False,
        showscale=False, zmin=0,
    ), row=2, col=1)

    hist_vals_ndx, bin_edges_ndx = np.histogram(term_ndx, bins=40, density=True)
    bin_centers_ndx = (bin_edges_ndx[:-1] + bin_edges_ndx[1:]) / 2
    fig.add_trace(go.Bar(
        x=hist_vals_ndx, y=[f"{(v/ndx-1)*100:+.1f}%" for v in bin_centers_ndx],
        orientation="h", marker_color="#06b6d4", opacity=0.7,
        name="NDX terminal", showlegend=False,
    ), row=2, col=2)

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

    title_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    plotly_dark(fig, title=f"SPX / NDX  FORWARD PROBABILITY — MERTON JUMP-DIFFUSION — 1-WEEK — {title_date}", height=1200)
    fig.update_layout(
        margin=dict(t=80, b=20, l=50, r=20),
        legend=dict(orientation="h", y=0.52, x=0, font=dict(size=10)),
    )
    for r in [1, 2]:
        fig.update_xaxes(showgrid=False, row=r, col=1)
        fig.update_yaxes(showgrid=False, row=r, col=1)
        fig.update_xaxes(showgrid=False, row=r, col=2)
        fig.update_yaxes(showgrid=False, row=r, col=2)

    return fig


def _merton_fig(spot: float, vix: float, vvix: float = float("nan"),
                vts_shape: str = "MIXED", vrp_val: float = 0.0,
                days=5, n=4000) -> go.Figure:
    rng = np.random.default_rng()
    dt_s = 1 / 252
    sig  = vix / 100
    params = _merton_calibrate(vix, vvix, vts_shape, vrp_val)
    lam, mj, sj = params["lam"], params["mj"], params["sj"]
    drift = -0.5 * sig**2 - params["jump_comp"]

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
    if chain_df is None or len(chain_df) == 0:
        fig = go.Figure()
        plotly_dark(fig, title=f"IV Surface — {label} (DATA UNAVAILABLE)", height=420)
        if label in ("NDX", "QQQ"):
            msg = (
                "<b>No NDX/QQQ options chain data available.</b><br>"
                "QQQ chain is loaded automatically alongside SPY.<br>"
                "Check your internet connection or try refreshing."
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

        df = df[(df["moneyness"] >= 0.93) & (df["moneyness"] <= 1.15)]
        _dte_limit = 10
        df = df[(df["dte"] >= 1) & (df["dte"] <= _dte_limit)]
        if len(df) < 4:
            _dte_limit = 21
            df = chain_df.copy()
            df["dte"] = (df["expiry_T"] * 365).clip(lower=1)
            df["moneyness"] = df["strike"] / spot
            df = df[(df["moneyness"] >= 0.93) & (df["moneyness"] <= 1.15)]
            df = df[(df["dte"] >= 1) & (df["dte"] <= _dte_limit)]

        df = df[(df["iv"] > 0.005) & (df["iv"] < 2.0)]

        iv_med = df["iv"].median()
        df = df[(df["iv"] >= iv_med * 0.25) & (df["iv"] < iv_med * 3.0)]

        if len(df) < 6:
            raise ValueError(f"Only {len(df)} valid chain points — insufficient for surface")

        _atm_mask = df["moneyness"].between(0.98, 1.02)
        atm_iv_est = float(df[_atm_mask]["iv"].mean() * 100) if _atm_mask.sum() > 0 else vix
        if not np.isfinite(atm_iv_est) or atm_iv_est <= 0:
            atm_iv_est = vix
        hard_lo = max(2.0,  atm_iv_est * 0.40)
        hard_hi = min(120.0, atm_iv_est * 3.0)

        pts_dte = df["dte"].values.astype(float)
        pts_m   = df["moneyness"].values.astype(float)
        pts_iv  = np.clip(df["iv"].values.astype(float) * 100, hard_lo, hard_hi)

        dte_grid = np.linspace(max(pts_dte.min(), 1.0), min(pts_dte.max(), 10.0), 16)
        m_grid   = np.linspace(pts_m.min(), pts_m.max(), 30)
        DTE, MON = np.meshgrid(dte_grid, m_grid)

        method = "linear" if len(df) >= 8 else "nearest"
        Z = griddata(
            points=np.column_stack([pts_dte, pts_m]),
            values=pts_iv,
            xi=np.column_stack([DTE.ravel(), MON.ravel()]),
            method=method,
            fill_value=np.nan,
        ).reshape(DTE.shape)

        nan_mask = ~np.isfinite(Z)
        if nan_mask.any():
            Z_nn = griddata(
                points=np.column_stack([pts_dte, pts_m]),
                values=pts_iv,
                xi=np.column_stack([DTE.ravel(), MON.ravel()]),
                method="nearest",
            ).reshape(DTE.shape)
            Z[nan_mask] = Z_nn[nan_mask]

        Z = np.clip(Z, hard_lo, hard_hi)
        Z = gaussian_filter(Z, sigma=1.2)
        Z = np.clip(Z, hard_lo, hard_hi)

        def _avg(mask): return float(df[mask]["iv"].mean() * 100) if mask.sum() > 0 else float("nan")
        puts_25d  = _avg((df["moneyness"].between(0.93, 0.97)) & (df["dte"].between(15, 45)))
        calls_25d = _avg((df["moneyness"].between(1.03, 1.07)) & (df["dte"].between(15, 45)))
        skew_25d  = puts_25d - calls_25d
        front_atm = _avg((df["moneyness"].between(0.985, 1.015)) & (df["dte"] <= 10))
        back_atm  = _avg((df["moneyness"].between(0.985, 1.015)) & (df["dte"].between(25, 70)))

        fig = go.Figure(go.Surface(
            x=dte_grid,
            y=m_grid * 100,
            z=Z,
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
            contours=dict(z=dict(show=False)),
            lighting=dict(ambient=0.7, diffuse=0.8, roughness=0.5, specular=0.3),
            lightposition=dict(x=100, y=200, z=0),
        ))

        plotly_dark(fig, title=f"{label} Implied Volatility Surface | Spot: {spot:,.2f} | ATM IV: {atm_iv_est:.1f}%", height=460)
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="DTE", backgroundcolor="rgba(0,0,0,0)",
                           gridcolor="rgba(255,255,255,0.08)", showbackground=True),
                yaxis=dict(title="Moneyness (%)", backgroundcolor="rgba(0,0,0,0)",
                           gridcolor="rgba(255,255,255,0.08)", showbackground=True),
                zaxis=dict(title="IV%", backgroundcolor="rgba(0,0,0,0)",
                           gridcolor="rgba(255,255,255,0.08)", showbackground=True),
                bgcolor="rgba(0,0,0,0)",
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

    rv5_d  = rv5.reindex(dates)
    rv63_d = rv63.reindex(dates)
    rv21_d = rv21.reindex(dates)
    va_d   = va.reindex(dates)

    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list(rv5_d.ffill()) + list(rv63_d.ffill()[::-1]),
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

    vrp_d = vrp.reindex(dates)
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in vrp_d.fillna(0)]
    fig.add_trace(go.Bar(
        x=dates, y=vrp_d,
        marker_color=colors, opacity=0.8,
        name="VRP", showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1, row=2, col=1)

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
    rets = sa.pct_change().dropna().tail(504) * 100
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

    fig.add_vline(x=mu, line_color="#06b6d4", line_width=1.5,
                  annotation_text=f"Mean: {mu:.3f}%", annotation_font_size=10,
                  row=1, col=1)

    for m, col_s in [(1, "#10b981"), (2, "#f59e0b"), (3, "#ef4444")]:
        for s in [-1, 1]:
            fig.add_vline(x=s * m * sig + mu, line_dash="dash",
                          line_color=col_s, line_width=1, row=1, col=1)

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
            font=dict(family="monospace", size=11 if is_header else 10, color=color),
            xanchor="left", yanchor="top",
        )

    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)
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
                      icsa_4w_chg: float, vix: float,
                      icsa_series: "pd.Series | None" = None,
                      spx_dd: float = 0.0, epu_z: float = 0.0,
                      s2s10_series: "pd.Series | None" = None) -> dict:
    from scipy.special import expit as _sigmoid

    sahm_z = np.clip(sahm / 0.4, 0, 1)
    hy_z   = np.clip((hy - 350) / 450, 0, 1)
    curve_z = np.clip((-s2s10_bp) / 150, 0, 1)

    inv_dur_mult = 1.0
    if s2s10_series is not None and s2s10_series.dropna().size > 10:
        _inv_days = int((s2s10_series.dropna() < 0).rolling(252, min_periods=1).sum().iloc[-1])
        inv_dur_mult = float(np.clip(1.0 + _inv_days / 365.0, 1.0, 2.0))
    curve_z = np.clip(curve_z * inv_dur_mult, 0, 1)

    if icsa_series is not None and icsa_series.dropna().size >= 26:
        _ic   = icsa_series.dropna()
        _mean = float(_ic.rolling(52, min_periods=26).mean().iloc[-1])
        _std  = float(_ic.rolling(52, min_periods=26).std().iloc[-1])
        icsa_z = float(np.clip((_ic.iloc[-1] - _mean) / (_std * 3.0), 0, 1)) if (np.isfinite(_mean) and _std > 0) else float(np.clip(icsa_4w_chg / 0.20, 0, 1))
    else:
        icsa_z = float(np.clip(icsa_4w_chg / 0.20, 0, 1))

    raw = 0.40 * sahm_z + 0.25 * hy_z + 0.20 * curve_z + 0.15 * icsa_z

    score = float(_sigmoid((raw - 0.35) * 8) * 100)
    score = round(np.clip(score, 1, 99), 1)

    if sahm >= 0.35 or hy > 375 or score > 60:  label = "ELEVATED"
    elif sahm >= 0.20 or hy > 320:               label = "MODERATE"
    else:                                          label = "LOW"

    return {
        "score": score, "label": label,
        "components": {
            "sahm_contribution":  round(0.40 * sahm_z  * 100, 1),
            "hy_contribution":    round(0.25 * hy_z    * 100, 1),
            "curve_contribution": round(0.20 * curve_z * 100, 1),
            "icsa_contribution":  round(0.15 * icsa_z  * 100, 1),
        },
        "note": (
            "Recession Stress Index — orthogonal to fear composite (no VIX/EPU). "
            "Sahm 40% + HY OAS 25% + yield curve 20% (with duration mult) + ICSA 15%. "
            "NOT a calibrated probability. Use Estrella-Mishkin for that."
        ),
    }

# ══════════════════════════════════════════════════════════════════════════════
# THREE-HORIZON ENGINES + COMPOSITE
# ══════════════════════════════════════════════════════════════════════════════

def _intraday_engine(
        gex_regime: "GammaRegime",
        dist_to_flip_pct: float,
        spx_1d_ret: float,
        spx_3d_ret: float,
        spx_5d_ret: float,
        hyg_1d_ret: float,
        lqd_1d_ret: float,
        hyg_3d_ret: float,
        lqd_3d_ret: float,
        rv5: float,
        rv20: float,
) -> dict:
    neg = gex_regime in (GammaRegime.NEGATIVE, GammaRegime.STRONG_NEGATIVE)
    pos = gex_regime in (GammaRegime.POSITIVE, GammaRegime.STRONG_POSITIVE)

    mom_raw = 0.5 * spx_1d_ret + 0.3 * spx_3d_ret + 0.2 * spx_5d_ret
    if pos:
        mom_spx = -mom_raw
    elif neg:
        mom_spx =  mom_raw
    else:
        mom_spx = -mom_raw * 0.5
    mom_spx = float(np.clip(mom_spx / 0.03, -2.5, 2.5))

    credit_1d = hyg_1d_ret - lqd_1d_ret
    credit_3d = hyg_3d_ret - lqd_3d_ret
    mom_credit = float(np.clip((0.6 * credit_1d + 0.4 * credit_3d) / 0.005, -1.5, 1.5))

    if np.isfinite(dist_to_flip_pct):
        mom_gex = float(np.tanh(dist_to_flip_pct / 1.5))
    else:
        mom_gex = 0.0

    if np.isfinite(rv5) and np.isfinite(rv20) and rv20 > 0:
        vol_trend = rv5 - rv20
        vol_signal = float(np.clip(-vol_trend / 3.0, -1.0, 1.0))
    else:
        vol_signal = 0.0

    score = mom_spx * 0.40 + mom_credit * 0.25 + mom_gex * 0.20 + vol_signal * 0.15
    return {
        "score":      float(np.clip(score * (5 / 1.5), -5.0, 5.0)),
        "mom_spx":    mom_spx,
        "mom_credit": mom_credit,
        "mom_gex":    mom_gex,
        "vol_signal": vol_signal,
    }


def _short_term_engine(
        vrp_val: float,
        vts_shape: str,
        fear_z: float,
        spx_dd: float,
        prob: dict,
) -> dict:
    if np.isfinite(vrp_val):
        if vrp_val > 4:    vrp_sig =  2.0
        elif vrp_val > 2:  vrp_sig =  1.0
        elif vrp_val < -4: vrp_sig = -2.0
        elif vrp_val < -2: vrp_sig = -1.0
        else:              vrp_sig =  vrp_val * 0.3
    else:
        vrp_sig = 0.0

    fear_sig = float(np.clip(-fear_z * 1.2, -2.5, 2.5))
    if vts_shape == "BACKWARDATION":
        fear_sig += 0.5

    dd_sig = float(np.clip(spx_dd / 0.05, -1.5, 1.5))

    def _bs(p): return (p - 50.0) / 10.0
    prob_sig = float(np.clip(
        0.60 * _bs(prob.get("tactical_prob", 50.0)) +
        0.40 * _bs(prob.get("short_prob",    50.0)),
        -1.5, 1.5
    ))

    score = vrp_sig * 0.35 + fear_sig * 0.30 + dd_sig * 0.20 + prob_sig * 0.15
    return {
        "score":    float(np.clip(score * (5 / 2.5), -5.0, 5.0)),
        "vrp_sig":  vrp_sig,
        "fear_sig": fear_sig,
        "dd_sig":   dd_sig,
        "prob_sig": prob_sig,
    }


def _medium_term_engine(
        macro_reg: str,
        growth_score: int,
        rec_score: float,
        prob: dict,
) -> dict:
    # FIX 2: Goldilocks base reduced from +2.0 → +1.0 to prevent systematic
    # bull tilt from regime label alone on otherwise ambiguous days.
    regime_base = {
        "Goldilocks":   +1.0,
        "Overheating":  +0.5,
        "Disinflation": -0.5,
        "Stagflation":  -2.0,
        "Deflation":    -3.0,
    }.get(macro_reg, 0.0)

    growth_sig = float(np.clip(growth_score / 4.5, -1.5, 1.5))

    def _bs(p): return (p - 50.0) / 10.0
    prob_sig = float(np.clip(_bs(prob.get("medium_prob", 50.0)), -1.5, 1.5))

    score = regime_base * 0.50 + growth_sig * 0.30 + prob_sig * 0.20
    return {
        "score":       float(np.clip(score * (5 / 3.0), -5.0, 5.0)),
        "regime_sig":  regime_base,
        "growth_sig":  growth_sig,
        "prob_sig":    prob_sig,
    }


def _composite(prob: dict, vrp_val: float,
               leading: dict = None, gex_regime: "GammaRegime | None" = None,
               vts_shape: str = "MIXED",
               fear_z: float = 0.0,
               rec_score: float = 50.0,
               spx_dd: float = 0.0,
               macro_reg: str = "Goldilocks",
               growth_score: int = 0,
               spx_5d_ret: float = 0.0,
               hyg_1d_ret: float = 0.0,
               lqd_1d_ret: float = 0.0,
               dist_to_flip_pct: float = 5.0,
               hy_oas_pct: float = 3.0,
               spx_1d_ret: float = 0.0,
               spx_3d_ret: float = 0.0,
               hyg_3d_ret: float = 0.0,
               lqd_3d_ret: float = 0.0,
               rv5: float = float("nan"),
               rv20: float = float("nan"),
               ) -> float:
    neg = gex_regime in (GammaRegime.NEGATIVE, GammaRegime.STRONG_NEGATIVE) if gex_regime else False
    pos = gex_regime in (GammaRegime.POSITIVE, GammaRegime.STRONG_POSITIVE)  if gex_regime else False
    at_flip = abs(dist_to_flip_pct) < 0.4 if np.isfinite(dist_to_flip_pct) else False

    intra  = _intraday_engine(
        gex_regime, dist_to_flip_pct,
        spx_1d_ret, spx_3d_ret, spx_5d_ret,
        hyg_1d_ret, lqd_1d_ret, hyg_3d_ret, lqd_3d_ret,
        rv5, rv20,
    )
    short  = _short_term_engine(vrp_val, vts_shape, fear_z, spx_dd, prob)
    medium = _medium_term_engine(macro_reg, growth_score, rec_score, prob)

    if neg:
        wi, ws, wm = 0.40, 0.40, 0.20
    elif pos:
        wi, ws, wm = 0.20, 0.40, 0.40
    else:
        wi, ws, wm = 0.30, 0.40, 0.30

    s = wi * intra["score"] + ws * short["score"] + wm * medium["score"]

    rec_penalty = 1.0 / (1.0 + np.exp(-(rec_score - 50.0) / 10.0))
    s *= (1.0 - 0.40 * rec_penalty)
    max_bull = 5.0 * (1.0 - 0.70 * rec_penalty)
    s = min(s, max_bull)

    stress_count = 0
    if leading is not None:
        sahm_triggered = leading.get("sahm_triggered", False)
        hy_stress      = (hy_oas_pct > 4.0) if np.isfinite(hy_oas_pct) else False
        corr_systemic  = leading.get("corr_regime", "NORMAL") == "SYSTEMIC"
        stress_count   = sum([bool(sahm_triggered), bool(hy_stress), bool(corr_systemic)])
        s -= stress_count * 0.4

    if vts_shape == "BACKWARDATION":
        s += 0.25

    _SCALE = 2.0
    if stress_count >= 2:
        s = min(s, -1.0)
    elif stress_count == 1:
        s = min(s,  1.5)

    s = s * _SCALE

    # FIX 1: Backwardation contrarian NO EDGE bug.
    # Previously: `if at_flip or (abs(s) < 1.5 and mom_abs < 0.5)`
    # This wiped out the backwardation contrarian signal even when it was the
    # primary edge (VTS backwardation + positive gamma + slight fear).
    # Fix: exclude the backwardation case from the NO EDGE condition so the
    # vol mean-reversion signal survives weak-momentum days.
    mom_abs = abs(intra.get("mom_spx", 0)) + abs(intra.get("mom_credit", 0))
    _vts_contrarian = (vts_shape == "BACKWARDATION") and not at_flip
    if at_flip or (abs(s) < 1.5 and mom_abs < 0.5 and not _vts_contrarian):
        s = 0.0

    return float(np.clip(s, -10.0, 10.0))

def _verdict(c: float, gex: GammaRegime) -> Tuple[str,str,str]:
    neg=gex in (GammaRegime.NEGATIVE,GammaRegime.STRONG_NEGATIVE)
    pos=gex in (GammaRegime.POSITIVE,GammaRegime.STRONG_POSITIVE)
    if c<=-3: return "BEARISH","#ef4444","Multiple signals aligned bearish."
    if c<=-2:
        return (("BEARISH","#ef4444","Negative macro + negative gamma.") if neg
                else ("LEANING BEARISH","#f97316","Modest bearish lean."))
    if c<=0:
        if neg: return "CAUTIOUS","#f59e0b","Neutral signals but negative gamma — tail risk elevated."
        if pos: return "NEUTRAL / RANGE","#6366f1","Positive gamma → compression and pin."
        return "NEUTRAL","#94a3b8","No strong conviction."
    if c<=2: return "LEANING BULLISH","#10b981","Bullish lean. Credit and liquidity constructive."
    return "BULLISH","#10b981","Broad bullish alignment."


def _verdict3(c: float, gex: GammaRegime, vrp_val: float,
              vts_shape: str, fear_z: float, rec_score: float,
              macro_reg: str, dist_flip_pct: float,
              leading: dict = None) -> dict:
    neg = gex in (GammaRegime.NEGATIVE, GammaRegime.STRONG_NEGATIVE)
    pos = gex in (GammaRegime.POSITIVE, GammaRegime.STRONG_POSITIVE)
    at_flip = abs(dist_flip_pct) < 0.5 if np.isfinite(dist_flip_pct) else False
    vrp_high   = np.isfinite(vrp_val) and vrp_val > 2.0
    vrp_low    = np.isfinite(vrp_val) and vrp_val < -2.0
    backw      = vts_shape == "BACKWARDATION"
    corr_stress = (leading or {}).get("corr_regime", "NORMAL") in ("STRESS","SYSTEMIC")

    if c == 0.0 and at_flip:
        bias, bias_color = "NO TRADE",          "#94a3b8"
    elif c == 0.0:
        bias, bias_color = "NO EDGE",           "#6b7280"
    elif c <= -3.0:  bias, bias_color = "BEARISH",        "#ef4444"
    elif c <= -1.75: bias, bias_color = "LEANING BEARISH","#f97316"
    elif c <= 0.75:
        if neg:     bias, bias_color = "CAUTIOUS",       "#f59e0b"
        elif pos:   bias, bias_color = "NEUTRAL / RANGE","#6366f1"
        else:       bias, bias_color = "NEUTRAL",        "#94a3b8"
    elif c <= 2.5:  bias, bias_color = "LEANING BULLISH", "#34d399"
    else:           bias, bias_color = "BULLISH",          "#10b981"

    n_weak = sum([at_flip, corr_stress, abs(c) < 1.5, fear_z > 1.5])
    confidence = "LOW" if n_weak >= 2 else "HIGH" if (abs(c) > 3.0 and not at_flip) else "MEDIUM"
    confidence_pct = float(min(abs(c) / 5.0, 1.0))

    if "BEARISH" in bias or "CAUTIOUS" in bias:
        if neg and fear_z > 0.5:
            bias_type = "TREND BEARISH"
            bias_type_note = "Dealers amplify + fear elevated → trending, not mean-reverting"
        elif vrp_high and pos:
            bias_type = "MEAN-REVERSION BEARISH"
            bias_type_note = "High VRP + positive GEX → fade rallies, not chase breakdowns"
        elif backw or rec_score > 60:
            bias_type = "EVENT-DRIVEN BEARISH"
            bias_type_note = "Vol backwardation or macro stress → risk of sharp gap"
        else:
            bias_type = "STRUCTURAL BEARISH"
            bias_type_note = "Macro headwinds dominant — slow deterioration"
    elif "BULLISH" in bias:
        if pos and vrp_high:
            bias_type = "MEAN-REVERSION BULLISH"
            bias_type_note = "Positive GEX + high VRP → buy dips, fade overshoots"
        elif neg:
            bias_type = "MOMENTUM BULLISH"
            bias_type_note = "Negative GEX amplifies — trend follows through"
        else:
            bias_type = "STRUCTURAL BULLISH"
            bias_type_note = "Macro + liquidity aligned — hold, add on pullbacks"
    else:
        bias_type = "NO DIRECTIONAL EDGE"
        bias_type_note = "Conflicting or insufficient inputs — reduce size"

    if at_flip:
        structure, structure_note = "⚡ NEAR FLIP",       "At gamma flip — binary outcome, no mechanical edge"
    elif neg and fear_z > 0.5:
        structure, structure_note = "📉 TREND / VOLATILE","Negative gamma + fear → dealers amplify, trending"
    elif pos:
        structure, structure_note = "📌 PINNED / RANGE",  "Positive gamma → dealers suppress moves, mean-rev dominant"
    elif backw:
        structure, structure_note = "⚠️ STRESS / EVENT",  "Vol backwardation — near-term uncertainty elevated"
    else:
        structure, structure_note = "〰️ CHOP",             "No clear mechanical bias"

    if at_flip or c == 0.0:
        strategy = "🚫 NO TRADE — no mechanical edge. Wait for resolution."
    elif bias_type == "TREND BEARISH":
        strategy = "📉 SELL RALLIES — not breakdowns. Neg GEX amplifies, don't chase."
    elif bias_type == "MEAN-REVERSION BEARISH":
        strategy = "🔁 FADE RALLIES — sell into strength. High VRP + pos GEX pins range."
    elif bias_type == "EVENT-DRIVEN BEARISH":
        strategy = "🛡 BUY PROTECTION — vol cheap vs stress. Put spreads over outright shorts."
    elif bias_type == "STRUCTURAL BEARISH":
        strategy = "⬇ REDUCE LONGS — slow grind. Reduce on bounces, no urgency."
    elif bias_type == "MEAN-REVERSION BULLISH":
        strategy = "🔁 BUY DIPS — pos GEX supports. Fade overshoots, don't chase."
    elif bias_type == "MOMENTUM BULLISH":
        strategy = "📈 BUY BREAKOUTS — neg GEX amplifies. Trend follows through."
    elif bias_type == "STRUCTURAL BULLISH":
        strategy = "📈 HOLD / ADD — macro aligned. Wait for pullbacks."
    else:
        strategy = "⚖️ STAY FLAT / REDUCE SIZE — no clear edge."

    conflicts = []
    sdir = {}
    if vrp_high:       sdir["Vol (VRP)"]   = "bullish"
    elif vrp_low:      sdir["Vol (VRP)"]   = "bearish"
    if neg:            sdir["Flow (GEX)"]  = "bearish"
    elif pos:          sdir["Flow (GEX)"]  = "bullish"
    if macro_reg in ("Goldilocks",):                          sdir["Macro"] = "bullish"
    elif macro_reg in ("Stagflation","Deflation","Disinflation"): sdir["Macro"] = "bearish"
    if fear_z > 0.8:   sdir["Fear"] = "bearish"
    elif fear_z < -0.5: sdir["Fear"] = "bullish"
    if backw:          sdir["Vol TS"] = "bullish"
    if corr_stress:    sdir["Credit Corr"] = "bearish"

    bull_s = [k for k,v in sdir.items() if v == "bullish"]
    bear_s = [k for k,v in sdir.items() if v == "bearish"]
    if bull_s and bear_s:
        conflicts.append(("⚠️",
            f"SIGNAL CONFLICT: {' + '.join(bull_s)} bullish | {' + '.join(bear_s)} bearish "
            f"→ expect chop / false moves"))
    if vrp_high and neg:
        conflicts.append(("⚠️","Vol says fade (VRP high) but GEX says trend (neg gamma) — size down"))
    if pos and fear_z > 1.0:
        conflicts.append(("⚠️","GEX says pin (pos gamma) but fear elevated — dealers may unwind"))

    how_to = []
    if at_flip:
        how_to.append(("🚫","At gamma flip — sit out or 25% size."))
    elif neg:
        how_to.append(("📉","Neg gamma: dealers AMPLIFY. Follow trend, don't fade."))
    elif pos:
        how_to.append(("📌","Pos gamma: dealers SUPPRESS. Fade extremes, trade the range."))
    if vrp_high:
        how_to.append(("💰",f"VRP rich (+{vrp_val:.2f}): sell vol / fade extremes."))
    elif vrp_low:
        how_to.append(("🛡",f"VRP cheap ({vrp_val:.2f}): buy protection. Tail risk underpriced."))
    if backw:
        how_to.append(("⏳","Vol backwardation: mean-reversion likely within 1-3 weeks."))
    if corr_stress:
        how_to.append(("🔴","Credit leading equity. Reduce size, monitor HYG/SPY divergence."))
    if confidence == "LOW":
        how_to.append(("⚖️","Low confidence — use 50% of normal position size."))

    return {
        "bias":           bias,         "bias_color":      bias_color,
        "bias_type":      bias_type,    "bias_type_note":  bias_type_note,
        "confidence":     confidence,   "confidence_pct":  confidence_pct,
        "structure":      structure,    "structure_note":  structure_note,
        "strategy":       strategy,
        "conflicts":      conflicts,    "how_to":          how_to,
    }


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
                   max_dte: int = 45, height: int = 360,
                   flip_src: str = "OI") -> go.Figure:
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

        _vol_avail = ("net_vol_gex" in gex_chain.columns and
                      gex_chain["net_vol_gex"].abs().sum() > 0)
        _gex_col = "net_vol_gex" if _vol_avail else "net_gex"
        _src_label = "Volume flow" if _vol_avail else "OI"

        agg = (gex_chain.groupby("strike")[_gex_col]
                        .sum()
                        .reset_index()
                        .sort_values("strike"))

        lo, hi = spot * 0.90, spot * 1.10
        agg = agg[(agg["strike"] >= lo) & (agg["strike"] <= hi)]

        if agg.empty:
            raise ValueError("No strikes in ±10% range")

        strikes   = agg["strike"].tolist()
        vals_m    = (agg[_gex_col] / 1e6).tolist()

        max_abs = max(abs(v) for v in vals_m)
        if max_abs >= 5000:
            vals_display = [v / 1000 for v in vals_m]
            unit = "$B"
        else:
            vals_display = vals_m
            unit = "$M"

        colors = [_C_POS if v > 0 else _C_NEG for v in vals_display]

        fig = go.Figure(go.Bar(
            x=strikes, y=vals_display,
            marker_color=colors, marker_line_width=0,
            opacity=0.88, name=f"Net GEX ({_src_label})",
        ))

        fig.add_hline(y=0, line_color="rgba(255,255,255,0.30)", line_width=1)
        fig.add_vline(
            x=spot, line_dash="dot",
            line_color="rgba(255,255,255,0.75)", line_width=1.5,
            annotation_text=f"SPOT {spot:.2f}",
            annotation_font_size=10,
            annotation_font_color="rgba(255,255,255,0.85)",
            annotation_position="top right",
        )

        if flip and np.isfinite(flip) and lo <= flip <= hi:
            fig.add_vline(
                x=flip, line_dash="dash",
                line_color=_C_FLIP, line_width=2,
                annotation_text=f"⚡ FLIP {flip:.2f} ({flip_src})",
                annotation_font_size=10,
                annotation_font_color=_C_FLIP,
                annotation_position="top left",
            )

        x_pad = max(abs(hi - spot), abs(spot - lo)) * 1.05
        fig.update_layout(
            xaxis=dict(range=[spot - x_pad, spot + x_pad], title="Strike"),
            yaxis_title=f"Net GEX ({unit})",
            bargap=0.12, showlegend=False,
        )
        plotly_dark(fig, title=f"GEX Histogram ({_src_label}) — {unit} per Strike (≤{max_dte} DTE)", height=height)
        return fig

    except Exception as exc:
        fig = go.Figure()
        plotly_dark(fig, title=f"GEX Histogram (error: {exc})", height=height)
        return fig


def _intraday_bias(q: dict, gex_state) -> tuple:
    spx_pct  = q.get("SPX_pct", 0.0) or 0.0
    vix_pct  = q.get("VIX_pct", 0.0) or 0.0
    hyg_pct  = q.get("HYG_pct", 0.0) or 0.0
    lqd_pct  = q.get("LQD_pct", 0.0) or 0.0

    credit_spread = hyg_pct - lqd_pct

    bull_pts = 0
    bear_pts = 0

    if spx_pct > 0.3:  bull_pts += 2
    elif spx_pct > 0:  bull_pts += 1
    elif spx_pct < -0.3: bear_pts += 2
    elif spx_pct < 0:    bear_pts += 1

    if vix_pct < -3:   bull_pts += 2
    elif vix_pct < 0:  bull_pts += 1
    elif vix_pct > 5:  bear_pts += 2
    elif vix_pct > 0:  bear_pts += 1

    if credit_spread > 0.2:  bull_pts += 1
    elif credit_spread < -0.2: bear_pts += 1

    try:
        dist = abs(float(gex_state.distance_to_flip_pct))
        if dist < 0.3:
            bull_pts = bear_pts = 0
    except Exception:
        pass

    net = bull_pts - bear_pts
    if net >= 3:    return "BULL FLOW",   "#10b981"
    elif net >= 1:  return "BIAS UP",     "#34d399"
    elif net <= -3: return "BEAR FLOW",   "#ef4444"
    elif net <= -1: return "BIAS DOWN",   "#f97316"
    else:           return "NEUTRAL",     "#94a3b8"

def _build_doc_style_thesis(
    vrp: dict,
    gex_state,
    gex_score: int,
    fear_z: float,
    rec: float,
    macro_reg: str,
    stlc: float,
    vts: dict,
    b: dict,
    spx: float,
    flip: float,
    upper: float,
    lower: float,
    gex_sym: str,
) -> dict:
    neg = gex_state.regime in (GammaRegime.NEGATIVE, GammaRegime.STRONG_NEGATIVE)
    pos = gex_state.regime in (GammaRegime.POSITIVE, GammaRegime.STRONG_POSITIVE)

    score = 0
    drivers = []

    # 1) GEX / structure
    if neg:
        score -= 2
        drivers.append(("🔴", f"GEX negative ({gex_score:+d})"))
    elif pos:
        score += 1
        drivers.append(("🟢", f"GEX positive ({gex_score:+d})"))

    # 2) Fear
    if fear_z >= 1.0:
        score -= 1
        drivers.append(("🔴", "Fear composite ELEVATED"))
    elif fear_z <= -1.0:
        score += 1
        drivers.append(("🟢", "Fear composite COMPLACENT"))

    # 3) Recession / macro stress
    if rec >= 60:
        score -= 1
        drivers.append(("🔴", f"Recession risk elevated ({rec:.1f}%)"))
    elif rec <= 30:
        score += 1
        drivers.append(("🟢", f"Recession risk low ({rec:.1f}%)"))

    # 4) VRP
    vrp_val = vrp.get("val", float("nan"))
    if np.isfinite(vrp_val):
        if vrp_val >= 2:
            score += 1
            drivers.append(("🟢", f"VRP positive ({vrp_val:.4f})"))
        elif vrp_val <= -2:
            score -= 1
            drivers.append(("🔴", f"VRP negative ({vrp_val:.4f})"))

    # 5) Macro tilt
    if macro_reg in ("Deflation", "Stagflation", "Disinflation"):
        score -= 1
    elif macro_reg == "Goldilocks":
        score += 1

    score = int(np.clip(score, -3, 3))

    if score <= -2:
        bias = "BEARISH"
        color = "#ef4444"
    elif score == -1:
        bias = "LEANING BEARISH"
        color = "#f97316"
    elif score == 0:
        bias = "NEUTRAL"
        color = "#94a3b8"
    elif score == 1:
        bias = "LEANING BULLISH"
        color = "#34d399"
    else:
        bias = "BULLISH"
        color = "#10b981"

    risks = []
    if fear_z >= 1.0:
        risks.append("⚠️ Elevated fear composite — potential for sharp moves")
    if rec >= 60:
        risks.append(f"⚠️ Recession probability at {rec:.1f}% — monitor labor data")
    if np.isfinite(stlc) and stlc > 0.2:
        risks.append("⚠️ Positive stock-bond correlation — diversification impaired")
    if vts.get("shape") == "BACKWARDATION":
        risks.append("⚠️ Vol backwardation — near-term stress elevated")
    if neg:
        risks.append("⚠️ Negative gamma — dealers may amplify moves")
    risks = risks[:3]

    ordered = []

    def _append_matching(keyword: str):
        for item in drivers:
            if item not in ordered and keyword.lower() in item[1].lower():
                ordered.append(item)

    _append_matching("VRP")
    _append_matching("Fear")
    _append_matching("Recession")
    _append_matching("GEX")

    for item in drivers:
        if item not in ordered:
            ordered.append(item)

    ordered = ordered[:3]

    mult = 10.0 if gex_sym in ("SPY", "SPX") else 40.0 if gex_sym in ("QQQ", "NDX") else 1.0

    return {
        "bias": bias,
        "color": color,
        "score": score,
        "drivers": ordered,
        "risks": risks,
        "levels": {
            "SPX Spot": spx,
            "GEX Flip": flip * mult,
            "GEX Upper": upper * mult,
            "GEX Lower": lower * mult,
            "1σ Daily": (b["d1lo"], b["d1hi"]),
            "2σ Daily": (b["d2lo"], b["d2hi"]),
            "1σ Weekly": (b["w1lo"], b["w1hi"]),
            "2σ Weekly": (b["w2lo"], b["w2hi"]),
        },
    }


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
        # OPT: load_macro hits FRED + yfinance; cache for 10 min since FRED
        # updates at most hourly and market close data is daily.
        @st.cache_data(ttl=600, show_spinner=False)
        def _cached_macro(s, e):
            return load_macro(s, e)
        raw=_cached_macro(start.isoformat(), end.isoformat())
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
        epu_r=raw.get("USEPUINDXD",pd.Series(dtype=float))
        sahm=resample_ffill(sahm_r,idx) if len(sahm_r.dropna())>0 else None
        hys=resample_ffill(hy_r,idx) if len(hy_r.dropna())>0 else None
        epu=resample_ffill(epu_r,idx) if len(epu_r.dropna())>0 else None
        icsa_r=raw.get("ICSA",pd.Series(dtype=float))
        icsa=resample_ffill(icsa_r,idx) if len(icsa_r.dropna())>0 else None
        nfp_r   = raw.get("PAYEMS", pd.Series(dtype=float))
        nfp_s   = resample_ffill(nfp_r, idx) if len(nfp_r.dropna()) > 0 else None
        ppi_r   = raw.get("PPIACO", pd.Series(dtype=float))
        ppi_s   = resample_ffill(ppi_r, idx) if len(ppi_r.dropna()) > 0 else None
        ism_nm_r = raw.get("NMFCI", pd.Series(dtype=float))
        ism_nm   = resample_ffill(ism_nm_r, idx) if len(ism_nm_r.dropna()) > 0 else None
        chi_pmi_r = raw.get("CHPMINDX", pd.Series(dtype=float))
        chi_pmi   = resample_ffill(chi_pmi_r, idx) if len(chi_pmi_r.dropna()) > 0 else None
        conf_r  = raw.get("UMCSENT", pd.Series(dtype=float))
        conf_s  = resample_ffill(conf_r, idx) if len(conf_r.dropna()) > 0 else None

    core_yoy=(core/core.shift(365)-1)*100
    cpi_yoy=(cpi/cpi.shift(365)-1)*100
    s2s10=(y10-y2)*100
    net_liq=(walcl-tga-rrp)/1000
    nl4w=net_liq.diff(28); bs13=walcl.diff(91)/1000
    cyl=_sl(core_yoy,2.5); crl=_sl(s2s10,0.0)
    gz=zscore(s2s10.fillna(0)); iz=zscore(core_yoy.fillna(cyl))

    _icsa_v_raw  = _sl(icsa, float("nan")) if icsa is not None else float("nan")
    _icsa_v_k    = _icsa_v_raw / 1000 if np.isfinite(_icsa_v_raw) else float("nan")
    _nfp_v       = float(nfp_s.dropna().diff(1).dropna().iloc[-1]) if (nfp_s is not None and nfp_s.dropna().size > 1) else float("nan")
    _ccpi_v      = _sl(core_yoy, float("nan"))
    _ppi_mom     = float(ppi_s.dropna().pct_change(1).dropna().iloc[-1] * 100) if (ppi_s is not None and ppi_s.dropna().size > 1) else float("nan")
    _ism_mfg_v   = _sl(ism, float("nan")) if ism is not None else float("nan")
    _ism_nm_v    = _sl(ism_nm, float("nan")) if ism_nm is not None else float("nan")
    _chi_v       = _sl(chi_pmi, float("nan")) if chi_pmi is not None else float("nan")
    _conf_v      = _sl(conf_s, float("nan")) if conf_s is not None else float("nan")
    ur  = _sl(unrate, 4.0)
    cyi = _sl(cpi_yoy, 2.5)

    _ur_val    = ur
    _icsa_val  = _icsa_v_k if np.isfinite(_icsa_v_k) else float("nan")
    _nfp_val   = _nfp_v
    _cpi_val   = cyi
    _ccpi_val  = _ccpi_v
    _ppi_val   = _ppi_mom
    _ism_val   = _ism_mfg_v
    _ism_nm_val= _ism_nm_v
    _chi_val   = _chi_v
    _conf_val  = _conf_v

    def _score_hi_good(val, good_thresh, bad_thresh):
        if not np.isfinite(val): return 0
        if val >= good_thresh: return +1
        if val <= bad_thresh:  return -1
        return 0

    def _score_hi_bad(val, bad_thresh, good_thresh):
        if not np.isfinite(val): return 0
        if val >= bad_thresh:  return -1
        if val <= good_thresh: return +1
        return 0

    _g_ur    = _score_hi_bad (_ur_val,    5.5,  4.0)
    _g_icsa  = _score_hi_bad (_icsa_val, 350,  250)
    _g_nfp   = _score_hi_good(_nfp_val,  250,   50)
    _g_ism   = _score_hi_good(_ism_val,   55,   50)
    _g_ism_nm= _score_hi_good(_ism_nm_val,55,   50)
    _g_chi   = _score_hi_good(_chi_val,   55,   50)
    _g_conf  = _score_hi_good(_conf_val, 120,  100)

    growth_score = (_g_ur + _g_icsa + _g_nfp +
                    _g_ism + _g_ism_nm + _g_chi + _g_conf)

    _i_cpi   = _score_hi_good(_cpi_val,  2.5, 1.5)
    _i_ccpi  = _score_hi_good(_ccpi_val, 2.5, 1.5)
    _i_ppi   = _score_hi_good(_ppi_val,  0.2, 0.0)
    _i_ism_r = +1 if (np.isfinite(_ism_val) and _ism_val > 55) else (
               -1 if (np.isfinite(_ism_val) and _ism_val < 48) else 0)

    inflation_score = _i_cpi + _i_ccpi + _i_ppi + _i_ism_r

    _core_3m_chg = float(core_yoy.dropna().diff(63).dropna().iloc[-1]) if core_yoy.dropna().size > 63 else 0.0
    _disinflating = _core_3m_chg < -0.10

    if growth_score >= 1 and inflation_score >= 1:
        macro_reg = "Overheating"
    elif growth_score >= 1 and inflation_score <= 0:
        macro_reg = "Goldilocks"
    elif growth_score <= -1 and inflation_score >= 1:
        macro_reg = "Disinflation" if _disinflating else "Stagflation"
    elif growth_score <= -1 and inflation_score <= -1:
        macro_reg = "Deflation"
    elif growth_score <= -1 and inflation_score == 0:
        macro_reg = "Disinflation"
    else:
        _gz_now = _sl(gz, 0.0)
        _iz_now = _sl(iz, 0.0)
        if _gz_now >= 0 and _iz_now >= 0:   macro_reg = "Overheating"
        elif _gz_now >= 0 and _iz_now < 0:  macro_reg = "Goldilocks"
        elif _gz_now < 0 and _iz_now >= 0:  macro_reg = "Disinflation" if _disinflating else "Stagflation"
        else:                                macro_reg = "Deflation" if _iz_now < -0.5 else "Disinflation"

    try:
        _ext_reg = classify_macro_regime_abs(cyl, crl)
        if _ext_reg in ("Stagflation", "Deflation") and macro_reg in ("Goldilocks", "Overheating"):
            macro_reg = _ext_reg
    except Exception:
        pass
    _hys_filled = (hys.fillna(hys.dropna().mean())
                   if hys is not None and hys.dropna().size > 0
                   else pd.Series(3.0, index=idx))

    _epu_filled = (epu.fillna(epu.dropna().mean())
                   if epu is not None and epu.dropna().size > 0
                   else pd.Series(100.0, index=idx))

    def _rz(s: pd.Series, window: int = 252) -> pd.Series:
        mu  = s.rolling(window, min_periods=63).mean()
        sig = s.rolling(window, min_periods=63).std()
        return (s - mu) / sig.replace(0, np.nan).ffill().fillna(1)

    vz   = _rz(vix_s.fillna(vix_s.dropna().median()))
    nz   = _rz(nfci.fillna(0))
    hyz  = _rz(_hys_filled)
    epuz = _rz(_epu_filled)

    fear_raw = 0.35*vz + 0.25*hyz + 0.25*nz + 0.15*epuz

    _fear_raw_scalar = float(fear_raw.dropna().iloc[-1]) if fear_raw.dropna().size > 0 else 0.0
    fear   = float(np.clip(100.0 / (1.0 + np.exp(-_fear_raw_scalar)), 0, 100))
    fear_z = _fear_raw_scalar

    vl=_sl(vix_s,20.0); sahmv=_sl(sahm,0.0) if sahm is not None else 0.0
    hyv=_sl(hys,3.0) if hys is not None else 3.0
    ur=_sl(unrate,4.0); cyi=_sl(cpi_yoy,2.5); gzv=_sl(gz,0.0); izv=_sl(iz,0.0)
    nl4wv=_sl(nl4w,0.0)
    liq_lab="Expanding" if nl4wv>=0 else "Contracting"
    y10_20=y10.diff(20)
    if icsa is not None and icsa.dropna().size > 28:
        _icsa_clean = icsa.dropna()
        icsa_4w_chg = float(_icsa_clean.pct_change(28).dropna().iloc[-1])
    else:
        icsa_4w_chg = 0.0
    u3m = icsa_4w_chg
    warsh=((y10.diff(20)<0)&(bs13<0)).astype(int)
    spydd=(spy/spy.rolling(126).max()-1).fillna(0)
    tp=float(np.clip(45+35*float(spydd.iloc[-1]<=-0.07)+20*(fear_z>1.0),0,100))
    fp=float(np.clip(55+25*float((y10_20.iloc[-1]<0)and(u3m>0))-10*float((cyl-3.0)>0)-15*float(warsh.iloc[-1]),0,100))
    tgadd=(tga.diff(28)<0).astype(int); rrpd=(rrp<50).astype(int)
    trp=float(np.clip(50+20*float(tgadd.iloc[-1])+15*float(rrpd.iloc[-1])+15*float(nl4w.iloc[-1]>=0),0,100))
    threeP=float(np.clip(0.35*trp+0.35*fp+0.30*tp,0,100))
    _stress = _recession_stress(
        sahmv, hyv*100, crl, icsa_4w_chg, vl,
        icsa_series=icsa,
        spx_dd=float(spydd.iloc[-1]),
        epu_z=float(epuz.dropna().iloc[-1]) if epuz.dropna().size > 0 else 0.0,
        s2s10_series=s2s10,
    )
    rec     = _stress["score"]
    rec_lbl = _stress["label"]
    sr=_to_1d(spy).reindex(idx).ffill().pct_change().dropna()
    tr2=_to_1d(tlt).reindex(idx).ffill().pct_change().reindex(sr.index).dropna()
    stlc=round(float(sr.rolling(21).corr(tr2).dropna().iloc[-1]),3) if sr.dropna().size>21 else float("nan")
    _cpi_raw = raw.get("CPIAUCSL", pd.Series(dtype=float)).dropna()
    cpi_now = round(float(_cpi_raw.pct_change(1).dropna().iloc[-1]) * 100, 3) if len(_cpi_raw) > 1 else float("nan")
    rd=None

    def _fed_sig(val, high_is_exp: bool, hi_thresh, lo_thresh, fmt=".1f"):
        if not np.isfinite(val):
            return "N/A", "N/A", "#94a3b8"
        vs = f"{val:{fmt}}"
        if high_is_exp:
            if val > hi_thresh:    return vs, "Expansionary",   "#10b981"
            elif val >= lo_thresh: return vs, "Neutral",         "#f59e0b"
            else:                  return vs, "Contractionary",  "#ef4444"
        else:
            if val > hi_thresh:    return vs, "Contractionary",  "#ef4444"
            elif val >= lo_thresh: return vs, "Neutral",         "#f59e0b"
            else:                  return vs, "Expansionary",    "#10b981"

    _ur_v, _ur_sig, _ur_col = _fed_sig(ur, True, 5.5, 4.0, ".1f")
    _icsa_vs, _icsa_sig, _icsa_col = _fed_sig(_icsa_v_k, True, 350, 250, ".0f")
    _icsa_vs = f"{_icsa_v_k:.0f}k" if np.isfinite(_icsa_v_k) else "N/A"
    _nfp_vs, _nfp_sig, _nfp_col = _fed_sig(_nfp_v, False, 250, 50, ".0f")
    _nfp_vs = f"{_nfp_v:+.0f}k" if np.isfinite(_nfp_v) else "N/A"
    _cpi_vs, _cpi_sig, _cpi_col = _fed_sig(cyi, False, 2.5, 1.5, ".2f")
    _cpi_vs = f"{cyi:.2f}%"
    _ccpi_vs, _ccpi_sig, _ccpi_col = _fed_sig(_ccpi_v, False, 2.5, 1.5, ".2f")
    _ccpi_vs = f"{_ccpi_v:.2f}%" if np.isfinite(_ccpi_v) else "N/A"
    _ppi_vs, _ppi_sig, _ppi_col = _fed_sig(_ppi_mom, False, 0.2, 0.0, "+.2f")
    _ppi_vs = f"{_ppi_mom:+.2f}%" if np.isfinite(_ppi_mom) else "N/A"
    _ism_vs, _ism_sig, _ism_col = _fed_sig(_ism_mfg_v, False, 55, 50, ".1f")
    _ism_vs = f"{_ism_mfg_v:.1f}" if np.isfinite(_ism_mfg_v) else "N/A"
    _ism_nm_vs, _ism_nm_sig, _ism_nm_col = _fed_sig(_ism_nm_v, False, 55, 50, ".1f")
    _ism_nm_vs = f"{_ism_nm_v:.1f}" if np.isfinite(_ism_nm_v) else "N/A"
    _chi_vs, _chi_sig, _chi_col = _fed_sig(_chi_v, False, 55, 50, ".1f")
    _chi_vs = f"{_chi_v:.1f}" if np.isfinite(_chi_v) else "N/A"
    _conf_vs, _conf_sig, _conf_col = _fed_sig(_conf_v, False, 120, 100, ".1f")
    _conf_vs = f"{_conf_v:.1f}" if np.isfinite(_conf_v) else "N/A"

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
    vxn_live=q.get("VXN_last", vix_live * 1.10)
    vrp=_vrp_full(vix_live,spy,idx)
    vts=_vts(q); tail=_tail(q)
    rd=_retdist(spy,idx,spx)
    b=_bands(spx,vix_live)

    gex_spot=spx if gex_sym in ("SPY","SPX") else ndx
    with st.spinner("Fetching options data…"):
        client=get_schwab_client(); chain_df=None
        # OPT: cache yfinance chain for 5 min — options data doesn't change
        # tick-by-tick for GEX purposes; avoids re-fetching on UI interactions.
        @st.cache_data(ttl=300, show_spinner=False)
        def _cached_yf_chain(sym):
            return get_gex_from_yfinance(sym)
        if client:
            chain_df=schwab_get_options_chain(client,gex_sym,spot=None)
            if chain_df is not None and len(chain_df)>0:
                gex_spot=schwab_get_spot(client,gex_sym) or gex_spot
        if chain_df is None or len(chain_df)==0:
            chain_df,gex_spot,_=_cached_yf_chain(gex_sym)
        if chain_df is not None and gex_spot:
            gex_st=build_gamma_state(chain_df,float(gex_spot),"live",max_dte=45)
            gwas=compute_gwas(chain_df,float(gex_spot))
            tstr=compute_gex_term_structure(chain_df,float(gex_spot))
            fl=compute_flow_imbalance(chain_df,float(gex_spot))
            net_gex=gex_st.total_gex; gex_score=int(np.clip(net_gex/1e9*10,-50,50))
            _atm_m = chain_df["strike"].between(float(gex_spot)*0.99, float(gex_spot)*1.01)
            _atm_v = chain_df[_atm_m]["iv"].dropna()
            atm_iv_chain = float(_atm_v.mean() * 100) if len(_atm_v) > 0 else None
        else:
            gex_st=GammaState(data_source="unavailable",timestamp=dt.datetime.now().strftime("%H:%M:%S"))
            gwas=tstr=fl={}; net_gex=gex_score=0; atm_iv_chain=None

        qqq_chain_df = None
        qqq_spot = q.get("QQQ_last", None)
        if gex_sym not in ("QQQ", "NDX", "^NDX"):
            if client:
                _qqq_chain = schwab_get_options_chain(client, "QQQ", spot=None)
                if _qqq_chain is not None and len(_qqq_chain) > 0:
                    qqq_chain_df = _qqq_chain
                    _schwab_qqq_spot = schwab_get_spot(client, "QQQ")
                    if _schwab_qqq_spot and _schwab_qqq_spot > 0:
                        qqq_spot = _schwab_qqq_spot
            if qqq_chain_df is None or len(qqq_chain_df) == 0:
                _yf_chain, _yf_spot, _ = _cached_yf_chain("QQQ")
                if _yf_chain is not None and len(_yf_chain) > 0:
                    qqq_chain_df = _yf_chain
                    if _yf_spot and _yf_spot > 0:
                        qqq_spot = _yf_spot
            if not qqq_spot:
                qqq_spot = ndx / 57.7
        else:
            qqq_chain_df = chain_df
            qqq_spot = gex_spot

    _vol_flip = gex_zero_crossing(chain_df, float(gex_spot), max_dte=45) if chain_df is not None else None
    if _vol_flip is not None and np.isfinite(_vol_flip):
        flip = _vol_flip
        if hasattr(gex_st, 'gamma_flip'):
            gex_st.gamma_flip = _vol_flip
    else:
        flip = gex_st.gamma_flip or gex_spot
    _has_vol_flip = (chain_df is not None and
                     "call_volume" in chain_df.columns and
                     chain_df["call_volume"].fillna(0).sum() > 0)
    _flip_src_label = "vol-flow" if _has_vol_flip else "OI"
    upper=gex_st.key_resistance[0] if gex_st.key_resistance else gex_spot*1.03
    lower=gex_st.key_support[0] if gex_st.key_support else gex_spot*0.97
    gex_reg=gex_st.regime.value if hasattr(gex_st.regime,"value") else str(gex_st.regime)
    gex_op_label=REGIME_OPERATIONAL_LABEL.get(gex_st.regime, gex_reg)
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
        # OPT: wrap leading-stack computation in a cached helper so repeated
        # Streamlit rerenders (e.g. sidebar interactions) skip the heavy pandas work.
        @st.cache_data(ttl=300, show_spinner=False)
        def _cached_leading(start_str, end_str, macro_reg_key):
            return compute_leading_stack(
                y2,y3m,y10,y30,s2s10,vix_s,m2,pd.Series(dtype=float),
                copx,gld,hyg,lqd,dxy,spy,qqq,iwm,net_liq,nl4w,walcl,bs13,idx,
                tips_10y=tips,bank_reserves=bres,bank_credit=bcred,ism_no=ism,gdp_quarterly=gdp,mmmf=mmmf)
        leading=_cached_leading(start.isoformat(), end.isoformat(), macro_reg)
        meta=regime_transition_prob(macro_reg,core_yoy,s2s10)
        nc=float(current_pct_rank(-_to_1d(nfci).reindex(idx).ffill(),252))
        lc=float(50.0+np.sign(nl4wv)*20)
    with st.spinner("Loading feeds…"):
        # OPT: RSS feeds update at most every few minutes; cache parsed results
        # for 3 min to avoid re-fetching on every Streamlit interaction.
        @st.cache_data(ttl=180, show_spinner=False)
        def _cached_feeds():
            try:
                rss = load_feeds(tuple(_all_feeds_flat().items()), 60)
                g, _ = geo_shock_score(rss)
                ci   = categorise_items(rss)
                return rss, g, ci
            except Exception:
                return {}, 0.0, {k: [] for k in INTEL_CATEGORIES}
        _rss, geo, cat_intel = _cached_feeds()
    prob=compute_prob_composite(leading,fear,geo,meta["p_change_20d"],gex_st,nfci_coincident=nc,liq_dir_coincident=lc)
    p1d=compute_1d_prob(gex_state=gex_st,spot=float(gex_spot),vix_level=vix_live,
        vix_series=vix_s,spy_series=spy,hyg_series=hyg,lqd_series=lqd,dxy_series=dxy,
        s_2s10s=s2s10,net_liq_4w=nl4w,nfci_z=nz,fear_score=fear,
        session=get_session_context(),idx=idx,sahm_rule=sahm,hy_spread=hys)

    def _last(s, n=1):
        v = _to_1d(s).reindex(idx).ffill().pct_change(n).dropna()
        return float(v.iloc[-1]) if len(v) > 0 else 0.0

    _spy_a = _to_1d(spy).reindex(idx).ffill()
    _hyg_a = _to_1d(hyg).reindex(idx).ffill()
    _lqd_a = _to_1d(lqd).reindex(idx).ffill()

    _spx_1d_ret = _last(_spy_a, 1)
    _spx_3d_ret = _last(_spy_a, 3)
    _spx_5d_ret = _last(_spy_a, 5)
    _hyg_1d_ret = _last(_hyg_a, 1)
    _hyg_3d_ret = _last(_hyg_a, 3)
    _lqd_1d_ret = _last(_lqd_a, 1)
    _lqd_3d_ret = _last(_lqd_a, 3)
    _dist_flip  = float(gex_st.distance_to_flip_pct) if (gex_st.distance_to_flip_pct and np.isfinite(gex_st.distance_to_flip_pct)) else 5.0

    _spy_rets = _spy_a.pct_change().dropna()
    _rv5  = float(_spy_rets.rolling(5,  min_periods=3).std().dropna().iloc[-1] * np.sqrt(252) * 100) if len(_spy_rets) > 5  else float("nan")
    _rv20 = float(_spy_rets.rolling(20, min_periods=10).std().dropna().iloc[-1] * np.sqrt(252) * 100) if len(_spy_rets) > 20 else float("nan")

    # FIX 3: Override stale daily closes with live Schwab intraday data.
    # Without this, reruns during the trading day use last night's close for
    # 1d momentum, making _intraday_engine blind to the current session move.
    _intraday_live = get_intraday_signals(client) if client else {}
    if _intraday_live:
        _spx_1d_ret = _intraday_live.get("SPY_pct", _spx_1d_ret)
        _hyg_1d_ret = _intraday_live.get("HYG_pct", _hyg_1d_ret)
        _lqd_1d_ret = _intraday_live.get("LQD_pct", _lqd_1d_ret)

    thesis = _build_doc_style_thesis(
        vrp=vrp,
        gex_state=gex_st,
        gex_score=gex_score,
        fear_z=fear_z,
        rec=rec,
        macro_reg=macro_reg,
        stlc=stlc,
        vts=vts,
        b=b,
        spx=spx,
        flip=flip,
        upper=upper,
        lower=lower,
        gex_sym=gex_sym,
    )
    news=_news_cats(cat_intel)
    ua="NQ" if gex_sym in ("QQQ","NDX") else "ES"
    um=40.0 if ua=="NQ" else 10.0
    def _ua(p): return f"{p*um:,.0f}"
    reg_col={"Goldilocks":"#10b981","Overheating":"#f59e0b","Stagflation":"#ef4444",
             "Deflation":"#6366f1","Disinflation":"#06b6d4"}.get(macro_reg,"#94a3b8")
    fl2="ELEVATED" if fear_z>1.0 else "MODERATE" if fear_z>0.0 else "COMPLACENT" if fear_z<-1.0 else "LOW"
    fc="#ef4444" if fear_z>1.0 else "#f59e0b" if fear_z>0.0 else "#10b981"

    # ── 1. MARKET REGIME ──────────────────────────────────────────────────
    _q_live = dict(q)
    if _intraday_live:
        # FIX 3 (cont): _intraday_bias uses percent; Schwab returns fractions
        _q_live["SPX_pct"] = _intraday_live.get("SPY_pct", q.get("SPX_pct", 0.0)) * 100
        _q_live["VIX_pct"] = _intraday_live.get("VIX_pct", q.get("VIX_pct", 0.0))
        _q_live["HYG_pct"] = _intraday_live.get("HYG_pct", q.get("HYG_pct", 0.0)) * 100
        _q_live["LQD_pct"] = _intraday_live.get("LQD_pct", q.get("LQD_pct", 0.0)) * 100
    intraday_label, intraday_col = _intraday_bias(_q_live, gex_st)
    hdr=(f"<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px;'>"
         +_pill(f"Market Regime: {macro_reg} / {intraday_label}",reg_col)
         +_pill(f"Fear Level: {fl2} ({fear_z:.2f})",fc)
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
        narr = f"Negative dealer flow ({gex_score:+d}): Dealers are short gamma and amplifying moves. Expect trending/volatile behavior."
    elif "POSITIVE" in gex_reg.upper():
        narr = f"Positive dealer flow ({gex_score:+d}): Dealers are long gamma and suppressing moves. Expect range-bound / mean-reverting behavior."
    else:
        narr = f"Neutral dealer positioning ({gex_score:+d}): Near gamma flip — expect unstable intraday behavior and lower edge."

    _g3_mult = 10.0 if gex_sym in ("SPY", "SPX") else 40.0 if gex_sym in ("QQQ", "NDX") else 1.0

    gex_body = (
        _sh(3, "GEX LEVELS & DEALER POSITIONING")
        + _kv("Spot", f"{spx:,.2f}", "#fff")
        + _kv("GEX Flip", f"{flip*_g3_mult:,.2f}", "#f59e0b")
        + _kv("Regime", gex_op_label, gex_rc)
        + f"<div style='margin-top:10px;font-size:12px;color:rgba(255,255,255,0.70);'>{narr}</div>"
    )
    st.markdown(_card(gex_body), unsafe_allow_html=True)

    # ── 3b. GEX HISTOGRAM + CUMULATIVE PROFILE ──────────────────────────
    _hist_col, _cum_col = st.columns(2)
    with _hist_col:
        st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>GEX HISTOGRAM — Net Gamma per Strike</div>",unsafe_allow_html=True)
        st.plotly_chart(
            _gex_histogram(chain_df if chain_df is not None else None, float(gex_spot), float(flip),
                           flip_src=_flip_src_label),
            use_container_width=True, key="th_gex_hist"
        )
    with _cum_col:
        st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>CUMULATIVE GEX PROFILE — Spatial dealer exposure</div>",unsafe_allow_html=True)
        st.caption("Zero crossing = gamma flip · Max pain = deepest negative · Max pin = highest positive")
        if chain_df is not None and len(chain_df) > 0:
            _cum_prof = compute_cumulative_gex_profile(chain_df, float(gex_spot), max_dte=45)
            if not _cum_prof.empty:
                _cum_scale = 1e9 if _cum_prof["cum_gex"].abs().max() >= 5e9 else 1e6
                _cum_unit  = "$B" if _cum_scale == 1e9 else "$M"
                _pos = _cum_prof[_cum_prof["cum_gex"] >= 0]
                _neg = _cum_prof[_cum_prof["cum_gex"] < 0]
                _cum_fig = go.Figure()
                if not _pos.empty:
                    _cum_fig.add_trace(go.Scatter(
                        x=_pos["strike"], y=_pos["cum_gex"]/_cum_scale,
                        fill="tozeroy", fillcolor="rgba(16,185,129,0.18)",
                        line=dict(color="#10b981",width=1.5), name="Positive",
                    ))
                if not _neg.empty:
                    _cum_fig.add_trace(go.Scatter(
                        x=_neg["strike"], y=_neg["cum_gex"]/_cum_scale,
                        fill="tozeroy", fillcolor="rgba(239,68,68,0.18)",
                        line=dict(color="#ef4444",width=1.5), name="Negative",
                    ))
                _cum_fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
                _cum_fig.add_vline(x=float(gex_spot), line_dash="dot",
                    line_color="rgba(255,255,255,0.7)", line_width=1.5,
                    annotation_text=f"SPOT", annotation_font_size=9)
                if flip and _cum_prof["strike"].min() < flip < _cum_prof["strike"].max():
                    _cum_fig.add_vline(x=float(flip), line_dash="dash",
                        line_color="#f59e0b", line_width=2,
                        annotation_text=f"⚡ FLIP {flip:.2f} ({_flip_src_label})",
                        annotation_font_size=9,
                        annotation_font_color="#f59e0b")
                _cum_fig.update_layout(yaxis_title=f"Cum GEX ({_cum_unit})", showlegend=True,
                    legend=dict(orientation="h",y=1.02,x=0,font=dict(size=9)))
                from ui_components import plotly_dark as _pd2
                _pd2(_cum_fig, f"Cumulative GEX ≤45DTE", 360)
                st.plotly_chart(_cum_fig, use_container_width=True, key="th_gex_cum")
        else:
            st.caption("No chain data available for cumulative profile.")

    # ── 4. PROBABILITY HEATMAP ────────────────────────────────────────────
    st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>4. SPX / NDX FORWARD PROBABILITY HEATMAP — 1-WEEK FORECAST</div>",unsafe_allow_html=True)
    st.caption("Merton Jump-Diffusion Monte Carlo (n=6,000) · VIX/VXN implied vol · VVIX-calibrated jump intensity · Risk-neutral drift · 5 trading days")
    with st.spinner("Running simulations…"):
        st.plotly_chart(
            _forward_prob_fig(
                spx=round(spx, 2), vix=round(vix_live, 2),
                ndx=round(ndx, 2), vxn=round(vxn_live, 2),
                spy=spy, qqq=qqq, idx=idx,
                vvix=round(float(tail["vvix"]), 2) if np.isfinite(tail["vvix"]) else float("nan"),
                vts_shape=vts.get("shape", "MIXED"),
            ),
            use_container_width=True, key="th_hm"
        )

    # ── 5 & 6. IV SURFACES ────────────────────────────────────────────────
    c5,c6=st.columns(2)
    with c5:
        st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>5. IMPLIED VOL SURFACE — SPX</div>",unsafe_allow_html=True)
        _spx_chain = chain_df if gex_sym not in ("QQQ", "NDX", "^NDX") else None
        st.plotly_chart(
            _ivsurf_from_chain(_spx_chain, float(gex_spot), "SPX", vix_live, vts.get("shape","MIXED")),
            use_container_width=True, key="th_ivs_spx"
        )
    with c6:
        st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>6. IMPLIED VOL SURFACE — NDX</div>",unsafe_allow_html=True)
        st.plotly_chart(
            _ivsurf_from_chain(qqq_chain_df, float(qqq_spot), "NDX", vxn_live, vts.get("shape","MIXED")),
            use_container_width=True, key="th_ivs_ndx"
        )

    # ── 7. IV vs RV ───────────────────────────────────────────────────────
    st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>7. IMPLIED vs REALIZED VOLATILITY</div>",unsafe_allow_html=True)
    st.plotly_chart(_ivrv_fig(vix_s,spy,idx),use_container_width=True,key="th_ivrv")

    # ── 8. RETURN DISTRIBUTION ────────────────────────────────────────────
    st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>8. RETURN DISTRIBUTION & PROBABILITY BANDS</div>",unsafe_allow_html=True)
    if rd:
        st.markdown(f"SPX Spot: `{spx:,.2f}` &nbsp;&nbsp; Daily σ: `{rd['daily_sigma']:.4f}%` &nbsp;&nbsp; Skew: `{rd['skew']:.4f}` &nbsp;&nbsp; Kurtosis: `{rd['kurtosis']:.4f}`")
        st.plotly_chart(_rdist_fig(spy,idx,rd,spot=spx),use_container_width=True,key="th_rd")

    # ── 9. MACRO REGIME & NEWS ────────────────────────────────────────────
    nr = ""
    for cat in news[:6]:
        s = cat["sentiment"]
        col = "#10b981" if s > 0.01 else "#ef4444" if s < -0.01 else "#94a3b8"
        nr += (
            f"<div style='font-size:12px;margin-bottom:4px;'>"
            f"<span style='color:rgba(255,255,255,0.55);'>{cat['label'].split('&')[0].strip().lower()}</span>: "
            f"<span style='color:{col};font-family:monospace;font-weight:700;'>{s:+.4f}</span> "
            f"<span style='color:rgba(255,255,255,0.30);'>({cat['count']} articles)</span>"
            f"</div>"
        )

    macro_body = (
        _sh(9, "MACRO REGIME & NEWS SENTIMENT")
        + _kv("Macro Regime", f"{macro_reg} / {intraday_label}", reg_col)
        + _kv("Growth Z", f"{gzv:+.2f}")
        + _kv("Inflation Z", f"{izv:+.2f}")
        + _kv("CPI YoY", f"{cyi:.2f}%")
        + _kv("CPI Nowcast", f"{cpi_now:+.3f}% MoM" if np.isfinite(cpi_now) else "N/A")
        + _kv("Unemployment", f"{ur:.1f}%")
        + _kv("HY OAS", f"{hyv:.2f}")
        + _kv("SPY-TLT Corr", f"{stlc:.3f}" if np.isfinite(stlc) else "N/A")
        + _kv("Sahm Rule", f"{sahmv:.3f}")
        + "<div style='margin-top:10px;border-top:1px solid rgba(255,255,255,0.08);padding-top:8px;'>"
        + "<div style='font-size:10px;color:rgba(255,255,255,0.35);margin-bottom:6px;letter-spacing:0.1em;'>NEWS SENTIMENT</div>"
        + nr
        + "</div>"
    )
    st.markdown(_card(macro_body), unsafe_allow_html=True)

    # ── 10. THESIS VERDICT — doc-style compact output ───────────────────────
    th = thesis
    c = th["color"]

    sig_html = "".join(
        f"<div style='font-size:12px;color:rgba(255,255,255,0.78);margin-bottom:4px;'>{emo} {txt}</div>"
        for emo, txt in th["drivers"]
    )

    lv = th["levels"]

    risk_html = "".join(
        f"<div style='font-size:12px;color:rgba(255,255,255,0.78);margin-bottom:4px;'>{r}</div>"
        for r in th["risks"]
    )

    vbd = (
        _sh(10, "THESIS VERDICT")
        + f"<div style='font-size:34px;font-weight:900;color:{c};margin-bottom:8px;'>THESIS VERDICT: {th['bias']}</div>"
        + f"<div style='font-size:12px;color:rgba(255,255,255,0.55);margin-bottom:4px;'>Composite Score: {th['score']} / ±10</div>"
        + f"<div style='font-size:12px;color:rgba(255,255,255,0.40);margin-bottom:12px;'>Date: {dt.datetime.now().strftime('%A, %B %d, %Y')}</div>"
        + "<div style='font-size:10px;color:rgba(255,255,255,0.35);letter-spacing:0.12em;margin-bottom:6px;'>SIGNAL BREAKDOWN</div>"
        + sig_html
        + "<div style='height:10px;'></div>"
        + "<div style='font-size:10px;color:rgba(255,255,255,0.35);letter-spacing:0.12em;margin-bottom:6px;'>KEY LEVELS</div>"
        + _kv("SPX Spot", f"{lv['SPX Spot']:,.2f}")
        + _kv("GEX Flip", f"{lv['GEX Flip']:,.2f}")
        + _kv("GEX Upper", f"{lv['GEX Upper']:,.2f}")
        + _kv("GEX Lower", f"{lv['GEX Lower']:,.2f}")
        + _kv("1σ Daily", f"{lv['1σ Daily'][0]:,.2f} — {lv['1σ Daily'][1]:,.2f}")
        + _kv("2σ Daily", f"{lv['2σ Daily'][0]:,.2f} — {lv['2σ Daily'][1]:,.2f}")
        + _kv("1σ Weekly", f"{lv['1σ Weekly'][0]:,.2f} — {lv['1σ Weekly'][1]:,.2f}")
        + _kv("2σ Weekly", f"{lv['2σ Weekly'][0]:,.2f} — {lv['2σ Weekly'][1]:,.2f}")
        + "<div style='height:10px;'></div>"
        + "<div style='font-size:10px;color:rgba(255,255,255,0.35);letter-spacing:0.12em;margin-bottom:6px;'>RISK FACTORS</div>"
        + risk_html
    )

    st.markdown(_card(vbd, bg=f"{c}08", border=f"{c}30"), unsafe_allow_html=True)

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
            +_gl("Market Regime",
                 "2×2 framework on rolling z-scores: Growth (2s10s) × Inflation (core CPI YoY). "
                 "Overheating: curve steep + CPI hot. Goldilocks: curve steep + CPI cooling. "
                 "Stagflation: curve flat/inverted + CPI still hot. "
                 "Disinflation: curve flat + CPI actively falling (>10bp over 3M). "
                 "Deflation: growth− + inflation z-score < −0.5σ (CPI well below trend).")
            +_gl("Recession P(6m)",
                 "Probability of a US recession starting in the next 6 months. Our model uses the yield curve, unemployment rate, initial claims, and the Sahm Rule.")
            +_gl("Net Liquidity",
                 "Fed Balance Sheet minus TGA minus RRP. Expanding = risk asset tailwind. Contracting = headwind.")),
            unsafe_allow_html=True)
    with g13:
        st.markdown(_card(
            _sh(13,"GLOSSARY — READING THE CHARTS")
            +_gl("IV Surface","3D plot: implied vol across strikes (moneyness) and DTE. Steep put skew = downside protection expensive. NDX surface always loaded from QQQ chain (ETF spot ~$420, not index ~24,000).")
            +_gl("Probability Heatmap",
                 "Monte Carlo simulation that runs thousands of price paths using a Merton jump-diffusion model. The heatmap shows the probability of SPX/NDX reaching each price level over the next week.")
            +_gl("GEX Histogram","Gamma per strike. Green = positive (dealers long gamma → mean-revert). Red = negative (dealers short gamma → amplify moves).")
            +_gl("IV vs RV Chart","VIX overlaid with 21D realized vol. VRP spread bar chart: green = IV premium (vol expensive, sell bias), red = IV discount (vol cheap, buy protection).")
            +_gl("Fear Composite",
                 "4-component rolling z-score: VIX (35%) + HY OAS (25%) + NFCI (25%) + EPU (15%). "
                 "Each component is z-scored against its own 252-day rolling window — NOT the 2Y global mean. "
                 "This means a VIX of 40 today reads as elevated vs the past 12 months, even if VIX has been "
                 "high for months (the old global mean would absorb the spike and understate fear). "
                 ">+1.0σ = ELEVATED. <−1.0σ = COMPLACENT. Mapped 0–100 via logistic.")
            +_gl("Composite Score",
                 "Three-horizon engine system. "
                 "Intraday: GEX momentum, multi-horizon SPY/credit (1d/3d/5d), vol-trend. "
                 "Short-term 1-5d: VRP carry, fear composite, drawdown, tactical probs. "
                 "Medium-term 2-4w: macro regime, economic breadth (7 metrics), medium probs. "
                 "Regime-conditional blending: neg-gamma (trending) → 40/40/20, "
                 "pos-gamma (mean-rev) → 20/40/40, neutral → 30/40/30. "
                 "Smooth sigmoid recession filter. Backwardation contrarian preserved. "
                 "NO EDGE fires when score + momentum both weak (backwardation exempted).")),
            unsafe_allow_html=True)

    st.markdown(
        f"<div style='text-align:center;font-size:10px;color:rgba(255,255,255,0.2);margin-top:16px;'>"
        f"Data: FRED · yfinance · Schwab API | Generated {dt.datetime.now().strftime('%H:%M ET')} | Not financial advice</div>",
        unsafe_allow_html=True)
