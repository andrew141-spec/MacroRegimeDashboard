# pages/wim.py — render_world_intelligence_monitor
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

def render_world_intelligence_monitor(
    categorised_intel: Dict,
    alerts: List[str],
    setups: List[Dict],
    failure_modes: List,
    prob: Dict,
    geo_shock: float,
    geo_triggers: List[str],
    live_enabled: bool,
    refresh_sec: int,
    session: Dict,
):
    """
    Full-featured intelligence panel replacing the basic feed list.
    Organises headlines into 7 thesis-mapped signal categories,
    shows per-category shock scores, driver alerts, active setups,
    failure mode warnings, and probability breakdown.
    """
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Header ──────────────────────────────────────────────────────────
    now_str = dt.datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
  <div>
    <div style="font-family:var(--mono);font-size:10px;letter-spacing:1.2px;
                text-transform:uppercase;color:var(--muted);">World Intelligence Monitor</div>
    <div style="font-size:9px;color:var(--dim);margin-top:1px;">
      {now_str} · {'🟢 auto' if live_enabled else '⚫ manual'} · {refresh_sec}s
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    if st.button("⟳ Refresh feeds", use_container_width=True, key="wim_refresh"):
        st.cache_data.clear(); st.rerun()

    # ── OVERALL THREAT COMPOSITE ─────────────────────────────────────────
    # Weighted average of all category shock scores → single regime signal
    cat_scores = {k: category_shock_score(categorised_intel.get(k, []))
                  for k in INTEL_CATEGORIES}
    # Weighted composite (geo and fed_policy carry more weight)
    CAT_WEIGHTS = {
        "fed_policy": 0.22, "fiscal_debt": 0.12, "inflation_labor": 0.18,
        "trade_tariffs": 0.14, "geopolitical": 0.18, "markets_liquidity": 0.10,
        "ai_tech": 0.06,
    }
    composite_shock = sum(cat_scores[k] * w for k, w in CAT_WEIGHTS.items())
    composite_shock = float(np.clip(composite_shock, 0, 100))

    shock_c = "var(--green)" if composite_shock < 15 else (
              "var(--yellow)" if composite_shock < 30 else (
              "var(--orange)" if composite_shock < 55 else "var(--red)"))
    shock_label = ("LOW" if composite_shock < 15 else
                   "ELEVATED" if composite_shock < 30 else
                   "HIGH" if composite_shock < 55 else "CRITICAL")

    st.markdown(f"""
<div style="background:rgba(0,0,0,0.25);border:1px solid rgba(255,255,255,0.08);
            border-radius:12px;padding:10px 12px;margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div style="font-family:var(--mono);font-size:9px;letter-spacing:1px;
                text-transform:uppercase;color:var(--muted);">Global Shock Composite</div>
    <div style="font-family:var(--mono);font-size:11px;font-weight:700;
                color:{shock_c};">{shock_label} · {composite_shock:.0f}</div>
  </div>
  <div style="background:rgba(255,255,255,0.06);border-radius:999px;height:4px;
              width:100%;margin:6px 0 2px;">
    <div style="width:{composite_shock:.0f}%;height:4px;border-radius:999px;
                background:{shock_c};transition:width 400ms;"></div>
  </div>
  <div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:5px;">
    {"".join([
        f'<span style="font-family:var(--mono);font-size:9px;padding:2px 6px;'
        f'border-radius:999px;background:{INTEL_CATEGORIES[k]["bg"]};'
        f'border:1px solid {INTEL_CATEGORIES[k]["border"]};'
        f'color:{INTEL_CATEGORIES[k]["color"]};">'
        f'{INTEL_CATEGORIES[k]["icon"]} {cat_scores[k]:.0f}</span>'
        for k in INTEL_CATEGORIES
    ])}
  </div>
</div>""", unsafe_allow_html=True)

    # ── DRIVER ALERTS (score moves) ──────────────────────────────────────
    if alerts:
        alerts_html = "".join([
            f'<div style="padding:3px 0;font-size:11px;border-bottom:1px solid '
            f'rgba(255,255,255,0.04);">{a}</div>'
            for a in alerts[:6]
        ])
        st.markdown(f"""
<div style="background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.22);
            border-radius:12px;padding:9px 12px;margin-bottom:8px;">
  <div style="font-family:var(--mono);font-size:9px;letter-spacing:1px;
              text-transform:uppercase;color:var(--yellow);margin-bottom:5px;">
    ⚡ Score Driver Alerts
  </div>
  {alerts_html}
</div>""", unsafe_allow_html=True)

    # ── ACTIVE SETUP SIGNALS ─────────────────────────────────────────────
    active_setups = [s for s in setups if s["active"] and s["score"].tradeable]
    if active_setups:
        setup_rows = "".join([
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
            f'<span style="font-family:var(--mono);font-size:10px;'
            f'color:{"var(--green)" if s["score"].composite>=0.75 else "var(--yellow)"};">'
            f'S{s["setup"]} {s["name"][:22]}</span>'
            f'<span style="font-family:var(--mono);font-size:9px;color:var(--muted);">'
            f'{s["score"].composite:.2f} · {s["effective_size"]*100:.0f}%sz</span>'
            f'</div>'
            for s in active_setups
        ])
        st.markdown(f"""
<div style="background:rgba(139,92,246,0.07);border:1px solid rgba(139,92,246,0.22);
            border-radius:12px;padding:9px 12px;margin-bottom:8px;">
  <div style="font-family:var(--mono);font-size:9px;letter-spacing:1px;
              text-transform:uppercase;color:var(--purple);margin-bottom:5px;">
    🎯 Active GEX Setups
  </div>
  {setup_rows}
</div>""", unsafe_allow_html=True)

    # ── FAILURE MODE WARNINGS ────────────────────────────────────────────
    triggered_fm = [fm for fm in failure_modes if fm[3]]
    if triggered_fm:
        fm_rows = "".join([
            f'<div style="font-size:10px;padding:2px 0;color:var(--orange);">'
            f'[FM{fid}] {fname}</div>'
            for fid, fname, _, _ in triggered_fm
        ])
        st.markdown(f"""
<div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.22);
            border-radius:12px;padding:9px 12px;margin-bottom:8px;">
  <div style="font-family:var(--mono);font-size:9px;letter-spacing:1px;
              text-transform:uppercase;color:var(--red);margin-bottom:4px;">
    ⚠ Failure Mode Warnings
  </div>
  {fm_rows}
</div>""", unsafe_allow_html=True)

    # ── PROBABILITY STRIP ────────────────────────────────────────────────
    bp = prob["bull_prob"]; bc = "var(--green)" if bp>60 else ("var(--red)" if bp<40 else "var(--yellow)")
    st.markdown(f"""
<div style="background:rgba(0,0,0,0.20);border:1px solid rgba(255,255,255,0.07);
            border-radius:12px;padding:9px 12px;margin-bottom:8px;">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
    <span style="font-family:var(--mono);font-size:9px;color:var(--muted);">BULL PROB</span>
    <span style="font-family:var(--mono);font-size:11px;font-weight:700;color:{bc};">{bp:.0f}%</span>
  </div>
  {pbar(bp, bc)}
  <div style="display:flex;justify-content:space-between;margin-top:5px;font-size:9px;color:var(--dim);font-family:var(--mono);">
    <span>Fwd-looking {prob["fwd_prob"]:.0f}%</span>
    <span>Coincident {prob["coincident_prob"]:.0f}%</span>
    <span>GEX {prob["gex_adjustment"]:+.1f}pt</span>
    <span>Geo −{prob["geo_drag"]:.0f}pt</span>
  </div>
  {"<div style=\"margin-top:5px;font-size:9px;color:var(--yellow);font-family:var(--mono);\">⚡ DIVERGENCE "+str(round(prob['divergence_gap']))+"ppt</div>" if prob["divergent"] else ""}
</div>""", unsafe_allow_html=True)

    # ── 7-CATEGORY INTELLIGENCE GRID ────────────────────────────────────
    st.markdown("""<div style="font-family:var(--mono);font-size:9px;letter-spacing:1.2px;
        text-transform:uppercase;color:var(--muted);margin:8px 0 5px;">Signal Categories</div>""",
        unsafe_allow_html=True)

    for cat_key, cat_data in INTEL_CATEGORIES.items():
        items_in_cat = categorised_intel.get(cat_key, [])
        shock = cat_scores[cat_key]
        shock_c_cat = ("var(--green)" if shock < 10 else
                       "var(--yellow)" if shock < 25 else
                       "var(--orange)" if shock < 50 else "var(--red)")
        n_items = len(items_in_cat)

        with st.expander(
            f"{cat_data['icon']} {cat_data['label']}  ·  {n_items} signals  ·  {shock:.0f}",
            expanded=(shock > 20)  # auto-expand high-shock categories
        ):
            if not items_in_cat:
                st.markdown("<span class='small'>No signals in this category right now.</span>",
                            unsafe_allow_html=True)
            else:
                for score, it in items_in_cat[:6]:
                    pub = it.published[:10] if it.published else ""
                    intensity = ("🔴" if score >= 8 else "🟡" if score >= 4 else "⚪")
                    title_disp = it.title[:72] + ("…" if len(it.title) > 72 else "")
                    link_part = (f' <a href="{it.link}" target="_blank" '
                                 f'style="color:var(--dim);font-size:9px;">↗</a>'
                                 if it.link else "")
                    st.markdown(
                        f'<div style="padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
                        f'<div style="font-size:11px;">{intensity} {title_disp}{link_part}</div>'
                        f'<div style="font-family:var(--mono);font-size:9px;color:var(--dim);'
                        f'margin-top:1px;">{it.source}'
                        f'{" · " + pub if pub else ""} · score {score:.0f}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # Regime impact note
            impact = cat_data.get("regime_impact", "")
            impact_map = {
                "three_puts": "Impacts: Three Puts composite",
                "treasury_put": "Impacts: Treasury Put score",
                "fed_put": "Impacts: Fed Put / cut threshold",
                "trump_put": "Impacts: Trump Put / intervention P",
                "geo_shock": "Impacts: Geo shock drag on bull prob",
                "market_index": "Impacts: Market index score",
                "bubble_score": "Impacts: Bubble monitor",
            }
            if impact in impact_map:
                st.markdown(
                    f'<div style="font-family:var(--mono);font-size:9px;color:var(--dim);'
                    f'margin-top:6px;padding-top:4px;border-top:1px solid rgba(255,255,255,0.05);">'
                    f'{impact_map[impact]}</div>',
                    unsafe_allow_html=True
                )

    # ── THESIS PUTS STATUS ────────────────────────────────────────────────
    st.markdown("""<div style="font-family:var(--mono);font-size:9px;letter-spacing:1.2px;
        text-transform:uppercase;color:var(--muted);margin:8px 0 5px;">Three Puts Status</div>""",
        unsafe_allow_html=True)

    # Derive a narrative status for each put from category shock scores
    fed_stress  = cat_scores["fed_policy"]
    fiscal_ok   = cat_scores["fiscal_debt"] < 30
    tariff_heat = cat_scores["trade_tariffs"]
    geo_heat    = cat_scores["geopolitical"]

    puts_status = [
        ("🏛 Treasury Put",
         "Active" if fiscal_ok else "Under Stress",
         "var(--green)" if fiscal_ok else "var(--orange)",
         "TGA + RRP + balance sheet supporting liquidity"
         if fiscal_ok else "Fiscal signals showing stress — watch TGA drawdown"),
        ("🦅 Fed Put",
         "Conditional" if fed_stress < 40 else "At Risk",
         "var(--yellow)" if fed_stress < 40 else "var(--red)",
         "Warsh/balance-sheet risk: rates may fall while BS shrinks"
         if fed_stress > 25 else "Rate cut window intact if CPI cooperates"),
        ("🎭 Trump Put",
         "Active" if tariff_heat < 50 else "Tested",
         "var(--green)" if tariff_heat < 50 else "var(--yellow)",
         "90-day pause logic intact — policy reversal if SPX -7%+"
         if tariff_heat < 50 else "Elevated tariff/trade signals — watch for policy pivot"),
    ]

    for name, status, color, note in puts_status:
        st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:flex-start;
            padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
  <div>
    <div style="font-size:11px;font-weight:600;">{name}</div>
    <div style="font-size:9px;color:var(--dim);margin-top:1px;">{note}</div>
  </div>
  <div style="font-family:var(--mono);font-size:10px;font-weight:700;
              color:{color};white-space:nowrap;margin-left:8px;">{status}</div>
</div>""", unsafe_allow_html=True)

    # ── FOOTER ────────────────────────────────────────────────────────────
    total_feeds = len(_all_feeds_flat())
    total_items = sum(len(v) for v in categorised_intel.values())
    st.markdown(f"""
<div style="margin-top:8px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.05);">
  <div style="font-family:var(--mono);font-size:9px;color:var(--dim);">
    {total_feeds} feeds · {total_items} categorised signals · cached 5min
  </div>
  <div style="font-family:var(--mono);font-size:9px;color:var(--dim);margin-top:1px;">
    Sources: Fed · FOMC · BLS · BEA · Treasury · Reuters · BBC
  </div>
</div>""", unsafe_allow_html=True)


