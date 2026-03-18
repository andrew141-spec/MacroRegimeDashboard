# pages/dashboard.py — render_dashboard
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
from config import GammaState, GammaRegime, FeedItem, SetupScore, REGIME_COLORS, CSS
from utils import _to_1d, zscore, resample_ffill, yf_close, kelly, current_pct_rank
from config import _get_secret
from ui_components import pill, pbar, sec_hdr, plotly_dark, regime_chip, autorefresh_js, colored, gauge
from gex_engine import build_gamma_state, compute_gex_from_chain
from schwab_api import get_schwab_client, schwab_get_spot, schwab_get_options_chain, schwab_run_auth_flow, schwab_complete_auth, _get_supabase, SCHWAB_AVAILABLE, SUPABASE_AVAILABLE
from data_loaders import load_macro, get_gex_from_yfinance, get_fwd_pe
from intel_monitor import load_feeds, geo_shock_score, score_relevance, categorise_items, category_shock_score, _all_feeds_flat, INTEL_CATEGORIES
from signals import compute_leading_stack, compute_1d_prob
from probability import compute_prob_composite, get_session_context, evaluate_setups, check_failure_modes, classify_macro_regime_abs, regime_transition_prob, driver_alerts
from page_wim import render_world_intelligence_monitor

def render_dashboard():
    """Main integrated dashboard."""
    st.markdown(CSS, unsafe_allow_html=True)

    # ── Persist user inputs across full page reloads via query params ──────
    qp = st.query_params
    def _qp(key, default):
        v = qp.get(key)
        return v if v is not None else default

    st.sidebar.markdown("### Controls")
    start = st.sidebar.date_input("Start", value=dt.date.today()-dt.timedelta(days=730), key="dash_start")
    end   = st.sidebar.date_input("End",   value=dt.date.today(), key="dash_end")
    ticker_tile = st.sidebar.text_input("Ticker Tile", _qp("ticker", "QQQ"), key="dash_ticker").upper().strip()
    cpi_thresh  = st.sidebar.number_input("Core CPI cut threshold", float(_qp("cpi", "3.0")), step=0.1, key="dash_cpi_thresh")
    gex_symbol  = st.sidebar.text_input("GEX Symbol", _qp("gex", "SPY"), key="dash_gex_symbol").upper().strip()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Live Feed")
    live_enabled = st.sidebar.toggle("Auto refresh", True, key="dash_live_enabled")
    refresh_sec  = int(st.sidebar.slider("Refresh (s)", 30, 300, 90, 15, key="dash_refresh_sec"))

    # Write current values into URL so they survive the next reload
    qp["ticker"] = ticker_tile
    qp["gex"]    = gex_symbol
    qp["cpi"]    = str(round(cpi_thresh, 2))

    autorefresh_js(refresh_sec, live_enabled)

    idx = pd.date_range(start, end, freq="D")

    with st.spinner("Loading macro + market data..."):
        raw = load_macro(start.isoformat(), end.isoformat())
        def r(k): return resample_ffill(raw.get(k, pd.Series(dtype=float)), idx)
        y3m=r("DGS3MO"); y2=r("DGS2"); y10=r("DGS10"); y30=r("DGS30")
        cpi=r("CPIAUCSL"); core=r("CPILFESL"); unrate=r("UNRATE"); claims=r("ICSA")
        walcl=r("WALCL"); tga=r("WTREGEN"); rrp=r("RRPONTSYD"); m2=r("M2SL")
        nfci=resample_ffill(raw.get("NFCI",pd.Series(dtype=float)),idx).fillna(0)
        vix=r("VIX"); spy=r("SPY"); tlt=r("TLT"); qqq=r("QQQ")
        copx=r("COPX"); gld=r("GLD"); hyg=r("HYG"); lqd=r("LQD")
        dxy=r("UUP"); iwm=r("IWM")
        # New series
        tips_10y      = r("DFII10")
        bank_reserves = r("WRBWFRBL")
        bank_credit   = r("TOTBKCR")
        ism_no_raw    = raw.get("AMTMNO", pd.Series(dtype=float))
        ism_no        = ism_no_raw if len(ism_no_raw.dropna()) > 4 else None
        gdp_quarterly = r("GDPC1")
        mmmf          = r("WRMFSL")

    # ── DERIVED ──
    core_yoy = (core/core.shift(365)-1)*100
    s_2s10s  = (y10-y2)*100; s_3m10y = (y10-y3m)*100
    # Net liquidity = Fed balance sheet minus fiscal/repo drains (pure plumbing).
    # Note: M2 was removed vs v2 (walcl+M2-rrp-tga). The v2 formula was a
    # broader "system money" measure; this is a tighter "Fed-driven liquidity"
    # measure. Both are valid; this version avoids double-counting M2 with the
    # M2 YoY signal in the medium-term bucket.
    net_liq  = (walcl - tga - rrp) / 1000.0
    net_liq_4w = net_liq.diff(28)
    bs_13w   = walcl.diff(91)/1000.0
    y10_20 = y10.diff(20); y3m_20 = y3m.diff(20)
    policy_mistrust = ((y3m_20>0)&(y10_20<0)).astype(int)
    warsh_decouple  = ((y10_20<0)&(bs_13w<0)).astype(int)
    tga_dd = (tga.diff(28)<0).astype(int); rrp_dep = (rrp<50).astype(int)

    # Absolute regime classification (not z-score based)
    core_yoy_latest  = float(core_yoy.dropna().iloc[-1]) if core_yoy.dropna().size else 2.5
    curve_raw_latest = float(s_2s10s.dropna().iloc[-1]) if s_2s10s.dropna().size else 0.0
    macro_regime = classify_macro_regime_abs(core_yoy_latest, curve_raw_latest)

    # Z-scores kept for display / legacy compatibility only
    growth_z = zscore(s_2s10s.fillna(0))
    infl_z   = zscore(core_yoy.fillna(core_yoy_latest))

    liq_axis = np.tanh(_to_1d(bs_13w.fillna(0)+net_liq_4w.fillna(0)))
    cap_axis = -np.tanh(_to_1d(y10_20.fillna(0)))
    def section_label(cap, liq):
        if cap and liq: return "C"
        if cap: return "D"
        if liq: return "B"
        return "A"
    section = section_label(bool(cap_axis.iloc[-1]>0.15), bool(liq_axis.iloc[-1]>0.15))

    vix_z  = zscore(vix.fillna(float(vix.dropna().median()) if vix.dropna().size else 20.0))
    nfci_z = zscore(nfci.fillna(0))
    inv       = (s_2s10s<0).astype(int)
    liq_tight = (net_liq_4w<0).astype(int)
    fear_raw   = 0.45*vix_z+0.35*nfci_z+0.10*inv+0.10*liq_tight
    fear_score = float(((fear_raw.iloc[-1]+2)/4).clip(0,1)*100)

    if section=="C" and fear_score<60 and float(net_liq_4w.iloc[-1])>=0: risk_regime="Risk-On"
    elif fear_score>=70 or section in("A","D"): risk_regime="Risk-Off"
    else: risk_regime="Neutral"

    idx_raw = 0.45*np.tanh(float(liq_axis.iloc[-1]))+0.20*np.tanh(float(cap_axis.iloc[-1]))+              0.15*np.tanh(float(growth_z.iloc[-1]))-0.12*np.tanh(float(infl_z.iloc[-1]))-              0.08*(fear_score/100)
    market_index_score = float(np.clip(idx_raw,-1,1)*100)

    spy_6m_high = spy.rolling(126).max(); dd = (spy/spy_6m_high-1).fillna(0)
    trump_put = float(np.clip(45+35*float(dd.iloc[-1]<=-0.07)+20*(fear_score>60),0,100))
    unemp_3m  = float(unrate.diff(90).iloc[-1]) if len(unrate)>90 else 0.0
    fed_put   = float(np.clip(55+25*float((y10_20.iloc[-1]<0)and(unemp_3m>0))-                10*float((core_yoy_latest-cpi_thresh)>0)-15*float(warsh_decouple.iloc[-1]),0,100))
    treas_put = float(np.clip(50+20*float(tga_dd.iloc[-1])+15*float(rrp_dep.iloc[-1])+15*float(net_liq_4w.iloc[-1]>=0),0,100))
    three_puts = float(np.clip(0.35*treas_put+0.35*fed_put+0.30*trump_put,0,100))
    liq_anxiety = float(np.clip(50+30*float(warsh_decouple.iloc[-1])+20*float(policy_mistrust.iloc[-1]),0,100))
    slr_proxy  = float((nfci_z.iloc[-1]>1.0)and(float(net_liq_4w.iloc[-1])<0))
    spy_mom    = float(spy.pct_change(21).iloc[-1]) if spy.dropna().size>25 else 0.0
    earn_decay = float((spy_mom<0)and(float(y10_20.iloc[-1])>=0))
    exhaustion = float(np.clip(100*(0.40*float(policy_mistrust.iloc[-1])+0.35*slr_proxy+0.25*earn_decay),0,100))

    mag7 = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA"]
    pe_vals = [v for v in [get_fwd_pe(t) for t in mag7] if np.isfinite(v)]
    mag7_pe = float(np.nanmean(pe_vals)) if pe_vals else np.nan
    mom_6m = float(qqq.pct_change(126).iloc[-1]) if qqq.dropna().size>130 else 0.0
    # Bubble score: continuous logistic functions replacing arbitrary binary thresholds.
    # PE component centred at 38x (midpoint between fair ~25x and expensive ~50x).
    # Momentum component centred at 25% 6M return. PE weighted 65% vs 35% momentum.
    if np.isfinite(mag7_pe) and mag7_pe > 0:
        pe_component  = float(100 / (1 + math.exp(-0.25 * (mag7_pe - 38))))
    else:
        pe_component  = 50.0
    mom_component = float(100 / (1 + math.exp(-15 * (mom_6m - 0.25))))
    bubble_score  = 0.65 * pe_component + 0.35 * mom_component
    bubble_label  = "Extreme" if bubble_score >= 78 else ("Elevated" if bubble_score >= 58 else "Normal")
    tga_4w = float(tga.diff(28).iloc[-1]) if tga.dropna().size>35 else 0.0
    rrp_level = float(rrp.iloc[-1]) if rrp.dropna().size else np.nan
    bs_4w = float(walcl.diff(28).iloc[-1]) if walcl.dropna().size>35 else 0.0
    stealth_score = 35*(1 if tga_4w<-50 else 0)+35*(1 if np.isfinite(rrp_level) and rrp_level<100 else 0)+30*(1 if bs_4w>0 else 0)
    stealth_label = "Active" if stealth_score>=60 else ("Mild" if stealth_score>=30 else "Off")

    # ── GEX ──
    chain_df, spot, gex_source = get_gex_from_yfinance(gex_symbol)
    if chain_df is not None:
        gex_state = build_gamma_state(chain_df, spot, gex_source)
        st.session_state["_last_good_gex"] = {
            "state": gex_state,
            "spot":  spot,
            "saved_at": dt.datetime.now().strftime("%a %b %d %H:%M ET"),
        }
        gex_stale = False
    else:
        cached = st.session_state.get("_last_good_gex")
        if cached:
            gex_state = cached["state"]
            spot      = cached["spot"]
            gex_stale = cached["saved_at"]
        else:
            gex_state = GammaState(data_source="unavailable", timestamp=dt.datetime.now().strftime("%H:%M:%S"))
            gex_stale = False

    # ── LEADING + PROB ──
    leading = compute_leading_stack(
        y2, y3m, y10, y30, s_2s10s, vix, m2, claims,
        copx, gld, hyg, lqd, dxy, spy, qqq, iwm,
        net_liq, net_liq_4w, walcl, bs_13w, idx,
        tips_10y=tips_10y, bank_reserves=bank_reserves,
        bank_credit=bank_credit, ism_no=ism_no,
        gdp_quarterly=gdp_quarterly, mmmf=mmmf,
    )
    meta = regime_transition_prob(macro_regime, core_yoy, s_2s10s)
    all_rss = load_feeds(tuple(_all_feeds_flat().items()), 120)
    geo_shock, geo_triggers = geo_shock_score(all_rss)
    relevant_rss = score_relevance(all_rss, 12)
    categorised_intel = categorise_items(all_rss)
    # Pre-compute coincident inputs that require local series (nfci, net_liq_4w, idx)
    nfci_coincident    = float(current_pct_rank(-_to_1d(nfci).reindex(idx).ffill(), 252))
    liq_dir_now        = float(net_liq_4w.dropna().iloc[-1]) if net_liq_4w.dropna().size else 0.0
    liq_dir_coincident = float(50.0 + np.sign(liq_dir_now) * 20)   # 70=expanding, 30=draining

    prob = compute_prob_composite(
        leading, fear_score, geo_shock, meta["p_change_20d"], gex_state,
        nfci_coincident=nfci_coincident,
        liq_dir_coincident=liq_dir_coincident,
    )

    # ── 1-DAY MODEL ──────────────────────────────────────────────────────
    vix_level = float(vix.dropna().iloc[-1]) if vix.dropna().size else 20.0
    prob_1d = compute_1d_prob(
        gex_state=gex_state,
        spot=spot,
        vix_level=vix_level,
        vix_series=vix,
        spy_series=spy,
        hyg_series=hyg,
        lqd_series=lqd,
        dxy_series=dxy,
        s_2s10s=s_2s10s,
        net_liq_4w=net_liq_4w,
        nfci_z=nfci_z,
        fear_score=fear_score,
        session=get_session_context(),
        idx=idx,
    )
    session = get_session_context()
    # vix_level already computed above in 1D model block
    failure_modes = check_failure_modes(gex_state, session, vix_level, session["is_data_day"])
    setups = evaluate_setups(gex_state, session, spot, fear_score, vix_level)

    # ── STATE TRACKING ──
    now_state = {
        "Fear": fear_score, "Three Puts": three_puts, "Liquidity Anxiety": liq_anxiety,
        "Exhaustion": exhaustion, "Market Index": market_index_score, "Bull Prob": prob["bull_prob"],
        "Risk Regime": risk_regime, "Macro Regime": macro_regime, "Bubble": bubble_label,
        "Stealth QE": stealth_label, "Section": section,
        "GEX Regime": gex_state.regime.value,
        "Overall": "Bullish" if prob["bull_prob"]>60 else ("Bearish" if prob["bull_prob"]<40 else "Neutral"),
    }
    if "prev_state" not in st.session_state: st.session_state.prev_state = now_state.copy()
    alerts = driver_alerts(st.session_state.prev_state, now_state)
    st.session_state.prev_state = now_state.copy()

    # ── REGIME LABEL — derived from macro regime + risk signal, not raw arithmetic ──
    # Per architecture critique: the old overall_raw used ad-hoc denominators
    # (120, 140) and arbitrary thresholds (0.20, 0.05, -0.10) with no
    # principled basis.  Replaced with regime-first classification:
    # macro regime is the primary anchor; fear/liquidity modify the label.
    regime_to_bias = {
        "Goldilocks":          ("Constructive", "yellow"),
        "Overheating":         ("Cautious",     "orange"),
        "Stagflation":         ("Defensive",    "red"),
        "Deflation/Recession": ("Defensive",    "red"),
    }
    base_label, base_tone = regime_to_bias.get(macro_regime, ("Neutral", "orange"))
    # Modify by risk regime and prob composite
    bp_centre = prob.get("bull_rounded", 50)
    if risk_regime == "Risk-On" and bp_centre >= 55:
        o_label, o_tone = "Bullish (Risk-On)", "green"
    elif risk_regime == "Risk-Off" or bp_centre <= 40:
        o_label, o_tone = "Defensive (Risk-Off)", "red"
    elif macro_regime == "Goldilocks" and bp_centre >= 55:
        o_label, o_tone = "Bullish (Goldilocks)", "green"
    elif macro_regime in ("Overheating", "Stagflation") and bp_centre <= 45:
        o_label, o_tone = "Defensive", "red"
    else:
        o_label, o_tone = base_label, base_tone

    # ════════════════════════════════════════════════════
    # RENDER LAYOUT
    # ════════════════════════════════════════════════════
    main_col, right_col = st.columns([2.35, 1.0], gap="large")

    with main_col:
        # HEADER
        st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:6px;">
  <div>
    <h2 style="margin:0;font-size:22px;">⚡ Quant Regime Dashboard</h2>
    <div class="small" style="margin-top:2px;">GEX + Macro + Probabilistic Engine · {session['time_et']}</div>
  </div>
  <div style="display:flex;gap:5px;flex-wrap:wrap;justify-content:flex-end;">
    {pill("Macro",macro_regime)} {pill("Section",section)} {pill("Risk",risk_regime)}
    {pill("Bubble",bubble_label)} {pill("Stealth QE",stealth_label)}
    {regime_chip(gex_state.regime)}
  </div>
</div>""", unsafe_allow_html=True)

        # SESSION CONTEXT BAR
        sm = session["size_mult"]
        sc = "var(--green)" if sm >= 0.9 else ("var(--yellow)" if sm >= 0.5 else ("var(--orange)" if sm > 0 else "var(--red)"))
        st.markdown(f"""
<div class='warn-card' style='{"background:rgba(16,185,129,0.07);border-color:rgba(16,185,129,0.30);" if session["prime_time"] else ""}margin-bottom:8px;'>
  <span style="font-family:var(--mono);font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);">SESSION</span>
  <span style="margin-left:10px;font-weight:700;color:{sc};">{session['window']}</span>
  <span style="margin-left:8px;font-size:11px;color:var(--muted);">Liquidity: <b>{session['liquidity']}</b> · Size mult: <b>{sm:.2f}x</b></span>
  <span style="margin-left:8px;font-size:11px;color:var(--dim);">{session['note']}</span>
</div>""", unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── PROBABILITY ROW — 1D hero + three forward horizons ──
        st.markdown(f"{sec_hdr('DIRECTIONAL SIGNAL — BY HORIZON')}", unsafe_allow_html=True)

        p1d  = prob_1d.get("prob_1d",  50.0)
        p5d  = prob.get("prob_5d",  prob["bull_prob"])
        p21d = prob.get("prob_21d", prob["bull_prob"])
        p63d = prob.get("prob_63d", prob["bull_prob"])
        k1d  = prob_1d.get("kelly_1d", 0.0)
        k5d  = prob.get("kelly_5d",  0.0)
        k21d = prob.get("kelly_21d", 0.0)
        k63d = prob.get("kelly_63d", 0.0)

        def _hcol(p): return "#10b981" if p>60 else ("#ef4444" if p<40 else "#f59e0b")
        def _hbar(p, color):
            return (f"<div style='background:rgba(255,255,255,0.06);border-radius:999px;"
                    f"height:3px;width:100%;margin:3px 0;'>"
                    f"<div style='width:{p:.0f}%;height:3px;border-radius:999px;"
                    f"background:{color};'></div></div>")

        # ── 1D HERO ROW — full-width, dominant visual ───────────────────
        c1d    = _hcol(p1d)
        lo1d   = prob_1d.get("lo_1d", p1d - 15)
        hi1d   = prob_1d.get("hi_1d", p1d + 15)
        unc1d  = prob_1d.get("unc_1d", 15.0)
        dom    = prob_1d.get("dominant_signal", "")
        dom_d  = prob_1d.get("dominant_dir", "")
        gex_c  = prob_1d.get("gex_confidence", 0.5)
        sess_ok = prob_1d.get("session_valid", True)
        gex_regime_str = gex_state.regime.value
        flip_px = prob_1d.get("flip_proximity", 1.0)

        # Full-width 1D card — this is the most actionable number for an intraday trader
        hero_col, meta_col = st.columns([1.6, 1.0])
        with hero_col:
            sess_warn = "⚠ Thin session — signal less reliable" if not sess_ok else ""
            st.markdown(f"""
<div style='background:linear-gradient(135deg,rgba(16,185,129,0.10),rgba(16,185,129,0.04));
            border:1.5px solid rgba(16,185,129,0.35);border-radius:16px;padding:18px 22px;'>
  <div style='display:flex;align-items:baseline;gap:14px;'>
    <div>
      <div style='font-family:var(--mono);font-size:10px;letter-spacing:1.2px;
                  text-transform:uppercase;color:var(--green);margin-bottom:4px;'>
        1-Day Bull Probability · GEX-Conditioned
      </div>
      <div style='font-size:52px;font-weight:700;color:{c1d};font-family:var(--mono);
                  line-height:1;letter-spacing:-1px;'>{p1d:.0f}%</div>
      <div style='font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:4px;'>
        Range {lo1d:.0f}–{hi1d:.0f}% · ±{unc1d:.0f}pp uncertainty
      </div>
    </div>
    <div style='flex:1;'>
      {_hbar(p1d, c1d)}
      <div style='margin-top:8px;display:flex;gap:14px;flex-wrap:wrap;'>
        <span style='font-family:var(--mono);font-size:10px;color:var(--muted);'>
          35%K: <b style='color:{c1d};'>{k1d*100:.0f}%</b>
        </span>
        <span style='font-family:var(--mono);font-size:10px;color:var(--muted);'>
          GEX regime: <b>{gex_regime_str}</b>
        </span>
        <span style='font-family:var(--mono);font-size:10px;color:var(--muted);'>
          Dominant: <b>{dom}</b> ({dom_d})
        </span>
        <span style='font-family:var(--mono);font-size:10px;color:var(--muted);'>
          Conf: <b>{gex_c:.2f}</b> · Flip prox: <b>{flip_px:.2f}</b>
        </span>
      </div>
      {f"<div style='margin-top:6px;font-family:var(--mono);font-size:10px;color:var(--yellow);'>{sess_warn}</div>" if sess_warn else ""}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        with meta_col:
            # 1D component breakdown inline (no expander)
            scores_1d = {
                "VIX TS":      prob_1d.get("score_vts",   50),
                "Momentum":    prob_1d.get("score_mom",   50),
                "Credit/FX":   prob_1d.get("score_micro", 50),
                "Curve":       prob_1d.get("score_curve", 50),
                "Liquidity":   prob_1d.get("score_liq",   50),
            }
            rows_html = "".join([
                f"<div style='display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.04);'>"
                f"<span style='font-family:var(--mono);font-size:10px;color:var(--muted);'>{k}</span>"
                f"<span style='font-family:var(--mono);font-size:10px;color:{_hcol(v)};font-weight:700;'>{v:.0f}</span>"
                f"</div>"
                for k, v in scores_1d.items()
            ])
            st.markdown(f"""<div class='panel' style='height:100%;'>
  <div class='panel-title'>1D Component Scores</div>
  {rows_html}
  <div style='font-family:var(--mono);font-size:9px;color:var(--dim);margin-top:6px;'>
    AUC realistic: 0.52–0.55 · Not investment advice
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── FORWARD HORIZONS ROW — smaller cards below 1D ──────────────
        ph1, ph2, ph3 = st.columns(3)
        with ph1:
            c = _hcol(p5d)
            st.markdown(f"""<div class='prob-card'>
              <div class='panel-title'>5-Day <span style='color:var(--dim);font-size:9px;'>(Tactical)</span></div>
              <div style='font-size:26px;font-weight:700;color:{c};font-family:var(--mono);'>{p5d:.0f}%</div>
              {_hbar(p5d, c)}
              <div class='small'>35%K: {k5d*100:.0f}% · R:R~1.3</div>
              <div class='small' style='color:var(--dim);'>VIX TS · DXY 5D</div>
            </div>""", unsafe_allow_html=True)
        with ph2:
            c = _hcol(p21d)
            st.markdown(f"""<div class='prob-card'>
              <div class='panel-title'>21-Day <span style='color:var(--dim);font-size:9px;'>(Short-term)</span></div>
              <div style='font-size:26px;font-weight:700;color:{c};font-family:var(--mono);'>{p21d:.0f}%</div>
              {_hbar(p21d, c)}
              <div class='small'>½K: {k21d*100:.0f}% · R:R~2.0</div>
              <div class='small' style='color:var(--dim);'>HYG/LQD · SC · Liq 4W · ISM</div>
            </div>""", unsafe_allow_html=True)
        with ph3:
            c = _hcol(p63d)
            p_ch = meta["p_change_20d"]; persist = meta["persistence"]
            pc_c = "#ef4444" if p_ch>55 else ("#f59e0b" if p_ch>35 else "#10b981")
            st.markdown(f"""<div class='prob-card'>
              <div class='panel-title'>63-Day <span style='color:var(--dim);font-size:9px;'>(Medium-term)</span></div>
              <div style='font-size:26px;font-weight:700;color:{c};font-family:var(--mono);'>{p63d:.0f}%</div>
              {_hbar(p63d, c)}
              <div class='small'>½K: {k63d*100:.0f}% · R:R~2.5</div>
              <div class='small' style='color:var(--dim);'>Curve · Cu/Au · Credit · Real Rate</div>
              <div style='margin-top:4px;font-size:9px;font-family:var(--mono);color:{pc_c};'>
                P(regime Δ 20d): {p_ch:.0f}% · held {persist}d
              </div>
            </div>""", unsafe_allow_html=True)

        # Cross-horizon divergence
        if prob["divergent"] or abs(p1d - p63d) > 20:
            div1 = f"1D: {p1d:.0f}%"
            div5 = f"5D: {p5d:.0f}%"
            div63 = f"63D: {p63d:.0f}%"
            st.markdown(f"""<div class='warn-card' style='margin-top:6px;'>
              ⚡ <b>CROSS-HORIZON SIGNAL</b> — {div1} · {div5} · {div63}
              <span class='small'> · Trade the horizon matching your setup timeframe</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── GEX SNAPSHOT ──
        st.markdown(f"{sec_hdr('GAMMA EXPOSURE · ' + gex_state.regime.value + ' · ' + gex_state.timestamp)}", unsafe_allow_html=True)

        if gex_stale:
            st.markdown(f"""<div class='warn-card' style='margin-bottom:8px;'>
              🕐 <b>LAST KNOWN GEX</b> — market closed · data from <b>{gex_stale}</b> · levels valid for context only, not live trading
            </div>""", unsafe_allow_html=True)
        elif "prev close" in gex_state.data_source or "cached" in gex_state.data_source:
            st.markdown(f"""<div class='warn-card' style='margin-bottom:8px;'>
              📋 <b>EOD GEX</b> — market closed · {gex_state.data_source} · levels are indicative until RTH opens
            </div>""", unsafe_allow_html=True)
        elif gex_state.data_source == "unavailable":
            st.markdown("""<div class='alert-card' style='margin-bottom:8px;'>
              ⚠ <b>GEX UNAVAILABLE</b> — no options data yet · will populate once market opens and a refresh occurs
            </div>""", unsafe_allow_html=True)
        gx1, gx2, gx3 = st.columns(3)
        with gx1:
            flip_disp = f"{gex_state.gamma_flip:.2f}" if gex_state.gamma_flip else "N/A"
            dist_c = "var(--green)" if gex_state.distance_to_flip_pct > 1 else ("var(--red)" if gex_state.distance_to_flip_pct < -1 else "var(--yellow)")
            st.markdown(f"""<div class='panel'>
              <div class='panel-title'>Gamma Flip Level</div>
              <div style='font-size:24px;font-weight:700;font-family:var(--mono);color:var(--yellow);'>{flip_disp}</div>
              <div class='small'>Distance: <span style='color:{dist_c};font-weight:700;'>{gex_state.distance_to_flip_pct:+.2f}%</span></div>
              <div style='margin-top:6px;'>{pbar(50+gex_state.distance_to_flip_pct*5, "var(--yellow)")}</div>
              <div class='small' style='margin-top:4px;'>Stability: {gex_state.regime_stability:.2f} · Source: {gex_state.data_source}</div>
            </div>""", unsafe_allow_html=True)
        with gx2:
            res_str = " · ".join([f"{r:.0f}" for r in gex_state.key_resistance[:3]]) or "N/A"
            sup_str = " · ".join([f"{s:.0f}" for s in gex_state.key_support[:3]]) or "N/A"
            st.markdown(f"""<div class='panel'>
              <div class='panel-title'>GEX Resistance</div>
              <div style='font-family:var(--mono);font-size:13px;color:var(--red);font-weight:700;'>{res_str}</div>
              <div class='panel-title' style='margin-top:10px;'>GEX Support</div>
              <div style='font-family:var(--mono);font-size:13px;color:var(--green);font-weight:700;'>{sup_str}</div>
              <div class='small' style='margin-top:8px;'>Spot: {spot:.2f}</div>
            </div>""", unsafe_allow_html=True)
        with gx3:
            total_sign = "+" if gex_state.total_gex >= 0 else ""
            tgex_c = "var(--green)" if gex_state.total_gex > 0 else "var(--red)"
            regime_c = REGIME_COLORS.get(gex_state.regime, "var(--text)")
            st.markdown(f"""<div class='panel'>
              <div class='panel-title'>Net GEX</div>
              <div style='font-size:20px;font-weight:700;font-family:var(--mono);color:{tgex_c};'>{total_sign}{gex_state.total_gex/1e6:.1f}M</div>
              <div style='margin-top:8px;'>{regime_chip(gex_state.regime)}</div>
              <div class='small' style='margin-top:8px;'>
                {'Positive gamma: dealer hedging STABILIZES moves. Mean-reversion favored.' if gex_state.total_gex>0 else 'Negative gamma: dealer hedging AMPLIFIES moves. Trend continuation risk.'}
              </div>
            </div>""", unsafe_allow_html=True)

        # GEX by strike chart
        if gex_state.gex_by_strike:
            strikes = sorted(gex_state.gex_by_strike.keys())
            near = [s for s in strikes if spot * 0.93 < s < spot * 1.07]
            if near:
                vals = [gex_state.gex_by_strike[s] for s in near]
                colors_gex = ["#10b981" if v > 0 else "#ef4444" for v in vals]
                fig_gex = go.Figure(go.Bar(x=near, y=vals, marker_color=colors_gex, opacity=0.80,
                                           name="Net GEX"))
                if gex_state.gamma_flip:
                    fig_gex.add_vline(x=gex_state.gamma_flip, line_dash="dash",
                                      line_color="#f59e0b", annotation_text="FLIP", annotation_font_size=10)
                fig_gex.add_vline(x=spot, line_dash="dot", line_color="rgba(255,255,255,0.5)",
                                  annotation_text="SPOT", annotation_font_size=10)
                fig_gex = plotly_dark(fig_gex, "GEX by Strike (near-the-money)", 240)
                st.plotly_chart(fig_gex, use_container_width=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── OVERALL + SCORES ──
        st.markdown(f"{sec_hdr('OVERALL INDICATOR')}", unsafe_allow_html=True)
        left_ov, right_ov = st.columns([1.1, 1.4])
        with left_ov:
            st.markdown(f"""<div class='panel'>
              <div class='panel-title'>Composite Read</div>
              <div style='font-size:24px;font-weight:700;'>{colored(o_label, o_tone)}</div>
              <div class='small' style='margin-top:6px;'>Macro: <b>{macro_regime}</b></div>
              <div style='margin-top:8px;'><div class='small'>Geo Shock</div>{pbar(geo_shock,"var(--orange)")}</div>
              <div style='margin-top:6px;'><div class='small'>Regime Change P(20d)</div>{pbar(p_ch,"var(--purple)")}</div>
            </div>""", unsafe_allow_html=True)
            s1,s2,s3,s4 = st.columns(4)
            s1.metric("Fear",f"{fear_score:.0f}"); s2.metric("3-Puts",f"{three_puts:.0f}")
            s3.metric("Bubble",bubble_label); s4.metric("Stealth QE",stealth_label)
        with right_ov:
            st.markdown("<div class='panel'><div class='panel-title'>Score Drivers</div></div>", unsafe_allow_html=True)
            for item in [
                f"GEX: {gex_state.regime.value} · flip dist: {gex_state.distance_to_flip_pct:+.2f}%",
                f"Bubble: {bubble_label} · Mag7 PE: {'N/A' if not np.isfinite(mag7_pe) else f'{mag7_pe:.1f}x'}",
                f"Stealth QE: {stealth_label} · TGA 4W: {tga_4w:+.0f}B",
                f"5D: {prob['prob_5d']:.0f}% | 21D: {prob['prob_21d']:.0f}% | 63D: {prob['prob_63d']:.0f}% {'⚡ DIVERGENCE' if prob['divergent'] else ''}",
                f"Fwd {prob['fwd_prob']:.0f}% vs Coincident {prob['coincident_prob']:.0f}%",
            ]:
                st.markdown(f"- {item}")
            st.markdown("<div class='small' style='margin-top:6px;color:var(--dim);letter-spacing:0.5px;'>NARRATIVE CONTEXT (not in probability model):</div>", unsafe_allow_html=True)
            for item in [
                f"Fear: {'HIGH' if fear_score>=70 else 'elevated' if fear_score>=55 else 'contained'} ({fear_score:.0f})",
                f"Three Puts: {'strong' if three_puts>=65 else 'mixed' if three_puts>=45 else 'weak'} ({three_puts:.0f})",
                f"Liq Anxiety: {liq_anxiety:.0f} · Exhaustion: {exhaustion:.0f}",
            ]:
                st.markdown(f"<div class='small' style='color:var(--dim);'>· {item}</div>", unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ── KEY SCORES GAUGES ──
        st.markdown(f"{sec_hdr('KEY SCORES')}", unsafe_allow_html=True)
        g1,g2,g3,g4,g5 = st.columns(5)
        g1.plotly_chart(gauge(fear_score,"Fear"), use_container_width=True)
        g2.plotly_chart(gauge((market_index_score+100)/2,"Market Index"), use_container_width=True)
        g3.plotly_chart(gauge(three_puts,"Three Puts"), use_container_width=True)
        g4.plotly_chart(gauge(liq_anxiety,"Liq Anxiety"), use_container_width=True)
        g5.plotly_chart(gauge(exhaustion,"Exhaustion"), use_container_width=True)

        # ── MARKET STRUCTURE ──
        st.markdown(f"{sec_hdr('MARKET STRUCTURE')}", unsafe_allow_html=True)
        ml1, ml2 = st.columns([1.35,1.0]); mr1, mr2 = st.columns([1.35,1.0])

        yc = pd.DataFrame({"Tenor":["3M","2Y","10Y","30Y"],
                           "Yield":[float(y3m.iloc[-1]),float(y2.iloc[-1]),float(y10.iloc[-1]),float(y30.iloc[-1])]})
        fig_yc = go.Figure(go.Scatter(x=yc.Tenor, y=yc.Yield, mode="lines+markers+text",
                                      text=[f"{v:.2f}" for v in yc.Yield], textposition="top center",
                                      line=dict(color="#3b82f6",width=2), marker=dict(size=6)))
        ml1.plotly_chart(plotly_dark(fig_yc,"Rate Curve Snapshot",270), use_container_width=True)

        mat = pd.DataFrame([[1,2],[3,4]],index=["Inflation ↓","Inflation ↑"],columns=["Growth ↓","Growth ↑"])
        fig_mat = px.imshow(mat, text_auto=False, aspect="auto",
                            color_continuous_scale=[[0,"#10b981"],[0.33,"#f59e0b"],[0.66,"#f97316"],[1,"#ef4444"]])
        fig_mat.add_trace(go.Scatter(x=[0 if float(growth_z.iloc[-1])<0 else 1],
                                     y=[0 if float(infl_z.iloc[-1])<0 else 1],
                                     mode="markers", marker=dict(size=18,symbol="x",color="white")))
        fig_mat.update_coloraxes(showscale=False)
        ml2.plotly_chart(plotly_dark(fig_mat,"Regime Map",270), use_container_width=True)

        fig_sp = go.Figure(go.Scatter(x=s_2s10s.index, y=s_2s10s.values, mode="lines",
                                      line=dict(color="#8b5cf6",width=1.5)))
        fig_sp.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.25)")
        mr1.plotly_chart(plotly_dark(fig_sp,"2s10s Spread (bp)",270), use_container_width=True)

        fig_lq = go.Figure(go.Scatter(x=net_liq.index, y=net_liq.values, mode="lines",
                                      line=dict(color="#06b6d4",width=1.5)))
        mr2.plotly_chart(plotly_dark(fig_lq,"Net Liquidity Proxy ($T)",270), use_container_width=True)

        # Ticker tile
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(f"{sec_hdr(f'TICKER: {ticker_tile}')}", unsafe_allow_html=True)
        px_t = yf_close(ticker_tile, start, end, idx)
        r5  = float(px_t.pct_change(5).iloc[-1]) if px_t.dropna().size>6 else np.nan
        r21 = float(px_t.pct_change(21).iloc[-1]) if px_t.dropna().size>22 else np.nan
        t1,t2,t3,t4 = st.columns(4)
        t1.metric("Symbol",ticker_tile); t2.metric("5D Return",f"{r5*100:.2f}%" if np.isfinite(r5) else "N/A")
        t3.metric("21D Return",f"{r21*100:.2f}%" if np.isfinite(r21) else "N/A")
        t4.metric("2s10s (bp)",f"{float(s_2s10s.iloc[-1]):.0f}")
        tc = "#10b981" if (np.isfinite(r21) and r21>0) else "#ef4444"
        # Convert hex colour to rgba for fill — Plotly rejects 8-digit hex
        def _hex_to_rgba(hex_col: str, alpha: float) -> str:
            h = hex_col.lstrip("#")
            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            return f"rgba({r},{g},{b},{alpha})"
        fig_px = go.Figure(go.Scatter(x=px_t.index, y=px_t.values, mode="lines",
                                      line=dict(color=tc, width=1.5),
                                      fill="tozeroy",
                                      fillcolor=_hex_to_rgba(tc, 0.10)))
        st.plotly_chart(plotly_dark(fig_px,f"{ticker_tile} Price",240), use_container_width=True)
        st.caption("Educational use only. Not investment advice.")

    # ── RIGHT PANEL ──
    with right_col:
        render_world_intelligence_monitor(
            categorised_intel=categorised_intel,
            alerts=alerts,
            setups=setups,
            failure_modes=failure_modes,
            prob=prob,
            geo_shock=geo_shock,
            geo_triggers=geo_triggers,
            live_enabled=live_enabled,
            refresh_sec=refresh_sec,
            session=session,
        )


# ============================================================
# WORLD INTELLIGENCE MONITOR — render function
# ============================================================
