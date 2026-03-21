# page_thesis.py — Daily Thesis Briefing
# Structured morning brief: macro regime, thesis verdict, key levels, risk factors, news sentiment
import math, datetime as dt
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from config import GammaState, GammaRegime, CSS
from utils import _to_1d, zscore, resample_ffill, current_pct_rank
from ui_components import plotly_dark, sec_hdr
from data_loaders import load_macro, get_gex_from_yfinance, get_fwd_pe
from gex_engine import build_gamma_state, compute_gwas, compute_gex_term_structure
from schwab_api import get_schwab_client, schwab_get_spot, schwab_get_options_chain
from signals import compute_leading_stack, compute_1d_prob
from probability import (compute_prob_composite, get_session_context,
                         classify_macro_regime_abs, regime_transition_prob)
from intel_monitor import (load_feeds, geo_shock_score, categorise_items,
                            category_shock_score, _all_feeds_flat, INTEL_CATEGORIES)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_last(s: pd.Series, default=float("nan")) -> float:
    try:
        v = s.dropna()
        return float(v.iloc[-1]) if len(v) else default
    except Exception:
        return default

def _score_to_int(score: float, neutral=50.0) -> int:
    """Convert 0-100 probability score to signed integer ±10 composite point."""
    return int(round((score - neutral) / 5))

def _vrp(vix_level: float, spy_series: pd.Series, idx: pd.DatetimeIndex) -> float:
    """Variance Risk Premium = VIX - 21D realized vol (annualised %)."""
    spy_a   = _to_1d(spy_series).reindex(idx).ffill()
    rvol_21 = spy_a.pct_change().rolling(21, min_periods=10).std().iloc[-1]
    if rvol_21 and np.isfinite(rvol_21):
        return round(vix_level - rvol_21 * np.sqrt(252) * 100, 4)
    return float("nan")

def _cpi_nowcast(cpi_series: pd.Series) -> float:
    """Approximate MoM CPI nowcast from last monthly change."""
    try:
        mom = cpi_series.pct_change(21).dropna()
        return round(float(mom.iloc[-1]) * 100, 3)
    except Exception:
        return float("nan")

def _recession_prob(sahm_val: float, three_puts: float, hy_spread_val: float) -> float:
    """Composite recession probability 0–100 from Sahm, HY spread and backstop score."""
    # Sahm Rule: 0=clear, 0.3=warning, 0.5=triggered
    sahm_component = float(np.clip(sahm_val / 0.5 * 60, 0, 60))
    # HY spread: <300=fine, 400=caution, 600=stress, 1000=systemic
    hy_component   = float(np.clip((hy_spread_val - 300) / 700 * 25, 0, 25))
    # Three puts: inverse — low backstop = higher recession risk
    puts_component = float(np.clip((100 - three_puts) / 100 * 15, 0, 15))
    return round(min(sahm_component + hy_component + puts_component, 99), 1)

def _composite_score(prob_dict: dict, fear_score: float, vrp_val: float) -> int:
    """
    Single signed integer ±10 thesis score.
    Positive = bullish, negative = bearish.
    Mirrors the composite score in the reference example.
    """
    bull = prob_dict.get("bull_prob", 50.0)
    score = 0
    # Probability component: ±4 max
    score += int(round((bull - 50) / 12.5))
    # Fear: elevated fear = bearish (-2 max)
    if fear_score > 70:   score -= 2
    elif fear_score > 55: score -= 1
    elif fear_score < 35: score += 1
    # VRP: positive = options overpriced = mild bearish for direction
    if np.isfinite(vrp_val):
        if vrp_val > 3:   score -= 1   # fear premium elevated
        elif vrp_val < -2: score += 1  # realized > implied = complacency or trending
    # GEX component handled at display time
    return int(np.clip(score, -10, 10))

def _thesis_verdict(composite: int, gex_regime: GammaRegime) -> Tuple[str, str, str]:
    """Return (verdict_label, color, explanation)."""
    gex_neg = gex_regime in (GammaRegime.NEGATIVE, GammaRegime.STRONG_NEGATIVE)
    gex_pos = gex_regime in (GammaRegime.POSITIVE, GammaRegime.STRONG_POSITIVE)

    if composite <= -4:
        return "BEARISH", "#ef4444", "Multiple signals aligned bearish. Negative macro + fear elevated."
    elif composite <= -2:
        if gex_neg:
            return "BEARISH", "#ef4444", "Bearish composite confirmed by negative gamma regime — moves amplify."
        return "LEANING BEARISH", "#f97316", "Modest bearish lean. Watch GEX flip for confirmation."
    elif composite <= 1:
        if gex_neg:
            return "CAUTIOUS", "#f59e0b", "Neutral signals but negative gamma — tail risk elevated."
        elif gex_pos:
            return "NEUTRAL / RANGE", "#6366f1", "Neutral macro, positive gamma — expect compression and pin."
        return "NEUTRAL", "#94a3b8", "No strong directional conviction. Reduce size, wait for setup."
    elif composite <= 3:
        if gex_pos:
            return "LEANING BULLISH", "#10b981", "Bullish lean supported by positive gamma — dealers absorb dips."
        return "LEANING BULLISH", "#10b981", "Modest bullish lean. Credit and liquidity constructive."
    else:
        return "BULLISH", "#10b981", "Broad bullish alignment. High conviction — size up on setups."

def _sigma_bands(spot: float, vix_level: float) -> Dict:
    """Compute 1σ / 2σ daily and weekly bands around spot."""
    daily_vol  = vix_level / 100 / np.sqrt(252)
    weekly_vol = vix_level / 100 / np.sqrt(52)
    return {
        "daily_1s_lo":  round(spot * (1 - daily_vol),  2),
        "daily_1s_hi":  round(spot * (1 + daily_vol),  2),
        "daily_2s_lo":  round(spot * (1 - 2*daily_vol), 2),
        "daily_2s_hi":  round(spot * (1 + 2*daily_vol), 2),
        "weekly_1s_lo": round(spot * (1 - weekly_vol),  2),
        "weekly_1s_hi": round(spot * (1 + weekly_vol),  2),
        "weekly_2s_lo": round(spot * (1 - 2*weekly_vol), 2),
        "weekly_2s_hi": round(spot * (1 + 2*weekly_vol), 2),
    }

def _news_sentiment_by_category(categorised_intel: dict) -> List[Dict]:
    """
    Compute per-category sentiment score and article count.
    Returns list of dicts sorted by absolute score descending.
    """
    out = []
    for cat_key, items in categorised_intel.items():
        if not items:
            continue
        meta = INTEL_CATEGORIES.get(cat_key, {})
        label = meta.get("label", cat_key)
        shock = category_shock_score(items)
        # Normalise shock (0-100) to sentiment (-1 to +1): high shock = negative
        # Shock>60 = fear/risk, <30 = quiet (mildly positive)
        sentiment = round(-(shock - 30) / 70, 4)
        out.append({
            "category": cat_key,
            "label":    label,
            "icon":     meta.get("icon", "📰"),
            "sentiment": sentiment,
            "shock":    shock,
            "count":    len(items),
        })
    return sorted(out, key=lambda x: abs(x["sentiment"]), reverse=True)


# ── CSS card helpers ──────────────────────────────────────────────────────────

_CARD = """
<div style='background:{bg};border:1px solid {border};border-radius:12px;padding:16px 20px;margin-bottom:12px;'>
  {body}
</div>
"""

def _card(body: str, bg="rgba(255,255,255,0.03)", border="rgba(255,255,255,0.10)") -> str:
    return _CARD.format(bg=bg, border=border, body=body)

def _section_header(n: int, title: str) -> str:
    return f"<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>{n}. {title}</div>"

def _kv(label: str, value: str, color: str = "rgba(255,255,255,0.85)") -> str:
    return (
        f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
        f"margin-bottom:3px;font-size:13px;'>"
        f"<span style='color:rgba(255,255,255,0.55);'>{label}</span>"
        f"<span style='font-family:monospace;font-weight:600;color:{color};'>{value}</span>"
        f"</div>"
    )

def _pill(text: str, color: str = "#6366f1") -> str:
    return (
        f"<span style='background:{color}22;color:{color};border:1px solid {color}44;"
        f"border-radius:6px;padding:2px 8px;font-size:11px;font-weight:600;"
        f"margin-right:4px;'>{text}</span>"
    )

def _signal_row(emoji: str, text: str) -> str:
    return f"<div style='font-size:12px;margin-bottom:3px;'>{emoji} {text}</div>"


# ── Main render ───────────────────────────────────────────────────────────────

def render_thesis_page():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## 📋 Daily Thesis Briefing")
    st.markdown(
        f"*Generated {dt.datetime.now().strftime('%A, %B %d, %Y — %H:%M ET')}* — "
        "Not financial advice."
    )

    # ── Sidebar controls ──────────────────────────────────────────────────
    st.sidebar.markdown("### Thesis Controls")
    start = st.sidebar.date_input("Data Start",
                                   value=dt.date.today() - dt.timedelta(days=730),
                                   key="thesis_start")
    end   = st.sidebar.date_input("Data End",
                                   value=dt.date.today(),
                                   key="thesis_end")
    gex_symbol = st.sidebar.text_input("GEX Symbol", "SPY", key="thesis_gex_sym").upper().strip()
    show_es_nq = st.sidebar.toggle("Show ES/NQ levels", True, key="thesis_esnq")

    if st.sidebar.button("🔄 Refresh Thesis", use_container_width=True, key="thesis_refresh"):
        st.cache_data.clear()
        st.rerun()

    idx = pd.date_range(start, end, freq="D")

    # ── Load all data (shared with dashboard — cached) ────────────────────
    with st.spinner("Loading macro data..."):
        raw = load_macro(start.isoformat(), end.isoformat())
        def r(k): return resample_ffill(raw.get(k, pd.Series(dtype=float)), idx)

        y3m=r("DGS3MO"); y2=r("DGS2"); y10=r("DGS10"); y30=r("DGS30")
        cpi=r("CPIAUCSL"); core=r("CPILFESL"); unrate=r("UNRATE")
        walcl=r("WALCL"); tga=r("WTREGEN"); rrp=r("RRPONTSYD"); m2=r("M2SL")
        nfci=resample_ffill(raw.get("NFCI", pd.Series(dtype=float)), idx).fillna(0)
        vix_s=r("VIX"); spy=r("SPY"); tlt=r("TLT"); qqq=r("QQQ")
        copx=r("COPX"); gld=r("GLD"); hyg=r("HYG"); lqd=r("LQD")
        dxy=r("UUP"); iwm=r("IWM")
        tips_10y     = r("DFII10")
        bank_reserves= r("WRBWFRBL")
        bank_credit  = r("TOTBKCR")
        ism_no_raw   = raw.get("AMTMNO", pd.Series(dtype=float))
        ism_no       = ism_no_raw if len(ism_no_raw.dropna()) > 4 else None
        gdp_quarterly= r("GDPC1")
        mmmf         = r("WRMFSL")
        _sahm_raw    = raw.get("SAHM_RULE", pd.Series(dtype=float))
        _hy_raw      = raw.get("BAMLH0A0HYM2", pd.Series(dtype=float))
        sahm_rule    = resample_ffill(_sahm_raw, idx) if len(_sahm_raw.dropna()) > 0 else None
        hy_spread    = resample_ffill(_hy_raw, idx)   if len(_hy_raw.dropna()) > 0 else None

    # ── Derived macro ─────────────────────────────────────────────────────
    core_yoy        = (core / core.shift(365) - 1) * 100
    cpi_yoy         = (cpi  / cpi.shift(365)  - 1) * 100
    s_2s10s         = (y10 - y2) * 100
    net_liq         = (walcl - tga - rrp) / 1000.0
    net_liq_4w      = net_liq.diff(28)
    bs_13w          = walcl.diff(91) / 1000.0
    core_yoy_latest = _safe_last(core_yoy, 2.5)
    curve_raw_latest= _safe_last(s_2s10s, 0.0)
    macro_regime    = classify_macro_regime_abs(core_yoy_latest, curve_raw_latest)

    growth_z  = zscore(s_2s10s.fillna(0))
    infl_z    = zscore(core_yoy.fillna(core_yoy_latest))
    vix_z     = zscore(vix_s.fillna(20.0))
    nfci_z    = zscore(nfci.fillna(0))
    inv       = (s_2s10s < 0).astype(int)
    liq_tight = (net_liq_4w < 0).astype(int)
    fear_raw  = 0.45*vix_z + 0.35*nfci_z + 0.10*inv + 0.10*liq_tight
    fear_score= float(((fear_raw.iloc[-1] + 2) / 4).clip(0, 1) * 100)

    vix_level = _safe_last(vix_s, 20.0)
    vrp_val   = _vrp(vix_level, spy, idx)
    unrate_v  = _safe_last(unrate, 4.0)
    sahm_v    = _safe_last(sahm_rule, 0.0) if sahm_rule is not None else 0.0
    hy_v      = _safe_last(hy_spread, 300.0) if hy_spread is not None else 300.0
    cpi_yoy_v = _safe_last(cpi_yoy, 2.5)
    cpi_now   = _cpi_nowcast(cpi)

    # Recession proxy (no Sahm component alone — blend three signals)
    y10_20   = y10.diff(20)
    unemp_3m = float(unrate.diff(90).iloc[-1]) if len(unrate) > 90 else 0.0
    warsh    = ((y10.diff(20) < 0) & (bs_13w < 0)).astype(int)
    trump_put= float(np.clip(45 + 35*float((spy/spy.rolling(126).max()-1).iloc[-1] <= -0.07)
                             + 20*(fear_score > 60), 0, 100))
    fed_put  = float(np.clip(55 + 25*float((y10_20.iloc[-1] < 0) and (unemp_3m > 0))
                             - 10*float((core_yoy_latest - 3.0) > 0)
                             - 15*float(warsh.iloc[-1]), 0, 100))
    tga_dd   = (tga.diff(28) < 0).astype(int)
    rrp_dep  = (rrp < 50).astype(int)
    treas_put= float(np.clip(50 + 20*float(tga_dd.iloc[-1])
                             + 15*float(rrp_dep.iloc[-1])
                             + 15*float(net_liq_4w.iloc[-1] >= 0), 0, 100))
    three_puts= float(np.clip(0.35*treas_put + 0.35*fed_put + 0.30*trump_put, 0, 100))
    rec_prob  = _recession_prob(sahm_v, three_puts, hy_v)

    # SPY-TLT correlation
    spy_ret   = _to_1d(spy).reindex(idx).ffill().pct_change().dropna()
    tlt_ret   = _to_1d(tlt).reindex(idx).ffill().pct_change().reindex(spy_ret.index).dropna()
    spy_tlt_corr = round(float(spy_ret.rolling(21).corr(tlt_ret).dropna().iloc[-1]), 3) \
                   if spy_ret.dropna().size > 21 else float("nan")

    # ── Leading stack + probability ───────────────────────────────────────
    with st.spinner("Computing signals..."):
        leading = compute_leading_stack(
            y2, y3m, y10, y30, s_2s10s, vix_s, m2,
            pd.Series(dtype=float),   # claims placeholder
            copx, gld, hyg, lqd, dxy, spy, qqq, iwm,
            net_liq, net_liq_4w, walcl, bs_13w, idx,
            tips_10y=tips_10y, bank_reserves=bank_reserves,
            bank_credit=bank_credit, ism_no=ism_no,
            gdp_quarterly=gdp_quarterly, mmmf=mmmf,
        )
        meta  = regime_transition_prob(macro_regime, core_yoy, s_2s10s)
        nfci_coin = float(current_pct_rank(-_to_1d(nfci).reindex(idx).ffill(), 252))
        liq_dir   = float(net_liq_4w.dropna().iloc[-1]) if net_liq_4w.dropna().size else 0.0
        liq_coin  = float(50.0 + np.sign(liq_dir) * 20)

    # ── GEX ──────────────────────────────────────────────────────────────
    with st.spinner("Fetching GEX data..."):
        client = get_schwab_client()
        chain_df = spot = gex_source = None
        if client:
            chain_df = schwab_get_options_chain(client, gex_symbol, spot=None)
            if chain_df is not None and len(chain_df) > 0:
                spot      = schwab_get_spot(client, gex_symbol) or float(chain_df["strike"].median())
                gex_source = "Schwab (live)"
        if chain_df is None or len(chain_df) == 0:
            chain_df, spot, gex_source = get_gex_from_yfinance(gex_symbol)

        if chain_df is not None and spot:
            gex_state = build_gamma_state(chain_df, spot, gex_source or "unknown", max_dte=45)
            gwas      = compute_gwas(chain_df, spot)
            term_str  = compute_gex_term_structure(chain_df, spot)
        else:
            gex_state = GammaState(data_source="unavailable",
                                   timestamp=dt.datetime.now().strftime("%H:%M:%S"))
            gwas, term_str = {}, {}
            spot = float(_to_1d(spy).dropna().iloc[-1]) * 10 if _to_1d(spy).dropna().size else 5500.0

    # ── Prob composite ────────────────────────────────────────────────────
    with st.spinner("Building probability model..."):
        # Quick geo shock from feeds
        try:
            all_rss = load_feeds(tuple(_all_feeds_flat().items()), 60)
            geo_shock, _ = geo_shock_score(all_rss)
            categorised_intel = categorise_items(all_rss)
        except Exception:
            geo_shock = 0.0
            categorised_intel = {k: [] for k in INTEL_CATEGORIES}

        prob = compute_prob_composite(
            leading, fear_score, geo_shock, meta["p_change_20d"], gex_state,
            nfci_coincident=nfci_coin, liq_dir_coincident=liq_coin,
        )

        prob_1d = compute_1d_prob(
            gex_state=gex_state, spot=spot, vix_level=vix_level,
            vix_series=vix_s, spy_series=spy, hyg_series=hyg,
            lqd_series=lqd, dxy_series=dxy, s_2s10s=s_2s10s,
            net_liq_4w=net_liq_4w, nfci_z=nfci_z, fear_score=fear_score,
            session=get_session_context(), idx=idx,
            sahm_rule=sahm_rule, hy_spread=hy_spread,
        )

    # ── Composite score + verdict ─────────────────────────────────────────
    composite   = _composite_score(prob, fear_score, vrp_val)
    # GEX adjustment: negative gamma adds -1 to composite, strong negative -2
    if gex_state.regime == GammaRegime.STRONG_NEGATIVE: composite = max(composite - 2, -10)
    elif gex_state.regime == GammaRegime.NEGATIVE:      composite = max(composite - 1, -10)
    elif gex_state.regime == GammaRegime.STRONG_POSITIVE: composite = min(composite + 1, 10)

    verdict, verdict_color, verdict_expl = _thesis_verdict(composite, gex_state.regime)

    # ── Sigma bands ───────────────────────────────────────────────────────
    bands = _sigma_bands(spot, vix_level)

    # ── Key GEX levels ────────────────────────────────────────────────────
    flip  = gex_state.gamma_flip if gex_state.gamma_flip else spot
    upper = max(gex_state.key_resistance[:1] + [spot * 1.03], default=spot * 1.03)
    upper = upper[0] if isinstance(upper, list) else upper
    lower = min(gex_state.key_support[:1] + [spot * 0.97], default=spot * 0.97)
    lower = lower[0] if isinstance(lower, list) else lower

    # ── ES / NQ levels ────────────────────────────────────────────────────
    multiplier = 40.0 if gex_symbol in ("QQQ", "NDX") else 10.0
    ua_label   = "NQ" if multiplier == 40 else "ES"

    def _to_ua(spy_price: float) -> str:
        return f"{spy_price * multiplier:,.0f}"

    # ── News sentiment ────────────────────────────────────────────────────
    news_cats = _news_sentiment_by_category(categorised_intel)
    top_cats  = news_cats[:5]  # top 5 most active

    # ────────────────────────────────────────────────────────────────────
    # RENDER
    # ────────────────────────────────────────────────────────────────────

    # ── SECTION 1: Macro Regime ───────────────────────────────────────────
    regime_color_map = {
        "Goldilocks":   "#10b981",
        "Overheating":  "#f59e0b",
        "Stagflation":  "#ef4444",
        "Deflation":    "#6366f1",
        "Neutral":      "#94a3b8",
    }
    reg_color = regime_color_map.get(macro_regime, "#94a3b8")

    growth_z_v = float(growth_z.dropna().iloc[-1]) if growth_z.dropna().size else 0.0
    infl_z_v   = float(infl_z.dropna().iloc[-1])   if infl_z.dropna().size   else 0.0

    macro_body = f"""
{_section_header(1, "MACRO REGIME & NEWS SENTIMENT")}
<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>
  <div style='font-size:18px;font-weight:700;color:{reg_color};'>{macro_regime}</div>
  {_pill("VIX " + f"{vix_level:.1f}", "#6366f1")}
  {_pill("Fear " + f"{fear_score:.0f}", "#ef4444" if fear_score > 60 else "#f59e0b" if fear_score > 40 else "#10b981")}
</div>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:0 24px;'>
  <div>
    {_kv("Growth Z",    f"{growth_z_v:+.2f}", "#10b981" if growth_z_v > 0 else "#ef4444")}
    {_kv("Inflation Z", f"{infl_z_v:+.2f}", "#f59e0b" if infl_z_v > 0.5 else "#94a3b8")}
    {_kv("CPI YoY",     f"{cpi_yoy_v:.2f}%")}
    {_kv("CPI Nowcast", f"{cpi_now:+.3f}% MoM")}
  </div>
  <div>
    {_kv("Unemployment", f"{unrate_v:.1f}%")}
    {_kv("HY OAS",       f"{hy_v:.2f}")}
    {_kv("SPY-TLT Corr", f"{spy_tlt_corr:.3f}" if np.isfinite(spy_tlt_corr) else "N/A",
         "#ef4444" if (np.isfinite(spy_tlt_corr) and spy_tlt_corr > 0.2) else "#94a3b8")}
    {_kv("Sahm Rule",    f"{sahm_v:.3f}",
         "#ef4444" if sahm_v >= 0.5 else "#f59e0b" if sahm_v >= 0.3 else "#10b981")}
  </div>
</div>
"""
    # News sentiment inline
    if top_cats:
        macro_body += "<div style='margin-top:10px;border-top:1px solid rgba(255,255,255,0.08);padding-top:8px;'>"
        macro_body += "<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:5px;letter-spacing:0.1em;'>NEWS SENTIMENT</div>"
        macro_body += "<div style='display:flex;flex-wrap:wrap;gap:6px;'>"
        for cat in top_cats:
            sent = cat["sentiment"]
            col  = "#10b981" if sent > 0.01 else "#ef4444" if sent < -0.01 else "#94a3b8"
            sign = "+" if sent > 0 else ""
            macro_body += (
                f"<span style='font-size:11px;font-family:monospace;'>"
                f"<span style='color:rgba(255,255,255,0.55);'>{cat['icon']} {cat['label'].split('&')[0].strip()}</span>: "
                f"<span style='color:{col};font-weight:600;'>{sign}{sent:.4f}</span> "
                f"<span style='color:rgba(255,255,255,0.3);'>({cat['count']} articles)</span>"
                f"</span>"
            )
        macro_body += "</div></div>"

    st.markdown(_card(macro_body), unsafe_allow_html=True)

    # ── SECTION 2: Thesis Verdict ─────────────────────────────────────────
    comp_color = "#10b981" if composite > 0 else "#ef4444" if composite < 0 else "#94a3b8"
    prob_5d  = prob.get("prob_5d",  50.0)
    prob_21d = prob.get("prob_21d", 50.0)
    prob_63d = prob.get("prob_63d", 50.0)
    bull_p   = prob.get("bull_prob", 50.0)

    signals_html = ""
    vrp_str = f"{vrp_val:+.4f}" if np.isfinite(vrp_val) else "N/A"
    vrp_pos = np.isfinite(vrp_val) and vrp_val > 0
    signals_html += _signal_row(
        "✅" if vrp_pos else "⚠️",
        f"VRP {'positive' if vrp_pos else 'negative'} ({vrp_str})"
    )
    fear_lbl = "ELEVATED" if fear_score > 60 else "MODERATE" if fear_score > 40 else "LOW"
    signals_html += _signal_row(
        "⚠️" if fear_score > 55 else "✅",
        f"Fear composite {fear_lbl}"
    )
    signals_html += _signal_row(
        "🔴" if rec_prob > 60 else "🟡" if rec_prob > 35 else "🟢",
        f"Recession risk {'elevated' if rec_prob > 60 else 'moderate' if rec_prob > 35 else 'low'} ({rec_prob:.1f}%)"
    )
    # GEX regime signal
    gex_regime_str = gex_state.regime.value if hasattr(gex_state.regime, "value") else str(gex_state.regime)
    signals_html += _signal_row(
        "🟢" if "positive" in gex_regime_str.lower() else "🔴" if "negative" in gex_regime_str.lower() else "⚪",
        f"GEX regime: {gex_regime_str} ({gex_state.distance_to_flip_pct:+.1f}% from flip)"
    )
    # Dominant signal from 1D model
    dom_sig  = prob_1d.get("dominant_signal", "—")
    dom_dir  = prob_1d.get("dominant_dir", "neutral")
    signals_html += _signal_row(
        "📊",
        f"Dominant 1D signal: {dom_sig} ({dom_dir})"
    )

    verdict_body = f"""
{_section_header(2, "THESIS VERDICT: " + verdict)}
<div style='display:flex;align-items:baseline;gap:16px;margin-bottom:10px;'>
  <div style='font-size:28px;font-weight:800;color:{verdict_color};letter-spacing:-0.5px;'>{verdict}</div>
  <div style='font-size:13px;color:rgba(255,255,255,0.5);'>Composite Score: <span style='font-family:monospace;font-weight:700;color:{comp_color};'>{composite:+d}</span> / ±10</div>
  <div style='font-size:12px;color:rgba(255,255,255,0.4);'>Date: {dt.date.today().strftime("%A, %B %d, %Y")}</div>
</div>
<div style='font-size:12px;color:rgba(255,255,255,0.6);margin-bottom:10px;font-style:italic;'>{verdict_expl}</div>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px 24px;margin-bottom:10px;'>
  <div>
    <div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>PROBABILITIES</div>
    {_kv("Bull Composite", f"{bull_p:.0f}%", "#10b981" if bull_p > 55 else "#ef4444" if bull_p < 45 else "#94a3b8")}
    {_kv("5D (Tactical)",  f"{prob_5d:.0f}%",  "#10b981" if prob_5d  > 55 else "#ef4444" if prob_5d  < 45 else "#94a3b8")}
    {_kv("21D (Short)",    f"{prob_21d:.0f}%", "#10b981" if prob_21d > 55 else "#ef4444" if prob_21d < 45 else "#94a3b8")}
    {_kv("63D (Medium)",   f"{prob_63d:.0f}%", "#10b981" if prob_63d > 55 else "#ef4444" if prob_63d < 45 else "#94a3b8")}
  </div>
  <div>
    <div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>SIGNAL BREAKDOWN</div>
    {signals_html}
  </div>
</div>
"""
    st.markdown(_card(verdict_body, bg=f"{verdict_color}08", border=f"{verdict_color}30"),
                unsafe_allow_html=True)

    # ── SECTION 3: Key Levels ─────────────────────────────────────────────
    gwas_above = gwas.get("gwas_above")
    gwas_below = gwas.get("gwas_below")
    dur_label  = term_str.get("durability", "unknown").upper()
    frag_pct   = term_str.get("fragility_ratio", 0.5) * 100

    def _level_row(label, spy_val, es_nq_val=None, color="#94a3b8"):
        val_str = f"`{spy_val:.2f}`"
        if show_es_nq and es_nq_val is not None:
            val_str += f"  →  **{ua_label} {es_nq_val:,.0f}**"
        return f"**{label}:** {val_str}  \n"

    # Build ES/NQ block first so it can be injected cleanly into the grid
    dur_color = "#ef4444" if dur_label == "FRAGILE" else "#10b981" if dur_label == "DURABLE" else "#94a3b8"
    gwas_above_str = f"{gwas_above:.2f}" if gwas_above else "N/A"
    gwas_below_str = f"{gwas_below:.2f}" if gwas_below else "N/A"

    esnq_block = ""
    if show_es_nq:
        esnq_block = (
            "<div style='margin-top:8px;border-top:1px solid rgba(255,255,255,0.08);padding-top:6px;'>"
            + f"<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>{ua_label} EQUIVALENT</div>"
            + _kv(f"{ua_label} Spot",  _to_ua(spot))
            + _kv(f"{ua_label} Flip",  _to_ua(flip),  "#f59e0b")
            + _kv(f"{ua_label} Upper", _to_ua(upper), "#10b981")
            + _kv(f"{ua_label} Lower", _to_ua(lower), "#ef4444")
            + _kv(f"{ua_label} GWAS↑", _to_ua(gwas_above) if gwas_above else "N/A", "#6366f1")
            + _kv(f"{ua_label} GWAS↓", _to_ua(gwas_below) if gwas_below else "N/A", "#6366f1")
            + "</div>"
        )

    levels_body = (
        _section_header(3, "KEY LEVELS")
        + "<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px 32px;'>"
        # Left column — GEX levels
        + "<div>"
        + f"<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>GEX LEVELS ({gex_symbol})</div>"
        + _kv(f"{gex_symbol} Spot", f"{spot:.2f}", "#fff")
        + _kv("GEX Flip",           f"{flip:.2f}",  "#f59e0b")
        + _kv("GEX Upper",          f"{upper:.2f}", "#10b981")
        + _kv("GEX Lower",          f"{lower:.2f}", "#ef4444")
        + _kv("GWAS Above",         gwas_above_str, "#6366f1")
        + _kv("GWAS Below",         gwas_below_str, "#6366f1")
        + _kv("GEX Duration",       f"{dur_label} ({frag_pct:.0f}% weekly)", dur_color)
        + "</div>"
        # Right column — sigma bands + optional ES/NQ
        + "<div>"
        + f"<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>σ BANDS (VIX={vix_level:.1f})</div>"
        + _kv("1σ Daily",  f"{bands['daily_1s_lo']:.2f} — {bands['daily_1s_hi']:.2f}")
        + _kv("2σ Daily",  f"{bands['daily_2s_lo']:.2f} — {bands['daily_2s_hi']:.2f}")
        + _kv("1σ Weekly", f"{bands['weekly_1s_lo']:.2f} — {bands['weekly_1s_hi']:.2f}")
        + _kv("2σ Weekly", f"{bands['weekly_2s_lo']:.2f} — {bands['weekly_2s_hi']:.2f}")
        + esnq_block
        + "</div>"
        + "</div>"
    )

    st.markdown(_card(levels_body), unsafe_allow_html=True)

    # ── SECTION 4: Risk Factors ───────────────────────────────────────────
    risk_items = []

    if fear_score > 60:
        risk_items.append(("⚠️", f"Elevated fear composite ({fear_score:.0f}/100) — potential for sharp moves"))
    if rec_prob > 50:
        risk_items.append(("🔴", f"Recession probability {rec_prob:.1f}% — monitor labor data and Sahm rule ({sahm_v:.3f})"))
    if np.isfinite(spy_tlt_corr) and spy_tlt_corr > 0.20:
        risk_items.append(("⚠️", f"Positive stock-bond correlation ({spy_tlt_corr:.3f}) — diversification impaired"))
    if gex_state.regime in (GammaRegime.NEGATIVE, GammaRegime.STRONG_NEGATIVE):
        risk_items.append(("🔴", f"Negative gamma regime — dealer hedging amplifies moves. No fading."))
    if np.isfinite(vrp_val) and vrp_val < -2:
        risk_items.append(("⚠️", f"IV below realized vol (VRP={vrp_val:.2f}) — market underpricing risk"))
    dur = term_str.get("durability")
    if dur == "fragile":
        risk_items.append(("⚠️", f"GEX regime fragile — {frag_pct:.0f}% of gamma in ≤7 DTE. Levels expire by Friday."))
    if leading.get("corr_regime") in ("STRESS", "SYSTEMIC"):
        risk_items.append(("🔴", f"Cross-asset correlation regime: {leading.get('corr_regime')} — credit leading equity lower"))
    if prob.get("divergent", False):
        risk_items.append(("📊", f"Horizon divergence: tactical {prob_5d:.0f}% vs medium {prob_63d:.0f}%. Mixed signals."))
    if geo_shock > 50:
        risk_items.append(("🌍", f"Elevated geopolitical shock score ({geo_shock:.0f}/100)"))
    sahm_trig = sahm_v >= 0.50
    if sahm_trig:
        risk_items.append(("🔴", f"Sahm Rule TRIGGERED ({sahm_v:.3f} ≥ 0.50) — recession onset signal active"))
    if hy_v > 450:
        risk_items.append(("⚠️", f"HY OAS elevated ({hy_v:.0f}bp) — credit stress building"))

    if not risk_items:
        risk_items.append(("✅", "No major risk flags active. Conditions broadly constructive."))

    risk_rows = "".join(_signal_row(e, t) for e, t in risk_items)
    risk_body = f"""
{_section_header(4, "RISK FACTORS")}
{risk_rows}
"""
    st.markdown(_card(risk_body, bg="rgba(239,68,68,0.04)", border="rgba(239,68,68,0.15)"),
                unsafe_allow_html=True)

    # ── SECTION 5: Session Context & Execution Notes ──────────────────────
    session = get_session_context()
    sess_name  = session.get("name", "Unknown")
    sess_mult  = session.get("size_mult", 0.5)
    prob_1d_v  = prob_1d.get("prob_1d", 50.0)
    kelly_1d   = prob_1d.get("kelly_1d", 0.0)
    kelly_21d  = prob.get("kelly_21d", 0.0)

    gex_interp = prob_1d.get("regime_interp", gex_regime_str)
    using_intra= prob_1d.get("using_intraday", False)

    exec_notes = []
    if sess_mult < 0.5:
        exec_notes.append(f"⏰ Off-hours session ({sess_name}) — reduce size by {int((1-sess_mult)*100)}%")
    if gex_state.regime in (GammaRegime.STRONG_POSITIVE, GammaRegime.POSITIVE):
        exec_notes.append("📍 Positive gamma: fade moves to GWAS, mean-reversion trades preferred")
    elif gex_state.regime in (GammaRegime.NEGATIVE, GammaRegime.STRONG_NEGATIVE):
        exec_notes.append("🌊 Negative gamma: follow momentum, wider stops, no early reversals")
    if abs(gex_state.distance_to_flip_pct) < 0.75:
        exec_notes.append(f"⚡ Within 0.75% of gamma flip — binary risk, reduce size 50%")
    if using_intra:
        exec_notes.append("📡 Live Schwab intraday signals active in 1D model")
    else:
        exec_notes.append("📋 1D model using prior-close data (Schwab not connected or pre-market)")

    exec_rows = "".join(f"<div style='font-size:12px;margin-bottom:3px;'>{n}</div>" for n in exec_notes)

    session_body = f"""
{_section_header(5, "SESSION & EXECUTION")}
<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px 24px;'>
  <div>
    {_kv("Session",       sess_name)}
    {_kv("Size Multiplier", f"{sess_mult:.1f}×")}
    {_kv("1D Bull Prob",  f"{prob_1d_v:.0f}%",
         "#10b981" if prob_1d_v > 55 else "#ef4444" if prob_1d_v < 45 else "#94a3b8")}
    {_kv("Kelly 1D",      f"{kelly_1d*100:.1f}%")}
    {_kv("Kelly 21D",     f"{kelly_21d*100:.1f}%")}
  </div>
  <div>
    <div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>EXECUTION NOTES</div>
    {exec_rows}
    <div style='font-size:11px;color:rgba(255,255,255,0.4);margin-top:6px;font-style:italic;'>{gex_interp}</div>
  </div>
</div>
"""
    st.markdown(_card(session_body), unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='text-align:center;font-size:10px;color:rgba(255,255,255,0.25);margin-top:16px;'>"
        f"GEX source: {gex_source or 'N/A'} · "
        f"Data as of {dt.datetime.now().strftime('%H:%M ET')} · "
        f"Not financial advice — quantitative research tool only."
        f"</div>",
        unsafe_allow_html=True,
    )
