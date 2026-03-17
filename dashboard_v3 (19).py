# dashboard_v3.py
# ============================================================
# PREDICTIVE MACRO + GEX REGIME DASHBOARD — Full Integration
# ============================================================
#
# ARCHITECTURE OVERVIEW
# ─────────────────────
# Three pillars unified into one operational dashboard:
#
#   MACRO LAYER (from dashboard_v2)
#   ├── Stochastic Probability Engine (Bayesian bull/bear %)
#   ├── 9 Leading Indicators (M2, credit impulse, Cu/Au, etc.)
#   ├── Regime Transition Model (P(change) in 20 days)
#   ├── Half-Kelly sizing framework
#   └── WorldMonitor geo-risk feeds
#
#   GEX LAYER (from AMT/GEX notes)
#   ├── Live options chain GEX via IBKR TWS/Gateway
#   ├── Gamma flip detection (zero-crossing of cumulative GEX)
#   ├── Gamma regime classification (Strong+/+/Neutral/−/Strong−)
#   ├── Key support/resistance gamma walls
#   ├── Regime stability score (distance from flip boundary)
#   ├── 5 Trade Setups: Bounce, Fade, Flip Breakout, Exhaustion, Pin
#   └── Setup scoring matrix (gamma + orderflow + TPO + freshness + event)
#
#   EXECUTION LAYER (from notes 08/09/10)
#   ├── Session structure overlay (time-of-day context)
#   ├── OpEx calendar awareness
#   ├── Failure Mode checklist (8 failure modes from note 10)
#   ├── Pre-trade checklist (all 10 items from note 07)
#   ├── Position sizing table by setup + volatility regime
#   └── Live IBKR portfolio P&L + positions
#
# IBKR INTEGRATION
# ─────────────────
# Uses schwab-py for OAuth2 REST access to Schwab/TOS options data.
# Requires TWS or IB Gateway running on localhost:7497 (paper)
# or localhost:7496 (live). Enable API in TWS settings.
# All IBKR features gracefully degrade if connection is unavailable.
#
# HOW TO RUN
# ──────────
# pip install streamlit yfinance fredapi schwab-py scipy plotly pandas numpy
# python -m streamlit run dashboard_v3.py
# ============================================================

import os, re, time, math, datetime as dt, asyncio, threading
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

# ── Schwab API + Supabase token storage ────────────────────────────────────
# schwab-py: OAuth2 + REST access to Schwab/TOS options data
# supabase:  cloud token storage so auth persists on Streamlit Community Cloud
# Install: pip install schwab-py supabase
import asyncio, json, tempfile

SCHWAB_AVAILABLE = False
try:
    import schwab
    SCHWAB_AVAILABLE = True
except ImportError:
    pass

SUPABASE_AVAILABLE = False
try:
    from supabase import create_client as _supa_create_client, Client as _SupaClient
    SUPABASE_AVAILABLE = True
except ImportError:
    pass

def _get_secret(key: str, fallback: str = "") -> str:
    """Read from st.secrets (Streamlit Cloud) with os.getenv fallback (local)."""
    try:
        return str(st.secrets[key])
    except Exception:
        return os.getenv(key, fallback)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Quant Regime Dashboard",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME — Terminal Aurora (upgraded from v2)
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,300&display=swap');

:root {
  --bg0:    #04060C;
  --bg1:    #080E1C;
  --bg2:    #0D1628;
  --panel:  rgba(255,255,255,0.038);
  --border: rgba(255,255,255,0.09);
  --text:   rgba(255,255,255,0.92);
  --muted:  rgba(255,255,255,0.52);
  --dim:    rgba(255,255,255,0.28);

  --green:   #10b981;
  --teal:    #06b6d4;
  --yellow:  #f59e0b;
  --orange:  #f97316;
  --red:     #ef4444;
  --crimson: #dc2626;
  --purple:  #8b5cf6;
  --blue:    #3b82f6;
  --sky:     #38bdf8;

  --gex-pos: #10b981;
  --gex-neg: #ef4444;
  --gex-flip:#f59e0b;

  --mono: 'Space Mono', ui-monospace, monospace;
  --sans: 'DM Sans', system-ui, sans-serif;

  --shadow-lg: 0 16px 48px rgba(0,0,0,0.60);
  --shadow-sm: 0 4px 16px rgba(0,0,0,0.40);
  --glow-green: 0 0 20px rgba(16,185,129,0.20);
  --glow-red:   0 0 20px rgba(239,68,68,0.20);
  --glow-blue:  0 0 20px rgba(59,130,246,0.15);
}

html, body, [class*="css"] {
  font-family: var(--sans) !important;
  background:
    radial-gradient(ellipse 140% 60% at 15% -5%, rgba(59,130,246,0.07), transparent 60%),
    radial-gradient(ellipse 100% 50% at 88% 8%,  rgba(139,92,246,0.06), transparent 55%),
    radial-gradient(ellipse 80%  40% at 50% 100%, rgba(16,185,129,0.04), transparent 60%),
    linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 50%, var(--bg2) 100%) !important;
  color: var(--text) !important;
  min-height: 100vh;
}

.block-container {
  padding-top: 0.8rem !important;
  padding-bottom: 1rem !important;
  max-width: 1600px !important;
}

/* ─── PANELS ─── */
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px 12px;
  box-shadow: var(--shadow-sm);
  backdrop-filter: blur(16px);
  transition: border-color 140ms, transform 120ms;
}
.panel:hover { border-color: rgba(255,255,255,0.14); transform: translateY(-1px); }
.panel-title {
  font-family: var(--mono);
  font-size: 9.5px;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0 0 8px 0;
}

/* ─── CARDS ─── */
.prob-card {
  background: rgba(59,130,246,0.06);
  border: 1px solid rgba(59,130,246,0.22);
  border-radius: 14px;
  padding: 14px;
}
.gex-card {
  background: rgba(16,185,129,0.05);
  border: 1px solid rgba(16,185,129,0.18);
  border-radius: 14px;
  padding: 14px;
}
.warn-card {
  background: rgba(245,158,11,0.06);
  border: 1px solid rgba(245,158,11,0.28);
  border-radius: 12px;
  padding: 10px 14px;
  font-size: 12px;
}
.alert-card {
  background: rgba(239,68,68,0.07);
  border: 1px solid rgba(239,68,68,0.32);
  border-radius: 12px;
  padding: 10px 14px;
  font-size: 12px;
}
.setup-card {
  background: rgba(139,92,246,0.06);
  border: 1px solid rgba(139,92,246,0.22);
  border-radius: 12px;
  padding: 12px 14px;
}
.api-card {
  background: rgba(6,182,212,0.05);
  border: 1px solid rgba(6,182,212,0.18);
  border-radius: 14px;
  padding: 14px;
}
.failure-card {
  background: rgba(220,38,38,0.06);
  border: 1px solid rgba(220,38,38,0.25);
  border-radius: 12px;
  padding: 10px 14px;
}

/* ─── BADGES ─── */
.badge {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 4px 9px; border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.03);
  color: var(--muted); font-size: 10.5px;
  font-family: var(--mono);
}
.badge b { color: var(--text); font-weight: 700; }

/* ─── REGIME CHIP ─── */
.regime-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 12px; border-radius: 8px;
  font-family: var(--mono); font-size: 10px;
  font-weight: 700; letter-spacing: 0.8px;
  text-transform: uppercase;
}

/* ─── PROBABILITY BAR ─── */
.pbar-wrap { background: rgba(255,255,255,0.06); border-radius: 999px; height: 5px; width: 100%; margin: 3px 0 1px; }
.pbar-fill  { border-radius: 999px; height: 5px; transition: width 400ms ease; }
.pbar-label { font-family: var(--mono); font-size: 10px; color: var(--muted); }

/* ─── GEX LEVEL BARS ─── */
.gex-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
  font-family: var(--mono); font-size: 11px;
}
.gex-row:last-child { border-bottom: none; }
.gex-pos { color: var(--green); }
.gex-neg { color: var(--red); }
.gex-flip-marker { color: var(--yellow); font-weight: 700; }

/* ─── SETUP SCORE ─── */
.score-dot {
  display: inline-block; width: 8px; height: 8px;
  border-radius: 50%; margin-right: 5px;
}

/* ─── CHECKLIST ─── */
.check-row {
  display: flex; align-items: flex-start; gap: 8px;
  padding: 4px 0; font-size: 12px; line-height: 1.4;
}
.check-ok   { color: var(--green); font-family: var(--mono); }
.check-warn { color: var(--yellow); font-family: var(--mono); }
.check-fail { color: var(--red); font-family: var(--mono); }

/* ─── TERMINAL LOG ─── */
.term {
  font-family: var(--mono); font-size: 11px; line-height: 1.5;
  background: rgba(0,0,0,0.30); border: 1px solid rgba(255,255,255,0.06);
  border-radius: 10px; padding: 10px 12px; max-height: 280px; overflow-y: auto;
}
.term-row { padding: 2px 0; }
.term-ts   { color: var(--dim); margin-right: 8px; }
.term-hi   { color: var(--sky); }
.term-ok   { color: var(--green); }
.term-warn { color: var(--yellow); }
.term-err  { color: var(--red); }

/* ─── MISC ─── */
.small { font-size: 11px; color: var(--muted); }
.mono  { font-family: var(--mono); }
hr { border-color: rgba(255,255,255,0.07) !important; }
h1,h2,h3 { font-family: var(--sans) !important; letter-spacing: -0.2px; }
div[data-testid="stMetricValue"] { font-family: var(--mono) !important; }

/* ─── SECTION HEADER ─── */
.section-hdr {
  display: flex; align-items: center; gap: 10px;
  margin: 18px 0 10px;
}
.section-hdr::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(to right, rgba(255,255,255,0.12), transparent);
}
.section-hdr span {
  font-family: var(--mono); font-size: 10px; letter-spacing: 1.4px;
  text-transform: uppercase; color: var(--muted); white-space: nowrap;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# ENUMS & DATACLASSES
# ============================================================
class GammaRegime(Enum):
    STRONG_POSITIVE = "STRONG +"
    POSITIVE        = "POSITIVE"
    NEUTRAL         = "NEUTRAL"
    NEGATIVE        = "NEGATIVE"
    STRONG_NEGATIVE = "STRONG −"

REGIME_COLORS = {
    GammaRegime.STRONG_POSITIVE: "var(--green)",
    GammaRegime.POSITIVE:        "var(--teal)",
    GammaRegime.NEUTRAL:         "var(--yellow)",
    GammaRegime.NEGATIVE:        "var(--orange)",
    GammaRegime.STRONG_NEGATIVE: "var(--red)",
}

REGIME_BG = {
    GammaRegime.STRONG_POSITIVE: "rgba(16,185,129,0.12)",
    GammaRegime.POSITIVE:        "rgba(6,182,212,0.09)",
    GammaRegime.NEUTRAL:         "rgba(245,158,11,0.08)",
    GammaRegime.NEGATIVE:        "rgba(249,115,22,0.09)",
    GammaRegime.STRONG_NEGATIVE: "rgba(239,68,68,0.12)",
}

@dataclass
class GammaState:
    regime: GammaRegime = GammaRegime.NEUTRAL
    gamma_flip: float = 0.0
    distance_to_flip_pct: float = 0.0
    total_gex: float = 0.0
    gex_by_strike: Dict[float, float] = field(default_factory=dict)
    key_support: List[float] = field(default_factory=list)
    key_resistance: List[float] = field(default_factory=list)
    regime_stability: float = 0.5
    data_source: str = "unavailable"
    timestamp: str = ""

@dataclass
class SetupScore:
    gamma_alignment: float = 0.5
    orderflow_confirmation: float = 0.5
    tpo_context: float = 0.5
    level_freshness: float = 0.9
    event_risk: float = 1.0

    @property
    def composite(self) -> float:
        return (
            self.gamma_alignment * 0.30 +
            self.orderflow_confirmation * 0.30 +
            self.tpo_context * 0.15 +
            self.level_freshness * 0.15 +
            self.event_risk * 0.10
        )

    @property
    def tradeable(self) -> bool:
        return (self.composite >= 0.65 and
                self.gamma_alignment >= 0.5 and
                self.orderflow_confirmation >= 0.5)

@dataclass
class FeedItem:
    title: str
    link: str
    published: str
    source: str

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("## ⚡ Quant Dashboard")
page = st.sidebar.radio(
    "Module",
    ["Dashboard", "GEX Engine", "Trade Setups", "Execution", "Probability Engine", "Schwab/TOS", "Guide"],
    index=0
)
st.sidebar.markdown("---")

# ============================================================
# HELPERS — CORE MATH
# ============================================================
def _to_1d(x, index=None) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0] if x.shape[1] >= 1 else pd.Series(dtype=float)
    if isinstance(x, pd.Series):
        s = x.copy()
        if index is not None: s = s.reindex(index)
        return pd.Series(np.asarray(s).squeeze(), index=s.index, dtype=float)
    arr = np.squeeze(np.asarray(x))
    if arr.ndim != 1: arr = arr.reshape(-1)
    return pd.Series(arr, index=index, dtype=float) if index is not None else pd.Series(arr, dtype=float)

def zscore(x: pd.Series) -> pd.Series:
    s = _to_1d(x)
    if len(s) == 0: return s
    mu, sd = float(np.nanmean(s.values)), float(np.nanstd(s.values))
    if not np.isfinite(sd) or sd == 0: return s * 0.0
    return (s - mu) / sd

def rolling_pct(s: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling historical percentile rank over a trailing `window`-day window.

    CORRECT USE: read only the CURRENT (last) value as today's signal.
      Each value is computed using only the trailing `window` observations
      available at that point in time — no future data is used.

    CAUTION ON HISTORICAL CHARTS: plotting the full time series is valid
      but can be misleading. The economic regime contained in any given
      252-day window changes over time, so a "80th percentile" in 2020
      and a "80th percentile" today are not comparable in absolute terms.
      Label historical charts accordingly (see `rolling_pct_for_chart`).
    """
    s = _to_1d(s)
    def _p(arr: np.ndarray) -> float:
        v = arr[-1]
        if not np.isfinite(v):
            return np.nan
        finite = arr[np.isfinite(arr)]
        if len(finite) < 5:
            return np.nan
        return float(scipy_stats.percentileofscore(finite, v, kind="rank"))
    return s.rolling(window, min_periods=20).apply(_p, raw=True)


def current_pct_rank(s: pd.Series, window: int = 252) -> float:
    """
    Return ONLY the current (latest) percentile rank.
    Preferred over rolling_pct when you only need today's value —
    avoids computing the full time series unnecessarily.
    """
    s = _to_1d(s).dropna()
    if len(s) < 5:
        return 50.0
    lookback = s.iloc[-window:]
    v = float(lookback.iloc[-1])
    return float(scipy_stats.percentileofscore(lookback.values, v, kind="rank"))

def bayesian_blend(prior: float, likelihood: float, w=0.45) -> float:
    def logit(p): return math.log(max(p, 0.5) / max(100 - p, 0.5) + 1e-9)
    lo = (1 - w) * logit(prior) + w * logit(likelihood)
    return float(100 / (1 + math.exp(-lo)))

def kelly(p_win: float, payoff=1.0) -> float:
    p_loss = 1 - p_win / 100
    p_win_f = p_win / 100
    k = (payoff * p_win_f - p_loss) / payoff
    return float(np.clip(k, 0, 1))

def resample_ffill(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    s = _to_1d(s)
    if len(s) == 0: return pd.Series(index=idx, data=np.nan, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index().reindex(idx).ffill()

def yf_close(symbol, start, end, idx) -> pd.Series:
    end_excl = (end + dt.timedelta(days=1)).isoformat()
    def _extract(df):
        if df is None or len(df) == 0: return pd.Series(index=idx, data=np.nan, dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            col = df["Close"].iloc[:, 0] if "Close" in df.columns.get_level_values(0) else df.iloc[:, 0]
        else:
            col = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        col = _to_1d(col); col.index = pd.to_datetime(col.index)
        return col.sort_index().reindex(idx).ffill()
    for _ in range(2):
        try:
            df = yf.download(symbol, start=start.isoformat(), end=end_excl,
                             auto_adjust=True, progress=False, group_by="column", threads=False)
            s = _extract(df)
            if int(s.notna().sum()) > 10: return s
        except: pass
    try:
        return _extract(yf.Ticker(symbol).history(start=start.isoformat(), end=end_excl, auto_adjust=True))
    except:
        return pd.Series(index=idx, data=np.nan, dtype=float)

# ============================================================
# HELPERS — UI
# ============================================================
def pill(label, value):
    return f"<span class='badge'>{label}: <b>{value}</b></span>"

def colored(v, tone):
    m = {"green":"var(--green)","teal":"var(--teal)","yellow":"var(--yellow)",
         "orange":"var(--orange)","red":"var(--red)","purple":"var(--purple)","blue":"var(--blue)"}
    return f"<span style='color:{m.get(tone,'var(--text)')}; font-weight:700'>{v}</span>"

def pbar(prob, color="var(--blue)"):
    p = float(np.clip(prob, 0, 100))
    return (f"<div class='pbar-wrap'><div class='pbar-fill' style='width:{p:.0f}%;background:{color};'></div></div>"
            f"<div class='pbar-label'>{p:.0f}%</div>")

def regime_chip(regime: GammaRegime):
    c = REGIME_COLORS.get(regime, "var(--text)")
    bg = REGIME_BG.get(regime, "rgba(255,255,255,0.05)")
    return (f"<span class='regime-chip' style='color:{c};background:{bg};border:1px solid {c}33;'>"
            f"{regime.value}</span>")

def sec_hdr(label):
    return f"<div class='section-hdr'><span>{label}</span></div>"

def plotly_dark(fig, title="", height=300):
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color="rgba(255,255,255,0.6)")),
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.85)", family="Space Mono"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
        legend=dict(font=dict(size=10)),
    )
    return fig

def gauge(value, title, vmin=0, vmax=100):
    value = float(np.clip(value, vmin, vmax))
    frac = (value - vmin) / (vmax - vmin)
    color = "#10b981" if frac < 0.35 else ("#f59e0b" if frac < 0.55 else ("#f97316" if frac < 0.75 else "#ef4444"))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": title, "font": {"size": 10}},
        number={"font": {"size": 18, "family": "Space Mono"}},
        gauge={
            "axis": {"range": [vmin, vmax], "tickfont": {"size": 9}},
            "bar": {"thickness": 0.22, "color": color},
            "steps": [
                {"range": [vmin, vmin+(vmax-vmin)*0.25], "color": "rgba(16,185,129,0.18)"},
                {"range": [vmin+(vmax-vmin)*0.25, vmin+(vmax-vmin)*0.50], "color": "rgba(245,158,11,0.18)"},
                {"range": [vmin+(vmax-vmin)*0.50, vmin+(vmax-vmin)*0.75], "color": "rgba(249,115,22,0.18)"},
                {"range": [vmin+(vmax-vmin)*0.75, vmax], "color": "rgba(239,68,68,0.18)"},
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=6,r=6,t=38,b=6),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="rgba(255,255,255,0.85)"))
    return fig

def autorefresh_js(seconds, enabled):
    if not enabled: return
    st.components.v1.html(
        f"<script>setTimeout(()=>{{window.parent.location.reload();}},{seconds*1000});</script>",
        height=0)

# ============================================================
# GEX ENGINE
# ============================================================
def bs_gamma(S, K, T, sigma, r=0.05) -> float:
    if T <= 0 or sigma <= 0: return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return scipy_norm.pdf(d1) / (S * sigma * math.sqrt(T))

def compute_gex_from_chain(chain: pd.DataFrame, spot: float,
                            multiplier=100, r=0.05) -> pd.DataFrame:
    """
    chain must have: strike, expiry_T, iv, call_oi, put_oi
    GEX sign convention: calls sold by dealers = positive (stabilizing)
                         puts sold by dealers = negative (destabilizing)
    """
    chain = chain.copy()
    chain["gamma"] = chain.apply(
        lambda row: bs_gamma(spot, row["strike"], row["expiry_T"], row["iv"], r), axis=1)
    chain["call_gex"] =  chain["call_oi"] * chain["gamma"] * multiplier * spot
    chain["put_gex"]  = -chain["put_oi"] * chain["gamma"] * multiplier * spot
    chain["net_gex"]  =  chain["call_gex"] + chain["put_gex"]
    return chain

def find_gamma_flip(chain: pd.DataFrame) -> float:
    sc = chain.sort_values("strike").reset_index(drop=True)
    cum = sc["net_gex"].cumsum()
    signs = cum.values[:-1] * cum.values[1:]
    idx = np.where(signs < 0)[0]
    if len(idx) == 0: return np.nan
    i = idx[0]
    s1, s2 = sc["strike"].iloc[i], sc["strike"].iloc[i+1]
    g1, g2 = cum.iloc[i], cum.iloc[i+1]
    return s1 + (s2 - s1) * (-g1) / (g2 - g1) if (g2 - g1) != 0 else s1

def classify_gex_regime(spot: float, flip: float) -> Tuple[GammaRegime, float, float]:
    if not np.isfinite(flip): return GammaRegime.NEUTRAL, 0.0, 0.5
    dist_pct = (spot - flip) / flip * 100
    if dist_pct >  2.0: regime = GammaRegime.STRONG_POSITIVE
    elif dist_pct > 0.5: regime = GammaRegime.POSITIVE
    elif dist_pct > -0.5: regime = GammaRegime.NEUTRAL
    elif dist_pct > -2.0: regime = GammaRegime.NEGATIVE
    else:                  regime = GammaRegime.STRONG_NEGATIVE
    stability = float(np.clip(min(abs(dist_pct - 0.5), abs(dist_pct + 0.5)) / 2.0, 0, 1))
    return regime, dist_pct, stability

def build_gamma_state(chain: pd.DataFrame, spot: float, source: str = "yfinance") -> GammaState:
    gex_chain = compute_gex_from_chain(chain, spot)
    flip = find_gamma_flip(gex_chain)
    regime, dist, stability = classify_gex_regime(spot, flip)
    by_strike = dict(zip(gex_chain["strike"].tolist(), gex_chain["net_gex"].tolist()))
    top_support    = gex_chain[gex_chain["net_gex"] < 0].nsmallest(5, "net_gex")["strike"].tolist()
    top_resistance = gex_chain[gex_chain["net_gex"] > 0].nlargest(5, "net_gex")["strike"].tolist()
    return GammaState(
        regime=regime, gamma_flip=float(flip) if np.isfinite(flip) else 0.0,
        distance_to_flip_pct=dist, total_gex=float(gex_chain["net_gex"].sum()),
        gex_by_strike=by_strike, key_support=top_support, key_resistance=top_resistance,
        regime_stability=stability, data_source=source,
        timestamp=dt.datetime.now().strftime("%H:%M:%S"),
    )

# ============================================================
# IBKR CONNECTION
# ============================================================
# ============================================================
# SCHWAB API — Connection + Data Functions
# ============================================================

# ============================================================
# SCHWAB + SUPABASE TOKEN LAYER
# ============================================================
# Architecture:
#   1. OAuth2 flow runs in the browser (Schwab → callback URL)
#   2. Token JSON is stored in Supabase (persists across Streamlit deploys)
#   3. schwab-py reads the token from a temp file on each request
#      (schwab-py needs a file path, so we write/read to tmp on each use)
#
# Supabase table schema (create once in Supabase SQL editor):
#   CREATE TABLE schwab_tokens (
#     id      TEXT PRIMARY KEY DEFAULT 'shared',
#     token   JSONB NOT NULL,
#     updated TIMESTAMPTZ DEFAULT NOW()
#   );
# ============================================================

def _get_supabase() -> Optional[object]:
    """Return a Supabase client using secrets."""
    if not SUPABASE_AVAILABLE:
        return None
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_KEY")   # use the anon/public key
    if not url or not key:
        return None
    try:
        return _supa_create_client(url, key)
    except Exception:
        return None


def _supabase_load_token() -> Optional[Dict]:
    """Load the Schwab token dict from Supabase."""
    sb = _get_supabase()
    if sb is None:
        return None
    try:
        res = sb.table("schwab_tokens").select("token").eq("id", "shared").execute()
        if res.data:
            return res.data[0]["token"]
    except Exception:
        pass
    return None


def _supabase_save_token(token_dict: Dict) -> bool:
    """Upsert the Schwab token dict to Supabase."""
    sb = _get_supabase()
    if sb is None:
        return False
    try:
        sb.table("schwab_tokens").upsert({
            "id": "shared",
            "token": token_dict,
        }).execute()
        return True
    except Exception:
        return False


def _token_to_tempfile(token_dict: Dict) -> str:
    """
    Write token dict to a temp file and return its path.
    schwab-py requires a file path for token storage; we use a temp file
    as a bridge between Supabase (our real store) and the library.
    The temp file is recreated on every call — no filesystem persistence.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="schwab_token_"
    )
    json.dump(token_dict, tmp)
    tmp.flush()
    tmp.close()
    return tmp.name


def _token_from_tempfile(path: str) -> Optional[Dict]:
    """Read token from temp file (after schwab-py may have refreshed it)."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_resource(ttl=300)   # re-check token every 5 min
def get_schwab_client():
    """
    Return an authenticated Schwab client backed by Supabase token storage.

    Flow on each call:
      1. Load token JSON from Supabase
      2. Write to temp file (schwab-py needs a file path)
      3. Build client — schwab-py auto-refreshes if token is near expiry
      4. Read back refreshed token and save to Supabase
    """
    if not SCHWAB_AVAILABLE:
        return None
    client_id     = _get_secret("SCHWAB_CLIENT_ID")
    client_secret = _get_secret("SCHWAB_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    token_dict = _supabase_load_token()
    if token_dict is None:
        return None   # not yet authorised

    try:
        tmp_path = _token_to_tempfile(token_dict)
        client   = schwab.auth.client_from_token_file(
            tmp_path, client_id, client_secret
        )
        # Save any refreshed token back to Supabase
        refreshed = _token_from_tempfile(tmp_path)
        if refreshed and refreshed != token_dict:
            _supabase_save_token(refreshed)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return client
    except Exception:
        return None


def schwab_run_auth_flow(client_id: str, client_secret: str,
                          redirect_uri: str) -> Optional[str]:
    """
    Start the OAuth2 flow. Returns the Schwab authorization URL.
    The user opens this URL, logs in, and is redirected to redirect_uri.
    """
    if not SCHWAB_AVAILABLE:
        return None
    try:
        # schwab-py's OAuth2 helper — we use a temp path as placeholder
        tmp_path = tempfile.mktemp(suffix=".json", prefix="schwab_auth_")
        oauth    = schwab.auth.OAuth2Client(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            token_path=tmp_path,
        )
        auth_url, state = oauth.authorization_url()
        # Store state for CSRF validation
        st.session_state["_schwab_oauth_state"]    = state
        st.session_state["_schwab_oauth_tmp_path"] = tmp_path
        st.session_state["_schwab_oauth_redir"]    = redirect_uri
        return auth_url
    except Exception as e:
        return f"error: {e}"


def schwab_complete_auth(client_id: str, client_secret: str,
                          redirect_uri: str, callback_url: str) -> Tuple[bool, str]:
    """
    Complete the OAuth2 flow. Saves the token to Supabase.
    Returns (success: bool, message: str).
    """
    if not SCHWAB_AVAILABLE:
        return False, "schwab-py not installed"
    try:
        tmp_path = st.session_state.get("_schwab_oauth_tmp_path",
                                         tempfile.mktemp(suffix=".json"))
        oauth = schwab.auth.OAuth2Client(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            token_path=tmp_path,
        )
        oauth.fetch_token(authorization_response=callback_url)
        token_dict = _token_from_tempfile(tmp_path)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        if token_dict is None:
            return False, "Token file empty after exchange"
        if not SUPABASE_AVAILABLE or _get_supabase() is None:
            # Fallback: store in session_state only (local dev without Supabase)
            st.session_state["_schwab_token_local"] = token_dict
            return True, "Token stored in session (no Supabase — local mode)"
        saved = _supabase_save_token(token_dict)
        if saved:
            get_schwab_client.clear()
            return True, "Token saved to Supabase ✓"
        else:
            return False, "Supabase save failed — check SUPABASE_URL and SUPABASE_KEY"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def schwab_get_spot(client, symbol: str = "SPY") -> Optional[float]:
    """Get current quote from Schwab API."""
    if client is None:
        return None
    try:
        resp = client.get_quote(symbol)
        if resp.status_code == 200:
            data = resp.json()
            q = data.get(symbol, {})
            # Schwab quote structure: quote.lastPrice or quote.mark
            quote = q.get("quote", {})
            price = (quote.get("lastPrice")
                     or quote.get("mark")
                     or quote.get("closePrice"))
            return float(price) if price else None
    except Exception:
        pass
    return None


def schwab_get_options_chain(client, symbol: str = "SPY",
                              spot: Optional[float] = None) -> Optional[pd.DataFrame]:
    """
    Fetch options chain from Schwab API for GEX computation.

    Schwab's /marketdata/v1/chains endpoint returns:
        - impliedVolatility  per strike  ← real per-strike IV (no surface fitting needed)
        - openInterest       per strike  ← exchange-reported OI
        - delta, gamma, theta, vega      ← we use gamma directly
        - daysToExpiration               ← T

    This is cleaner than the IBKR approach: real per-strike Greeks from
    Schwab's own model, no SVI surface approximation required.

    Parameters:
        contractType: "ALL" — calls and puts
        strikeCount:  30    — ±15 strikes around ATM
        range:        "NTM" — near-the-money
    """
    if client is None:
        return None
    try:
        today    = dt.date.today()
        spot_est = spot or 500.0

        # Request near-the-money chain, 2 nearest expirations
        resp = client.get_option_chain(
            symbol,
            contract_type=schwab.client.Client.Options.ContractType.ALL,
            strike_count=30,
            include_underlying_quote=True,
            strategy=schwab.client.Client.Options.Strategy.SINGLE,
            option_type=schwab.client.Client.Options.Type.ALL,
        )

        if resp.status_code != 200:
            st.session_state["_schwab_chain_error"] = (
                f"API error {resp.status_code}: {resp.text[:200]}"
            )
            return None

        data = resp.json()
        rows = []

        for right_key, right_char in [("callExpDateMap", "C"), ("putExpDateMap", "P")]:
            exp_map = data.get(right_key, {})
            for exp_str, strikes_dict in exp_map.items():
                # exp_str format: "2025-01-17:30" (date:DTE)
                try:
                    exp_date_str = exp_str.split(":")[0]
                    exp_dt = dt.date.fromisoformat(exp_date_str)
                    T = max((exp_dt - today).days / 365.0, 1.0 / 365.0)
                except Exception:
                    continue

                for strike_str, contracts in strikes_dict.items():
                    try:
                        strike = float(strike_str)
                        # Skip far out-of-money
                        if not (spot_est * 0.88 <= strike <= spot_est * 1.12):
                            continue
                        for contract in contracts:
                            iv  = float(contract.get("volatility", 0) or 0) / 100.0
                            oi  = int(contract.get("openInterest", 0) or 0)
                            # Use Schwab's own gamma if available
                            gk  = float(contract.get("gamma", 0) or 0)

                            if iv <= 0 or not np.isfinite(iv):
                                iv = 0.18
                            rows.append({
                                "strike":      strike,
                                "expiry_T":    T,
                                "iv":          float(np.clip(iv, 0.01, 5.0)),
                                "call_oi":     oi if right_char == "C" else 0,
                                "put_oi":      oi if right_char == "P" else 0,
                                "schwab_gamma": gk if right_char == "C" else -gk,
                            })
                    except Exception:
                        continue

        if not rows:
            st.session_state["_schwab_chain_error"] = "No chain data in response"
            return None

        df = (pd.DataFrame(rows)
                .groupby(["strike", "expiry_T"])
                .agg(
                    iv=("iv", "mean"),
                    call_oi=("call_oi", "sum"),
                    put_oi=("put_oi", "sum"),
                )
                .reset_index())
        return df

    except Exception as e:
        st.session_state["_schwab_chain_error"] = f"{type(e).__name__}: {e}"
        return None


# ============================================================
# FALLBACK SYNTHETIC GEX (from yfinance options)
# ============================================================
@st.cache_data(ttl=900)
def get_gex_from_yfinance(symbol="SPY") -> Tuple[Optional[pd.DataFrame], float, str]:
    """Pull option chain from yfinance and compute GEX."""
    try:
        ticker = yf.Ticker(symbol)
        spot = ticker.info.get("regularMarketPrice") or ticker.info.get("previousClose") or 580.0
        exps = ticker.options
        if not exps: return None, float(spot), "yfinance (no options)"
        rows = []
        today = dt.date.today()
        for exp in exps[:3]:
            try:
                exp_dt = dt.datetime.strptime(exp, "%Y-%m-%d").date()
                T = max((exp_dt - today).days / 365.0, 1/365)
                chain = ticker.option_chain(exp)
                calls = chain.calls[["strike","impliedVolatility","openInterest"]].copy()
                calls.columns = ["strike","iv","call_oi"]
                calls["put_oi"] = 0
                puts  = chain.puts[["strike","impliedVolatility","openInterest"]].copy()
                puts.columns = ["strike","iv","put_oi"]
                puts["call_oi"] = 0
                for df_leg in [calls, puts]:
                    df_leg["expiry_T"] = T
                    df_leg["iv"] = df_leg["iv"].fillna(0.20).clip(0.05, 5.0)
                    df_leg[["call_oi","put_oi"]] = df_leg[["call_oi","put_oi"]].fillna(0).astype(int)
                    rows.append(df_leg)
            except: pass
        if not rows: return None, float(spot), "yfinance (chain error)"
        full = pd.concat(rows, ignore_index=True)
        # Near-the-money filter
        full = full[(full["strike"] > spot * 0.88) & (full["strike"] < spot * 1.12)]
        agg = full.groupby(["strike","expiry_T"]).agg(
            iv=("iv","mean"), call_oi=("call_oi","sum"), put_oi=("put_oi","sum")
        ).reset_index()
        return agg, float(spot), "yfinance"
    except Exception as e:
        return None, 580.0, f"error: {e}"

# ============================================================
# DATA LOADING — MACRO
# ============================================================
@st.cache_data(ttl=1800)
def load_macro(start_iso, end_iso):
    start, end = dt.date.fromisoformat(start_iso), dt.date.fromisoformat(end_iso)
    key  = _get_secret("FRED_API_KEY")
    fred = Fred(api_key=key) if key else Fred()

    def fs(sid):
        s = fred.get_series(sid, observation_start=start_iso, observation_end=end_iso)
        s = pd.Series(s); s.index = pd.to_datetime(s.index)
        return s.sort_index()

    out = {}

    # ── Existing series (keep) ────────────────────────────────────────────
    for sid in ["DGS3MO","DGS2","DGS10","DGS30",
                "CPIAUCSL","CPILFESL","UNRATE","WALCL","WTREGEN","RRPONTSYD","M2SL","NFCI"]:
        try:    out[sid] = fs(sid)
        except: out[sid] = pd.Series(dtype=float)
    try:    out["ICSA"] = fs("ICSA")
    except: out["ICSA"] = pd.Series(dtype=float)

    # ── NEW: Real yields (10Y TIPS) ───────────────────────────────────────
    # DFII10: 10-Year Treasury Inflation-Indexed Security yield
    # Critical driver of equity multiples — completely absent before
    try:    out["DFII10"] = fs("DFII10")
    except: out["DFII10"] = pd.Series(dtype=float)

    # ── NEW: Bank reserves at Fed ────────────────────────────────────────
    # WRBWFRBL: reserve balances. Below ~$3T = repo stress risk
    try:    out["WRBWFRBL"] = fs("WRBWFRBL")
    except: out["WRBWFRBL"] = pd.Series(dtype=float)

    # ── NEW: Bank credit (for true credit impulse) ────────────────────────
    # TOTBKCR: total bank credit — flow gives credit impulse
    try:    out["TOTBKCR"] = fs("TOTBKCR")
    except: out["TOTBKCR"] = pd.Series(dtype=float)

    # ── NEW: ISM Manufacturing New Orders ────────────────────────────────
    # AMTMNO: new orders — leads headline ISM by 1-2m, GDP by 3-6m
    try:    out["AMTMNO"] = fs("AMTMNO")
    except: out["AMTMNO"] = pd.Series(dtype=float)

    # ── NEW: Money market fund assets (for net liquidity calc) ───────────
    # WRMFSL: captures RRP→MMMF migration
    try:    out["WRMFSL"] = fs("WRMFSL")
    except: out["WRMFSL"] = pd.Series(dtype=float)

    # ── NEW: GDP (for credit impulse denominator) ─────────────────────────
    # GDPC1: quarterly real GDP, will be interpolated to daily
    try:    out["GDPC1"] = fs("GDPC1")
    except: out["GDPC1"] = pd.Series(dtype=float)

    # ── Market series (keep + no change) ─────────────────────────────────
    idx = pd.date_range(start, end, freq="D")
    for sym in ["^VIX","SPY","TLT","QQQ","COPX","GLD","HYG","LQD","UUP","IWM"]:
        k = sym.replace("^","")
        out[k] = yf_close(sym, start, end, idx).dropna()

    return out

@st.cache_data(ttl=3600)
def get_fwd_pe(ticker):
    try:
        pe = yf.Ticker(ticker).info.get("forwardPE", np.nan)
        return float(pe) if pe else np.nan
    except: return np.nan

# ============================================================
# WORLD INTELLIGENCE MONITOR — Feed Infrastructure
# ============================================================
# Organised into 7 signal categories matching the thesis document.
# Each category has: feeds, keywords, impact weights, and a colour.
# ============================================================

INTEL_CATEGORIES = {
    "fed_policy": {
        "label": "Fed & Monetary Policy",
        "icon": "🏛",
        "color": "var(--blue)",
        "bg": "rgba(59,130,246,0.08)",
        "border": "rgba(59,130,246,0.25)",
        "feeds": {
            "Fed Releases":  "https://www.federalreserve.gov/feeds/press_all.xml",
            "FOMC":          "https://www.federalreserve.gov/feeds/fomcpressreleases.xml",
        },
        "keywords": ["fed","fomc","powell","rate cut","rate hike","balance sheet",
                     "quantitative","qt","qe","tapering","dot plot","basis points",
                     "federal reserve","monetary","inflation target","repo","rrp",
                     "reverse repo","tga","treasury general","slr","warsh","mirren"],
        "weights":  {"rate cut":10,"rate hike":10,"balance sheet":8,"qt":8,"qe":8,
                     "powell":6,"fomc":6,"slr":9,"warsh":7,"repo":7,"rrp":7},
        "regime_impact": "three_puts",   # maps to thesis put
    },
    "fiscal_debt": {
        "label": "Fiscal & Debt",
        "icon": "💵",
        "color": "var(--teal)",
        "bg": "rgba(6,182,212,0.07)",
        "border": "rgba(6,182,212,0.22)",
        "feeds": {
            "US Treasury":   "https://home.treasury.gov/press-center/press-releases/rss",
            "BEA":           "https://www.bea.gov/news/rss.xml",
        },
        "keywords": ["treasury","deficit","debt ceiling","t-bill","bond auction",
                     "fiscal","spending bill","tax cut","tariff revenue","stablecoin",
                     "genius act","big beautiful bill","debt issuance","tga drawdown",
                     "budget","appropriations","continuing resolution","shutdown"],
        "weights":  {"debt ceiling":10,"t-bill":7,"tariff revenue":8,"tax cut":7,
                     "tga":8,"deficit":6,"shutdown":9,"genius act":7,"stablecoin":6},
        "regime_impact": "treasury_put",
    },
    "inflation_labor": {
        "label": "Inflation & Labor",
        "icon": "📊",
        "color": "var(--yellow)",
        "bg": "rgba(245,158,11,0.07)",
        "border": "rgba(245,158,11,0.22)",
        "feeds": {
            "BLS":           "https://www.bls.gov/feed/news_release/rss.xml",
            "BEA PCE":       "https://www.bea.gov/news/rss.xml",
        },
        "keywords": ["cpi","pce","inflation","core inflation","jobless claims","payroll",
                     "unemployment","nonfarm","labor","wage","consumer price","producer price",
                     "ppi","shelter","housing cost","services inflation","sticky",
                     "jobs report","ism","pmi","manufacturing","services"],
        "weights":  {"cpi":10,"pce":10,"nonfarm":9,"jobless claims":8,"unemployment":8,
                     "inflation":7,"pmi":6,"ism":6,"wage":7,"shelter":7},
        "regime_impact": "fed_put",
    },
    "trade_tariffs": {
        "label": "Trade & Tariffs",
        "icon": "🌐",
        "color": "var(--orange)",
        "bg": "rgba(249,115,22,0.07)",
        "border": "rgba(249,115,22,0.22)",
        "feeds": {
            "Reuters World": "https://feeds.reuters.com/reuters/worldNews",
        },
        "keywords": ["tariff","trade war","trade deal","sanction","export control",
                     "import duty","90-day pause","liberation day","china trade",
                     "supply chain","reshoring","onshoring","wto","retaliatory",
                     "trade deficit","current account","customs","duties"],
        "weights":  {"tariff":10,"trade war":9,"sanction":8,"90-day pause":8,
                     "liberation day":7,"export control":7,"trade deal":7,"china":5},
        "regime_impact": "trump_put",
    },
    "geopolitical": {
        "label": "Geopolitical Risk",
        "icon": "⚔",
        "color": "var(--red)",
        "bg": "rgba(239,68,68,0.07)",
        "border": "rgba(239,68,68,0.22)",
        "feeds": {
            "BBC World":     "https://feeds.bbci.co.uk/news/world/rss.xml",
            "Reuters World": "https://feeds.reuters.com/reuters/worldNews",
        },
        "keywords": ["war","conflict","military","attack","invasion","missile","drone",
                     "nuclear","nato","iran","russia","ukraine","israel","gaza",
                     "strait of hormuz","oil supply","crude","opec","escalation",
                     "coup","regime change","sanctions","embargo","blockade"],
        "weights":  {"war":10,"nuclear":10,"invasion":9,"missile":8,"iran":7,
                     "strait of hormuz":9,"escalation":8,"crude":5,"opec":5},
        "regime_impact": "geo_shock",
    },
    "markets_liquidity": {
        "label": "Markets & Liquidity",
        "icon": "📈",
        "color": "var(--purple)",
        "bg": "rgba(139,92,246,0.07)",
        "border": "rgba(139,92,246,0.22)",
        "feeds": {
            "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
        },
        "keywords": ["margin debt","liquidity","credit spread","vix","volatility",
                     "stock market","nasdaq","s&p","mag7","magnificent seven",
                     "earnings","forward pe","valuation","buyback","ipo",
                     "hedge fund","short selling","options","gamma","repo rate",
                     "bank reserve","m2","money supply","capital flows"],
        "weights":  {"margin debt":9,"liquidity":7,"vix":7,"credit spread":8,
                     "mag7":6,"magnificent seven":6,"forward pe":7,"repo rate":7,
                     "m2":6,"money supply":6},
        "regime_impact": "market_index",
    },
    "ai_tech": {
        "label": "AI & Tech Cycle",
        "icon": "🤖",
        "color": "var(--sky)",
        "bg": "rgba(56,189,248,0.07)",
        "border": "rgba(56,189,248,0.20)",
        "feeds": {
            "Reuters Tech":  "https://feeds.reuters.com/reuters/technologyNews",
        },
        "keywords": ["artificial intelligence","ai","nvidia","microsoft","google",
                     "amazon","meta","apple","openai","data center","capex",
                     "semiconductor","chip","gpu","hyperscaler","inference",
                     "antitrust","regulation","ipo","valuation","bubble",
                     "margin debt","finra","leverage"],
        "weights":  {"nvidia":6,"openai":6,"data center":7,"semiconductor":6,
                     "antitrust":7,"bubble":8,"margin debt":8,"capex":5},
        "regime_impact": "bubble_score",
    },
}

# Flat feed map for bulk loading (deduplicated)
def _all_feeds_flat() -> Dict[str, str]:
    seen_urls, flat = set(), {}
    for cat_data in INTEL_CATEGORIES.values():
        for name, url in cat_data["feeds"].items():
            if url not in seen_urls:
                flat[name] = url
                seen_urls.add(url)
    return flat

# Backwards-compat aliases used elsewhere in the file
FEEDS_MACRO = {k: v for k, v in _all_feeds_flat().items()
               if any(k in cat["feeds"] for cat in [INTEL_CATEGORIES["fed_policy"],
                       INTEL_CATEGORIES["fiscal_debt"], INTEL_CATEGORIES["inflation_labor"]])}
FEEDS_GEO   = {k: v for k, v in _all_feeds_flat().items()
               if any(k in cat["feeds"] for cat in [INTEL_CATEGORIES["geopolitical"],
                       INTEL_CATEGORIES["trade_tariffs"]])}

RELEVANCE_KW = list({kw for cat in INTEL_CATEGORIES.values() for kw in cat["keywords"]})

# ── Geo shock: negation patterns that cancel keyword matches ──────────────
# If any negation phrase appears in the same title, the geo keyword is
# discarded.  This catches "price war", "culture war", "star wars",
# "cyber attack" on mundane targets, "missile" in sports, etc.
_GEO_NEGATION_PHRASES = [
    "price war", "culture war", "star wars", "trade war",  # trade war handled by TRADE category
    "cyber attack", "heart attack", "panic attack", "asthma attack",
    "shark attack", "dog attack", "pepper spray", "sales",
    "marketing", "ad campaign", "award", "concert", "music",
    "film", "movie", "game", "sport", "pitch", "pitcher",
    "midfielder", "linebacker", "quarterback",
    "missile defense", "anti-missile",   # these are non-escalatory
    "nuclear plant", "nuclear power", "nuclear energy",  # civilian contexts
    "tensions ease", "tension relief",   # de-escalation
]

# ── Geo signals require co-occurrence with geographic/political entities ──
# A keyword only scores if one of these context terms also appears in
# the same title (ensures the headline is actually about a place/actor).
_GEO_CONTEXT_REQUIRED = {
    "war":       ["russia","ukraine","israel","iran","gaza","china","taiwan",
                  "north korea","nato","middle east","military","troops","army"],
    "attack":    ["iran","israel","ukraine","russia","nato","military","troops",
                  "airstrikes","bombing","drone","navy","base","port"],
    "missile":   ["iran","russia","china","north korea","ukraine","hypersonic",
                  "ballistic","cruise","icbm","launch","intercept"],
    "invasion":  ["russia","china","taiwan","ukraine","nato","troops","military"],
    "nuclear":   ["iran","russia","china","north korea","weapon","warhead",
                  "icbm","deterrent","treaty"],
    "escalat":   ["iran","russia","ukraine","israel","china","taiwan","nato",
                  "military","conflict","troops"],
    "blockade":  ["strait","hormuz","taiwan","shipping","navy","port","cargo"],
}
# Keywords that score without context requirement (intrinsically specific)
_GEO_NO_CONTEXT_NEEDED = {"strait of hormuz","nuclear warhead","military invasion",
                           "nato article 5","weapons of mass"}

# Source multipliers for geo scoring — geo-specialist feeds score higher
_GEO_SOURCE_WEIGHT = {
    "Reuters World": 1.0,
    "BBC World":     1.0,
    "Reuters Business": 0.35,   # business news mis-fires on "attack" etc.
    "Fed Releases":  0.0,
    "FOMC":          0.0,
    "BLS":           0.0,
    "BEA":           0.0,
    "US Treasury":   0.0,
}

def _fetch_url(url, timeout=7):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as r: return r.read().decode("utf-8", errors="ignore")

def _strip(s): return re.sub(r"\s+", " ", (s or "").strip())

def _parse_feed(xml_text, source, max_items=15):
    items = []
    try: root = ET.fromstring(xml_text)
    except: return items
    channel = root.find("channel")
    nodes = channel.findall("item")[:max_items] if channel is not None else []
    for it in nodes:
        t = it.find("title"); l = it.find("link"); p = it.find("pubDate") or it.find("date")
        title = _strip(t.text) if t is not None and t.text else ""
        if title: items.append(FeedItem(title, _strip(l.text) if l is not None else "",
                                        _strip(p.text) if p is not None else "", source))
    return items

@st.cache_data(ttl=300)
def load_feeds(feed_tuple, max_total=120):
    all_items = []
    for source, url in dict(feed_tuple).items():
        try: all_items.extend(_parse_feed(_fetch_url(url), source))
        except: pass
    return all_items[:max_total]

def score_relevance(items, max_keep=12):
    kw = [k.lower() for k in RELEVANCE_KW]
    scored = [(sum(1 for k in kw if k in (it.title+" "+it.source).lower()), it) for it in items]
    return [it for s, it in sorted(scored, key=lambda x:-x[0]) if s > 0][:max_keep]

def categorise_items(items: List[FeedItem]) -> Dict[str, List[Tuple[float, FeedItem]]]:
    """
    Assign each headline to its highest-scoring category.
    Returns {cat_key: [(score, item), ...]} sorted by score descending.
    """
    cat_results: Dict[str, List] = {k: [] for k in INTEL_CATEGORIES}
    for it in items:
        txt = (it.title + " " + it.source).lower()
        best_cat, best_score = None, 0.0
        for cat_key, cat_data in INTEL_CATEGORIES.items():
            # Weighted keyword score
            score = 0.0
            weights = cat_data["weights"]
            for kw in cat_data["keywords"]:
                if kw in txt:
                    score += weights.get(kw, 3.0)
            if score > best_score:
                best_score, best_cat = score, cat_key
        if best_cat and best_score > 0:
            cat_results[best_cat].append((best_score, it))
    # Sort each category by score
    for k in cat_results:
        cat_results[k].sort(key=lambda x: -x[0])
    return cat_results

def category_shock_score(cat_items: List[Tuple[float, FeedItem]]) -> float:
    """0–100 shock score for a single category based on weighted hits."""
    if not cat_items: return 0.0
    raw = sum(min(s, 10) for s, _ in cat_items[:10])
    return float(np.clip(raw / 100 * 100, 0, 100))

def _parse_item_age_hours(published: str) -> float:
    """
    Parse RSS pubDate string and return age in hours.
    Returns 6.0 (half a trading session) if parsing fails — a conservative
    default that doesn't inflate or deflate stale items.
    """
    if not published:
        return 6.0
    # Try common RSS date formats
    for fmt in ("%a, %d %b %Y %H:%M:%S %z",
                "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = dt.datetime.strptime(published[:31].strip(), fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            age = (dt.datetime.now(dt.timezone.utc) - parsed).total_seconds() / 3600
            return max(float(age), 0.0)
        except Exception:
            pass
    return 6.0


def _geo_item_score(it: FeedItem) -> Tuple[float, str]:
    """
    Score a single headline for geopolitical significance.

    Returns (score, reason_string).  Score 0 means the item failed
    at least one filter (negation, context requirement, source weight).

    Scoring logic:
    1. Check source weight — geo-irrelevant feeds return 0 immediately.
    2. Detect negation phrases — if any present, return 0.
    3. Check high-weight keywords with context requirements.
    4. Apply temporal decay: score *= exp(-0.04 * age_hours)
       Half-life ~17h: a 6h-old item keeps ~79% score,
       24h-old item keeps ~39%, 48h-old item keeps ~15%.
    5. Deduplicate: caller is responsible for title-hash dedup.
    """
    source_w = _GEO_SOURCE_WEIGHT.get(it.source, 0.5)
    if source_w == 0.0:
        return 0.0, ""

    txt = it.title.lower()

    # Negation check — any negation phrase cancels all geo scoring for this item
    for neg in _GEO_NEGATION_PHRASES:
        if neg in txt:
            return 0.0, f"[negated by '{neg}']"

    # High-severity keywords with context requirements
    HIGH_KW_SCORES = {
        "war": 9.0, "invasion": 9.0, "nuclear": 8.5, "missile": 7.5,
        "attack": 7.0, "escalat": 7.0, "blockade": 8.0,
        "strait of hormuz": 10.0,
    }
    # Medium-severity keywords (no context requirement needed — specific enough)
    MED_KW_SCORES = {
        "sanction": 5.0, "embargo": 5.5, "coup": 6.0,
        "military strike": 8.0, "airstrike": 8.0, "troops deployed": 7.0,
    }

    base_score = 0.0
    reason = ""

    for kw, kw_score in HIGH_KW_SCORES.items():
        if kw in txt:
            if kw in _GEO_NO_CONTEXT_NEEDED or any(kw in phrase for phrase in _GEO_NO_CONTEXT_NEEDED):
                base_score = kw_score
                reason = f"[{kw}]"
                break
            # Check context requirement
            required = _GEO_CONTEXT_REQUIRED.get(kw, [])
            if any(ctx in txt for ctx in required):
                base_score = kw_score
                reason = f"[{kw}+context]"
                break
            # Keyword present but no geographic context — low score
            base_score = max(base_score, kw_score * 0.2)
            reason = f"[{kw} no-context]"

    if base_score == 0.0:
        for kw, kw_score in MED_KW_SCORES.items():
            if kw in txt:
                base_score = kw_score
                reason = f"[{kw}]"
                break

    if base_score == 0.0:
        return 0.0, ""

    # Apply source weight
    scored = base_score * source_w

    # Temporal decay: half-life ~17h
    age_h = _parse_item_age_hours(it.published)
    decay = math.exp(-0.04 * age_h)
    scored *= decay

    return float(scored), reason


def geo_shock_score(items: List[FeedItem]) -> Tuple[float, List[str]]:
    """
    Compute a geopolitical shock score 0–100 from headlines.

    Improvements over naive keyword matching:
    - Negation detection eliminates "price war", "cyber attack" on mundane
      targets, "nuclear power plant" (civilian), etc.
    - Context co-occurrence requirement: "war" only scores if a geographic
      or military entity also appears in the same headline.
    - Source weighting: Reuters World/BBC World score at 1.0; business feeds
      that mis-fire on "attack" etc. score at 0.0–0.35.
    - Temporal decay (half-life ~17h): events age out naturally, so
      yesterday's unchanged headlines don't accumulate infinitely.
    - Title deduplication: the same event appearing across multiple feeds
      is counted once (using a 10-word title prefix hash).
    """
    seen_hashes: set = set()
    triggers: List[str] = []
    total_score = 0.0

    for it in items:
        score, reason = _geo_item_score(it)
        if score <= 0:
            continue
        # Deduplicate by first 10 words of title (catches cross-feed dupes)
        title_hash = " ".join(it.title.lower().split()[:10])
        if title_hash in seen_hashes:
            continue
        seen_hashes.add(title_hash)
        total_score += score
        severity = "⚠" if score >= 5.0 else "◈"
        triggers.append(f"{severity} {reason} {it.title[:70]}")

    # Normalise: cap at 100, but use a softer ceiling via tanh
    # This prevents a single event cluster from maxing the scale
    normalised = float(math.tanh(total_score / 40.0) * 100)
    return float(np.clip(normalised, 0, 100)), triggers[:8]

# ============================================================
# LEADING INDICATORS
# ============================================================
def compute_leading_stack(
    y2, y3m, y10, y30, s_2s10s, vix, m2, claims,
    copx, gld, hyg, lqd, dxy, spy, qqq, iwm,
    net_liq, net_liq_4w, walcl, bs_13w, idx,
    # NEW signals
    tips_10y=None, bank_reserves=None, bank_credit=None,
    ism_no=None, gdp_quarterly=None, mmmf=None,
) -> Dict:
    """
    Signal stack reorganised by forecast horizon per the architecture critique.

    Signals are grouped into three horizon buckets and NEVER blended across
    buckets in the probability composite.  Each bucket feeds the corresponding
    sub-model (tactical / short_term / medium_term).

    HORIZON_MAP — signal → horizon:
        Tactical   (1–5 days):    GEX regime (via gex_state), VIX term structure,
                                  DXY 5D momentum
        Short-term (1–4 weeks):   HYG/LQD, small-cap leadership,
                                  net liquidity 4W impulse, ISM NO momentum
        Medium-term (1–3 months): Curve phase (not just level), copper/gold 13W,
                                  real credit impulse, real rate regime,
                                  reserve adequacy, M2 acceleration

    Percentile ranking uses current_pct_rank() (trailing 252-day window,
    current observation only — no full time-series construction).

    Magnitude information is preserved alongside the percentile rank
    where economically meaningful (e.g. real rate level vs threshold).
    """
    R = {}

    # ════════════════════════════════════════════════════════════════
    # HORIZON 1 — TACTICAL (1–5 days)
    # ════════════════════════════════════════════════════════════════

    # T1. VIX term structure: VIX / 63D realised vol ratio
    # High premium → fear priced in → bearish next-week signal (inverted)
    spy_a   = _to_1d(spy).reindex(idx).ffill()
    vix_a   = _to_1d(vix).reindex(idx).ffill()
    spy_ret = spy_a.pct_change()
    rvol    = spy_ret.rolling(63, min_periods=20).std() * np.sqrt(252) * 100
    vts     = vix_a / rvol.replace(0, np.nan)
    R["vix_ts_pct"]    = current_pct_rank(-vts, 63)   # 63-day window for tactical
    R["vix_ts_level"]  = float(vts.dropna().iloc[-1]) if vts.dropna().size else np.nan

    # T2. DXY 5D momentum (short window for tactical)
    dxy_a  = _to_1d(dxy).reindex(idx).ffill()
    dxy_5d = dxy_a.pct_change(5) * 100
    R["dxy_5d_pct"]    = current_pct_rank(-dxy_5d, 63)   # inverted: DXY up = risk-off

    # ════════════════════════════════════════════════════════════════
    # HORIZON 2 — SHORT-TERM (1–4 weeks)
    # ════════════════════════════════════════════════════════════════

    # S1. HYG/LQD ratio 1M momentum — credit spread compression/widening
    hyg_a = _to_1d(hyg).reindex(idx).ffill()
    lqd_a = _to_1d(lqd).reindex(idx).ffill()
    if hyg_a.dropna().size > 21 and lqd_a.dropna().size > 21:
        hl_mom = (hyg_a / lqd_a.replace(0, np.nan)).pct_change(21) * 100
        R["hyg_lqd_pct"] = current_pct_rank(hl_mom, 252)
    else:
        R["hyg_lqd_pct"] = 50.0

    # S2. Small-cap vs large-cap leadership (IWM/SPY) 3-week momentum
    iwm_a = _to_1d(iwm).reindex(idx).ffill()
    if iwm_a.dropna().size > 21 and spy_a.dropna().size > 21:
        rs_mom = (iwm_a / spy_a.replace(0, np.nan)).pct_change(21) * 100
        R["smallcap_pct"] = current_pct_rank(rs_mom, 252)
    else:
        R["smallcap_pct"] = 50.0

    # S3. Net liquidity 4-week impulse
    R["liq_impulse_4w_pct"]   = current_pct_rank(net_liq_4w, 252)
    R["liq_impulse_4w_level"] = float(net_liq_4w.dropna().iloc[-1]) if net_liq_4w.dropna().size else 0.0

    # S4. ISM Manufacturing New Orders momentum (if available)
    if ism_no is not None and len(ism_no.dropna()) > 6:
        ism_no_r = resample_ffill(ism_no, idx)
        ism_mom  = ism_no_r.diff(2)   # 2-month change (monthly series)
        # Classify: above/below 50 AND rising/falling → 4 quadrants
        ism_latest    = float(ism_no_r.dropna().iloc[-1]) if ism_no_r.dropna().size else 50.0
        ism_mom_latest = float(ism_mom.dropna().iloc[-1]) if ism_mom.dropna().size else 0.0
        # Bull signal: below 50 but rising (contraction decelerating) OR above 50 rising
        ism_bull = float(ism_latest + ism_mom_latest * 3)   # momentum-boosted score
        R["ism_no_pct"]    = current_pct_rank(ism_no_r + ism_mom * 2, 252)
        R["ism_level"]     = ism_latest
        R["ism_momentum"]  = ism_mom_latest
        # Quadrant classification
        if ism_latest > 50 and ism_mom_latest > 0:
            R["ism_quadrant"] = "Expansion Accelerating"
        elif ism_latest > 50 and ism_mom_latest <= 0:
            R["ism_quadrant"] = "Expansion Decelerating"
        elif ism_latest <= 50 and ism_mom_latest < 0:
            R["ism_quadrant"] = "Contraction Accelerating"
        else:
            R["ism_quadrant"] = "Contraction Decelerating"  # the buy signal
    else:
        R["ism_no_pct"]   = 50.0
        R["ism_level"]    = np.nan
        R["ism_momentum"] = np.nan
        R["ism_quadrant"] = "Unknown"

    # ════════════════════════════════════════════════════════════════
    # HORIZON 3 — MEDIUM-TERM (1–3 months)
    # ════════════════════════════════════════════════════════════════

    # M1. Yield curve PHASE — not just level or simple acceleration
    # Bull steepening (both rates falling, long faster) vs
    # Bear steepening (short falling, long stable/rising) are OPPOSITE signals
    y10_20d = y10.diff(20); y2_20d = y2.diff(20)
    bull_steepen = ((y10_20d < 0) & (y2_20d < y10_20d)).astype(float)  # both down, long faster
    bear_steepen = ((y2_20d  < 0) & (y10_20d >= 0)).astype(float)      # front end down, long sticky
    bear_flatten = ((y2_20d  > 0) & (y10_20d > y2_20d)).astype(float)  # bear flatten (late tighten)
    bull_flatten = ((y10_20d < 0) & (y2_20d  >= 0)).astype(float)      # long-end rally (risk-off)

    # Encode phase as numeric: bull_steepen=+2, bear_steepen=+1, neutral=0,
    # bull_flatten=-1 (safe haven bid), bear_flatten=-2 (policy error risk)
    curve_phase_raw = 2*bull_steepen + 1*bear_steepen - 1*bull_flatten - 2*bear_flatten
    R["curve_phase_pct"]  = current_pct_rank(curve_phase_raw, 252)
    R["curve_phase_label"] = (
        "Bull Steepen"  if bool(bull_steepen.iloc[-1])  else
        "Bear Steepen"  if bool(bear_steepen.iloc[-1])  else
        "Bull Flatten"  if bool(bull_flatten.iloc[-1])  else
        "Bear Flatten"  if bool(bear_flatten.iloc[-1])  else
        "Parallel Shift"
    )
    R["curve_raw"]  = float(s_2s10s.dropna().iloc[-1]) if s_2s10s.dropna().size else 0.0
    R["curve_inverted"] = R["curve_raw"] < 0

    # M2. Copper/Gold ratio 13-week momentum (medium-term window)
    copx_a = _to_1d(copx).reindex(idx).ffill()
    gld_a  = _to_1d(gld).reindex(idx).ffill()
    if copx_a.dropna().size > 91 and gld_a.dropna().size > 91:
        cg_mom = (copx_a / gld_a.replace(0, np.nan)).pct_change(91) * 100
        R["copper_gold_pct"]   = current_pct_rank(cg_mom, 252)
        R["copper_gold_13w"]   = float(cg_mom.dropna().iloc[-1]) if cg_mom.dropna().size else 0.0
    else:
        R["copper_gold_pct"]  = 50.0
        R["copper_gold_13w"]  = 0.0

    # M3. REAL credit impulse — change in bank credit flow / GDP
    # Biggs et al (2010): this leads equity returns by ~6 months
    if bank_credit is not None and gdp_quarterly is not None        and len(bank_credit.dropna()) > 91 and len(gdp_quarterly.dropna()) > 4:
        bc_r   = resample_ffill(bank_credit, idx)
        gdp_r  = resample_ffill(gdp_quarterly, idx).ffill()  # quarterly → daily ffill
        gdp_r  = gdp_r.replace(0, np.nan)
        credit_flow    = bc_r.diff(91)             # new credit created (quarterly flow)
        credit_impulse = credit_flow.diff(91) / gdp_r  # change in flow / GDP
        R["credit_impulse_pct"]   = current_pct_rank(credit_impulse, 252)
        R["credit_impulse_level"] = float(credit_impulse.dropna().iloc[-1]) if credit_impulse.dropna().size else 0.0
    else:
        # Fallback: M2 acceleration proxy (labelled clearly as inferior)
        ci     = m2.diff(91) / m2.shift(91) * 100
        ci_roc = ci.diff(63).dropna()
        R["credit_impulse_pct"]   = current_pct_rank(ci_roc, 252)
        R["credit_impulse_level"] = float(ci_roc.dropna().iloc[-1]) if ci_roc.dropna().size else 0.0
    R["credit_impulse_source"] = "TOTBKCR/GDP" if (bank_credit is not None and len(bank_credit.dropna()) > 91) else "M2 proxy"

    # M4. Real rate regime (DFII10: 10Y TIPS yield)
    # Financial repression (<0%) vs neutral (0-1.5%) vs compression (>1.5%) vs danger (>2.5%)
    if tips_10y is not None and len(tips_10y.dropna()) > 20:
        tips_r = resample_ffill(tips_10y, idx)
        real_rate_now = float(tips_r.dropna().iloc[-1]) if tips_r.dropna().size else np.nan
        if np.isfinite(real_rate_now):
            if real_rate_now < 0:
                rr_regime = "Repression"
            elif real_rate_now < 1.5:
                rr_regime = "Neutral"
            elif real_rate_now < 2.5:
                rr_regime = "Compression"
            else:
                rr_regime = "Danger"
            # Momentum matters too: rising real rates = valuation headwind
            rr_mom = tips_r.diff(63)   # 3M change
            R["real_rate_regime"]  = rr_regime
            R["real_rate_level"]   = real_rate_now
            R["real_rate_mom_pct"] = current_pct_rank(-rr_mom, 252)  # inverted: rising = bearish
            # Bull signal: level below 1.5% AND falling (multiple expansion supported)
            R["real_rate_pct"] = float(np.clip(
                (1.5 - real_rate_now) / 3.5 * 50 + 50 +   # level component
                current_pct_rank(-rr_mom, 252) * 0.2,       # momentum component
                5, 95
            ))
        else:
            R["real_rate_regime"] = "Unknown"; R["real_rate_level"] = np.nan
            R["real_rate_pct"]    = 50.0;      R["real_rate_mom_pct"] = 50.0
    else:
        R["real_rate_regime"] = "Unavailable"; R["real_rate_level"] = np.nan
        R["real_rate_pct"]    = 50.0;          R["real_rate_mom_pct"] = 50.0

    # M5. Reserve adequacy — bank reserves at Fed
    # Below ~$3T threshold = repo stress risk (Sep 2019 analogue)
    if bank_reserves is not None and len(bank_reserves.dropna()) > 20:
        res_r = resample_ffill(bank_reserves, idx)
        res_now_b = float(res_r.dropna().iloc[-1]) if res_r.dropna().size else np.nan  # billions
        # Historical mean for normalisation (or use hard threshold)
        res_mean  = float(res_r.rolling(504, min_periods=50).mean().dropna().iloc[-1])                     if res_r.rolling(504).mean().dropna().size else 3000.0
        adequacy_ratio = res_now_b / max(res_mean, 1.0)
        R["reserve_adequacy_ratio"] = adequacy_ratio
        R["reserve_level_bn"]       = res_now_b
        # Regime classification
        if adequacy_ratio >= 0.90:
            R["reserve_regime"] = "Ample"
        elif adequacy_ratio >= 0.75:
            R["reserve_regime"] = "Watch"
        else:
            R["reserve_regime"] = "Alert"   # repo stress risk zone
        R["reserve_pct"] = float(np.clip(adequacy_ratio * 50 + 25, 5, 95))
    else:
        R["reserve_adequacy_ratio"] = np.nan
        R["reserve_regime"]         = "Unavailable"
        R["reserve_level_bn"]       = np.nan
        R["reserve_pct"]            = 50.0

    # M6. M2 YoY growth (keep — medium-term liquidity backdrop)
    m2_yoy = (m2 / m2.shift(365) - 1) * 100
    R["m2_yoy"]     = float(m2_yoy.dropna().iloc[-1]) if m2_yoy.dropna().size else np.nan
    R["m2_yoy_pct"] = current_pct_rank(m2_yoy, 252)

    # M7. Net liquidity 13-week impulse (medium-term window)
    liq_13w = net_liq.diff(91)
    R["liq_impulse_13w_pct"]   = current_pct_rank(liq_13w, 252)
    R["liq_impulse_13w_level"] = float(liq_13w.dropna().iloc[-1]) if liq_13w.dropna().size else 0.0

    return R

# ============================================================
# PROBABILISTIC COMPOSITE
# ============================================================

# ============================================================
# 1-DAY PROBABILITY MODEL
# ============================================================
def compute_1d_prob(
    gex_state: GammaState,
    spot: float,
    vix_level: float,
    vix_series: pd.Series,      # full VIX history for percentile
    spy_series: pd.Series,      # SPY history for realised vol + momentum
    hyg_series: pd.Series,      # HYG for intraday credit proxy
    lqd_series: pd.Series,
    dxy_series: pd.Series,      # UUP for dollar intraday
    s_2s10s: pd.Series,         # curve raw bp
    net_liq_4w: pd.Series,      # 4W liquidity impulse
    nfci_z: pd.Series,          # NFCI z-score series
    fear_score: float,
    session: Dict,
    idx: pd.DatetimeIndex,
) -> Dict:
    """
    1-Day directional probability — built around GEX mechanics.

    Signal architecture: 6 factors, each genuinely 1-day relevant.
    No signal from the 5-day or longer stack is included here because
    they answer a different question.

    ── FACTOR 1: GEX Regime & Proximity ──────────────────────────
    The single most important 1-day structural signal.
    Positive gamma → dealers buy dips, sell rallies → mean-reversion bias.
    Negative gamma → dealers amplify moves → trend/momentum bias.
    Flip proximity → binary outcome risk, compress toward 50.

    ── FACTOR 2: VIX Term Structure (intraday horizon) ──────────
    VIX / 5-day realised vol (not 63-day as in tactical bucket).
    The very near-term vol premium tells you whether fear is building
    TODAY, not over the next week.  High premium = options market
    pricing in near-term event risk.

    ── FACTOR 3: Intraday Momentum (5D SPY return) ──────────────
    Empirically: the strongest single-day predictor for next-day
    direction in the short run is the last 5 days of price momentum
    (Jegadeesh & Titman short-horizon reversion vs. momentum depends
    on GEX regime — positive GEX inverts it, negative GEX extends it).
    This is why we need GEX to interpret momentum correctly.

    ── FACTOR 4: Credit / Dollar Intraday Microstructure ─────────
    HYG/LQD intraday change and DXY intraday change are the two
    fastest-moving cross-asset signals.  Both lead equity by minutes
    to hours on a daily basis.  1-day change, not 21-day.

    ── FACTOR 5: Curve Inversion Status ─────────────────────────
    A simple binary: is the curve inverted?  On any given day, an
    inverted curve means carry is negative (short-term funding costs
    more than long-term), which creates a persistent structural headwind
    for risk assets regardless of regime.

    ── FACTOR 6: Session Context Multiplier ─────────────────────
    A 1-day probability is only meaningful if you can act on it.
    Outside prime-time (10:30-12:00 ET), liquidity is lower and
    intraday signals are noisier.  The session multiplier compresses
    the signal toward 50 during thin periods.

    ── GEX DIRECTION INTERPRETATION ─────────────────────────────
    This is the key conceptual point that separates this model from
    naive approaches:

    GEX does NOT give a direction by itself.  But it tells you HOW
    to interpret the other signals:

    In POSITIVE gamma:
      - Momentum signals should be FADED (dealers suppress extremes)
      - Credit/dollar signals are PRIMARY direction indicators
      - Base rate is mean-reversion to gamma flip

    In NEGATIVE gamma:
      - Momentum signals should be FOLLOWED (dealers amplify)
      - Credit/dollar signals CONFIRM the direction
      - Base rate is continuation

    In NEUTRAL (near flip):
      - No GEX signal — all signals equal weight
      - Compress toward 50, widen uncertainty

    ── PERFORMANCE EXPECTATION ──────────────────────────────────
    Realistic 1-day AUC: 0.52–0.55.  This is consistent with the
    academic literature on daily equity direction.  Anyone claiming
    >0.58 on 1-day equity direction is overfitting.  The value of
    this model is not a high AUC — it is conditional positioning:
    knowing WHEN to trade (session + GEX regime) and HOW MUCH to
    risk (Kelly with honest uncertainty).
    """
    # ── Factor 1: GEX regime ─────────────────────────────────────────────
    regime   = gex_state.regime
    dist_pct = abs(gex_state.distance_to_flip_pct)
    stability = gex_state.regime_stability

    # Base directional bias from GEX: positive = mean-reversion (neutral 50),
    # negative = amplification (still 50 directionally — GEX ≠ direction)
    # What GEX gives us is CONFIDENCE in the other signals, not direction itself
    gex_signal_confidence = float(np.clip(stability, 0.3, 1.0))

    # Flip proximity penalty — when within 0.5% of flip, all signals lose meaning
    flip_proximity_penalty = float(np.clip(1.0 - max(0, (0.75 - dist_pct) / 0.75), 0.4, 1.0))

    # ── Factor 2: VIX term structure (5D realised vol — intraday window) ──
    spy_a   = _to_1d(spy_series).reindex(idx).ffill()
    rvol_5d = spy_a.pct_change().rolling(5, min_periods=3).std() * np.sqrt(252) * 100
    vts_1d  = vix_level / float(rvol_5d.dropna().iloc[-1]) if rvol_5d.dropna().size and float(rvol_5d.dropna().iloc[-1]) > 0 else 1.0

    # High VIX/RVol ratio = fear premium = near-term bearish signal (inverted)
    # Ratio > 1.3: market pricing in more fear than recent realised → bearish 1D
    # Ratio < 0.8: complacency relative to realised vol → slightly bullish 1D
    if vts_1d > 1.5:   vts_score = 30.0   # high fear premium
    elif vts_1d > 1.2: vts_score = 42.0
    elif vts_1d > 0.9: vts_score = 50.0   # neutral zone
    elif vts_1d > 0.7: vts_score = 58.0
    else:              vts_score = 65.0   # complacency / vol crush

    # ── Factor 3: 5D SPY momentum (regime-conditioned) ───────────────────
    spy_5d_ret = float(spy_a.pct_change(5).dropna().iloc[-1]) if spy_a.pct_change(5).dropna().size else 0.0
    spy_1d_ret = float(spy_a.pct_change(1).dropna().iloc[-1]) if spy_a.pct_change(1).dropna().size else 0.0

    # Key insight: momentum interpretation depends on GEX regime
    if regime in (GammaRegime.STRONG_POSITIVE, GammaRegime.POSITIVE):
        # Positive gamma → FADE momentum (dealers mean-revert)
        # Recent strength = resistance ahead; recent weakness = support ahead
        if spy_5d_ret > 0.015:   mom_score = 38.0   # overbought into GEX resistance
        elif spy_5d_ret > 0.005: mom_score = 46.0
        elif spy_5d_ret > -0.005: mom_score = 52.0
        elif spy_5d_ret > -0.015: mom_score = 57.0
        else:                     mom_score = 63.0   # oversold, GEX support below
    elif regime in (GammaRegime.NEGATIVE, GammaRegime.STRONG_NEGATIVE):
        # Negative gamma → FOLLOW momentum (dealers amplify)
        if spy_5d_ret > 0.015:   mom_score = 62.0   # momentum continuation
        elif spy_5d_ret > 0.005: mom_score = 56.0
        elif spy_5d_ret > -0.005: mom_score = 50.0
        elif spy_5d_ret > -0.015: mom_score = 44.0
        else:                     mom_score = 36.0   # continuation downside
    else:
        # Neutral / near flip — equal weight, slight short-term reversion
        mom_score = 50.0 - spy_5d_ret * 200   # mild reversion
        mom_score = float(np.clip(mom_score, 35, 65))

    # ── Factor 4: Credit & dollar microstructure (1-day change) ─────────
    hyg_a   = _to_1d(hyg_series).reindex(idx).ffill()
    lqd_a   = _to_1d(lqd_series).reindex(idx).ffill()
    dxy_a   = _to_1d(dxy_series).reindex(idx).ffill()

    hyg_1d  = float(hyg_a.pct_change(1).dropna().iloc[-1]) if hyg_a.pct_change(1).dropna().size else 0.0
    lqd_1d  = float(lqd_a.pct_change(1).dropna().iloc[-1]) if lqd_a.pct_change(1).dropna().size else 0.0
    dxy_1d  = float(dxy_a.pct_change(1).dropna().iloc[-1]) if dxy_a.pct_change(1).dropna().size else 0.0

    # HYG/LQD spread tightening (HYG up relative to LQD) = risk-on = bullish
    credit_1d_signal = hyg_1d - lqd_1d * 0.5   # HYG weighted more
    # Dollar weakening = risk-on = bullish (inverted)
    dollar_signal    = -dxy_1d

    # Combine: both pointing same direction = stronger signal
    micro_raw = credit_1d_signal * 2000 + dollar_signal * 1500  # scale to approx -1/+1
    micro_score = float(np.clip(50.0 + micro_raw * 20, 20, 80))

    # ── Factor 5: Curve inversion (structural 1D headwind/tailwind) ───────
    curve_bp = float(s_2s10s.dropna().iloc[-1]) if s_2s10s.dropna().size else 0.0
    if curve_bp < -50:    curve_score = 38.0   # deeply inverted — persistent headwind
    elif curve_bp < -10:  curve_score = 45.0   # inverted — mild headwind
    elif curve_bp < 10:   curve_score = 50.0   # near flat — neutral
    elif curve_bp < 50:   curve_score = 54.0   # normal — mild tailwind
    else:                 curve_score = 57.0   # steep — tailwind (carry positive)

    # ── Factor 6: Net liquidity 1-day direction ─────────────────────────
    # The 4W impulse is a slow signal; for 1D we use its sign + recent acceleration
    liq_4w_level = float(net_liq_4w.dropna().iloc[-1]) if net_liq_4w.dropna().size else 0.0
    liq_accel_1d  = float(net_liq_4w.diff(1).dropna().iloc[-1]) if net_liq_4w.diff(1).dropna().size else 0.0
    liq_score = float(np.clip(52.0 + np.sign(liq_4w_level) * 5 + np.sign(liq_accel_1d) * 3, 30, 70))

    # ── Combine factors with GEX-regime-aware weights ────────────────────
    #
    # In POSITIVE GAMMA:   credit/dollar microstructure matters most (what's
    #                       actually flowing TODAY), momentum is faded
    # In NEGATIVE GAMMA:   momentum matters most (dealers amplifying),
    #                       credit confirms, VIX TS is critical
    # In NEUTRAL:          equal weights, all compressed toward 50
    #
    if regime in (GammaRegime.STRONG_POSITIVE, GammaRegime.POSITIVE):
        w = {"vts": 0.18, "mom": 0.12, "micro": 0.35, "curve": 0.12, "liq": 0.23}
    elif regime in (GammaRegime.NEGATIVE, GammaRegime.STRONG_NEGATIVE):
        w = {"vts": 0.22, "mom": 0.30, "micro": 0.25, "curve": 0.10, "liq": 0.13}
    else:
        w = {"vts": 0.20, "mom": 0.20, "micro": 0.25, "curve": 0.15, "liq": 0.20}

    raw_1d = (
        w["vts"]   * vts_score   +
        w["mom"]   * mom_score   +
        w["micro"] * micro_score +
        w["curve"] * curve_score +
        w["liq"]   * liq_score
    )

    # ── GEX confidence scaling ─────────────────────────────────────────
    # Compress toward 50 when GEX is uncertain (near flip) or stability low
    # This is the correct way to incorporate GEX into direction: as a
    # confidence scalar on the other signals, not as a direction vote
    scaled = 50.0 + (raw_1d - 50.0) * gex_signal_confidence * flip_proximity_penalty

    # ── Session context compression ───────────────────────────────────
    # 1-day probability is only reliable during prime-time liquidity
    session_mult = session.get("size_mult", 0.5)
    if session_mult < 0.5:
        # Outside prime time: compress toward 50 (signals are noisier)
        scaled = 50.0 + (scaled - 50.0) * (session_mult * 1.5)

    # ── Fear overlay (coincident risk-off condition) ──────────────────
    # High fear compresses upside potential: fear>70 → cap bull signal at 60
    # This reflects the empirical observation that in high-fear regimes,
    # upside surprises are muted (dealers are not supporting rallies)
    if fear_score > 70:
        scaled = min(scaled, 60.0 + (scaled - 60.0) * 0.4)
    elif fear_score > 55:
        scaled = min(scaled, 65.0 + (scaled - 65.0) * 0.7)

    prob_1d = float(np.clip(scaled, 10, 90))

    # ── Uncertainty band (wider than longer horizons — more noise) ─────
    # 1-day direction has ~±15pp structural uncertainty
    base_unc_1d = 15.0
    # Widen further near gamma flip (regime is unstable)
    flip_extra  = max(0, (0.75 - dist_pct) / 0.75) * 8.0
    # Widen in high VIX (vol-of-vol is high)
    vix_extra   = max(0, (vix_level - 20) / 30) * 5.0
    unc_1d = round(base_unc_1d + flip_extra + vix_extra, 1)

    lo_1d = float(np.clip(prob_1d - unc_1d, 5, 95))
    hi_1d = float(np.clip(prob_1d + unc_1d, 5, 95))

    # ── Kelly for 1D (conservative — shortest horizon, most noise) ────
    kelly_1d = kelly(prob_1d, payoff=1.0) * 0.35   # 35% Kelly (more conservative than 50%)

    # ── Narrative ─────────────────────────────────────────────────────
    regime_interp = {
        GammaRegime.STRONG_POSITIVE: "Strong pos. gamma: dealers PIN price, fade extremes",
        GammaRegime.POSITIVE:        "Pos. gamma: mean-reversion bias, fade momentum",
        GammaRegime.NEUTRAL:         "Near gamma flip: binary risk, reduce size",
        GammaRegime.NEGATIVE:        "Neg. gamma: moves amplified, follow momentum",
        GammaRegime.STRONG_NEGATIVE: "Strong neg. gamma: cascades possible, trend continuation",
    }

    # Dominant signal
    scores = {"VIX TS": vts_score, "Momentum": mom_score,
              "Credit/FX": micro_score, "Curve": curve_score, "Liquidity": liq_score}
    dominant = max(scores, key=lambda k: abs(scores[k] - 50))
    dominant_dir = "bullish" if scores[dominant] > 50 else "bearish"

    return {
        "prob_1d":        prob_1d,
        "lo_1d":          lo_1d,
        "hi_1d":          hi_1d,
        "unc_1d":         unc_1d,
        "kelly_1d":       kelly_1d,
        # Component scores (0-100, 50=neutral)
        "score_vts":      vts_score,
        "score_mom":      mom_score,
        "score_micro":    micro_score,
        "score_curve":    curve_score,
        "score_liq":      liq_score,
        "vts_ratio":      vts_1d,
        "spy_5d_ret":     spy_5d_ret,
        "credit_1d":      credit_1d_signal,
        "dollar_1d":      dollar_signal,
        "dominant_signal": dominant,
        "dominant_dir":    dominant_dir,
        "regime_interp":   regime_interp.get(regime, ""),
        "gex_confidence":  gex_signal_confidence,
        "flip_proximity":  flip_proximity_penalty,
        "session_valid":   session_mult >= 0.5,
        "_note": (
            f"1D model: GEX-regime-conditioned. "
            f"Realistic AUC 0.52-0.55. "
            f"Range: {lo_1d:.0f}-{hi_1d:.0f}%."
        ),
    }


def compute_prob_composite(leading, fear_score, three_puts_score, liq_anxiety,
                            exhaustion, market_index, geo_shock, regime_change_p,
                            gex_state: GammaState,
                            nfci_coincident: float = 50.0,
                            liq_dir_coincident: float = 50.0) -> Dict:
    """
    Three-horizon probability composite — separating signals by forecast window.

    ARCHITECTURE:
      Tactical bucket  (1–5d):   VIX term structure, DXY 5D momentum
      Short-term bucket (1–4w):  HYG/LQD, small-cap, liq impulse 4W, ISM NO
      Medium-term bucket (1–3m): Curve phase, Cu/Au 13W, credit impulse,
                                 real rate regime, reserve adequacy, M2 YoY,
                                 liq impulse 13W

    Signals are averaged WITHIN each bucket before blending across buckets.
    This eliminates intra-bucket double-counting (the core critique).

    GEX feeds execution context only (not direction). It contributes a
    volatility regime label and a size scalar, not a directional probability.

    Uncertainty band is explicit and regime-aware.
    """

    # ── Tactical bucket ───────────────────────────────────────────────────
    tactical_signals = [
        leading.get("vix_ts_pct",      50.0),
        leading.get("dxy_5d_pct",      50.0),
    ]
    tactical_prob = float(np.mean(tactical_signals))

    # ── Short-term bucket ────────────────────────────────────────────────
    short_signals = [
        leading.get("hyg_lqd_pct",          50.0),
        leading.get("smallcap_pct",          50.0),
        leading.get("liq_impulse_4w_pct",    50.0),
        leading.get("ism_no_pct",            50.0),
    ]
    short_prob = float(np.mean(short_signals))

    # ── Medium-term bucket ───────────────────────────────────────────────
    medium_signals = [
        leading.get("curve_phase_pct",       50.0),
        leading.get("copper_gold_pct",        50.0),
        leading.get("credit_impulse_pct",     50.0),
        leading.get("real_rate_pct",          50.0),
        leading.get("reserve_pct",            50.0),
        leading.get("m2_yoy_pct",             50.0),
        leading.get("liq_impulse_13w_pct",    50.0),
    ]
    medium_prob = float(np.mean(medium_signals))

    # ── Coincident conditions (3 orthogonal signals) ────────────────────
    # Each measures a genuinely distinct underlying factor:
    #   fear_score   → VIX-based equity stress (volatility surface)
    #   nfci_signal  → NFCI-based financial conditions (credit/funding)
    #   liq_dir      → current net liquidity direction (Fed plumbing flow)
    #
    # Explicitly excluded from this bucket because they share underlying
    # data with the signals above:
    #   three_puts_score  — contains y10_20 (yield curve, already in short bucket)
    #   liq_anxiety       — contains policy_mistrust (yield curve again)
    #   exhaustion        — contains policy_mistrust (yield curve a 3rd time)
    #   market_index      — contains growth_z (2s10s z-score, yield curve again)
    #
    # The yield curve appeared in all 5 elements of the old bucket.
    # Effective signal count was ~2. Now it is 3 with minimal overlap.
    # nfci_coincident and liq_dir_coincident are pre-computed in render_dashboard
    # and passed as arguments — they reference local series (nfci, net_liq_4w, idx)
    # that are not in scope inside this standalone function.
    coincident_conditions = float(np.mean([
        100 - fear_score,       # VIX / equity stress
        nfci_coincident,        # NFCI financial conditions (credit, funding)
        liq_dir_coincident,     # net liquidity direction: 70=expanding, 30=draining
    ]))

    # ── Cross-bucket blend ────────────────────────────────────────────────
    # Weights reflect empirical lead times — medium-term leads most,
    # tactical is highest-frequency noise.
    # Coincident conditions are a reality check, not a leading indicator.
    BUCKET_W = {
        "tactical":    0.10,
        "short_term":  0.30,
        "medium_term": 0.40,
        "coincident":  0.20,
    }
    raw_composite = (
        BUCKET_W["tactical"]   * tactical_prob  +
        BUCKET_W["short_term"] * short_prob     +
        BUCKET_W["medium_term"]* medium_prob    +
        BUCKET_W["coincident"] * coincident_conditions
    )

    # ── GEX volatility regime adjustment ─────────────────────────────────
    # GEX does NOT vote on direction — it adjusts for volatility regime.
    # Positive gamma → mean-reversion expected → compress toward 50 slightly
    # Negative gamma → amplification expected → expand away from 50 slightly
    gex_adjustment = 0.0
    if gex_state.regime in (GammaRegime.STRONG_POSITIVE, GammaRegime.POSITIVE):
        # Dampen extremes in positive gamma (moves are suppressed)
        gex_adjustment = (50.0 - raw_composite) * 0.08 * gex_state.regime_stability
    elif gex_state.regime in (GammaRegime.NEGATIVE, GammaRegime.STRONG_NEGATIVE):
        # Amplify signal in negative gamma (moves are amplified)
        gex_adjustment = (raw_composite - 50.0) * 0.08 * gex_state.regime_stability

    raw = float(np.clip(raw_composite + gex_adjustment, 5, 95))

    # ── Geo drag ──────────────────────────────────────────────────────────
    geo_drag = float(geo_shock / 100 * 15)   # max 15pt (was 25pt — geo scoring is noisy)

    # ── Regime uncertainty compression ───────────────────────────────────
    unc        = regime_change_p / 100
    compressed = 50.0 + (raw - 50.0) * (1.0 - unc * 0.45) - geo_drag
    bull_prob  = float(np.clip(compressed, 5, 95))

    # ── Honest uncertainty band ───────────────────────────────────────────
    # Effective independent groups: ~3-4 (not 14 signals)
    # Base uncertainty ±10pp + regime + geo widening
    base_unc     = 10.0
    regime_extra = unc * 5.0
    geo_extra    = (geo_shock / 100) * 3.0
    unc_band     = round(base_unc + regime_extra + geo_extra, 1)
    bull_lo      = float(np.clip(bull_prob - unc_band, 5, 95))
    bull_hi      = float(np.clip(bull_prob + unc_band, 5, 95))

    # ── Per-horizon Kelly fractions ───────────────────────────────────────
    # Each horizon paired with its natural payoff assumption:
    #   Tactical (5D):    mean-reversion trades, typical R:R ~1.3 (Setup 5 pin)
    #   Short-term (21D): directional with stop, typical R:R ~2.0 (Setup 1/2)
    #   Medium-term (63D): swing position, typical R:R ~2.5 (Setup 3/4)
    # Half-Kelly throughout for position sizing discipline.
    kelly_5d  = kelly(tactical_prob,  payoff=1.3) * 0.5
    kelly_21d = kelly(short_prob,     payoff=2.0) * 0.5
    kelly_63d = kelly(medium_prob,    payoff=2.5) * 0.5

    return {
        # ── Per-horizon probabilities (the primary outputs) ───────────────
        "prob_5d":       tactical_prob,
        "prob_21d":      short_prob,
        "prob_63d":      medium_prob,
        "coincident":    coincident_conditions,
        # ── Uncertainty-bounded composite (for display only) ─────────────
        "bull_prob":     bull_prob,
        "bull_rounded":  round(bull_prob / 5) * 5,
        "bull_lo":       bull_lo,
        "bull_hi":       bull_hi,
        "uncertainty":   unc_band,
        "bear_prob":     100 - bull_prob,
        # ── Per-horizon Kelly fractions ───────────────────────────────────
        "kelly_5d":      kelly_5d,
        "kelly_21d":     kelly_21d,
        "kelly_63d":     kelly_63d,
        "half_kelly":    kelly_21d,   # kept for legacy UI references
        # ── Forward-looking vs coincident (renamed from leading/lagging) ──
        "fwd_prob":      float(np.mean([short_prob, medium_prob])),
        "coincident_prob": coincident_conditions,
        # Legacy aliases so WIM / other UI code doesn't break
        "leading_prob":  float(np.mean([short_prob, medium_prob])),
        "lagging_prob":  coincident_conditions,
        "tactical_prob": tactical_prob,
        "short_prob":    short_prob,
        "medium_prob":   medium_prob,
        "group_scores": {
            "tactical":   tactical_prob,
            "short_term": short_prob,
            "medium":     medium_prob,
            "coincident": coincident_conditions,
        },
        # ── Adjustments ───────────────────────────────────────────────────
        "gex_adjustment": gex_adjustment,
        "geo_drag":       geo_drag,
        # ── Cross-horizon divergence ──────────────────────────────────────
        "asymmetry":     abs(bull_prob - 50) / 50 * 100,
        "divergent":     abs(tactical_prob - medium_prob) > 20,
        "divergence_gap":abs(tactical_prob - medium_prob),
        "_precision_note": (
            f"3-horizon model: 5D={tactical_prob:.0f}% | "
            f"21D={short_prob:.0f}% | 63D={medium_prob:.0f}%. "
            f"Composite range: {bull_lo:.0f}–{bull_hi:.0f}%."
        ),
    }

# ============================================================
# SESSION CONTEXT
# ============================================================
def get_session_context() -> Dict:
    now_et = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=4)
    h, m = now_et.hour, now_et.minute
    total_mins = h * 60 + m
    weekday = now_et.weekday()

    sessions = [
        (0,   9*60+30,  "Globex",      "Very Low",    0.0,  "Monitor only — do NOT trade gamma setups"),
        (9*60+30, 9*60+45, "RTH Open",  "Very High",  0.0,  "Opening rotation — do NOT trade (noise > signal)"),
        (9*60+45, 10*60+30,"IB Forming","High",        0.5,  "First cautious entries possible if confirmation is strong"),
        (10*60+30,12*60,   "Morning",   "High/Stable", 1.0,  "PRIME TIME — deploy all setups at full regime-adjusted size"),
        (12*60,   14*60,   "Midday",    "Reduced",     0.35, "Reduce size 50%+ — thin book, false signals common"),
        (14*60,   15*60,   "Afternoon", "Increasing",  0.65, "Reassess 0DTE gamma — selective entries only"),
        (15*60,   16*60,   "Close/MOC", "Very High",   0.25, "MOC flow overrides gamma — pin trades only after 15:30"),
        (16*60,   24*60,   "Post-RTH",  "Low",         0.0,  "No gamma setups — monitor only"),
    ]
    for start_m, end_m, name, liq, mult, note in sessions:
        if start_m <= total_mins < end_m:
            is_opex_friday = (weekday == 4)
            is_data_day = False  # Could extend with calendar API
            return {
                "window": name, "liquidity": liq, "size_mult": mult,
                "note": note, "time_et": now_et.strftime("%H:%M ET"),
                "is_opex_friday": is_opex_friday,
                "is_data_day": is_data_day,
                "prime_time": name == "Morning",
            }
    return {"window": "Unknown", "liquidity": "Unknown", "size_mult": 0.5,
            "note": "Cannot determine session", "time_et": now_et.strftime("%H:%M ET"),
            "is_opex_friday": False, "is_data_day": False, "prime_time": False}

# ============================================================
# TRADE SETUP EVALUATOR (5 setups from note 07)
# ============================================================
SETUPS = {
    1: ("Gamma Support Bounce",    "Positive regime. Price at GEX support from above. Dealers must buy.", 0.55, 1.0),
    2: ("Gamma Resistance Fade",   "Positive regime. Price at GEX resistance from below. Dealers must sell.", 0.52, 1.0),
    3: ("Gamma Flip Breakout",     "Price crossing gamma flip. Regime transition — amplified directional move.", 0.45, 0.75),
    4: ("Exhaustion Reversal",     "Extended negative gamma move. Volume climax + delta divergence = reversal.", 0.40, 0.50),
    5: ("0DTE Gamma Pin",          "Late session. Dominant 0DTE strike = gravitational pin. Fade deviations.", 0.65, 1.0),
}

def evaluate_setups(gex_state: GammaState, session: Dict, spot: float,
                    fear_score: float, vix_level: float) -> List[Dict]:
    results = []
    regime = gex_state.regime
    flip = gex_state.gamma_flip
    dist_pct = gex_state.distance_to_flip_pct
    stability = gex_state.regime_stability
    size_mult = session["size_mult"]
    vol_adj = 0.75 if vix_level > 25 else (0.5 if vix_level > 35 else 1.0)

    near_flip = abs(dist_pct) < 0.75
    at_support = any(abs(spot - s) / spot < 0.003 for s in gex_state.key_support)
    at_resistance = any(abs(spot - r) / spot < 0.003 for r in gex_state.key_resistance)
    late_session = session["window"] in ("Afternoon", "Close/MOC")
    prime_time = session["prime_time"]

    for setup_num, (name, desc, base_hr, base_size) in SETUPS.items():
        active = False
        score = SetupScore()
        note = ""

        if setup_num == 1:
            active = (regime in (GammaRegime.STRONG_POSITIVE, GammaRegime.POSITIVE) and at_support)
            score.gamma_alignment = 0.9 if regime == GammaRegime.STRONG_POSITIVE else 0.75
            score.orderflow_confirmation = 0.5  # Needs real-time confirmation
            score.tpo_context = 0.6
            score.level_freshness = 0.9
            score.event_risk = 0.8 if not session["is_data_day"] else 0.2
            note = "WAIT for absorption on footprint + delta flip before entry"

        elif setup_num == 2:
            active = (regime in (GammaRegime.STRONG_POSITIVE, GammaRegime.POSITIVE) and at_resistance)
            score.gamma_alignment = 0.85 if regime == GammaRegime.STRONG_POSITIVE else 0.70
            score.orderflow_confirmation = 0.5
            score.tpo_context = 0.6
            score.level_freshness = 0.85
            score.event_risk = 0.8 if not session["is_data_day"] else 0.2
            note = "Marginally lower hit rate than Setup 1 due to structural long bias"

        elif setup_num == 3:
            active = near_flip and session["window"] in ("IB Forming","Morning","Afternoon")
            score.gamma_alignment = 0.8 if near_flip else 0.3
            score.orderflow_confirmation = 0.4  # Need initiative selling confirmation
            score.tpo_context = 0.5
            score.level_freshness = 0.9
            score.event_risk = 0.7 if not session["is_data_day"] else 0.15
            note = f"Watch for initiative break through flip ({flip:.0f}). No absorption = regime change."

        elif setup_num == 4:
            neg_regime = regime in (GammaRegime.NEGATIVE, GammaRegime.STRONG_NEGATIVE)
            active = neg_regime and abs(dist_pct) > 2.0
            score.gamma_alignment = 0.7 if neg_regime else 0.2
            score.orderflow_confirmation = 0.3  # Needs exhaustion evidence
            score.tpo_context = 0.5
            score.level_freshness = 0.8
            score.event_risk = 0.7 if not session["is_data_day"] else 0.2
            note = "LOWEST hit rate. 50% size only. Wait for volume climax + delta divergence. 3:1 R:R minimum."

        elif setup_num == 5:
            active = late_session and len(gex_state.key_resistance) > 0
            score.gamma_alignment = 0.8 if late_session else 0.2
            score.orderflow_confirmation = 0.5
            score.tpo_context = 0.6
            score.level_freshness = 0.95
            score.event_risk = 0.9 if not session["is_data_day"] else 0.3
            note = "Only valid after 14:00 ET. Pin strengthens as expiry approaches. Exit before 15:50."

        effective_size = base_size * size_mult * vol_adj
        results.append({
            "setup": setup_num, "name": name, "desc": desc,
            "active": active, "score": score, "est_hit_rate": base_hr,
            "effective_size": effective_size, "note": note,
            "regime_ok": active,
        })
    return results

# ============================================================
# FAILURE MODE CHECKER
# ============================================================
FAILURE_MODES = {
    1: ("Stale GEX Data",          "GEX data may not reflect intraday OI changes. No reaction at levels = dead level."),
    2: ("Gamma Level Crowding",     "Level published by multiple services. Front-running and stop hunts likely."),
    3: ("Regime Transition Mid-Trade","Price approaching gamma flip. Mean-reversion thesis invalidated if flip breaks."),
    4: ("Exogenous Shock Override", "Macro event flow > dealer hedging. GEX levels become irrelevant."),
    5: ("Footprint Spoofing",       "Single large absorption print may be manufactured. Require 3+ bars of confirmation."),
    6: ("Correlation Regime Break", "Cross-asset drive (bonds/FX) overriding options-mechanical flow."),
    7: ("OpEx Regime Decay",        "Expiring options removing gamma. Post-OpEx vacuum — levels weakening."),
    8: ("Delta-Hedging Timing Lag", "Dealers hedge in bursts, not continuously. Level is a zone, not a precise price."),
}

def check_failure_modes(gex_state: GammaState, session: Dict,
                        vix_level: float, is_data_day=False) -> List[Tuple[int, str, str, bool]]:
    """Returns list of (mode_id, name, desc, triggered)"""
    triggered = []
    dist = abs(gex_state.distance_to_flip_pct)

    fm1 = gex_state.data_source == "yfinance" and session["window"] in ("Afternoon","Close/MOC")
    fm3 = dist < 0.5
    fm4 = is_data_day or vix_level > 28
    fm6 = vix_level > 25
    fm7 = session["is_opex_friday"]
    fm8 = True  # Always relevant

    active = {1: fm1, 2: False, 3: fm3, 4: fm4, 5: False, 6: fm6, 7: fm7, 8: fm8}
    return [(k, FAILURE_MODES[k][0], FAILURE_MODES[k][1], active.get(k, False))
            for k in FAILURE_MODES]

# ============================================================
# REGIME TRANSITION PROBABILITY
# ============================================================
def classify_macro_regime_abs(core_cpi_yoy: float, curve_raw_bp: float,
                               gdp_qoq: Optional[float] = None) -> str:
    """
    Classify macro regime using ABSOLUTE economic thresholds, not z-scores.

    Z-score classification (old approach) was wrong because:
    - The boundary at z=0 shifts with the sample window
    - In a sustained bear market, all readings look "normal" relative to recent history
    - Small oscillations around z=0 cause spurious regime flips

    Absolute thresholds:
    - Inflation: CPI YoY vs 3.0% (elevated) and 2.0% (target) thresholds
    - Growth proxy: yield curve level vs 0bp (inversion), not vs its mean
    """
    high_inflation = core_cpi_yoy > 3.0
    inv_curve      = curve_raw_bp < 0

    if not high_inflation and not inv_curve:
        return "Goldilocks"
    elif not high_inflation and inv_curve:
        return "Deflation/Recession"
    elif high_inflation and not inv_curve:
        return "Overheating"
    else:
        return "Stagflation"


def regime_transition_prob(
    macro_regime: str,
    core_cpi_yoy: pd.Series,   # raw CPI series, not z-scored
    curve_raw: pd.Series,      # 2s10s raw bp, not z-scored
    window: int = 63
) -> Dict:
    """
    Estimate P(regime change in next 20 days) using absolute thresholds
    and a simple empirical Markov persistence measure.

    Improvements over the old z-score approach:
    1. Absolute thresholds (CPI>3%, curve<0) — not regime-window-relative
    2. Regime changes only counted when the absolute classification flips,
       not when a z-score oscillates around zero
    3. Serial correlation accounted for via persistence penalty
    4. Reports days in current regime and days since last change
    """
    cpi_s   = _to_1d(core_cpi_yoy).dropna()
    curve_s = _to_1d(curve_raw).dropna()

    if len(cpi_s) < 21 or len(curve_s) < 21:
        return {"current": macro_regime, "p_change_20d": 30.0, "persistence": 0,
                "last_change_days": None}

    # Align on common index
    common = cpi_s.index.intersection(curve_s.index)
    if len(common) < 21:
        return {"current": macro_regime, "p_change_20d": 30.0, "persistence": 0,
                "last_change_days": None}

    cpi_c   = cpi_s.reindex(common)
    curve_c = curve_s.reindex(common)

    # Classify regime at each date using absolute thresholds
    regimes = pd.Series([
        classify_macro_regime_abs(float(cpi_c.iloc[i]), float(curve_c.iloc[i]))
        for i in range(len(common))
    ], index=common)

    current       = regimes.iloc[-1]
    regime_changes = (regimes != regimes.shift(1)).iloc[1:]  # True on change days

    # How long in current regime?
    persistence = 0
    for i in range(len(regimes) - 1, -1, -1):
        if regimes.iloc[i] == current:
            persistence += 1
        else:
            break

    # Days since last regime change
    change_indices = regime_changes[regime_changes].index
    last_change_days = int((regimes.index[-1] - change_indices[-1]).days)                        if len(change_indices) > 0 else None

    # Rolling change rate over trailing window
    recent      = regime_changes.iloc[-window:]
    n_changes   = int(recent.sum())
    change_rate = n_changes / max(window, 1)

    # P(at least one change in next 20 days)
    # Using geometric: P = 1 - (1 - rate)^20, but with serial correlation correction
    # Macro regimes are highly autocorrelated — penalise based on persistence
    # Longer persistence → lower effective rate (momentum)
    momentum_discount = max(0.4, 1.0 - 0.015 * persistence)   # decays to 40% floor
    adj_rate   = change_rate * momentum_discount
    p_change   = float(np.clip((1 - (1 - adj_rate) ** 20) * 100, 0, 88))

    return {
        "current":          current,
        "p_change_20d":     p_change,
        "persistence":      persistence,
        "last_change_days": last_change_days,
        "n_changes_window": n_changes,
    }

# ============================================================
# DRIVER ALERTS
# ============================================================
def driver_alerts(prev, now) -> List[str]:
    alerts = []
    thresholds = {"Fear":8,"Three Puts":8,"Liquidity Anxiety":10,"Exhaustion":10,
                  "Market Index":12,"Bull Prob":8}
    for k, thr in thresholds.items():
        if k in prev and k in now and np.isfinite(prev.get(k,np.nan)) and np.isfinite(now.get(k,np.nan)):
            d = now[k] - prev[k]
            if abs(d) >= thr:
                alerts.append(f"{'↑' if d>0 else '↓'} {k}: {d:+.1f} (thr {thr:.0f})")
    for k in ["Risk Regime","Macro Regime","Bubble","Stealth QE","Section","Overall","GEX Regime"]:
        if k in prev and k in now and prev[k] != now[k]:
            alerts.append(f"⚡ STATE CHANGE → {k}: {prev[k]} ▶ {now[k]}")
    return alerts[:12]

# ============================================================
# PAGES
# ============================================================

def render_dashboard():
    """Main integrated dashboard."""
    st.sidebar.markdown("### Controls")
    start = st.sidebar.date_input("Start", value=dt.date.today()-dt.timedelta(days=730))
    end   = st.sidebar.date_input("End",   value=dt.date.today())
    ticker_tile = st.sidebar.text_input("Ticker Tile", "QQQ").upper().strip()
    cpi_thresh  = st.sidebar.number_input("Core CPI cut threshold", 3.0, step=0.1)
    gex_symbol  = st.sidebar.text_input("GEX Symbol", "SPY")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Live Feed")
    live_enabled = st.sidebar.toggle("Auto refresh", True)
    refresh_sec  = int(st.sidebar.slider("Refresh (s)", 30, 300, 90, 15))
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
    else:
        gex_state = GammaState(data_source="unavailable", timestamp=dt.datetime.now().strftime("%H:%M:%S"))

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
        leading, fear_score, three_puts, liq_anxiety, exhaustion,
        market_index_score, geo_shock, meta["p_change_20d"], gex_state,
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

        # ── PROBABILITY ROW — 1D + three forward horizons ──
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

        ph0, ph1, ph2, ph3 = st.columns(4)

        # ── 1D CARD — primary GEX-driven model ───────────────────────────
        with ph0:
            c1d   = _hcol(p1d)
            lo1d  = prob_1d.get("lo_1d", p1d - 15)
            hi1d  = prob_1d.get("hi_1d", p1d + 15)
            unc1d = prob_1d.get("unc_1d", 15.0)
            dom   = prob_1d.get("dominant_signal", "")
            dom_d = prob_1d.get("dominant_dir", "")
            gex_c = prob_1d.get("gex_confidence", 0.5)
            sess_ok = prob_1d.get("session_valid", True)
            gex_regime_str = gex_state.regime.value
            st.markdown(f"""<div style='background:rgba(16,185,129,0.07);border:1px solid rgba(16,185,129,0.28);
                          border-radius:14px;padding:14px;'>
              <div style='font-family:var(--mono);font-size:9px;letter-spacing:1px;
                          text-transform:uppercase;color:var(--green);margin-bottom:3px;'>
                1-Day Bull Prob <span style='color:var(--dim);'>· GEX-conditioned</span>
              </div>
              <div style='font-size:30px;font-weight:700;color:{c1d};font-family:var(--mono);'>{p1d:.0f}%</div>
              {_hbar(p1d, c1d)}
              <div style='font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:2px;'>
                Range: {lo1d:.0f}–{hi1d:.0f}% · ±{unc1d:.0f}pp
              </div>
              <div style='font-family:var(--mono);font-size:9px;color:var(--dim);margin-top:3px;'>
                35%K: {k1d*100:.0f}% · GEX conf: {gex_c:.2f}
              </div>
              <div style='margin-top:5px;padding-top:5px;border-top:1px solid rgba(255,255,255,0.07);
                          font-size:10px;color:var(--muted);'>
                {gex_regime_str}
              </div>
              <div style='font-size:9px;color:var(--dim);margin-top:2px;'>
                Lead: <b>{dom}</b> ({dom_d})
                {"· ⚠ Thin session" if not sess_ok else ""}
              </div>
            </div>""", unsafe_allow_html=True)

        with ph1:
            c = _hcol(p5d)
            st.markdown(f"""<div class='prob-card'>
              <div class='panel-title'>5-Day <span style='color:var(--dim);font-size:9px;'>(Tactical)</span></div>
              <div style='font-size:28px;font-weight:700;color:{c};font-family:var(--mono);'>{p5d:.0f}%</div>
              {_hbar(p5d, c)}
              <div class='small'>35%K: {k5d*100:.0f}% · R:R~1.3</div>
              <div class='small' style='color:var(--dim);'>VIX TS · DXY 5D</div>
            </div>""", unsafe_allow_html=True)
        with ph2:
            c = _hcol(p21d)
            st.markdown(f"""<div class='prob-card'>
              <div class='panel-title'>21-Day <span style='color:var(--dim);font-size:9px;'>(Short-term)</span></div>
              <div style='font-size:28px;font-weight:700;color:{c};font-family:var(--mono);'>{p21d:.0f}%</div>
              {_hbar(p21d, c)}
              <div class='small'>½K: {k21d*100:.0f}% · R:R~2.0</div>
              <div class='small' style='color:var(--dim);'>HYG/LQD · SC · Liq 4W · ISM</div>
            </div>""", unsafe_allow_html=True)
        with ph3:
            c = _hcol(p63d)
            unc  = prob.get("uncertainty", 10.0)
            bp   = prob["bull_prob"]
            bull_lo = prob.get("bull_lo", bp - unc); bull_hi = prob.get("bull_hi", bp + unc)
            p_ch = meta["p_change_20d"]; persist = meta["persistence"]
            pc_c = "#ef4444" if p_ch>55 else ("#f59e0b" if p_ch>35 else "#10b981")
            st.markdown(f"""<div class='prob-card'>
              <div class='panel-title'>63-Day <span style='color:var(--dim);font-size:9px;'>(Medium-term)</span></div>
              <div style='font-size:28px;font-weight:700;color:{c};font-family:var(--mono);'>{p63d:.0f}%</div>
              {_hbar(p63d, c)}
              <div class='small'>½K: {k63d*100:.0f}% · R:R~2.5</div>
              <div class='small' style='color:var(--dim);'>Curve · Cu/Au · Credit · Real Rate</div>
              <div style='margin-top:5px;padding-top:4px;border-top:1px solid rgba(255,255,255,0.07);
                          font-size:9px;font-family:var(--mono);color:{pc_c};'>
                P(regime Δ 20d): {p_ch:.0f}% · held {persist}d
              </div>
            </div>""", unsafe_allow_html=True)

        # 1D component breakdown
        with st.expander("1-Day Signal Components", expanded=False):
            scores = {
                "VIX Term Structure":   prob_1d.get("score_vts", 50),
                "SPY 5D Momentum":      prob_1d.get("score_mom", 50),
                "Credit/Dollar Micro":  prob_1d.get("score_micro", 50),
                "Yield Curve":          prob_1d.get("score_curve", 50),
                "Net Liquidity":        prob_1d.get("score_liq", 50),
            }
            regime_interp = prob_1d.get("regime_interp", "")
            vts_r  = prob_1d.get("vts_ratio", 1.0)
            spy_r  = prob_1d.get("spy_5d_ret", 0.0)
            cr_sig = prob_1d.get("credit_1d", 0.0)
            dl_sig = prob_1d.get("dollar_1d", 0.0)
            gc_val = prob_1d.get("gex_confidence", 0.5)
            fp_val = prob_1d.get("flip_proximity", 1.0)

            st.markdown(f"**GEX Regime:** {gex_state.regime.value} · {regime_interp}", unsafe_allow_html=True)
            st.markdown(f"**GEX confidence:** {gc_val:.2f} · **Flip proximity multiplier:** {fp_val:.2f}")
            st.markdown(f"**Inputs:** VIX/RVol5D = {vts_r:.2f} · SPY 5D = {spy_r*100:+.1f}% · Credit 1D = {cr_sig*10000:.0f}bps · Dollar 1D = {dl_sig*100:+.2f}%")

            fig_comp = go.Figure(go.Bar(
                x=[v - 50 for v in scores.values()],
                y=list(scores.keys()),
                orientation="h",
                marker_color=["#10b981" if v > 50 else "#ef4444" for v in scores.values()],
                text=[f"{v:.0f}" for v in scores.values()],
                textposition="outside",
            ))
            fig_comp.add_vline(x=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
            st.plotly_chart(plotly_dark(fig_comp, "1D Component Scores (deviation from 50)", 220), use_container_width=True)
            st.caption("1-day AUC realistic expectation: 0.52–0.55. This is a positioning aid, not a prediction. Not investment advice.")

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
                f"Fear {'high' if fear_score>=70 else 'elevated' if fear_score>=55 else 'contained'} ({fear_score:.0f}/100)",
                f"Three Puts {'strong' if three_puts>=65 else 'mixed' if three_puts>=45 else 'weak'} ({three_puts:.0f}/100)",
                f"GEX Regime: {gex_state.regime.value} · dist to flip: {gex_state.distance_to_flip_pct:+.2f}%",
                f"Bubble: {bubble_label} · Mag7 PE: {'N/A' if not np.isfinite(mag7_pe) else f'{mag7_pe:.1f}x'}",
                f"Stealth QE: {stealth_label} · TGA 4W: {tga_4w:+.0f}B",
                f"Liq Anxiety: {'high' if liq_anxiety>=70 else 'elevated' if liq_anxiety>=55 else 'contained'} ({liq_anxiety:.0f})",
                f"Exhaustion: {'elevated' if exhaustion>=60 else 'mild' if exhaustion>0 else '0/100'}",
                f"Fwd-looking {prob['fwd_prob']:.0f}% vs Coincident {prob['coincident_prob']:.0f}% {'⚡ DIVERGENCE' if prob['divergent'] else ''}",
            ]:
                st.markdown(f"- {item}")

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


def render_gex_engine():
    """Deep-dive GEX analysis page."""
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


def render_execution_page():
    """Execution checklist and failure modes."""
    st.markdown("## 📋 Execution Framework")

    session = get_session_context()
    chain_df, spot, source = get_gex_from_yfinance("SPY")
    gs = build_gamma_state(chain_df, spot, source) if chain_df is not None else GammaState()
    vix_df = yf.Ticker("^VIX").history(period="1d")
    vix_level = float(vix_df["Close"].iloc[-1]) if len(vix_df) > 0 else 20.0

    # Pre-trade checklist (from note 07)
    st.markdown("### Pre-Trade Checklist")
    st.markdown("<div class='small'>Run this before every trade. All boxes must be checked or the trade does not exist.</div>", unsafe_allow_html=True)

    items = [
        ("Gamma level identified (current, not yesterday's)", gs.data_source != "unavailable"),
        ("Gamma regime confirmed above/below flip", gs.gamma_flip != 0),
        (f"Session context favorable (current: {session['window']}, mult: {session['size_mult']:.2f}x)",
         session["size_mult"] >= 0.5),
        ("Orderflow confirmation present (absorption/initiative/exhaustion on footprint)", None),  # Manual
        ("Setup maps to one of the 5 defined setups (no guesses)", None),
        ("Invalidation (stop) defined at structural level", None),
        (f"Risk within limits (VIX: {vix_level:.0f} → {'REDUCE 50%' if vix_level>30 else 'REDUCE 25%' if vix_level>22 else 'Normal'})",
         vix_level <= 22),
        ("R:R acceptable (≥2:1 for Setups 1-3, ≥3:1 for Setup 4, ≥1.3:1 for Setup 5)", None),
        (f"No disqualifying macro event within 2h (data day: {session['is_data_day']})",
         not session["is_data_day"]),
        (f"Not in opening 15min or post-15:30 (current: {session['window']})",
         session["window"] not in ("RTH Open", "Close/MOC")),
    ]

    for label, status in items:
        if status is True: icon, cls = "✅", "check-ok"
        elif status is False: icon, cls = "❌", "check-fail"
        else: icon, cls = "☐", "check-warn"
        st.markdown(f"<div class='check-row'><span class='{cls}'>{icon}</span><span>{label}</span></div>",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Failure Mode Detector")
    failure_modes = check_failure_modes(gs, session, vix_level, session["is_data_day"])
    n_triggered = sum(1 for fm in failure_modes if fm[3])
    if n_triggered == 0:
        st.success("No failure modes currently triggered. Framework conditions appear nominal.")
    else:
        st.warning(f"{n_triggered} failure mode(s) active — review before trading.")

    for fid, fname, fdesc, triggered in failure_modes:
        c_card = "alert-card" if triggered else "panel"
        icon = "⚠" if triggered else "✓"
        c_text = "#ef4444" if triggered else "var(--muted)"
        with st.expander(f"{icon} FM{fid}: {fname}", expanded=triggered):
            st.markdown(f"<div class='small'>{fdesc}</div>", unsafe_allow_html=True)
            st.markdown(f"**Status:** {'🔴 TRIGGERED' if triggered else '🟢 Clear'}")

    st.markdown("---")
    st.markdown("### Session Structure Guide")
    session_data = [
        ("Globex (overnight)", "Very Low", "0.0x", "Monitor only — no gamma setups"),
        ("RTH Open (09:30-09:45)", "Very High/Chaotic", "0.0x", "DO NOT TRADE — opening rotation"),
        ("IB Forming (09:45-10:30)", "High", "0.5x", "First cautious entries if strong confirmation"),
        ("Morning (10:30-12:00)", "High/Stable", "1.0x", "PRIME TIME — deploy all setups"),
        ("Midday (12:00-14:00)", "Reduced", "0.35x", "Reduce 50%+ — false signals common"),
        ("Afternoon (14:00-15:00)", "Increasing", "0.65x", "Reassess 0DTE, selective entries"),
        ("Close/MOC (15:00-16:00)", "Very High", "0.25x", "MOC overrides gamma — pin trades only"),
    ]
    df_sess = pd.DataFrame(session_data, columns=["Session","Liquidity","Size Mult","Action"])
    st.dataframe(df_sess, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### Gamma Regime → Setup Matrix")
    matrix_data = [
        ("Strong Positive (>2%)","Balanced/D-profile","1, 2 (bounce/fade)","Breakout trades"),
        ("Positive (0.5-2%)","IB near GEX levels","1, 2","Forcing breakouts"),
        ("Neutral (±0.5%)","Narrow IB","3 (watch for flip)","Full-size at extremes"),
        ("Negative (-0.5 to -2%)","Trending","3 continuation, 4 if exhaustion","Counter-trend fades"),
        ("Strong Negative (>-2%)","Extreme displacement ≥2ATR","4 (exhaustion reversal)","Any continuation"),
        ("High 0DTE, single strike","Late session, near pin","5 (gamma pin)","Directional bets"),
    ]
    df_matrix = pd.DataFrame(matrix_data, columns=["Gamma Regime","Market Structure","Applicable Setups","Avoid"])
    st.dataframe(df_matrix, hide_index=True, use_container_width=True)


def render_schwab_page():
    """Schwab / ThinkorSwim OAuth2 + Supabase token storage."""
    st.markdown("## 📈 Schwab / ThinkorSwim Connection")

    # ── Pre-flight checks ────────────────────────────────────────────────
    missing = []
    if not SCHWAB_AVAILABLE:   missing.append("`pip install schwab-py`")
    if not SUPABASE_AVAILABLE: missing.append("`pip install supabase`")
    if missing:
        st.error("Missing packages: " + " · ".join(missing))

    # ── Connection status card ───────────────────────────────────────────
    client = get_schwab_client()
    if client is not None:
        st.success("✅ Schwab API connected")
        col_t, col_r = st.columns(2)
        with col_t:
            if st.button("🔍 Test — fetch SPY quote", use_container_width=True):
                with st.spinner("Fetching..."):
                    px = schwab_get_spot(client, "SPY")
                if px:
                    st.success(f"SPY: ${px:.2f} ✓")
                else:
                    st.error("Failed — token may have expired. Re-authorise below.")
        with col_r:
            if st.button("🗑 Revoke & re-authorise", use_container_width=True):
                sb = _get_supabase()
                if sb:
                    try: sb.table("schwab_tokens").delete().eq("id","shared").execute()
                    except: pass
                st.session_state.pop("_schwab_token_local", None)
                get_schwab_client.clear()
                st.rerun()
        st.divider()

    # ── Setup tabs ───────────────────────────────────────────────────────
    tab_setup, tab_auth, tab_supabase = st.tabs([
        "1️⃣ Schwab App Setup", "2️⃣ Authorise", "3️⃣ Supabase Setup"
    ])

    with tab_setup:
        st.markdown("""
### Create a Schwab Developer App

1. Go to **[developer.schwab.com](https://developer.schwab.com)**
2. Sign in with your **Schwab brokerage account** (same as thinkorswim)
3. Click **"Create App"**
4. Fill in:
   - App Name: anything (e.g. `GEX Dashboard`)
   - **Callback URL:** your Streamlit app URL + `/` e.g.
     `https://yourapp.streamlit.app/`
     *(for local testing use `https://127.0.0.1`)*
   - API Product: **Trader API - Individual**
5. Submit — individual accounts are usually approved instantly
6. Copy your **App Key** (= Client ID) and **Secret**

Then add these to your Streamlit secrets (`.streamlit/secrets.toml` locally,
or the Secrets panel on share.streamlit.io):

```toml
SCHWAB_CLIENT_ID     = "your-app-key-here"
SCHWAB_CLIENT_SECRET = "your-secret-here"
SCHWAB_REDIRECT_URI  = "https://yourapp.streamlit.app/"
```
""")
        cid_check = _get_secret("SCHWAB_CLIENT_ID")
        if cid_check:
            st.success(f"✅ SCHWAB_CLIENT_ID found in secrets ({cid_check[:8]}...)")
        else:
            st.warning("SCHWAB_CLIENT_ID not in secrets yet")

    with tab_supabase:
        st.markdown("""
### Set up Supabase for token storage

Supabase is a free Postgres database that stores the Schwab OAuth token
so it persists across Streamlit Cloud deploys (Streamlit's filesystem resets
on every redeploy — a local `.json` file would be wiped).

**Step 1 — Create a free Supabase project**
1. Go to **[supabase.com](https://supabase.com)** → New project
2. Note your **Project URL** and **anon/public API key** from
   Project Settings → API

**Step 2 — Create the tokens table** (run in Supabase SQL Editor):
```sql
CREATE TABLE schwab_tokens (
  id      TEXT PRIMARY KEY DEFAULT 'shared',
  token   JSONB NOT NULL,
  updated TIMESTAMPTZ DEFAULT NOW()
);

-- Row-level security: allow the anon key to read/write this table
ALTER TABLE schwab_tokens ENABLE ROW LEVEL SECURITY;
CREATE POLICY "team_access" ON schwab_tokens
  FOR ALL USING (true) WITH CHECK (true);
```

**Step 3 — Add to Streamlit secrets:**
```toml
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "eyJ..."   # anon/public key
```
""")
        sb_url = _get_secret("SUPABASE_URL")
        sb_key = _get_secret("SUPABASE_KEY")
        if sb_url and sb_key:
            sb = _get_supabase()
            if sb:
                st.success("✅ Supabase connected")
            else:
                st.error("Supabase credentials found but connection failed")
        else:
            st.warning("SUPABASE_URL / SUPABASE_KEY not in secrets yet")

    with tab_auth:
        st.markdown("### Authorise with Schwab")

        client_id     = _get_secret("SCHWAB_CLIENT_ID")
        client_secret = _get_secret("SCHWAB_CLIENT_SECRET")
        redirect_uri  = _get_secret("SCHWAB_REDIRECT_URI", "https://127.0.0.1")

        if not client_id or not client_secret:
            st.warning("Add SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET to secrets first (tab 1)")
            return

        if client is not None:
            st.success("Already connected — no need to re-authorise unless you revoked the token above.")
            return

        st.info(f"Redirect URI: `{redirect_uri}`  ← must match your Schwab app exactly")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Step 1 — Generate auth URL**")
            if st.button("🔗 Generate Auth URL", use_container_width=True):
                url = schwab_run_auth_flow(client_id, client_secret, redirect_uri)
                if url and not str(url).startswith("error"):
                    st.session_state["_schwab_auth_url_display"] = url
                else:
                    st.error(f"Failed: {url}")

            stored_url = st.session_state.get("_schwab_auth_url_display","")
            if stored_url:
                st.markdown("**Open this URL in your browser:**")
                st.code(stored_url, language="text")
                st.caption(
                    "Log in with Schwab. You'll be redirected to your callback URL — "
                    "the page may not load. Copy the full URL from the address bar."
                )

        with col2:
            st.markdown("**Step 2 — Paste callback URL**")
            callback = st.text_input(
                "Paste the full redirect URL",
                placeholder="https://yourapp.streamlit.app/?code=ABC&session=...",
                key="schwab_callback_input",
            )
            if st.button("✅ Complete Auth", type="primary", use_container_width=True):
                if not callback.strip():
                    st.warning("Paste the redirect URL first")
                else:
                    with st.spinner("Exchanging token..."):
                        ok, msg = schwab_complete_auth(
                            client_id, client_secret, redirect_uri, callback.strip()
                        )
                    if ok:
                        st.success(f"✅ {msg}")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"Auth failed: {msg}")
                        st.markdown("""
**Common causes:**
- Auth code expired (> 30s) — generate a new URL and try immediately
- Redirect URI mismatch — check it matches your Schwab app exactly
- Wrong Client ID or Secret
- Supabase not configured — token can't be saved
""")


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
