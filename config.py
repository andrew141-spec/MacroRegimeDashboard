# config.py — imports, page config, CSS, dataclasses, sidebar nav
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
    page_title="Regime Dashboard",
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
  --bg0:    #000000;
  --bg1:    #040404;
  --bg2:    #080808;
  --panel:  #050505;
  --panel2: #090909;
  --border: #1a1a1a;

  --text:   rgba(255,255,255,0.88);
  --muted:  rgba(255,255,255,0.50);
  --dim:    rgba(255,255,255,0.24);

  --green:   #22c55e;
  --teal:    #16a34a;
  --yellow:  #eab308;
  --orange:  #f97316;
  --red:     #ef4444;
  --crimson: #dc2626;
  --purple:  #8b5cf6;
  --blue:    #3b82f6;
  --sky:     #38bdf8;

  --gex-pos: #22c55e;
  --gex-neg: #ef4444;
  --gex-flip:#eab308;

  --mono: 'Space Mono', ui-monospace, monospace;
  --sans: 'DM Sans', system-ui, sans-serif;

  --shadow-lg: none;
  --shadow-sm: none;
  --glow-green: none;
  --glow-red: none;
  --glow-blue: none;
}

html, body, [class*="css"] {
  font-family: var(--sans) !important;
  background:
    linear-gradient(rgba(34,197,94,0.045), rgba(34,197,94,0)) top/100% 1px no-repeat,
    linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 55%, var(--bg2) 100%) !important;
  color: var(--text) !important;
  min-height: 100vh;
}

.stApp {
  background:
    linear-gradient(rgba(34,197,94,0.045), rgba(34,197,94,0)) top/100% 1px no-repeat,
    #000 !important;
}

.block-container {
  padding-top: 0.35rem !important;
  padding-bottom: 0.75rem !important;
  max-width: 100% !important;
}

/* keep sidebar toggle visible */
#MainMenu, footer { visibility: hidden; }
header {
  background: transparent !important;
}

/* general text */
h1, h2, h3, h4, h5, h6 {
  font-family: var(--sans) !important;
  letter-spacing: -0.2px;
  color: var(--text) !important;
}
.small { font-size: 11px; color: var(--muted); }
.mono  { font-family: var(--mono); }
hr { border-color: rgba(255,255,255,0.07) !important; }

/* panels */
.panel {
  background: linear-gradient(180deg, var(--panel2), var(--panel));
  border: 1px solid var(--border);
  border-radius: 0;
  padding: 10px 12px 10px;
  box-shadow: none;
  backdrop-filter: none;
  transition: border-color 120ms ease, background 120ms ease;
}
.panel:hover {
  border-color: rgba(34,197,94,0.35);
  transform: none;
  background: #080808;
}
.panel-title {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 1.8px;
  text-transform: uppercase;
  color: var(--green);
  margin: 0 0 8px 0;
}

/* cards */
.prob-card,
.gex-card,
.warn-card,
.alert-card,
.setup-card,
.api-card,
.failure-card {
  background: #050505;
  border: 1px solid #1a1a1a;
  border-radius: 0;
  padding: 10px 12px;
  box-shadow: none;
}

.prob-card    { border-left: 1px solid #3b82f6; }
.gex-card     { border-left: 1px solid #22c55e; }
.warn-card    { border-left: 1px solid #eab308; }
.alert-card   { border-left: 1px solid #ef4444; }
.setup-card   { border-left: 1px solid #8b5cf6; }
.api-card     { border-left: 1px solid #06b6d4; }
.failure-card { border-left: 1px solid #dc2626; }

/* badges */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 8px;
  border-radius: 0;
  border: 1px solid var(--border);
  background: #090909;
  color: var(--muted);
  font-size: 10px;
  font-family: var(--mono);
}
.badge b {
  color: var(--text);
  font-weight: 700;
}

/* regime chip */
.regime-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 0;
  font-family: var(--mono);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  border: 1px solid currentColor;
}

/* probability bar */
.pbar-wrap {
  background: rgba(255,255,255,0.06);
  border-radius: 0;
  height: 4px;
  width: 100%;
  margin: 3px 0 1px;
}
.pbar-fill {
  border-radius: 0;
  height: 4px;
  transition: width 400ms ease;
}
.pbar-label {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
}

/* gex rows */
.gex-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 3px 0;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  font-family: var(--mono);
  font-size: 10.5px;
}
.gex-row:last-child { border-bottom: none; }
.gex-pos { color: var(--gex-pos); }
.gex-neg { color: var(--gex-neg); }
.gex-flip-marker { color: var(--gex-flip); font-weight: 700; }

/* setup score */
.score-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 0;
  margin-right: 5px;
}

/* checklist */
.check-row {
  display: flex;
  align-items: flex-start;
  gap: 7px;
  padding: 2px 0;
  font-size: 11px;
  line-height: 1.35;
}
.check-ok   { color: var(--green);  font-family: var(--mono); }
.check-warn { color: var(--yellow); font-family: var(--mono); }
.check-fail { color: var(--red);    font-family: var(--mono); }

/* terminal log */
.term {
  font-family: var(--mono);
  font-size: 10.5px;
  line-height: 1.45;
  background: #030303;
  border: 1px solid #1a1a1a;
  border-radius: 0;
  padding: 8px 10px;
  max-height: 280px;
  overflow-y: auto;
}
.term-row { padding: 2px 0; }
.term-ts   { color: var(--dim); margin-right: 8px; }
.term-hi   { color: var(--sky); }
.term-ok   { color: var(--green); }
.term-warn { color: var(--yellow); }
.term-err  { color: var(--red); }

/* streamlit metric cards */
div[data-testid="stMetric"] {
  background: #050505 !important;
  border: 1px solid #1a1a1a !important;
  border-radius: 0 !important;
  padding: 8px 10px !important;
  box-shadow: none !important;
}
div[data-testid="stMetricValue"] {
  font-family: var(--mono) !important;
}
div[data-testid="stMetricLabel"] {
  font-family: var(--mono) !important;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  color: var(--muted) !important;
}

/* buttons */
.stButton > button,
.stDownloadButton > button {
  width: 100% !important;
  background: #090909 !important;
  color: var(--green) !important;
  border: 1px solid rgba(34,197,94,0.38) !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  font-family: var(--mono) !important;
  font-size: 10px !important;
  letter-spacing: 1.8px !important;
  text-transform: uppercase !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
  background: rgba(34,197,94,0.08) !important;
  border-color: rgba(34,197,94,0.60) !important;
}

/* inputs */
.stTextInput input,
.stTextArea textarea,
.stSelectbox > div > div {
  background: #080808 !important;
  color: var(--text) !important;
  border: 1px solid #1a1a1a !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  font-family: var(--mono) !important;
  font-size: 11px !important;
}

/* sidebar */
[data-testid="stSidebar"] {
  background: #000 !important;
  border-right: 1px solid #1a1a1a !important;
}
[data-testid="stSidebar"] * {
  color: var(--text) !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea,
[data-testid="stSidebar"] .stSelectbox > div > div {
  background: #080808 !important;
  border: 1px solid #1a1a1a !important;
  border-radius: 0 !important;
  font-family: var(--mono) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
  font-family: var(--mono) !important;
}

/* section header */
.section-hdr {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 14px 0 8px;
}
.section-hdr::after {
  content: '';
  flex: 1;
  height: 1px;
  background: linear-gradient(to right, rgba(34,197,94,0.28), transparent);
}
.section-hdr span {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 1.6px;
  text-transform: uppercase;
  color: var(--green);
  white-space: nowrap;
}

/* tables */
table {
  border-collapse: collapse !important;
}
th, td {
  border-color: #1a1a1a !important;
}

/* scrollbars */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #020202; }
::-webkit-scrollbar-thumb {
  background: #1f1f1f;
  border-radius: 0;
}
::-webkit-scrollbar-thumb:hover {
  background: #2b2b2b;
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

# Operational labels: what dealers are actually doing in each regime.
# Used for trader-facing display instead of the abstract +/- labels.
REGIME_OPERATIONAL_LABEL = {
    GammaRegime.STRONG_POSITIVE: "STRONG_BUY_DIPS",
    GammaRegime.POSITIVE:        "DEALERS_BUY_DIPS",
    GammaRegime.NEUTRAL:         "NEAR_FLIP",
    GammaRegime.NEGATIVE:        "DEALERS_SELL_RALLIES",
    GammaRegime.STRONG_NEGATIVE: "STRONG_SELL_RALLIES",
}

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

# page is set in app.py — imported from there
page = None  # placeholder; overridden by app.py before any render call

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
