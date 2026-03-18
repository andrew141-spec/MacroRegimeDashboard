# data_loaders.py — FRED, yfinance, options chain data fetching
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
from config import _get_secret, GammaState
from utils import _to_1d, resample_ffill, yf_close
from gex_engine import build_gamma_state

# ── Disk cache for GEX chain — survives rate limits and Streamlit reloads ────
_GEX_CACHE_DIR = tempfile.gettempdir()

def _gex_cache_path(symbol: str) -> str:
    return os.path.join(_GEX_CACHE_DIR, f"gex_chain_{symbol.upper()}.json")

def _save_gex_cache(symbol: str, records: list, spot: float, source: str):
    try:
        with open(_gex_cache_path(symbol), "w") as f:
            json.dump({"chain": records, "spot": spot, "source": source,
                       "saved_at": dt.datetime.now().isoformat()}, f)
    except Exception:
        pass

def _load_gex_cache(symbol: str) -> Tuple[Optional[pd.DataFrame], float, str]:
    try:
        path = _gex_cache_path(symbol)
        if not os.path.exists(path):
            return None, 0.0, ""
        with open(path) as f:
            p = json.load(f)
        age_h = (dt.datetime.now() - dt.datetime.fromisoformat(p["saved_at"])).total_seconds() / 3600
        if age_h > 24:
            return None, 0.0, ""
        chain = pd.DataFrame(p["chain"])
        if chain.empty:
            return None, 0.0, ""
        saved = dt.datetime.fromisoformat(p["saved_at"]).strftime("%a %b %d %H:%M")
        return chain, float(p["spot"]), f"cached (EOD {saved})"
    except Exception:
        return None, 0.0, ""


def _fetch_with_retry(fn, retries=3, delay=2):
    """Call fn() with retries on rate limit errors."""
    for attempt in range(retries):
        try:
            return fn(), None
        except Exception as e:
            err = str(e)
            if "Too Many Requests" in err or "Rate" in err or "429" in err:
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
            return None, err
    return None, "max retries exceeded"


@st.cache_data(ttl=1800)
def get_gex_from_yfinance(symbol="SPY") -> Tuple[Optional[pd.DataFrame], float, str]:
    """Pull option chain from yfinance. Retries on rate limits, falls back to disk cache."""
    today = dt.date.today()
    now_et = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=4)
    total_mins = now_et.hour * 60 + now_et.minute
    market_open = (now_et.weekday() < 5) and (9*60+30) <= total_mins < (16*60)

    try:
        ticker = yf.Ticker(symbol)

        # ── Spot price ────────────────────────────────────────────────────
        spot = None
        hist_result, hist_err = _fetch_with_retry(lambda: ticker.history(period="5d"))
        if hist_result is not None and not hist_result.empty:
            spot = float(hist_result["Close"].iloc[-1])
        if not spot:
            try:
                fi = ticker.fast_info
                spot = float(fi.last_price or fi.previous_close or 0) or None
            except Exception:
                pass
        if not spot:
            spot = 580.0

        source_label = "yfinance (live)" if market_open else "yfinance (EOD)"

        # ── Options chain with retry ──────────────────────────────────────
        exps_result, exps_err = _fetch_with_retry(lambda: ticker.options)
        exps = exps_result or []

        rows = []
        for exp in exps[:10]:
            try:
                exp_dt = dt.datetime.strptime(exp, "%Y-%m-%d").date()
                T = max((exp_dt - today).days / 365.0, 1/365)
                chain_result, _ = _fetch_with_retry(lambda e=exp: ticker.option_chain(e), retries=2, delay=1)
                if chain_result is None:
                    continue
                calls = chain_result.calls[["strike","impliedVolatility","openInterest"]].copy()
                calls.columns = ["strike","iv","call_oi"]
                calls["put_oi"] = 0
                puts  = chain_result.puts[["strike","impliedVolatility","openInterest"]].copy()
                puts.columns = ["strike","iv","put_oi"]
                puts["call_oi"] = 0
                for df_leg in [calls, puts]:
                    df_leg["expiry_T"] = T
                    df_leg["iv"] = df_leg["iv"].fillna(0.20).clip(0.05, 5.0)
                    df_leg[["call_oi","put_oi"]] = df_leg[["call_oi","put_oi"]].fillna(0).astype(int)
                    rows.append(df_leg)
            except Exception:
                pass

        if rows:
            full = pd.concat(rows, ignore_index=True)
            full = full[(full["strike"] > spot * 0.88) & (full["strike"] < spot * 1.12)]
            if not full.empty:
                agg = full.groupby(["strike","expiry_T"]).agg(
                    iv=("iv","mean"), call_oi=("call_oi","sum"), put_oi=("put_oi","sum")
                ).reset_index()
                _save_gex_cache(symbol, agg.to_dict("records"), spot, source_label)
                return agg, float(spot), source_label

        # ── Rate limited or empty — use disk cache ────────────────────────
        cached_chain, cached_spot, cached_label = _load_gex_cache(symbol)
        if cached_chain is not None:
            return cached_chain, float(spot), cached_label

        err_detail = exps_err or hist_err or "empty chain"
        return None, float(spot), f"no data ({err_detail})"

    except Exception as e:
        cached_chain, cached_spot, cached_label = _load_gex_cache(symbol)
        if cached_chain is not None:
            return cached_chain, cached_spot, cached_label
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

