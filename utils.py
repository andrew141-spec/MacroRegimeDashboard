# utils.py — core math and data utilities
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

def _bs_iv_from_price(S: float, K: float, T: float, option_price: float,
                      right: str, r: float = 0.05) -> float:
    """Newton-Raphson Black-Scholes implied volatility solver."""
    if option_price <= 0.01 or T <= 0:
        return 0.20
    intrinsic = max(S - K, 0) if right == "C" else max(K - S, 0)
    if option_price <= intrinsic:
        return 0.20
    sigma = 0.20
    for _ in range(50):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if right == "C":
            price = S * scipy_norm.cdf(d1) - K * math.exp(-r * T) * scipy_norm.cdf(d2)
        else:
            price = K * math.exp(-r * T) * scipy_norm.cdf(-d2) - S * scipy_norm.cdf(-d1)
        vega = S * scipy_norm.pdf(d1) * math.sqrt(T)
        if vega < 1e-8:
            break
        diff  = price - option_price
        sigma -= diff / vega
        sigma  = max(0.001, min(sigma, 10.0))
        if abs(diff) < 1e-5:
            break
    return float(np.clip(sigma, 0.01, 5.0))

# ============================================================
# HELPERS — UI
# ============================================================
