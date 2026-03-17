# gex_engine.py — GEX computation: gamma per strike, flip detection, regime
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
from config import GammaState, GammaRegime, SetupScore
from utils import _to_1d, _bs_iv_from_price

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
