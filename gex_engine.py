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
# GEX ENGINE — GEX + VEX (Vanna) + CEX (Charm)
# ============================================================

@dataclass
class DealerGreeks:
    """Per-strike aggregated dealer Greek exposures."""
    # GEX — reaction to price
    gex_by_strike:   Dict[float, float] = field(default_factory=dict)
    # VEX — reaction to IV changes
    vex_by_strike:   Dict[float, float] = field(default_factory=dict)
    # CEX — reaction to time decay
    cex_by_strike:   Dict[float, float] = field(default_factory=dict)
    # Key nodes (largest absolute value per Greek)
    key_nodes_gex:   List[Tuple[float, float]] = field(default_factory=list)  # (strike, value)
    key_nodes_vex:   List[Tuple[float, float]] = field(default_factory=list)
    key_nodes_cex:   List[Tuple[float, float]] = field(default_factory=list)
    # OTM anchors for weekly bias (strikes far from spot)
    otm_anchors:     List[Tuple[float, float]] = field(default_factory=list)  # (strike, net_gex)
    # Vanna/Charm alignment signal
    vanna_charm_aligned: bool  = False
    vanna_direction:     str   = "neutral"  # pressure direction when IV rising
    vanna_sign:          str   = "neutral"  # "positive", "negative", "neutral" — raw sign
    charm_direction:     str   = "neutral"
    data_source:         str   = "unknown"


def _d1d2(S, K, T, sigma, r=0.05):
    if T <= 0 or sigma <= 0: return 0.0, 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def bs_gamma(S, K, T, sigma, r=0.05) -> float:
    if T <= 0 or sigma <= 0: return 0.0
    d1, _ = _d1d2(S, K, T, sigma, r)
    return scipy_norm.pdf(d1) / (S * sigma * math.sqrt(T))

def bs_vanna(S, K, T, sigma, r=0.05) -> float:
    """Vanna = dDelta/dIV = d2Delta/dSdSigma.
    Measures how dealer delta changes when IV changes.
    Vanna = -phi(d1) * d2 / sigma
    Sign convention (per notes):
    Positive vanna: IV rises → dealers BUY, IV falls → dealers SELL
    Negative vanna: IV rises → dealers SELL, IV falls → dealers BUY
    """
    if T <= 0 or sigma <= 0: return 0.0
    d1, d2 = _d1d2(S, K, T, sigma, r)
    return -scipy_norm.pdf(d1) * d2 / sigma

def bs_charm(S, K, T, sigma, r=0.05) -> float:
    """Charm = dDelta/dTime (delta decay per day).
    Measures how dealer delta changes as time passes.
    Charm = -phi(d1) * (2*r*T - d2*sigma*sqrt(T)) / (2*T*sigma*sqrt(T))
    Sign convention: positive charm → dealer delta increases over time → they buy (upward drift).
    """
    if T <= 0 or sigma <= 0: return 0.0
    d1, d2 = _d1d2(S, K, T, sigma, r)
    return -scipy_norm.pdf(d1) * (2 * r * T - d2 * sigma * math.sqrt(T)) / (2 * T * sigma * math.sqrt(T))

def compute_gex_from_chain(chain: pd.DataFrame, spot: float,
                            multiplier=100, r=0.05) -> pd.DataFrame:
    """
    chain must have: strike, expiry_T, iv, call_oi, put_oi
    GEX sign convention: calls sold by dealers = positive (stabilizing)
                         puts sold by dealers = negative (destabilizing)
    Also computes VEX (vanna) and CEX (charm) with same sign convention.
    """
    chain = chain.copy()
    chain["gamma"] = chain.apply(
        lambda row: bs_gamma(spot, row["strike"], row["expiry_T"], row["iv"], r), axis=1)
    chain["vanna"] = chain.apply(
        lambda row: bs_vanna(spot, row["strike"], row["expiry_T"], row["iv"], r), axis=1)
    chain["charm"] = chain.apply(
        lambda row: bs_charm(spot, row["strike"], row["expiry_T"], row["iv"], r), axis=1)

    # GEX per equation: Σᵢ (Γᵢ · OIᵢ · ContractSize · S²)
    # bs_gamma = φ(d1) / (S·σ·√T), so: bs_gamma * OI * C * S² = φ(d1)*OI*C*S / (σ·√T)
    # This is dollar gamma — the $ move in dealer delta per 1% move in spot.
    # We then divide by 100 to normalise to "per 1-point move" for display.
    # calls: dealers long gamma = stabilizing (+)
    # puts:  dealers short gamma = destabilizing (-)
    S2 = spot * spot / 100.0   # S²/100 — standard GEX normalisation
    chain["call_gex"] =  chain["call_oi"] * chain["gamma"] * multiplier * S2
    chain["put_gex"]  = -chain["put_oi"] * chain["gamma"] * multiplier * S2
    chain["net_gex"]  =  chain["call_gex"] + chain["put_gex"]

    # VEX and CEX use same normalisation
    chain["call_vex"] =  chain["call_oi"] * chain["vanna"] * multiplier * S2
    chain["put_vex"]  = -chain["put_oi"] * chain["vanna"] * multiplier * S2
    chain["net_vex"]  =  chain["call_vex"] + chain["put_vex"]

    chain["call_cex"] =  chain["call_oi"] * chain["charm"] * multiplier * S2
    chain["put_cex"]  = -chain["put_oi"] * chain["charm"] * multiplier * S2
    chain["net_cex"]  =  chain["call_cex"] + chain["put_cex"]

    return chain


def compute_dealer_greeks(chain: pd.DataFrame, spot: float,
                           source: str = "yfinance") -> DealerGreeks:
    """Compute full GEX/VEX/CEX profile, key nodes, OTM anchors, and alignment signals."""
    gex_chain = compute_gex_from_chain(chain, spot)

    # Aggregate by strike (sum across expirations)
    agg = gex_chain.groupby("strike").agg(
        net_gex=("net_gex", "sum"),
        net_vex=("net_vex", "sum"),
        net_cex=("net_cex", "sum"),
    ).reset_index()

    gex_by_strike = dict(zip(agg["strike"], agg["net_gex"]))
    vex_by_strike = dict(zip(agg["strike"], agg["net_vex"]))
    cex_by_strike = dict(zip(agg["strike"], agg["net_cex"]))

    # Key nodes = largest ABSOLUTE value (per Module 2: size matters most)
    def _key_nodes(by_strike: dict, n=5) -> List[Tuple[float, float]]:
        return sorted(by_strike.items(), key=lambda x: abs(x[1]), reverse=True)[:n]

    key_nodes_gex = _key_nodes(gex_by_strike)
    key_nodes_vex = _key_nodes(vex_by_strike)
    key_nodes_cex = _key_nodes(cex_by_strike)

    # OTM anchors: strikes >3% away from spot with significant absolute GEX
    otm_threshold = spot * 0.03
    otm = {k: v for k, v in gex_by_strike.items() if abs(k - spot) > otm_threshold}
    otm_anchors = sorted(otm.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    # Vanna direction: net VEX near spot (within 2%)
    # Convention per notes:
    #   Positive vanna: IV rises → dealers BUY, IV falls → dealers SELL
    #   Negative vanna: IV rises → dealers SELL, IV falls → dealers BUY
    # We store the vanna sign and let the UI interpret based on IV regime.
    # vanna_direction here = pressure direction IF IV IS RISING (most actionable context)
    ntm_vex = sum(v for k, v in vex_by_strike.items() if abs(k - spot) / spot < 0.02)
    # Positive vanna + rising IV = dealers buy = bullish pressure
    # Negative vanna + rising IV = dealers sell = bearish pressure
    vanna_direction = "bullish" if ntm_vex > 0 else ("bearish" if ntm_vex < 0 else "neutral")
    vanna_sign = "positive" if ntm_vex > 0 else ("negative" if ntm_vex < 0 else "neutral")

    # Charm direction: net CEX near spot
    ntm_cex = sum(v for k, v in cex_by_strike.items() if abs(k - spot) / spot < 0.02)
    charm_direction = "bullish" if ntm_cex > 0 else ("bearish" if ntm_cex < 0 else "neutral")

    # Alignment: both pointing same direction
    vanna_charm_aligned = (vanna_direction == charm_direction) and vanna_direction != "neutral"

    return DealerGreeks(
        gex_by_strike=gex_by_strike,
        vex_by_strike=vex_by_strike,
        cex_by_strike=cex_by_strike,
        key_nodes_gex=key_nodes_gex,
        key_nodes_vex=key_nodes_vex,
        key_nodes_cex=key_nodes_cex,
        otm_anchors=otm_anchors,
        vanna_charm_aligned=vanna_charm_aligned,
        vanna_direction=vanna_direction,
        vanna_sign=vanna_sign,
        charm_direction=charm_direction,
        data_source=source,
    )

def find_gamma_flip(chain: pd.DataFrame) -> float:
    """
    Zero-gamma (gamma flip) level.

    Finds the strike where the net GEX profile crosses zero by:
    1. Aggregating net_gex by strike (summing across expirations)
    2. Sorting strikes
    3. Interpolating the zero-crossing between adjacent strikes

    This uses the already-computed net_gex column from compute_gex_from_chain,
    finding where: net_gex(K_i) changes sign between adjacent strikes.
    The interpolated zero is the gamma flip level.
    """
    if "strike" not in chain.columns or len(chain) == 0:
        return np.nan

    # Aggregate by strike (sum across expirations for same strike)
    agg = chain.groupby("strike")["net_gex"].sum().reset_index()
    agg = agg.sort_values("strike").reset_index(drop=True)

    strikes = agg["strike"].values
    gex_vals = agg["net_gex"].values

    # Find all zero crossings (sign changes between adjacent strikes)
    signs = gex_vals[:-1] * gex_vals[1:]
    crossings = np.where(signs < 0)[0]

    if len(crossings) == 0:
        return np.nan

    # Pick the zero crossing closest to the current spot (most actionable flip)
    # We don't have spot here so use the midpoint of the strike range as proxy,
    # but caller should pass spot — for now use the crossing with largest
    # adjacent GEX magnitude (most dealer flow around it = most meaningful flip)
    crossing_mids = np.array([(strikes[i] + strikes[i+1]) / 2.0 for i in crossings])
    crossing_mag  = np.array([abs(gex_vals[i]) + abs(gex_vals[i+1]) for i in crossings])
    best_i = crossings[np.argmax(crossing_mag)]

    # Linear interpolation between the two bracketing strikes
    s1, s2 = strikes[best_i], strikes[best_i + 1]
    g1, g2 = gex_vals[best_i], gex_vals[best_i + 1]
    if (g2 - g1) == 0:
        return float(s1)
    return float(s1 + (s2 - s1) * (-g1) / (g2 - g1))


def find_local_extrema(chain: pd.DataFrame) -> Dict[str, List[Tuple[float, float]]]:
    """
    Find local maxima and minima of the per-strike net GEX profile.

    Local maxima = GEX peaks (strongest resistance walls / dealer selling zones)
    Local minima = GEX troughs (strongest support walls / dealer amplification zones)
    Zero crossings = gamma flip candidates

    Returns dict with keys: 'maxima', 'minima', 'zero_crossings'
    Each value is a list of (strike, net_gex) tuples sorted by abs(gex) descending.
    """
    if "strike" not in chain.columns or len(chain) == 0:
        return {"maxima": [], "minima": [], "zero_crossings": []}

    agg = chain.groupby("strike")["net_gex"].sum().reset_index()
    agg = agg.sort_values("strike").reset_index(drop=True)
    strikes = agg["strike"].values
    gex_vals = agg["net_gex"].values
    n = len(strikes)

    maxima, minima, zero_crossings = [], [], []

    for i in range(1, n - 1):
        prev, curr, nxt = gex_vals[i-1], gex_vals[i], gex_vals[i+1]
        if curr > prev and curr > nxt:
            maxima.append((float(strikes[i]), float(curr)))
        elif curr < prev and curr < nxt:
            minima.append((float(strikes[i]), float(curr)))

    # Zero crossings via interpolation
    sign_changes = np.where(gex_vals[:-1] * gex_vals[1:] < 0)[0]
    for i in sign_changes:
        s1, s2 = strikes[i], strikes[i+1]
        g1, g2 = gex_vals[i], gex_vals[i+1]
        if (g2 - g1) != 0:
            zc = s1 + (s2 - s1) * (-g1) / (g2 - g1)
            zero_crossings.append((float(zc), 0.0))

    # Sort by absolute GEX magnitude descending
    maxima.sort(key=lambda x: -abs(x[1]))
    minima.sort(key=lambda x: -abs(x[1]))

    return {"maxima": maxima, "minima": minima, "zero_crossings": zero_crossings}

def find_local_extrema(gex_by_strike: Dict[float, float],
                       min_prominence: float = 0.0
                       ) -> Tuple[List[Tuple[float,float]], List[Tuple[float,float]]]:
    """
    Find local maxima and minima in the per-strike GEX profile.
    A local maximum is a strike whose GEX value is greater than both neighbours.
    A local minimum is a strike whose GEX value is less than both neighbours.
    Returns (maxima, minima) each as list of (strike, gex_value).
    """
    if len(gex_by_strike) < 3:
        return [], []
    strikes = sorted(gex_by_strike.keys())
    vals    = [gex_by_strike[s] for s in strikes]
    maxima, minima = [], []
    for i in range(1, len(strikes) - 1):
        v_prev, v, v_next = vals[i-1], vals[i], vals[i+1]
        if v > v_prev and v > v_next:
            maxima.append((strikes[i], v))
        elif v < v_prev and v < v_next:
            minima.append((strikes[i], v))
    # Sort by absolute magnitude — largest effects first
    maxima.sort(key=lambda x: abs(x[1]), reverse=True)
    minima.sort(key=lambda x: abs(x[1]), reverse=True)
    return maxima, minima


def find_zero_crossings(gex_by_strike: Dict[float, float]) -> List[float]:
    """
    Find strikes where per-strike net GEX crosses zero (sign changes).
    These are local transition points — where individual strike gamma
    flips from stabilizing to destabilizing (or vice versa).
    Distinct from the gamma flip (total GEX = 0).
    Uses linear interpolation between adjacent strikes.
    """
    if len(gex_by_strike) < 2:
        return []
    strikes = sorted(gex_by_strike.keys())
    vals    = [gex_by_strike[s] for s in strikes]
    crossings = []
    for i in range(len(strikes) - 1):
        v1, v2 = vals[i], vals[i+1]
        if v1 * v2 < 0:  # sign change
            s1, s2 = strikes[i], strikes[i+1]
            # Linear interpolation: x where v1 + (v2-v1)*(x-s1)/(s2-s1) = 0
            x = s1 + (-v1) * (s2 - s1) / (v2 - v1)
            crossings.append(float(x))
    return crossings


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

    # Aggregate net_gex by strike (sum across expirations)
    by_strike_series = gex_chain.groupby("strike")["net_gex"].sum()
    by_strike = by_strike_series.to_dict()

    top_support    = gex_chain[gex_chain["net_gex"] < 0].nsmallest(5, "net_gex")["strike"].tolist()
    top_resistance = gex_chain[gex_chain["net_gex"] > 0].nlargest(5, "net_gex")["strike"].tolist()

    # Local structure
    local_maxima, local_minima = find_local_extrema(by_strike)
    zero_crossings = find_zero_crossings(by_strike)

    return GammaState(
        regime=regime, gamma_flip=float(flip) if np.isfinite(flip) else 0.0,
        distance_to_flip_pct=dist, total_gex=float(gex_chain["net_gex"].sum()),
        gex_by_strike=by_strike, key_support=top_support, key_resistance=top_resistance,
        regime_stability=stability, data_source=source,
        timestamp=dt.datetime.now().strftime("%H:%M:%S"),
        local_maxima=local_maxima[:5],
        local_minima=local_minima[:5],
        zero_crossings=zero_crossings,
    )

# ============================================================
# IBKR CONNECTION
# ============================================================
# ============================================================
