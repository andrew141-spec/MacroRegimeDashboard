# gex_engine.py — GEX computation: gamma per strike, flip detection, regime, advanced analytics
import math, datetime as dt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm
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
    # Prefer Schwab's own model gamma when available (non-zero, finite).
    # Schwab's gamma accounts for their vol surface assumptions and is more
    # accurate than our flat-vol BS approximation.
    # Fall back to BS gamma for yfinance data which has no per-contract gamma.
    has_schwab_gamma = (
        "schwab_gamma" in chain.columns
        and chain["schwab_gamma"].notna().any()
        and (chain["schwab_gamma"].abs() > 0).any()
    )
    if has_schwab_gamma:
        # Schwab gamma is always positive (it's the absolute rate of delta change).
        # Sign is handled downstream via call_oi (+) and put_oi (-) convention.
        # Use vectorised operation for performance instead of row-wise apply.
        bs_g = chain.apply(
            lambda row: bs_gamma(spot, row["strike"], row["expiry_T"], row["iv"], r), axis=1)
        schwab_valid = (
            chain["schwab_gamma"].notna() &
            chain["schwab_gamma"].apply(np.isfinite) &
            (chain["schwab_gamma"] > 0)
        )
        chain["gamma"] = np.where(schwab_valid, chain["schwab_gamma"], bs_g)
    else:
        chain["gamma"] = chain.apply(
            lambda row: bs_gamma(spot, row["strike"], row["expiry_T"], row["iv"], r), axis=1)
    chain["vanna"] = chain.apply(
        lambda row: bs_vanna(spot, row["strike"], row["expiry_T"], row["iv"], r), axis=1)
    chain["charm"] = chain.apply(
        lambda row: bs_charm(spot, row["strike"], row["expiry_T"], row["iv"], r), axis=1)

    # GEX per the equation: Σ(Γᵢ · OIᵢ · ContractSize · S²)
    # calls: dealers long gamma = stabilizing (+)
    # puts:  dealers short gamma = destabilizing (-)
    # S² = spot² — the dollar gamma term that gives GEX its units of $ per 1% move
    chain["call_gex"] =  chain["call_oi"] * chain["gamma"] * multiplier * (spot ** 2)
    chain["put_gex"]  = -chain["put_oi"] * chain["gamma"] * multiplier * (spot ** 2)
    chain["net_gex"]  =  chain["call_gex"] + chain["put_gex"]

    # VEX: positive vanna + rising IV → dealers buy (bullish pressure on IV spike)
    #      positive vanna + falling IV → dealers sell (bearish on vol crush)
    chain["call_vex"] =  chain["call_oi"] * chain["vanna"] * multiplier * (spot ** 2)
    chain["put_vex"]  = -chain["put_oi"] * chain["vanna"] * multiplier * (spot ** 2)
    chain["net_vex"]  =  chain["call_vex"] + chain["put_vex"]

    # CEX: positive charm → time decay forces dealers to buy (upward drift)
    chain["call_cex"] =  chain["call_oi"] * chain["charm"] * multiplier * (spot ** 2)
    chain["put_cex"]  = -chain["put_oi"] * chain["charm"] * multiplier * (spot ** 2)
    chain["net_cex"]  =  chain["call_cex"] + chain["put_cex"]

    return chain


def compute_dealer_greeks(chain: pd.DataFrame, spot: float,
                           source: str = "yfinance", max_dte: int = 45) -> DealerGreeks:
    """Compute full GEX/VEX/CEX profile, key nodes, OTM anchors, and alignment signals."""
    near_chain = chain[chain["expiry_T"] <= max_dte / 365.0].copy()
    if near_chain.empty:
        near_chain = chain.copy()
    gex_chain = compute_gex_from_chain(near_chain, spot)

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
    Gamma flip = the price where per-strike net GEX transitions from negative
    to positive (or vice versa) — i.e. where the bars cross zero.

    Method: aggregate by strike (sum across all expirations), then find the
    adjacent pair of strikes where net_gex changes sign and interpolate.

    Fallback: if net_gex is the same sign everywhere (common when put OI
    heavily dominates, as in SPY), return the strike closest to zero — this
    is the "vol trigger" level used by SpotGamma and GEXBot.

    NOTE: the cumulative sum method was removed because it fails whenever
    put OI dominates (the cumsum stays negative throughout, returning nan
    even though there is a clear per-strike zero-crossing in the bars).
    """
    # Aggregate by strike across all expirations
    by_strike = (chain.groupby("strike")["net_gex"]
                      .sum()
                      .reset_index()
                      .sort_values("strike")
                      .reset_index(drop=True))

    if len(by_strike) < 2:
        return np.nan

    vals    = by_strike["net_gex"].values
    strikes = by_strike["strike"].values

    # Primary: find where per-strike net_gex changes sign (bar crosses zero)
    bar_sign_changes = np.where(vals[:-1] * vals[1:] < 0)[0]

    if len(bar_sign_changes) > 0:
        # If multiple sign changes, pick the one closest to ATM (largest |GEX| transition)
        best_i = bar_sign_changes[
            np.argmax([abs(vals[j+1] - vals[j]) for j in bar_sign_changes])
        ]
        s1, s2 = float(strikes[best_i]),  float(strikes[best_i + 1])
        g1, g2 = float(vals[best_i]),     float(vals[best_i + 1])
        return s1 + (s2 - s1) * (-g1) / (g2 - g1) if (g2 - g1) != 0 else s1

    # Fallback: no sign change — all bars same sign (heavy put or call skew).
    # Return the strike with net_gex closest to zero. This is still a valid
    # "vol trigger" level — the point of minimum dealer net gamma.
    nearest_idx = int(np.argmin(np.abs(vals)))
    return float(strikes[nearest_idx])

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

def nearest_expiry_chain(chain: pd.DataFrame) -> pd.DataFrame:
    """Return only the nearest expiration — for 0DTE-style bar chart display."""
    if chain is None or chain.empty:
        return chain
    min_T = chain["expiry_T"].min()
    return chain[chain["expiry_T"] <= min_T + 1/365].copy()


def build_gamma_state(chain: pd.DataFrame, spot: float, source: str = "yfinance",
                      max_dte: int = 45) -> GammaState:
    """
    Build gamma state from options chain.
    max_dte: only include expirations within this many days (default 45).
    Far-dated options have negligible gamma and distort the flip level.
    SpotGamma/GEXBot use near-term OI only for spot GEX calculations.
    """
    import datetime as _dt
    # Filter to near-term expirations before GEX computation
    # expiry_T is in years; 45 DTE = 45/365 ≈ 0.123
    near_chain = chain[chain["expiry_T"] <= max_dte / 365.0].copy()
    if near_chain.empty:
        near_chain = chain.copy()  # fallback: use all if filter removes everything
    gex_chain = compute_gex_from_chain(near_chain, spot)
    flip = find_gamma_flip(gex_chain)
    regime, dist, stability = classify_gex_regime(spot, flip)

    # Aggregate net_gex BY STRIKE (sum across all expirations) before computing
    # support/resistance levels and the by_strike dict.
    # The raw gex_chain has one row per (strike, expiry) — using it directly
    # causes the same strike to appear multiple times in top_support/resistance,
    # and by_strike only gets the last expiry's value instead of the total.
    agg = (gex_chain.groupby("strike")["net_gex"]
                    .sum()
                    .reset_index()
                    .sort_values("strike"))

    by_strike = dict(zip(agg["strike"].tolist(), agg["net_gex"].tolist()))

    # Positive GEX strikes above spot = resistance walls (dealers sell into rallies)
    pos_above = agg[(agg["net_gex"] > 0) & (agg["strike"] > spot)]
    top_resistance = pos_above.nlargest(5, "net_gex")["strike"].tolist()

    # Negative GEX strikes below spot = support zones (dealers amplify falls)
    neg_below = agg[(agg["net_gex"] < 0) & (agg["strike"] < spot)]
    top_support = neg_below.nsmallest(5, "net_gex")["strike"].tolist()

    # Fallback: if no pos/neg on the correct side, use all pos/neg strikes
    if not top_resistance:
        top_resistance = agg[agg["net_gex"] > 0].nlargest(5, "net_gex")["strike"].tolist()
    if not top_support:
        top_support = agg[agg["net_gex"] < 0].nsmallest(5, "net_gex")["strike"].tolist()

    return GammaState(
        regime=regime, gamma_flip=float(flip) if np.isfinite(flip) else 0.0,
        distance_to_flip_pct=dist, total_gex=float(gex_chain["net_gex"].sum()),
        gex_by_strike=by_strike, key_support=top_support, key_resistance=top_resistance,
        regime_stability=stability, data_source=source,
        timestamp=dt.datetime.now().strftime("%H:%M:%S"),
    )

# ============================================================
# ADVANCED ANALYTICS — GWAS / TERM STRUCTURE / FLOW IMBALANCE
# ============================================================

def compute_gwas(chain: pd.DataFrame, spot: float) -> dict:
    """
    Gamma-Weighted Average Strike (GWAS) above and below spot.

    Instead of discrete "wall at $X", this gives a probabilistic gravity
    centre for positive-gamma pinning zones — the diffuse region where
    dealer hedging creates mean-reversion pressure.

    Returns dict with:
      gwas_above  — gamma-weighted avg strike of all strikes above spot
      gwas_below  — gamma-weighted avg strike of all strikes below spot
      gwas_net    — net gravity centre across all strikes (call-weighted)
      total_gex_above / total_gex_below — magnitudes for sizing
    """
    if chain is None or chain.empty:
        return {}

    # Work from per-strike aggregated GEX (sum across expirations)
    gex_chain = compute_gex_from_chain(chain, spot)
    agg = gex_chain.groupby("strike")["net_gex"].sum().reset_index()

    above = agg[agg["strike"] > spot].copy()
    below = agg[agg["strike"] < spot].copy()

    def _wt_avg(df):
        weights = df["net_gex"].abs()
        total = weights.sum()
        if total == 0:
            return None, 0.0
        return float((df["strike"] * weights).sum() / total), float(total)

    gwas_above, mag_above = _wt_avg(above)
    gwas_below, mag_below = _wt_avg(below)

    # Net gravity: positive-GEX strikes only (true pin zones)
    pos = agg[agg["net_gex"] > 0]
    wt_pos = pos["net_gex"].sum()
    gwas_net = float((pos["strike"] * pos["net_gex"]).sum() / wt_pos) if wt_pos > 0 else None

    return {
        "gwas_above":       gwas_above,
        "gwas_below":       gwas_below,
        "gwas_net":         gwas_net,
        "total_gex_above":  mag_above,
        "total_gex_below":  mag_below,
    }


def compute_gex_term_structure(chain: pd.DataFrame, spot: float) -> dict:
    """
    GEX Term Structure: ratio of near-term to longer-dated gamma.

    Splits chain into:
      ≤7 DTE  — weekly / 0DTE positioning (expires fast, fragile)
      8-45 DTE — monthly positioning (durable regime support)

    A positive gamma regime driven by weeklies evaporates by Friday.
    Monthly gamma provides structural support that persists across sessions.

    Returns:
      gex_0_7dte       — total net GEX for ≤7 DTE
      gex_8_45dte      — total net GEX for 8-45 DTE
      fragility_ratio  — 0_7 / total (0=all monthlies, 1=all weeklies)
      durability       — "durable" / "fragile" / "mixed"
      dte_buckets      — list of (label, gex) for bar chart
    """
    if chain is None or chain.empty:
        return {}

    gex_chain = compute_gex_from_chain(chain, spot)
    gex_chain["dte"] = (gex_chain["expiry_T"] * 365).round().astype(int)

    buckets = [
        ("0–7 DTE",   gex_chain[gex_chain["dte"] <= 7]["net_gex"].sum()),
        ("8–21 DTE",  gex_chain[(gex_chain["dte"] > 7)  & (gex_chain["dte"] <= 21)]["net_gex"].sum()),
        ("22–45 DTE", gex_chain[(gex_chain["dte"] > 21) & (gex_chain["dte"] <= 45)]["net_gex"].sum()),
        ("46+ DTE",   gex_chain[gex_chain["dte"] > 45]["net_gex"].sum()),
    ]

    gex_0_7   = float(gex_chain[gex_chain["dte"] <= 7]["net_gex"].sum())
    gex_8_45  = float(gex_chain[(gex_chain["dte"] > 7) & (gex_chain["dte"] <= 45)]["net_gex"].sum())
    total_abs = abs(gex_0_7) + abs(gex_8_45)
    fragility = abs(gex_0_7) / total_abs if total_abs > 0 else 0.5

    if fragility > 0.65:
        durability = "fragile"   # regime collapses as weeklies expire
    elif fragility < 0.30:
        durability = "durable"   # monthly gamma dominates → structural
    else:
        durability = "mixed"

    return {
        "gex_0_7dte":      gex_0_7,
        "gex_8_45dte":     gex_8_45,
        "fragility_ratio": fragility,
        "durability":      durability,
        "dte_buckets":     buckets,
    }


def compute_flow_imbalance(chain: pd.DataFrame, spot: float) -> dict:
    """
    Put/Call Dollar Volume Imbalance.

    Volume = what's happening NOW. OI = yesterday's positions.
    Dollar premium = volume × mid_price (or IV-approximated premium).

    If 'volume' column present (Schwab intraday), uses it directly.
    Falls back to OI as a proxy when volume is unavailable.

    Returns:
      put_dollar_vol   — total put dollar premium (volume × approx price)
      call_dollar_vol  — total call dollar premium
      pc_ratio         — put / call dollar ratio (>1.2 = fear, <0.8 = greed)
      flow_bias        — "bearish" / "bullish" / "neutral"
      put_pct          — put share of total dollar flow
    """
    if chain is None or chain.empty:
        return {}

    df = chain.copy()
    has_volume = "volume" in df.columns and df["volume"].notna().any() and (df["volume"] > 0).any()

    # Approximate premium per contract: Black-Scholes call/put price
    # (fast approximation using ATM IV for simplicity; good enough for ratio)
    def _approx_premium(row, is_call):
        S, K, T, iv = spot, row["strike"], max(row.get("expiry_T", 0.01), 1/365), row.get("iv", 0.20)
        if T <= 0 or iv <= 0:
            return max(S - K, 0) if is_call else max(K - S, 0)
        d1 = (math.log(S / K) + (0.05 + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
        d2 = d1 - iv * math.sqrt(T)
        if is_call:
            return float(S * scipy_norm.cdf(d1) - K * math.exp(-0.05 * T) * scipy_norm.cdf(d2))
        else:
            return float(K * math.exp(-0.05 * T) * scipy_norm.cdf(-d2) - S * scipy_norm.cdf(-d1))

    # Use volume if available, else OI as proxy
    flow_col = "volume" if has_volume else None

    call_rows = df[df["call_oi"] > 0].copy() if not has_volume else df[df.get("call_volume", pd.Series(dtype=float)) > 0].copy()
    put_rows  = df[df["put_oi"] > 0].copy()

    # Simpler: use OI × premium as the dollar weight
    call_premium = df.apply(lambda r: _approx_premium(r, True)  * (r.get("volume", r["call_oi"]) if has_volume else r["call_oi"]) * 100, axis=1).sum()
    put_premium  = df.apply(lambda r: _approx_premium(r, False) * (r.get("volume", r["put_oi"])  if has_volume else r["put_oi"])  * 100, axis=1).sum()

    total = call_premium + put_premium
    pc_ratio = put_premium / call_premium if call_premium > 0 else 1.0
    put_pct  = put_premium / total if total > 0 else 0.5

    if pc_ratio > 1.3:
        flow_bias = "bearish"   # heavy put premium = fear / hedging
    elif pc_ratio < 0.77:
        flow_bias = "bullish"   # call premium dominant = speculation / complacency
    else:
        flow_bias = "neutral"

    return {
        "put_dollar_vol":  float(put_premium),
        "call_dollar_vol": float(call_premium),
        "pc_ratio":        float(pc_ratio),
        "flow_bias":       flow_bias,
        "put_pct":         float(put_pct),
        "using_volume":    has_volume,
    }

