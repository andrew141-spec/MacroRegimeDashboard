# gex_engine.py — GEX computation: gamma per strike, flip detection, regime
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
        # Use Schwab gamma where non-zero, fill gaps with BS
        chain["gamma"] = chain.apply(
            lambda row: (
                abs(float(row["schwab_gamma"]))
                if ("schwab_gamma" in row.index
                    and np.isfinite(float(row["schwab_gamma"]))
                    and abs(float(row["schwab_gamma"])) > 0)
                else bs_gamma(spot, row["strike"], row["expiry_T"], row["iv"], r)
            ), axis=1
        )
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
    Zero-gamma level where: Sigma_i (Gamma_i * OI_i * ContractSize * S^2) = 0
    Linearly interpolates the zero-crossing of cumulative net_gex across strikes.
    net_gex already embeds S^2 from compute_gex_from_chain.
    """
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
