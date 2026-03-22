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
    Sign convention:
    Positive vanna: IV rises → dealers BUY, IV falls → dealers SELL
    Negative vanna: IV rises → dealers SELL, IV falls → dealers BUY
    """
    if T <= 0 or sigma <= 0: return 0.0
    d1, d2 = _d1d2(S, K, T, sigma, r)
    return -scipy_norm.pdf(d1) * d2 / sigma


def bs_vega(S, K, T, sigma, r=0.05) -> float:
    """True Vega = dPrice/dIV = S * phi(d1) * sqrt(T).
    Dollar sensitivity of option price to a 1-unit (100%) change in IV.
    VEX = OI * vega * multiplier  →  dollar gamma per 1-vol move (matches Doc 1 definition).
    For per-1-vol-point (1%), divide by 100 at display time.
    """
    if T <= 0 or sigma <= 0: return 0.0
    d1, _ = _d1d2(S, K, T, sigma, r)
    return S * scipy_norm.pdf(d1) * math.sqrt(T)


def bs_charm(S, K, T, sigma, r=0.05, option_type: str = "call") -> float:
    """Charm = dDelta/dTime (annualised rate of delta decay).
    Full per-side formula (matches Doc 1):
      call_charm = base - r * exp(-rT) * N(d2)
      put_charm  = base + r * exp(-rT) * N(-d2)
    where base = -phi(d1)*(2rT - d2*sigma*sqrt(T)) / (2T*sigma*sqrt(T))

    Sign convention: positive charm → dealer delta increases over time → they buy (upward drift).
    """
    if T <= 0 or sigma <= 0: return 0.0
    d1, d2 = _d1d2(S, K, T, sigma, r)
    base = -scipy_norm.pdf(d1) * (2 * r * T - d2 * sigma * math.sqrt(T)) / (2 * T * sigma * math.sqrt(T))
    if option_type == "put":
        return base + r * math.exp(-r * T) * scipy_norm.cdf(-d2)
    # default: call
    return base - r * math.exp(-r * T) * scipy_norm.cdf(d2)

def compute_gex_from_chain(chain: pd.DataFrame, spot: float,
                            multiplier=100, r=0.05) -> pd.DataFrame:
    """
    chain must have: strike, expiry_T, iv, call_oi, put_oi
    Optional volume columns: call_volume, put_volume (used for 0DTE flow profile)

    GEX sign convention: calls sold by dealers = positive (stabilizing)
                         puts sold by dealers = negative (destabilizing)

    Formulas (Doc 1 standard — no 0.01 baked in):
      GEX  = OI × Γ × S² × multiplier          (raw dollar gamma)
      VEX  = OI × vega × multiplier             (true vega exposure, Doc 1 definition)
      CEX  = OI × charm × multiplier            (per-side charm, full r-correction)
      VNNX = OI × vanna × multiplier            (vanna exposure — separate from VEX)

    The 0.01 per-1%-move normalisation belongs at the DISPLAY layer, not here.
    """
    chain = chain.copy()

    # ── Gamma source: prefer per-side Schwab gamma, fallback to BS ────────
    bs_g = chain.apply(
        lambda row: bs_gamma(spot, row["strike"], row["expiry_T"], row["iv"], r), axis=1)

    has_call_gamma = ("call_gamma" in chain.columns and
                      (chain["call_gamma"].fillna(0) > 0).any())
    has_put_gamma  = ("put_gamma"  in chain.columns and
                      (chain["put_gamma"].fillna(0)  > 0).any())
    has_schwab_gamma = (
        "schwab_gamma" in chain.columns
        and chain["schwab_gamma"].notna().any()
        and (chain["schwab_gamma"].abs() > 0).any()
    )

    if has_call_gamma and has_put_gamma:
        call_g_valid   = chain["call_gamma"].fillna(0) > 0
        put_g_valid    = chain["put_gamma"].fillna(0)  > 0
        call_gamma_col = np.where(call_g_valid, chain["call_gamma"].fillna(0), bs_g)
        put_gamma_col  = np.where(put_g_valid,  chain["put_gamma"].fillna(0),  bs_g)
    elif has_schwab_gamma:
        schwab_valid = (
            chain["schwab_gamma"].notna() &
            chain["schwab_gamma"].apply(np.isfinite) &
            (chain["schwab_gamma"] > 0)
        )
        blended = np.where(schwab_valid, chain["schwab_gamma"], bs_g)
        call_gamma_col = blended
        put_gamma_col  = blended
    else:
        call_gamma_col = bs_g.values
        put_gamma_col  = bs_g.values

    # ── GEX: raw dollar gamma (Doc 1: OI × Γ × S² × 100, NO 0.01) ───────
    chain["call_gex"] =  chain["call_oi"] * call_gamma_col * multiplier * (spot ** 2)
    chain["put_gex"]  = -chain["put_oi"]  * put_gamma_col  * multiplier * (spot ** 2)
    chain["net_gex"]  =  chain["call_gex"] + chain["put_gex"]

    # ── Volume-based GEX for 0DTE flow profile (separate columns) ─────────
    # Uses volume instead of OI — measures real-time dealer hedging pressure.
    # OTM-only: calls at K >= spot, puts at K <= spot.
    has_call_vol = ("call_volume" in chain.columns and
                    (chain["call_volume"].fillna(0) > 0).any())
    has_put_vol  = ("put_volume"  in chain.columns and
                    (chain["put_volume"].fillna(0)  > 0).any())
    if has_call_vol and has_put_vol:
        call_vol = chain["call_volume"].fillna(0)
        put_vol  = chain["put_volume"].fillna(0)
    else:
        # Fallback: use OI as volume proxy (less accurate)
        call_vol = chain["call_oi"].fillna(0)
        put_vol  = chain["put_oi"].fillna(0)

    # OTM filter mask: calls above spot, puts below spot (Doc 2 §4 & §5)
    otm_call_mask = chain["strike"] >= spot
    otm_put_mask  = chain["strike"] <= spot

    chain["call_vol_gex"] = np.where(
        otm_call_mask,
         call_vol * call_gamma_col * multiplier * (spot ** 2),
        0.0
    )
    chain["put_vol_gex"] = np.where(
        otm_put_mask,
        -put_vol * put_gamma_col * multiplier * (spot ** 2),
        0.0
    )
    chain["net_vol_gex"] = chain["call_vol_gex"] + chain["put_vol_gex"]

    # ── Vanna (dDelta/dIV) — kept as VNNX, separate from VEX ─────────────
    chain["vanna"] = chain.apply(
        lambda row: bs_vanna(spot, row["strike"], row["expiry_T"], row["iv"], r), axis=1)
    chain["call_vnnx"] =  chain["call_oi"] * chain["vanna"] * multiplier
    chain["put_vnnx"]  = -chain["put_oi"]  * chain["vanna"] * multiplier
    chain["net_vnnx"]  =  chain["call_vnnx"] + chain["put_vnnx"]

    # ── VEX: TRUE Vega exposure (Doc 1: OI × vega × multiplier) ──────────
    # vega = S × phi(d1) × sqrt(T)  — dollar price sensitivity to IV
    chain["vega"] = chain.apply(
        lambda row: bs_vega(spot, row["strike"], row["expiry_T"], row["iv"], r), axis=1)
    chain["call_vex"] =  chain["call_oi"] * chain["vega"] * multiplier
    chain["put_vex"]  = -chain["put_oi"]  * chain["vega"] * multiplier
    chain["net_vex"]  =  chain["call_vex"] + chain["put_vex"]

    # ── CEX: Charm with full per-side r-correction (Doc 1 formula) ────────
    chain["call_charm"] = chain.apply(
        lambda row: bs_charm(spot, row["strike"], row["expiry_T"], row["iv"], r, "call"), axis=1)
    chain["put_charm"] = chain.apply(
        lambda row: bs_charm(spot, row["strike"], row["expiry_T"], row["iv"], r, "put"), axis=1)
    # Backward-compat alias: charm = call charm (blended; sign is correct for calls/puts separately)
    chain["charm"] = chain["call_charm"]
    chain["call_cex"] =  chain["call_oi"] * chain["call_charm"] * multiplier
    chain["put_cex"]  = -chain["put_oi"]  * chain["put_charm"]  * multiplier
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
        net_gex=("net_gex",  "sum"),
        net_vex=("net_vex",  "sum"),   # true vega exposure (Doc 1 definition)
        net_vnnx=("net_vnnx", "sum"),  # vanna exposure (dDelta/dIV — direction signal)
        net_cex=("net_cex",  "sum"),
    ).reset_index()

    gex_by_strike  = dict(zip(agg["strike"], agg["net_gex"]))
    vex_by_strike  = dict(zip(agg["strike"], agg["net_vex"]))   # true vega
    vnnx_by_strike = dict(zip(agg["strike"], agg["net_vnnx"]))  # vanna
    cex_by_strike  = dict(zip(agg["strike"], agg["net_cex"]))

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

    # Vanna direction: use NET VNNX (true vanna) near spot (within 2%)
    # Convention:
    #   Positive vanna: IV rises → dealers BUY, IV falls → dealers SELL
    #   Negative vanna: IV rises → dealers SELL, IV falls → dealers BUY
    ntm_vnnx = sum(v for k, v in vnnx_by_strike.items() if abs(k - spot) / spot < 0.02)
    vanna_direction = "bullish" if ntm_vnnx > 0 else ("bearish" if ntm_vnnx < 0 else "neutral")
    vanna_sign      = "positive" if ntm_vnnx > 0 else ("negative" if ntm_vnnx < 0 else "neutral")

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

def _net_gamma_at_spot(chain: pd.DataFrame, x: float,
                        multiplier: int = 100, r: float = 0.05) -> float:
    """
    Compute total net dealer gamma if spot were at price x.

    Recomputes BS gamma for every strike using x as the spot price.
    This is required for the TRUE gamma flip — the level where total
    net gamma (as a function of hypothetical spot) equals zero.

    G_net(x) = Σ_calls[ Γ_i(x) · OI_i · 100 ] - Σ_puts[ Γ_i(x) · OI_i · 100 ]
    """
    total = 0.0
    for _, row in chain.iterrows():
        K, T, iv = float(row["strike"]), float(row["expiry_T"]), float(row["iv"])
        if T <= 0 or iv <= 0:
            continue
        g = bs_gamma(x, K, T, iv, r)
        c_oi = float(row.get("call_oi", 0))
        p_oi = float(row.get("put_oi",  0))
        total += g * multiplier * (c_oi - p_oi)
    return total


def find_gamma_flip(chain: pd.DataFrame, spot: float = None,
                    scan_pct: float = 0.10, n_points: int = 200) -> float:
    """
    True zero-gamma level: the spot price x where total net dealer gamma = 0.

    Method (correct):
        Scan a price grid x ∈ [spot*(1-scan_pct), spot*(1+scan_pct)] and
        compute G_net(x) = Σ Γ_i(x)·OI_i·100 for calls minus puts at each x.
        Find where G_net crosses zero and interpolate.

    This is the proper definition: gamma at each strike changes as spot moves,
    so the flip level requires recomputing greeks at each hypothetical price.

    Fallback (when no zero crossing found):
        Return the x with G_net closest to zero — the "vol trigger" / minimum
        gamma level, equivalent to SpotGamma's zero-gamma fallback.

    Reference: Carr & Madan (2001); SpotGamma methodology notes (2021).
    """
    if chain is None or len(chain) == 0:
        return np.nan

    # Build a deduplicated chain for efficiency (aggregate OI by strike/expiry)
    cols = [c for c in ["strike","expiry_T","iv","call_oi","put_oi"] if c in chain.columns]
    chain_agg = (chain[cols]
                 .groupby(["strike","expiry_T"])
                 .agg({"iv":"mean","call_oi":"sum","put_oi":"sum"})
                 .reset_index())

    # Determine scan range
    if spot is None or not np.isfinite(spot) or spot <= 0:
        # Estimate spot from chain midpoint
        spot = float(chain_agg["strike"].median())

    x_lo = spot * (1 - scan_pct)
    x_hi = spot * (1 + scan_pct)
    x_grid = np.linspace(x_lo, x_hi, n_points)

    # Vectorised computation of G_net at each grid point
    g_vals = np.array([_net_gamma_at_spot(chain_agg, float(x)) for x in x_grid])

    if not np.any(np.isfinite(g_vals)):
        return float(spot)  # no valid data

    # Find zero crossings
    sign_changes = np.where(g_vals[:-1] * g_vals[1:] < 0)[0]

    if len(sign_changes) > 0:
        # Interpolate within the crossing interval with largest transition
        best_i = sign_changes[np.argmax([abs(g_vals[j+1] - g_vals[j])
                                          for j in sign_changes])]
        x1, x2 = x_grid[best_i], x_grid[best_i + 1]
        g1, g2 = g_vals[best_i], g_vals[best_i + 1]
        return float(x1 + (x2 - x1) * (-g1) / (g2 - g1)) if (g2 - g1) != 0 else float(x1)

    # No crossing: return x with G_net closest to zero
    return float(x_grid[np.argmin(np.abs(g_vals))])

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


def compute_cumulative_gex_profile(chain: pd.DataFrame, spot: float,
                                    max_dte: int = 45) -> pd.DataFrame:
    """
    Cumulative GEX profile — how total dealer gamma exposure changes as spot
    moves through each strike level.

    Method: for each strike K (sorted low→high), compute the cumulative sum of
    net_gex for all strikes <= K.  The result tells you:
      - Where cumulative GEX is most negative  → max pain / highest amplification zone
      - Where it crosses zero                  → the gamma flip (same as find_gamma_flip)
      - Where it peaks positive                → strongest pinning / resistance wall
      - Shape of curve                         → steep = sharp regime change at that level

    This is the profile shown by SpotGamma's "GEX Profile" chart and GEXBot's
    cumulative view — it gives spatial context that individual strike bars lack.

    Returns DataFrame with columns: strike, net_gex, cum_gex, regime
      regime: "negative" | "positive" (sign of cum_gex at each strike)
    """
    near = chain[chain["expiry_T"] <= max_dte / 365.0].copy()
    if near.empty:
        near = chain.copy()

    gex_chain = compute_gex_from_chain(near, spot)

    agg = (gex_chain.groupby("strike")["net_gex"]
                    .sum()
                    .reset_index()
                    .sort_values("strike")
                    .reset_index(drop=True))

    agg["cum_gex"] = agg["net_gex"].cumsum()
    agg["regime"]  = np.where(agg["cum_gex"] >= 0, "positive", "negative")
    return agg



def compute_max_pain(chain: pd.DataFrame) -> float:
    """
    Max Pain = candidate expiration price x that minimises total payout to holders.

    Pain(x) = Σ_i[ OI^call_i · max(0, x - K_i) ]   ← calls ITM when x > K_i
            + Σ_i[ OI^put_i  · max(0, K_i - x) ]    ← puts  ITM when x < K_i

    x is searched over all listed strikes (standard industry convention).
    Note: x is the HYPOTHETICAL expiration price, K_i are the fixed strikes.

    Reference: Bollen & Whaley (2004); Avellaneda & Lipkin (2003).
    """
    if chain is None or chain.empty:
        return float("nan")

    by_strike = (chain.groupby("strike")
                      .agg(call_oi=("call_oi","sum"), put_oi=("put_oi","sum"))
                      .reset_index()
                      .sort_values("strike"))

    K_arr     = by_strike["strike"].values.astype(float)
    call_ois  = by_strike["call_oi"].values.astype(float)
    put_ois   = by_strike["put_oi"].values.astype(float)

    min_pain  = float("inf")
    best_x    = float("nan")

    for x in K_arr:                            # search over listed strikes
        call_pain = np.sum(call_ois * np.maximum(0.0, x - K_arr))  # calls ITM: x > K_i
        put_pain  = np.sum(put_ois  * np.maximum(0.0, K_arr - x))  # puts  ITM: K_i > x
        total = float(call_pain + put_pain)
        if total < min_pain:
            min_pain = total
            best_x   = float(x)

    return best_x


def compute_volume_weighted_strike(chain: pd.DataFrame) -> dict:
    """
    Volume-weighted average strike (VWAS) — the options activity centre of mass.

    K_vol = Σ(K_i * V_i) / Σ(V_i)

    Computed separately for calls, puts, and combined.
    Distinct from the gamma flip: VWAS shows WHERE trading is happening,
    gamma flip shows WHERE dealer hedging regime changes.

    Reference: standard VWAP calculation applied to strike space.
    """
    if chain is None or chain.empty or "call_volume" not in chain.columns:
        return {"combined": float("nan"), "calls": float("nan"), "puts": float("nan")}

    by_strike = (chain.groupby("strike")
                      .agg(call_vol=("call_volume","sum"),
                           put_vol=("put_volume","sum"))
                      .reset_index())

    K = by_strike["strike"].values
    cv = by_strike["call_vol"].values.astype(float)
    pv = by_strike["put_vol"].values.astype(float)

    def _vwas(vols):
        total = vols.sum()
        return float((K * vols).sum() / total) if total > 0 else float("nan")

    return {
        "combined": _vwas(cv + pv),
        "calls":    _vwas(cv),
        "puts":     _vwas(pv),
    }


def compute_call_put_walls(chain: pd.DataFrame, spot: float,
                            max_dte: int = 45) -> dict:
    """
    Call wall = strike with highest gamma-weighted call OI above spot.
    Put wall  = strike with highest gamma-weighted put OI below spot.

    Uses gamma-weighted OI (OI * gamma) rather than raw OI because a high-OI
    strike at low gamma has minimal dealer hedging impact. The gamma-weighted
    version matches the SpotGamma / GEXBot methodology.

    Reference: SpotGamma SGVI methodology notes; Bollen & Whaley (2004).
    """
    if chain is None or chain.empty:
        return {"call_wall": float("nan"), "put_wall": float("nan"),
                "call_wall_gex": 0.0,     "put_wall_gex": 0.0}

    near = chain[chain["expiry_T"] <= max_dte / 365.0].copy()
    if near.empty:
        near = chain.copy()

    gex_chain = compute_gex_from_chain(near, spot)
    agg = (gex_chain.groupby("strike")
                    .agg(call_gex=("call_gex","sum"), put_gex=("put_gex","sum"))
                    .reset_index())

    # Call wall: highest positive (call) GEX above spot
    above = agg[agg["strike"] > spot].copy()
    if not above.empty:
        idx = above["call_gex"].idxmax()
        call_wall     = float(above.loc[idx, "strike"])
        call_wall_gex = float(above.loc[idx, "call_gex"])
    else:
        call_wall, call_wall_gex = float("nan"), 0.0

    # Put wall: most negative (put) GEX below spot
    below = agg[agg["strike"] < spot].copy()
    if not below.empty:
        idx = below["put_gex"].idxmin()
        put_wall     = float(below.loc[idx, "strike"])
        put_wall_gex = float(below.loc[idx, "put_gex"])
    else:
        put_wall, put_wall_gex = float("nan"), 0.0

    return {
        "call_wall":     call_wall,
        "put_wall":      put_wall,
        "call_wall_gex": call_wall_gex,
        "put_wall_gex":  put_wall_gex,
    }

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
    flip = find_gamma_flip(near_chain, spot=spot)  # true zero-gamma scan over spot grid
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
    Put/Call Dollar Premium Imbalance.

    When call_volume / put_volume columns are present (populated from Schwab
    totalVolume), computes volume-weighted premium — genuine intraday FLOW.

    When volume columns are absent or all-zero, falls back to OI-weighted
    premium.  That measures accumulated positional INVENTORY, not flow.
    These are different signals and often diverge near expiry.  The
    'using_volume' key makes the mode explicit so callers can label correctly.

    Returns:
      put_dollar_vol   — total put dollar premium
      call_dollar_vol  — total call dollar premium
      pc_ratio         — put / call dollar ratio (>1.3 = fear, <0.77 = greed)
      flow_bias        — "bearish" / "bullish" / "neutral"
      put_pct          — put share of total dollar premium
      using_volume     — True if volume-based (flow), False if OI-based (inventory)
    """
    if chain is None or chain.empty:
        return {}

    df = chain.copy()

    # Prefer per-side volume columns (call_volume / put_volume) from
    # schwab_get_options_chain (totalVolume).  A single generic 'volume'
    # column is not expected from Schwab which separates calls and puts.
    has_call_vol = ("call_volume" in df.columns
                    and df["call_volume"].notna().any()
                    and (df["call_volume"] > 0).any())
    has_put_vol  = ("put_volume"  in df.columns
                    and df["put_volume"].notna().any()
                    and (df["put_volume"] > 0).any())
    has_volume   = has_call_vol and has_put_vol

    # Approximate premium per contract via Black-Scholes mid-price.
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

    if has_volume:
        # Volume-weighted: genuine intraday flow
        call_premium = df.apply(
            lambda r: _approx_premium(r, True)  * float(r.get("call_volume", 0)) * 100,
            axis=1).sum()
        put_premium  = df.apply(
            lambda r: _approx_premium(r, False) * float(r.get("put_volume",  0)) * 100,
            axis=1).sum()
    else:
        # OI-weighted: positional inventory proxy (NOT flow)
        call_premium = df.apply(
            lambda r: _approx_premium(r, True)  * float(r.get("call_oi", 0)) * 100,
            axis=1).sum()
        put_premium  = df.apply(
            lambda r: _approx_premium(r, False) * float(r.get("put_oi",  0)) * 100,
            axis=1).sum()

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
