# gex_engine.py — GEX computation: fully vectorized, no apply/iterrows
import math, datetime as dt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm
from config import GammaState, GammaRegime, SetupScore
from utils import _to_1d, _bs_iv_from_price

@dataclass
class DealerGreeks:
    gex_by_strike:       Dict[float, float] = field(default_factory=dict)
    vex_by_strike:       Dict[float, float] = field(default_factory=dict)
    cex_by_strike:       Dict[float, float] = field(default_factory=dict)
    key_nodes_gex:       List[Tuple[float, float]] = field(default_factory=list)
    key_nodes_vex:       List[Tuple[float, float]] = field(default_factory=list)
    key_nodes_cex:       List[Tuple[float, float]] = field(default_factory=list)
    otm_anchors:         List[Tuple[float, float]] = field(default_factory=list)
    vanna_charm_aligned: bool = False
    vanna_direction:     str  = "neutral"
    vanna_sign:          str  = "neutral"
    charm_direction:     str  = "neutral"
    data_source:         str  = "unknown"


# ── Vectorized BS primitives ──────────────────────────────────────────────────

def _vec_d1d2(S, K, T, sigma, r=0.05):
    T = np.maximum(T, 1e-8); sigma = np.maximum(sigma, 1e-8)
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5*sigma**2)*T) / (sigma*sqT)
    return d1, d1 - sigma*sqT

def _vec_gamma(S, K, T, sigma, r=0.05):
    T = np.maximum(T,1e-8); sigma = np.maximum(sigma,1e-8)
    d1,_ = _vec_d1d2(S,K,T,sigma,r)
    return scipy_norm.pdf(d1) / (S * sigma * np.sqrt(T))

def _vec_vanna(S, K, T, sigma, r=0.05):
    T = np.maximum(T,1e-8); sigma = np.maximum(sigma,1e-8)
    d1,d2 = _vec_d1d2(S,K,T,sigma,r)
    return -scipy_norm.pdf(d1)*d2 / sigma

def _vec_charm(S, K, T, sigma, r=0.05):
    """Returns (call_charm, put_charm) arrays."""
    T = np.maximum(T,1e-8); sigma = np.maximum(sigma,1e-8); sqT = np.sqrt(T)
    d1,d2 = _vec_d1d2(S,K,T,sigma,r)
    base   = -scipy_norm.pdf(d1)*(2*r*T - d2*sigma*sqT) / (2*T*sigma*sqT)
    rd     = r * np.exp(-r*T)
    return base - rd*scipy_norm.cdf(d2), base + rd*scipy_norm.cdf(-d2)

# Scalar wrappers for external compatibility
def _d1d2(S,K,T,sigma,r=0.05):
    if T<=0 or sigma<=0: return 0.0,0.0
    d1=(math.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T)); return d1, d1-sigma*math.sqrt(T)
def bs_gamma(S,K,T,sigma,r=0.05):
    if T<=0 or sigma<=0: return 0.0
    d1,_=_d1d2(S,K,T,sigma,r); return scipy_norm.pdf(d1)/(S*sigma*math.sqrt(T))
def bs_vanna(S,K,T,sigma,r=0.05):
    if T<=0 or sigma<=0: return 0.0
    d1,d2=_d1d2(S,K,T,sigma,r); return -scipy_norm.pdf(d1)*d2/sigma
def bs_charm(S,K,T,sigma,r=0.05,option_type="call"):
    if T<=0 or sigma<=0: return 0.0
    d1,d2=_d1d2(S,K,T,sigma,r)
    base=-scipy_norm.pdf(d1)*(2*r*T-d2*sigma*math.sqrt(T))/(2*T*sigma*math.sqrt(T))
    return base+(r*math.exp(-r*T)*scipy_norm.cdf(-d2)) if option_type=="put" else base-(r*math.exp(-r*T)*scipy_norm.cdf(d2))


# ── Main chain computation — single vectorized pass ───────────────────────────

def compute_gex_from_chain(chain: pd.DataFrame, spot: float,
                            multiplier: int = 100, r: float = 0.05) -> pd.DataFrame:
    """
    Compute GEX/VEX/CEX/vol-GEX in one NumPy pass. No apply(), no iterrows().
    ~50-100x faster than the previous row-by-row implementation.
    """
    chain = chain.copy()
    if len(chain) == 0:
        return chain

    S    = float(spot)
    K    = chain["strike"].to_numpy(dtype=float)
    T    = np.maximum(chain["expiry_T"].to_numpy(dtype=float), 1e-8)
    iv   = np.maximum(chain["iv"].to_numpy(dtype=float), 1e-8)
    c_oi = chain["call_oi"].to_numpy(dtype=float)
    p_oi = chain["put_oi"].to_numpy(dtype=float)
    mult = float(multiplier)
    S2   = S * S

    # Gamma source priority: per-side Schwab > blended Schwab > BS
    bs_g = _vec_gamma(S, K, T, iv, r)
    has_cg = ("call_gamma" in chain.columns and (chain["call_gamma"].fillna(0)>0).any())
    has_pg = ("put_gamma"  in chain.columns and (chain["put_gamma"].fillna(0)>0).any())
    has_sg = ("schwab_gamma" in chain.columns and chain["schwab_gamma"].notna().any()
              and (chain["schwab_gamma"].abs()>0).any())

    if has_cg and has_pg:
        cg = chain["call_gamma"].fillna(0).to_numpy(dtype=float)
        pg = chain["put_gamma"].fillna(0).to_numpy(dtype=float)
        call_g = np.where(cg>0, cg, bs_g)
        put_g  = np.where(pg>0, pg, bs_g)
    elif has_sg:
        sg = chain["schwab_gamma"].fillna(0).to_numpy(dtype=float)
        bl = np.where(np.isfinite(sg)&(sg>0), sg, bs_g)
        call_g = put_g = bl
    else:
        call_g = put_g = bs_g

    # GEX
    chain["call_gex"] =  c_oi * call_g * mult * S2
    chain["put_gex"]  = -p_oi * put_g  * mult * S2
    chain["net_gex"]  = chain["call_gex"].to_numpy() + chain["put_gex"].to_numpy()

    # Volume GEX (OTM only)
    has_cv = ("call_volume" in chain.columns and (chain["call_volume"].fillna(0)>0).any())
    has_pv = ("put_volume"  in chain.columns and (chain["put_volume"].fillna(0)>0).any())
    cv = chain["call_volume"].fillna(0).to_numpy(dtype=float) if has_cv else c_oi
    pv = chain["put_volume"].fillna(0).to_numpy(dtype=float)  if has_pv else p_oi
    chain["call_vol_gex"] = np.where(K>=S,  cv*call_g*mult*S2, 0.0)
    chain["put_vol_gex"]  = np.where(K<=S, -pv*put_g *mult*S2, 0.0)
    chain["net_vol_gex"]  = chain["call_vol_gex"].to_numpy() + chain["put_vol_gex"].to_numpy()

    # VEX (vanna)
    vanna = _vec_vanna(S, K, T, iv, r)
    chain["vanna"]    =  vanna
    chain["call_vex"] =  c_oi * vanna * mult
    chain["put_vex"]  = -p_oi * vanna * mult
    chain["net_vex"]  = chain["call_vex"].to_numpy() + chain["put_vex"].to_numpy()

    # CEX (charm, per-side)
    cc, pc = _vec_charm(S, K, T, iv, r)
    chain["call_charm"] = cc
    chain["put_charm"]  = pc
    chain["charm"]      = cc  # backward-compat
    chain["call_cex"]   =  c_oi * cc * mult
    chain["put_cex"]    = -p_oi * pc * mult
    chain["net_cex"]    = chain["call_cex"].to_numpy() + chain["put_cex"].to_numpy()

    return chain


# ── Gamma flip — vectorized grid scan ────────────────────────────────────────

def _net_gamma_at_spot(chain: pd.DataFrame, x: float,
                        multiplier: int = 100, r: float = 0.05) -> float:
    K      = chain["strike"].to_numpy(dtype=float)
    T      = np.maximum(chain["expiry_T"].to_numpy(dtype=float), 1e-8)
    iv     = np.maximum(chain["iv"].to_numpy(dtype=float), 1e-8)
    net_oi = chain["call_oi"].to_numpy(dtype=float) - chain["put_oi"].to_numpy(dtype=float)
    return float(np.sum(_vec_gamma(x, K, T, iv, r) * net_oi))


def find_gamma_flip(chain: pd.DataFrame, spot: float = None,
                    scan_pct: float = 0.20, n_points: int = 400,
                    r: float = 0.05) -> float:
    if chain is None or len(chain) == 0:
        return np.nan
    cols = [c for c in ["strike","expiry_T","iv","call_oi","put_oi"] if c in chain.columns]
    agg  = (chain[cols].groupby(["strike","expiry_T"])
                       .agg({"iv":"mean","call_oi":"sum","put_oi":"sum"})
                       .reset_index())
    if spot is None or not np.isfinite(spot) or spot <= 0:
        spot = float(agg["strike"].median())
    k_min, k_max = float(agg["strike"].min()), float(agg["strike"].max())
    x_lo = min(spot*(1-scan_pct), k_min*0.98)
    x_hi = max(spot*(1+scan_pct), k_max*1.02)
    x_grid = np.linspace(x_lo, x_hi, int(n_points))
    g_vals = np.array([_net_gamma_at_spot(agg, float(x), r=r) for x in x_grid])
    if not np.any(np.isfinite(g_vals)):
        return float(spot)
    sc = np.where(g_vals[:-1]*g_vals[1:] < 0)[0]
    if len(sc) > 0:
        i = sc[np.argmin(np.abs(x_grid[sc] - spot))]
        x1,x2,g1,g2 = x_grid[i],x_grid[i+1],g_vals[i],g_vals[i+1]
        return float(x1+(x2-x1)*(-g1)/(g2-g1)) if (g2-g1)!=0 else float(x1)
    return float(x_grid[np.argmin(np.abs(g_vals))])


def classify_gex_regime(spot: float, flip: float) -> Tuple[GammaRegime, float, float]:
    if not np.isfinite(flip): return GammaRegime.NEUTRAL, 0.0, 0.5
    dist_pct = (spot-flip)/flip*100
    if   dist_pct >  2.0: regime = GammaRegime.STRONG_POSITIVE
    elif dist_pct >  0.5: regime = GammaRegime.POSITIVE
    elif dist_pct > -0.5: regime = GammaRegime.NEUTRAL
    elif dist_pct > -2.0: regime = GammaRegime.NEGATIVE
    else:                  regime = GammaRegime.STRONG_NEGATIVE
    stability = float(np.clip(min(abs(dist_pct-0.5),abs(dist_pct+0.5))/2.0, 0, 1))
    return regime, dist_pct, stability


def compute_cumulative_gex_profile(chain: pd.DataFrame, spot: float,
                                    max_dte: int = 45) -> pd.DataFrame:
    near = chain[chain["expiry_T"] <= max_dte/365.0].copy()
    if near.empty: near = chain.copy()
    gc = compute_gex_from_chain(near, spot)
    agg = gc.groupby("strike")["net_gex"].sum().reset_index().sort_values("strike").reset_index(drop=True)
    agg["cum_gex"] = agg["net_gex"].cumsum()
    agg["regime"]  = np.where(agg["cum_gex"]>=0,"positive","negative")
    return agg


def compute_max_pain(chain: pd.DataFrame) -> float:
    if chain is None or chain.empty: return float("nan")
    bs = (chain.groupby("strike").agg(call_oi=("call_oi","sum"),put_oi=("put_oi","sum"))
               .reset_index().sort_values("strike"))
    K = bs["strike"].to_numpy(dtype=float)
    co = bs["call_oi"].to_numpy(dtype=float)
    po = bs["put_oi"].to_numpy(dtype=float)
    cp = np.maximum(0.0, K[:,None]-K[None,:])*co[None,:]
    pp = np.maximum(0.0, K[None,:]-K[:,None])*po[None,:]
    return float(K[np.argmin(cp.sum(1)+pp.sum(1))])


def compute_volume_weighted_strike(chain: pd.DataFrame) -> dict:
    if chain is None or chain.empty or "call_volume" not in chain.columns:
        return {"combined":float("nan"),"calls":float("nan"),"puts":float("nan")}
    bs = chain.groupby("strike").agg(cv=("call_volume","sum"),pv=("put_volume","sum")).reset_index()
    K=bs["strike"].to_numpy(dtype=float); cv=bs["cv"].to_numpy(dtype=float); pv=bs["pv"].to_numpy(dtype=float)
    def _w(v): t=v.sum(); return float((K*v).sum()/t) if t>0 else float("nan")
    return {"combined":_w(cv+pv),"calls":_w(cv),"puts":_w(pv)}


def compute_call_put_walls(chain: pd.DataFrame, spot: float, max_dte: int = 45) -> dict:
    if chain is None or chain.empty:
        return {"call_wall":float("nan"),"put_wall":float("nan"),"call_wall_gex":0.0,"put_wall_gex":0.0}
    near = chain[chain["expiry_T"]<=max_dte/365.0].copy()
    if near.empty: near = chain.copy()
    gc  = compute_gex_from_chain(near, spot)
    agg = gc.groupby("strike").agg(call_gex=("call_gex","sum"),put_gex=("put_gex","sum")).reset_index()
    ab  = agg[agg["strike"]>spot]; bl = agg[agg["strike"]<spot]
    cw  = (float(ab.loc[ab["call_gex"].idxmax(),"strike"]), float(ab["call_gex"].max())) if not ab.empty else (float("nan"),0.0)
    pw  = (float(bl.loc[bl["put_gex"].idxmin(),"strike"]),  float(bl["put_gex"].min()))  if not bl.empty else (float("nan"),0.0)
    return {"call_wall":cw[0],"put_wall":pw[0],"call_wall_gex":cw[1],"put_wall_gex":pw[1]}


def nearest_expiry_chain(chain: pd.DataFrame) -> pd.DataFrame:
    if chain is None or chain.empty: return chain
    return chain[chain["expiry_T"] <= chain["expiry_T"].min()+1/365].copy()


def compute_dealer_greeks(chain: pd.DataFrame, spot: float,
                           source: str = "yfinance", max_dte: int = 45) -> DealerGreeks:
    near = chain[chain["expiry_T"]<=max_dte/365.0].copy()
    if near.empty: near = chain.copy()
    gc  = compute_gex_from_chain(near, spot)
    agg = gc.groupby("strike").agg(net_gex=("net_gex","sum"),net_vex=("net_vex","sum"),
                                    net_cex=("net_cex","sum")).reset_index()
    gbs = dict(zip(agg["strike"],agg["net_gex"]))
    vbs = dict(zip(agg["strike"],agg["net_vex"]))
    cbs = dict(zip(agg["strike"],agg["net_cex"]))
    def _kn(d,n=5): return sorted(d.items(),key=lambda x:abs(x[1]),reverse=True)[:n]
    otm = {k:v for k,v in gbs.items() if abs(k-spot)>spot*0.03}
    ntm_v = sum(v for k,v in vbs.items() if abs(k-spot)/spot<0.02)
    ntm_c = sum(v for k,v in cbs.items() if abs(k-spot)/spot<0.02)
    vd = "bullish" if ntm_v>0 else ("bearish" if ntm_v<0 else "neutral")
    cd = "bullish" if ntm_c>0 else ("bearish" if ntm_c<0 else "neutral")
    return DealerGreeks(
        gex_by_strike=gbs, vex_by_strike=vbs, cex_by_strike=cbs,
        key_nodes_gex=_kn(gbs), key_nodes_vex=_kn(vbs), key_nodes_cex=_kn(cbs),
        otm_anchors=sorted(otm.items(),key=lambda x:abs(x[1]),reverse=True)[:10],
        vanna_charm_aligned=(vd==cd and vd!="neutral"),
        vanna_direction=vd, vanna_sign=("positive" if ntm_v>0 else ("negative" if ntm_v<0 else "neutral")),
        charm_direction=cd, data_source=source,
    )


def build_gamma_state(chain: pd.DataFrame, spot: float, source: str = "yfinance",
                      max_dte: int = 45) -> GammaState:
    import datetime as _dt
    near = chain[chain["expiry_T"]<=max_dte/365.0].copy()
    if near.empty: near = chain.copy()
    gc   = compute_gex_from_chain(near, spot)
    flip = find_gamma_flip(near, spot=spot)
    regime, dist, stability = classify_gex_regime(spot, flip)
    agg  = gc.groupby("strike")["net_gex"].sum().reset_index().sort_values("strike")
    bys  = dict(zip(agg["strike"].tolist(), agg["net_gex"].tolist()))
    pa   = agg[(agg["net_gex"]>0)&(agg["strike"]>spot)]
    nb   = agg[(agg["net_gex"]<0)&(agg["strike"]<spot)]
    tr   = pa.nlargest(5,"net_gex")["strike"].tolist() or agg[agg["net_gex"]>0].nlargest(5,"net_gex")["strike"].tolist()
    ts   = nb.nsmallest(5,"net_gex")["strike"].tolist() or agg[agg["net_gex"]<0].nsmallest(5,"net_gex")["strike"].tolist()
    return GammaState(
        regime=regime, gamma_flip=float(flip) if np.isfinite(flip) else 0.0,
        distance_to_flip_pct=dist, total_gex=float(gc["net_gex"].sum()),
        gex_by_strike=bys, key_support=ts, key_resistance=tr,
        regime_stability=stability, data_source=source,
        timestamp=_dt.datetime.now().strftime("%H:%M:%S"),
    )


# ── Advanced analytics ────────────────────────────────────────────────────────

def compute_gwas(chain: pd.DataFrame, spot: float) -> dict:
    if chain is None or chain.empty: return {}
    agg = compute_gex_from_chain(chain,spot).groupby("strike")["net_gex"].sum().reset_index()
    def _wa(df):
        w=df["net_gex"].abs().to_numpy(); t=w.sum()
        return (float((df["strike"].to_numpy()*w).sum()/t),float(t)) if t>0 else (None,0.0)
    ga,ma = _wa(agg[agg["strike"]>spot]); gb,mb = _wa(agg[agg["strike"]<spot])
    pos=agg[agg["net_gex"]>0]; wp=pos["net_gex"].sum()
    gn = float((pos["strike"]*pos["net_gex"]).sum()/wp) if wp>0 else None
    return {"gwas_above":ga,"gwas_below":gb,"gwas_net":gn,"total_gex_above":ma,"total_gex_below":mb}


def compute_gex_term_structure(chain: pd.DataFrame, spot: float) -> dict:
    if chain is None or chain.empty: return {}
    gc = compute_gex_from_chain(chain,spot)
    gc["dte"] = (gc["expiry_T"]*365).round().astype(int)
    b = [
        ("0–7 DTE",   float(gc[gc["dte"]<=7]["net_gex"].sum())),
        ("8–21 DTE",  float(gc[(gc["dte"]>7)&(gc["dte"]<=21)]["net_gex"].sum())),
        ("22–45 DTE", float(gc[(gc["dte"]>21)&(gc["dte"]<=45)]["net_gex"].sum())),
        ("46+ DTE",   float(gc[gc["dte"]>45]["net_gex"].sum())),
    ]
    g07=b[0][1]; g845=b[1][1]+b[2][1]; ta=abs(g07)+abs(g845)
    frag=abs(g07)/ta if ta>0 else 0.5
    dur="fragile" if frag>0.65 else ("durable" if frag<0.30 else "mixed")
    return {"gex_0_7dte":g07,"gex_8_45dte":g845,"fragility_ratio":frag,"durability":dur,"dte_buckets":b}


def compute_flow_imbalance(chain: pd.DataFrame, spot: float) -> dict:
    if chain is None or chain.empty: return {}
    df=chain.copy()
    has_v = (("call_volume" in df.columns and df["call_volume"].notna().any() and (df["call_volume"]>0).any()) and
             ("put_volume"  in df.columns and df["put_volume"].notna().any()  and (df["put_volume"]>0).any()))
    S=float(spot); K=df["strike"].to_numpy(dtype=float)
    T=np.maximum(df["expiry_T"].fillna(0.01).to_numpy(dtype=float),1/365)
    iv=np.maximum(df["iv"].fillna(0.20).to_numpy(dtype=float),1e-8); r=0.05
    d1,d2=_vec_d1d2(S,K,T,iv,r); disc=np.exp(-r*T)
    cp=np.maximum(S*scipy_norm.cdf(d1)-K*disc*scipy_norm.cdf(d2),0.0)
    pp=np.maximum(K*disc*scipy_norm.cdf(-d2)-S*scipy_norm.cdf(-d1),0.0)
    cv = df["call_volume"].fillna(0).to_numpy(dtype=float) if has_v else df["call_oi"].fillna(0).to_numpy(dtype=float)
    pv = df["put_volume"].fillna(0).to_numpy(dtype=float)  if has_v else df["put_oi"].fillna(0).to_numpy(dtype=float)
    cprem=float(np.sum(cp*cv*100)); pprem=float(np.sum(pp*pv*100))
    tot=cprem+pprem; pcr=pprem/cprem if cprem>0 else 1.0; ppt=pprem/tot if tot>0 else 0.5
    fb="bearish" if pcr>1.3 else ("bullish" if pcr<0.77 else "neutral")
    return {"put_dollar_vol":pprem,"call_dollar_vol":cprem,"pc_ratio":pcr,
            "flow_bias":fb,"put_pct":ppt,"using_volume":has_v}
