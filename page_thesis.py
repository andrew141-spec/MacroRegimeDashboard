# page_thesis.py — Daily Thesis Briefing (full 13-section report)
import math, datetime as dt
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from scipy.stats import skew as _scipy_skew, kurtosis as _scipy_kurt

from config import GammaState, GammaRegime, CSS
from utils import _to_1d, zscore, resample_ffill, current_pct_rank
from ui_components import plotly_dark
from data_loaders import load_macro, get_gex_from_yfinance
from gex_engine import (build_gamma_state, compute_gwas,
                         compute_gex_term_structure, compute_flow_imbalance)
from schwab_api import get_schwab_client, schwab_get_spot, schwab_get_options_chain
from signals import compute_leading_stack, compute_1d_prob
from probability import (compute_prob_composite, get_session_context,
                          classify_macro_regime_abs, regime_transition_prob)
from intel_monitor import (load_feeds, geo_shock_score, categorise_items,
                            category_shock_score, _all_feeds_flat, INTEL_CATEGORIES)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _sl(s, d=float("nan")):
    try:
        v = s.dropna()
        return float(v.iloc[-1]) if len(v) else d
    except Exception:
        return d

@st.cache_data(ttl=60)
def _quotes() -> Dict:
    out = {}
    pairs = [("SPX","^GSPC"),("NDX","^NDX"),("VIX","^VIX"),("VIX3M","^VIX3M"),
             ("VVIX","^VVIX"),("DXY","DX-Y.NYB"),("GLD","GLD"),("TLT","TLT"),
             ("TNX","^TNX"),("IRX","^IRX"),("SPY","SPY"),("QQQ","QQQ"),
             ("HYG","HYG"),("ES","ES=F"),("NQ","NQ=F")]
    for k, sym in pairs:
        try:
            h = yf.Ticker(sym).history(period="5d")
            if not h.empty:
                out[k+"_last"] = float(h["Close"].iloc[-1])
                out[k+"_prev"] = float(h["Close"].iloc[-2]) if len(h)>1 else out[k+"_last"]
                out[k+"_pct"]  = (out[k+"_last"]/out[k+"_prev"]-1)*100
        except Exception:
            pass
    return out

def _vrp_full(vix: float, spy: pd.Series, idx) -> Dict:
    sa   = _to_1d(spy).reindex(idx).ffill()
    rets = sa.pct_change()
    rv21 = rets.rolling(21, min_periods=10).std() * np.sqrt(252) * 100
    vrp  = vix - rv21
    val  = _sl(vrp); z = _sl(zscore(vrp)); pct = current_pct_rank(vrp, 252)
    rv21v = _sl(rv21)
    reg  = ("RICH" if val>2 else "CHEAP" if val<-1 else "FAIR") if np.isfinite(val) else "N/A"
    return {"val":val,"z":z,"pct":float(pct),"regime":reg,"rv21":rv21v,"spread":val}

def _vts(q: Dict) -> Dict:
    v=q.get("VIX_last",float("nan")); v3=q.get("VIX3M_last",float("nan"))
    ratio = v/v3 if (v3 and v3>0) else float("nan")
    carry = (v3-v)/v*100 if (v and v>0 and np.isfinite(v3)) else float("nan")
    shape = ("BACKWARDATION" if (np.isfinite(ratio) and ratio>1.05)
             else "CONTANGO" if (np.isfinite(ratio) and ratio<0.95) else "MIXED")
    return {"ratio":ratio,"carry":carry,"shape":shape}

def _tail(q: Dict) -> Dict:
    vvix=q.get("VVIX_last",float("nan")); vix=q.get("VIX_last",20.0)
    ratio = vvix/vix if (np.isfinite(vvix) and vix>0) else float("nan")
    reg   = ("ELEVATED" if (np.isfinite(ratio) and ratio>5.5)
             else "MODERATE" if (np.isfinite(ratio) and ratio>4.5) else "LOW")
    return {"vvix":vvix,"ratio":ratio,"regime":reg}

def _retdist(spy: pd.Series, idx, spot: float) -> Dict:
    sa   = _to_1d(spy).reindex(idx).ffill()
    rets = sa.pct_change().dropna()
    if len(rets)<30: return {}
    ds = float(rets.rolling(21,min_periods=10).std().iloc[-1])*100
    sk = float(_scipy_skew(rets.tail(252)))
    ku = float(_scipy_kurt(rets.tail(252), fisher=True))
    return {"daily_sigma":ds,"skew":sk,"kurtosis":ku}

def _merton_fig(spot: float, vix: float, days=5, n=4000) -> go.Figure:
    np.random.seed(42)
    dt_s=1/252; sig=vix/100
    lam,mj,sj=0.10,-0.015,0.025
    paths=np.zeros((n,days+1)); paths[:,0]=spot
    for t in range(1,days+1):
        z=np.random.standard_normal(n); nj=np.random.poisson(lam*dt_s,n)
        j=np.random.normal(mj,sj,n)*nj
        paths[:,t]=paths[:,t-1]*np.exp((-0.5*sig**2)*dt_s+sig*np.sqrt(dt_s)*z+j)
    lo=spot*0.93; hi=spot*1.07; lvls=np.linspace(lo,hi,24); bkt=(hi-lo)/24
    Z=np.zeros((24,days))
    for d in range(days):
        f=paths[:,d+1]
        for i,lv in enumerate(lvls):
            Z[i,d]=np.mean((f>=lv)&(f<lv+bkt))*100
    y_labels=[f"{lv:.0f}" for lv in lvls]
    fig=go.Figure(go.Heatmap(z=Z,x=[f"Day {i+1}" for i in range(days)],
        y=y_labels,colorscale="Viridis",showscale=True,
        colorbar=dict(title="Prob%",thickness=12)))
    # Categorical y-axis: add_hline fails. Use add_shape with paper coords instead.
    closest_idx=int(np.argmin(np.abs(lvls-spot)))
    y_paper=closest_idx/max(len(lvls)-1,1)
    fig.add_shape(type="line",xref="paper",yref="paper",
                  x0=0,x1=1,y0=y_paper,y1=y_paper,
                  line=dict(color="white",width=1.5,dash="dash"))
    fig.add_annotation(xref="paper",yref="paper",x=0.01,y=y_paper,
                       text=f"Spot {spot:.0f}",showarrow=False,
                       font=dict(color="white",size=10),
                       xanchor="left",yanchor="bottom")
    plotly_dark(fig,title="Probability Heatmap — Merton Jump-Diffusion (1-week)",height=420)
    fig.update_layout(xaxis_title="Trading Day Forward",yaxis_title="Price Level")
    return fig

def _ivsurf_fig(vix: float, label: str) -> go.Figure:
    ms=np.linspace(0.88,1.12,15); dtes=np.array([7,14,21,45,90,180])
    Z=np.zeros((len(ms),len(dtes)))
    for i,m in enumerate(ms):
        for j,d in enumerate(dtes):
            mn=m-1.0
            Z[i,j]=max((vix/100-0.30*mn-0.002*d)*100,1.0)
    fig=go.Figure(go.Surface(x=dtes,y=[f"{m*100:.0f}%" for m in ms],z=Z,
        colorscale="RdYlGn_r",showscale=True,colorbar=dict(title="IV%",thickness=12),opacity=0.90))
    plotly_dark(fig,title=f"IV Surface — {label} (Model)",height=380)
    fig.update_layout(scene=dict(xaxis_title="DTE",yaxis_title="Moneyness",
                                  zaxis_title="IV%",bgcolor="rgba(0,0,0,0)"))
    return fig

def _ivrv_fig(vix_s: pd.Series, spy: pd.Series, idx) -> go.Figure:
    sa=_to_1d(spy).reindex(idx).ffill(); va=_to_1d(vix_s).reindex(idx).ffill()
    rv21=(sa.pct_change().rolling(21,min_periods=10).std()*np.sqrt(252)*100)
    vrp=(va-rv21).dropna(); dates=vrp.index[-252:]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=va.reindex(dates),name="VIX",
                             line=dict(color="#6366f1",width=1.8)))
    fig.add_trace(go.Scatter(x=dates,y=rv21.reindex(dates),name="21D RV",
                             line=dict(color="#10b981",width=1.8)))
    vs=vrp.reindex(dates)
    fig.add_trace(go.Bar(x=dates,y=vs,name="VRP",yaxis="y2",opacity=0.55,
                         marker_color=["#10b981" if v>=0 else "#ef4444" for v in vs]))
    plotly_dark(fig,title="IV vs Realized Volatility — 1 Year",height=340)
    fig.update_layout(yaxis=dict(title="Vol (%)"),
                      yaxis2=dict(title="VRP",overlaying="y",side="right",showgrid=False),
                      legend=dict(orientation="h",y=1.02))
    return fig

def _rdist_fig(spy: pd.Series, idx, rd: Dict) -> go.Figure:
    sa=_to_1d(spy).reindex(idx).ffill()
    rets=sa.pct_change().dropna().tail(252)*100
    sig=rd.get("daily_sigma",1.0)
    fig=go.Figure()
    fig.add_trace(go.Histogram(x=rets,nbinsx=60,name="Actual",
                               marker_color="#6366f1",opacity=0.75,histnorm="probability density"))
    xn=np.linspace(float(rets.min()),float(rets.max()),200)
    yn=np.exp(-0.5*(xn/sig)**2)/(sig*np.sqrt(2*np.pi))
    fig.add_trace(go.Scatter(x=xn,y=yn,name="Normal",
                             line=dict(color="#f59e0b",width=1.8,dash="dot")))
    for m,col in [(1,"#10b981"),(2,"#f59e0b")]:
        for s in [-1,1]:
            fig.add_vline(x=s*m*sig,line_dash="dash",line_color=col,line_width=1)
    plotly_dark(fig,title="Return Distribution & Probability Bands (1Y)",height=320)
    fig.update_layout(xaxis_title="Daily Return (%)",yaxis_title="Density")
    return fig

def _rec_prob(sahm: float, puts: float, hy: float) -> float:
    return round(min(np.clip(sahm/0.5*60,0,60)+np.clip((hy-300)/700*25,0,25)+np.clip((100-puts)/100*15,0,15),99),1)

def _composite(prob: dict, fear: float, vrp_val: float, gex: GammaRegime) -> int:
    s=int(round((prob.get("bull_prob",50)-50)/12.5))
    if fear>70: s-=2
    elif fear>55: s-=1
    elif fear<35: s+=1
    if np.isfinite(vrp_val):
        if vrp_val>3: s-=1
        elif vrp_val<-2: s+=1
    if gex==GammaRegime.STRONG_NEGATIVE: s-=2
    elif gex==GammaRegime.NEGATIVE: s-=1
    elif gex==GammaRegime.STRONG_POSITIVE: s+=1
    return int(np.clip(s,-10,10))

def _verdict(c: int, gex: GammaRegime) -> Tuple[str,str,str]:
    neg=gex in (GammaRegime.NEGATIVE,GammaRegime.STRONG_NEGATIVE)
    pos=gex in (GammaRegime.POSITIVE,GammaRegime.STRONG_POSITIVE)
    if c<=-4: return "BEARISH","#ef4444","Multiple signals aligned bearish."
    if c<=-2:
        return (("BEARISH","#ef4444","Negative macro + negative gamma.") if neg
                else ("LEANING BEARISH","#f97316","Modest bearish lean."))
    if c<=1:
        if neg: return "CAUTIOUS","#f59e0b","Neutral signals but negative gamma — tail risk elevated."
        if pos: return "NEUTRAL / RANGE","#6366f1","Positive gamma → compression and pin."
        return "NEUTRAL","#94a3b8","No strong conviction."
    if c<=3: return "LEANING BULLISH","#10b981","Bullish lean. Credit and liquidity constructive."
    return "BULLISH","#10b981","Broad bullish alignment."

def _bands(spot: float, vix: float) -> Dict:
    dv=vix/100/np.sqrt(252); wv=vix/100/np.sqrt(52)
    return {k:round(v,2) for k,v in {
        "d1lo":spot*(1-dv),"d1hi":spot*(1+dv),
        "d2lo":spot*(1-2*dv),"d2hi":spot*(1+2*dv),
        "w1lo":spot*(1-wv),"w1hi":spot*(1+wv),
        "w2lo":spot*(1-2*wv),"w2hi":spot*(1+2*wv)}.items()}

def _news_cats(cat_intel: dict) -> List[Dict]:
    out=[]
    for k,items in cat_intel.items():
        if not items: continue
        m=INTEL_CATEGORIES.get(k,{})
        sh=category_shock_score(items)
        out.append({"label":m.get("label",k),"icon":m.get("icon","📰"),
                    "sentiment":round(-(sh-30)/70,4),"count":len(items)})
    return sorted(out,key=lambda x:abs(x["sentiment"]),reverse=True)

# ── HTML helpers ──────────────────────────────────────────────────────────────

def _card(body,bg="rgba(255,255,255,0.03)",border="rgba(255,255,255,0.10)"):
    return (f"<div style='background:{bg};border:1px solid {border};border-radius:12px;"
            f"padding:16px 20px;margin-bottom:14px;'>{body}</div>")

def _sh(n,t):
    return (f"<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);"
            f"letter-spacing:0.15em;text-transform:uppercase;margin-bottom:8px;'>{n}. {t}</div>")

def _kv(label,value,color="rgba(255,255,255,0.85)"):
    return (f"<div style='display:flex;justify-content:space-between;align-items:baseline;"
            f"margin-bottom:3px;font-size:13px;'>"
            f"<span style='color:rgba(255,255,255,0.50);'>{label}</span>"
            f"<span style='font-family:monospace;font-weight:600;color:{color};'>{value}</span></div>")

def _tk(sym,price,pct):
    if not np.isfinite(price): return ""
    pc=("#10b981" if pct>=0 else "#ef4444") if np.isfinite(pct) else "#94a3b8"
    ps=(f"+{pct:.2f}%" if pct>=0 else f"{pct:.2f}%") if np.isfinite(pct) else ""
    return (f"<div style='text-align:center;'>"
            f"<div style='font-size:10px;color:rgba(255,255,255,0.4);'>{sym}</div>"
            f"<div style='font-family:monospace;font-size:14px;font-weight:700;'>{price:,.2f}</div>"
            f"<div style='font-size:11px;color:{pc};font-weight:600;'>{ps}</div></div>")

def _pill(text,color="#6366f1"):
    return (f"<span style='background:{color}22;color:{color};border:1px solid {color}44;"
            f"border-radius:6px;padding:2px 8px;font-size:11px;font-weight:600;"
            f"margin-right:4px;display:inline-block;'>{text}</span>")

def _sig(e,t):
    return f"<div style='font-size:12px;margin-bottom:3px;'>{e} {t}</div>"

def _gl(term,defn):
    return (f"<div style='margin-bottom:10px;'>"
            f"<div style='font-size:13px;font-weight:700;color:rgba(255,255,255,0.9);margin-bottom:2px;'>{term}</div>"
            f"<div style='font-size:12px;color:rgba(255,255,255,0.55);line-height:1.55;'>{defn}</div></div>")

# ── Main ──────────────────────────────────────────────────────────────────────

def render_thesis_page():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## 📋 Daily Thesis Briefing")
    st.markdown(f"*{dt.datetime.now().strftime('%A, %B %d, %Y — %H:%M ET')}*  |  Not financial advice.")

    st.sidebar.markdown("### Thesis Controls")
    start=st.sidebar.date_input("Data Start",dt.date.today()-dt.timedelta(days=730),key="th_start")
    end=st.sidebar.date_input("Data End",dt.date.today(),key="th_end")
    gex_sym=st.sidebar.text_input("GEX Symbol","SPY",key="th_gex").upper().strip()
    show_ua=st.sidebar.toggle("Show ES/NQ levels",True,key="th_ua")
    if st.sidebar.button("🔄 Refresh",use_container_width=True,key="th_ref"):
        st.cache_data.clear(); st.rerun()

    idx=pd.date_range(start,end,freq="D")

    with st.spinner("Loading macro data…"):
        raw=load_macro(start.isoformat(),end.isoformat())
        def r(k): return resample_ffill(raw.get(k,pd.Series(dtype=float)),idx)
        y3m=r("DGS3MO");y2=r("DGS2");y10=r("DGS10");y30=r("DGS30")
        cpi=r("CPIAUCSL");core=r("CPILFESL");unrate=r("UNRATE")
        walcl=r("WALCL");tga=r("WTREGEN");rrp=r("RRPONTSYD");m2=r("M2SL")
        nfci=resample_ffill(raw.get("NFCI",pd.Series(dtype=float)),idx).fillna(0)
        vix_s=r("VIX");spy=r("SPY");tlt=r("TLT");qqq=r("QQQ")
        copx=r("COPX");gld=r("GLD");hyg=r("HYG");lqd=r("LQD");dxy=r("UUP");iwm=r("IWM")
        tips=r("DFII10");bres=r("WRBWFRBL");bcred=r("TOTBKCR");gdp=r("GDPC1");mmmf=r("WRMFSL")
        ism_r=raw.get("AMTMNO",pd.Series(dtype=float))
        ism=ism_r if len(ism_r.dropna())>4 else None
        sahm_r=raw.get("SAHM_RULE",pd.Series(dtype=float))
        hy_r=raw.get("BAMLH0A0HYM2",pd.Series(dtype=float))
        sahm=resample_ffill(sahm_r,idx) if len(sahm_r.dropna())>0 else None
        hys=resample_ffill(hy_r,idx) if len(hy_r.dropna())>0 else None

    core_yoy=(core/core.shift(365)-1)*100
    cpi_yoy=(cpi/cpi.shift(365)-1)*100
    s2s10=(y10-y2)*100
    net_liq=(walcl-tga-rrp)/1000
    nl4w=net_liq.diff(28); bs13=walcl.diff(91)/1000
    cyl=_sl(core_yoy,2.5); crl=_sl(s2s10,0.0)
    macro_reg=classify_macro_regime_abs(cyl,crl)
    gz=zscore(s2s10.fillna(0)); iz=zscore(core_yoy.fillna(cyl))
    vz=zscore(vix_s.fillna(20)); nz=zscore(nfci.fillna(0))
    inv=(s2s10<0).astype(int); lt=(nl4w<0).astype(int)
    fear_raw=0.45*vz+0.35*nz+0.10*inv+0.10*lt
    fear=float(((fear_raw.iloc[-1]+2)/4).clip(0,1)*100)
    vl=_sl(vix_s,20.0); sahmv=_sl(sahm,0.0) if sahm is not None else 0.0
    hyv=_sl(hys,300.0) if hys is not None else 300.0
    ur=_sl(unrate,4.0); cyi=_sl(cpi_yoy,2.5); gzv=_sl(gz,0.0); izv=_sl(iz,0.0)
    nl4wv=_sl(nl4w,0.0)
    liq_lab="Expanding" if nl4wv>=0 else "Contracting"
    y10_20=y10.diff(20); u3m=float(unrate.diff(90).iloc[-1]) if len(unrate)>90 else 0.0
    warsh=((y10.diff(20)<0)&(bs13<0)).astype(int)
    spydd=(spy/spy.rolling(126).max()-1).fillna(0)
    tp=float(np.clip(45+35*float(spydd.iloc[-1]<=-0.07)+20*(fear>60),0,100))
    fp=float(np.clip(55+25*float((y10_20.iloc[-1]<0)and(u3m>0))-10*float((cyl-3.0)>0)-15*float(warsh.iloc[-1]),0,100))
    tgadd=(tga.diff(28)<0).astype(int); rrpd=(rrp<50).astype(int)
    trp=float(np.clip(50+20*float(tgadd.iloc[-1])+15*float(rrpd.iloc[-1])+15*float(nl4w.iloc[-1]>=0),0,100))
    threeP=float(np.clip(0.35*trp+0.35*fp+0.30*tp,0,100))
    rec=_rec_prob(sahmv,threeP,hyv)
    sr=_to_1d(spy).reindex(idx).ffill().pct_change().dropna()
    tr2=_to_1d(tlt).reindex(idx).ffill().pct_change().reindex(sr.index).dropna()
    stlc=round(float(sr.rolling(21).corr(tr2).dropna().iloc[-1]),3) if sr.dropna().size>21 else float("nan")
    cpi_now=round(float(cpi.pct_change(21).dropna().iloc[-1])*100,3) if cpi.pct_change(21).dropna().size else float("nan")
    rd=_retdist(spy,idx,0)

    with st.spinner("Fetching live quotes…"):
        q=_quotes()

    spx=q.get("SPX_last",_sl(spy)*10); ndx=q.get("NDX_last",_sl(qqq)*40)
    es=q.get("ES_last",spx); nq=q.get("NQ_last",ndx)
    dxyv=q.get("DXY_last",float("nan")); gldv=q.get("GLD_last",float("nan"))
    tnxv=q.get("TNX_last",float("nan"))
    tnxv=tnxv/10 if (np.isfinite(tnxv) and tnxv>20) else tnxv
    irxv=q.get("IRX_last",float("nan"))
    irxv=irxv/10 if (np.isfinite(irxv) and irxv>20) else irxv
    s2s10v=(tnxv-irxv)*100 if (np.isfinite(tnxv) and np.isfinite(irxv)) else crl
    vix_live=q.get("VIX_last",vl)
    vrp=_vrp_full(vix_live,spy,idx)
    vts=_vts(q); tail=_tail(q)
    rd=_retdist(spy,idx,spx)
    b=_bands(spx,vix_live)

    gex_spot=spx if gex_sym in ("SPY","SPX") else ndx
    with st.spinner("Fetching options data…"):
        client=get_schwab_client(); chain_df=None
        if client:
            chain_df=schwab_get_options_chain(client,gex_sym,spot=None)
            if chain_df is not None and len(chain_df)>0:
                gex_spot=schwab_get_spot(client,gex_sym) or gex_spot
        if chain_df is None or len(chain_df)==0:
            chain_df,gex_spot,_=get_gex_from_yfinance(gex_sym)
        if chain_df is not None and gex_spot:
            gex_st=build_gamma_state(chain_df,float(gex_spot),"live",max_dte=45)
            gwas=compute_gwas(chain_df,float(gex_spot))
            tstr=compute_gex_term_structure(chain_df,float(gex_spot))
            fl=compute_flow_imbalance(chain_df,float(gex_spot))
            net_gex=gex_st.total_gex; gex_score=int(np.clip(net_gex/1e9*10,-50,50))
        else:
            gex_st=GammaState(data_source="unavailable",timestamp=dt.datetime.now().strftime("%H:%M:%S"))
            gwas=tstr=fl={}; net_gex=gex_score=0

    flip=gex_st.gamma_flip or gex_spot
    upper=gex_st.key_resistance[0] if gex_st.key_resistance else gex_spot*1.03
    lower=gex_st.key_support[0] if gex_st.key_support else gex_spot*0.97
    gex_reg=gex_st.regime.value if hasattr(gex_st.regime,"value") else str(gex_st.regime)
    gex_rc=("#ef4444" if "NEGATIVE" in gex_reg.upper()
            else "#10b981" if "POSITIVE" in gex_reg.upper() else "#f59e0b")
    gwas_a=gwas.get("gwas_above") if gwas else None
    gwas_b=gwas.get("gwas_below") if gwas else None
    dur=tstr.get("durability","N/A").upper() if tstr else "N/A"
    frag=tstr.get("fragility_ratio",0.5)*100 if tstr else 50.0
    pcr=fl.get("pc_ratio",1.0) if fl else 1.0
    fb=fl.get("flow_bias","neutral") if fl else "neutral"

    with st.spinner("Computing signals…"):
        leading=compute_leading_stack(
            y2,y3m,y10,y30,s2s10,vix_s,m2,pd.Series(dtype=float),
            copx,gld,hyg,lqd,dxy,spy,qqq,iwm,net_liq,nl4w,walcl,bs13,idx,
            tips_10y=tips,bank_reserves=bres,bank_credit=bcred,ism_no=ism,gdp_quarterly=gdp,mmmf=mmmf)
        meta=regime_transition_prob(macro_reg,core_yoy,s2s10)
        nc=float(current_pct_rank(-_to_1d(nfci).reindex(idx).ffill(),252))
        lc=float(50.0+np.sign(nl4wv)*20)
    with st.spinner("Loading feeds…"):
        try:
            rss=load_feeds(tuple(_all_feeds_flat().items()),60)
            geo,_=geo_shock_score(rss); cat_intel=categorise_items(rss)
        except Exception:
            geo=0.0; cat_intel={k:[] for k in INTEL_CATEGORIES}
    prob=compute_prob_composite(leading,fear,geo,meta["p_change_20d"],gex_st,nfci_coincident=nc,liq_dir_coincident=lc)
    p1d=compute_1d_prob(gex_state=gex_st,spot=float(gex_spot),vix_level=vix_live,
        vix_series=vix_s,spy_series=spy,hyg_series=hyg,lqd_series=lqd,dxy_series=dxy,
        s_2s10s=s2s10,net_liq_4w=nl4w,nfci_z=nz,fear_score=fear,
        session=get_session_context(),idx=idx,sahm_rule=sahm,hy_spread=hys)

    comp=_composite(prob,fear,vrp["val"],gex_st.regime)
    vrd,vc,ve=_verdict(comp,gex_st.regime)
    news=_news_cats(cat_intel)
    ua="NQ" if gex_sym in ("QQQ","NDX") else "ES"
    um=40.0 if ua=="NQ" else 10.0
    def _ua(p): return f"{p*um:,.0f}"
    reg_col={"Goldilocks":"#10b981","Overheating":"#f59e0b","Stagflation":"#ef4444","Deflation":"#6366f1"}.get(macro_reg,"#94a3b8")
    fl2="ELEVATED" if fear>60 else "MODERATE" if fear>40 else "LOW"
    fc="#ef4444" if fear>60 else "#f59e0b" if fear>40 else "#10b981"

    # ── 1. MARKET REGIME ──────────────────────────────────────────────────
    hdr=(f"<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px;'>"
         +_pill(f"Market Regime: {macro_reg}",reg_col)
         +_pill(f"Fear Level: {fl2} ({fear:.2f})",fc)
         +_pill(f"Liquidity: {liq_lab} (${abs(nl4wv):.0f}B)","#10b981" if nl4wv>=0 else "#ef4444")
         +_pill(f"Recession P(6m): {rec:.1f}%","#ef4444" if rec>50 else "#f59e0b" if rec>30 else "#10b981")
         +"</div>")
    strip=("<div style='display:grid;grid-template-columns:repeat(9,1fr);gap:6px;'>"
           +_tk("SPX",spx,q.get("SPX_pct",float("nan")))
           +_tk("NDX",ndx,q.get("NDX_pct",float("nan")))
           +_tk("VIX",vix_live,q.get("VIX_pct",float("nan")))
           +_tk("ES",es,q.get("ES_pct",float("nan")))
           +_tk("NQ",nq,q.get("NQ_pct",float("nan")))
           +_tk("DXY",dxyv,q.get("DXY_pct",float("nan")))
           +_tk("10Y",tnxv,float("nan"))
           +_tk("2s10s",s2s10v,float("nan"))
           +_tk("GLD",gldv,q.get("GLD_pct",float("nan")))
           +"</div>")
    st.markdown(_card(_sh(1,"MARKET REGIME")+hdr+strip),unsafe_allow_html=True)

    # ── 2. VOLATILITY REGIME ──────────────────────────────────────────────
    vc2=("#ef4444" if vrp["regime"]=="CHEAP" else "#10b981" if vrp["regime"]=="RICH" else "#94a3b8")
    tc=("#ef4444" if vts["shape"]=="BACKWARDATION" else "#10b981" if vts["shape"]=="CONTANGO" else "#f59e0b")
    trc=("#ef4444" if tail["regime"]=="ELEVATED" else "#f59e0b" if tail["regime"]=="MODERATE" else "#10b981")
    vol=(  _sh(2,"VOLATILITY REGIME")
         +"<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px 32px;'><div>"
         +"<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>VRP</div>"
         +_kv("Value",f"{vrp['val']:+.4f}" if np.isfinite(vrp['val']) else "N/A",vc2)
         +_kv("Z-Score",f"{vrp['z']:.2f}" if np.isfinite(vrp['z']) else "N/A")
         +_kv("Pct",f"{vrp['pct']:.0f}%" if np.isfinite(vrp['pct']) else "N/A")
         +_kv("Regime",vrp["regime"],vc2)
         +"<div style='margin-top:8px;font-size:10px;color:rgba(255,255,255,0.4);letter-spacing:0.1em;'>TERM STRUCTURE</div>"
         +_kv("Shape",vts["shape"],tc)
         +_kv("VIX/VIX3M",f"{vts['ratio']:.3f}" if np.isfinite(vts.get('ratio',float('nan'))) else "N/A")
         +_kv("Carry",f"{vts['carry']:.2f}%" if np.isfinite(vts.get('carry',float('nan'))) else "N/A")
         +"<div style='margin-top:8px;font-size:10px;color:rgba(255,255,255,0.4);letter-spacing:0.1em;'>TAIL RISK</div>"
         +_kv("VVIX",f"{tail['vvix']:.1f}" if np.isfinite(tail['vvix']) else "N/A")
         +_kv("VVIX/VIX",f"{tail['ratio']:.2f}" if np.isfinite(tail['ratio']) else "N/A",trc)
         +_kv("Regime",tail["regime"],trc)
         +"</div><div>"
         +"<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>IV vs RV</div>"
         +_kv("VIX (IV)",f"{vix_live:.2f}")
         +_kv("21D RV",f"{vrp['rv21']:.2f}%" if np.isfinite(vrp['rv21']) else "N/A")
         +_kv("VRP Spread",f"{vrp['spread']:+.2f}" if np.isfinite(vrp['spread']) else "N/A",vc2)
         +"<div style='margin-top:8px;font-size:10px;color:rgba(255,255,255,0.4);letter-spacing:0.1em;'>ATM IV</div>"
         +_kv("SPX ATM IV",f"{vix_live:.1f}%")
         +_kv("NDX ATM IV",f"{vix_live*0.88:.1f}%")
         +"</div></div>")
    st.markdown(_card(vol),unsafe_allow_html=True)

    # ── 3. GEX & DEALER POSITIONING ───────────────────────────────────────
    if "NEGATIVE" in gex_reg.upper():
        narr=f"Negative dealer flow ({gex_score:+d}): Dealers short gamma — amplifying moves. Expect trending/volatile behavior."
    elif "POSITIVE" in gex_reg.upper():
        narr=f"Positive dealer flow ({gex_score:+d}): Dealers long gamma — suppressing moves. Expect mean-reversion."
    else:
        narr=f"Neutral dealer positioning ({gex_score:+d}): Near gamma flip — binary risk, reduce size."

    gleft=("<div>"
           +f"<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>GEX LEVELS ({gex_sym})</div>"
           +_kv(f"{gex_sym} Spot",f"{float(gex_spot):.2f}","#fff")
           +_kv("GEX Flip",f"{flip:.2f}","#f59e0b")
           +_kv("GEX Upper",f"{upper:.2f}","#10b981")
           +_kv("GEX Lower",f"{lower:.2f}","#ef4444")
           +_kv("GWAS Above",f"{gwas_a:.2f}" if gwas_a else "N/A","#6366f1")
           +_kv("GWAS Below",f"{gwas_b:.2f}" if gwas_b else "N/A","#6366f1")
           +_kv("GEX Regime",gex_reg,gex_rc)
           +"</div>")
    gright=("<div>"
            +"<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>OPTIONS FLOW</div>"
            +_kv("P/C Dollar Ratio",f"{pcr:.2f}","#ef4444" if pcr>1.3 else "#10b981" if pcr<0.8 else "#94a3b8")
            +_kv("Flow Bias",fb.upper(),"#ef4444" if fb=="bearish" else "#10b981" if fb=="bullish" else "#94a3b8")
            +_kv("GEX Duration",f"{dur} ({frag:.0f}% weekly)","#ef4444" if dur=="FRAGILE" else "#10b981")
            +_kv("Net GEX",f"${gex_st.total_gex/1e9:.2f}B" if gex_st.total_gex else "N/A")
            +_kv("Dist to Flip",f"{gex_st.distance_to_flip_pct:+.2f}%"))
    if show_ua:
        gright+=("<div style='margin-top:8px;border-top:1px solid rgba(255,255,255,0.08);padding-top:6px;'>"
                 +f"<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>{ua} EQUIVALENT</div>"
                 +_kv(f"{ua} Spot",_ua(float(gex_spot)))
                 +_kv(f"{ua} Flip",_ua(flip),"#f59e0b")
                 +_kv(f"{ua} Upper",_ua(upper),"#10b981")
                 +_kv(f"{ua} Lower",_ua(lower),"#ef4444")
                 +"</div>")
    gright+="</div>"
    gex_body=(_sh(3,"GEX LEVELS & DEALER POSITIONING")
              +"<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px 32px;'>"
              +gleft+gright+"</div>"
              +f"<div style='margin-top:10px;border-top:1px solid rgba(255,255,255,0.08);padding-top:8px;"
              +f"font-size:12px;color:rgba(255,255,255,0.65);font-style:italic;'>"
              +f"<span style='color:{gex_rc};font-weight:700;'>{gex_reg}</span> — {narr}</div>")
    st.markdown(_card(gex_body),unsafe_allow_html=True)

    # ── 4. PROBABILITY HEATMAP ────────────────────────────────────────────
    st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>4. PROBABILITY HEATMAP — MC Simulation</div>",unsafe_allow_html=True)
    st.caption("1-week forward | Merton jump-diffusion | SPX")
    st.plotly_chart(_merton_fig(spx,vix_live),use_container_width=True,key="th_hm")

    # ── 5 & 6. IV SURFACES ────────────────────────────────────────────────
    c5,c6=st.columns(2)
    with c5:
        st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>5. IMPLIED VOL SURFACE — SPX</div>",unsafe_allow_html=True)
        st.plotly_chart(_ivsurf_fig(vix_live,"SPX"),use_container_width=True,key="th_ivs_spx")
    with c6:
        st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>6. IMPLIED VOL SURFACE — NDX</div>",unsafe_allow_html=True)
        st.plotly_chart(_ivsurf_fig(vix_live*0.92,"NDX"),use_container_width=True,key="th_ivs_ndx")

    # ── 7. IV vs RV ───────────────────────────────────────────────────────
    st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>7. IMPLIED vs REALIZED VOLATILITY</div>",unsafe_allow_html=True)
    st.plotly_chart(_ivrv_fig(vix_s,spy,idx),use_container_width=True,key="th_ivrv")

    # ── 8. RETURN DISTRIBUTION ────────────────────────────────────────────
    st.markdown("<div style='font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px;'>8. RETURN DISTRIBUTION & PROBABILITY BANDS</div>",unsafe_allow_html=True)
    if rd:
        st.markdown(f"SPX Spot: `{spx:,.2f}` &nbsp;&nbsp; Daily σ: `{rd['daily_sigma']:.4f}%` &nbsp;&nbsp; Skew: `{rd['skew']:.4f}` &nbsp;&nbsp; Kurtosis: `{rd['kurtosis']:.4f}`")
        st.plotly_chart(_rdist_fig(spy,idx,rd),use_container_width=True,key="th_rd")

    # ── 9. MACRO REGIME & NEWS ────────────────────────────────────────────
    nr=""
    for cat in news[:6]:
        s=cat["sentiment"]; col=("#10b981" if s>0.01 else "#ef4444" if s<-0.01 else "#94a3b8")
        sign="+" if s>0 else ""
        nr+=(f"<span style='font-size:11px;font-family:monospace;margin-right:10px;'>"
             f"<span style='color:rgba(255,255,255,0.5);'>{cat['icon']} {cat['label'].split('&')[0].strip().lower()}</span>: "
             f"<span style='color:{col};font-weight:600;'>{sign}{s:.4f}</span> "
             f"<span style='color:rgba(255,255,255,0.3);'>({cat['count']} articles)</span></span>")
    mac=(_sh(9,"MACRO REGIME & NEWS SENTIMENT")
         +f"<div style='font-size:16px;font-weight:700;color:{reg_col};margin-bottom:8px;'>{macro_reg}</div>"
         +"<div style='display:grid;grid-template-columns:1fr 1fr;gap:4px 32px;'><div>"
         +_kv("Growth Z",f"{gzv:+.2f}","#10b981" if gzv>0 else "#ef4444")
         +_kv("Inflation Z",f"{izv:+.2f}","#f59e0b" if izv>0.5 else "#94a3b8")
         +_kv("CPI YoY",f"{cyi:.2f}%")
         +_kv("CPI Nowcast",f"{cpi_now:+.3f}% MoM" if np.isfinite(cpi_now) else "N/A")
         +"</div><div>"
         +_kv("Unemployment",f"{ur:.1f}%")
         +_kv("HY OAS",f"{hyv:.2f}","#ef4444" if hyv>450 else "#94a3b8")
         +_kv("SPY-TLT Corr",f"{stlc:.3f}" if np.isfinite(stlc) else "N/A",
              "#ef4444" if (np.isfinite(stlc) and stlc>0.2) else "#94a3b8")
         +_kv("Sahm Rule",f"{sahmv:.3f}",
              "#ef4444" if sahmv>=0.5 else "#f59e0b" if sahmv>=0.3 else "#10b981")
         +"</div></div>"
         +f"<div style='margin-top:10px;border-top:1px solid rgba(255,255,255,0.08);padding-top:8px;'>"
         +"<div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:5px;letter-spacing:0.1em;'>NEWS SENTIMENT</div>"
         +f"<div style='display:flex;flex-wrap:wrap;gap:4px;'>{nr}</div></div>")
    st.markdown(_card(mac),unsafe_allow_html=True)

    # ── 10. THESIS VERDICT ────────────────────────────────────────────────
    cc="#10b981" if comp>0 else "#ef4444" if comp<0 else "#94a3b8"
    vp=np.isfinite(vrp["val"]) and vrp["val"]>0
    vs2=f"{vrp['val']:+.4f}" if np.isfinite(vrp["val"]) else "N/A"
    sigs=(_sig("✅" if vp else "⚠️",f"VRP {'positive' if vp else 'negative'} ({vs2})")
          +_sig("⚠️" if fear>55 else "✅",f"Fear composite {fl2}")
          +_sig("🔴" if rec>60 else "🟡" if rec>35 else "🟢",
                f"Recession risk {'elevated' if rec>60 else 'moderate' if rec>35 else 'low'} ({rec:.1f}%)")
          +_sig("🔴" if "NEGATIVE" in gex_reg.upper() else "🟢",f"GEX: {gex_reg}")
          +_sig("📊",f"Dominant 1D: {p1d.get('dominant_signal','—')} ({p1d.get('dominant_dir','neutral')})"))
    kls=(_kv("SPX Spot",f"{spx:,.2f}","#fff")
         +_kv("GEX Flip",f"{flip:,.2f}","#f59e0b")
         +_kv("GEX Upper",f"{upper:,.2f}","#10b981")
         +_kv("GEX Lower",f"{lower:,.2f}","#ef4444")
         +_kv("1σ Daily",f"{b['d1lo']:,.2f} — {b['d1hi']:,.2f}")
         +_kv("2σ Daily",f"{b['d2lo']:,.2f} — {b['d2hi']:,.2f}")
         +_kv("1σ Weekly",f"{b['w1lo']:,.2f} — {b['w1hi']:,.2f}")
         +_kv("2σ Weekly",f"{b['w2lo']:,.2f} — {b['w2hi']:,.2f}"))
    risks=[]
    if fear>60: risks.append(("⚠️","Elevated fear composite — potential for sharp moves"))
    if rec>50: risks.append(("🔴",f"Recession probability at {rec:.1f}% — monitor labor data"))
    if np.isfinite(stlc) and stlc>0.2: risks.append(("⚠️","Positive stock-bond correlation — diversification impaired"))
    if "NEGATIVE" in gex_reg.upper(): risks.append(("🔴","Negative gamma regime — dealer hedging amplifies moves. No fading."))
    if dur=="FRAGILE": risks.append(("⚠️",f"GEX regime fragile — {frag:.0f}% of gamma ≤7 DTE. Levels expire by Friday."))
    if leading.get("corr_regime") in ("STRESS","SYSTEMIC"): risks.append(("🔴",f"Cross-asset correlation: {leading.get('corr_regime')} — credit leading equity lower"))
    if vts["shape"]=="BACKWARDATION": risks.append(("⚠️","VIX backwardation — near-term stress priced above medium-term"))
    if not risks: risks.append(("✅","No major risk flags. Conditions broadly constructive."))
    rr="".join(_sig(e,t) for e,t in risks)
    vbd=(_sh(10,f"THESIS VERDICT: {vrd}")
         +f"<div style='display:flex;align-items:baseline;gap:16px;margin-bottom:8px;'>"
         +f"<div style='font-size:26px;font-weight:800;color:{vc};'>{vrd}</div>"
         +f"<div style='font-size:13px;color:rgba(255,255,255,0.5);'>Composite Score: <span style='font-family:monospace;font-weight:700;color:{cc};'>{comp:+d}</span> / ±10</div>"
         +f"<div style='font-size:12px;color:rgba(255,255,255,0.35);'>Date: {dt.date.today().strftime('%A, %B %d, %Y')}</div></div>"
         +f"<div style='font-size:12px;color:rgba(255,255,255,0.55);margin-bottom:10px;font-style:italic;'>{ve}</div>"
         +"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px 24px;'>"
         +"<div><div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>SIGNAL BREAKDOWN</div>"+sigs+"</div>"
         +"<div><div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>KEY LEVELS</div>"+kls+"</div>"
         +"<div><div style='font-size:10px;color:rgba(255,255,255,0.4);margin-bottom:4px;letter-spacing:0.1em;'>RISK FACTORS</div>"+rr+"</div>"
         +"</div>")
    st.markdown(_card(vbd,bg=f"{vc}08",border=f"{vc}30"),unsafe_allow_html=True)

    # ── 11-13. GLOSSARIES ─────────────────────────────────────────────────
    g11,g12,g13=st.tabs(["📖 Glossary: Vol","📖 Glossary: Macro","📖 Glossary: Charts"])
    with g11:
        st.markdown(_card(
            _sh(11,"GLOSSARY — WHAT EVERYTHING MEANS")
            +_gl("VRP (Variance Risk Premium)",
                 "Difference between implied vol (VIX) and realized vol. Positive VRP = options expensive vs actual moves. "
                 "Traders sell vol when VRP is high. Negative VRP = market underpricing realized moves → buy protection.")
            +_gl("GEX (Gamma Exposure)",
                 "Total gamma held by options dealers. Positive GEX = dealers long gamma → buy dips, sell rips → suppresses vol. "
                 "Negative GEX = dealers short gamma → amplify moves. GEX Flip = strike where dealer gamma flips sign.")
            +_gl("VIX Term Structure",
                 "Curve of implied vol across expirations. Contango (VIX3M > VIX) = normal. "
                 "Backwardation (VIX > VIX3M) = near-term stress elevated, hedging demand high.")
            +_gl("Tail Risk (VVIX/VIX)",
                 "VVIX = vol of VIX (vol-of-vol). High ratio = market pricing sharp VIX spikes = crash insurance expensive. "
                 "Ratio > 5.5 = elevated. < 4.5 = low.")),unsafe_allow_html=True)
    with g12:
        st.markdown(_card(
            _sh(12,"GLOSSARY — MACRO & PROBABILITY")
            +_gl("σ (Sigma / Standard Deviation)",
                 "1σ ≈ 68% of expected moves, 2σ ≈ 95%, 3σ ≈ 99.7%. Daily σ of 1% → ±1% range 68% of the time.")
            +_gl("Skewness & Kurtosis",
                 "Skewness = asymmetry. Negative skew = more large down moves. "
                 "High kurtosis = fat tails (extreme moves more common than normal distribution predicts).")
            +_gl("Recession P(6m)",
                 "Blends Sahm Rule, HY OAS, and Three Puts backstop. Sahm: 3M avg unemployment up 0.5% above 12M low = recession onset.")
            +_gl("Net Liquidity",
                 "Fed Balance Sheet minus TGA minus RRP. Expanding = risk asset tailwind. Contracting = headwind.")),unsafe_allow_html=True)
    with g13:
        st.markdown(_card(
            _sh(13,"GLOSSARY — READING THE CHARTS")
            +_gl("IV Surface","3D plot: implied vol across strikes (moneyness) and DTE. Steep put skew = downside protection expensive.")
            +_gl("Probability Heatmap","Monte Carlo using Merton jump-diffusion. Shows probability of SPX reaching each price level over next week.")
            +_gl("GEX Histogram","Gamma per strike. Green = positive (dealers stabilize). Red = negative (dealers amplify).")
            +_gl("IV vs RV Chart","VIX overlaid with 21D realized vol. VRP spread: green = IV premium (sell vol), red = IV discount (buy protection).")
            +_gl("Fear Composite","VIX z-score + NFCI + curve inversion + liquidity tightening. >60 = elevated fear.")),unsafe_allow_html=True)

    st.markdown(
        f"<div style='text-align:center;font-size:10px;color:rgba(255,255,255,0.2);margin-top:16px;'>"
        f"Data: FRED · yfinance · Schwab API | Generated {dt.datetime.now().strftime('%H:%M ET')} | Not financial advice</div>",
        unsafe_allow_html=True)
