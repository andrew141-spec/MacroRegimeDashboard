# page_guide.py — render_guide + render_probability_page
import os, math, datetime as dt
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from config import GammaState, GammaRegime, FeedItem, SetupScore, CSS
from utils import _to_1d, resample_ffill, rolling_pct
from ui_components import plotly_dark, sec_hdr
from data_loaders import load_macro
from signals import compute_leading_stack


# ============================================================
# PROBABILITY ENGINE PAGE (data-driven)
# ============================================================
def render_probability_page():
    """Probability engine deep-dive: signal overview bar chart + rolling history."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## 📊 Probability Engine")
    st.markdown("Per-signal percentile ranks and rolling history for all leading indicators.")

    start = st.sidebar.date_input("Start", value=dt.date.today() - dt.timedelta(days=730), key="pb_start")
    end   = st.sidebar.date_input("End",   value=dt.date.today(), key="pb_end")
    idx   = pd.date_range(start, end, freq="D")

    with st.spinner("Loading data..."):
        raw = load_macro(start.isoformat(), end.isoformat())
        def r(k): return resample_ffill(raw.get(k, pd.Series(dtype=float)), idx)
        y2=r("DGS2"); y3m=r("DGS3MO"); y10=r("DGS10"); y30=r("DGS30")
        m2=r("M2SL"); walcl=r("WALCL"); tga=r("WTREGEN"); rrp=r("RRPONTSYD")
        copx=r("COPX"); gld=r("GLD"); hyg=r("HYG"); lqd=r("LQD")
        dxy=r("UUP"); spy=r("SPY"); vix=r("VIX"); qqq=r("QQQ"); iwm=r("IWM")
        claims=r("ICSA")
        tips_10y=r("DFII10"); bank_reserves=r("WRBWFRBL")
        bank_credit=r("TOTBKCR")
        ism_no_raw = raw.get("AMTMNO", pd.Series(dtype=float))
        ism_no = ism_no_raw if len(ism_no_raw.dropna()) > 4 else None
        gdp_quarterly=r("GDPC1"); mmmf=r("WRMFSL")
        unrate=r("UNRATE")
        hy_spread_raw = raw.get("BAMLH0A0HYM2", pd.Series(dtype=float))

    net_liq    = (walcl - tga - rrp) / 1000.0
    net_liq_4w = net_liq.diff(28)
    bs_13w     = walcl.diff(91) / 1000.0
    s_2s10s    = (y10 - y2) * 100

    leading = compute_leading_stack(
        y2, y3m, y10, y30, s_2s10s, vix, m2, claims,
        copx, gld, hyg, lqd, dxy, spy, qqq, iwm,
        net_liq, net_liq_4w, walcl, bs_13w, idx,
        tips_10y=tips_10y, bank_reserves=bank_reserves,
        bank_credit=bank_credit, ism_no=ism_no,
        gdp_quarterly=gdp_quarterly, mmmf=mmmf,

    )

    SIGNAL_LABELS = {
        "vix_ts_pct":           "VIX Term Structure (VIX/VIX3M slope)",
        "corr_regime_pct":      "Correlation Stress Signal",
        "breadth_pct":          "Market Breadth (IWM/QQQ momentum)",
        "dxy_5d_pct":           "DXY 5D Momentum [5D]",
        "hyg_lqd_pct":          "HYG/LQD Ratio [21D]",
        "smallcap_pct":         "Small-Cap Leadership [21D]",
        "liq_impulse_4w_pct":   "Net Liquidity 4W [21D]",
        "ism_no_pct":           "ISM New Orders [21D]",
        "curve_phase_pct":      "Curve Phase [63D]",
        "copper_gold_pct":      "Copper/Gold 13W [63D]",
        "credit_impulse_pct":   "Credit Impulse [63D]",
        "real_rate_pct":        "Real Rate Regime [63D]",
        "reserve_pct":          "Reserve Adequacy [63D]",
        "m2_yoy_pct":           "M2 YoY Growth [63D]",
        "liq_impulse_13w_pct":  "Net Liquidity 13W [63D]",
    }

    pcts = [(label, leading.get(key)) for key, label in SIGNAL_LABELS.items()]
    # Only keep numeric values that can be plotted as percentile bars (0-100 range)
    pcts = [(label, float(val)) for label, val in pcts
            if val is not None and isinstance(val, (int, float))
            and not __import__("math").isnan(float(val))
            and 0.0 <= float(val) <= 100.0]

    if not pcts:
        st.warning("No signal data available yet. Check data sources.")
        return

    def _bar_colour(v):
        try: v = float(v)
        except (TypeError, ValueError): return "#6b7280"
        if v > 65:   return "#10b981"
        elif v > 55: return "#34d399"
        elif v > 45: return "#f59e0b"
        elif v > 35: return "#f97316"
        else:        return "#ef4444"

    colours = [_bar_colour(v) for _, v in pcts]
    fig_p = go.Figure(go.Bar(
        x=[v for _, v in pcts], y=[l for l, _ in pcts],
        orientation="h", marker_color=colours,
        text=[f"{v:.0f}th" for _, v in pcts], textposition="outside",
    ))
    fig_p.add_vline(x=50, line_dash="dot", line_color="rgba(255,255,255,0.25)")
    fig_p.add_vline(x=80, line_dash="dot", line_color="rgba(16,185,129,0.30)")
    fig_p.add_vline(x=20, line_dash="dot", line_color="rgba(239,68,68,0.30)")
    fig_p.update_layout(xaxis=dict(range=[0, 115]))
    st.plotly_chart(
        plotly_dark(fig_p, "Leading Signals — Historical Percentile (50=Neutral · 80=Top Quintile)", 520),
        use_container_width=True
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Real Rate Regime", leading.get("real_rate_regime","N/A"),
                 f"{leading.get('real_rate_level',float('nan')):.2f}%" if not (leading.get('real_rate_level',float('nan')) != leading.get('real_rate_level',float('nan'))) else "N/A")
    col_b.metric("Reserve Adequacy", leading.get("reserve_regime","N/A"),
                 f"${leading.get('reserve_level_bn',float('nan')):,.0f}B" if not (leading.get('reserve_level_bn',float('nan')) != leading.get('reserve_level_bn',float('nan'))) else "N/A")
    col_c.metric("ISM New Orders", leading.get("ism_quadrant","Unknown"),
                 f"{leading.get('ism_level',float('nan')):.1f}" if not (leading.get('ism_level',float('nan')) != leading.get('ism_level',float('nan'))) else "N/A")

    st.markdown("---")
    st.markdown("### Rolling Percentile History")
    spy_r = spy.reindex(idx).ffill()
    series_map = {
        "VIX Term Structure [5D]":    vix.reindex(idx).ffill() / (spy_r.pct_change().rolling(63,min_periods=20).std()*np.sqrt(252)*100).replace(0,np.nan),
        "DXY 5D Momentum [5D]":       -dxy.reindex(idx).ffill().pct_change(5)*100,
        "HYG/LQD Ratio [21D]":        hyg.reindex(idx).ffill()/lqd.reindex(idx).ffill().replace(0,np.nan),
        "Small-Cap Leadership [21D]":  iwm.reindex(idx).ffill()/spy_r.replace(0,np.nan),
        "Net Liquidity 4W [21D]":      net_liq_4w,
        "ISM New Orders [21D]":        resample_ffill(raw.get("AMTMNO", pd.Series(dtype=float)), idx),
        "Curve Phase [63D]":           s_2s10s,
        "Copper/Gold 13W [63D]":       copx.reindex(idx).ffill()/gld.reindex(idx).ffill().replace(0,np.nan),
        "Credit Impulse [63D]":        m2.diff(91)/m2.shift(91)*100,
        "Real Rate Regime [63D]":      r("DFII10"),
        "Reserve Adequacy [63D]":      r("WRBWFRBL"),
        "M2 YoY Growth [63D]":         (m2/m2.shift(365)-1)*100,
        "Net Liquidity 13W [63D]":     net_liq.diff(91),
    }

    available = [l for l, _ in pcts if l in series_map]
    if available:
        choice = st.selectbox("Signal", available)
        s_plot = series_map.get(choice, s_2s10s)
        pct_s  = rolling_pct(s_plot.dropna(), 252)
        fig1 = go.Figure(go.Scatter(x=s_plot.index, y=s_plot.values, mode="lines",
                                     line=dict(color="#3b82f6", width=1.5)))
        st.plotly_chart(plotly_dark(fig1, choice, 240), use_container_width=True)
        fig2 = go.Figure(go.Scatter(x=pct_s.index, y=pct_s.values, mode="lines",
                                     line=dict(color="#8b5cf6"), fill="tozeroy",
                                     fillcolor="rgba(139,92,246,0.08)"))
        fig2.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.2)")
        fig2.add_hline(y=80, line_dash="dot", line_color="rgba(16,185,129,0.3)")
        fig2.add_hline(y=20, line_dash="dot", line_color="rgba(239,68,68,0.3)")
        st.plotly_chart(plotly_dark(fig2, f"{choice} — Rolling 252D Percentile Rank", 200),
                        use_container_width=True)
        st.caption("Percentile rank: trailing 252-day window, current value only.")


# ============================================================
# GUIDE PAGE
# ============================================================
def render_guide():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## 📖 Dashboard Guide")
    st.markdown("*Every concept explained in depth — what it is, why it exists, how it is computed, and how it feeds into the model.*")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🏠 Overview",
        "⚡ GEX Engine",
        "🔬 Greeks Deep Dive",
        "📊 Probability Engine",
        "🌍 Intel Monitor",
        "📋 Execution",
        "⚠️ Risk Signals",
        "🔧 Setup & Data",
    ])

    # ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### What is this dashboard?")
        st.markdown("""
This is a **quantitative trading dashboard** that combines three distinct analytical pillars into one screen:

1. **WHERE** — Mechanical support and resistance from options dealer positioning (GEX)
2. **WHY** — Macro backdrop and directional probability from 13 economic signals
3. **WHEN** — Session context, setup scoring, and sizing rules that tell you whether conditions are good enough to act

The goal is not to predict the market. It is to trade **with the odds slightly in your favour**, know when conditions support a trade, and know when to sit on your hands.

---

### The Three Pillars

#### Pillar 1 — GEX: Where the mechanical flows are

When you buy an options contract, a dealer (market maker) sold it to you. That dealer is not speculating — they need to remain delta-neutral. So they buy or sell the underlying to hedge their exposure. This hedging creates **predictable, mechanical price flows** that are independent of anyone's opinion about fundamentals.

**Positive GEX (above the gamma flip)**
- Dealers are net long gamma: a price drop forces them to buy the underlying, a price rise forces them to sell it.
- Their hedging *opposes* moves — it is a natural shock absorber.
- Result: volatility is suppressed, large moves fade, options pin to strikes.

**Negative GEX (below the gamma flip)**
- Dealers are net short gamma: a price drop forces them to sell more, a price rise forces them to buy more.
- Their hedging *amplifies* moves — there is no natural floor from dealer flows.
- Result: cascades become possible, moves are self-reinforcing, trends extend further than fundamentals would suggest.

The **gamma flip** is the price level where this switches. It is the most important structural level on the chart.

#### Pillar 2 — Macro Probability Engine: Why the market should move

Thirteen signals across three forward-looking time horizons (5-day, 21-day, 63-day) are measured as percentile ranks against their own history and blended into directional probability estimates. Each signal measures a genuinely different underlying economic factor — credit conditions, liquidity, growth, inflation, yield curve dynamics, and dollar strength.

#### Pillar 3 — Execution Framework: When and how large

Even with correct direction and correct structure, a trade still fails if the timing is wrong. The execution framework enforces session-based size multipliers, a pre-trade checklist, setup classification, and explicit failure mode monitoring. It prevents overtrading in thin conditions and sizes positions to the honest edge.

---

### How to Read the Main Dashboard

**Session bar (top)**
Shows the current session window (Globex / RTH Open / IB Forming / Morning / Midday / Afternoon / Close). The size multiplier is the key number — 0.0 means do not trade, 1.0 means full position size.

**Probability row**
Four cards: 1-Day, 5-Day, 21-Day, 63-Day. Each shows the bull probability (0-100%) and the associated Kelly fraction. Values above 60% are bullish-tilted, below 40% are bearish-tilted, 40-60% are noise.

**GEX snapshot**
The gamma flip level, the current GEX regime, and whether vanna/charm are aligned. This tells you the mechanical context for any move that occurs.

**GEX chart (GEX Engine tab)**
Bars by strike. Green = dealers long gamma here (stabilising / support / resistance). Red = dealers short gamma here (amplifying / fuel). The tallest bars are the most important levels. The yellow line is the flip.

**Intel panel (right side)**
Headlines scored and categorised by market relevance. Driver alerts fire when any signal changes materially between refreshes.
""")

    # ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### GEX Engine — Full Explanation")
        st.markdown("""
#### What GEX measures

GEX (Gamma Exposure) answers one question: **how much dollar hedging must dealers do for each 1% move in the underlying?**

The formula per strike:
```
GEX = Gamma × Open Interest × Contract Size × Spot Price²

Where:
  Gamma        = rate of change of delta (from Black-Scholes)
  Open Interest = number of open contracts at this strike
  Contract Size = 100 shares (standard US options)
  Spot Price²   = converts to dollar-gamma units ($)
```

**Sign convention (critical):**
- Calls: dealers assume they sold calls → they are **short calls** → they are **long gamma** → their hedging opposes moves (stabilising) → sign = **positive**
- Puts: dealers assume they sold puts → they are **short puts** → they are **short gamma** → their hedging amplifies moves (destabilising) → sign = **negative**
- Net GEX at each strike = call GEX + put GEX

A strike with net positive GEX means the call open interest is large enough to make dealers net buyers on dips and sellers on rallies at that level. A strike with net negative GEX does the opposite.

#### The Gamma Flip (Vol Trigger)

The flip is the price where the **sum of all net GEX across all strikes** transitions from negative to positive. It is found by aggregating GEX by strike and interpolating the zero-crossing.

When all strikes are the same sign (heavy put skew is common in SPY), there is no zero-crossing in the literal sense. In this case the flip falls back to the **minimum-GEX strike** — the point of least dealer support, often called the vol trigger. This is still a meaningful level: it is where dealer gamma is weakest.

**Distance from flip** is shown as a percentage. Within ±0.5% of the flip is the transition zone — regime is ambiguous and all GEX-based signals are compressed toward 50%.

#### How Gamma Is Calculated: Black-Scholes vs Schwab Model

**Without Schwab connected (yfinance data):**
Gamma is computed analytically using the Black-Scholes formula:
```
d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
Gamma = N'(d1) / (S·σ·√T)

Where:
  S = spot price
  K = strike price
  T = time to expiration (years)
  σ = implied volatility (from yfinance)
  r = risk-free rate (5% assumed)
  N'= standard normal PDF
```
This uses a flat IV approximation — one IV per contract, not a full vol surface. Adequate for the flip level and regime classification, but less accurate for the per-strike bar heights.

**With Schwab connected:**
Schwab's own market model computes gamma per contract from their full vol surface. This is the `schwab_gamma` field returned per contract. Where it is non-zero and finite, it replaces the BS-computed gamma. Where it is missing or zero, BS fills in the gap. Schwab's gamma accounts for skew, term structure, and microstructure effects that flat-vol BS misses — the flip level is more accurate intraday.

#### The GEX Bar Chart

Each bar represents a strike. Its height is the **sum of net GEX across all expiration dates** at that strike. The chart tells you:

- **Tallest green bar**: the strongest stabilising level. Dealers will mechanically buy (from above) or sell (from below) most aggressively here.
- **Tallest red bar**: the point of maximum destabilisation. A break through here is amplified by dealer flows.
- **OTM anchors**: green bars more than 3% from spot — these define the structural range the market is expected to stay within under current dealer positioning.

#### The 5 GEX-Based Setups

| Setup | Regime required | Trigger | Mechanics |
|-------|----------------|---------|-----------|
| **1 — Gamma Support Bounce** | Positive (above flip) | Price touches a large positive-GEX strike from above | Dealers must buy as price falls to this level. Support is mechanical, not opinion. Wait for footprint absorption before entering long. |
| **2 — Gamma Resistance Fade** | Positive (above flip) | Price reaches a large positive-GEX strike from below | Dealers must sell as price rises to this level. Resistance is mechanical. Slightly lower hit rate than Setup 1 because equities have a structural upward bias. |
| **3 — Gamma Flip Breakout** | Neutral (within 0.75% of flip) | Price breaks through the flip and holds | Regime transition. Positive→Negative means dealer flows switch from opposing moves to amplifying them. The breakout can cascade. Needs initiative flow confirmation — no absorption = no trade. |
| **4 — Exhaustion Reversal** | Negative (below flip), extended >2% | Volume climax + delta divergence on footprint | Rarest setup (lowest hit rate). In negative gamma, dealers amplify moves. This only reverses when exhaustion is genuine — large volume, price no longer moving down, delta flipping. 3:1 R:R minimum, 50% size only. |
| **5 — 0DTE Gamma Pin** | Any (dominant strike present) | After 2pm ET, or OpEx morning 10:30-12pm | Large 0DTE (same-day expiry) open interest creates a gravitational pin. The strike acts like an attractor — price deviations fade back to it. Most potent in the final 90 minutes before expiry. |

#### VEX, CEX and Vanna/Charm Alignment

The engine also computes two secondary Greek exposures:

**VEX (Vanna Exposure)**
Vanna = how dealer delta changes when implied volatility changes.
```
Vanna = -N'(d1) × d2 / σ
```
- Positive VEX + IV rising → dealers buy → upward pressure
- Positive VEX + IV falling → dealers sell → downward pressure
- Negative VEX + IV rising → dealers sell → downward pressure (amplified by vol spike)

This is why VIX spike days can cause cascading equity selling even without obvious fundamental catalysts — vanna flows force dealer sales as IV rises. Knowing the vanna sign tells you whether a volatility expansion is bullish or bearish for the underlying.

**CEX (Charm Exposure)**
Charm = how dealer delta changes as time passes (delta decay).
```
Charm = -N'(d1) × [2rT - d2·σ·√T] / [2T·σ·√T]
```
- Positive CEX near spot → as expiry approaches, dealers buy → upward drift
- This is the mechanical basis for the OpEx pin trade and the post-market-open morning drift in positive gamma regimes

**Vanna/Charm Alignment**
When both vanna and charm are pointing the same direction near spot, the dealer hedging flows from two independent sources (vol changes and time decay) are reinforcing each other. This is a higher-conviction mechanical context. The dashboard shows this alignment status in the GEX panel.
""")

    # ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Options Greeks: A Complete Reference")
        st.markdown("""
This tab explains every Greek used in the engine from first principles. You do not need to memorise these. Understanding them conceptually tells you *why* the model behaves the way it does.

---

#### Delta (Δ)

**What it is:** The rate of change of an option's price per $1 move in the underlying.
- Call delta: 0 to 1. Deep in-the-money call ≈ 1.0 (moves dollar-for-dollar with stock). At-the-money call ≈ 0.5. Far OTM call ≈ 0.
- Put delta: -1 to 0.

**Why dealers care:** A dealer who sold you a call with delta 0.5 must own 50 shares of the underlying to hedge their exposure. If the stock rallies and the call becomes delta 0.7, they must buy 20 more shares. This buying is **automatic, mechanical, and forced** — it has nothing to do with their view on the stock.

**In GEX context:** Delta hedging is the act of maintaining a delta-neutral position. Every GEX trade setup is ultimately about predicting where dealer delta-hedging flows will go.

---

#### Gamma (Γ)

**What it is:** The rate of change of delta per $1 move in the underlying. Also: how fast the dealer must re-hedge.
```
Gamma = N'(d1) / (S × σ × √T)
```
- Gamma is highest for at-the-money options near expiration.
- Gamma is low for deep in/out-of-the-money options and for options with long time to expiry.

**Why it matters for GEX:** Gamma is the multiplier in the GEX formula. A strike with large open interest but low gamma contributes less to GEX than a strike with smaller OI but high gamma (typically near-term ATM strikes). This is why 0DTE options dominate the GEX chart even though they represent a small fraction of total notional OI.

**Long vs short gamma:**
- Long gamma: you profit from large moves in either direction. Dealers who sold you options are usually short gamma.
- Short gamma: you profit from small moves (decay). Market makers earn the bid-ask spread but are exposed to large moves.

The GEX sign convention: dealers are assumed to have sold options to the public → dealers are net short gamma overall → positive GEX requires sufficient call OI to overcome the natural short-gamma position.

---

#### Vanna (dΔ/dσ)

**What it is:** How delta changes when implied volatility changes. Equivalently, how a dealer's hedge ratio must change when IV moves.
```
Vanna = -N'(d1) × d2 / σ
```

**Intuition:** An out-of-the-money option has low delta (say 0.15). If IV spikes, that OTM option now has a higher probability of expiring in-the-money, so its delta rises (say to 0.30). The dealer who sold that option must now buy more shares to re-hedge. If this is happening simultaneously across thousands of strikes, the forced buying (or selling) from vanna creates directional flows in the underlying.

**Why the sign matters:**
- Near-spot options with positive vanna: IV rising → dealer buys underlying (vanna supports rallies during vol spikes)
- Near-spot options with negative vanna: IV rising → dealer sells underlying (vanna amplifies equity selloffs)

Most equity markets in normal conditions have negative aggregate vanna near-spot — which is why VIX spikes correlate with equity declines beyond what fundamentals explain.

---

#### Charm (dΔ/dt)

**What it is:** How delta changes as time passes. Also called delta decay.
```
Charm = -N'(d1) × [2rT - d2·σ·√T] / [2T·σ·√T]
```

**Intuition:** An option that is slightly in-the-money with 30 days to expiry has a certain delta. As time passes toward expiry, that option becomes more binary — it will either expire in-the-money (delta→1) or out-of-the-money (delta→0). As delta changes through time, dealers must adjust their hedge continuously.

**Why it matters in practice:**
- Near expiration (0DTE, WeeklyDTE), charm flows dominate intraday dealer hedging
- Positive aggregate charm near spot → as the trading day progresses, dealers buy → creates an intraday upward drift
- This is the mechanical basis for the well-documented "late morning drift" on OpEx Fridays

**Vanna/Charm Alignment:** When both are pointing the same direction near spot, you have two independent dealer hedging flows (one from vol, one from time) reinforcing each other. This is the strongest version of a GEX-regime signal.

---

#### Implied Volatility (IV)

**What it is:** The market's consensus forecast of future volatility, *implied* from the current options price. Not a prediction of direction — just magnitude.

**How it's used in GEX:** IV is the σ input to all Greek calculations. Higher IV → lower gamma (for a given time to expiry). This means that in high-IV environments, the absolute GEX values are smaller — dealers need to re-hedge less per dollar move because their hedges are already less sensitive to price.

**VIX vs per-strike IV:** VIX is the 30-day implied volatility of the S&P 500 index (from near-term options). Per-strike IV is specific to each contract. With yfinance, the engine uses per-contract IV. With Schwab, it uses the exact IV from Schwab's model per contract (more accurate, accounts for vol smile/skew).

---

#### The Black-Scholes Model

All four Greeks above (delta, gamma, vanna, charm) are computed analytically from the Black-Scholes option pricing formula:
```
Call Price = S·N(d1) - K·e^(-rT)·N(d2)

d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
d2 = d1 - σ·√T

Where:
  S  = current spot price
  K  = option strike price
  T  = time to expiry (in years; 1 day = 1/365)
  σ  = implied volatility (annualised)
  r  = risk-free rate (5% assumed)
  N  = cumulative standard normal distribution
  N' = standard normal probability density function
```

Black-Scholes assumes constant volatility across all strikes — a simplification. Real markets have a volatility smile (OTM puts have higher IV than ATM). Schwab's model corrects for this; the BS approximation used with yfinance data does not. For the purposes of identifying the flip level and key GEX nodes, the approximation is adequate.
""")

    # ─────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Probability Engine — Full Explanation")
        st.markdown("""
#### The Core Architecture

The probability engine answers: *what are the odds of the market being higher in X days?*

It does this with a **three-horizon bucket model**, keeping short-term and medium-term signals completely separate. This design choice is deliberate:

- If you mix a 5-day signal (VIX term structure) with a 63-day signal (yield curve phase), the faster signal dominates in the short run but the slower signal is more predictive over time. Mixing them into a single score gives you a number that isn't optimal for either horizon.
- Each bucket answers a different question. The 5-day bucket is for tactical positioning. The 63-day bucket is for swing positions.

The four buckets and their weights in the composite:
```
Tactical    (5D)  : 10% weight — high-frequency, noisy
Short-term  (21D) : 30% weight
Medium-term (63D) : 40% weight — most predictive empirically
Coincident  (now) : 20% weight — reality check, not leading
```

---

#### How Each Signal Is Computed

Every signal is converted to a **percentile rank**: where does today's reading sit relative to the past 252 trading days (approximately 1 year)?
- 80th percentile = more bullish than 80% of the past year → bullish signal
- 50th percentile = exactly at the median → neutral
- 20th percentile = more bearish than 80% of the past year → bearish signal

This normalisation has three advantages: (1) it is dimensionless — all signals are on the same 0-100 scale, (2) it is self-adjusting to the current regime — a reading of 300bp on the HY spread means something different in 2009 vs 2024, and (3) it makes the composite robust to outliers.

---

#### Tactical Bucket (5-Day Horizon)

**VIX Term Structure**
- Raw: VIX level ÷ 63-day realised volatility of SPY (annualised, %)
- What it measures: the vol premium the market is paying for near-term insurance vs what has actually happened
- High ratio (>1.3): fear is overpriced relative to recent history → contrarian bullish → high percentile rank → high score
- Low ratio (<0.8): complacency → vol may mean-revert upward → bearish tail risk
- In the 1-day model, this uses 5-day realised vol (more sensitive to recent events)

**DXY 5-Day Momentum**
- Raw: 5-day percentage change in UUP (dollar index ETF), sign-inverted
- What it measures: the 5-day dollar trend, where a rising dollar is bearish for risk assets
- Inverted because dollar strength = capital rotating out of risk assets
- A falling dollar (negative DXY change) → high score (bullish)

---

#### Short-Term Bucket (21-Day Horizon)

**HYG/LQD Ratio**
- Raw: HYG price ÷ LQD price (ratio of high-yield to investment-grade bond ETFs)
- What it measures: whether credit markets are embracing or avoiding risk
- HYG rising faster than LQD = credit spreads tightening = risk-on → high score
- HYG falling relative to LQD = credit stress building → low score
- Critical: this leads equity stress by **days to weeks**. Credit markets see trouble before equity markets because institutional credit investors are generally better-informed than equity retail flow.

**Small-Cap Leadership (IWM/SPY)**
- Raw: IWM price ÷ SPY price
- What it measures: breadth of the economic expansion
- Small caps are more economically sensitive than mega-caps. They rely on domestic consumption, bank loans, and are exposed to higher interest rates.
- IWM leading SPY = broad economic confidence. IWM lagging = markets hiding in mega-cap safety.
- In a healthy bull market, small caps lead. In a defensive/late-cycle environment, they lag.

**Net Liquidity 4-Week Impulse**
- Raw: Change over 28 days of (Fed Balance Sheet − Treasury Cash Account − Reverse Repo Facility)
  ```
  Net Liquidity = WALCL - WTREGEN - RRPONTSYD  (all in $bn)
  Net Liq 4W    = Net_Liq(today) - Net_Liq(28 days ago)
  ```
- What it measures: the 4-week change in how much "free" money is in the financial system
- The Fed balance sheet is the source of base money. The TGA is a drain (Treasury holds cash at the Fed). RRP is a drain (dealers park cash at the Fed overnight instead of deploying it in markets). Net liquidity is what's left flowing through the financial system.
- Expanding = supportive. Contracting = headwind. This is the single most mechanically direct link between Fed policy and market flows.

**ISM New Orders**
- Raw: AMTMNO series (Institute for Supply Management Manufacturing New Orders index)
- What it measures: forward-looking manufacturing demand. Businesses place new orders before they produce and hire. This leads GDP by 3-6 months.
- Above 50 = expansion. Below 50 = contraction. But the **direction** matters more than the level:
  - Above 50 + rising: best (expanding and accelerating)
  - Above 50 + falling: watch (expanding but slowing)
  - Below 50 + rising: recovery signal (contracting but turning)
  - Below 50 + falling: worst (contracting and accelerating)
- Monthly data — carries forward between releases.

---

#### Medium-Term Bucket (63-Day Horizon)

**Yield Curve Phase**
- Raw: 2-year Treasury yield minus 10-year Treasury yield (2s10s spread, in basis points)
- What it measures: not just the level of the curve, but its *rate of change* and *direction*. Four phases:

| Phase | Description | Signal |
|-------|-------------|--------|
| Bull steepening | Both rates falling, long end falls faster | Most bullish — growth expectations recovering, recession fears fading |
| Bear steepening | Short end falls faster than long end | Mixed — Fed cutting into weakness, long rates rising (inflation or fiscal concern) |
| Bull flattening | Long end rallying (falling yields), short end stable | Risk-off — flight to safety, growth slowing |
| Bear flattening | Short end rising faster | Late cycle — Fed tightening, policy error risk |

An inverted curve (2Y > 10Y) signals that short-term funding costs more than long-term capital — banks stop lending profitably, credit creation slows. Every US recession since WWII was preceded by curve inversion.

**Copper/Gold 13-Week Ratio**
- Raw: COPX (copper miner ETF) price ÷ GLD (gold ETF) price over 13 weeks
- What it measures: the growth/fear balance in global commodity markets
- Copper is an industrial input — demand rises with economic activity. Gold is a fear asset — demand rises with uncertainty.
- Copper/Gold rising = global growth confidence. Gold/Copper rising = global fear or growth doubt.
- Historically leads global PMI by 2-3 months because copper buyers must plan industrial production in advance.

**Credit Impulse**
- Raw: (Change in bank credit over 91 days) ÷ (Bank credit 91 days ago) × 100
  ```
  Credit Impulse = (TOTBKCR - TOTBKCR.shift(91)) / TOTBKCR.shift(91) × 100
  ```
- What it measures: not the level of credit, not the growth rate, but the **acceleration** of credit creation
- Academic basis: Biggs, Mayer, Pick (2010) — the Credit Impulse leads GDP and equity returns by ~6 months
- Why acceleration matters: when banks suddenly lend more aggressively after a slow period, that new money enters the economy immediately and shows up in spending data 1-2 quarters later. The inflection is the signal, not the level.

**Real Rate Regime**
- Raw: DFII10 (10-year TIPS yield, daily from FRED)
- What it measures: the inflation-adjusted (real) return on 10-year Treasuries
- Real rates above 2.5%: punitive discount rate for equities. Cash earns more than enough to compensate for risk. Equity valuations compress. Growth stocks hit hardest.
- Real rates 0-2%: neutral zone.
- Real rates below 0% (financial repression): cash earns nothing real. Capital pushed into risk assets. Historically associated with strong equity and commodity performance.

**Reserve Adequacy**
- Raw: WRBWFRBL (excess reserves of depository institutions at the Federal Reserve, $bn)
- What it measures: how much cushion is in the banking system's plumbing
- Below ~$3 trillion: historically associated with funding stress. The September 2019 repo spike (overnight rates briefly hit 10%) occurred when reserves approached this level.
- When reserves are low, banks hoard cash and reduce inter-bank lending → repo rates spike → funding costs rise across the system → credit tightens.

**M2 YoY Growth**
- Raw: (M2SL / M2SL.shift(365) - 1) × 100
- What it measures: how fast the total money supply is growing year-over-year
- More money in the system → more capital available to chase assets → supportive for equity prices
- M2 growth leads equity returns by approximately 6-9 months empirically
- Negative M2 YoY (quantitative tightening combined with weak bank lending) has preceded every major equity drawdown — 2000, 2008, and the 2022 bear market

**Net Liquidity 13-Week Impulse**
- Same construction as the 4-week version, but over 91 days
- Captures slower-moving structural liquidity trends rather than week-to-week fluctuations

---

#### Coincident Conditions (Not Leading — A Reality Check)

Three signals measure what is happening **right now**, not what is expected. These are not leading indicators. They are used as a 20%-weight reality check to prevent the model from being wildly wrong when current conditions are clearly stressed:

1. **Fear score (VIX-based equity stress)**: a composite of VIX level, VIX term structure, and short-term price action. High fear compresses bullish signals.
2. **NFCI (National Financial Conditions Index)**: the Federal Reserve's weekly measure of credit, risk, and leverage conditions in US financial markets. Above 0 = tighter than average. Below 0 = looser than average.
3. **Net liquidity direction**: is the 4-week liquidity impulse currently positive (expanding) or negative (contracting)?

These were deliberately chosen to be orthogonal — they measure genuinely different underlying things (equity vol surface, credit/funding conditions, central bank plumbing). Earlier versions of this bucket had 5+ signals that all contained yield-curve information, which effectively made the bucket triple-count one factor.

---

#### The 1-Day Model

The 1-day model is architecturally separate from the 5/21/63-day models. It is built specifically around GEX mechanics.

**Why GEX conditions the 1-day model:**

GEX does not give a direction. But it tells you *how to interpret* the other signals:

| GEX Regime | How to read momentum | Primary signal |
|------------|---------------------|----------------|
| Positive gamma | FADE momentum | Credit/dollar microstructure (what is flowing today) |
| Negative gamma | FOLLOW momentum | SPY 5-day momentum (dealers amplify) |
| Near flip | Compress toward 50% | All signals equally uncertain |

The five factors:
1. **VIX/RVol ratio** — Using 5-day realised vol (not 63-day). Very short horizon fear premium.
2. **SPY 5D momentum** — Regime-conditioned: fade in positive gamma, follow in negative gamma.
3. **Credit/dollar microstructure** — 1-day changes in HYG, LQD, and DXY. These are the fastest cross-asset signals and lead equity by minutes to hours intraday.
4. **Curve inversion status** — Binary structural headwind/tailwind. An inverted curve is a persistent drag regardless of regime.
5. **Net liquidity sign** — Sign and recent acceleration of the 4-week liquidity impulse.

**Session compression:** Outside prime-time (10:30am–12pm ET), the 1-day signal is compressed toward 50%. GEX-based setups only work reliably in liquid conditions.

**Honest accuracy ceiling:** Realistic 1-day AUC is 0.52–0.55. This is consistent with the academic literature. Anyone claiming >0.58 on daily equity direction is overfitting. The value is in *conditional positioning* — knowing when conditions are even slightly better, and sizing accordingly.

---

#### GEX Volatility Regime Adjustment

GEX does not vote on direction in the composite model. It adjusts for expected volatility regime:

- **Positive gamma regime**: compress the composite slightly toward 50. Moves are expected to be mean-reverting, so extreme readings are less likely to persist.
- **Negative gamma regime**: expand the composite slightly away from 50. Moves are expected to be amplified, so directional signals carry more weight.

This adjustment is capped at 8% of the distance to 50, scaled by regime stability. It is a second-order effect — the signal itself comes from the economic indicators, not from GEX.

---

#### Geo Shock Overlay

The geo shock score (0-100) is computed separately from the Intel Monitor and applied as a drag on the bull probability:

```
geo_drag = (geo_shock / 100) × 15    # max 15 percentage-point compression
```

A geo shock of 50 compresses the bull probability by 7.5 percentage points. This reflects the fact that geopolitical risk events add uncertainty orthogonal to all economic signals — they can occur regardless of liquidity, credit, or curve conditions.

---

#### Regime Uncertainty Compression

The regime transition probability (P(regime change in 20 days)) compresses the composite:
```
compressed = 50 + (raw - 50) × (1 - uncertainty × 0.45) - geo_drag
```
At 50% transition probability, the composite is compressed 22.5% toward 50 (plus geo drag). At 0% transition probability, no compression. This prevents high-conviction directional calls during periods of macro regime instability.
""")

    # ─────────────────────────────────────────────────────────
    with tab5:
        st.markdown("### Intel Monitor — Full Explanation")
        st.markdown("""
#### What it does

The Intel Monitor scans **real-time news RSS feeds** from primary sources (Federal Reserve, BLS, BEA, Reuters, BBC, Al Jazeera, CNBC, FT) across 7 categories. Every 5 minutes, it re-fetches all feeds, scores each headline, and categorises it.

---

#### The 7 Categories

**Fed & Monetary Policy**
Tracks Federal Reserve statements, FOMC decisions, balance sheet operations, and regulatory policy. Sourced from the Fed's own press release feed, FOMC release feed, SEC, and White House.

Why it matters: the Fed is the single largest driver of long-term equity valuations through its control of the risk-free rate and liquidity supply. A surprise hawkish statement overrides virtually all other signals.

Key weighted terms: `rate cut +10`, `rate hike +10`, `balance sheet +8`, `QT +8`, `QE +8`, `SLR +9` (supplementary leverage ratio — affects how much capital banks must hold against Treasuries, which directly affects repo market liquidity)

**Fiscal & Debt**
Tracks Treasury operations, debt ceiling negotiations, budget legislation, and TGA (Treasury General Account) movements. Sourced from Treasury press releases, BEA, State Dept.

Why it matters: the TGA drawdown mechanism is one of the primary sources of near-term liquidity injections into markets. When Treasury spends down its Fed cash account, those dollars flow into the banking system and show up in net liquidity within days. Debt ceiling crises constrain this mechanism and create bond supply disruptions.

**Inflation & Labor**
Tracks CPI, PCE, NFP, ISM, jobless claims. Sourced from BLS, BEA, CNBC Economy, FT.

Why it matters: the Fed's reaction function depends entirely on the inflation/labor picture. High inflation with tight labor = Fed stays tight = higher real rates = equity headwind. Inflation breaking lower with rising unemployment = Fed can cut = rate tailwind.

**Trade & Tariffs**
Tracks tariff announcements, trade negotiations, sanctions, and export controls. Sourced from Reuters World, CNBC Trade.

Why it matters: tariffs create supply-side inflation (harder for the Fed to cut), disrupt global supply chains (negative for global growth expectations), and reduce corporate earnings on import-exposed companies. Tariff news in this dashboard feeds into the geo shock component.

**Geopolitical Risk**
Tracks conflicts, military activity, sanctions, strategic chokepoints. Sourced from Reuters, BBC, Al Jazeera, RFE/RL, Sky News.

Why it matters: the geo shock score from this category directly compresses the bull probability. See geo shock scoring below.

**Markets & Liquidity**
Tracks credit spreads, volatility events, repo market, M2 announcements. Sourced from Reuters Business, MarketWatch, Investopedia.

Why it matters: credit market stress shows up in headlines before it shows up in economic data. An unexpected credit event (like a major default or repo spike) is a coincident stress signal that should immediately reduce risk.

**AI & Tech Cycle**
Tracks AI capex announcements, semiconductor policy, antitrust cases. Sourced from Reuters Tech.

Why it matters: the current market is heavily concentrated in Mag-7 (Apple, Microsoft, Nvidia, Alphabet, Meta, Amazon, Tesla). A bubble score for this concentration feeds into the structural risk framework. AI-driven capex cycles affect both growth expectations and the sustainability of current valuations.

---

#### How Headlines Are Scored

Each headline is scored using a **weighted keyword match** across all 7 categories. The headline and its source name are searched for each category's keyword list:

```python
score = sum(weights.get(kw, 3.0) for kw in category_keywords if kw in headline_text)
```

The headline is assigned to its **highest-scoring category**. If two categories score equally, the first in order wins. If no category scores above 0, the headline is discarded.

Weights range from 3 (generic keyword) to 10 (critical term). Examples:
- `rate cut = 10`, `rate hike = 10` — the highest priority signals
- `debt ceiling = 10`, `shutdown = 9`, `SLR = 9` — near-critical
- `inflation = 7`, `cpi = 10`, `pce = 10` — primary economic data
- `tariff = 6`, `sanction = 7`, `blockade = 8` — escalating geopolitical terms

---

#### Geo Shock Scoring

The geo shock score is calculated separately and uses a more sophisticated classifier than simple keyword matching:

**Word-boundary matching:** The classifier uses regex word boundaries (`\\b` anchors) for short terms. This prevents `war` from matching `award`, `forward`, or `warrants`. Longer phrases are still substring-matched.

**Exclusion phrases:** Checked first, before any keyword matching. Examples:
- `"price war"` → not a conflict
- `"nuclear energy"`, `"nuclear plant"` → not a weapons event
- `"war on inflation"` → policy rhetoric, not conflict

**Country baseline risk:** High-risk countries get a bonus multiplier on any matched headline:
- Iran, Russia, North Korea, Yemen: high multiplier
- Taiwan, China, Ukraine, Israel, Pakistan: medium-high
- Others: standard

**Strategic chokepoints:** 9 critical maritime/geographic locations (Strait of Hormuz, Malacca Strait, Taiwan Strait, Suez Canal, Bab-el-Mandeb, Black Sea, Panama Canal, South China Sea, Persian Gulf). A chokepoint only scores if a **disruption keyword** also appears in the same headline. "Suez Canal expansion" → 0. "Suez Canal blockade" → 12.

**Temporal decay:** Headlines older than 6 hours decay in weight. Breaking news scores higher than a story that has been in the feed for a day.

The final geo shock score is clipped to 0-100 and applied as a compression on the bull probability.

---

#### Driver Alerts

Driver alerts fire whenever any tracked signal changes by more than its threshold between consecutive dashboard refreshes:

| Signal | Threshold | What triggers it |
|--------|-----------|-----------------|
| Fear Score | ±8 points | VIX-based stress composite changed materially |
| Bull Probability | ±8 points | Overall composite shifted significantly |
| Three Puts Score | ±8 points | Fed/Treasury/Political put availability changed |
| Liquidity Anxiety | ±10 points | Liquidity stress indicator moved |
| Market Index | ±12 points | Broader market composite shifted |
| Macro Regime | any change | Goldilocks/Overheating/Stagflation/Deflation classification flipped |
| Risk Regime | any change | Overall risk classification changed |
| GEX Regime | any change | Positive/Neutral/Negative gamma regime flipped |

These are **state-change notifications** — they tell you something material moved, not just noise. When a driver alert fires, the specific value is shown alongside the direction of change.

---

#### Economic Calendar Integration

The session context module maintains a hard-coded calendar of major scheduled events:
- FOMC decision days (all meetings through 2026)
- CPI release dates
- NFP/jobs report dates
- PCE release dates
- GDP advance estimate dates

On scheduled data days, the session size multiplier is automatically reduced:
- FOMC day: ×0.5 (entire session is event-driven — GEX levels often break)
- Other major data (CPI, NFP, PCE, GDP): ×0.75

The intuition: on major data days, large macro surprises (3-sigma NFP, unexpected Fed language) override dealer hedging flows completely. GEX levels that have been reliable all week become irrelevant for hours. Reducing size on these days is not optional — it is baked into the framework.
""")

    # ─────────────────────────────────────────────────────────
    with tab6:
        st.markdown("### Execution Framework — Full Explanation")
        st.markdown("""
#### Session Windows and Size Multipliers

The time of day is not a secondary consideration — it is a primary input to position sizing. GEX-based dealer hedging flows are only reliable when liquidity is high and bid-ask spreads are tight. In thin conditions, the same price levels exist but the flows that enforce them are absent.

| Session | Time (ET) | Size multiplier | Rationale |
|---------|-----------|----------------|-----------|
| Globex (overnight) | Until 9:30am | **0.0×** | GEX data is end-of-day. Overnight moves are unrelated to dealer gamma flows. No setups. |
| RTH Open | 9:30–9:45am | **0.0×** | Opening rotation is driven by overnight order imbalance, not gamma mechanics. Price often gaps through GEX levels without dealer response. |
| IB Forming | 9:45–10:30am | **0.5×** | Initial Balance forming. Market is still discovering price. GEX levels can attract price but flow confirmation is noisy. Only trade with strong confirmation. |
| **Morning (Prime Time)** | **10:30am–12pm** | **1.0×** | **Peak GEX reliability.** Options flow is actively updating, bid-ask spreads are tight, dealer hedging is most responsive. All 5 setups are valid here. |
| Midday | 12–2pm | **0.35×** | Book is thin. Algorithmic programs fill the vacuum with low-information flow. False touches at GEX levels are common. Reduce size by 50%+. |
| Afternoon | 2–3pm | **0.65×** | Liquidity returns as 0DTE gamma becomes more acute. Reassess 0DTE-specific setups. |
| Close/MOC | 3–4pm | **0.25×** | Market-on-close (MOC) imbalances and index-rebalancing flows override GEX mechanics. Setup 5 (0DTE pin) is the only valid setup after 3:30pm. |
| Post-RTH | After 4pm | **0.0×** | No gamma setups. Monitor only. |

On FOMC days: multiply the base size multiplier by 0.5. On other major data days (CPI, NFP, PCE, GDP): multiply by 0.75.

---

#### The 5 Setup Scoring System

Each setup is scored on five dimensions, each 0-1:

| Dimension | What it measures |
|-----------|-----------------|
| **Gamma alignment** | How strongly the GEX regime supports this setup type |
| **Orderflow confirmation** | Whether footprint/DOM shows absorption/initiative consistent with the setup |
| **TPO context** | Whether the market profile structure supports the trade (prior value areas, single prints) |
| **Level freshness** | How recently the GEX level was tested — fresh levels work better than crowded ones |
| **Event risk** | Whether scheduled events could override the setup |

Setup score = weighted average of these five dimensions. The dashboard shows estimated hit rates per setup:
- Setup 1 (Gamma Bounce): ~55% historical
- Setup 2 (Gamma Fade): ~52% (lower because of structural long bias)
- Setup 3 (Flip Breakout): ~45% (confirmation is hard to get cleanly)
- Setup 4 (Exhaustion Reversal): ~40% (rarest, lowest conviction)
- Setup 5 (0DTE Pin): ~65% (highest, but only valid in a narrow time window)

---

#### Pre-Trade Checklist

Before every trade, **all relevant boxes must be confirmed** or the trade does not exist:

**Auto-verified from live data (the dashboard checks these):**
1. ✅ GEX level identified from current session data
2. ✅ Gamma regime confirmed (above or below flip, not within ±0.5%)
3. ✅ Session size multiplier ≥ 0.5 (trading is active)
7. ✅ Risk within VIX-adjusted limits (VIX >35 = 50% size, VIX 25-35 = 75% size)
9. ✅ No FOMC/CPI/NFP within 2 hours
10. ✅ Not in first 15 minutes (RTH open) or post-3:30pm (unless Setup 5)

**Requires your judgment (cannot be automated):**
4. ⬜ Orderflow confirmation: absorption print (Setup 1/2), initiative flow (Setup 3), or volume climax + delta divergence (Setup 4)
5. ⬜ Setup classification: does this trade map to one of the 5 defined setups? If not, pass.
6. ⬜ Stop defined at a structural level (GEX level invalidated, regime changed) before entry
8. ⬜ R:R acceptable: ≥2:1 for Setups 1-3, ≥3:1 for Setup 4, ≥1.3:1 for Setup 5

---

#### The 8 Failure Modes

These are the most common reasons GEX-based trades fail even when every check is green:

**FM1 — Stale GEX Data**
OI only updates once per day (post-close, from OCC). Intraday changes in positioning are not reflected. If a large institutional order has altered the OI structure since the previous close (e.g., a major roll), the bars on the chart are wrong. *Diagnostic: require 3 separate touches at the level before trusting it intraday.*

**FM2 — Gamma Level Crowding**
Round-number flip levels (500, 580, 600) that have been published by multiple GEX services (SpotGamma, GEXBot, etc.) attract heavy front-running. Retail and institutional traders all expect the level to hold — which creates congestion and stop-hunting rather than clean absorption. *Diagnostic: the dashboard flags flip levels at round numbers (÷5 == 0) with high regime stability.*

**FM3 — Regime Transition Mid-Trade**
A trade entered in positive gamma can find itself in negative gamma if the flip level breaks during the trade. Mean-reversion logic becomes invalid the moment price holds below the flip. *Diagnostic: monitor the flip distance continuously. If the flip is breached and held for 3+ bars, the thesis is invalidated.*

**FM4 — Exogenous Shock Override**
A 3-sigma macro event (unexpected Fed statement, surprise NFP, geopolitical escalation) creates information that overrides dealer hedging flows entirely. Dealers still mechanically hedge, but the underlying move is large enough that GEX levels become noise. *Rule: exit at market price. Do not defend GEX levels against macro flow.*

**FM5 — Footprint Spoofing**
A single large absorption print on the footprint chart may be manufactured — placed at the bid/ask to appear as absorption, then pulled before execution. *Rule: require 3+ consecutive bars of confirmed absorption before entering. One print is noise.*

**FM6 — Correlation Regime Break**
When bonds or FX are driving equity direction more than options mechanics — a common occurrence during major macro regime transitions — GEX levels stop working. The correlation between GEX flows and price action breaks down. *Diagnostic: VIX >25 or the correlation between SPY and TLT flipping positive are warning signs.*

**FM7 — OpEx Regime Decay**
Options that expire on a given OpEx day are removed from the system. The remaining open interest is from the next expiry cycle, and those contracts have lower gamma (longer time to expiry). GEX levels weaken after the dominant expiry. *Rule: treat all GEX levels as weaker for 1-2 days after monthly OpEx.*

**FM8 — Delta-Hedging Timing Lag**
Dealers do not hedge continuously. They hedge in batches when their delta exposure exceeds internal thresholds. A GEX level is a zone (±4-8 ticks), not a laser-precise price. *Rule: always use a buffer on stops placed at GEX levels. A stop at exactly 5800 against a 5800 GEX node will be run.*

---

#### Kelly Fraction and Position Sizing

The Kelly criterion is the mathematically optimal fraction of capital to bet given a known edge and payoff ratio:
```
f* = (edge × payoff − loss_prob) / payoff

Where:
  edge      = P(win) - 0.5 (the true edge above coinflip)
  payoff    = R:R ratio (how much you win vs how much you lose)
  loss_prob = 1 - P(win)
```

Example: P(win) = 60%, payoff (R:R) = 2:1
```
f* = (0.60 × 2 − 0.40) / 2 = (1.20 - 0.40) / 2 = 0.40 = 40% of capital
```

The dashboard shows **half-Kelly** (35% for 1-day, 50% for 21-day and 63-day). Half-Kelly is standard professional practice because:
1. The true edge is almost always overestimated
2. Full Kelly with an overestimated edge causes catastrophic drawdowns
3. Half-Kelly loses only ~25% of the theoretical maximum growth rate but eliminates blowup risk

The final position size = Kelly fraction × session size multiplier × VIX adjustment:
- VIX >35: ×0.5 (extremely high vol — option prices are expensive, ranges are wide)
- VIX 25-35: ×0.75
- VIX <25: ×1.0
""")

    # ─────────────────────────────────────────────────────────
    with tab7:
        st.markdown("### Structural Risk Signals — Full Explanation")
        st.markdown("""
These are regime gates, not directional trading signals. When triggered, they reduce the weight given to bullish signals across the entire model. They are the difference between a bull setup in a healthy environment and a bull setup in a stress environment.

---

#### Sahm Rule

**Origin:** Developed by economist Claudia Sahm (2019), initially for automatic fiscal stabiliser design. Adopted by traders as a real-time recession onset indicator.

**Formula:**
```
Sahm = (3-month average UNRATE) − (12-month minimum UNRATE)
```

**Why this formula works:** Unemployment rises slowly at first, then accelerates. A small increase from the cycle low is the early warning — it precedes the acceleration that everyone recognises as a recession. By the time unemployment is obviously elevated, the recession is already several months old.

The 3-month average smooths out monthly noise. Comparing to the 12-month low anchors to the cycle peak of the labour market.

**Historical reliability:** Every US recession since World War II has been preceded or accompanied by a Sahm reading ≥0.50. There have been no false positives at the 0.50 threshold in the post-WWII sample.

**Thresholds and dashboard behaviour:**

| Reading | Status | Dashboard effect |
|---------|--------|-----------------|
| < 0.10 | Normal | No adjustment |
| 0.10 – 0.30 | Elevated | Monitor — early softening |
| 0.30 – 0.50 | Watch zone | Mild compression of bullish signals |
| **≥ 0.50** | **Triggered** | **Significant compression — base rate of bull market outcome is lower** |

**Important nuance:** A triggered Sahm Rule does not mean sell everything immediately. Recessions unfold over 12-18 months and equity markets have historically rallied significantly *during* recessions (often pricing in the recovery). The Sahm trigger changes the *base rate assumption* — bull probability starts from a lower prior.

---

#### HY Spread (ICE BofA High Yield OAS)

**What it is:** The Option-Adjusted Spread (OAS) on the ICE BofA High Yield bond index (FRED ticker: BAMLH0A0HYM2). This is the extra yield (in basis points) investors demand to hold speculative-grade (junk) corporate debt instead of equivalent Treasuries.

**OAS vs raw spread:** OAS removes the value of embedded call options in bonds (which distort the raw spread). It is a cleaner measure of pure credit risk premium.

**Why credit leads equity:**
1. Institutional credit investors are generally better-informed than equity retail investors
2. Corporate treasurers manage credit maturities actively — stress shows up in funding decisions before it shows up in earnings
3. Credit defaults are existential events; equity drawdowns are not. Credit markets price tail risk faster.

**Historical context:**

| Level | Context | Regime implication |
|-------|---------|-------------------|
| < 300bp | Complacency | Tighter than historical average (~400bp). Potential for sharp widening. |
| 300–450bp | Normal | Healthy credit conditions. Consistent with expansion. |
| 450–600bp | Elevated | Risk appetite deteriorating. Credit tightening. |
| **600–1000bp** | **Recession pricing** | Default risk rising. Consistent with late cycle or contraction. |
| > 1000bp | Systemic stress | GFC-level. Many normal signals break down. |

**The HYG/LQD signal vs the HY spread level:** These are different questions.
- HYG/LQD momentum (in the probability model) asks: is credit risk appetite *improving or deteriorating* in the last 21 days?
- HY spread level asks: where are we *absolutely* in the credit cycle?

Both are needed. You can have a healthy HYG/LQD momentum reading while still being in a high-stress environment if spreads are at 700bp and slowly tightening.

---

#### Three Puts (Narrative Context Only)

These are three structural backstops that can prevent or limit severe market declines. They are **not inputs to the probability model** (they share underlying data with signals already in the model — including them would double-count yield curve and unemployment data). They are shown as context for understanding the macro backdrop.

**Fed Put**
The Federal Reserve's implicit commitment to provide liquidity and cut rates if financial conditions become too tight or if the economy contracts severely.

Score is high (Fed Put available) when:
- 10-year yields are falling (bond market pricing rate cuts)
- Unemployment is rising (cutting window is open — Fed mandate triggered)
- Core CPI is below 3% (inflation doesn't constrain the Fed)

Score is low (Fed Put constrained) when:
- Core CPI > 3% — the Fed cannot cut without losing inflation credibility
- This is the worst case: recession risk present but Fed is unable to respond

**Treasury Put**
The Treasury's ability to inject liquidity through TGA drawdowns and RRP drainage.

- TGA drawdown: Treasury spending down its Fed cash account floods the banking system with dollars within days. Every billion drained from the TGA appears in net liquidity.
- RRP drainage: When the Fed's reverse repo facility shrinks, money that was parked there (earning the RRP rate) moves into money market funds, repo markets, and eventually risk assets.
- When both TGA and RRP are near zero, this mechanism is exhausted.

Score reflects: how much runway remains in these liquidity-injection mechanisms.

**Political/Trump Put**
The market's expectation that significant political intervention (tariff pauses, pressure on the Fed, direct support measures) will occur to prevent sustained market declines.

Score rises when:
- Market is in the "intervention zone" (>7% off 6-month highs)
- Fear is elevated (VIX elevated, sentiment negative)
- Political actors face electoral incentives to support markets

This is the most subjective of the three. It is narrative context, not a quantitative model.

---

#### NFCI (National Financial Conditions Index)

**Source:** Federal Reserve Bank of Chicago, published weekly.

**Construction:** NFCI is a composite of 105 financial variables across three sub-categories:
- Risk (equity volatility, credit spreads, funding costs)
- Credit (bank lending standards, credit growth, leverage)
- Leverage (debt-to-income ratios, asset prices relative to trend)

Each variable is measured in z-score terms relative to its own history.

**Interpretation:**
- NFCI = 0: financial conditions exactly at historical average
- NFCI > 0: tighter than average (restrictive)
- NFCI < 0: looser than average (accommodative)

**Dashboard use:** NFCI is one of the three coincident condition signals. It is orthogonal to the equity-based fear score (NFCI measures credit/funding stress, not equity vol) and to the net liquidity direction (NFCI is a composite index, not a flow measure).

---

#### The Fear Score

The fear score (0-100) is a VIX-based composite:
- VIX absolute level (current vs historical)
- VIX term structure (front/back ratio — fear premium)
- Short-term SPY price action

Fear > 70: high fear regime. The 1-day probability model caps the bull signal at a lower maximum. This reflects the empirical observation that in high-fear regimes, dealers are not supporting rallies — they are hedging their own book aggressively.

Fear > 55 but ≤ 70: moderate stress. Mild signal dampening.
""")

    # ─────────────────────────────────────────────────────────
    with tab8:
        st.markdown("### Setup, Data Sources & Practical Notes")
        st.markdown("""
#### Data Sources and Update Frequency

| Data | Source | FRED Ticker | Update frequency |
|------|--------|------------|-----------------|
| 2-Year Treasury yield | FRED | DGS2 | Daily |
| 10-Year Treasury yield | FRED | DGS10 | Daily |
| 30-Year Treasury yield | FRED | DGS30 | Daily |
| 3-Month Treasury yield | FRED | DGS3MO | Daily |
| 10-Year TIPS (real rate) | FRED | DFII10 | Daily |
| M2 money supply | FRED | M2SL | Weekly |
| Fed balance sheet | FRED | WALCL | Weekly |
| Treasury cash (TGA) | FRED | WTREGEN | Weekly |
| Reverse Repo (RRP) | FRED | RRPONTSYD | Daily |
| Bank reserves | FRED | WRBWFRBL | Weekly |
| Bank credit | FRED | TOTBKCR | Weekly |
| HY Spread (OAS) | FRED | BAMLH0A0HYM2 | Daily |
| ISM New Orders | FRED | AMTMNO | Monthly |
| Unemployment rate | FRED | UNRATE | Monthly |
| Initial jobless claims | FRED | ICSA | Weekly |
| Money market funds | FRED | WRMFSL | Weekly |
| GDP (real, chained) | FRED | GDPC1 | Quarterly |
| NFCI | FRED | NFCI | Weekly |
| Equity prices (SPY, QQQ, IWM, VIX, HYG, LQD, UUP, COPX, GLD) | yfinance | — | ~15min delayed |
| Options chain (OI, IV) | yfinance | — | OI: post-close; IV: delayed |
| Options chain (IV only) | Schwab API | — | Real-time (with auth) |
| News headlines | RSS (Fed, BLS, BEA, Reuters, BBC, Al Jazeera, CNBC, FT) | — | 5-minute cache |

---

#### GEX Data Limitations (Critical to Understand)

**Open Interest updates once per day.** The Options Clearing Corporation publishes OI data after market close — typically around 6:30am ET the following morning. During the trading day, every OI bar on the GEX chart reflects the previous night's positioning.

This means:
- The bar heights are static during the trading day
- Large intraday flows (institutional rolls, new positions) are invisible until the next morning
- The gamma flip level is based on yesterday's OI

**With Schwab connected:**
Schwab provides live implied volatility per contract. Since IV is an input to the gamma calculation (`Gamma = N'(d1) / (S × σ × √T)`), changing IV changes gamma even with fixed OI. The flip level therefore moves modestly intraday based on vol surface shifts. This makes the level more accurate — but it is still based on static OI.

**Without Schwab:**
Both OI and IV are end-of-day. The flip level is completely static until the next morning's OCC data.

**Practical implication:** Always treat GEX levels as zones (±0.3%), not precise prices. Require multiple touches and footprint confirmation before trusting a level intraday. If a level fails cleanly (3+ bars holding below it), assume the OI structure has changed and the level is stale.

---

#### Schwab API Setup

1. Go to developer.schwab.com and create an app (Trader API — Individual)
2. Set the callback URL to exactly your Streamlit app URL (e.g., `https://your-app.streamlit.app/`)
3. Add your App Key and App Secret to Streamlit secrets (see below)
4. In the dashboard, navigate to the Schwab/TOS tab and click **Authorise**
5. Complete the OAuth2 flow — log in with your Schwab credentials and approve the app
6. The token is stored in Supabase and persists across Streamlit restarts

With Schwab connected, the GEX chart shows real-time IV-adjusted gamma. Without it, the dashboard still functions — it uses yfinance IV, which is accurate enough for the flip level and regime classification.

---

#### Streamlit Secrets Configuration

```toml
# .streamlit/secrets.toml

FRED_API_KEY         = "your_fred_api_key"          # from fred.stlouisfed.org/api
SCHWAB_CLIENT_ID     = "your_schwab_app_key"         # from developer.schwab.com
SCHWAB_CLIENT_SECRET = "your_schwab_secret"          # from developer.schwab.com
SCHWAB_REDIRECT_URI  = "https://your-app.streamlit.app/"   # must match exactly

# Supabase (for Schwab token persistence across restarts)
SUPABASE_URL         = "https://xxxx.supabase.co"
SUPABASE_KEY         = "sb_publishable_xxxx"         # use the anon/public key
```

Required Supabase table (create once in the SQL editor):
```sql
CREATE TABLE schwab_tokens (
  id      TEXT PRIMARY KEY DEFAULT 'shared',
  token   JSONB NOT NULL,
  updated TIMESTAMPTZ DEFAULT NOW()
);
```

---

#### Caching and Refresh Behaviour

| Data type | Cache TTL | Notes |
|-----------|-----------|-------|
| FRED macro data | 30 minutes | Infrequently updated — 30 min is appropriate |
| yfinance options chain | 60 seconds | OI is static intraday; short cache is fine |
| Schwab options chain | 60 seconds | IV is live; short cache captures intraday vol changes |
| News feeds (RSS) | 5 minutes | Balances freshness against rate limits |
| Schwab authentication | 5 minutes | Token refreshed automatically by schwab-py |

The GEX Engine page has manual refresh controls in the sidebar (30s to 30min). Shorter refresh intervals are only meaningful with Schwab connected — yfinance IV doesn't update frequently enough to benefit from sub-60s refresh.

---

#### Performance Expectations

Be honest about what this model can and cannot do:

| Horizon | Realistic AUC | What drives it |
|---------|--------------|----------------|
| 1-Day direction | 0.52–0.55 | Near the theoretical ceiling for daily equity direction |
| 5-Day direction | 0.54–0.57 | VIX term structure and DXY momentum have modest short-term predictive value |
| 21-Day direction | 0.55–0.60 | Credit and liquidity signals have more time to resolve |
| 63-Day direction | 0.58–0.64 | Macro signals are most predictive at this horizon |

AUC of 0.55 means the model is right 55% of the time on its high-conviction calls, versus a 50% coin flip. That is a real edge — but it requires correct sizing (Kelly) and strict setup discipline to monetise. A 55% AUC model with poor sizing (too large) or poor selectivity (trading every signal) will still lose money.

---

#### Disclaimer

This dashboard is for **educational and research purposes only**. It is not investment advice. Probability outputs, Kelly fractions, and setup scores are quantitative research tools to assist your own decision-making — they are not recommendations to buy or sell any security.

All trading involves substantial risk of loss. Past performance of any signal, indicator, or model does not guarantee future results. The probability outputs reflect historical statistical relationships that may not persist. Macro regimes change. Model assumptions can be wrong. Position sizes derived from Kelly fractions assume edge estimates that are uncertain.

Use this dashboard as one input among many. Always apply your own judgment. Always use stops.
""")
