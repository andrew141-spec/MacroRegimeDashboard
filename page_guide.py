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
        unrate=unrate, hy_spread=hy_spread_raw,
    )

    SIGNAL_LABELS = {
        "vix_ts_pct":           "VIX Term Structure [5D]",
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
    pcts = [(label, val) for label, val in pcts if val is not None]

    if not pcts:
        st.warning("No signal data available yet. Check data sources.")
        return

    def _bar_colour(v):
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
    st.markdown("*Everything explained in plain language. No finance degree required.*")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏠 Overview",
        "⚡ GEX Engine",
        "📊 Probability",
        "🌍 Intel Monitor",
        "📋 Execution",
        "⚠️ Risk Signals",
        "🔧 Setup & Data",
    ])

    # ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### What is this dashboard?")
        st.markdown("""
This is a **quantitative trading dashboard** that combines three things into one screen:

1. **Where** the market has mechanical support and resistance (GEX — options dealer positioning)
2. **Why** the market should go up or down (macro indicators — interest rates, inflation, liquidity)
3. **When** to act (session context, setup scoring, execution framework)

The goal is not to predict the market. It is to help you **trade with the odds slightly in your favour**, know when conditions are good enough to trade, and know when to stay out.

---

### The three pillars

**Pillar 1 — GEX (Gamma Exposure)**

When you buy an options contract, the dealer who sold it to you has to hedge their risk. They do this by buying or selling the underlying stock/index. This creates predictable mechanical flows.

- When dealers are **net long gamma** (positive GEX): price moves cause them to trade *against* the move. Price drops → they buy. Price rises → they sell. This **suppresses volatility** and creates mean-reversion.
- When dealers are **net short gamma** (negative GEX): price moves cause them to trade *with* the move. Price drops → they sell more. This **amplifies volatility** and creates trending conditions.

The **gamma flip** (vol trigger) is the price level where dealer positioning switches from stabilising to amplifying. Above it = positive regime (mean-reversion expected). Below it = negative regime (trend continuation expected).

**Pillar 2 — Macro Probability Engine**

Thirteen signals across three time horizons tell you the directional bias for the next day, week, and month. Each signal is measured as a percentile rank — where is today's reading compared to the last 252 trading days? A reading of 80 means conditions are more bullish than 80% of recent history. A reading of 20 means they are more bearish than 80% of recent history.

**Pillar 3 — Execution Framework**

Even with the right direction, timing and sizing matter. The execution framework tells you:
- What session you're in (Prime Time vs thin market)
- Which of the 5 setups applies right now
- How large to size the trade (Kelly fraction × session multiplier)
- What can go wrong (8 failure modes)

---

### How to read the main dashboard

**Top of page:**
- The session bar tells you what market period you're in and the size multiplier
- The 1-Day probability (large number, left) is the most actionable signal for intraday traders
- The 5D / 21D / 63D cards show the directional bias at different time horizons

**Middle of page:**
- GEX snapshot shows the gamma flip level and whether you're in positive or negative gamma regime
- The bar chart shows GEX by strike — green bars = dealers long gamma there (support/resistance), red bars = dealers short gamma there (amplifying)

**Right panel:**
- Real-time headlines categorised by type (Fed, trade, geo, markets)
- Driver alerts fire when any signal moves by more than a threshold between refreshes
""")

    # ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### GEX Engine — Explained")

        st.markdown("""
#### What is GEX?

GEX stands for **Gamma Exposure**. It measures how much dollar hedging activity dealers must do for every 1% move in the underlying asset.

The formula is:
```
GEX = Gamma × Open Interest × Contract Size × Spot Price²
```

- **Gamma** is how fast a dealer's hedge ratio changes as price moves (from the Black-Scholes options model)
- **Open Interest** is the number of open contracts
- **Contract Size** is 100 shares per standard US options contract
- **Spot²** converts to dollar terms

#### The two GEX signs

| Sign | Meaning | Market behaviour |
|------|---------|-----------------|
| **Positive (green bars)** | Dealers are net long gamma at this strike | They buy when price falls here, sell when it rises — acts as a **shock absorber** |
| **Negative (red bars)** | Dealers are net short gamma at this strike | They sell when price falls here, buy when it rises — acts as **fuel for moves** |

#### The gamma flip (vol trigger)

This is the price where the *total* dealer gamma exposure crosses zero. It is the most important level on the chart.

- **Above the flip**: positive gamma regime. The market tends to mean-revert. Options pin to strikes. Large moves often fade.
- **Below the flip**: negative gamma regime. The market tends to trend. Moves can accelerate. Stop-running is common.

The yellow dashed line on the GEX chart marks this level. The percentage shown is how far the current price is from the flip.

#### VEX and CEX (advanced)

The GEX Engine also computes two related Greeks:

**VEX (Vanna Exposure)** — how dealer hedges change when *implied volatility* changes.
- Positive VEX + rising IV → dealers buy → upward pressure
- Negative VEX + rising IV → dealers sell → downward pressure
- This is why VIX spikes can cause cascading moves in the underlying

**CEX (Charm Exposure)** — how dealer hedges change as *time passes*.
- Relevant near options expiration (OpEx)
- Positive CEX → as time passes, dealers buy → upward drift
- This is the mechanical basis for the "OpEx pin" trade

#### The 5 GEX-based trade setups

| Setup | When it applies | What to do |
|-------|----------------|-----------|
| **1 — Gamma Bounce** | Positive regime, price at a large positive GEX strike | Buy the dip — dealers mechanically buy here |
| **2 — Gamma Fade** | Positive regime, price at a large positive GEX strike from below | Fade the rally — dealers mechanically sell here |
| **3 — Flip Breakout** | Price within 0.75% of the gamma flip | Watch for a break through the flip — if it holds, regime changes |
| **4 — Exhaustion Reversal** | Negative regime, price extended >2% below flip | The rarest setup. Requires volume climax + delta divergence. 3:1 R:R minimum |
| **5 — 0DTE Pin** | After 2pm ET, large OI at a nearby strike | Fade deviations from the dominant strike. The strike acts like a gravitational magnet |

#### Why GEX data has limits

- **Open Interest only updates once per day** (after market close, via the Options Clearing Corporation). The bars on the chart are end-of-day data, not live.
- **With Schwab connected**: implied volatility updates live, which changes gamma calculations even though OI is fixed. More accurate flip levels intraday.
- **Without Schwab**: both OI and IV are end-of-day. The flip level is static until the next morning.
- **The flip is a zone, not a price**: dealers hedge in batches, not continuously. Treat it as a ±0.3% zone.
""")

    # ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Probability Engine — Explained")

        st.markdown("""
#### The core idea

The probability engine converts 13 market signals into a single question: **what are the odds the market is higher in X days?**

It does this across four time horizons, kept completely separate:

| Horizon | Question | Signals |
|---------|----------|---------|
| **1-Day** | Tomorrow's directional bias | GEX regime, VIX term structure, momentum, credit, curve |
| **5-Day** | This week | VIX term structure, DXY momentum |
| **21-Day** | This month | HYG/LQD, small-cap leadership, net liquidity, ISM |
| **63-Day** | This quarter | Curve phase, copper/gold, credit impulse, real rates, reserves, M2 |

The 1-day model is the most actionable for intraday traders. The 63-day model is for swing positions.

#### How each signal works

**VIX Term Structure** — The ratio of VIX (expected future volatility) to actual recent volatility. When this ratio is high, the market is pricing in more fear than has actually happened. That tends to be a short-term bullish contrarian signal (fear is already priced in). When low, complacency can precede a sharp move.

**DXY 5-Day Momentum** — A rising dollar (DXY up) is usually bad for risk assets. Capital flows into the US dollar and out of equities. A falling dollar is usually tailwind for stocks. This is measured over 5 days for the tactical signal.

**HYG/LQD Ratio** — HYG is a high-yield (junk) bond ETF. LQD is investment-grade bonds. When junk bonds outperform investment-grade (ratio rising), credit markets are saying risk appetite is healthy. When the ratio falls, credit stress is building — which tends to lead equity stress by days to weeks.

**Small-Cap Leadership** — The ratio of IWM (small-cap) to SPY (large-cap). Small companies are more sensitive to the economy and credit conditions than large ones. When small caps lead, it suggests broad economic confidence. When they lag, it suggests markets are hiding in safety.

**Net Liquidity 4W** — The Federal Reserve's balance sheet minus the Treasury's cash account minus the reverse repo facility. This is the best proxy for how much "free" liquidity is in the financial system. Expanding = supportive for risk assets. Contracting = headwind.

**ISM New Orders** — The manufacturing new orders component from the Institute for Supply Management survey. It leads GDP by about 3-6 months. The *direction* (rising vs falling) matters more than the level. The four quadrants are: above 50 + rising (best), above 50 + falling (watch), below 50 + rising (recovery signal), below 50 + falling (worst).

**Yield Curve Phase** — Not just whether the curve is inverted, but *how* it's moving:
- Bull steepening (both rates falling, long end faster) = most bullish — recession fears easing
- Bear steepening (short end falling faster) = mixed — Fed may be cutting into weakness
- Bull flattening (long end rallying) = risk-off flight to safety
- Bear flattening (short end rising faster) = late-cycle tightening, policy error risk

**Copper/Gold Ratio** — Copper is used in industrial production (growth). Gold is bought in fear (safety). When copper outperforms gold, it signals economic confidence. When gold outperforms, it signals fear or growth concerns. Measured over 13 weeks to capture the medium-term trend.

**Real Credit Impulse** — The change in the *change* of bank credit divided by GDP. Not the level of credit, not the growth rate — the *acceleration*. Academic research (Biggs, 2010) shows this leads equity returns by about 6 months. It measures whether the private sector is actually getting new money to spend.

**Real Rate Regime** — The yield on 10-year TIPS (Treasury Inflation-Protected Securities). This is the "real" interest rate — what you earn after inflation. When real rates are above 2.5%, they create a severe headwind for equity valuations (the discount rate is punitive). Below 0% (financial repression), stocks often do well because cash earns nothing.

**Reserve Adequacy** — Bank reserves held at the Federal Reserve. Below roughly $3 trillion has historically correlated with funding stress (the September 2019 repo market seizure happened with reserves near this level). Adequate reserves = smooth plumbing. Low reserves = repo stress risk.

**M2 YoY Growth** — The year-over-year growth rate of M2 money supply. This leads equity markets by approximately 6-9 months empirically. When money supply is expanding, there is more capital chasing assets. Contracting M2 YoY has preceded every major equity drawdown.

**Net Liquidity 13W** — Same as the 4-week version above, but measured over 13 weeks for the medium-term signal.

#### How the composite is built

Each signal is converted to a **percentile rank**: where does today's reading sit compared to the last 252 trading days?
- 80th percentile = more bullish than 80% of the past year
- 50th percentile = exactly neutral
- 20th percentile = more bearish than 80% of the past year

Signals within each horizon bucket are averaged. Then the four buckets are blended with weights:
- Tactical: 10% (noisy, high-frequency)
- Short-term: 30%
- Medium-term: 40% (most predictive)
- Coincident conditions: 20% (reality check)

Two structural risk gates can compress all signals toward 50%:
- **Sahm Rule triggered**: recession onset signal — all bullish readings are discounted
- **HY Spread ≥600bp**: recession pricing in credit markets — consistent with elevated drawdown risk

#### The 1-day model is different

The 1-day model is **GEX-regime conditioned**. GEX doesn't give a direction — it tells you how to *interpret* the other signals:

- In **positive gamma**: momentum signals are *faded*. If the market is up 1% in 5 days, that's a headwind (dealers will sell). Credit/dollar flows are the primary signals.
- In **negative gamma**: momentum signals are *followed*. If the market is down 1% in 5 days, it may continue (dealers amplify). Direction matters more than mean-reversion.
- **Near the flip**: all signals are compressed toward 50%. Uncertainty is highest here.

#### What the percentage means (and doesn't)

A 65% bull probability means: based on current conditions, similar historical setups have resolved upward about 65% of the time. It does NOT mean the market will be up tomorrow. One-day equity direction has a realistic accuracy ceiling of about 55% AUC. The value is in conditional positioning — knowing *when* the odds are better, not predicting individual outcomes.

#### Kelly fraction

The Kelly fraction is a mathematical formula for optimal bet sizing given a known edge:
```
f* = (edge × payoff ratio − loss probability) / payoff ratio
```
At 60% win probability with 2:1 R:R: f* = (0.6×2 − 0.4) / 2 = 0.40 (40% of capital)

The dashboard shows **half-Kelly** (20% in this example). Half-Kelly is standard practice because the true edge is always uncertain — full Kelly with an overestimated edge causes severe drawdowns.

*This is educational. It is not investment advice.*
""")

    # ─────────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Intel Monitor — Explained")

        st.markdown("""
#### What it does

The Intel Monitor scans **real-time news RSS feeds** across 7 categories and scores each headline for market relevance and urgency.

The 7 categories:

| Category | What it tracks | Why it matters |
|----------|---------------|----------------|
| **Fed & Monetary Policy** | Fed statements, FOMC decisions, balance sheet changes | Directly drives the Fed Put probability and liquidity conditions |
| **Fiscal & Debt** | Treasury auctions, debt ceiling, budget, TGA | Affects net liquidity — Treasury cash drawdowns inject money into markets |
| **Inflation & Labor** | CPI, PCE, NFP, ISM, wages | Determines whether the Fed can cut or must stay tight |
| **Trade & Tariffs** | Tariff announcements, trade deals, sanctions | Affects growth expectations and supply chains |
| **Geopolitical Risk** | Conflict, military activity, chokepoints | Drives the geo shock overlay that compresses bull probabilities |
| **Markets & Liquidity** | Credit spreads, volatility, repo, M2 | Coincident market stress signal |
| **AI & Tech Cycle** | Capex, semiconductor, antitrust | Bubble score for Mag7 concentration risk |

#### The geo shock score

The geo shock score (0-100) measures how much active geopolitical stress appears in current headlines. It feeds directly into the probability engine — a high geo shock score compresses the bull probability toward 50% (uncertainty increases regardless of macro conditions).

The classifier uses word-boundary matching (so "war" won't trigger on "award" or "forward"), checks exclusion phrases first ("price war", "nuclear energy" are not conflicts), and gives bonus weight to headlines involving high-risk countries (Iran, Russia, Ukraine, North Korea, Taiwan) at baseline-calibrated levels.

The 9 strategic chokepoints (Strait of Hormuz, Malacca Strait, Taiwan Strait, Suez Canal, etc.) receive a high score only when a disruption keyword also appears in the same headline — "Suez Canal expansion" scores 0, "Suez Canal blocked" scores 12.

#### Driver alerts

The right panel shows driver alerts whenever any signal changes by more than its threshold between refreshes. These are state-change notifications — they tell you something material moved, not just noise.

Thresholds:
- Fear score: ±8 points
- Bull probability: ±8 points
- Three Puts: ±8 points
- Liquidity anxiety: ±10 points
- Macro regime change: immediate alert

#### Economic calendar

The session context automatically knows when major economic releases are scheduled (FOMC, CPI, NFP, PCE, GDP). On these days, the session size multiplier is reduced (FOMC: ×0.5, other data days: ×0.75) because GEX levels become less reliable when large macro surprises override dealer hedging flows.
""")

    # ─────────────────────────────────────────────────────────
    with tab5:
        st.markdown("### Execution Framework — Explained")

        st.markdown("""
#### Session windows and size multipliers

The time of day matters enormously for GEX-based trades. Dealer hedging is most reliable during high-liquidity periods when the options market is active and bid-ask spreads are tight.

| Session | Time (ET) | Size multiplier | Why |
|---------|-----------|----------------|-----|
| Globex (overnight) | Until 9:30am | 0.0× | No GEX setups — gamma levels are end-of-day data |
| RTH Open | 9:30–9:45am | 0.0× | Opening rotation is chaotic — noise dominates signal |
| IB Forming | 9:45–10:30am | 0.5× | Initial balance being set — proceed with strong confirmation only |
| **Morning (Prime Time)** | **10:30am–12pm** | **1.0×** | **Best GEX session — full size if setup confirms** |
| Midday | 12–2pm | 0.35× | Thin book, false signals common — reduce size by 50%+ |
| Afternoon | 2–3pm | 0.65× | Reassess 0DTE gamma as expiry approaches |
| Close/MOC | 3–4pm | 0.25× | MOC (market on close) flow overrides GEX — pin trades only after 3:30 |
| Post-RTH | After 4pm | 0.0× | No gamma setups |

#### The pre-trade checklist

Before every trade, all 10 boxes must be checked or the trade does not exist:

1. ✅ GEX level identified from current data (not yesterday's)
2. ✅ Gamma regime confirmed (above or below flip)
3. ✅ Session context is favorable (size multiplier ≥0.5)
4. ⬜ Orderflow confirmation present (absorption, initiative, or exhaustion on footprint)
5. ⬜ Setup maps to one of the 5 defined setups (no improvised trades)
6. ⬜ Stop (invalidation) defined at a structural level before entry
7. ✅ Risk within VIX-adjusted limits
8. ⬜ R:R acceptable (≥2:1 for setups 1-3, ≥3:1 for setup 4, ≥1.3:1 for setup 5)
9. ✅ No FOMC, CPI, or NFP release within 2 hours
10. ✅ Not in opening 15 minutes or post-3:30pm

Items 1, 2, 3, 7, 9, 10 are auto-checked from live data. Items 4, 5, 6, 8 require your judgment.

#### The 8 failure modes

These are the most common reasons GEX-based trades fail even when everything looks correct:

**FM1 — Stale GEX Data**: OI only updates once per day. If the market structure has changed significantly since the previous close, the GEX levels are wrong. Solution: require 3 touches at the level before trusting it.

**FM2 — Gamma Level Crowding**: When a flip level is at a round number (500, 580, 600) and has been widely published, retail and institutional traders front-run it. The level becomes a self-defeating prophecy — too crowded to work cleanly. Solution: reduce expected hold percentage by 30-50%.

**FM3 — Regime Transition Mid-Trade**: Price approaching the gamma flip invalidates mean-reversion logic. A trade entered in positive gamma can find itself in negative gamma if the flip breaks. Solution: monitor the flip distance and exit if it's breached and held.

**FM4 — Exogenous Shock Override**: FOMC statements, unexpected economic data, or geopolitical events can override dealer hedging flows entirely. A 3-sigma news event makes GEX levels irrelevant for hours. Solution: exit at market price. Do not defend GEX levels against macro flow.

**FM5 — Footprint Spoofing**: A single large absorption print on the footprint chart may be manufactured (placed and then pulled from DOM). Require 3+ bars of confirmed absorption before entering. One print is noise.

**FM6 — Correlation Regime Break**: When bonds and FX are driving equity direction more than options mechanics, GEX levels don't work. This happens during major macro regime shifts. Solution: downweight GEX, follow the macro driver.

**FM7 — OpEx Regime Decay**: Options expiring on a given day remove gamma from the system. Post-OpEx, the remaining gamma is from the next expiry cycle and the levels weaken until new OI builds. Solution: after major expirations (monthly OpEx), treat all GEX levels as weaker for 1-2 days.

**FM8 — Delta-Hedging Timing Lag**: Dealers do not hedge continuously. They hedge in batches when their delta exposure exceeds internal thresholds. A GEX level is a zone (±4-8 ticks), not a precise price. Always use a buffer on stops placed at GEX levels.
""")

    # ─────────────────────────────────────────────────────────
    with tab6:
        st.markdown("### Structural Risk Signals — Explained")

        st.markdown("""
These are not directional trading signals. They are **regime gates** — when triggered, they reduce the weight given to bullish signals across the entire model.

---

#### Sahm Rule

**What it is**: Developed by economist Claudia Sahm in 2019. It measures the difference between the 3-month average unemployment rate and the 12-month low in unemployment.

**Formula**: `Sahm = (3M avg UNRATE) - (12M low UNRATE)`

**Why it works**: Unemployment tends to rise slowly at first, then accelerate. By the time the media is calling a recession, the job market has already turned. The Sahm Rule catches the *onset* — the first significant uptick from the cycle low — before it's obvious.

**Thresholds**:
- Below 0.10: Normal. No recession signal.
- 0.10–0.30: Elevated. Labour market softening but not alarming.
- 0.30–0.50: Watch zone. Monitor closely. Signal is approaching threshold.
- **≥0.50: Triggered. Historical recession onset signal.** Every post-WWII US recession has triggered this.

**How it affects the dashboard**: A triggered Sahm Rule compresses all bullish signals. The probability engine adjusts downward because the base rate of a bull market outcome is lower during recession onset conditions. This is not a trading signal on its own — recessions can unfold over 12-18 months and markets often rally during them. It is context.

---

#### HY Spread (ICE BofA High Yield OAS)

**What it is**: The Option-Adjusted Spread on ICE BofA's high-yield (junk) bond index. This measures the *extra yield* investors demand to hold high-risk corporate debt instead of Treasuries.

**Why it matters**: The HYG/LQD ratio (used in the short-term signal stack) tells you the *momentum* — is the spread tightening or widening? The HY spread level tells you *where* you are in the credit cycle. These are different questions.

**Regimes**:
- **Below 300bp**: Complacency. Credit markets are relaxed. Historical average is around 400bp. Sub-300 is unusually tight.
- **300–450bp**: Normal. Healthy credit conditions. Consistent with expansion.
- **450–600bp**: Elevated. Risk appetite is deteriorating. Credit conditions tightening.
- **600–1000bp**: Recession pricing. Credit markets are pricing in significant default risk. Consistent with late cycle or contraction.
- **Above 1000bp**: Systemic stress. GFC territory. Possible credit market seizure.

**How it affects the dashboard**: Spreads above 600bp trigger the stress gate, which compresses bullish probability signals. Spreads above 1000bp would indicate a systemically stressed environment where most normal signals break down.

---

#### Three Puts (narrative context only)

Three structural interventions that can prevent severe market declines:

**Fed Put**: The Federal Reserve's willingness to cut rates if financial conditions tighten too much. Score is high when: 10-year yields are falling, unemployment is rising (cutting window), and inflation is below the 3% threshold. Score is low when inflation is hot and the Fed's hands are tied.

**Treasury Put**: The Treasury's ability to inject liquidity through TGA (Treasury General Account) drawdowns and RRP (Reverse Repo) drainage. When the Treasury spends down its cash account, that money enters the financial system. When RRP is low, that liquidity has already been deployed.

**Trump/Political Put**: The market's expectation that significant political intervention will occur to prevent sustained market declines. Score rises when the market is in the "intervention zone" (>7% off 6-month highs) and fear is elevated.

These are **narrative context only** — they inform your understanding of the macro backdrop but are not inputs to the probability model because they share underlying data (yield curve, unemployment) with signals already in the model.
""")

    # ─────────────────────────────────────────────────────────
    with tab7:
        st.markdown("### Setup, Data Sources & Practical Notes")

        st.markdown("""
#### Data sources

| Data | Source | Update frequency |
|------|--------|-----------------|
| Interest rates (2Y, 10Y, 30Y, 3M) | FRED (Federal Reserve) | Daily |
| Inflation (CPI, Core CPI, PCE) | FRED | Monthly |
| Unemployment (UNRATE, ICSA claims) | FRED | Monthly / Weekly |
| Fed balance sheet (WALCL) | FRED | Weekly |
| Treasury cash (TGA/WTREGEN) | FRED | Weekly |
| Reverse Repo (RRPONTSYD) | FRED | Daily |
| M2 money supply | FRED | Weekly |
| NFCI (financial conditions) | FRED | Weekly |
| Real yields (DFII10 TIPS) | FRED | Daily |
| Bank reserves (WRBWFRBL) | FRED | Weekly |
| Bank credit (TOTBKCR) | FRED | Weekly |
| ISM New Orders (AMTMNO) | FRED | Monthly |
| HY Spread (BAMLH0A0HYM2) | FRED | Daily |
| Options chain (GEX) | yfinance (OI: daily, IV: realtime with Schwab) | See below |
| Equity prices (SPY, QQQ, VIX, etc.) | yfinance | Delayed ~15min |
| Sector ETFs (HYG, LQD, IWM, etc.) | yfinance | Delayed ~15min |
| News feeds | RSS (Fed, BLS, BEA, Reuters, BBC, Al Jazeera, etc.) | 5-minute cache |

#### GEX data limitation (important)

Options Open Interest only updates **once per day**, published by the Options Clearing Corporation after market close (typically around 6:30am ET the next morning).

This means:
- The GEX bars (heights) on the chart are **end-of-day data**. They do not change during the trading day.
- The gamma flip level is **static until the next morning**.
- With Schwab connected, live implied volatility changes the *gamma calculations* even though OI is fixed, so the flip level moves modestly intraday.
- For a truly live GEX, you would need a direct OCC data feed or a professional data provider.

This is a fundamental constraint of using free data — not a dashboard limitation.

#### Schwab API setup

1. Go to developer.schwab.com and create an app (Trader API — Individual)
2. Set callback URL to your Streamlit app URL
3. Add your App Key and Secret to Streamlit secrets
4. Navigate to the Schwab/TOS tab and click Authorise
5. Complete the OAuth flow — your token is stored in Supabase and persists across restarts

With Schwab connected, you get live implied volatility for more accurate GEX calculations.

#### Streamlit secrets required

```toml
FRED_API_KEY         = "your_fred_api_key"
SCHWAB_CLIENT_ID     = "your_schwab_app_key"
SCHWAB_CLIENT_SECRET = "your_schwab_secret"
SCHWAB_REDIRECT_URI  = "https://your-app.streamlit.app/"
SUPABASE_URL         = "https://xxxx.supabase.co"
SUPABASE_KEY         = "sb_publishable_xxxx"
```

#### Auto-refresh and caching

The dashboard caches data to avoid hitting API rate limits:
- Macro data (FRED): 30-minute cache
- GEX chain (yfinance): 60-second cache
- News feeds: 5-minute cache

The GEX Engine page has its own refresh controls in the sidebar with intervals from 30 seconds to 30 minutes. Shorter intervals only help meaningfully if Schwab is connected (for live IV).

#### Disclaimer

This dashboard is for **educational and research purposes only**. It is not investment advice. Probability outputs, Kelly fractions, and setup scores are quantitative tools to assist your own decision-making — they are not recommendations to buy or sell any security. Past performance of any signal or indicator does not guarantee future results. All trading involves substantial risk of loss.
""")
