# app.py — entry point
# Run: streamlit run app.py
#
# Module structure:
#   app.py              ← entry point (this file, ~30 lines)
#   config.py           ← imports, page config, CSS, dataclasses, nav
#   utils.py            ← math helpers (zscore, kelly, rolling_pct etc)
#   ui_components.py    ← Streamlit/Plotly display helpers
#   gex_engine.py       ← GEX computation (gamma, flip, regime)
#   schwab_api.py       ← Schwab OAuth2 + Supabase token storage
#   data_loaders.py     ← FRED, yfinance, options chain fetching
#   intel_monitor.py    ← RSS feeds, geo scoring, World Intel Monitor
#   signals.py          ← leading indicator stack + 1-day model
#   probability.py      ← probability composite, session, setups, regime
#   pages/
#     dashboard.py      ← main dashboard render
#     wim.py            ← World Intelligence Monitor page
#     gex.py            ← GEX Engine + Trade Setups pages
#     execution.py      ← Execution page
#     schwab.py         ← Schwab/TOS auth page
#     guide.py          ← Guide page

import streamlit as st

# config must be imported first — it calls st.set_page_config()
from config import page  # noqa: F401 (triggers page config + nav)

from pages.dashboard  import render_dashboard
from pages.wim        import render_world_intelligence_monitor
from pages.gex        import render_gex_engine, render_setups_page
from pages.execution  import render_execution_page
from pages.schwab     import render_schwab_page
from pages.guide      import render_guide, render_probability_page

# ── Router ──────────────────────────────────────────────────────────
if   page == "Dashboard":           render_dashboard()
elif page == "GEX Engine":          render_gex_engine()
elif page == "Trade Setups":        render_setups_page()
elif page == "Execution":           render_execution_page()
elif page == "Probability Engine":  render_probability_page()
elif page == "Schwab/TOS":          render_schwab_page()
elif page == "Guide":               render_guide()
