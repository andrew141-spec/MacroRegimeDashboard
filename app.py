# app.py — entry point
# Run: streamlit run app.py
#
# Flat structure — all modules at root level, no subdirectories.
# This avoids sys.path issues on Streamlit Cloud.

import streamlit as st

# config must be imported first — calls st.set_page_config()
from config import page

from page_dashboard  import render_dashboard
from page_wim        import render_world_intelligence_monitor
from page_gex        import render_gex_engine, render_setups_page
from page_execution  import render_execution_page
from page_schwab     import render_schwab_page
from page_guide      import render_guide, render_probability_page

# ── Router ──────────────────────────────────────────────────────────
if   page == "Dashboard":           render_dashboard()
elif page == "GEX Engine":          render_gex_engine()
elif page == "Trade Setups":        render_setups_page()
elif page == "Execution":           render_execution_page()
elif page == "Probability Engine":  render_probability_page()
elif page == "Schwab/TOS":          render_schwab_page()
elif page == "Guide":               render_guide()
