# app.py — entry point
# Run: streamlit run app.py

import sys
import os

# Ensure the project root is on the Python path
# This is required on Streamlit Cloud where the working directory
# may not be automatically added to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
