# pages/schwab.py — render_schwab_page
import os, re, time, math, datetime as dt, json, tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from fredapi import Fred
from scipy import stats as scipy_stats
from scipy.stats import norm as scipy_norm
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from config import GammaState, GammaRegime, FeedItem, SetupScore, CSS
from utils import _to_1d, zscore, resample_ffill, yf_close, kelly, current_pct_rank
from config import _get_secret
from ui_components import pill, pbar, sec_hdr, plotly_dark, regime_chip, autorefresh_js
from gex_engine import build_gamma_state, compute_gex_from_chain
from schwab_api import get_schwab_client, schwab_get_spot, schwab_get_options_chain, schwab_run_auth_flow, schwab_complete_auth, _get_supabase, SCHWAB_AVAILABLE, SUPABASE_AVAILABLE
from data_loaders import load_macro, get_gex_from_yfinance, get_fwd_pe
from intel_monitor import load_feeds, geo_shock_score, score_relevance, categorise_items, category_shock_score, _all_feeds_flat, INTEL_CATEGORIES
from signals import compute_leading_stack, compute_1d_prob
from probability import compute_prob_composite, get_session_context, evaluate_setups, check_failure_modes, classify_macro_regime_abs, regime_transition_prob, driver_alerts

def render_schwab_page():
    """Schwab / ThinkorSwim OAuth2 + Supabase token storage."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("## 📈 Schwab / ThinkorSwim Connection")

    # ── Pre-flight checks ────────────────────────────────────────────────
    missing = []
    if not SCHWAB_AVAILABLE:   missing.append("`pip install schwab-py`")
    if not SUPABASE_AVAILABLE: missing.append("`pip install supabase`")
    if missing:
        st.error("Missing packages: " + " · ".join(missing))

    # ── Connection status card ───────────────────────────────────────────
    client = get_schwab_client()
    if client is not None:
        st.success("✅ Schwab API connected")
        col_t, col_r = st.columns(2)
        with col_t:
            if st.button("🔍 Test — fetch SPY quote", use_container_width=True):
                with st.spinner("Fetching..."):
                    px = schwab_get_spot(client, "SPY")
                if px:
                    st.success(f"SPY: ${px:.2f} ✓")
                else:
                    st.error("Failed — token may have expired. Re-authorise below.")
        with col_r:
            if st.button("🗑 Revoke & re-authorise", use_container_width=True):
                sb = _get_supabase()
                if sb:
                    try: sb.table("schwab_tokens").delete().eq("id","shared").execute()
                    except: pass
                st.session_state.pop("_schwab_token_local", None)
                st.session_state.pop("_schwab_client_obj", None)
        st.session_state.pop("_schwab_client_ts", None)
                st.rerun()
        st.divider()

    # ── Setup tabs ───────────────────────────────────────────────────────
    tab_setup, tab_auth, tab_supabase = st.tabs([
        "1️⃣ Schwab App Setup", "2️⃣ Authorise", "3️⃣ Supabase Setup"
    ])

    with tab_setup:
        st.markdown("""
### Create a Schwab Developer App

1. Go to **[developer.schwab.com](https://developer.schwab.com)**
2. Sign in with your **Schwab brokerage account** (same as thinkorswim)
3. Click **"Create App"**
4. Fill in:
   - App Name: anything (e.g. `GEX Dashboard`)
   - **Callback URL:** your Streamlit app URL + `/` e.g.
     `https://yourapp.streamlit.app/`
     *(for local testing use `https://127.0.0.1`)*
   - API Product: **Trader API - Individual**
5. Submit — individual accounts are usually approved instantly
6. Copy your **App Key** (= Client ID) and **Secret**

Then add these to your Streamlit secrets (`.streamlit/secrets.toml` locally,
or the Secrets panel on share.streamlit.io):

```toml
SCHWAB_CLIENT_ID     = "your-app-key-here"
SCHWAB_CLIENT_SECRET = "your-secret-here"
SCHWAB_REDIRECT_URI  = "https://yourapp.streamlit.app/"
```
""")
        cid_check = _get_secret("SCHWAB_CLIENT_ID")
        if cid_check:
            st.success(f"✅ SCHWAB_CLIENT_ID found in secrets ({cid_check[:8]}...)")
        else:
            st.warning("SCHWAB_CLIENT_ID not in secrets yet")

    with tab_supabase:
        st.markdown("""
### Set up Supabase for token storage

Supabase is a free Postgres database that stores the Schwab OAuth token
so it persists across Streamlit Cloud deploys (Streamlit's filesystem resets
on every redeploy — a local `.json` file would be wiped).

**Step 1 — Create a free Supabase project**
1. Go to **[supabase.com](https://supabase.com)** → New project
2. Note your **Project URL** and **anon/public API key** from
   Project Settings → API

**Step 2 — Create the tokens table** (run in Supabase SQL Editor):
```sql
CREATE TABLE schwab_tokens (
  id      TEXT PRIMARY KEY DEFAULT 'shared',
  token   JSONB NOT NULL,
  updated TIMESTAMPTZ DEFAULT NOW()
);

-- Row-level security: allow the anon key to read/write this table
ALTER TABLE schwab_tokens ENABLE ROW LEVEL SECURITY;
CREATE POLICY "team_access" ON schwab_tokens
  FOR ALL USING (true) WITH CHECK (true);
```

**Step 3 — Add to Streamlit secrets:**
```toml
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "eyJ..."   # anon/public key
```
""")
        sb_url = _get_secret("SUPABASE_URL")
        sb_key = _get_secret("SUPABASE_KEY")
        if sb_url and sb_key:
            sb = _get_supabase()
            if sb:
                st.success("✅ Supabase connected")
            else:
                st.error("Supabase credentials found but connection failed")
        else:
            st.warning("SUPABASE_URL / SUPABASE_KEY not in secrets yet")

    with tab_auth:
        st.markdown("### Authorise with Schwab")

        client_id     = _get_secret("SCHWAB_CLIENT_ID")
        client_secret = _get_secret("SCHWAB_CLIENT_SECRET")
        redirect_uri  = _get_secret("SCHWAB_REDIRECT_URI", "https://127.0.0.1")

        if not client_id or not client_secret:
            st.warning("Add SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET to secrets first (tab 1)")
            return

        if client is not None:
            st.success("Already connected — use 'Revoke & re-authorise' above if you need a fresh token.")
            # Don't return — fall through so user can still re-auth if needed

        st.info(f"Redirect URI: `{redirect_uri}`  ← must match your Schwab app exactly")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Step 1 — Generate auth URL**")
            if st.button("🔗 Generate Auth URL", use_container_width=True):
                url = schwab_run_auth_flow(client_id, client_secret, redirect_uri)
                if url and not str(url).startswith("error"):
                    st.session_state["_schwab_auth_url_display"] = url
                else:
                    st.error(f"Failed: {url}")

            stored_url = st.session_state.get("_schwab_auth_url_display","")
            if stored_url:
                st.markdown("**Open this URL in your browser:**")
                st.code(stored_url, language="text")
                st.caption(
                    "Log in with Schwab. You'll be redirected to your callback URL — "
                    "the page may not load. Copy the full URL from the address bar."
                )

        with col2:
            st.markdown("**Step 2 — Paste callback URL**")
            callback = st.text_input(
                "Paste the full redirect URL",
                placeholder="https://yourapp.streamlit.app/?code=ABC&session=...",
                key="schwab_callback_input",
            )
            if st.button("✅ Complete Auth", type="primary", use_container_width=True):
                if not callback.strip():
                    st.warning("Paste the redirect URL first")
                else:
                    with st.spinner("Exchanging token..."):
                        ok, msg = schwab_complete_auth(
                            client_id, client_secret, redirect_uri, callback.strip()
                        )
                    if ok:
                        st.success(f"✅ {msg}")
                        st.balloons()
                        st.session_state.pop("_schwab_client_obj", None)
        st.session_state.pop("_schwab_client_ts", None)  # ensure cache is cleared before rerun
                        time.sleep(0.5)            # brief pause so Supabase write propagates
                        st.rerun()
                    else:
                        st.error(f"Auth failed: {msg}")
                        st.markdown("""
**Common causes:**
- Auth code expired (> 30s) — generate a new URL and try immediately
- Redirect URI mismatch — check it matches your Schwab app exactly
- Wrong Client ID or Secret
- Supabase not configured — token can't be saved
""")
