# schwab_api.py — Schwab OAuth2 client, Supabase token storage, options chain
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
from config import GammaState, _get_secret

# ── Optional dependencies ────────────────────────────────────────────────────
try:
    import schwab
    SCHWAB_AVAILABLE = True
except ImportError:
    SCHWAB_AVAILABLE = False

try:
    from supabase import create_client as _supabase_create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


# SCHWAB API — Connection + Data Functions
# ============================================================

# ============================================================
# SCHWAB + SUPABASE TOKEN LAYER
# ============================================================
# Architecture:
#   1. OAuth2 flow runs in the browser (Schwab → callback URL)
#   2. Token JSON is stored in Supabase (persists across Streamlit deploys)
#   3. schwab-py reads the token from a temp file on each request
#      (schwab-py needs a file path, so we write/read to tmp on each use)
#
# Supabase table schema (create once in Supabase SQL editor):
#   CREATE TABLE schwab_tokens (
#     id      TEXT PRIMARY KEY DEFAULT 'shared',
#     token   JSONB NOT NULL,
#     updated TIMESTAMPTZ DEFAULT NOW()
#   );
# ============================================================

def _get_supabase() -> Optional[object]:
    """Return a Supabase client using secrets."""
    if not SUPABASE_AVAILABLE:
        return None
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_KEY")   # use the anon/public key
    if not url or not key:
        return None
    try:
        return _supabase_create_client(url, key)
    except Exception:
        return None


def _supabase_load_token() -> Optional[Dict]:
    """Load the Schwab token dict from Supabase."""
    sb = _get_supabase()
    if sb is None:
        return None
    try:
        res = sb.table("schwab_tokens").select("token").eq("id", "shared").execute()
        if res.data:
            return res.data[0]["token"]
    except Exception:
        pass
    return None


def _supabase_save_token(token_dict: Dict) -> bool:
    """Upsert the Schwab token dict to Supabase."""
    sb = _get_supabase()
    if sb is None:
        return False
    try:
        sb.table("schwab_tokens").upsert({
            "id": "shared",
            "token": token_dict,
        }).execute()
        return True
    except Exception:
        return False


def _token_to_tempfile(token_dict: Dict) -> str:
    """
    Write token dict to a temp file and return its path.
    schwab-py requires a file path for token storage; we use a temp file
    as a bridge between Supabase (our real store) and the library.
    The temp file is recreated on every call — no filesystem persistence.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="schwab_token_"
    )
    json.dump(token_dict, tmp)
    tmp.flush()
    tmp.close()
    return tmp.name


def _token_from_tempfile(path: str) -> Optional[Dict]:
    """Read token from temp file (after schwab-py may have refreshed it)."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_resource(ttl=300)   # re-check token every 5 min
def get_schwab_client():
    """
    Return an authenticated Schwab client backed by Supabase token storage.

    Flow on each call:
      1. Load token JSON from Supabase
      2. Write to temp file (schwab-py needs a file path)
      3. Build client — schwab-py auto-refreshes if token is near expiry
      4. Read back refreshed token and save to Supabase
    """
    if not SCHWAB_AVAILABLE:
        return None
    client_id     = _get_secret("SCHWAB_CLIENT_ID")
    client_secret = _get_secret("SCHWAB_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    token_dict = _supabase_load_token()
    if token_dict is None:
        return None   # not yet authorised

    try:
        tmp_path = _token_to_tempfile(token_dict)
        client   = schwab.auth.client_from_token_file(
            tmp_path, client_id, client_secret
        )
        # Save any refreshed token back to Supabase
        refreshed = _token_from_tempfile(tmp_path)
        if refreshed and refreshed != token_dict:
            _supabase_save_token(refreshed)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return client
    except Exception:
        return None


def schwab_run_auth_flow(client_id: str, client_secret: str,
                          redirect_uri: str) -> Optional[str]:
    """
    Start the OAuth2 flow. Returns the Schwab authorization URL.
    The user opens this URL, logs in, and is redirected to redirect_uri.
    """
    if not SCHWAB_AVAILABLE:
        return None
    try:
        # schwab-py's OAuth2 helper — we use a temp path as placeholder
        tmp_path = tempfile.mktemp(suffix=".json", prefix="schwab_auth_")
        oauth    = schwab.auth.OAuth2Client(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            token_path=tmp_path,
        )
        auth_url, state = oauth.authorization_url()
        # Store state for CSRF validation
        st.session_state["_schwab_oauth_state"]    = state
        st.session_state["_schwab_oauth_tmp_path"] = tmp_path
        st.session_state["_schwab_oauth_redir"]    = redirect_uri
        return auth_url
    except Exception as e:
        return f"error: {e}"


def schwab_complete_auth(client_id: str, client_secret: str,
                          redirect_uri: str, callback_url: str) -> Tuple[bool, str]:
    """
    Complete the OAuth2 flow. Saves the token to Supabase.
    Returns (success: bool, message: str).
    """
    if not SCHWAB_AVAILABLE:
        return False, "schwab-py not installed"
    try:
        tmp_path = st.session_state.get("_schwab_oauth_tmp_path",
                                         tempfile.mktemp(suffix=".json"))
        oauth = schwab.auth.OAuth2Client(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            token_path=tmp_path,
        )
        oauth.fetch_token(authorization_response=callback_url)
        token_dict = _token_from_tempfile(tmp_path)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        if token_dict is None:
            return False, "Token file empty after exchange"
        if not SUPABASE_AVAILABLE or _get_supabase() is None:
            # Fallback: store in session_state only (local dev without Supabase)
            st.session_state["_schwab_token_local"] = token_dict
            return True, "Token stored in session (no Supabase — local mode)"
        saved = _supabase_save_token(token_dict)
        if saved:
            get_schwab_client.clear()
            return True, "Token saved to Supabase ✓"
        else:
            return False, "Supabase save failed — check SUPABASE_URL and SUPABASE_KEY"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def schwab_get_spot(client, symbol: str = "SPY") -> Optional[float]:
    """Get current quote from Schwab API."""
    if client is None:
        return None
    try:
        resp = client.get_quote(symbol)
        if resp.status_code == 200:
            data = resp.json()
            q = data.get(symbol, {})
            # Schwab quote structure: quote.lastPrice or quote.mark
            quote = q.get("quote", {})
            price = (quote.get("lastPrice")
                     or quote.get("mark")
                     or quote.get("closePrice"))
            return float(price) if price else None
    except Exception:
        pass
    return None


def schwab_get_options_chain(client, symbol: str = "SPY",
                              spot: Optional[float] = None) -> Optional[pd.DataFrame]:
    """
    Fetch options chain from Schwab API for GEX computation.

    Schwab's /marketdata/v1/chains endpoint returns:
        - impliedVolatility  per strike  ← real per-strike IV (no surface fitting needed)
        - openInterest       per strike  ← exchange-reported OI
        - delta, gamma, theta, vega      ← we use gamma directly
        - daysToExpiration               ← T

    This is cleaner than the IBKR approach: real per-strike Greeks from
    Schwab's own model, no SVI surface approximation required.

    Parameters:
        contractType: "ALL" — calls and puts
        strikeCount:  30    — ±15 strikes around ATM
        range:        "NTM" — near-the-money
    """
    if client is None:
        return None
    try:
        today    = dt.date.today()
        spot_est = spot or 500.0

        # Request near-the-money chain, 2 nearest expirations
        resp = client.get_option_chain(
            symbol,
            contract_type=schwab.client.Client.Options.ContractType.ALL,
            strike_count=30,
            include_underlying_quote=True,
            strategy=schwab.client.Client.Options.Strategy.SINGLE,
            option_type=schwab.client.Client.Options.Type.ALL,
        )

        if resp.status_code != 200:
            st.session_state["_schwab_chain_error"] = (
                f"API error {resp.status_code}: {resp.text[:200]}"
            )
            return None

        data = resp.json()
        rows = []

        for right_key, right_char in [("callExpDateMap", "C"), ("putExpDateMap", "P")]:
            exp_map = data.get(right_key, {})
            for exp_str, strikes_dict in exp_map.items():
                # exp_str format: "2025-01-17:30" (date:DTE)
                try:
                    exp_date_str = exp_str.split(":")[0]
                    exp_dt = dt.date.fromisoformat(exp_date_str)
                    T = max((exp_dt - today).days / 365.0, 1.0 / 365.0)
                except Exception:
                    continue

                for strike_str, contracts in strikes_dict.items():
                    try:
                        strike = float(strike_str)
                        # Skip far out-of-money
                        if not (spot_est * 0.88 <= strike <= spot_est * 1.12):
                            continue
                        for contract in contracts:
                            iv  = float(contract.get("volatility", 0) or 0) / 100.0
                            oi  = int(contract.get("openInterest", 0) or 0)
                            # Use Schwab's own gamma if available
                            gk  = float(contract.get("gamma", 0) or 0)

                            if iv <= 0 or not np.isfinite(iv):
                                iv = 0.18
                            rows.append({
                                "strike":      strike,
                                "expiry_T":    T,
                                "iv":          float(np.clip(iv, 0.01, 5.0)),
                                "call_oi":     oi if right_char == "C" else 0,
                                "put_oi":      oi if right_char == "P" else 0,
                                "schwab_gamma": gk if right_char == "C" else -gk,
                            })
                    except Exception:
                        continue

        if not rows:
            st.session_state["_schwab_chain_error"] = "No chain data in response"
            return None

        df = (pd.DataFrame(rows)
                .groupby(["strike", "expiry_T"])
                .agg(
                    iv=("iv", "mean"),
                    call_oi=("call_oi", "sum"),
                    put_oi=("put_oi", "sum"),
                )
                .reset_index())
        return df

    except Exception as e:
        st.session_state["_schwab_chain_error"] = f"{type(e).__name__}: {e}"
        return None


