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
    Start the OAuth2 flow using the current schwab-py manual flow API.
    Returns the Schwab authorization URL for the user to open in their browser.
    """
    if not SCHWAB_AVAILABLE:
        return None
    try:
        # Build the auth URL directly via the Schwab OAuth endpoint —
        # schwab-py no longer exposes OAuth2Client; use the documented URL format.
        import urllib.parse
        params = urllib.parse.urlencode({
            "client_id":     client_id,
            "redirect_uri":  redirect_uri,
            "response_type": "code",
        })
        auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?{params}"
        st.session_state["_schwab_oauth_redir"] = redirect_uri
        return auth_url
    except Exception as e:
        return f"error: {e}"


def schwab_complete_auth(client_id: str, client_secret: str,
                          redirect_uri: str, callback_url: str) -> Tuple[bool, str]:
    """
    Complete the OAuth2 flow using schwab.auth.client_from_manual_flow.
    Saves the resulting token to Supabase.
    Returns (success: bool, message: str).
    """
    if not SCHWAB_AVAILABLE:
        return False, "schwab-py not installed"
    try:
        tmp_path = tempfile.mktemp(suffix=".json", prefix="schwab_token_")

        # client_from_manual_flow is the current schwab-py API for server/cloud use.
        # It takes the full redirected callback URL, exchanges the code, and writes
        # the token to tmp_path.
        client = schwab.auth.client_from_manual_flow(
            api_key=client_id,
            app_secret=client_secret,
            callback_url=redirect_uri,
            token_path=tmp_path,
            redirected_url=callback_url,
        )

        token_dict = _token_from_tempfile(tmp_path)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        if token_dict is None:
            return False, "Token file empty after exchange"

        if not SUPABASE_AVAILABLE or _get_supabase() is None:
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
    symbol = symbol.strip().upper()
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
    # Normalise symbol — strip whitespace and uppercase
    # A trailing space or wrong case causes Schwab 400 "Invalid Parameter/Value"
    symbol = symbol.strip().upper()
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

        # Try to get a live quote first so strike_count is centred correctly
        if not spot:
            try:
                q = client.get_quote(symbol).json()
                q_data = q.get(symbol, {})
                spot = float(q_data.get("quote", q_data).get("lastPrice", 0) or 0) or None
            except Exception:
                pass
        spot_est = spot or 500.0

        # Restore proper params now that symbol is normalised
        st.session_state["_schwab_chain_debug"] = f"Requesting chain for symbol='{symbol}' spot_est={spot_est:.2f}"
        resp = client.get_option_chain(
            symbol,
            contract_type=schwab.client.Client.Options.ContractType.ALL,
            strike_count=40,  # ±20 from ATM — wider than 20 to capture full GEX landscape
        )

        if resp.status_code != 200:
            st.session_state["_schwab_chain_error"] = (
                f"API error {resp.status_code}: {resp.text[:200]}"
            )
            return None

        data = resp.json()

        # Use the underlying price from the response if available — most accurate
        underlying = data.get("underlying", {})
        if underlying:
            api_spot = float(underlying.get("last", 0) or underlying.get("mark", 0) or 0)
            if api_spot > 0:
                spot_est = api_spot

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
                        # Filter to ±15% — wide enough to never miss valid strikes
                        # (the user's heatmap range controls what's displayed)
                        if not (spot_est * 0.85 <= strike <= spot_est * 1.15):
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
                                "schwab_gamma": gk,  # always positive from Schwab API — sign applied via call_oi/put_oi convention in gex_engine
                            })
                    except Exception:
                        continue

        if not rows:
            # Debug: log what the response actually contained
            call_keys = list(data.get("callExpDateMap", {}).keys())[:3]
            put_keys  = list(data.get("putExpDateMap",  {}).keys())[:3]
            all_strikes = []
            for exp_map in [data.get("callExpDateMap", {}), data.get("putExpDateMap", {})]:
                for strikes_dict in exp_map.values():
                    all_strikes.extend([float(k) for k in strikes_dict.keys()])
            strike_range = f"{min(all_strikes):.0f}–{max(all_strikes):.0f}" if all_strikes else "none"
            st.session_state["_schwab_chain_error"] = (
                f"No rows after filtering. spot_est={spot_est:.2f}, "                f"raw strikes in response: {strike_range}, "                f"filter: {spot_est*0.85:.0f}–{spot_est*1.15:.0f}, "                f"expirations (calls): {call_keys}"
            )
            return None

        df = (pd.DataFrame(rows)
                .groupby(["strike", "expiry_T"])
                .agg(
                    iv=("iv", "mean"),
                    call_oi=("call_oi", "sum"),
                    put_oi=("put_oi", "sum"),
                    schwab_gamma=("schwab_gamma", "mean"),
                )
                .reset_index())
        return df

    except Exception as e:
        st.session_state["_schwab_chain_error"] = f"{type(e).__name__}: {e}"
        return None


