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
    # schwab-py's client_from_token_file expects {"token": {...}} at the top level.
    # Wrap if the dict doesn't already have that structure.
    if "token" not in token_dict:
        file_payload = {"token": token_dict}
    else:
        file_payload = token_dict
    json.dump(file_payload, tmp)
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


def get_schwab_client():
    """
    Return an authenticated Schwab client backed by Supabase token storage.
    Uses session-state caching (TTL 5 min) so that:
      - A None result (no token yet) is NEVER cached — each page load re-checks Supabase.
      - A live client IS cached for up to 5 min to avoid repeated token file I/O.
    """
    # ── Fast path: return cached client if still fresh ───────────────────
    cached   = st.session_state.get("_schwab_client_obj")
    cached_t = st.session_state.get("_schwab_client_ts", 0)
    if cached is not None and (time.time() - cached_t) < 300:
        return cached

    if not SCHWAB_AVAILABLE:
        return None
    client_id     = _get_secret("SCHWAB_CLIENT_ID")
    client_secret = _get_secret("SCHWAB_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    # Try Supabase first, fall back to session state (set during local/no-Supabase auth)
    token_dict = _supabase_load_token()
    if token_dict is None:
        token_dict = st.session_state.get("_schwab_token_local")
    if token_dict is None:
        return None   # not yet authorised — caller falls back to yfinance

    # Clear any previous token-expiry flag before attempting
    st.session_state.pop("_schwab_token_expired", None)
    st.session_state.pop("_schwab_auth_error", None)

    try:
        tmp_path = _token_to_tempfile(token_dict)
        import warnings as _warnings
        with _warnings.catch_warnings(record=True):
            _warnings.simplefilter("always")
            client = schwab.auth.client_from_token_file(
                tmp_path, client_id, client_secret
            )
        # Save any refreshed token back to Supabase.
        # _token_from_tempfile reads the {"token": {...}} wrapped file;
        # unwrap before saving so Supabase always stores the flat token dict.
        refreshed_raw = _token_from_tempfile(tmp_path)
        refreshed = refreshed_raw.get("token", refreshed_raw) if refreshed_raw else None
        if refreshed and refreshed != token_dict:
            _supabase_save_token(refreshed)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        # Cache the live client in session state (TTL enforced at top of function)
        st.session_state["_schwab_client_obj"] = client
        st.session_state["_schwab_client_ts"]  = time.time()
        return client
    except Exception as e:
        err_str = str(e)
        # Evict any stale cached client so next call retries fresh
        st.session_state.pop("_schwab_client_obj", None)
        st.session_state.pop("_schwab_client_ts", None)
        # Detect refresh-token expiry / revocation — these require re-authorization,
        # not just a retry.  Surface a clear flag so the UI can prompt the user.
        _REFRESH_INDICATORS = (
            "refresh_token_authentication_error",
            "unsupported_token_type",
            "refresh_token",
            "OAuthError",
            "invalid_grant",
            "token format has changed",
        )
        is_token_expired = any(ind in err_str for ind in _REFRESH_INDICATORS)
        if is_token_expired:
            st.session_state["_schwab_token_expired"] = True
            st.session_state["_schwab_auth_error"] = (
                f"Schwab refresh token expired or revoked — please re-authorize "
                f"on the Schwab/TOS tab to restore live IV data. "
                f"(Detail: {err_str[:200]})"
            )
        else:
            st.session_state["_schwab_auth_error"] = (
                f"Schwab client error: {type(e).__name__}: {err_str[:200]}"
            )
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
    Complete the OAuth2 flow by directly exchanging the authorization code
    for tokens via the Schwab /oauth/token endpoint.

    schwab-py's client_from_manual_flow() is interactive (stdin prompts) and
    does not accept a pre-captured callback URL — it cannot be used in a
    Streamlit/server context.  We replicate the token exchange it performs
    internally using plain requests + base64, then build the token dict in
    the same format schwab-py expects so client_from_token_file works normally.

    Args:
        client_id:    Schwab app key (Client ID)
        client_secret: Schwab app secret
        redirect_uri: The callback URL registered in your Schwab app
        callback_url: The full URL the browser was redirected to after login
                      (contains ?code=...&session=...)
    Returns:
        (success: bool, message: str)
    """
    import urllib.parse
    import base64
    import requests as _requests

    try:
        # ── 1. Extract the authorization code from the redirected URL ────────
        parsed = urllib.parse.urlparse(callback_url)
        params = urllib.parse.parse_qs(parsed.query)

        # Schwab encodes the code as "code=...@" — strip the trailing @
        raw_code = params.get("code", [None])[0]
        if not raw_code:
            return False, (
                "No 'code' parameter found in the callback URL. "
                "Make sure you pasted the full URL from your browser's address bar."
            )
        # Schwab appends a literal '@' to the code value — keep it as-is for
        # the exchange request; stripping it causes a 400 error.
        auth_code = raw_code  # e.g. "Ab1Cd2Ef3...@"

        # ── 2. Exchange code for tokens ──────────────────────────────────────
        credentials = base64.b64encode(
            f"{client_id}:{client_secret}".encode("utf-8")
        ).decode("utf-8")

        headers = {
            "Authorization": f"Basic {credentials}",
            "Content-Type":  "application/x-www-form-urlencoded",
        }
        payload = {
            "grant_type":   "authorization_code",
            "code":         auth_code,
            "redirect_uri": redirect_uri,
        }

        resp = _requests.post(
            "https://api.schwabapi.com/v1/oauth/token",
            headers=headers,
            data=payload,
            timeout=15,
        )

        if resp.status_code != 200:
            return False, (
                f"Token exchange failed ({resp.status_code}): {resp.text[:300]}"
            )

        token_data = resp.json()

        # ── 3. Build token dict in the format schwab-py's token file expects ─
        # schwab-py stores: creation_timestamp + expires_in alongside the raw
        # OAuth fields so it can calculate when to refresh.
        token_dict = {
            **token_data,
            "creation_timestamp": time.time(),
        }

        # ── 4. Persist ───────────────────────────────────────────────────────
        # Clear the cache FIRST so the next get_schwab_client() call re-reads
        # from Supabase rather than returning the cached None from before auth.
        st.session_state.pop("_schwab_client_obj", None)
        st.session_state.pop("_schwab_client_ts", None)

        if not SUPABASE_AVAILABLE or _get_supabase() is None:
            st.session_state["_schwab_token_local"] = token_dict
            return True, "Token stored in session (no Supabase — local mode)"

        saved = _supabase_save_token(token_dict)
        if not saved:
            return False, "Token exchange succeeded but Supabase save failed — check SUPABASE_URL and SUPABASE_KEY"

        # ── 5. (Verification skipped for fresh tokens) ───────────────────────
        # A freshly-exchanged token is valid by construction.
        # Calling client_from_token_file here can raise spurious warnings/
        # exceptions (e.g. "token format has changed") for tokens created by
        # older schwab-py versions, causing false auth failures.
        # The real smoke-test happens the first time get_schwab_client() is
        # called from the dashboard — if the token is truly broken it will
        # surface there with a clear re-auth prompt.

        return True, "Authenticated and token saved to Supabase ✓"

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"




def get_intraday_signals(client, symbols: list = None) -> dict:
    """
    Fetch current session % change for key cross-asset signals via Schwab quotes.
    Returns dict of {SYM_pct: float, SYM_price: float}.
    Used to make compute_1d_prob genuinely intraday-aware rather than one-day-lagged.
    Call every 60-90 seconds during market hours.
    """
    if client is None:
        return {}
    if symbols is None:
        symbols = ["SPY", "HYG", "LQD", "UUP", "QQQ", "IWM", "TLT"]
    
    out = {}
    try:
        # Batch quote request — one API call for all symbols
        resp = client.get_quotes(symbols)
        if resp.status_code != 200:
            return {}
        data = resp.json()
        for sym in symbols:
            q = data.get(sym, {}).get("quote", {})
            pct  = float(q.get("netPercentChangeInDouble", 0) or 0) / 100.0
            last = float(q.get("lastPrice", 0) or q.get("mark", 0) or 0)
            out[f"{sym}_pct"]   = pct
            out[f"{sym}_price"] = last
    except Exception:
        pass
    return out

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
                            vol = int(contract.get("totalVolume", 0) or 0)
                            rows.append({
                                "strike":       strike,
                                "expiry_T":     T,
                                "iv":           float(np.clip(iv, 0.01, 5.0)),
                                "call_oi":      oi if right_char == "C" else 0,
                                "put_oi":       oi if right_char == "P" else 0,
                                "call_volume":  vol if right_char == "C" else 0,
                                "put_volume":   vol if right_char == "P" else 0,
                                # Keep gamma separate per side — critical for GEX correctness.
                                # OTM call gamma ≠ ITM put gamma at the same strike.
                                # Averaging them produces wrong GEX for both legs.
                                "call_gamma":   gk if right_char == "C" else 0.0,
                                "put_gamma":    gk if right_char == "P" else 0.0,
                                "schwab_gamma": gk,
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
                    call_volume=("call_volume", "sum"),
                    put_volume=("put_volume", "sum"),
                    # Max is correct here: one side is 0.0, the other is the real gamma.
                    # mean() would dilute a real gamma (e.g. 0.82) by averaging with 0.
                    call_gamma=("call_gamma", "max"),
                    put_gamma=("put_gamma", "max"),
                    schwab_gamma=("schwab_gamma", "mean"),  # kept for fallback
                )
                .reset_index())
        return df

    except Exception as e:
        st.session_state["_schwab_chain_error"] = f"{type(e).__name__}: {e}"
        return None
