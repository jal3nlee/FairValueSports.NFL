# app.py
import os, math
import pandas as pd
import streamlit as st
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from supabase import create_client, Client
from pathlib import Path
from PIL import Image

# =======================
# Auth (Supabase)
# =======================
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error("Auth not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY in your environment.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Reattach session tokens if present
if st.session_state.get("sb_access_token") and st.session_state.get("sb_refresh_token"):
    supabase.auth.set_session(
        access_token=st.session_state.sb_access_token,
        refresh_token=st.session_state.sb_refresh_token,
    )

# ===== BRANDING =====
ROOT = Path(__file__).parent.resolve()
ASSET_DIRS = [ROOT / "assets", ROOT / ".streamlit" / "assets"]

def find_asset(name: str):
    for d in ASSET_DIRS:
        p = d / name
        if p.is_file():
            return p
    return None

def newest_favicon():
    cands = []
    for d in ASSET_DIRS:
        if d.is_dir():
            cands += list(d.glob("favicon*.png"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)

LOGO_PATH = find_asset("logo.png")
FAVICON_PATH = newest_favicon()

favicon_img = None
if FAVICON_PATH:
    try:
        favicon_img = Image.open(FAVICON_PATH)
    except Exception:
        favicon_img = None

st.set_page_config(
    page_title="Fair Value Betting",
    page_icon=(favicon_img if favicon_img else "🏈"),
    layout="wide",
    initial_sidebar_state="expanded",
)

HEADER_W = 560
SIDEBAR_W = 320

if LOGO_PATH:
    st.image(str(LOGO_PATH), width=HEADER_W)

# ---- Session state priming ----
st.session_state.setdefault("sb_session", None)
st.session_state.setdefault("sb_access_token", None)
st.session_state.setdefault("sb_refresh_token", None)
st.session_state.setdefault("show_auth", False)

def _store_session(sess):
    st.session_state.sb_session = sess
    try:
        st.session_state.sb_access_token = getattr(sess, "access_token", None) or sess.get("access_token")
        st.session_state.sb_refresh_token = getattr(sess, "refresh_token", None) or sess.get("refresh_token")
    except Exception:
        pass

def _clear_session():
    st.session_state.sb_session = None
    st.session_state.sb_access_token = None
    st.session_state.sb_refresh_token = None

def _maybe_refresh_session():
    if st.session_state.sb_session is None and st.session_state.sb_refresh_token:
        try:
            res = supabase.auth.refresh_session()
            if res and getattr(res, "session", None):
                _store_session(res.session)
        except Exception:
            _clear_session()

_maybe_refresh_session()

def auth_view():
    tabs = st.tabs(["Sign in", "Create account"])
    with tabs[0]:
        with st.form("signin_form", clear_on_submit=False):
            email = st.text_input("Email", key="signin_email")
            password = st.text_input("Password", type="password", key="signin_pw")
            submit = st.form_submit_button("Sign in")
        if submit:
            try:
                supabase.auth.sign_out()
            except Exception:
                pass
            try:
                res = supabase.auth.sign_in_with_password({"email": (email or "").strip(), "password": password})
                sess = getattr(res, "session", None) or getattr(res, "session", {}) or None
                if sess:
                    _store_session(sess)
                    st.success("Signed in.")
                    st.session_state.show_auth = False
                    st.rerun()
                else:
                    st.error("Sign-in failed. Please try again.")
            except Exception as e:
                msg = str(e)
                if "Invalid login credentials" in msg:
                    st.error("Invalid email or password.")
                elif "Email not confirmed" in msg or "confirmed_at" in msg:
                    st.error("Please verify your email, then sign in.")
                else:
                    st.error(f"Sign-in failed: {msg or 'Unknown error.'}")
    with tabs[1]:
        with st.form("signup_form", clear_on_submit=False):
            full_name = st.text_input("Name", key="signup_name")
            email2 = st.text_input("Email", key="signup_email")
            pw2 = st.text_input("Password", type="password", key="signup_pw")
            submit2 = st.form_submit_button("Create account")
        if submit2:
            if not full_name.strip():
                st.warning("Please enter your full name.")
            elif not email2 or not pw2:
                st.warning("Email and password are required.")
            else:
                try:
                    supabase.auth.sign_up(
                        {"email": email2.strip(), "password": pw2, "options": {"data": {"full_name": full_name.strip()}}}
                    )
                    st.success("Account created. Check your email to verify, then sign in.")
                except Exception as e:
                    st.error(f"Sign-up failed: {str(e) or 'Try a different email or password.'}")

authed = st.session_state.sb_session is not None

# --- Sidebar ---
with st.sidebar:
    if LOGO_PATH:
        st.image(str(LOGO_PATH), width=SIDEBAR_W)
    else:
        st.write("Fair Value Betting")

st.sidebar.divider()

with st.sidebar:
    if authed:
        u = getattr(st.session_state.sb_session, "user", None)
        user_email = getattr(u, "email", None) or (getattr(u, "user_metadata", {}) or {}).get("email")
        st.success(f"Signed in{f' as {user_email}' if user_email else ''}.")
        if st.button("Log out"):
            try:
                supabase.auth.sign_out()
            except Exception:
                pass
            _clear_session()
            st.rerun()
    else:
        st.info("Free full access in September—create a free account to unlock filters and sorting.")
        if st.button("Sign in / Create account"):
            st.session_state.show_auth = True

    if st.session_state.show_auth and not authed:
        with st.expander("Account", expanded=True):
            auth_view()

# =======================
# NFL app config + helpers + odds model
# =======================

# (Paste your entire big block here: NFL config, odds helpers, fetchers,
# compute functions, display functions, etc.)
# This includes:
#  - american_to_implied_prob()
#  - devig_two_way()
#  - nfl_week_window_utc()
#  - fetch_market_lines()
#  - build_market_from_lines_moneyline/spread/total()
#  - compute_consensus_fair_probs_*
#  - best_prices_*
#  - make_rows_*
#  - run_app() function definition
# (The full block you pasted earlier.)

# ---- Run app ----
run_app()
