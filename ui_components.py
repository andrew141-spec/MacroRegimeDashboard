# ui_components.py — reusable Streamlit / Plotly helpers
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
from config import GammaRegime, REGIME_COLORS, REGIME_BG

def pill(label, value):
    return f"<span class='badge'>{label}: <b>{value}</b></span>"

def colored(v, tone):
    m = {"green":"var(--green)","teal":"var(--teal)","yellow":"var(--yellow)",
         "orange":"var(--orange)","red":"var(--red)","purple":"var(--purple)","blue":"var(--blue)"}
    return f"<span style='color:{m.get(tone,'var(--text)')}; font-weight:700'>{v}</span>"

def pbar(prob, color="var(--blue)"):
    p = float(np.clip(prob, 0, 100))
    return (f"<div class='pbar-wrap'><div class='pbar-fill' style='width:{p:.0f}%;background:{color};'></div></div>"
            f"<div class='pbar-label'>{p:.0f}%</div>")

def regime_chip(regime: GammaRegime):
    c = REGIME_COLORS.get(regime, "var(--text)")
    bg = REGIME_BG.get(regime, "rgba(255,255,255,0.05)")
    return (f"<span class='regime-chip' style='color:{c};background:{bg};border:1px solid {c}33;'>"
            f"{regime.value}</span>")

def sec_hdr(label):
    return f"<div class='section-hdr'><span>{label}</span></div>"

def plotly_dark(fig, title="", height=300):
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color="rgba(255,255,255,0.6)")),
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.85)", family="Space Mono"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
        legend=dict(font=dict(size=10)),
    )
    return fig

def gauge(value, title, vmin=0, vmax=100):
    value = float(np.clip(value, vmin, vmax))
    frac = (value - vmin) / (vmax - vmin)
    color = "#10b981" if frac < 0.35 else ("#f59e0b" if frac < 0.55 else ("#f97316" if frac < 0.75 else "#ef4444"))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": title, "font": {"size": 10}},
        number={"font": {"size": 18, "family": "Space Mono"}},
        gauge={
            "axis": {"range": [vmin, vmax], "tickfont": {"size": 9}},
            "bar": {"thickness": 0.22, "color": color},
            "steps": [
                {"range": [vmin, vmin+(vmax-vmin)*0.25], "color": "rgba(16,185,129,0.18)"},
                {"range": [vmin+(vmax-vmin)*0.25, vmin+(vmax-vmin)*0.50], "color": "rgba(245,158,11,0.18)"},
                {"range": [vmin+(vmax-vmin)*0.50, vmin+(vmax-vmin)*0.75], "color": "rgba(249,115,22,0.18)"},
                {"range": [vmin+(vmax-vmin)*0.75, vmax], "color": "rgba(239,68,68,0.18)"},
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=6,r=6,t=38,b=6),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="rgba(255,255,255,0.85)"))
    return fig

def autorefresh_js(seconds, enabled):
    if not enabled: return
    st.components.v1.html(
        f"<script>setTimeout(()=>{{window.parent.location.reload();}},{seconds*1000});</script>",
        height=0)
