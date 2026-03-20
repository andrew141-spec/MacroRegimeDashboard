# intel_monitor.py — World Intelligence Monitor: RSS feeds, geo scoring
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
from config import FeedItem

INTEL_CATEGORIES = {
    "fed_policy": {
        "label": "Fed & Monetary Policy",
        "icon": "🏛",
        "color": "var(--blue)",
        "bg": "rgba(59,130,246,0.08)",
        "border": "rgba(59,130,246,0.25)",
        "feeds": {
            "Fed Releases":  "https://www.federalreserve.gov/feeds/press_all.xml",
            "FOMC":          "https://www.federalreserve.gov/feeds/fomcpressreleases.xml",
            "SEC":           "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&dateb=&owner=include&count=10&search_text=&output=atom",
            "White House":   "https://www.whitehouse.gov/feed/",
        },
        "keywords": ["fed","fomc","powell","rate cut","rate hike","balance sheet",
                     "quantitative","qt","qe","tapering","dot plot","basis points",
                     "federal reserve","monetary","inflation target","repo","rrp",
                     "reverse repo","tga","treasury general","slr","warsh","mirren"],
        "weights":  {"rate cut":10,"rate hike":10,"balance sheet":8,"qt":8,"qe":8,
                     "powell":6,"fomc":6,"slr":9,"warsh":7,"repo":7,"rrp":7},
        "regime_impact": "three_puts",   # maps to thesis put
    },
    "fiscal_debt": {
        "label": "Fiscal & Debt",
        "icon": "💵",
        "color": "var(--teal)",
        "bg": "rgba(6,182,212,0.07)",
        "border": "rgba(6,182,212,0.22)",
        "feeds": {
            "US Treasury":   "https://home.treasury.gov/press-center/press-releases/rss",
            "BEA":           "https://www.bea.gov/news/rss.xml",
        },
        "keywords": ["treasury","deficit","debt ceiling","t-bill","bond auction",
                     "fiscal","spending bill","tax cut","tariff revenue","stablecoin",
                     "genius act","big beautiful bill","debt issuance","tga drawdown",
                     "budget","appropriations","continuing resolution","shutdown"],
        "weights":  {"debt ceiling":10,"t-bill":7,"tariff revenue":8,"tax cut":7,
                     "tga":8,"deficit":6,"shutdown":9,"genius act":7,"stablecoin":6},
        "regime_impact": "treasury_put",
    },
    "inflation_labor": {
        "label": "Inflation & Labor",
        "icon": "📊",
        "color": "var(--yellow)",
        "bg": "rgba(245,158,11,0.07)",
        "border": "rgba(245,158,11,0.22)",
        "feeds": {
            "BLS":           "https://www.bls.gov/feed/news_release/rss.xml",
            "BEA PCE":       "https://www.bea.gov/news/rss.xml",
        },
        "keywords": ["cpi","pce","inflation","core inflation","jobless claims","payroll",
                     "unemployment","nonfarm","labor","wage","consumer price","producer price",
                     "ppi","shelter","housing cost","services inflation","sticky",
                     "jobs report","ism","pmi","manufacturing","services"],
        "weights":  {"cpi":10,"pce":10,"nonfarm":9,"jobless claims":8,"unemployment":8,
                     "inflation":7,"pmi":6,"ism":6,"wage":7,"shelter":7},
        "regime_impact": "fed_put",
    },
    "trade_tariffs": {
        "label": "Trade & Tariffs",
        "icon": "🌐",
        "color": "var(--orange)",
        "bg": "rgba(249,115,22,0.07)",
        "border": "rgba(249,115,22,0.22)",
        "feeds": {
            "Reuters World": "https://feeds.reuters.com/reuters/worldNews",
            "CNBC":          "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "FT":            "https://www.ft.com/rss/home",
            "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
        },
        "keywords": ["tariff","trade war","trade deal","sanction","export control",
                     "import duty","90-day pause","liberation day","china trade",
                     "supply chain","reshoring","onshoring","wto","retaliatory",
                     "trade deficit","current account","customs","duties"],
        "weights":  {"tariff":10,"trade war":9,"sanction":8,"90-day pause":8,
                     "liberation day":7,"export control":7,"trade deal":7,"china":5},
        "regime_impact": "trump_put",
    },
    "geopolitical": {
        "label": "Geopolitical Risk",
        "icon": "⚔",
        "color": "var(--red)",
        "bg": "rgba(239,68,68,0.07)",
        "border": "rgba(239,68,68,0.22)",
        "feeds": {
            "BBC World":       "https://feeds.bbci.co.uk/news/world/rss.xml",
            "BBC Middle East": "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
            "Reuters World":   "https://feeds.reuters.com/reuters/worldNews",
            "Al Jazeera":      "https://www.aljazeera.com/xml/rss/all.xml",
            "Guardian World":  "https://www.theguardian.com/world/rss",
            "France 24":       "https://www.france24.com/en/rss",
            "UN News":         "https://news.un.org/feed/subscribe/en/news/all/rss.xml",
            "CISA":            "https://www.cisa.gov/cybersecurity-advisories/all.xml",
            "State Dept":      "https://www.state.gov/rss-feeds/",
        },
        "keywords": ["war","conflict","military","attack","invasion","missile","drone",
                     "nuclear","nato","iran","russia","ukraine","israel","gaza",
                     "strait of hormuz","oil supply","crude","opec","escalation",
                     "coup","regime change","sanctions","embargo","blockade"],
        "weights":  {"war":10,"nuclear":10,"invasion":9,"missile":8,"iran":7,
                     "strait of hormuz":9,"escalation":8,"crude":5,"opec":5},
        "regime_impact": "geo_shock",
    },
    "markets_liquidity": {
        "label": "Markets & Liquidity",
        "icon": "📈",
        "color": "var(--purple)",
        "bg": "rgba(139,92,246,0.07)",
        "border": "rgba(139,92,246,0.22)",
        "feeds": {
            "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
        },
        "keywords": ["margin debt","liquidity","credit spread","vix","volatility",
                     "stock market","nasdaq","s&p","mag7","magnificent seven",
                     "earnings","forward pe","valuation","buyback","ipo",
                     "hedge fund","short selling","options","gamma","repo rate",
                     "bank reserve","m2","money supply","capital flows"],
        "weights":  {"margin debt":9,"liquidity":7,"vix":7,"credit spread":8,
                     "mag7":6,"magnificent seven":6,"forward pe":7,"repo rate":7,
                     "m2":6,"money supply":6},
        "regime_impact": "market_index",
    },
    "ai_tech": {
        "label": "AI & Tech Cycle",
        "icon": "🤖",
        "color": "var(--sky)",
        "bg": "rgba(56,189,248,0.07)",
        "border": "rgba(56,189,248,0.20)",
        "feeds": {
            "Reuters Tech":  "https://feeds.reuters.com/reuters/technologyNews",
        },
        "keywords": ["artificial intelligence","ai","nvidia","microsoft","google",
                     "amazon","meta","apple","openai","data center","capex",
                     "semiconductor","chip","gpu","hyperscaler","inference",
                     "antitrust","regulation","ipo","valuation","bubble",
                     "margin debt","finra","leverage"],
        "weights":  {"nvidia":6,"openai":6,"data center":7,"semiconductor":6,
                     "antitrust":7,"bubble":8,"margin debt":8,"capex":5},
        "regime_impact": "bubble_score",
    },
}

# Flat feed map for bulk loading (deduplicated)
def _all_feeds_flat() -> Dict[str, str]:
    seen_urls, flat = set(), {}
    for cat_data in INTEL_CATEGORIES.values():
        for name, url in cat_data["feeds"].items():
            if url not in seen_urls:
                flat[name] = url
                seen_urls.add(url)
    return flat

# Backwards-compat aliases used elsewhere in the file
FEEDS_MACRO = {k: v for k, v in _all_feeds_flat().items()
               if any(k in cat["feeds"] for cat in [INTEL_CATEGORIES["fed_policy"],
                       INTEL_CATEGORIES["fiscal_debt"], INTEL_CATEGORIES["inflation_labor"]])}
FEEDS_GEO   = {k: v for k, v in _all_feeds_flat().items()
               if any(k in cat["feeds"] for cat in [INTEL_CATEGORIES["geopolitical"],
                       INTEL_CATEGORIES["trade_tariffs"]])}

RELEVANCE_KW = list({kw for cat in INTEL_CATEGORIES.values() for kw in cat["keywords"]})

# ── Geo shock: negation patterns that cancel keyword matches ──────────────
# If any negation phrase appears in the same title, the geo keyword is
# discarded.  This catches "price war", "culture war", "star wars",
# "cyber attack" on mundane targets, "missile" in sports, etc.
_GEO_NEGATION_PHRASES = [
    "price war", "culture war", "star wars", "trade war",  # trade war handled by TRADE category
    "cyber attack", "heart attack", "panic attack", "asthma attack",
    "shark attack", "dog attack", "pepper spray", "sales",
    "marketing", "ad campaign", "award", "concert", "music",
    "film", "movie", "game", "sport", "pitch", "pitcher",
    "midfielder", "linebacker", "quarterback",
    "missile defense", "anti-missile",   # these are non-escalatory
    "nuclear plant", "nuclear power", "nuclear energy",  # civilian contexts
    "tensions ease", "tension relief",   # de-escalation
]

# ── Geo signals require co-occurrence with geographic/political entities ──
# A keyword only scores if one of these context terms also appears in
# the same title (ensures the headline is actually about a place/actor).
_GEO_CONTEXT_REQUIRED = {
    "war":       ["russia","ukraine","israel","iran","gaza","china","taiwan",
                  "north korea","nato","middle east","military","troops","army"],
    "attack":    ["iran","israel","ukraine","russia","nato","military","troops",
                  "airstrikes","bombing","drone","navy","base","port"],
    "missile":   ["iran","russia","china","north korea","ukraine","hypersonic",
                  "ballistic","cruise","icbm","launch","intercept"],
    "invasion":  ["russia","china","taiwan","ukraine","nato","troops","military"],
    "nuclear":   ["iran","russia","china","north korea","weapon","warhead",
                  "icbm","deterrent","treaty"],
    "escalat":   ["iran","russia","ukraine","israel","china","taiwan","nato",
                  "military","conflict","troops"],
    "blockade":  ["strait","hormuz","taiwan","shipping","navy","port","cargo"],
}
# Keywords that score without context requirement (intrinsically specific)
_GEO_NO_CONTEXT_NEEDED = {"strait of hormuz","nuclear warhead","military invasion",
                           "nato article 5","weapons of mass"}

# Source multipliers for geo scoring — geo-specialist feeds score higher
_GEO_SOURCE_WEIGHT = {
    "Reuters World": 1.0,
    "BBC World":     1.0,
    "Reuters Business": 0.35,   # business news mis-fires on "attack" etc.
    "Fed Releases":  0.0,
    "FOMC":          0.0,
    "BLS":           0.0,
    "BEA":           0.0,
    "US Treasury":   0.0,
}

# ── Country baseline risk scores (0–50, higher = more significant event) ─────
# When a headline mentions a high-risk country, boost geo score accordingly.
# Source: WorldMonitor pattern + practitioner calibration
COUNTRY_BASELINE_RISK: Dict[str, int] = {
    "US": 5, "RU": 35, "CN": 25, "UA": 50, "IR": 40,
    "IL": 45, "TW": 30, "KP": 45, "SA": 20, "SY": 50,
    "YE": 50, "PK": 35, "MM": 45, "LB": 40, "AF": 45,
}
COUNTRY_KEYWORDS: Dict[str, List[str]] = {
    "RU": ["russia", "moscow", "kremlin", "putin", "russian"],
    "CN": ["china", "beijing", "xi jinping", "pla", "chinese"],
    "IR": ["iran", "tehran", "irgc", "khamenei", "iranian"],
    "IL": ["israel", "idf", "gaza", "netanyahu", "israeli"],
    "TW": ["taiwan", "taipei", "tsai", "taiwanese"],
    "KP": ["north korea", "pyongyang", "kim jong", "dprk"],
    "UA": ["ukraine", "kyiv", "zelensky", "ukrainian"],
    "YE": ["yemen", "houthi", "sanaa", "yemeni"],
    "SA": ["saudi arabia", "riyadh", "mbs", "aramco"],
    "SY": ["syria", "damascus", "syrian"],
}

# ── Strategic chokepoints (WorldMonitor canonical list) ────────────────────
# Headlines mentioning these WITH a conflict keyword score much higher.
STRATEGIC_CHOKEPOINTS: Dict[str, List[str]] = {
    "Suez Canal":          ["suez", "suez canal"],
    "Malacca Strait":      ["malacca", "strait of malacca"],
    "Strait of Hormuz":    ["hormuz", "strait of hormuz"],
    "Bab el-Mandeb":       ["bab el-mandeb", "bab-el-mandeb", "mandeb"],
    "Panama Canal":        ["panama canal"],
    "Taiwan Strait":       ["taiwan strait"],
    "Cape of Good Hope":   ["cape of good hope"],
    "Gibraltar":           ["gibraltar"],
    "Bosporus":            ["bosphorus", "bosporus"],
}

def _country_risk_bonus(txt: str) -> float:
    """Return a risk bonus (0–15) based on countries mentioned in headline."""
    bonus = 0.0
    for code, keywords in COUNTRY_KEYWORDS.items():
        if any(kw in txt for kw in keywords):
            bonus = max(bonus, COUNTRY_BASELINE_RISK.get(code, 0) / 10.0)
    return min(bonus, 15.0)

def _chokepoint_bonus(txt: str) -> float:
    """Return bonus score if a strategic chokepoint is mentioned."""
    for name, aliases in STRATEGIC_CHOKEPOINTS.items():
        if any(alias in txt for alias in aliases):
            return 8.0  # chokepoint disruption is high-impact by definition
    return 0.0




def google_news_rss(query: str) -> str:
    """Generate a targeted Google News RSS feed URL for any query.
    
    Useful for creating highly specific feeds without a paid news API.
    Examples:
        google_news_rss("site:federalreserve.gov OR FOMC rate decision")
        google_news_rss("tariff trade war sanctions when:1d")
        google_news_rss("strait of hormuz OR suez canal disruption")
    """
    import urllib.parse
    params = urllib.parse.urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    return f"https://news.google.com/rss/search?{params}"

def _fetch_url(url, timeout=7):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as r: return r.read().decode("utf-8", errors="ignore")

def _strip(s): return re.sub(r"\s+", " ", (s or "").strip())

def _parse_feed(xml_text, source, max_items=15):
    items = []
    try: root = ET.fromstring(xml_text)
    except: return items
    channel = root.find("channel")
    nodes = channel.findall("item")[:max_items] if channel is not None else []
    for it in nodes:
        t = it.find("title"); l = it.find("link"); p = it.find("pubDate") or it.find("date")
        title = _strip(t.text) if t is not None and t.text else ""
        if title: items.append(FeedItem(title, _strip(l.text) if l is not None else "",
                                        _strip(p.text) if p is not None else "", source))
    return items

@st.cache_data(ttl=300)
def load_feeds(feed_tuple, max_total=120):
    all_items = []
    for source, url in dict(feed_tuple).items():
        try: all_items.extend(_parse_feed(_fetch_url(url), source))
        except: pass
    return all_items[:max_total]

# Pre-compile relevance patterns once (word-boundary for all)
_RELEVANCE_PATTERNS = [
    (kw, re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE))
    for kw in RELEVANCE_KW
]

def score_relevance(items, max_keep=12):
    scored = []
    for it in items:
        txt = (it.title + " " + it.source)
        count = sum(1 for _, pat in _RELEVANCE_PATTERNS if pat.search(txt))
        if count > 0:
            scored.append((count, it))
    scored.sort(key=lambda x: -x[0])
    return [it for _, it in scored[:max_keep]]

# Pre-compile word-boundary regex patterns for each keyword once at module load.
# This prevents "war" matching "hardware", "qt" matching "squat", etc.
def _build_kw_regex(keywords: List[str]) -> List[Tuple[str, re.Pattern]]:
    """Compile word-boundary regex for every keyword.
    \b..\b prevents 'war' matching 'hardware', 'qt' matching 'squat',
    'rate cut' matching 'rate cutback', etc.
    """
    patterns = []
    for kw in keywords:
        pat = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
        patterns.append((kw, pat))
    return patterns

_CAT_PATTERNS: Dict[str, List[Tuple[str, re.Pattern]]] = {}

def _get_cat_patterns(cat_key: str, cat_data: dict) -> List[Tuple[str, re.Pattern]]:
    """Lazily build and cache compiled regex patterns per category."""
    if cat_key not in _CAT_PATTERNS:
        _CAT_PATTERNS[cat_key] = _build_kw_regex(cat_data["keywords"])
    return _CAT_PATTERNS[cat_key]


def categorise_items(items: List[FeedItem]) -> Dict[str, List[Tuple[float, FeedItem]]]:
    """
    Assign each headline to its highest-scoring category using word-boundary regex.
    Returns {cat_key: [(score, item), ...]} sorted by score descending.
    """
    cat_results: Dict[str, List] = {k: [] for k in INTEL_CATEGORIES}
    for it in items:
        txt = (it.title + " " + it.source).lower()
        best_cat, best_score = None, 0.0
        for cat_key, cat_data in INTEL_CATEGORIES.items():
            score = 0.0
            weights = cat_data["weights"]
            for kw, pat in _get_cat_patterns(cat_key, cat_data):
                if pat.search(txt):
                    score += weights.get(kw, 3.0)
            if score > best_score:
                best_score, best_cat = score, cat_key
        if best_cat and best_score > 0:
            cat_results[best_cat].append((best_score, it))
    for k in cat_results:
        cat_results[k].sort(key=lambda x: -x[0])
    return cat_results

def category_shock_score(cat_items: List[Tuple[float, FeedItem]]) -> float:
    """0–100 shock score for a single category based on weighted hits."""
    if not cat_items: return 0.0
    raw = sum(min(s, 10) for s, _ in cat_items[:10])
    return float(np.clip(raw / 100 * 100, 0, 100))

def _parse_item_age_hours(published: str) -> float:
    """
    Parse RSS pubDate string and return age in hours.
    Returns 6.0 (half a trading session) if parsing fails — a conservative
    default that doesn't inflate or deflate stale items.
    """
    if not published:
        return 6.0
    # Try common RSS date formats
    for fmt in ("%a, %d %b %Y %H:%M:%S %z",
                "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = dt.datetime.strptime(published[:31].strip(), fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            age = (dt.datetime.now(dt.timezone.utc) - parsed).total_seconds() / 3600
            return max(float(age), 0.0)
        except Exception:
            pass
    return 6.0


def _geo_item_score(it: FeedItem) -> Tuple[float, str]:
    """
    Score a single headline for geopolitical significance.

    Returns (score, reason_string).  Score 0 means the item failed
    at least one filter (negation, context requirement, source weight).

    Scoring logic:
    1. Check source weight — geo-irrelevant feeds return 0 immediately.
    2. Detect negation phrases — if any present, return 0.
    3. Check high-weight keywords with context requirements.
    4. Apply temporal decay: score *= exp(-0.04 * age_hours)
       Half-life ~17h: a 6h-old item keeps ~79% score,
       24h-old item keeps ~39%, 48h-old item keeps ~15%.
    5. Deduplicate: caller is responsible for title-hash dedup.
    """
    source_w = _GEO_SOURCE_WEIGHT.get(it.source, 0.5)
    if source_w == 0.0:
        return 0.0, ""

    txt = it.title.lower()

    # Negation check — any negation phrase cancels all geo scoring for this item
    for neg in _GEO_NEGATION_PHRASES:
        if neg in txt:
            return 0.0, f"[negated by '{neg}']"

    # High-severity keywords with context requirements
    HIGH_KW_SCORES = {
        "war": 9.0, "invasion": 9.0, "nuclear": 8.5, "missile": 7.5,
        "attack": 7.0, "escalat": 7.0, "blockade": 8.0,
        "strait of hormuz": 10.0,
    }
    # Medium-severity keywords (no context requirement needed — specific enough)
    MED_KW_SCORES = {
        "sanction": 5.0, "embargo": 5.5, "coup": 6.0,
        "military strike": 8.0, "airstrike": 8.0, "troops deployed": 7.0,
    }

    base_score = 0.0
    reason = ""

    for kw, kw_score in HIGH_KW_SCORES.items():
        if re.search(r'\b' + re.escape(kw) + r'\b', txt):
            if kw in _GEO_NO_CONTEXT_NEEDED or any(kw in phrase for phrase in _GEO_NO_CONTEXT_NEEDED):
                base_score = kw_score
                reason = f"[{kw}]"
                break
            # Check context requirement
            required = _GEO_CONTEXT_REQUIRED.get(kw, [])
            if any(ctx in txt for ctx in required):
                base_score = kw_score
                reason = f"[{kw}+context]"
                break
            # Keyword present but no geographic context — low score
            base_score = max(base_score, kw_score * 0.2)
            reason = f"[{kw} no-context]"

    if base_score == 0.0:
        for kw, kw_score in MED_KW_SCORES.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', txt):
                base_score = kw_score
                reason = f"[{kw}]"
                break

    if base_score == 0.0:
        return 0.0, ""

    # Country risk bonus — Iran/Russia/etc. headlines score higher
    base_score += _country_risk_bonus(txt)
    # Chokepoint bonus — mentions of Hormuz/Suez/etc. are inherently significant
    base_score += _chokepoint_bonus(txt)

    # Apply source weight
    scored = base_score * source_w

    # Temporal decay: half-life ~17h
    age_h = _parse_item_age_hours(it.published)
    decay = math.exp(-0.04 * age_h)
    scored *= decay

    return float(scored), reason


def geo_shock_score(items: List[FeedItem]) -> Tuple[float, List[str]]:
    """
    Compute a geopolitical shock score 0–100 from headlines.

    Improvements over naive keyword matching:
    - Negation detection eliminates "price war", "cyber attack" on mundane
      targets, "nuclear power plant" (civilian), etc.
    - Context co-occurrence requirement: "war" only scores if a geographic
      or military entity also appears in the same headline.
    - Source weighting: Reuters World/BBC World score at 1.0; business feeds
      that mis-fire on "attack" etc. score at 0.0–0.35.
    - Temporal decay (half-life ~17h): events age out naturally, so
      yesterday's unchanged headlines don't accumulate infinitely.
    - Title deduplication: the same event appearing across multiple feeds
      is counted once (using a 10-word title prefix hash).
    """
    seen_hashes: set = set()
    triggers: List[str] = []
    total_score = 0.0

    for it in items:
        score, reason = _geo_item_score(it)
        if score <= 0:
            continue
        # Deduplicate by first 10 words of title (catches cross-feed dupes)
        title_hash = " ".join(it.title.lower().split()[:10])
        if title_hash in seen_hashes:
            continue
        seen_hashes.add(title_hash)
        total_score += score
        severity = "⚠" if score >= 5.0 else "◈"
        triggers.append(f"{severity} {reason} {it.title[:70]}")

    # Normalise: cap at 100, but use a softer ceiling via tanh
    # This prevents a single event cluster from maxing the scale
    normalised = float(math.tanh(total_score / 40.0) * 100)
    return float(np.clip(normalised, 0, 100)), triggers[:8]

# ============================================================
# LEADING INDICATORS
# ============================================================
