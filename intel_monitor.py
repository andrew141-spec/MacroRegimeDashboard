# intel_monitor.py — World Intelligence Monitor: RSS feeds, geo scoring
import re, math, datetime as dt
import urllib.parse
from typing import List, Dict, Tuple, Optional
import numpy as np
import streamlit as st
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
            "SEC":           "https://www.sec.gov/news/pressreleases.rss",
            "White House":   "https://news.google.com/rss/search?q=site%3Awhitehouse.gov&hl=en-US&gl=US&ceid=US%3Aen",
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
            "State Dept":    "https://news.google.com/rss/search?q=site%3Astate.gov&hl=en-US&gl=US&ceid=US%3Aen",
            "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
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
            "CNBC Economy":  "https://www.cnbc.com/id/20910258/device/rss/rss.html",
            "FT Economy":    "https://www.ft.com/rss/home/uk",
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
            "CNBC Trade":    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "FT Trade":      "https://www.ft.com/rss/home",
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


# Source multipliers for geo scoring — geo-specialist feeds score higher
_GEO_SOURCE_WEIGHT = {
    # Primary geo sources — full weight
    "Reuters World":   1.0,
    "BBC World":       1.0,
    "BBC Middle East": 1.1,   # higher precision for ME conflict
    "Al Jazeera":      1.0,
    "Guardian World":  0.9,
    "France 24":       0.9,
    "UN News":         0.8,
    "CISA":            0.7,   # cyber/infrastructure
    # Business/trade sources — partial geo relevance
    "Reuters Business":0.35,
    "CNBC Economy":    0.2,
    "CNBC Trade":      0.3,
    "FT Trade":        0.3,
    "FT Economy":      0.2,
    "Yahoo Finance":   0.1,
    # Policy sources — no geo scoring
    "Fed Releases":    0.0,
    "FOMC":            0.0,
    "BLS":             0.0,
    "BEA":             0.0,
    "BEA PCE":         0.0,
    "US Treasury":     0.0,
    "White House":     0.2,
    "State Dept":      0.6,   # diplomatic announcements matter
    "SEC":             0.0,
}

# ============================================================
# NEWS CLASSIFIER — word-boundary regex + exclusions + chokepoints
# ============================================================

# Google News RSS helper — targeted feeds without a paid API
def google_news_rss(query: str) -> str:
    params = urllib.parse.urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    return f"https://news.google.com/rss/search?{params}"

# Short keywords that must match as whole words (\bkw\b) to avoid
# "war" matching "award", "forward", "hardware" etc.
_SHORT_KEYWORDS = {
    "war", "coup", "ban", "vote", "riot", "hack", "flu", "ban",
    "nuke", "bomb", "arms", "raid",
}

# Compiled regex cache — built once per keyword
_REGEX_CACHE: Dict[str, re.Pattern] = {}

def _kw_regex(kw: str) -> re.Pattern:
    if kw not in _REGEX_CACHE:
        escaped = re.escape(kw)
        pattern = rf"\b{escaped}\b" if kw in _SHORT_KEYWORDS else escaped
        _REGEX_CACHE[kw] = re.compile(pattern, re.IGNORECASE)
    return _REGEX_CACHE[kw]

def _matches(kw: str, text: str) -> bool:
    return bool(_kw_regex(kw).search(text))

# Exclusion terms — if any appear in the title, suppress geo scoring entirely
_EXCLUSION_TERMS = [
    "price war", "culture war", "star wars", "trade war", "gang war",
    "cyber attack on business", "heart attack", "panic attack", "asthma attack",
    "shark attack", "dog attack", "snake attack", "animal attack",
    "nuclear power", "nuclear energy", "nuclear plant", "nuclear reactor",
    "nuclear fusion", "nuclear medicine",
    "tensions ease", "ceasefire", "peace deal", "peace talks",
    "sports", "recipe", "celebrity", "movie", "concert", "festival",
    "award", "forward", "hardware", "software", "password",
    "missile defense system", "anti-missile",
    "market war", "streaming war", "streaming wars", "tech war",
    "ad war", "advertising war", "ratings war",
]

# 9 strategic chokepoints — score when chokepoint + disruption keyword co-occur
_CHOKEPOINTS = {
    "suez canal":            12.0,
    "malacca strait":        11.0,
    "strait of hormuz":      12.0,
    "bab el-mandeb":         11.0,
    "bab-el-mandeb":         11.0,
    "panama canal":          10.0,
    "taiwan strait":         11.0,
    "cape of good hope":      9.0,
    "strait of gibraltar":    9.0,
    "bosphorus":              9.0,
    "strait of malacca":     11.0,
    "hormuz":                11.0,
}
_CHOKEPOINT_DISRUPTION_KW = [
    "blocked", "closed", "attack", "seized", "disrupted", "shutdown",
    "mine", "missile", "tanker", "navy", "military", "conflict",
    "houthi", "iran", "piracy", "pirate",
]

# Country baseline risk + keywords (steal structure from get-risk-scores.ts)
_COUNTRY_BASELINE = {
    "RU": 35, "CN": 25, "UA": 50, "IR": 40, "IL": 45,
    "TW": 30, "KP": 45, "SA": 20, "SY": 50, "YE": 50,
    "PK": 35, "MM": 45, "LB": 40, "AZ": 25, "ET": 30,
}
_COUNTRY_KEYWORDS = {
    "RU": ["russia", "moscow", "kremlin", "putin", "russian"],
    "CN": ["china", "beijing", "xi jinping", "chinese", "pla"],
    "IR": ["iran", "tehran", "irgc", "khamenei", "iranian"],
    "IL": ["israel", "idf", "gaza", "netanyahu", "israeli", "west bank"],
    "TW": ["taiwan", "taipei", "tsai", "taiwanese"],
    "KP": ["north korea", "pyongyang", "kim jong", "dprk"],
    "YE": ["yemen", "houthi", "sanaa", "yemeni"],
    "UA": ["ukraine", "kyiv", "zelensky", "ukrainian"],
    "SY": ["syria", "damascus", "syrian", "hts"],
    "LB": ["lebanon", "hezbollah", "beirut"],
    "SA": ["saudi", "riyadh", "mbs", "opec"],
}

# Critical keywords — very high confidence, word-boundary matched
_CRITICAL_KW: Dict[str, Tuple[float, str]] = {
    "nuclear strike":       (12.0, "military"),
    "nuclear attack":       (12.0, "military"),
    "nuclear warhead":      (11.0, "military"),
    "nato article 5":       (12.0, "military"),
    "declaration of war":   (11.0, "conflict"),
    "military invasion":    (11.0, "conflict"),
    "martial law":          (9.0,  "military"),
    "mass casualty":        (9.0,  "conflict"),
    "chemical attack":      (10.0, "terrorism"),
    "biological attack":    (10.0, "terrorism"),
    "weapons of mass":      (11.0, "military"),
    "nuclear detonation":   (12.0, "military"),
}

# High keywords — require word-boundary for short ones, context for ambiguous ones
_HIGH_KW: Dict[str, Tuple[float, str, Optional[List[str]]]] = {
    # (score, category, context_required or None)
    "invasion":       (9.0,  "conflict",  ["russia","china","taiwan","ukraine","nato","troops"]),
    "airstrike":      (8.5,  "conflict",  None),
    "airstrikes":     (8.5,  "conflict",  None),
    "missile launch": (8.5,  "military",  None),
    "troops deployed":(8.0,  "military",  None),
    "bombing":        (8.0,  "conflict",  ["military","civilians","city","town","base"]),
    "war":            (8.0,  "conflict",  ["russia","ukraine","israel","iran","gaza","china",
                                           "taiwan","nato","military","troops","army","conflict"]),
    "coup":           (8.0,  "military",  None),
    "nuclear":        (7.5,  "military",  ["iran","russia","china","north korea","weapon",
                                           "warhead","icbm","deterrent","treaty","test"]),
    "missile":        (7.0,  "military",  ["iran","russia","china","north korea","ukraine",
                                           "ballistic","cruise","icbm","launch","intercept"]),
    "attack":         (6.5,  "conflict",  ["iran","israel","ukraine","russia","nato","military",
                                           "troops","airstrike","drone","navy","base","port"]),
    "escalation":     (7.0,  "conflict",  ["iran","russia","ukraine","israel","china","taiwan",
                                           "nato","military","conflict","troops"]),
    "sanctions":      (5.5,  "economic",  None),
    "embargo":        (5.5,  "economic",  None),
    "blockade":       (7.5,  "conflict",  ["strait","hormuz","taiwan","shipping","navy","port"]),
    "riot":           (5.0,  "conflict",  ["government","police","military","capital","protest"]),
    "hack":           (5.5,  "cyber",     ["government","infrastructure","power","bank","hospital"]),
}


def _country_bonus(txt: str) -> float:
    """Return additional score based on highest-risk country mentioned.""",
    max_bonus = 0.0
    for code, kws in _COUNTRY_KEYWORDS.items():
        if any(kw in txt for kw in kws):
            baseline = _COUNTRY_BASELINE.get(code, 10)
            # Scale: baseline 50 → +2.0 bonus, baseline 10 → +0.4 bonus
            max_bonus = max(max_bonus, baseline / 25.0)
    return max_bonus


def _chokepoint_score(txt: str) -> Tuple[float, str]:
    """Return (score, label) if a strategic chokepoint + disruption co-occur.""",
    for cp, cp_score in _CHOKEPOINTS.items():
        if cp in txt:
            if any(dk in txt for dk in _CHOKEPOINT_DISRUPTION_KW):
                return cp_score, f"chokepoint:{cp}"
    return 0.0, ""


def classify_headline(title: str) -> Dict:
    """
    Classify a single headline for geo/market significance.
    Returns {level, category, score, reason, confidence}.
    Improvements over naive kw matching:
      - Word-boundary regex for short ambiguous keywords
      - Exclusion phrases checked first
      - Context co-occurrence for ambiguous keywords
      - Chokepoint detection
      - Country baseline risk bonus
    """
    lower = title.lower()

    # 1. Exclusion check — bail immediately on noise
    for ex in _EXCLUSION_TERMS:
        if ex in lower:
            return {"level": "info", "category": "general", "score": 0.0,
                    "reason": f"excluded:{ex}", "confidence": 0.1}

    # 2. Chokepoint check
    cp_score, cp_reason = _chokepoint_score(lower)
    if cp_score > 0:
        country_b = _country_bonus(lower)
        return {"level": "critical", "category": "chokepoint",
                "score": cp_score + country_b, "reason": cp_reason, "confidence": 0.95}

    # 3. Critical keywords
    for kw, (kw_score, cat) in _CRITICAL_KW.items():
        if _matches(kw, lower):
            country_b = _country_bonus(lower)
            return {"level": "critical", "category": cat,
                    "score": kw_score + country_b, "reason": kw, "confidence": 0.9}

    # 4. High keywords (with context check for ambiguous ones)
    best_score, best_cat, best_reason = 0.0, "general", ""
    for kw, (kw_score, cat, ctx_required) in _HIGH_KW.items():
        if not _matches(kw, lower):
            continue
        if ctx_required is None:
            # No context needed — keyword is specific enough
            effective = kw_score
        elif any(ctx in lower for ctx in ctx_required):
            effective = kw_score
        else:
            effective = kw_score * 0.15   # keyword present, no context → near-zero
        if effective > best_score:
            best_score, best_cat, best_reason = effective, cat, kw

    if best_score >= 1.0:
        country_b = _country_bonus(lower)
        level = "high" if best_score >= 5.0 else "medium"
        return {"level": level, "category": best_cat,
                "score": best_score + country_b, "reason": best_reason,
                "confidence": 0.8 if best_score >= 5.0 else 0.5}

    return {"level": "info", "category": "general", "score": 0.0,
            "reason": "", "confidence": 0.2}


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

def score_relevance(items, max_keep=12):
    kw = [k.lower() for k in RELEVANCE_KW]
    scored = [(sum(1 for k in kw if k in (it.title+" "+it.source).lower()), it) for it in items]
    return [it for s, it in sorted(scored, key=lambda x:-x[0]) if s > 0][:max_keep]

def categorise_items(items: List[FeedItem]) -> Dict[str, List[Tuple[float, FeedItem]]]:
    """
    Assign each headline to its highest-scoring category.
    Returns {cat_key: [(score, item), ...]} sorted by score descending.
    """
    cat_results: Dict[str, List] = {k: [] for k in INTEL_CATEGORIES}
    for it in items:
        txt = (it.title + " " + it.source).lower()
        best_cat, best_score = None, 0.0
        for cat_key, cat_data in INTEL_CATEGORIES.items():
            # Weighted keyword score
            score = 0.0
            weights = cat_data["weights"]
            for kw in cat_data["keywords"]:
                if kw in txt:
                    score += weights.get(kw, 3.0)
            if score > best_score:
                best_score, best_cat = score, cat_key
        if best_cat and best_score > 0:
            cat_results[best_cat].append((best_score, it))
    # Sort each category by score
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
    Score a single headline using classify_headline() + source weight + temporal decay.

    Key improvements over old implementation:
    - Word-boundary regex for short keywords: 'war' won't match 'award'/'forward'
    - Exclusion list handled by classifier (price war, nuclear plant, etc.)
    - 9 strategic chokepoints with disruption co-occurrence check
    - Country baseline risk bonus (Iran headline > Germany headline)
    - Context requirements embedded in _HIGH_KW dict, not a separate lookup
    """
    source_w = _GEO_SOURCE_WEIGHT.get(it.source, 0.4)
    if source_w == 0.0:
        return 0.0, ""

    result = classify_headline(it.title)
    base_score = result["score"]
    reason     = result["reason"]
    level      = result["level"]

    if base_score <= 0.0 or level == "info":
        return 0.0, ""

    # Apply source weight
    scored = base_score * source_w

    # Temporal decay: half-life ~17h
    age_h = _parse_item_age_hours(it.published)
    decay = math.exp(-0.04 * age_h)
    scored *= decay

    return float(scored), f"[{level}:{reason}]"


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
