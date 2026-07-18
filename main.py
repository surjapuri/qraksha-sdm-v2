"""
QRaksha / SDM — Official Identity Intelligence Database
FastAPI Backend v2.0

Features:
  • GitHub REST API persistent storage with local file fallback
  • JWT Role-Based Access Control (admin vs public tiers)
  • Manual / Bulk CSV-Excel-JSON / Wikidata OSINT / AI-LLM ingestion pipelines
  • Three-layer anti-impersonation engine with bilingual bilingual payloads
"""

import os
import io
import json
import base64
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import httpx
import pandas as pd
import requests
from fastapi import (
    Depends, FastAPI, File, HTTPException, Query,
    UploadFile, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# GitHub DB layer
GITHUB_TOKEN  = os.environ.get("GITHUB_TOKEN",  "")
GITHUB_OWNER  = os.environ.get("GITHUB_OWNER",  "")
REPO_NAME     = os.environ.get("REPO_NAME",     "")
FILE_PATH     = os.environ.get("FILE_PATH",     "database.json")
LOCAL_DB_PATH = "database.json"
USE_GITHUB    = all([GITHUB_TOKEN, GITHUB_OWNER, REPO_NAME])

# Auth
ADMIN_USERNAME      = os.environ.get("ADMIN_USERNAME",      "admin")
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", "")
JWT_SECRET_KEY      = os.environ.get(
    "JWT_SECRET_KEY",
    "CHANGE_ME_IN_PRODUCTION_qraksha_sdm_2025_secret",
)
JWT_ALGORITHM    = "HS256"
JWT_EXPIRE_HOURS = 12

# AI parser
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL   = "claude-sonnet-4-6"

# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="QRaksha SDM — Identity Intelligence API v2",
    description=(
        "Secure, multi-source identity registry and anti-impersonation engine "
        "for the QRaksha / SDM platform."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context    = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme  = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

class IdentityRecord(BaseModel):
    id: str = Field(..., description="Unique lowercase slug, e.g. 'pmo-india'")
    official_name: str
    official_name_hi: Optional[str] = None
    entity_type: str       # Individual | Institution
    category: str          # Government | Celebrity | Brand | Media | Finance | Education | NGO | Other
    official_website: Optional[str] = None
    official_x_handle: Optional[str] = None
    official_instagram_handle: Optional[str] = None
    verified_status: str = "Pending"       # Pending | Verified | Rejected
    confidence_score: int = Field(default=0, ge=0, le=100)
    source_urls: List[str] = []
    discovered_sources: List[str] = []    # Manual | Wikidata | AI_Parser | CSV_Bulk
    added_at: Optional[str] = None

    @field_validator("entity_type")
    @classmethod
    def _check_entity_type(cls, v: str) -> str:
        if v not in {"Individual", "Institution"}:
            raise ValueError("entity_type must be 'Individual' or 'Institution'")
        return v

    @field_validator("verified_status")
    @classmethod
    def _check_status(cls, v: str) -> str:
        if v not in {"Pending", "Verified", "Rejected"}:
            raise ValueError("verified_status must be 'Pending', 'Verified', or 'Rejected'")
        return v


class VerifyRequest(BaseModel):
    action: str                          # Verified | Rejected
    confidence_score: Optional[int] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRE_HOURS * 3600


class WikidataFetchRequest(BaseModel):
    query: str
    lang: str = "en"


class UnstructuredParseRequest(BaseModel):
    text: str
    hint: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# JWT AUTHENTICATION LAYER
# ─────────────────────────────────────────────────────────────────────────────

def _create_token(data: dict) -> str:
    payload = {
        **data,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is invalid or has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_admin(token: str = Depends(oauth2_scheme)) -> dict:
    """FastAPI dependency — verifies JWT and asserts admin role."""
    payload = _decode_token(token)
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required.")
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# GITHUB REST API PERSISTENCE LAYER
# ─────────────────────────────────────────────────────────────────────────────

def _gh_headers() -> dict:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _gh_url() -> str:
    return f"https://api.github.com/repos/{GITHUB_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"


def load_database() -> List[dict]:
    """Load records from GitHub repo; fall back to local file if unconfigured."""
    if USE_GITHUB:
        try:
            resp = requests.get(_gh_url(), headers=_gh_headers(), timeout=10)
            resp.raise_for_status()
            raw = base64.b64decode(resp.json()["content"]).decode("utf-8")
            return json.loads(raw)
        except Exception as exc:
            print(f"[GitHub load error] {exc} — using local fallback.")

    if os.path.exists(LOCAL_DB_PATH):
        try:
            with open(LOCAL_DB_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            print(f"[Local DB error] {exc}")
    return []


def save_database(records: List[dict], commit_msg: str = "Update identity database") -> bool:
    """Persist records to GitHub repo; fall back to local file."""
    payload_str = json.dumps(records, ensure_ascii=False, indent=2)

    if USE_GITHUB:
        try:
            get_resp = requests.get(_gh_url(), headers=_gh_headers(), timeout=10)
            sha = get_resp.json().get("sha", "") if get_resp.status_code == 200 else ""
            body: dict = {
                "message": commit_msg,
                "content": base64.b64encode(payload_str.encode()).decode(),
                "branch": "main",
            }
            if sha:
                body["sha"] = sha
            put = requests.put(_gh_url(), headers=_gh_headers(), json=body, timeout=20)
            put.raise_for_status()
            return True
        except Exception as exc:
            print(f"[GitHub save error] {exc} — writing locally.")

    try:
        with open(LOCAL_DB_PATH, "w", encoding="utf-8") as fh:
            fh.write(payload_str)
        return True
    except Exception as exc:
        print(f"[Local save error] {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ANTI-IMPERSONATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

# One-way canonical map — digit/symbol → letter equivalent.
# Applied to BOTH query AND token so spoofed handles share fingerprints.
CANONICAL_MAP: List[tuple] = [
    ("0",  "o"),  ("1",  "l"),  ("3",  "e"),  ("4",  "a"),
    ("5",  "s"),  ("6",  "g"),  ("7",  "t"),  ("8",  "b"),
    ("9",  "q"),  ("@",  "a"),  ("$",  "s"),
    ("rn", "m"),  ("vv", "w"),
]


def _normalise(text: str) -> str:
    """Strip URL prefixes, @, separators, and diacritics; lowercase."""
    text = text.lower().strip()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"www\.", "", text)
    text = re.sub(r"\.(com|org|in|gov|net|io|co)(/.*)?$", "", text)
    text = text.lstrip("@")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[_\-\.\s]+", "", text)
    return text


def _canonicalize(text: str) -> str:
    """Apply one-way lookalike substitution map to produce canonical fingerprint."""
    result = text
    for fake, real in CANONICAL_MAP:
        result = result.replace(fake, real)
    return result


def _extract_tokens(record: dict) -> tuple[List[str], List[str]]:
    """Return (exact_tokens, canonical_tokens) for all identity fields of a record."""
    exact, canonical = [], []
    for raw in [
        record.get("official_name", ""),
        record.get("official_name_hi", ""),
        record.get("official_x_handle", ""),
        record.get("official_instagram_handle", ""),
        record.get("official_website", ""),
    ]:
        if raw:
            norm = _normalise(str(raw))
            if norm:
                exact.append(norm)
                canonical.append(_canonicalize(norm))
    return exact, canonical


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def _is_typo_variant(q: str, token: str) -> bool:
    """True when query is ≥75% similar to token but not identical (Levenshtein)."""
    if not token or not q:
        return False
    max_len = max(len(q), len(token))
    if max_len == 0:
        return False
    return 0.75 <= (1 - _levenshtein(q, token) / max_len) < 1.0


def _is_substring_spoof(query: str, token: str) -> bool:
    """True when a short verified token is injected inside a longer fake handle."""
    if not token or len(token) < 3:
        return False
    return token in query and query != token


def run_impersonation_check(query: str, records: List[dict]) -> dict:
    """
    Three-layer impersonation matching engine.

    Rule A  → exact plain match on any verified token          → Risk 0  SAFE
    Rule B1 → canonical fingerprint collision (digit/symbol)   → Risk 85 HIGH_RISK
    Rule B2 → Levenshtein typo variant (≥75% similarity)      → Risk 85 HIGH_RISK
    Rule B3 → substring injection (raw or canonical)           → Risk 85 HIGH_RISK
    Default → no match found                                   → Risk 40 UNVERIFIED
    """
    verified   = [r for r in records if r.get("verified_status") == "Verified"]
    q_norm     = _normalise(query)
    q_can      = _canonicalize(q_norm)
    matched    = None
    risk_type  = None

    for record in verified:
        exact_toks, can_toks = _extract_tokens(record)

        # ── Rule A ────────────────────────────────────────────────────────────
        if q_norm in exact_toks:
            name_hi = record.get("official_name_hi") or record.get("official_name")
            return {
                "risk_score": 0,
                "risk_level": "SAFE",
                "matched_entity": record.get("official_name"),
                "entity_details": {
                    "id":               record.get("id"),
                    "category":         record.get("category"),
                    "confidence_score": record.get("confidence_score"),
                    "official_website": record.get("official_website"),
                },
                "message_en": (
                    f"✅ VERIFIED IDENTITY: '{record.get('official_name')}' is a confirmed, "
                    f"officially registered entity in the QRaksha SDM database. "
                    f"This profile is authentic."
                ),
                "message_hi": (
                    f"✅ सत्यापित पहचान: '{name_hi}' QRaksha SDM डेटाबेस में एक "
                    f"पुष्टि की गई, आधिकारिक रूप से पंजीकृत संस्था है। "
                    f"यह प्रोफ़ाइल प्रामाणिक है।"
                ),
            }

        # ── Rules B1 / B2 / B3 ───────────────────────────────────────────────
        for idx, token in enumerate(exact_toks):
            can_tok = can_toks[idx] if idx < len(can_toks) else ""

            if q_can == can_tok and q_norm != token:
                matched   = record
                risk_type = "character_substitution"
                break

            if _is_typo_variant(q_norm, token):
                matched   = record
                risk_type = "typo_variant"
                break

            if _is_substring_spoof(q_norm, token) or _is_substring_spoof(q_can, can_tok):
                matched   = record
                risk_type = "substring_injection"
                break

        if matched:
            break

    if matched:
        name    = matched.get("official_name")
        name_hi = matched.get("official_name_hi") or name
        return {
            "risk_score":    85,
            "risk_level":    "HIGH_RISK",
            "matched_entity": name,
            "spoof_type":    risk_type,
            "message_en": (
                f"🚨 HIGH RISK — IMPERSONATION DETECTED: The query '{query}' is a "
                f"lookalike / {risk_type.replace('_', ' ')} of the verified entity "
                f"'{name}'. This may be a fraudulent or spoof account. "
                f"Do NOT interact. Report it to the platform immediately."
            ),
            "message_hi": (
                f"🚨 उच्च जोखिम — नकली पहचान का पता चला: क्वेरी '{query}' सत्यापित "
                f"संस्था '{name_hi}' का एक लुकअलाइक या वेरिएंट है "
                f"(पहचान प्रकार: {risk_type.replace('_', ' ')})। "
                f"यह एक धोखाधड़ी/नकली खाता हो सकता है। इस प्रोफ़ाइल से बातचीत न करें। "
                f"इसे तुरंत प्लेटफ़ॉर्म पर रिपोर्ट करें।"
            ),
        }

    return {
        "risk_score":    40,
        "risk_level":    "UNVERIFIED",
        "matched_entity": None,
        "message_en": (
            f"⚠️ UNVERIFIED: '{query}' does not match any entity in the QRaksha SDM "
            f"verified registry. This profile has not been authenticated. "
            f"Exercise caution before trusting this account."
        ),
        "message_hi": (
            f"⚠️ असत्यापित: '{query}' QRaksha SDM सत्यापित रजिस्ट्री में किसी भी "
            f"संस्था से मेल नहीं खाती। इस प्रोफ़ाइल को प्रमाणित नहीं किया गया है। "
            f"इस खाते पर भरोसा करने से पहले सावधानी बरतें।"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORE CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

def calculate_confidence(record: dict) -> int:
    score = 0
    if record.get("official_website"):           score += 20
    if record.get("official_x_handle"):          score += 15
    if record.get("official_instagram_handle"):  score += 10
    if record.get("official_name_hi"):           score += 5
    score += min(len(record.get("source_urls", [])) * 10, 30)
    cat = record.get("category", "")
    if cat == "Government":                      score += 20
    elif cat in ("Celebrity", "Brand", "Media"): score += 10
    return min(score, 100)


# ─────────────────────────────────────────────────────────────────────────────
# BULK ROW NORMALISER
# ─────────────────────────────────────────────────────────────────────────────

def _pick(row: dict, *keys: str) -> Optional[str]:
    """Case-insensitive, underscore/space-tolerant column picker."""
    for key in keys:
        for col, val in row.items():
            if col.strip().lower().replace(" ", "_") == key.lower():
                if val is not None and str(val).strip() not in ("", "nan", "None"):
                    return str(val).strip()
    return None


def normalise_bulk_row(row: dict, source_tag: str = "CSV_Bulk") -> dict:
    """Map flexible CSV/Excel column names to the IdentityRecord schema."""
    name = _pick(row, "official_name", "name", "title") or "Unknown Entity"
    slug = _pick(row, "id", "slug", "identifier") or re.sub(
        r"[^a-z0-9\-]", "", name.lower().replace(" ", "-")
    )[:48]

    raw_sources = _pick(row, "source_urls", "sources", "source_url", "urls") or ""
    sources = [s.strip() for s in re.split(r"[\n|,;]", raw_sources) if s.strip()]

    return {
        "id":                        slug,
        "official_name":             name,
        "official_name_hi":          _pick(row, "official_name_hi", "hindi_name", "name_hi"),
        "entity_type":               _pick(row, "entity_type", "type") or "Institution",
        "category":                  _pick(row, "category", "cat") or "Other",
        "official_website":          _pick(row, "official_website", "website", "url", "web"),
        "official_x_handle":         _pick(row, "official_x_handle", "x_handle", "twitter", "twitter_handle"),
        "official_instagram_handle": _pick(row, "official_instagram_handle", "instagram", "ig_handle"),
        "verified_status":           "Pending",
        "confidence_score":          0,
        "source_urls":               sources,
        "discovered_sources":        [source_tag],
        "added_at":                  datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# WIKIDATA OSINT CONNECTOR
# ─────────────────────────────────────────────────────────────────────────────

_WD_SEARCH_URL = "https://www.wikidata.org/w/api.php"
_WD_SPARQL_URL = "https://query.wikidata.org/sparql"
_WD_AGENT      = "QRaksha-SDM/2.0 (identity-intelligence-platform; contact@qraksha.in)"


def _wikidata_search(query: str, lang: str) -> Optional[dict]:
    """Find the best-matching Wikidata item for a name query."""
    params = {
        "action":   "wbsearchentities",
        "format":   "json",
        "language": lang,
        "search":   query,
        "limit":    5,
        "type":     "item",
    }
    resp = requests.get(
        _WD_SEARCH_URL,
        params=params,
        headers={"User-Agent": _WD_AGENT},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("search", [])
    return results[0] if results else None


def _wikidata_sparql(qid: str) -> dict:
    """Fetch official URL, X handle, and Instagram handle for a Wikidata QID."""
    sparql = f"""
    SELECT ?website ?twitter ?instagram ?fbPage WHERE {{
      OPTIONAL {{ wd:{qid} wdt:P856 ?website. }}
      OPTIONAL {{ wd:{qid} wdt:P2002 ?twitter. }}
      OPTIONAL {{ wd:{qid} wdt:P2003 ?instagram. }}
      OPTIONAL {{ wd:{qid} wdt:P2013 ?fbPage. }}
    }} LIMIT 10
    """
    resp = requests.get(
        _WD_SPARQL_URL,
        params={"query": sparql, "format": "json"},
        headers={"User-Agent": _WD_AGENT, "Accept": "application/sparql-results+json"},
        timeout=15,
    )
    resp.raise_for_status()
    bindings = resp.json().get("results", {}).get("bindings", [])

    websites, twitters, instagrams = [], [], []
    for b in bindings:
        if b.get("website"):   websites.append(b["website"]["value"])
        if b.get("twitter"):   twitters.append(b["twitter"]["value"])
        if b.get("instagram"): instagrams.append(b["instagram"]["value"])

    return {
        "websites":   list(dict.fromkeys(websites)),
        "twitters":   list(dict.fromkeys(twitters)),
        "instagrams": list(dict.fromkeys(instagrams)),
    }


def fetch_wikidata_entity(query: str, lang: str = "en") -> dict:
    """Full Wikidata lookup: search → SPARQL property pull → merged response."""
    try:
        entity = _wikidata_search(query, lang)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Wikidata search failed: {exc}")

    if not entity:
        return {"found": False, "message": f"No Wikidata entity found for '{query}'."}

    qid       = entity["id"]
    label     = entity.get("label", query)
    desc      = entity.get("description", "")
    wd_url    = f"https://www.wikidata.org/wiki/{qid}"

    try:
        props = _wikidata_sparql(qid)
    except Exception:
        props = {"websites": [], "twitters": [], "instagrams": []}

    slug = re.sub(r"[^a-z0-9\-]", "", label.lower().replace(" ", "-"))[:48]

    source_urls = list(dict.fromkeys([wd_url] + props["websites"]))

    return {
        "found":                     True,
        "wikidata_id":               qid,
        "wikidata_url":              wd_url,
        # Pre-filled schema fields — ready to be staged or edited by admin
        "id":                        slug,
        "official_name":             label,
        "official_name_hi":          None,
        "entity_type":               "Institution",
        "category":                  "Other",
        "official_website":          props["websites"][0] if props["websites"] else None,
        "official_x_handle":         props["twitters"][0]   if props["twitters"]   else None,
        "official_instagram_handle": props["instagrams"][0] if props["instagrams"] else None,
        "source_urls":               source_urls,
        "discovered_sources":        ["Wikidata"],
        "description":               desc,
        "all_websites":              props["websites"],
        "all_x_handles":             props["twitters"],
        "all_instagram_handles":     props["instagrams"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# AI / LLM UNSTRUCTURED DOCUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM_PROMPT = (
    "You are a structured data extraction engine for a cybersecurity identity platform.\n"
    "Your task: read the supplied raw text and extract all identity information into a "
    "valid JSON object with EXACTLY the following keys:\n"
    "  id               (lowercase slug using hyphens, derived from official_name)\n"
    "  official_name    (string, required)\n"
    "  official_name_hi (Devanagari/Hindi name if present, else null)\n"
    "  entity_type      (exactly 'Individual' or 'Institution')\n"
    "  category         (one of: Government, Celebrity, Brand, Media, Finance, Education, NGO, Other)\n"
    "  official_website (full URL or null)\n"
    "  official_x_handle (handle WITHOUT @ prefix, or null)\n"
    "  official_instagram_handle (handle WITHOUT @ prefix, or null)\n"
    "  source_urls      (JSON array of all URLs found in the text — may be [])\n\n"
    "Rules:\n"
    "  - Respond ONLY with the raw JSON object. No markdown fences, no explanation.\n"
    "  - If a field is unknown, use null (not empty string).\n"
    "  - source_urls must be a valid JSON array, never a string."
)


def _regex_fallback_parser(text: str) -> dict:
    """Heuristic extractor used when Anthropic API key is absent."""
    urls    = re.findall(r"https?://[^\s\"'\),>]+", text)
    x_m     = re.search(r"(?:twitter\.com/|x\.com/|@)([A-Za-z0-9_]{1,50})", text)
    ig_m    = re.search(r"(?:instagram\.com/|@)([A-Za-z0-9_.]{1,50})", text)
    web     = next(
        (u for u in urls if not any(s in u for s in ["twitter", "x.com", "instagram", "facebook", "wikidata"])),
        None,
    )
    words   = re.sub(r"[^a-zA-Z\s]", "", text).split()
    name    = " ".join(words[:6]) if words else "Unknown Entity"
    slug    = re.sub(r"[^a-z0-9\-]", "", name.lower().replace(" ", "-"))[:48]

    return {
        "id":                        slug,
        "official_name":             name,
        "official_name_hi":          None,
        "entity_type":               "Institution",
        "category":                  "Other",
        "official_website":          web,
        "official_x_handle":         x_m.group(1) if x_m else None,
        "official_instagram_handle": ig_m.group(1) if ig_m else None,
        "source_urls":               list(dict.fromkeys(urls))[:10],
        "discovered_sources":        ["AI_Parser"],
        "verified_status":           "Pending",
        "confidence_score":          0,
        "added_at":                  datetime.now(timezone.utc).isoformat(),
        "_parser_note": (
            "Anthropic API key not configured — regex fallback was used. "
            "Review all fields carefully before staging."
        ),
    }


async def parse_with_ai(text: str, hint: Optional[str]) -> dict:
    """Async call to Anthropic Claude for identity schema extraction."""
    if not ANTHROPIC_API_KEY:
        return _regex_fallback_parser(text)

    user_message = text[:4500]
    if hint:
        user_message = f"Hint: {hint}\n\nRAW TEXT:\n{user_message}"
    else:
        user_message = f"RAW TEXT:\n{user_message}"

    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            resp = await client.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key":         ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      ANTHROPIC_MODEL,
                    "max_tokens": 900,
                    "system":     _EXTRACTION_SYSTEM_PROMPT,
                    "messages":   [{"role": "user", "content": user_message}],
                },
            )
            resp.raise_for_status()
            blocks   = resp.json().get("content", [])
            raw_json = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            # Strip accidental markdown fences
            raw_json = re.sub(r"```(?:json)?", "", raw_json).strip().strip("`").strip()
            parsed   = json.loads(raw_json)
    except json.JSONDecodeError:
        return _regex_fallback_parser(text)
    except Exception as exc:
        print(f"[AI Parser error] {exc}")
        return _regex_fallback_parser(text)

    # Enforce mandatory fields
    parsed.setdefault("verified_status",   "Pending")
    parsed.setdefault("confidence_score",  0)
    parsed.setdefault("discovered_sources", [])
    parsed.setdefault("added_at", datetime.now(timezone.utc).isoformat())
    if "AI_Parser" not in parsed["discovered_sources"]:
        parsed["discovered_sources"].append("AI_Parser")

    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────────────────────

# ── System status ─────────────────────────────────────────────────────────────
@app.get("/", tags=["System"])
def root():
    return {
        "platform": "QRaksha / SDM v2",
        "status":   "operational",
        "storage":  "github" if USE_GITHUB else "local",
        "ai_parser": "anthropic" if ANTHROPIC_API_KEY else "regex_fallback",
        "docs":     "/docs",
    }


# ── Authentication ─────────────────────────────────────────────────────────────
@app.post("/api/auth/token", response_model=TokenResponse, tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Exchange admin credentials for a 12-hour JWT.
    Public endpoint — no prior auth required.
    """
    if form_data.username != ADMIN_USERNAME:
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    if ADMIN_PASSWORD_HASH:
        if not pwd_context.verify(form_data.password, ADMIN_PASSWORD_HASH):
            raise HTTPException(status_code=401, detail="Invalid credentials.")
    else:
        # Dev-only fallback: password must equal literal "admin"
        # Set ADMIN_PASSWORD_HASH in production — NEVER leave this unset.
        if form_data.password != "admin":
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials. Configure ADMIN_PASSWORD_HASH env var in production.",
            )

    token = _create_token({"sub": form_data.username, "role": "admin"})
    return TokenResponse(access_token=token)


# ── Identity CRUD ──────────────────────────────────────────────────────────────
@app.get("/api/identities", response_model=List[IdentityRecord], tags=["Identities"])
def get_identities(_: dict = Depends(require_admin)):
    """Admin: fetch the full identity registry."""
    return load_database()


@app.post("/api/identities", response_model=IdentityRecord, status_code=201, tags=["Identities"])
def create_identity(record: IdentityRecord):
    """
    Public: stage a new identity submission for admin review.
    All submissions enter as 'Pending' with confidence_score = 0.
    """
    records = load_database()
    if any(r["id"] == record.id for r in records):
        raise HTTPException(status_code=409, detail=f"Identity '{record.id}' already exists.")

    new = record.model_dump()
    new["verified_status"]  = "Pending"
    new["confidence_score"] = 0
    new["added_at"]         = datetime.now(timezone.utc).isoformat()
    if "Manual" not in new.get("discovered_sources", []):
        new.setdefault("discovered_sources", []).append("Manual")

    records.append(new)
    if not save_database(records, commit_msg=f"Stage: {record.id}"):
        raise HTTPException(status_code=500, detail="Database write failed.")
    return new


@app.put(
    "/api/identities/{entity_id}/verify",
    response_model=IdentityRecord,
    tags=["Identities"],
)
def verify_identity(
    entity_id: str,
    payload: VerifyRequest,
    _: dict = Depends(require_admin),
):
    """Admin: approve or reject a staged identity. Triggers GitHub commit."""
    records = load_database()
    idx = next((i for i, r in enumerate(records) if r["id"] == entity_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Identity '{entity_id}' not found.")
    if payload.action not in ("Verified", "Rejected"):
        raise HTTPException(status_code=400, detail="action must be 'Verified' or 'Rejected'.")

    record                   = records[idx]
    record["verified_status"] = payload.action
    record["confidence_score"] = (
        min(100, max(0, payload.confidence_score))
        if payload.confidence_score is not None
        else (calculate_confidence(record) if payload.action == "Verified" else 0)
    )
    records[idx] = record

    if not save_database(records, commit_msg=f"Admin {payload.action}: {entity_id}"):
        raise HTTPException(status_code=500, detail="Database write failed.")
    return record


@app.delete("/api/identities/{entity_id}", status_code=204, tags=["Identities"])
def delete_identity(entity_id: str, _: dict = Depends(require_admin)):
    """Admin: permanently remove a record."""
    records     = load_database()
    new_records = [r for r in records if r["id"] != entity_id]
    if len(new_records) == len(records):
        raise HTTPException(status_code=404, detail=f"Identity '{entity_id}' not found.")
    if not save_database(new_records, commit_msg=f"Delete: {entity_id}"):
        raise HTTPException(status_code=500, detail="Database write failed.")


# ── Bulk Ingest ────────────────────────────────────────────────────────────────
@app.post("/api/identities/bulk", tags=["Ingestion"])
async def bulk_ingest(
    file: UploadFile = File(...),
    _: dict = Depends(require_admin),
):
    """
    Admin: upload a CSV, XLSX, or JSON file to batch-stage records.
    Supports flexible column naming (see normalize_bulk_row).
    """
    filename = (file.filename or "").lower()
    content  = await file.read()

    try:
        if filename.endswith(".json"):
            raw = json.loads(content.decode("utf-8"))
            rows = raw if isinstance(raw, list) else [raw]
            tag  = "CSV_Bulk"
        elif filename.endswith(".csv"):
            df   = pd.read_csv(io.BytesIO(content))
            rows = df.to_dict(orient="records")
            tag  = "CSV_Bulk"
        elif filename.endswith((".xlsx", ".xls")):
            df   = pd.read_excel(io.BytesIO(content))
            rows = df.to_dict(orient="records")
            tag  = "CSV_Bulk"
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Accepted: .csv, .xlsx, .xls, .json",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"File parse error: {exc}")

    records    = load_database()
    exists_ids = {r["id"] for r in records}
    inserted, skipped = [], []

    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            norm = normalise_bulk_row(row, tag)
            if norm["id"] in exists_ids:
                skipped.append({"id": norm["id"], "reason": "duplicate"})
                continue
            records.append(norm)
            exists_ids.add(norm["id"])
            inserted.append(norm["id"])
        except Exception as exc:
            skipped.append({"id": "?", "reason": str(exc)})

    if inserted:
        save_database(records, commit_msg=f"Bulk ingest: {len(inserted)} records added")

    return {
        "status":       "complete",
        "inserted":     len(inserted),
        "skipped":      len(skipped),
        "inserted_ids": inserted,
        "skipped_detail": skipped,
    }


# ── Wikidata OSINT Connector ──────────────────────────────────────────────────
@app.post("/api/identities/fetch-wikidata", tags=["Ingestion"])
def fetch_wikidata(body: WikidataFetchRequest):
    """
    Public: query Wikidata by entity name.
    Returns pre-filled IdentityRecord fields ready to preview and stage.
    """
    return fetch_wikidata_entity(body.query.strip(), body.lang)


# ── AI Unstructured Document Parser ──────────────────────────────────────────
@app.post("/api/identities/parse-unstructured", tags=["Ingestion"])
async def parse_unstructured(body: UnstructuredParseRequest):
    """
    Public: paste raw text — official notifications, social posts, news transcripts.
    Returns structured IdentityRecord fields extracted by Claude AI (or regex fallback).
    """
    if not body.text or len(body.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Supplied text is too short to parse.")
    return await parse_with_ai(body.text.strip(), body.hint)


# ── Anti-Impersonation Scanner ────────────────────────────────────────────────
@app.get("/api/check", tags=["Scanner"])
def check_identity(query: str = Query(..., min_length=1, description="Handle, URL, or name to verify")):
    """Public: anti-impersonation scan against the verified registry."""
    return run_impersonation_check(query.strip(), load_database())
