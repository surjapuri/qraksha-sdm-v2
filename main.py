"""
QRaksha / SDM — Official Identity Intelligence Database
FastAPI Backend v2.2 — Termux / Android Compatible

Pydantic v1 (pure Python, zero Rust) — works on:
  • Termux Android (aarch64)
  • Any Python 3.8+ environment
  • Render / Railway / VPS

Key differences from v2.1:
  • pydantic==1.10.13  (no pydantic-core, no Rust build)
  • Uses @validator decorator  (pydantic v1 style)
  • Uses .dict() for serialisation  (pydantic v1 style)
  • All other logic identical to v2.1
"""

import io
import json
import base64
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

# ── Load .env FIRST ───────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import httpx
import requests
from fastapi import (
    Depends, FastAPI, File, HTTPException,
    Query, UploadFile, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext

# Pydantic v1 imports (pure Python — no Rust needed)
from pydantic import BaseModel, Field, validator

# ── Pandas: optional, enables bulk CSV/Excel upload ───────────────────────────
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None                       # type: ignore[assignment]
    PANDAS_AVAILABLE = False
    print("[WARNING] pandas not installed — CSV/Excel bulk upload disabled.")
    print("          Fix: pip install pandas openpyxl")

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def _clean_env(key: str, default: str = "") -> str:
    """Read env var, strip accidental surrounding quotes."""
    val = os.environ.get(key, default)
    return val.strip().strip("\"'")


GITHUB_TOKEN  = _clean_env("GITHUB_TOKEN")
GITHUB_OWNER  = _clean_env("GITHUB_OWNER")
REPO_NAME     = _clean_env("REPO_NAME")
FILE_PATH     = _clean_env("FILE_PATH", "database.json")
LOCAL_DB_PATH = "database.json"

_token_looks_real = (
    GITHUB_TOKEN.startswith("ghp_") or
    GITHUB_TOKEN.startswith("github_pat_") or
    GITHUB_TOKEN.startswith("ghs_")
)
USE_GITHUB = all([GITHUB_TOKEN, GITHUB_OWNER, REPO_NAME]) and _token_looks_real

if GITHUB_TOKEN and not _token_looks_real:
    print("[WARNING] GITHUB_TOKEN looks like a placeholder — using local DB.")

ADMIN_USERNAME      = _clean_env("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = _clean_env("ADMIN_PASSWORD_HASH")
JWT_SECRET_KEY      = _clean_env(
    "JWT_SECRET_KEY",
    "CHANGE_ME_IN_PRODUCTION_qraksha_sdm_key",
)
JWT_ALGORITHM    = "HS256"
JWT_EXPIRE_HOURS = 12

ANTHROPIC_API_KEY = _clean_env("ANTHROPIC_API_KEY")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL   = "claude-sonnet-4-6"

print(f"[Boot] Storage  : {'GitHub (' + GITHUB_OWNER + '/' + REPO_NAME + ')' if USE_GITHUB else 'Local file'}")
print(f"[Boot] Auth     : {'bcrypt hash' if ADMIN_PASSWORD_HASH else 'DEV FALLBACK (password=admin)'}")
print(f"[Boot] AI parser: {'Anthropic' if ANTHROPIC_API_KEY else 'Regex fallback'}")
print(f"[Boot] Pandas   : {'yes' if PANDAS_AVAILABLE else 'no (pip install pandas)'}")

# ─────────────────────────────────────────────────────────────────────────────
# APP BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="QRaksha SDM — Identity Intelligence API v2.2",
    description="Termux/Android-compatible build. Pydantic v1, no Rust required.",
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC v1 DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

class IdentityRecord(BaseModel):
    id: str = Field(..., description="Unique lowercase slug, e.g. 'pmo-india'")
    official_name: str
    official_name_hi: Optional[str] = None
    entity_type: str        # Individual | Institution
    category: str           # Government|Celebrity|Brand|Media|Finance|Education|NGO|Other
    official_website: Optional[str] = None
    official_x_handle: Optional[str] = None
    official_instagram_handle: Optional[str] = None
    verified_status: str = "Pending"   # Pending | Verified | Rejected
    confidence_score: int = Field(default=0, ge=0, le=100)
    source_urls: List[str] = []
    discovered_sources: List[str] = []  # Manual|Wikidata|AI_Parser|CSV_Bulk
    added_at: Optional[str] = None

    # ── Pydantic v1 validators (no @classmethod decorator here) ───────────────
    @validator("entity_type")
    def _chk_entity_type(cls, v: str) -> str:
        if v not in {"Individual", "Institution"}:
            raise ValueError("entity_type must be 'Individual' or 'Institution'")
        return v

    @validator("verified_status")
    def _chk_status(cls, v: str) -> str:
        if v not in {"Pending", "Verified", "Rejected"}:
            raise ValueError("verified_status must be 'Pending', 'Verified', or 'Rejected'")
        return v

    class Config:
        # Pydantic v1 config class
        str_strip_whitespace = True


class VerifyRequest(BaseModel):
    action: str                         # Verified | Rejected
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
# JWT AUTHENTICATION
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
            detail="Token invalid or expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_admin(token: str = Depends(oauth2_scheme)) -> dict:
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
    return (
        f"https://api.github.com/repos/{GITHUB_OWNER}/{REPO_NAME}"
        f"/contents/{FILE_PATH}"
    )


def load_database() -> List[dict]:
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


def save_database(records: List[dict], commit_msg: str = "Update DB") -> bool:
    payload_str = json.dumps(records, ensure_ascii=False, indent=2)

    if USE_GITHUB:
        try:
            get_resp = requests.get(_gh_url(), headers=_gh_headers(), timeout=10)
            sha = get_resp.json().get("sha", "") if get_resp.status_code == 200 else ""
            body: Dict = {
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

CANONICAL_MAP: List[Tuple[str, str]] = [
    ("0", "o"), ("1", "l"), ("3", "e"), ("4", "a"),
    ("5", "s"), ("6", "g"), ("7", "t"), ("8", "b"),
    ("9", "q"), ("@", "a"), ("$", "s"),
    ("rn", "m"), ("vv", "w"),
]


def _normalise(text: str) -> str:
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
    result = text
    for fake, real in CANONICAL_MAP:
        result = result.replace(fake, real)
    return result


def _extract_tokens(record: dict) -> Tuple[List[str], List[str]]:
    exact: List[str] = []
    canonical: List[str] = []
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
    if not token or not q:
        return False
    max_len = max(len(q), len(token))
    return max_len > 0 and 0.75 <= (1 - _levenshtein(q, token) / max_len) < 1.0


def _is_substring_spoof(query: str, token: str) -> bool:
    if not token or len(token) < 3:
        return False
    return token in query and query != token


def run_impersonation_check(query: str, records: List[dict]) -> dict:
    """
    Rule A  → exact match              → Risk 0   SAFE
    Rule B1 → canonical fingerprint    → Risk 85  HIGH_RISK
    Rule B2 → Levenshtein ≥75%        → Risk 85  HIGH_RISK
    Rule B3 → substring injection      → Risk 85  HIGH_RISK
    Default → no match                 → Risk 40  UNVERIFIED
    """
    verified  = [r for r in records if r.get("verified_status") == "Verified"]
    q_norm    = _normalise(query)
    q_can     = _canonicalize(q_norm)
    matched   = None
    risk_type = None

    for record in verified:
        exact_toks, can_toks = _extract_tokens(record)

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
                    "officially registered entity in the QRaksha SDM database. "
                    "This profile is authentic."
                ),
                "message_hi": (
                    f"✅ सत्यापित पहचान: '{name_hi}' QRaksha SDM डेटाबेस में एक "
                    "पुष्टि की गई, आधिकारिक रूप से पंजीकृत संस्था है। "
                    "यह प्रोफ़ाइल प्रामाणिक है।"
                ),
            }

        for idx, token in enumerate(exact_toks):
            can_tok = can_toks[idx] if idx < len(can_toks) else ""
            if q_can == can_tok and q_norm != token:
                matched = record; risk_type = "character_substitution"; break
            if _is_typo_variant(q_norm, token):
                matched = record; risk_type = "typo_variant"; break
            if _is_substring_spoof(q_norm, token) or _is_substring_spoof(q_can, can_tok):
                matched = record; risk_type = "substring_injection"; break
        if matched:
            break

    if matched:
        name    = matched.get("official_name")
        name_hi = matched.get("official_name_hi") or name
        return {
            "risk_score":     85,
            "risk_level":     "HIGH_RISK",
            "matched_entity": name,
            "spoof_type":     risk_type,
            "message_en": (
                f"🚨 HIGH RISK — IMPERSONATION DETECTED: '{query}' is a "
                f"lookalike / {str(risk_type).replace('_', ' ')} of verified entity "
                f"'{name}'. This may be a fraudulent account. "
                "Do NOT interact. Report to the platform immediately."
            ),
            "message_hi": (
                f"🚨 उच्च जोखिम — नकली पहचान: '{query}' सत्यापित संस्था "
                f"'{name_hi}' का लुकअलाइक है "
                f"(प्रकार: {str(risk_type).replace('_', ' ')})। "
                "धोखाधड़ी हो सकती है — इस प्रोफ़ाइल से बातचीत न करें।"
            ),
        }

    return {
        "risk_score":     40,
        "risk_level":     "UNVERIFIED",
        "matched_entity": None,
        "message_en": (
            f"⚠️ UNVERIFIED: '{query}' does not match any entity in the "
            "QRaksha SDM verified registry. Exercise caution."
        ),
        "message_hi": (
            f"⚠️ असत्यापित: '{query}' QRaksha SDM रजिस्ट्री में नहीं मिला। "
            "इस खाते पर भरोसा करने से पहले सावधानी बरतें।"
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
    if cat == "Government":                       score += 20
    elif cat in ("Celebrity", "Brand", "Media"):  score += 10
    return min(score, 100)


# ─────────────────────────────────────────────────────────────────────────────
# BULK ROW NORMALISER
# ─────────────────────────────────────────────────────────────────────────────

def _pick(row: dict, *keys: str) -> Optional[str]:
    for key in keys:
        for col, val in row.items():
            if col.strip().lower().replace(" ", "_") == key.lower():
                if val is not None and str(val).strip() not in ("", "nan", "None"):
                    return str(val).strip()
    return None


def normalise_bulk_row(row: dict, source_tag: str = "CSV_Bulk") -> dict:
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
        "official_x_handle":         _pick(row, "official_x_handle", "x_handle", "twitter"),
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

_WD_SEARCH = "https://www.wikidata.org/w/api.php"
_WD_SPARQL = "https://query.wikidata.org/sparql"
_WD_AGENT  = "QRaksha-SDM/2.2 (identity-intelligence; contact@qraksha.in)"


def _wd_search(query: str, lang: str) -> Optional[dict]:
    resp = requests.get(
        _WD_SEARCH,
        params={"action": "wbsearchentities", "format": "json",
                "language": lang, "search": query, "limit": 5, "type": "item"},
        headers={"User-Agent": _WD_AGENT},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("search", [])
    return results[0] if results else None


def _wd_sparql(qid: str) -> dict:
    sparql = (
        f"SELECT ?website ?twitter ?instagram WHERE {{"
        f"  OPTIONAL {{ wd:{qid} wdt:P856 ?website. }}"
        f"  OPTIONAL {{ wd:{qid} wdt:P2002 ?twitter. }}"
        f"  OPTIONAL {{ wd:{qid} wdt:P2003 ?instagram. }}"
        f"}} LIMIT 10"
    )
    resp = requests.get(
        _WD_SPARQL,
        params={"query": sparql, "format": "json"},
        headers={"User-Agent": _WD_AGENT, "Accept": "application/sparql-results+json"},
        timeout=15,
    )
    resp.raise_for_status()
    bindings = resp.json().get("results", {}).get("bindings", [])
    sites, twitters, igrams = [], [], []
    for b in bindings:
        if b.get("website"):   sites.append(b["website"]["value"])
        if b.get("twitter"):   twitters.append(b["twitter"]["value"])
        if b.get("instagram"): igrams.append(b["instagram"]["value"])
    return {
        "websites":   list(dict.fromkeys(sites)),
        "twitters":   list(dict.fromkeys(twitters)),
        "instagrams": list(dict.fromkeys(igrams)),
    }


def fetch_wikidata_entity(query: str, lang: str = "en") -> dict:
    try:
        entity = _wd_search(query, lang)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Wikidata search failed: {exc}")
    if not entity:
        return {"found": False, "message": f"No Wikidata entity found for '{query}'."}
    qid    = entity["id"]
    label  = entity.get("label", query)
    wd_url = f"https://www.wikidata.org/wiki/{qid}"
    try:
        props = _wd_sparql(qid)
    except Exception:
        props = {"websites": [], "twitters": [], "instagrams": []}
    slug = re.sub(r"[^a-z0-9\-]", "", label.lower().replace(" ", "-"))[:48]
    return {
        "found":                     True,
        "wikidata_id":               qid,
        "wikidata_url":              wd_url,
        "id":                        slug,
        "official_name":             label,
        "official_name_hi":          None,
        "entity_type":               "Institution",
        "category":                  "Other",
        "official_website":          props["websites"][0]   if props["websites"]   else None,
        "official_x_handle":         props["twitters"][0]   if props["twitters"]   else None,
        "official_instagram_handle": props["instagrams"][0] if props["instagrams"] else None,
        "source_urls":               list(dict.fromkeys([wd_url] + props["websites"])),
        "discovered_sources":        ["Wikidata"],
        "description":               entity.get("description", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# AI / LLM UNSTRUCTURED DOCUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

_AI_SYSTEM = (
    "You are a data extraction engine. Extract identity fields from the raw text "
    "into a JSON object with EXACTLY these keys: id (lowercase-hyphen slug), "
    "official_name, official_name_hi (or null), "
    "entity_type ('Individual' or 'Institution'), "
    "category (Government/Celebrity/Brand/Media/Finance/Education/NGO/Other), "
    "official_website (URL or null), official_x_handle (without @), "
    "official_instagram_handle (without @), source_urls (array). "
    "Respond ONLY with the raw JSON. No markdown."
)


def _regex_fallback(text: str) -> dict:
    urls  = re.findall(r"https?://[^\s\"'\),>]+", text)
    x_m   = re.search(r"(?:twitter\.com/|x\.com/|@)([A-Za-z0-9_]{1,50})", text)
    ig_m  = re.search(r"(?:instagram\.com/|@)([A-Za-z0-9_.]{1,50})", text)
    web   = next((u for u in urls if not any(s in u for s in
                  ["twitter", "x.com", "instagram", "facebook", "wikidata"])), None)
    words = re.sub(r"[^a-zA-Z\s]", "", text).split()
    name  = " ".join(words[:6]) if words else "Unknown Entity"
    slug  = re.sub(r"[^a-z0-9\-]", "", name.lower().replace(" ", "-"))[:48]
    return {
        "id": slug, "official_name": name, "official_name_hi": None,
        "entity_type": "Institution", "category": "Other",
        "official_website": web,
        "official_x_handle": x_m.group(1) if x_m else None,
        "official_instagram_handle": ig_m.group(1) if ig_m else None,
        "source_urls": list(dict.fromkeys(urls))[:10],
        "discovered_sources": ["AI_Parser"],
        "verified_status": "Pending", "confidence_score": 0,
        "added_at": datetime.now(timezone.utc).isoformat(),
        "_parser_note": "Anthropic API key not set — regex fallback used. Review all fields.",
    }


async def parse_with_ai(text: str, hint: Optional[str]) -> dict:
    if not ANTHROPIC_API_KEY:
        return _regex_fallback(text)
    user_msg = (
        f"Hint: {hint}\n\nRAW TEXT:\n{text[:4500]}"
        if hint else f"RAW TEXT:\n{text[:4500]}"
    )
    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            resp = await client.post(
                ANTHROPIC_API_URL,
                headers={"x-api-key": ANTHROPIC_API_KEY,
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": ANTHROPIC_MODEL, "max_tokens": 900,
                      "system": _AI_SYSTEM,
                      "messages": [{"role": "user", "content": user_msg}]},
            )
            resp.raise_for_status()
            blocks   = resp.json().get("content", [])
            raw_json = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            raw_json = re.sub(r"```(?:json)?", "", raw_json).strip().strip("`").strip()
            parsed   = json.loads(raw_json)
    except Exception as exc:
        print(f"[AI Parser error] {exc}")
        return _regex_fallback(text)

    parsed.setdefault("verified_status",    "Pending")
    parsed.setdefault("confidence_score",   0)
    parsed.setdefault("discovered_sources", [])
    parsed.setdefault("added_at", datetime.now(timezone.utc).isoformat())
    if "AI_Parser" not in parsed["discovered_sources"]:
        parsed["discovered_sources"].append("AI_Parser")
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
def root():
    return {
        "platform":  "QRaksha / SDM v2.2",
        "status":    "operational",
        "pydantic":  "v1 (Termux-compatible)",
        "storage":   "github" if USE_GITHUB else "local",
        "ai_parser": "anthropic" if ANTHROPIC_API_KEY else "regex_fallback",
        "pandas":    PANDAS_AVAILABLE,
        "docs":      "/docs",
    }


# ── AUTH ──────────────────────────────────────────────────────────────────────
@app.post("/api/auth/token", response_model=TokenResponse, tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login → JWT (12 hours).
    • Hash set  → bcrypt verify
    • Hash empty → DEV MODE only, password must be literal 'admin'
    """
    if form_data.username != ADMIN_USERNAME:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    if ADMIN_PASSWORD_HASH:
        try:
            valid = pwd_context.verify(form_data.password, ADMIN_PASSWORD_HASH)
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="ADMIN_PASSWORD_HASH is malformed. Run generate_hash.py.",
            )
        if not valid:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
    else:
        if form_data.password != "admin":
            raise HTTPException(
                status_code=401,
                detail="DEV MODE: password must be 'admin'. Run generate_hash.py for production.",
            )

    token = _create_token({"sub": form_data.username, "role": "admin"})
    return TokenResponse(access_token=token)


# ── IDENTITIES ────────────────────────────────────────────────────────────────
@app.get("/api/identities", response_model=List[IdentityRecord], tags=["Identities"])
def get_identities(_: dict = Depends(require_admin)):
    return load_database()


@app.post("/api/identities", response_model=IdentityRecord, status_code=201, tags=["Identities"])
def create_identity(record: IdentityRecord):
    records = load_database()
    if any(r["id"] == record.id for r in records):
        raise HTTPException(status_code=409, detail=f"Identity '{record.id}' already exists.")

    # Pydantic v1 serialisation: use .dict()
    new = record.dict()
    new["verified_status"]  = "Pending"
    new["confidence_score"] = 0
    new["added_at"]         = datetime.now(timezone.utc).isoformat()
    new.setdefault("discovered_sources", [])
    if "Manual" not in new["discovered_sources"]:
        new["discovered_sources"].append("Manual")

    records.append(new)
    if not save_database(records, commit_msg=f"Stage: {record.id}"):
        raise HTTPException(status_code=500, detail="Database write failed.")
    return new


@app.put("/api/identities/{entity_id}/verify",
         response_model=IdentityRecord, tags=["Identities"])
def verify_identity(
    entity_id: str,
    payload: VerifyRequest,
    _: dict = Depends(require_admin),
):
    records = load_database()
    idx = next((i for i, r in enumerate(records) if r["id"] == entity_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Identity '{entity_id}' not found.")
    if payload.action not in ("Verified", "Rejected"):
        raise HTTPException(status_code=400, detail="action must be 'Verified' or 'Rejected'.")

    record = records[idx]
    record["verified_status"]  = payload.action
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
    records     = load_database()
    new_records = [r for r in records if r["id"] != entity_id]
    if len(new_records) == len(records):
        raise HTTPException(status_code=404, detail=f"Identity '{entity_id}' not found.")
    if not save_database(new_records, commit_msg=f"Delete: {entity_id}"):
        raise HTTPException(status_code=500, detail="Database write failed.")


# ── BULK INGEST ───────────────────────────────────────────────────────────────
@app.post("/api/identities/bulk", tags=["Ingestion"])
async def bulk_ingest(file: UploadFile = File(...), _: dict = Depends(require_admin)):
    if not PANDAS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="pandas not installed. Run: pip install pandas openpyxl",
        )
    filename = (file.filename or "").lower()
    content  = await file.read()
    try:
        if filename.endswith(".json"):
            raw  = json.loads(content.decode("utf-8"))
            rows = raw if isinstance(raw, list) else [raw]
            tag  = "CSV_Bulk"
        elif filename.endswith(".csv"):
            df   = pd.read_csv(io.BytesIO(content))        # type: ignore[union-attr]
            rows = df.to_dict(orient="records")
            tag  = "CSV_Bulk"
        elif filename.endswith((".xlsx", ".xls")):
            df   = pd.read_excel(io.BytesIO(content))      # type: ignore[union-attr]
            rows = df.to_dict(orient="records")
            tag  = "CSV_Bulk"
        else:
            raise HTTPException(status_code=400, detail="Use .csv, .xlsx, or .json")
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
        save_database(records, commit_msg=f"Bulk: {len(inserted)} records")

    return {"status": "complete", "inserted": len(inserted), "skipped": len(skipped),
            "inserted_ids": inserted, "skipped_detail": skipped}


# ── WIKIDATA ──────────────────────────────────────────────────────────────────
@app.post("/api/identities/fetch-wikidata", tags=["Ingestion"])
def fetch_wikidata(body: WikidataFetchRequest):
    return fetch_wikidata_entity(body.query.strip(), body.lang)


# ── AI PARSER ─────────────────────────────────────────────────────────────────
@app.post("/api/identities/parse-unstructured", tags=["Ingestion"])
async def parse_unstructured(body: UnstructuredParseRequest):
    if not body.text or len(body.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text too short to parse.")
    return await parse_with_ai(body.text.strip(), body.hint)


# ── SCANNER ───────────────────────────────────────────────────────────────────
@app.get("/api/check", tags=["Scanner"])
def check_identity(query: str = Query(..., min_length=1)):
    return run_impersonation_check(query.strip(), load_database())
