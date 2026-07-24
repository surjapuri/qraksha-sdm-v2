"""
QRaksha / SDM — Official Identity Intelligence Database
FastAPI Backend v2.3 — Termux/Ubuntu/Render Compatible
"""

import io
import json
import base64
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Dict

# ── .env manual loader ────────────────────────────────────────────────────────
def _load_dotenv_manual():
    for env_path in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'),
        os.path.join(os.getcwd(), '.env'),
        '.env',
    ]:
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, _, val = line.partition('=')
                    key = key.strip()
                    val = val.strip()
                    if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                        val = val[1:-1]
                    if key and key not in os.environ:
                        os.environ[key] = val
            break

_load_dotenv_manual()

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

import httpx
import requests
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, validator

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

GITHUB_TOKEN  = os.environ.get("GITHUB_TOKEN",  "").strip()
GITHUB_OWNER  = os.environ.get("GITHUB_OWNER",  "").strip()
REPO_NAME     = os.environ.get("REPO_NAME",     "").strip()
FILE_PATH     = os.environ.get("FILE_PATH",     "database.json").strip()
LOCAL_DB_PATH = "database.json"

_token_real = (GITHUB_TOKEN.startswith("ghp_") or
               GITHUB_TOKEN.startswith("github_pat_") or
               GITHUB_TOKEN.startswith("ghs_"))
USE_GITHUB = all([GITHUB_TOKEN, GITHUB_OWNER, REPO_NAME]) and _token_real

ADMIN_USERNAME      = os.environ.get("ADMIN_USERNAME",      "admin").strip()
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", "").strip()
JWT_SECRET_KEY      = os.environ.get("JWT_SECRET_KEY", "qraksha_secret_change_me").strip()
JWT_ALGORITHM       = "HS256"
JWT_EXPIRE_HOURS    = 12

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
SERPAPI_KEY    = os.environ.get("SERPAPI_KEY",    "").strip()

print(f"[Boot] Storage  : {'GitHub' if USE_GITHUB else 'Local file'}")
print(f"[Boot] Auth     : {'bcrypt hash' if ADMIN_PASSWORD_HASH else 'DEV FALLBACK (password=admin)'}")
print(f"[Boot] AI parser: {'Gemini' if GEMINI_API_KEY else 'Regex fallback'}")
print(f"[Boot] Pandas   : {'yes' if PANDAS_AVAILABLE else 'no'}")
print(f"[Boot] Hash len : {len(ADMIN_PASSWORD_HASH)}")

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="QRaksha SDM v2.3", version="2.3.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

class IdentityRecord(BaseModel):
    id: str
    official_name: str
    official_name_hi: Optional[str] = None
    entity_type: str
    category: str
    official_website: Optional[str] = None
    official_x_handle: Optional[str] = None
    official_instagram_handle: Optional[str] = None
    verified_status: str = "Pending"
    confidence_score: int = Field(default=0, ge=0, le=100)
    source_urls: List[str] = []
    discovered_sources: List[str] = []
    added_at: Optional[str] = None

    @validator("entity_type")
    def _chk_type(cls, v):
        if v not in {"Individual", "Institution"}:
            raise ValueError("Must be Individual or Institution")
        return v

    @validator("verified_status")
    def _chk_status(cls, v):
        if v not in {"Pending", "Verified", "Rejected"}:
            raise ValueError("Must be Pending, Verified, or Rejected")
        return v


class VerifyRequest(BaseModel):
    action: str
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
# JWT
# ─────────────────────────────────────────────────────────────────────────────

def _create_token(data: dict) -> str:
    payload = {**data,
               "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
               "iat": datetime.now(timezone.utc)}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Token invalid or expired.",
                           headers={"WWW-Authenticate": "Bearer"})


def require_admin(token: str = Depends(oauth2_scheme)) -> dict:
    payload = _decode_token(token)
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin required.")
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# GITHUB DB
# ─────────────────────────────────────────────────────────────────────────────

def _gh_headers():
    return {"Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}

def _gh_url():
    return f"https://api.github.com/repos/{GITHUB_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"


def load_database() -> List[dict]:
    if USE_GITHUB:
        try:
            r = requests.get(_gh_url(), headers=_gh_headers(), timeout=10)
            r.raise_for_status()
            return json.loads(base64.b64decode(r.json()["content"]).decode())
        except Exception as e:
            print(f"[GitHub load] {e}")

    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOCAL_DB_PATH)
    if not os.path.exists(db_path):
        db_path = LOCAL_DB_PATH
    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Local DB] {e}")
    return []


def save_database(records: List[dict], commit_msg: str = "Update DB") -> bool:
    payload_str = json.dumps(records, ensure_ascii=False, indent=2)
    if USE_GITHUB:
        try:
            gr = requests.get(_gh_url(), headers=_gh_headers(), timeout=10)
            sha = gr.json().get("sha", "") if gr.status_code == 200 else ""
            body: Dict = {"message": commit_msg,
                          "content": base64.b64encode(payload_str.encode()).decode(),
                          "branch": "main"}
            if sha:
                body["sha"] = sha
            pr = requests.put(_gh_url(), headers=_gh_headers(), json=body, timeout=20)
            pr.raise_for_status()
            return True
        except Exception as e:
            print(f"[GitHub save] {e}")

    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOCAL_DB_PATH)
    try:
        with open(db_path, "w", encoding="utf-8") as f:
            f.write(payload_str)
        return True
    except Exception as e:
        print(f"[Local save] {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ANTI-IMPERSONATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

CANONICAL_MAP: List[Tuple[str, str]] = [
    ("0","o"),("1","l"),("3","e"),("4","a"),("5","s"),
    ("6","g"),("7","t"),("8","b"),("9","q"),("@","a"),
    ("$","s"),("rn","m"),("vv","w"),
]

def _normalise(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"https?://","",t)
    t = re.sub(r"www\.","",t)
    t = re.sub(r"\.(com|org|in|gov|net|io|co)(/.*)?$","",t)
    t = t.lstrip("@")
    t = unicodedata.normalize("NFKD",t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[_\-\.\s]+","",t)
    return t

def _canonicalize(t: str) -> str:
    for f,r in CANONICAL_MAP: t = t.replace(f,r)
    return t

def _extract_tokens(rec: dict) -> Tuple[List[str], List[str]]:
    ex, ca = [], []
    for raw in [rec.get("official_name",""), rec.get("official_name_hi",""),
                rec.get("official_x_handle",""), rec.get("official_instagram_handle",""),
                rec.get("official_website","")]:
        if raw:
            n = _normalise(str(raw))
            if n: ex.append(n); ca.append(_canonicalize(n))
    return ex, ca

def _levenshtein(a: str, b: str) -> int:
    if len(a)<len(b): a,b=b,a
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i,ca in enumerate(a):
        curr=[i+1]
        for j,cb in enumerate(b):
            curr.append(min(prev[j+1]+1,curr[j]+1,prev[j]+(ca!=cb)))
        prev=curr
    return prev[-1]

def _typo(q: str, t: str) -> bool:
    if not t or not q: return False
    m = max(len(q),len(t))
    return m>0 and 0.75<=(1-_levenshtein(q,t)/m)<1.0

def _substr(q: str, t: str) -> bool:
    return bool(t) and len(t)>=3 and t in q and q!=t

def run_impersonation_check(query: str, records: List[dict]) -> dict:
    verified = [r for r in records if r.get("verified_status")=="Verified"]
    qn = _normalise(query); qc = _canonicalize(qn)
    matched = None; rtype = None

    for rec in verified:
        ex, ca = _extract_tokens(rec)
        if qn in ex:
            hi = rec.get("official_name_hi") or rec.get("official_name")
            return {"risk_score":0,"risk_level":"SAFE",
                    "matched_entity":rec.get("official_name"),
                    "entity_details":{"id":rec.get("id"),"category":rec.get("category"),
                                      "confidence_score":rec.get("confidence_score"),
                                      "official_website":rec.get("official_website")},
                    "message_en":f"✅ VERIFIED: '{rec.get('official_name')}' is authentic.",
                    "message_hi":f"✅ सत्यापित: '{hi}' प्रामाणिक है।"}
        for i,t in enumerate(ex):
            ct = ca[i] if i<len(ca) else ""
            if qc==ct and qn!=t: matched=rec; rtype="character_substitution"; break
            if _typo(qn,t): matched=rec; rtype="typo_variant"; break
            if _substr(qn,t) or _substr(qc,ct): matched=rec; rtype="substring_injection"; break
        if matched: break

    if matched:
        n=matched.get("official_name"); hi=matched.get("official_name_hi") or n
        return {"risk_score":85,"risk_level":"HIGH_RISK","matched_entity":n,"spoof_type":rtype,
                "message_en":f"🚨 HIGH RISK: '{query}' is a lookalike of '{n}'. Do NOT interact.",
                "message_hi":f"🚨 उच्च जोखिम: '{query}' — '{hi}' का लुकअलाइक है।"}

    return {"risk_score":40,"risk_level":"UNVERIFIED","matched_entity":None,
            "message_en":f"⚠️ UNVERIFIED: '{query}' not in registry.",
            "message_hi":f"⚠️ असत्यापित: '{query}' रजिस्ट्री में नहीं।"}


def calculate_confidence(r: dict) -> int:
    s=0
    if r.get("official_website"): s+=20
    if r.get("official_x_handle"): s+=15
    if r.get("official_instagram_handle"): s+=10
    if r.get("official_name_hi"): s+=5
    s+=min(len(r.get("source_urls",[]))*10,30)
    c=r.get("category","")
    if c=="Government": s+=20
    elif c in ("Celebrity","Brand","Media"): s+=10
    return min(s,100)


def _pick(row: dict, *keys: str) -> Optional[str]:
    for k in keys:
        for col,val in row.items():
            if col.strip().lower().replace(" ","_")==k.lower():
                if val is not None and str(val).strip() not in ("","nan","None"):
                    return str(val).strip()
    return None


def normalise_bulk_row(row: dict, tag: str="CSV_Bulk") -> dict:
    name=_pick(row,"official_name","name","title") or "Unknown"
    slug=_pick(row,"id","slug") or re.sub(r"[^a-z0-9\-]","",name.lower().replace(" ","-"))[:48]
    raw=_pick(row,"source_urls","sources","url") or ""
    src=[s.strip() for s in re.split(r"[\n|,;]",raw) if s.strip()]
    return {"id":slug,"official_name":name,
            "official_name_hi":_pick(row,"official_name_hi","hindi_name"),
            "entity_type":_pick(row,"entity_type","type") or "Institution",
            "category":_pick(row,"category","cat") or "Other",
            "official_website":_pick(row,"official_website","website","url"),
            "official_x_handle":_pick(row,"official_x_handle","twitter"),
            "official_instagram_handle":_pick(row,"official_instagram_handle","instagram"),
            "verified_status":"Pending","confidence_score":0,
            "source_urls":src,"discovered_sources":[tag],
            "added_at":datetime.now(timezone.utc).isoformat()}


def fetch_wikidata_entity(query: str, lang: str="en") -> dict:
    try:
        r=requests.get("https://www.wikidata.org/w/api.php",
            params={"action":"wbsearchentities","format":"json","language":lang,
                    "search":query,"limit":5,"type":"item"},
            headers={"User-Agent":"QRaksha-SDM/2.3"},timeout=10)
        res=r.json().get("search",[])
    except Exception as e:
        raise HTTPException(status_code=502,detail=f"Wikidata error: {e}")
    if not res: return {"found":False,"message":f"No entity for '{query}'."}
    e=res[0]; qid=e["id"]; label=e.get("label",query)
    wd=f"https://www.wikidata.org/wiki/{qid}"
    try:
        sp=(f"SELECT ?w ?t ?i WHERE {{OPTIONAL{{wd:{qid} wdt:P856 ?w.}}"
            f"OPTIONAL{{wd:{qid} wdt:P2002 ?t.}}OPTIONAL{{wd:{qid} wdt:P2003 ?i.}}}} LIMIT 5")
        sr=requests.get("https://query.wikidata.org/sparql",
            params={"query":sp,"format":"json"},
            headers={"User-Agent":"QRaksha-SDM/2.3","Accept":"application/sparql-results+json"},
            timeout=15)
        bs=sr.json().get("results",{}).get("bindings",[])
        ws=[b["w"]["value"] for b in bs if b.get("w")]
        ts=[b["t"]["value"] for b in bs if b.get("t")]
        ig=[b["i"]["value"] for b in bs if b.get("i")]
    except Exception: ws=ts=ig=[]
    slug=re.sub(r"[^a-z0-9\-]","",label.lower().replace(" ","-"))[:48]
    return {"found":True,"wikidata_id":qid,"wikidata_url":wd,"id":slug,
            "official_name":label,"official_name_hi":None,
            "entity_type":"Institution","category":"Other",
            "official_website":ws[0] if ws else None,
            "official_x_handle":ts[0] if ts else None,
            "official_instagram_handle":ig[0] if ig else None,
            "source_urls":list(dict.fromkeys([wd]+ws)),
            "discovered_sources":["Wikidata"],"description":e.get("description","")}


def _regex_fallback(text: str) -> dict:
    urls=re.findall(r"https?://[^\s\"'\),>]+",text)
    xm=re.search(r"(?:twitter\.com/|x\.com/|@)([A-Za-z0-9_]{1,50})",text)
    im=re.search(r"(?:instagram\.com/|@)([A-Za-z0-9_.]{1,50})",text)
    web=next((u for u in urls if not any(s in u for s in ["twitter","x.com","instagram","facebook"])),None)
    words=re.sub(r"[^a-zA-Z\s]","",text).split()
    name=" ".join(words[:6]) if words else "Unknown"
    slug=re.sub(r"[^a-z0-9\-]","",name.lower().replace(" ","-"))[:48]
    return {"id":slug,"official_name":name,"official_name_hi":None,
            "entity_type":"Institution","category":"Other","official_website":web,
            "official_x_handle":xm.group(1) if xm else None,
            "official_instagram_handle":im.group(1) if im else None,
            "source_urls":list(dict.fromkeys(urls))[:10],
            "discovered_sources":["AI_Parser"],"verified_status":"Pending",
            "confidence_score":0,"added_at":datetime.now(timezone.utc).isoformat(),
            "_parser_note":"Regex fallback — review all fields."}


async def parse_with_ai(text: str, hint: Optional[str]) -> dict:
    if not GEMINI_API_KEY: return _regex_fallback(text)
    sys_p=("Extract identity from text into JSON: id,official_name,official_name_hi,"
           "entity_type,category,official_website,official_x_handle,"
           "official_instagram_handle,source_urls. Raw JSON only.")
    msg=f"{'Hint:'+hint+chr(10) if hint else ''}TEXT:\n{text[:4500]}"
    try:
        async with httpx.AsyncClient(timeout=40) as c:
            r=await c.post(ANTHROPIC_API_URL,
                headers={"x-api-key":GEMINI_API_KEY,"anthropic-version":"2023-06-01",
                         "content-type":"application/json"},
                json={"model":ANTHROPIC_MODEL,"max_tokens":900,"system":sys_p,
                      "messages":[{"role":"user","content":msg}]})
            r.raise_for_status()
            raw="".join(b.get("text","") for b in r.json().get("content",[]) if b.get("type")=="text")
            raw=re.sub(r"```(?:json)?","",raw).strip().strip("`")
            parsed=json.loads(raw)
    except Exception as e:
        print(f"[AI] {e}"); return _regex_fallback(text)
    parsed.setdefault("verified_status","Pending")
    parsed.setdefault("confidence_score",0)
    parsed.setdefault("discovered_sources",["AI_Parser"])
    parsed.setdefault("added_at",datetime.now(timezone.utc).isoformat())
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"platform":"QRaksha/SDM v2.3","status":"operational",
            "storage":"github" if USE_GITHUB else "local",
            "ai_parser":"anthropic" if GEMINI_API_KEY else "regex_fallback",
            "pandas":PANDAS_AVAILABLE,"docs":"/docs"}


@app.post("/api/auth/token", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != ADMIN_USERNAME:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    if ADMIN_PASSWORD_HASH:
        try:
            valid = pwd_context.verify(form_data.password, ADMIN_PASSWORD_HASH)
        except Exception as exc:
            print(f"[Auth error] {exc} | hash={repr(ADMIN_PASSWORD_HASH[:30])}")
            raise HTTPException(status_code=500,
                detail=f"Hash error: {exc} | len={len(ADMIN_PASSWORD_HASH)}")
        if not valid:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
    else:
        if form_data.password != "admin":
            raise HTTPException(status_code=401,
                detail="DEV MODE: use password 'admin'.")

    return TokenResponse(access_token=_create_token({"sub":form_data.username,"role":"admin"}))


@app.get("/api/identities", response_model=List[IdentityRecord])
def get_identities(_: dict = Depends(require_admin)):
    return load_database()


@app.post("/api/identities", response_model=IdentityRecord, status_code=201)
def create_identity(record: IdentityRecord):
    records = load_database()
    if any(r["id"]==record.id for r in records):
        raise HTTPException(status_code=409, detail=f"'{record.id}' already exists.")
    new=record.dict()
    new.update({"verified_status":"Pending","confidence_score":0,
                "added_at":datetime.now(timezone.utc).isoformat()})
    new.setdefault("discovered_sources",[])
    if "Manual" not in new["discovered_sources"]:
        new["discovered_sources"].append("Manual")
    records.append(new)
    if not save_database(records, commit_msg=f"Stage:{record.id}"):
        raise HTTPException(status_code=500, detail="DB write failed.")
    return new


@app.put("/api/identities/{entity_id}/verify", response_model=IdentityRecord)
def verify_identity(entity_id: str, payload: VerifyRequest, _: dict=Depends(require_admin)):
    records=load_database()
    idx=next((i for i,r in enumerate(records) if r["id"]==entity_id),None)
    if idx is None: raise HTTPException(status_code=404,detail=f"'{entity_id}' not found.")
    if payload.action not in ("Verified","Rejected"):
        raise HTTPException(status_code=400,detail="action: Verified or Rejected")
    rec=records[idx]
    rec["verified_status"]=payload.action
    rec["confidence_score"]=(min(100,max(0,payload.confidence_score))
        if payload.confidence_score is not None
        else (calculate_confidence(rec) if payload.action=="Verified" else 0))
    records[idx]=rec
    if not save_database(records,commit_msg=f"Admin {payload.action}:{entity_id}"):
        raise HTTPException(status_code=500,detail="DB write failed.")
    return rec


@app.delete("/api/identities/{entity_id}", status_code=204)
def delete_identity(entity_id: str, _: dict=Depends(require_admin)):
    records=load_database()
    new=[r for r in records if r["id"]!=entity_id]
    if len(new)==len(records): raise HTTPException(status_code=404,detail="Not found.")
    if not save_database(new,commit_msg=f"Delete:{entity_id}"):
        raise HTTPException(status_code=500,detail="DB write failed.")


@app.post("/api/identities/bulk")
async def bulk_ingest(file: UploadFile=File(...), _: dict=Depends(require_admin)):
    if not PANDAS_AVAILABLE:
        raise HTTPException(status_code=503,detail="pandas not installed.")
    fn=(file.filename or "").lower(); content=await file.read()
    try:
        if fn.endswith(".json"):
            raw=json.loads(content.decode()); rows=raw if isinstance(raw,list) else [raw]
        elif fn.endswith(".csv"):
            rows=pd.read_csv(io.BytesIO(content)).to_dict(orient="records")
        elif fn.endswith((".xlsx",".xls")):
            rows=pd.read_excel(io.BytesIO(content)).to_dict(orient="records")
        else: raise HTTPException(status_code=400,detail="Use .csv/.xlsx/.json")
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=422,detail=f"Parse error:{e}")
    records=load_database(); exists={r["id"] for r in records}
    ins,skip=[],[]
    for row in rows:
        if not isinstance(row,dict): continue
        try:
            n=normalise_bulk_row(row)
            if n["id"] in exists: skip.append({"id":n["id"],"reason":"duplicate"}); continue
            records.append(n); exists.add(n["id"]); ins.append(n["id"])
        except Exception as e: skip.append({"id":"?","reason":str(e)})
    if ins: save_database(records,commit_msg=f"Bulk:{len(ins)}")
    return {"status":"complete","inserted":len(ins),"skipped":len(skip),"inserted_ids":ins}


@app.post("/api/identities/fetch-wikidata")
def fetch_wikidata(body: WikidataFetchRequest):
    return fetch_wikidata_entity(body.query.strip(), body.lang)


@app.post("/api/identities/parse-unstructured")
async def parse_unstructured(body: UnstructuredParseRequest):
    if not body.text or len(body.text.strip())<10:
        raise HTTPException(status_code=400,detail="Text too short.")
    return await parse_with_ai(body.text.strip(), body.hint)




@app.post("/api/osint/smart-extract")
async def smart_osint_extract(entity_name: str):
    """OSINT: SerpAPI + Gemini AI se entity data extract karo"""
    if not SERPAPI_KEY:
        raise HTTPException(status_code=503, detail="SERPAPI_KEY not configured.")
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not configured.")

    # Step 1: SerpAPI Google Search
    try:
        serp = requests.get(
            "https://serpapi.com/search.json",
            params={"q": f"{entity_name} official instagram twitter website India",
                    "api_key": SERPAPI_KEY, "num": 10},
            timeout=15,
        )
        serp.raise_for_status()
        results = serp.json().get("organic_results", [])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"SerpAPI error: {e}")

    snippets = [{"title": r.get("title"), "link": r.get("link"),
                 "snippet": r.get("snippet")} for r in results[:8]]

    # Step 2: Gemini AI Extraction
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""You are an OSINT analyst for QRaksha DB.
Analyze these search results for "{entity_name}" (Indian public figure).
Extract ONLY official verified accounts. Ignore fan pages and news articles.
Return ONLY valid JSON (no markdown):
{{
  "id": "lowercase-hyphen-slug",
  "official_name": "{entity_name}",
  "official_name_hi": "hindi name in devanagari",
  "entity_type": "Individual or Institution",
  "category": "Government/Celebrity/Brand/Media/Finance/Education/NGO/Other",
  "official_website": "url or null",
  "official_x_handle": "handle without @ or null",
  "official_instagram_handle": "handle without @ or null",
  "source_urls": ["official URLs found"]
}}

SEARCH DATA:
{json.dumps(snippets, ensure_ascii=False)}"""

        resp = model.generate_content(prompt)
        raw = re.sub(r"```(?:json)?", "", resp.text).strip().strip("`")
        extracted = json.loads(raw)
    except Exception as e:
        print(f"[Gemini] {e} — trying Wikidata fallback")
        try:
            wd = fetch_wikidata_entity(entity_name)
            if not wd.get("found"): raise Exception("Wikidata miss")
            extracted = wd
        except Exception as we:
            raise HTTPException(status_code=500, detail=f"Gemini:{e}|Wikidata:{we}")

    # Step 3: Normalize and save
    etype = extracted.get("entity_type", "Individual")
    record = {
        "id": extracted.get("id", re.sub(r"[^a-z0-9\-]", "",
               entity_name.lower().replace(" ", "-"))[:48]),
        "official_name":             extracted.get("official_name", entity_name),
        "official_name_hi":          extracted.get("official_name_hi"),
        "entity_type":               etype if etype in ("Individual","Institution") else "Individual",
        "category":                  extracted.get("category", "Other"),
        "official_website":          extracted.get("official_website"),
        "official_x_handle":         extracted.get("official_x_handle"),
        "official_instagram_handle": extracted.get("official_instagram_handle"),
        "verified_status":           "Pending",
        "confidence_score":          0,
        "source_urls":               extracted.get("source_urls", []),
        "discovered_sources":        ["Google_SerpAPI", "Gemini_AI"],
        "added_at":                  datetime.now(timezone.utc).isoformat(),
    }

    records = load_database()
    idx = next((i for i,r in enumerate(records) if r["id"]==record["id"]), None)
    if idx is not None:
        records[idx] = record; action = "updated"
    else:
        records.append(record); action = "added"
    save_database(records, commit_msg=f"OSINT: {entity_name}")

    return {"status": "success", "action": action,
            "message": f"'{entity_name}' DB mein {action} kar diya.",
            "data": record}



@app.get("/api/suggest")
def suggest_identities(q: str = Query("", min_length=1)):
    if len(q.strip()) < 2: return []
    records = load_database()
    q_low = q.strip().lower()
    seen, out = set(), []
    for r in records:
        if r.get("id","").startswith("__"): continue
        for field, label in [
            (r.get("official_name",""),"name"),
            (r.get("official_x_handle",""),"X"),
            (r.get("official_instagram_handle",""),"Instagram"),
            (r.get("official_name_hi",""),"Hindi"),
        ]:
            if field and q_low in field.lower() and r["id"] not in seen:
                seen.add(r["id"])
                out.append({"id":r.get("id"),"label":r.get("official_name"),
                    "label_hi":r.get("official_name_hi"),"x":r.get("official_x_handle"),
                    "ig":r.get("official_instagram_handle"),"website":r.get("official_website"),
                    "category":r.get("category"),"verified":r.get("verified_status")=="Verified",
                    "match":label})
                break
        if len(out)>=8: break
    return out


@app.post("/api/search-log")
def log_search(query: str, risk_level: str = "UNVERIFIED"):
    try:
        records = load_database()
        LID = "__search_log__"
        log = next((r for r in records if r.get("id")==LID), None)
        if not log:
            log={"id":LID,"official_name":"_Search Log","entity_type":"Institution",
                 "category":"Other","verified_status":"Verified","confidence_score":0,
                 "source_urls":[],"discovered_sources":["System"],"searches":[]}
            records.append(log)
        idx = next(i for i,r in enumerate(records) if r.get("id")==LID)
        log.setdefault("searches",[])
        log["searches"].append({"q":query,"risk":risk_level,"at":datetime.now(timezone.utc).isoformat()})
        log["searches"] = log["searches"][-500:]
        records[idx] = log
        save_database(records, commit_msg=f"Log:{query[:15]}")
        return {"logged":True}
    except Exception as e:
        return {"logged":False,"error":str(e)}

@app.get("/api/check")
def check_identity(query: str=Query(..., min_length=1)):
    return run_impersonation_check(query.strip(), load_database())
