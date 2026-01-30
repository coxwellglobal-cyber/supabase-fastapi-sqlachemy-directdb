import os
import re
import time
import logging
from typing import Any, List, Dict, Optional, Literal

from fastapi import FastAPI, HTTPException, Request, status, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("supabase-api")

# -----------------------------
# Env vars (Render injects these)
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
REX_API_KEY = os.getenv("REX_API_KEY", "")
RATE_LIMIT = os.getenv("RATE_LIMIT", "100/hour")

# Query safety knobs
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "100"))
MAX_LIMIT = int(os.getenv("MAX_LIMIT", "500"))

# Timeouts (ms)
STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "8000"))  # 8s
IDLE_TX_TIMEOUT_MS = int(os.getenv("IDLE_TX_TIMEOUT_MS", "8000"))      # 8s

# AI query knobs (view name)
PRODUCT_VIEW = os.getenv("PRODUCT_VIEW", "public.v_product_intelligence")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")
if not REX_API_KEY:
    raise ValueError("REX_API_KEY environment variable is required")

logger.info(f"Rate limit: {RATE_LIMIT}")
logger.info(f"DEFAULT_LIMIT={DEFAULT_LIMIT}, MAX_LIMIT={MAX_LIMIT}")
logger.info(f"STATEMENT_TIMEOUT_MS={STATEMENT_TIMEOUT_MS}, IDLE_TX_TIMEOUT_MS={IDLE_TX_TIMEOUT_MS}")
logger.info(f"PRODUCT_VIEW={PRODUCT_VIEW}")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Coxwell Supabase Read-only SQL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Rate limiting
# -----------------------------
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )

# -----------------------------
# DB Engine (pooler-friendly)
# -----------------------------
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
)

# -----------------------------
# Safety: allow ONLY SELECT, single statement, block dangerous keywords
# -----------------------------
DANGEROUS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|vacuum|analyze|copy|do)\b",
    re.IGNORECASE,
)

# SELECT-only but still dangerous
BAD_SELECT = re.compile(
    r"\b(pg_sleep|for\s+update|for\s+share)\b",
    re.IGNORECASE,
)

AGG_HINT = re.compile(
    r"\b(group\s+by|having|count\s*\(|sum\s*\(|avg\s*\(|min\s*\(|max\s*\()\b",
    re.IGNORECASE,
)

def is_select_only(sql: str) -> bool:
    s = (sql or "").strip()

    if not s.lower().startswith("select"):
        return False

    # block multi-statement; allow a single trailing ';'
    if ";" in s[:-1]:
        return False

    if DANGEROUS.search(s):
        return False

    if BAD_SELECT.search(s):
        return False

    return True

def clamp_limit(user_limit: Optional[int]) -> int:
    if user_limit is None:
        return DEFAULT_LIMIT
    if user_limit < 1:
        return DEFAULT_LIMIT
    if user_limit > MAX_LIMIT:
        return MAX_LIMIT
    return user_limit

def add_limit(sql: str, limit: int) -> str:
    """
    Force a LIMIT unless:
    - query is aggregation (keeps your previous logic)
    - query already has LIMIT
    """
    s = (sql or "").strip().rstrip(";")

    # don't force LIMIT on aggregations
    if AGG_HINT.search(s):
        return s

    # already has LIMIT
    if re.search(r"\blimit\b", s, flags=re.IGNORECASE):
        return s

    return f"{s} LIMIT {limit}"

def apply_session_safety(conn) -> None:
    """
    Enforce read-only + timeouts at DB-session level.
    """
    conn.exec_driver_sql("SET default_transaction_read_only = on")
    conn.exec_driver_sql(f"SET statement_timeout = '{STATEMENT_TIMEOUT_MS}'")
    conn.exec_driver_sql(f"SET idle_in_transaction_session_timeout = '{IDLE_TX_TIMEOUT_MS}'")

def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "****"
    return key[:4] + "****" + key[-4:]

def require_api_key(api_key: str) -> None:
    if api_key != REX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# -----------------------------
# Health endpoints
# -----------------------------
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "supabase-sql-api"}

# Protected dbcheck
@app.get("/dbcheck")
@limiter.limit(RATE_LIMIT)
async def dbcheck(
    request: Request,
    api_key: str = Query(..., description="API key required"),
) -> Dict[str, Any]:
    require_api_key(api_key)

    try:
        with engine.connect() as conn:
            apply_session_safety(conn)
            val = conn.execute(text("SELECT 1")).scalar()
        return {"db": "ok", "result": int(val)}
    except Exception as e:
        logger.exception("DB health check failed")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Main SQL endpoint (raw SQL, but SELECT-only)
# -----------------------------
@app.get("/sqlquery_alchemy/")
@limiter.limit(RATE_LIMIT)
async def sqlquery_alchemy(
    request: Request,
    sqlquery: str,
    api_key: str,
    limit: Optional[int] = Query(None, description="Optional row limit (max capped)"),
) -> Any:
    require_api_key(api_key)

    if not is_select_only(sqlquery):
        raise HTTPException(
            status_code=400,
            detail="Only single-statement SELECT queries are allowed (no pg_sleep, no FOR UPDATE/SHARE).",
        )

    final_limit = clamp_limit(limit)
    safe_sql = add_limit(sqlquery, limit=final_limit)

    client_ip = get_remote_address(request)
    start = time.time()

    logger.info(f"SQL IP={client_ip} key={mask_key(api_key)} limit={final_limit} sql={safe_sql[:300]}")

    try:
        with engine.connect() as conn:
            apply_session_safety(conn)

            result = conn.execute(text(safe_sql))
            rows = result.fetchall()
            cols = list(result.keys())

        duration_ms = int((time.time() - start) * 1000)
        logger.info(f"SQL OK IP={client_ip} ms={duration_ms} rows={len(rows)}")

        data: List[Dict[str, Any]] = [dict(zip(cols, row)) for row in rows]
        return data

    except SQLAlchemyError:
        duration_ms = int((time.time() - start) * 1000)
        logger.exception(f"SQL DB ERROR IP={client_ip} ms={duration_ms}")
        raise HTTPException(status_code=500, detail="Database error")
    except Exception:
        duration_ms = int((time.time() - start) * 1000)
        logger.exception(f"SQL UNEXPECTED ERROR IP={client_ip} ms={duration_ms}")
        raise HTTPException(status_code=500, detail="Unexpected error")

# -----------------------------
# AI Query Endpoint (intent -> safe SQL)
# -----------------------------
def _normalize(s: str) -> str:
    return (s or "").strip()

def build_product_query(question: str) -> (str, Dict[str, Any], int):
    """
    Convert a human question into a SAFE parameterized SQL query.
    ONLY queries PRODUCT_VIEW.
    """
    q = _normalize(question).lower()
    params: Dict[str, Any] = {}

    # Defaults
    limit = DEFAULT_LIMIT

    # Detect thickness like "25mm", "14 mm", "6mm"
    m = re.search(r"(\d+)\s*mm", q)
    thickness = int(m.group(1)) if m else None

    # Detect product class keywords
    # You can add more mappings later
    wants_multicell = "multicell" in q
    wants_multiwall = "multiwall" in q or "multiwall" in q
    wants_solid = "solid" in q
    wants_corrugated = "corrugated" in q

    # Detect color/finish keywords (simple contains)
    # These are LIKE filters
    color_like = None
    for c in ["clear", "opal", "bronze", "blue", "green", "smoke", "grey", "gray", "red", "yellow"]:
        if c in q:
            color_like = c
            break

    finish_like = None
    for f in ["uv", "anti-glare", "antiglare", "ir", "softlite", "anti-reflective", "anti reflective"]:
        if f in q:
            finish_like = f
            break

    # Detect “show all” / “list”
    list_mode = any(k in q for k in ["show", "list", "all", "products", "product", "find"])

    # Detect “top N”
    topn = re.search(r"\btop\s+(\d+)\b", q)
    if topn:
        limit = clamp_limit(int(topn.group(1)))

    where = []
    if thickness is not None:
        where.append("thickness = :thickness")
        params["thickness"] = thickness

    if wants_multicell:
        where.append("product_class ILIKE '%multicell%'")
    if wants_multiwall:
        where.append("product_class ILIKE '%multiwall%'")
    if wants_solid:
        where.append("product_class ILIKE '%solid%'")
    if wants_corrugated:
        where.append("product_class ILIKE '%corrugated%'")

    if color_like:
        where.append("color ILIKE :color")
        params["color"] = f"%{color_like}%"

    if finish_like:
        where.append("finish ILIKE :finish")
        params["finish"] = f"%{finish_like}%"

    # If nothing detected, do a general search on name/class/color/finish
    if not where:
        # Use original question as keyword
        keyword = _normalize(question)
        if keyword:
            where.append("(product_name ILIKE :kw OR product_class ILIKE :kw OR color ILIKE :kw OR finish ILIKE :kw)")
            params["kw"] = f"%{keyword}%"
        else:
            # empty question fallback
            where.append("1=1")

    where_sql = " AND ".join(where)

    sql = f"""
        SELECT id, product_name, thickness, product_class, color, finish
        FROM {PRODUCT_VIEW}
        WHERE {where_sql}
        ORDER BY thickness NULLS LAST, product_name
    """.strip()

    # Always apply limit for row queries
    sql = add_limit(sql, limit)

    return sql, params, limit

@app.post("/ai_query")
@limiter.limit(RATE_LIMIT)
async def ai_query(request: Request) -> Any:
    """
    POST body:
    {
      "question": "Show all 25mm multicell products",
      "api_key": "rex-....",
      "limit": 50   (optional)
    }
    """
    body = await request.json()
    question = body.get("question", "")
    api_key = body.get("api_key", "")
    user_limit = body.get("limit", None)

    require_api_key(api_key)

    # Build safe query
    sql, params, auto_limit = build_product_query(question)

    # Optional override limit (still capped)
    if isinstance(user_limit, int):
        final_limit = clamp_limit(user_limit)
        # replace limit by re-adding at end (safe)
        sql = re.sub(r"\blimit\s+\d+\b", "", sql, flags=re.IGNORECASE).strip()
        sql = add_limit(sql, final_limit)

    client_ip = get_remote_address(request)
    start = time.time()
    logger.info(f"AI IP={client_ip} key={mask_key(api_key)} q={question[:200]} sql={sql[:300]} params={params}")

    try:
        with engine.connect() as conn:
            apply_session_safety(conn)
            result = conn.execute(text(sql), params)
            rows = result.fetchall()
            cols = list(result.keys())

        duration_ms = int((time.time() - start) * 1000)
        logger.info(f"AI OK IP={client_ip} ms={duration_ms} rows={len(rows)}")

        data: List[Dict[str, Any]] = [dict(zip(cols, row)) for row in rows]
        return data

    except SQLAlchemyError:
        duration_ms = int((time.time() - start) * 1000)
        logger.exception(f"AI DB ERROR IP={client_ip} ms={duration_ms}")
        raise HTTPException(status_code=500, detail="Database error")
    except Exception:
        duration_ms = int((time.time() - start) * 1000)
        logger.exception(f"AI UNEXPECTED ERROR IP={client_ip} ms={duration_ms}")
        raise HTTPException(status_code=500, detail="Unexpected error")
