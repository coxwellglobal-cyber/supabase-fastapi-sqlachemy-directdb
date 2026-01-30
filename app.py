import os
import re
import time
import logging
from typing import Any, List, Dict, Optional

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

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")
if not REX_API_KEY:
    raise ValueError("REX_API_KEY environment variable is required")

logger.info(f"Rate limit: {RATE_LIMIT}")
logger.info(f"DEFAULT_LIMIT={DEFAULT_LIMIT}, MAX_LIMIT={MAX_LIMIT}")
logger.info(f"STATEMENT_TIMEOUT_MS={STATEMENT_TIMEOUT_MS}, IDLE_TX_TIMEOUT_MS={IDLE_TX_TIMEOUT_MS}")

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
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|vacuum|analyze)\b",
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

def add_limit(sql: str, limit: int) -> str:
    """
    Force a LIMIT unless:
    - query is aggregation (optional rule; keep your previous logic)
    - query already has LIMIT
    """
    s = (sql or "").strip().rstrip(";")

    # don't force LIMIT on aggregations (your previous behavior)
    if AGG_HINT.search(s):
        return s

    # already has LIMIT
    if re.search(r"\blimit\b", s, flags=re.IGNORECASE):
        return s

    return f"{s} LIMIT {limit}"

def clamp_limit(user_limit: Optional[int]) -> int:
    if user_limit is None:
        return DEFAULT_LIMIT
    if user_limit < 1:
        return DEFAULT_LIMIT
    if user_limit > MAX_LIMIT:
        return MAX_LIMIT
    return user_limit

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
    if api_key != REX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        with engine.connect() as conn:
            apply_session_safety(conn)
            val = conn.execute(text("SELECT 1")).scalar()
        return {"db": "ok", "result": int(val)}
    except Exception as e:
        logger.exception("DB health check failed")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Main endpoint
# -----------------------------
@app.get("/sqlquery_alchemy/")
@limiter.limit(RATE_LIMIT)
async def sqlquery_alchemy(
    request: Request,
    sqlquery: str,
    api_key: str,
    limit: Optional[int] = Query(None, description="Optional row limit (max capped)"),
) -> Any:
    # Auth
    if api_key != REX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Basic SQL safety
    if not is_select_only(sqlquery):
        raise HTTPException(
            status_code=400,
            detail="Only single-statement SELECT queries are allowed (no pg_sleep, no FOR UPDATE/SHARE).",
        )

    # Limit control
    final_limit = clamp_limit(limit)
    safe_sql = add_limit(sqlquery, limit=final_limit)

    # Logging (safe)
    client_ip = get_remote_address(request)
    start = time.time()

    logger.info(
        f"IP={client_ip} key={mask_key(api_key)} limit={final_limit} sql={safe_sql[:300]}"
    )

    try:
        with engine.connect() as conn:
            apply_session_safety(conn)

            result = conn.execute(text(safe_sql))
            rows = result.fetchall()
            cols = list(result.keys())

        duration_ms = int((time.time() - start) * 1000)
        rowcount = len(rows)

        logger.info(f"OK IP={client_ip} ms={duration_ms} rows={rowcount}")

        data: List[Dict[str, Any]] = [dict(zip(cols, row)) for row in rows]
        return data

    except SQLAlchemyError as e:
        duration_ms = int((time.time() - start) * 1000)
        logger.exception(f"DB ERROR IP={client_ip} ms={duration_ms}")
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        logger.exception(f"UNEXPECTED ERROR IP={client_ip} ms={duration_ms}")
        raise HTTPException(status_code=500, detail="Unexpected error")
