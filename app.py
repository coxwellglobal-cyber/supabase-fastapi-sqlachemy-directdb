import os
import logging
from typing import Any, List, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import create_engine, text, event
from sqlalchemy.exc import SQLAlchemyError

import psycopg2
from psycopg2.extras import RealDictCursor

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Optional monitoring middleware (only if installed + enabled)
try:
    from tigzig_api_monitor import APIMonitorMiddleware
except Exception:
    APIMonitorMiddleware = None

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("db-connector")

# ----------------------------
# Load .env only for local dev (NOT on Render)
# ----------------------------
if os.getenv("RENDER") is None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

# ----------------------------
# Env vars (REQUIRED)
# ----------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
REX_API_KEY = os.getenv("REX_API_KEY")
RATE_LIMIT = os.getenv("RATE_LIMIT", "100/hour")
ENABLE_MONITOR = os.getenv("ENABLE_MONITOR", "true").lower() == "true"

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

if not REX_API_KEY:
    raise ValueError("REX_API_KEY environment variable is required")

logger.info(f"RATE_LIMIT = {RATE_LIMIT}")

# ----------------------------
# Helpers: SQL safety + LIMIT rule
# ----------------------------
AGG_HINTS = (
    " group by ",
    " having ",
    " count(",
    " sum(",
    " min(",
    " max(",
    " avg(",
    " distinct ",
)

DISALLOWED_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "grant", "revoke", "comment", "vacuum", "analyze", "call", "do",
    "copy", "execute", "refresh", "set ", "show ", "transaction",
    "select for update", "for update"
)

def normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().split())

def block_multistatement(sql: str) -> None:
    s = sql.strip()
    # Allow ONE trailing semicolon only
    if ";" in s.rstrip(";"):
        raise HTTPException(status_code=400, detail="Multi-statement SQL is not allowed.")

def allow_only_select(sql: str) -> None:
    s = sql.strip().lower()
    if not s.startswith("select"):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed.")
    # Basic keyword block (defense-in-depth)
    for kw in DISALLOWED_KEYWORDS:
        if kw in s:
            raise HTTPException(status_code=400, detail=f"Disallowed keyword detected: {kw.strip()}")

def is_aggregation(sql: str) -> bool:
    s = f" {sql.lower()} "
    return any(h in s for h in AGG_HINTS)

def enforce_limit_100(sql: str) -> str:
    """
    Rule:
    - If retrieving rows: append LIMIT 100 (if no limit already)
    - If aggregation: do NOT add limit
    """
    s = normalize_sql(sql).rstrip(";")
    s_low = f" {s.lower()} "

    if " limit " in s_low:
        return s  # already has a limit

    if is_aggregation(s):
        return s  # aggregation: never add limit

    # Otherwise: row fetch
    return f"{s} LIMIT 100"

def validate_and_prepare(sqlquery: str) -> str:
    block_multistatement(sqlquery)
    allow_only_select(sqlquery)
    return enforce_limit_100(sqlquery)

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="Supabase SQL Read-Only Connector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )

# Optional monitoring middleware
if ENABLE_MONITOR and APIMonitorMiddleware is not None:
    app.add_middleware(
        APIMonitorMiddleware,
        app_name="SUPABASE_CONNECT_FASTAPI",
        include_prefixes=("/sqlquery_alchemy/", "/sqlquery_direct/"),
    )
else:
    logger.info("APIMonitorMiddleware disabled or not installed.")

# ----------------------------
# SQLAlchemy Engine (read-only)
# ----------------------------
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

@event.listens_for(engine, "connect")
def set_session_readonly(dbapi_connection, connection_record):
    try:
        # psycopg2 connection
        dbapi_connection.set_session(readonly=True, autocommit=False)
    except Exception as e:
        logger.warning(f"Could not set SQLAlchemy DB session readonly: {e}")

# ----------------------------
# Auth helper
# ----------------------------
def require_api_key(api_key: str):
    if api_key != REX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/sqlquery_alchemy/")
@limiter.limit(lambda: RATE_LIMIT)
async def sqlquery_alchemy(sqlquery: str, api_key: str, request: Request) -> List[Dict[str, Any]]:
    require_api_key(api_key)

    prepared_sql = validate_and_prepare(sqlquery)
    logger.info(f"[ALCHEMY] {prepared_sql}")

    try:
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                result = conn.execute(text(prepared_sql))
                rows = result.fetchall()
                cols = result.keys()
                trans.commit()
                return [dict(zip(cols, row)) for row in rows]
            except Exception:
                trans.rollback()
                raise

    except SQLAlchemyError as e:
        logger.exception("SQLAlchemy error")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error (alchemy)")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/sqlquery_direct/")
@limiter.limit(lambda: RATE_LIMIT)
async def sqlquery_direct(sqlquery: str, api_key: str, request: Request) -> List[Dict[str, Any]]:
    require_api_key(api_key)

    prepared_sql = validate_and_prepare(sqlquery)
    logger.info(f"[DIRECT] {prepared_sql}")

    conn = None
    try:
        # Connect directly using DATABASE_URL so sslmode=require is honored from the URL
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        conn.set_session(readonly=True, autocommit=False)

        with conn.cursor() as cur:
            cur.execute(prepared_sql)
            results = cur.fetchall()
            return list(results)

    except psycopg2.Error as e:
        logger.exception("PostgreSQL error")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error (direct)")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        if conn:
            conn.close()
