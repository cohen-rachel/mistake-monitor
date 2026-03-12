"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.config import settings
from app.database import engine, Base, async_session_factory
from app.seed import seed_mistake_types
from app.api import sessions, transcribe, analyze, insights, topics, rewrite, language_profiles

# logging.basicConfig(level=logging.DEBUG)

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO, # This sets the minimum level for all handlers
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"), 
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger(__name__)


def _cors_origins() -> list[str]:
    return [
        origin.strip()
        for origin in settings.cors_allow_origins.split(",")
        if origin.strip()
    ]


async def _ensure_sqlite_schema_compatibility() -> None:
    """Apply minimal SQLite schema upgrades for existing local DBs.

    SQLite `create_all` does not alter existing tables, so older dev DBs can
    miss newly added columns.
    """
    if not engine.url.drivername.startswith("sqlite"):
        return

    async with engine.begin() as conn:
        result = await conn.execute(text("PRAGMA table_info(users)"))
        columns = {row[1] for row in result.fetchall()}
        if "current_language_profile_id" not in columns:
            await conn.execute(
                text("ALTER TABLE users ADD COLUMN current_language_profile_id INTEGER")
            )
            logger.info("Applied SQLite compatibility migration for users.current_language_profile_id")

        result = await conn.execute(text("PRAGMA table_info(rewrite_attempts)"))
        rewrite_attempt_columns = {row[1] for row in result.fetchall()}
        if rewrite_attempt_columns and "language_code" not in rewrite_attempt_columns:
            await conn.execute(
                text("ALTER TABLE rewrite_attempts ADD COLUMN language_code VARCHAR(10)")
            )
            await conn.execute(
                text(
                    "UPDATE rewrite_attempts "
                    "SET language_code = COALESCE(language, 'en') "
                    "WHERE language_code IS NULL"
                )
            )
            await conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_rewrite_attempts_language_code "
                    "ON rewrite_attempts (language_code)"
                )
            )
            logger.info("Applied SQLite compatibility migration for rewrite_attempts.language_code")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _ensure_sqlite_schema_compatibility()
    logger.info("Database tables created")

    # Seed mistake types
    async with async_session_factory() as db:
        await seed_mistake_types(db)
    logger.info("Mistake types seeded")

    yield

    # Shutdown
    await engine.dispose()
    logger.info("Database engine disposed")


app = FastAPI(
    title="Language Tutor API",
    description="Personal language tutor — transcribe, analyze, and track language mistakes",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=settings.cors_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(sessions.router)
app.include_router(transcribe.router)
app.include_router(analyze.router)
app.include_router(insights.router)
app.include_router(topics.router)
app.include_router(rewrite.router)
app.include_router(language_profiles.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
