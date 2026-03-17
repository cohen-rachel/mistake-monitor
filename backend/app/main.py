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

        result = await conn.execute(text("PRAGMA table_info(mistakes)"))
        mistake_columns = {row[1] for row in result.fetchall()}
        if mistake_columns and "skill_family" not in mistake_columns:
            await conn.execute(
                text("ALTER TABLE mistakes ADD COLUMN skill_family VARCHAR(200)")
            )
            logger.info("Applied SQLite compatibility migration for mistakes.skill_family")
        if mistake_columns and "pattern_label" not in mistake_columns:
            await conn.execute(
                text("ALTER TABLE mistakes ADD COLUMN pattern_label VARCHAR(200)")
            )
            logger.info("Applied SQLite compatibility migration for mistakes.pattern_label")
        if mistake_columns and "canonical_wrong_example" not in mistake_columns:
            await conn.execute(
                text("ALTER TABLE mistakes ADD COLUMN canonical_wrong_example VARCHAR(500)")
            )
            logger.info("Applied SQLite compatibility migration for mistakes.canonical_wrong_example")
        if mistake_columns and "canonical_correct_example" not in mistake_columns:
            await conn.execute(
                text("ALTER TABLE mistakes ADD COLUMN canonical_correct_example VARCHAR(500)")
            )
            logger.info("Applied SQLite compatibility migration for mistakes.canonical_correct_example")

        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS mistake_memories ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "language_profile_id INTEGER NOT NULL, "
                "source_mistake_id INTEGER NOT NULL, "
                "mistake_type_code VARCHAR(50) NOT NULL, "
                "skill_family VARCHAR(200) NOT NULL, "
                "pattern_label VARCHAR(200) NOT NULL, "
                "wrong_form VARCHAR(500), "
                "correct_form VARCHAR(500), "
                "canonical_wrong_example VARCHAR(500), "
                "canonical_correct_example VARCHAR(500), "
                "explanation TEXT, "
                "status VARCHAR(20) NOT NULL DEFAULT 'open', "
                "occurrence_count INTEGER NOT NULL DEFAULT 1, "
                "improvement_count INTEGER NOT NULL DEFAULT 0, "
                "created_at DATETIME NOT NULL, "
                "last_seen_at DATETIME NOT NULL, "
                "FOREIGN KEY(language_profile_id) REFERENCES user_language_profiles(id), "
                "FOREIGN KEY(source_mistake_id) REFERENCES mistakes(id)"
                ")"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_mistake_memories_language_profile_id "
                "ON mistake_memories (language_profile_id)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_mistake_memories_skill_family "
                "ON mistake_memories (skill_family)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_mistake_memories_pattern_label "
                "ON mistake_memories (pattern_label)"
            )
        )

        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS improvement_events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "language_profile_id INTEGER NOT NULL, "
                "session_id INTEGER NOT NULL, "
                "memory_id INTEGER NOT NULL, "
                "event_type VARCHAR(20) NOT NULL, "
                "sentence_text TEXT, "
                "reason TEXT, "
                "confidence FLOAT, "
                "created_at DATETIME NOT NULL, "
                "FOREIGN KEY(language_profile_id) REFERENCES user_language_profiles(id), "
                "FOREIGN KEY(session_id) REFERENCES sessions(id), "
                "FOREIGN KEY(memory_id) REFERENCES mistake_memories(id)"
                ")"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_improvement_events_language_profile_id "
                "ON improvement_events (language_profile_id)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_improvement_events_session_id "
                "ON improvement_events (session_id)"
            )
        )


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
