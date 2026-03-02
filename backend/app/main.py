"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import engine, Base, async_session_factory
from app.seed import seed_mistake_types
from app.api import sessions, transcribe, analyze, insights, practice

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(sessions.router)
app.include_router(transcribe.router)
app.include_router(analyze.router)
app.include_router(insights.router)
app.include_router(practice.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}
