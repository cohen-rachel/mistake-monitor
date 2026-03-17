"""Topic practice endpoints: topic selection and historical comparison."""

from pathlib import Path

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import PracticeAttempt, Session, Transcript, Mistake, UserLanguageProfile, SessionLanguageProfile
from app.schemas import TopicItem, TopicListResponse, TopicAttemptItem, TopicHistoryResponse

router = APIRouter(prefix="/api/topics", tags=["topics"])

TOPIC_FILE = Path(__file__).resolve().parents[2] / "data" / "practice_topics.txt"


def _load_topics() -> list[TopicItem]:
    topics: list[TopicItem] = []
    if not TOPIC_FILE.exists():
        return topics

    for raw in TOPIC_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        level, key, prompt = parts
        topics.append(
            TopicItem(
                key=key,
                title=key.replace("_", " ").title(),
                prompt=prompt,
                level=level,
            )
        )
    return topics


async def _estimate_level(db: AsyncSession, user_id: int, language_code: str) -> str:
    """Estimate learner level from recent mistake density for a specific language."""
    result = await db.execute(
        select(Session.id, Transcript.raw_text)
        .join(Transcript, Transcript.session_id == Session.id)
        .join(SessionLanguageProfile, SessionLanguageProfile.session_id == Session.id)
        .join(UserLanguageProfile, UserLanguageProfile.id == SessionLanguageProfile.language_profile_id)
        .where(Session.user_id == user_id, UserLanguageProfile.language_code == language_code)
        .order_by(Session.created_at.desc())
        .limit(8)
    )
    rows = result.all()
    if not rows:
        return "beginner"

    session_ids = [row[0] for row in rows]
    mistake_result = await db.execute(
        select(Mistake.session_id, func.count(Mistake.id))
        .where(Mistake.session_id.in_(session_ids))
        .group_by(Mistake.session_id)
    )
    mistake_map = {sid: cnt for sid, cnt in mistake_result.all()}

    total_words = 0
    total_mistakes = 0
    for sid, raw_text in rows:
        words = len((raw_text or "").split())
        total_words += words
        total_mistakes += mistake_map.get(sid, 0)

    if total_words == 0:
        return "beginner"

    mistakes_per_100_words = (total_mistakes / total_words) * 100
    if mistakes_per_100_words > 12:
        return "beginner"
    if mistakes_per_100_words > 6:
        return "intermediate"
    return "advanced"


@router.get("", response_model=TopicListResponse)
async def list_topics(
    user_id: int = Query(1, ge=1),
    language_code: str = Query(..., description="Language code to filter topics"),
    db: AsyncSession = Depends(get_db),
):
    """Return all configured topic options, prioritizing the estimated learner level."""
    estimated_level = await _estimate_level(db, user_id=user_id, language_code=language_code)
    topics = _load_topics()

    prioritized = [t for t in topics if t.level == estimated_level]
    remaining = [t for t in topics if t.level != estimated_level]

    free_talk = TopicItem(
        key="free_talk",
        title="Free Talk",
        prompt="Speak freely about anything you want with no fixed prompt.",
        level=estimated_level,
    )
    return TopicListResponse(
        estimated_level=estimated_level,
        topics=[free_talk, *prioritized, *remaining],
    )


@router.get("/history", response_model=TopicHistoryResponse)
async def topic_history(
    topic_key: str = Query(...),
    user_id: int = Query(1, ge=1),
    language_code: str = Query(..., description="Language code to filter history"),
    db: AsyncSession = Depends(get_db),
):
    """Return historical attempts for a given topic for month-over-month comparison."""
    result = await db.execute(
        select(PracticeAttempt, Session, Transcript)
        .join(Session, Session.id == PracticeAttempt.session_id)
        .join(Transcript, Transcript.session_id == Session.id)
        .join(SessionLanguageProfile, SessionLanguageProfile.session_id == Session.id)
        .join(UserLanguageProfile, UserLanguageProfile.id == SessionLanguageProfile.language_profile_id)
        .where(
            PracticeAttempt.user_id == user_id,
            UserLanguageProfile.language_code == language_code,
            PracticeAttempt.topic_key == topic_key,
        )
        .order_by(PracticeAttempt.created_at.desc())
        .limit(30)
    )

    attempts: list[TopicAttemptItem] = []
    for attempt, session, transcript in result.all():
        mistake_count_result = await db.execute(
            select(func.count(Mistake.id)).where(Mistake.session_id == session.id)
        )
        mistake_count = mistake_count_result.scalar_one() or 0
        attempts.append(
            TopicAttemptItem(
                id=attempt.id,
                session_id=session.id,
                date=attempt.created_at.strftime("%Y-%m-%d %H:%M"),
                language=attempt.language,
                topic_key=attempt.topic_key,
                topic_text=attempt.topic_text,
                is_free_talk=attempt.is_free_talk,
                estimated_level=attempt.estimated_level,
                mistake_count=mistake_count,
                transcript=transcript.raw_text,
            )
        )

    return TopicHistoryResponse(topic_key=topic_key, attempts=attempts)
