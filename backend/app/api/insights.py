"""Insights endpoint: aggregated mistakes, trends, and recent corrections."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Mistake, MistakeType, Session, Transcript, RewriteAttempt, SessionLanguageProfile, UserLanguageProfile
from app.schemas import (
    InsightsResponse,
    MistakeCountItem,
    TrendPoint,
    RecentMistakeItem,
    ProgressPoint,
)

router = APIRouter(prefix="/api/insights", tags=["insights"])


def _extract_sentence(text: str, start_char: int | None, end_char: int | None) -> str:
    if not text:
        return ""
    if start_char is None or end_char is None:
        return text.strip()

    punct = ".!?。！？\n"
    left = max(0, start_char)
    right = min(len(text), end_char)

    while left > 0 and text[left - 1] not in punct:
        left -= 1
    while right < len(text) and text[right] not in punct:
        right += 1
    if right < len(text):
        right += 1

    return text[left:right].strip()


def _build_corrected_sentence(
    original_sentence: str,
    transcript_span: str | None,
    suggested_correction: str | None,
) -> str:
    if not original_sentence:
        return ""
    if transcript_span and suggested_correction and transcript_span in original_sentence:
        return original_sentence.replace(transcript_span, suggested_correction, 1)
    if suggested_correction:
        return f"{original_sentence} [{suggested_correction}]"
    return original_sentence


@router.get("", response_model=InsightsResponse)
async def get_insights(
    top_k: int = Query(10, ge=1, le=50),
    recent_n: int = Query(20, ge=1, le=100),
    language_profile_id: int = Query(..., description="Language profile ID to filter insights"),
    db: AsyncSession = Depends(get_db),
):
    """Get aggregated mistake insights: top types, trends, and recent mistakes for a given language profile."""

    # 1. Top K mistake types by count
    top_query = (
        select(
            MistakeType.code,
            MistakeType.label,
            func.count(Mistake.id).label("cnt"),
        )
        .join(Mistake, Mistake.mistake_type_id == MistakeType.id)
        .join(Session, Mistake.session_id == Session.id)
        .join(SessionLanguageProfile, SessionLanguageProfile.session_id == Session.id)
        .where(SessionLanguageProfile.language_profile_id == language_profile_id)
        .group_by(MistakeType.code, MistakeType.label)
        .order_by(func.count(Mistake.id).desc())
        .limit(top_k)
    )
    
    top_result = await db.execute(top_query)
    top_rows = top_result.all()
    top_mistakes = []
    for row in top_rows:
        code, label, count = row[0], row[1], row[2]
        latest_query = (
            select(Mistake.explanation_short)
            .join(MistakeType, Mistake.mistake_type_id == MistakeType.id)
            .join(Session, Mistake.session_id == Session.id)
            .join(SessionLanguageProfile, SessionLanguageProfile.session_id == Session.id)
            .where(MistakeType.code == code, SessionLanguageProfile.language_profile_id == language_profile_id)
            .order_by(Mistake.id.desc())
            .limit(1)
        )
        latest_result = await db.execute(latest_query)
        latest_summary = latest_result.scalar_one_or_none()

        description_query = select(MistakeType.description).where(MistakeType.code == code).limit(1)
        description_result = await db.execute(description_query)
        description = description_result.scalar_one_or_none()

        top_mistakes.append(
            MistakeCountItem(
                code=code,
                label=label,
                count=count,
                description=description,
                recent_mistake_summary=latest_summary,
            )
        )

    # 2. Trend data: mistake count per session per type
    trend_query = (
        select(
            Session.id,
            Session.created_at,
            MistakeType.code,
            func.count(Mistake.id).label("cnt"),
        )
        .join(Mistake, Mistake.session_id == Session.id)
        .join(MistakeType, Mistake.mistake_type_id == MistakeType.id)
        .join(SessionLanguageProfile, SessionLanguageProfile.session_id == Session.id)
        .where(SessionLanguageProfile.language_profile_id == language_profile_id)
        .group_by(Session.id, Session.created_at, MistakeType.code)
        .order_by(Session.created_at)
    )
    trend_result = await db.execute(trend_query)
    trends = [
        TrendPoint(
            session_id=row[0],
            date=row[1].strftime("%Y-%m-%d %H:%M") if row[1] else "",
            mistake_type_code=row[2],
            count=row[3],
        )
        for row in trend_result.all()
    ]

    # 3. Recent N individual mistakes with correction info
    recent_query = (
        select(Mistake, MistakeType, Session, Transcript)
        .join(MistakeType, Mistake.mistake_type_id == MistakeType.id)
        .join(Session, Mistake.session_id == Session.id)
        .join(Transcript, Transcript.session_id == Session.id)
        .join(SessionLanguageProfile, SessionLanguageProfile.session_id == Session.id)
        .where(SessionLanguageProfile.language_profile_id == language_profile_id)
        .order_by(Mistake.id.desc())
        .limit(recent_n)
    )
    recent_result = await db.execute(recent_query)
    recent_mistakes = []
    for mistake, mt, session, transcript in recent_result.all():
        original_sentence = _extract_sentence(
            transcript.raw_text or "",
            mistake.start_char,
            mistake.end_char,
        )
        corrected_sentence = _build_corrected_sentence(
            original_sentence,
            mistake.transcript_span,
            mistake.suggested_correction,
        )
        recent_mistakes.append(
            RecentMistakeItem(
                id=mistake.id,
                session_id=mistake.session_id,
                date=session.created_at.strftime("%Y-%m-%d %H:%M") if session.created_at else "",
                mistake_type_code=mt.code,
                mistake_type_label=mt.label,
                transcript_span=mistake.transcript_span,
                suggested_correction=mistake.suggested_correction,
                explanation_short=mistake.explanation_short,
                original_sentence=original_sentence,
                corrected_sentence=corrected_sentence,
            )
        )

    # 4. Session-level progress: error rate over time (mistakes per 100 words)
    session_query = (
        select(Session.id, Session.created_at, Transcript.raw_text)
        .join(Transcript, Transcript.session_id == Session.id)
        .join(SessionLanguageProfile, SessionLanguageProfile.session_id == Session.id)
        .where(SessionLanguageProfile.language_profile_id == language_profile_id)
        .order_by(Session.created_at)
    )
    session_result = await db.execute(session_query)
    session_rows = session_result.all()

    progress: list[ProgressPoint] = []
    for session_id, created_at, raw_text in session_rows:
        word_count = len((raw_text or "").split())
        mistake_count_result = await db.execute(
            select(func.count(Mistake.id)).where(Mistake.session_id == session_id)
        )
        mistake_count = mistake_count_result.scalar_one() or 0
        error_rate = (mistake_count / word_count * 100) if word_count > 0 else 0.0
        progress.append(
            ProgressPoint(
                session_id=session_id,
                date=created_at.strftime("%Y-%m-%d %H:%M") if created_at else "",
                word_count=word_count,
                mistake_count=mistake_count,
                error_rate_per_100_words=round(error_rate, 2),
            )
        )

    # 5. Improvement banners from rewrite training wins.
    improvement_banners: list[str] = []
    profile_result = await db.execute(
        select(UserLanguageProfile.language_code).where(UserLanguageProfile.id == language_profile_id)
    )
    language_code = profile_result.scalar_one_or_none()
    if language_code: #TODO: fix this to use new inputs

        rewrite_query = (
            select(RewriteAttempt)
            .where(RewriteAttempt.language_code == language_code)
            .order_by(RewriteAttempt.created_at.desc())
            .limit(200)
        )
        rewrite_result = await db.execute(rewrite_query)
        rewrite_attempts = rewrite_result.scalars().all()

        # Group by mistake pair and check for "was wrong before, now correct" pattern.
        grouped: dict[tuple[str, str], list[RewriteAttempt]] = {}
        for attempt in rewrite_attempts:
            wrong = (attempt.wrong_span or "").strip()
            corr = (attempt.expected_correction or "").strip()
            if not wrong or not corr:
                continue
            grouped.setdefault((wrong, corr), []).append(attempt)

        for (wrong, corr), attempts in grouped.items():
            ordered = sorted(attempts, key=lambda a: a.created_at)
            had_wrong = any(not a.is_correct for a in ordered[:-1])
            latest = ordered[-1]
            if had_wrong and latest.is_correct:
                improvement_banners.append(
                    f"Improvement win: You previously struggled with '{wrong}', and now corrected it to '{corr}'."
                )
            if len(improvement_banners) >= 3:
                break

    return InsightsResponse(
        top_mistakes=top_mistakes,
        trends=trends,
        recent_mistakes=recent_mistakes,
        progress=progress,
        improvement_banners=improvement_banners,
    )
