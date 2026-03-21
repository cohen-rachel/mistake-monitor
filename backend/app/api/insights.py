"""Insights endpoint: aggregated mistakes, trends, and recent corrections."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import (
    ImprovementEvent,
    Mistake,
    MistakeMemory,
    MistakeType,
    Session,
    Transcript,
    SessionLanguageProfile,
)
from app.schemas import (
    InsightsResponse,
    MistakeCountItem,
    TrendPoint,
    RecentMistakeItem,
    ProgressPoint,
    SpeakingWinItem,
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


_GENERIC_SKILL_FAMILIES = {
    "tense_aspect_usage",
    "subject_verb_agreement",
    "article_usage",
    "preposition_usage",
    "general_grammar",
    "other",
    "verb-tense",
    "preposition",
    "article",
    "word-order",
    "pronunciation",
    "false-friend",
    "pronoun",
    "pluralization",
    "vocabulary",
    "subject-verb-agreement",
    "verb conjugation and tense/aspect",
    "verb usage",
}


def _humanize_focus_label(value: str | None) -> str:
    if not value:
        return "grammar"
    cleaned = value.split(":", 1)[0].strip().strip("_-")
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned or "grammar"


def _is_generic_focus_value(value: str | None) -> bool:
    if not value:
        return True
    normalized = value.strip().lower()
    return normalized in _GENERIC_SKILL_FAMILIES


def _speaking_win_focus_label(memory: MistakeMemory, mistake_type_label: str | None) -> str:
    pattern_label = (memory.pattern_label or "").strip()
    if pattern_label and ":" not in pattern_label and not _is_generic_focus_value(pattern_label):
        return _humanize_focus_label(pattern_label)

    skill_family = (memory.skill_family or "").strip()
    if skill_family and not _is_generic_focus_value(skill_family):
        return _humanize_focus_label(skill_family)

    if mistake_type_label:
        return _humanize_focus_label(mistake_type_label)

    return "grammar"


def _speaking_win_summary(focus_label: str) -> str:
    return f"You showed improvement in {focus_label}."


def _specific_focus_label(
    pattern_label: str | None,
    skill_family: str | None,
    mistake_type_label: str | None,
) -> str:
    pattern_value = (pattern_label or "").strip()
    if pattern_value and ":" not in pattern_value and not _is_generic_focus_value(pattern_value):
        return _humanize_focus_label(pattern_value)

    skill_value = (skill_family or "").strip()
    if skill_value and not _is_generic_focus_value(skill_value):
        return _humanize_focus_label(skill_value)

    if mistake_type_label:
        return _humanize_focus_label(mistake_type_label)

    return "grammar"


def _specific_pattern_focus_label(
    pattern_label: str | None,
    skill_family: str | None,
) -> str | None:
    pattern_value = (pattern_label or "").strip()
    if pattern_value and ":" not in pattern_value and not _is_generic_focus_value(pattern_value):
        return _humanize_focus_label(pattern_value)

    skill_value = (skill_family or "").strip()
    if skill_value and not _is_generic_focus_value(skill_value):
        return _humanize_focus_label(skill_value)

    return None


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
            select(Mistake, Transcript)
            .join(MistakeType, Mistake.mistake_type_id == MistakeType.id)
            .join(Session, Mistake.session_id == Session.id)
            .join(Transcript, Transcript.session_id == Session.id)
            .join(SessionLanguageProfile, SessionLanguageProfile.session_id == Session.id)
            .where(MistakeType.code == code, SessionLanguageProfile.language_profile_id == language_profile_id)
            .order_by(Mistake.id.desc())
            .limit(1)
        )
        latest_result = await db.execute(latest_query)
        latest_row = latest_result.first()
        latest_summary = None
        if latest_row:
            latest_mistake, latest_transcript = latest_row
            latest_summary = _extract_sentence(
                latest_transcript.raw_text or "",
                latest_mistake.start_char,
                latest_mistake.end_char,
            )

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

    # 1b. More specific recurring error patterns, grouped by construction/focus area.
    pattern_query = (
        select(
            Mistake.pattern_label,
            Mistake.skill_family,
            MistakeType.code,
            MistakeType.label,
            func.count(Mistake.id).label("cnt"),
        )
        .join(MistakeType, Mistake.mistake_type_id == MistakeType.id)
        .join(Session, Mistake.session_id == Session.id)
        .join(SessionLanguageProfile, SessionLanguageProfile.session_id == Session.id)
        .where(SessionLanguageProfile.language_profile_id == language_profile_id)
        .group_by(
            Mistake.pattern_label,
            Mistake.skill_family,
            MistakeType.code,
            MistakeType.label,
        )
    )
    pattern_result = await db.execute(pattern_query)
    pattern_groups: dict[str, dict] = {}
    for pattern_label, skill_family, mistake_type_code, mistake_type_label, count in pattern_result.all():
        focus_label = _specific_pattern_focus_label(pattern_label, skill_family)
        if not focus_label:
            continue
        key = focus_label.lower()
        existing = pattern_groups.get(key)
        if existing is None:
            pattern_groups[key] = {
                "focus_label": focus_label,
                "count": count,
                "description": mistake_type_label,
                "recent_mistake_summary": None,
            }
        else:
            existing["count"] += count
            if existing["description"] is None and mistake_type_label:
                existing["description"] = mistake_type_label

    pattern_example_query = (
        select(Mistake, MistakeType, Transcript)
        .join(MistakeType, Mistake.mistake_type_id == MistakeType.id)
        .join(Session, Mistake.session_id == Session.id)
        .join(Transcript, Transcript.session_id == Session.id)
        .join(SessionLanguageProfile, SessionLanguageProfile.session_id == Session.id)
        .where(SessionLanguageProfile.language_profile_id == language_profile_id)
        .order_by(Mistake.id.desc())
        .limit(500)
    )
    pattern_example_result = await db.execute(pattern_example_query)
    for mistake, mistake_type, transcript in pattern_example_result.all():
        focus_label = _specific_pattern_focus_label(mistake.pattern_label, mistake.skill_family)
        if not focus_label:
            continue
        key = focus_label.lower()
        group = pattern_groups.get(key)
        if group is None or group["recent_mistake_summary"] is not None:
            continue
        group["recent_mistake_summary"] = _extract_sentence(
            transcript.raw_text or "",
            mistake.start_char,
            mistake.end_char,
        )

    common_patterns = [
        MistakeCountItem(
            code=group["focus_label"].lower().replace(" ", "-"),
            label=group["focus_label"],
            count=group["count"],
            description=group["description"],
            recent_mistake_summary=group["recent_mistake_summary"],
        )
        for group in sorted(pattern_groups.values(), key=lambda item: (-item["count"], item["focus_label"]))
        if group["count"] > 3
    ][:top_k]

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

    # 5. Speaking-win details from speaking-based improvement wins.
    improvement_banners: list[str] = []
    speaking_win_history: list[SpeakingWinItem] = []
    improvement_query = (
        select(ImprovementEvent, MistakeMemory, Mistake, MistakeType, Transcript)
        .join(MistakeMemory, MistakeMemory.id == ImprovementEvent.memory_id)
        .join(Mistake, Mistake.id == MistakeMemory.source_mistake_id)
        .join(MistakeType, MistakeType.id == Mistake.mistake_type_id)
        .join(Session, Session.id == Mistake.session_id)
        .join(Transcript, Transcript.session_id == Session.id)
        .where(
            ImprovementEvent.language_profile_id == language_profile_id,
            ImprovementEvent.event_type == "win",
        )
        .order_by(ImprovementEvent.created_at.desc())
        .limit(50)
    )
    improvement_result = await db.execute(improvement_query)
    for event, memory, source_mistake, mistake_type, source_transcript in improvement_result.all():
        focus_label = _speaking_win_focus_label(memory, mistake_type.label if mistake_type else None)
        summary = _speaking_win_summary(focus_label)
        previous_bad_sentence = _extract_sentence(
            source_transcript.raw_text or "",
            source_mistake.start_char,
            source_mistake.end_char,
        )
        speaking_win_history.append(
            SpeakingWinItem(
                event_id=event.id,
                memory_id=memory.id,
                created_at=event.created_at.strftime("%Y-%m-%d %H:%M") if event.created_at else "",
                summary=summary,
                focus_label=focus_label,
                previous_bad_sentence=previous_bad_sentence,
                improved_sentence=event.sentence_text,
                previous_wrong_span=source_mistake.transcript_span,
                suggested_correction=source_mistake.suggested_correction,
                reason=event.reason,
                confidence=event.confidence,
            )
        )

    latest_speaking_win = speaking_win_history[0] if speaking_win_history else None
    if latest_speaking_win is not None:
        improvement_banners.append(latest_speaking_win.summary)

    return InsightsResponse(
        top_mistakes=top_mistakes,
        common_patterns=common_patterns,
        trends=trends,
        recent_mistakes=recent_mistakes,
        progress=progress,
        improvement_banners=improvement_banners,
        latest_speaking_win=latest_speaking_win,
        speaking_win_history=speaking_win_history,
    )
