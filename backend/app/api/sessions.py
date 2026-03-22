"""Session endpoints: create and retrieve sessions."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from typing import Optional

from app.database import get_db
from app.config import settings
from app.models import (
    Mistake,
    Session,
    Transcript,
    User,
    UserLanguageProfile,
    SessionLanguageProfile,
    PracticeAttempt,
)
from app.schemas import SessionCreate, SessionOut, SessionDetailOut, SessionListOut
from app.services.stt.factory import get_final_stt_provider
from app.services.analysis import analyze_transcript

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

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


def _language_display_name(language: str) -> str:
    names = {
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "ja": "Japanese",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
    }
    return names.get(language, language.upper())


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
    return value.strip().lower() in _GENERIC_SKILL_FAMILIES


def _mistake_focus_label(mistake: Mistake) -> str | None:
    pattern_label = (mistake.pattern_label or "").strip()
    if pattern_label and ":" not in pattern_label and not _is_generic_focus_value(pattern_label):
        return _humanize_focus_label(pattern_label)

    skill_family = (mistake.skill_family or "").strip()
    if skill_family and not _is_generic_focus_value(skill_family):
        return _humanize_focus_label(skill_family)

    if mistake.mistake_type and mistake.mistake_type.label:
        return _humanize_focus_label(mistake.mistake_type.label)

    return None


def _session_focus_summary(
    session: Session,
) -> tuple[int, str | None, list[str], str | None, list[str]]:
    if not session.mistakes:
        return 0, None, [], None, []

    type_counts: dict[str, int] = {}
    counts: dict[str, int] = {}
    for mistake in session.mistakes:
        if mistake.mistake_type and mistake.mistake_type.label:
            type_label = _humanize_focus_label(mistake.mistake_type.label)
            type_counts[type_label] = type_counts.get(type_label, 0) + 1
        label = _mistake_focus_label(mistake)
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1

    ordered_type_labels = [
        label for label, _ in sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    if not counts:
        return (
            len(session.mistakes),
            ordered_type_labels[0] if ordered_type_labels else None,
            ordered_type_labels,
            None,
            [],
        )

    ordered_labels = [
        label for label, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return (
        len(session.mistakes),
        ordered_type_labels[0] if ordered_type_labels else None,
        ordered_type_labels,
        ordered_labels[0],
        ordered_labels,
    )


async def _load_session_detail(db: AsyncSession, session_id: int) -> Session:
    result = await db.execute(
        select(Session)
        .where(Session.id == session_id)
        .options(
            selectinload(Session.transcript),
            selectinload(Session.mistakes).selectinload(Mistake.mistake_type),
        )
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("", response_model=SessionDetailOut)
async def create_session(
    audio_file: Optional[UploadFile] = File(None),
    language_profile_id: int = Form(...), # Now required
    user_id: Optional[int] = Form(None),
    transcript_text: Optional[str] = Form(None),
    practice_topic_key: Optional[str] = Form(None),
    practice_topic_text: Optional[str] = Form(None),
    is_free_talk: bool = Form(False),
    estimated_level: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Create a new session.

    Accepts either:
    - An audio file (will be transcribed via STT)
    - A pre-built transcript_text (from real-time transcription)
    """
    # Ensure user exists (create default user if needed)
    if user_id:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
    else:
        # Use or create default user (id=1)
        result = await db.execute(select(User).where(User.id == 1))
        user = result.scalar_one_or_none()
        if not user:
            user = User(id=1)
            db.add(user)
            await db.flush()
        user_id = 1

    # Fetch the language profile
    language_profile = await db.get(UserLanguageProfile, language_profile_id)
    if not language_profile or language_profile.user_id != user_id:
        raise HTTPException(status_code=404, detail="Language profile not found or not owned by user")
    language_code = language_profile.language_code

    stt_provider_name = settings.final_stt_provider or settings.stt_provider
    stt_confidence = None
    tokens_with_timestamps = None

    # If audio file provided, transcribe it
    if audio_file and not transcript_text:
        audio_bytes = await audio_file.read()
        stt = get_final_stt_provider()
        result = await stt.transcribe(
            audio_bytes,
            None,
            filename=audio_file.filename,
            content_type=audio_file.content_type,
        )
        transcript_text = result.text
        stt_confidence = result.average_confidence
        tokens_with_timestamps = [
            {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "confidence": seg.confidence,
            }
            for seg in result.segments
        ]

    if not transcript_text:
        raise HTTPException(
            status_code=400,
            detail="Either audio_file or transcript_text must be provided",
        )

    # Create session
    session = Session(
        user_id=user_id,
        language=language_code,
        stt_provider=stt_provider_name,
        stt_confidence_summary=stt_confidence,
        status="transcribed",
    )
    db.add(session)
    await db.flush()

    # Link the session to the user's language profile
    db.add(
        SessionLanguageProfile(
            session_id=session.id,
            language_profile_id=language_profile.id,
        )
    )

    # Create transcript
    transcript = Transcript(
        session_id=session.id,
        raw_text=transcript_text,
        tokens_with_timestamps=tokens_with_timestamps,
    )
    db.add(transcript)

    # Persist practice metadata for this spoken answer when provided.
    if practice_topic_key or is_free_talk:
        db.add(
            PracticeAttempt(
                user_id=user_id,
                session_id=session.id,
                language=language_code,
                topic_key=practice_topic_key or "free_talk",
                topic_text=practice_topic_text or "Speak freely about any topic.",
                is_free_talk=is_free_talk,
                estimated_level=estimated_level,
            )
        )
    await db.commit()

    # If audio file was uploaded, auto-analyze
    if audio_file:
        try:
            await analyze_transcript(db, session.id)
        except Exception:
            pass  # Analysis failure shouldn't block session creation

    return await _load_session_detail(db, session.id)


@router.get("", response_model=SessionListOut)
async def list_sessions(
    language_profile_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
):
    """List all sessions for a given language profile, most recent first."""
    stmt = select(Session).options(
        selectinload(Session.mistakes).selectinload(Mistake.mistake_type)
    )
    if language_profile_id:
        stmt = stmt.join(SessionLanguageProfile).where(
            SessionLanguageProfile.language_profile_id == language_profile_id
        )
    stmt = stmt.order_by(Session.created_at.desc())
    result = await db.execute(stmt)
    sessions = result.scalars().all()
    session_items = []
    for session in sessions:
        (
            mistake_count,
            primary_mistake_type_label,
            mistake_type_labels,
            primary_focus_label,
            focus_labels,
        ) = _session_focus_summary(session)
        session_items.append(
            SessionOut(
                id=session.id,
                user_id=session.user_id,
                created_at=session.created_at,
                language=session.language,
                audio_meta=session.audio_meta,
                stt_provider=session.stt_provider,
                stt_confidence_summary=session.stt_confidence_summary,
                status=session.status,
                mistake_count=mistake_count,
                primary_mistake_type_label=primary_mistake_type_label,
                mistake_type_labels=mistake_type_labels,
                primary_focus_label=primary_focus_label,
                focus_labels=focus_labels,
            )
        )
    return SessionListOut(sessions=session_items)


@router.get("/{session_id}", response_model=SessionDetailOut)
async def get_session(session_id: int, db: AsyncSession = Depends(get_db)):
    """Get a session with full transcript and analysis."""
    return await _load_session_detail(db, session_id)
