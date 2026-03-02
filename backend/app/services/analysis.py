"""Analysis service — orchestrates LLM call, parses JSON, stores Mistakes."""

import json
import logging
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    Mistake,
    MistakeType,
    Session,
    Transcript,
    SessionLanguageProfile,
)
from app.schemas import AnalysisResult, LLMMistake
from app.services.llm.factory import get_llm_provider

logger = logging.getLogger(__name__)

COMMON_PROMPT_SUFFIX = (
    "When given a transcript string, output JSON exactly in the schema specified. "
    "For each suspected error, include type, span text, character indices, suggested "
    "correction, short pedagogical explanation (1-2 sentences), and confidence. "
    "Use canonical type codes: verb-tense, preposition, article, word-order, "
    "pronunciation, false-friend, pronoun, pluralization, vocabulary, "
    "subject-verb-agreement, other. If a token's STT confidence < 0.6, set "
    "\"stt_uncertain\": true. If you are uncertain whether something is an error, "
    "set \"uncertain\": true and provide a short reason. Output only valid JSON "
    "matching this schema: "
    '{"mistakes": [{"type": "str", "span_text": "str", "start_char": 0, '
    '"end_char": 0, "suggested_correction": "str", "explanation": "str", '
    '"confidence": 0.0, "stt_uncertain": false, "uncertain": false, '
    '"uncertain_reason": null}]}'
)

SYSTEM_PROMPT_BY_LANGUAGE = {
    "fr": (
        "You are a French grammar and pedagogy assistant focused on learner French. "
        "Prioritize: agreement (gender/number), article usage (le/la/les/un/une/des), "
        "verb conjugation and tense selection (present, passe compose, imparfait), "
        "prepositions (a/de/en/dans/chez), clitics and pronouns (y/en, me/te/se), "
        "negation (ne...pas), adjective placement, and false-friend vocabulary. "
        "Do not over-correct colloquial but acceptable spoken French. "
        + COMMON_PROMPT_SUFFIX
    ),
    "es": (
        "You are a Spanish grammar and pedagogy assistant focused on learner Spanish. "
        "Prioritize: verb conjugation and tense/aspect (preterito vs imperfecto), "
        "ser vs estar, por vs para, gender/number agreement, article usage, clitic "
        "pronouns (lo/la/le/se), reflexive constructions, prepositions, and common "
        "false-friend vocabulary. Do not over-correct regional but valid variants. "
        + COMMON_PROMPT_SUFFIX
    ),
    "ja": (
        "You are a Japanese grammar and pedagogy assistant focused on learner Japanese. "
        "Prioritize: particle errors (wa/ga/o/ni/de/e), politeness/register consistency "
        "(desu/masu vs plain form), verb/adjective conjugation, tense/negation, "
        "word order, counters/classifiers, and unnatural lexical choice. "
        "When relevant, suggest natural Japanese phrasing rather than literal translations. "
        + COMMON_PROMPT_SUFFIX
    ),
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a grammar and pedagogy assistant. " + COMMON_PROMPT_SUFFIX
)


def _build_user_prompt(
    transcript: str,
    language: str,
    prior_summary: Optional[list[dict]] = None,
) -> str:
    """Build the user prompt with transcript and optional prior mistake summary."""
    parts = [
        f"Language code: {language}",
        f'Transcript: "{transcript}"',
    ]
    if prior_summary:
        summary_text = ", ".join(
            f"{item['code']} ({item['count']})" for item in prior_summary[:10]
        )
        parts.append(f"\nPrior mistake summary (top types with counts): {summary_text}")
    return "\n".join(parts)


async def get_prior_mistake_summary(
    db: AsyncSession,
    user_id: int,
    language: str,
) -> list[dict]:
    """Get top 10 mistake types by count for the same user and language."""
    query = (
        select(MistakeType.code, func.count(Mistake.id).label("cnt"))
        .join(Mistake, Mistake.mistake_type_id == MistakeType.id)
        .join(Session, Mistake.session_id == Session.id)
        .group_by(MistakeType.code)
        .where(Session.user_id == user_id, Session.language == language)
        .order_by(func.count(Mistake.id).desc())
        .limit(10)
    )
    result = await db.execute(query)
    return [{"code": row[0], "count": row[1]} for row in result.all()]


def _parse_llm_response(raw: str) -> list[LLMMistake]:
    """Parse the LLM JSON response into validated Pydantic models."""
    # Try to extract JSON from the response (handle markdown fences etc.)
    text = raw.strip()
    if text.startswith("```"):
        # Strip markdown code fences
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            logger.warning(
                f"Extracted partial JSON from LLM response: {text[start:end][:500]}..."
            )
            data = json.loads(text[start:end])
        else:
            logger.error(f"Could not parse LLM response as JSON: {text[:200]}")
            return []

    result = AnalysisResult(**data)
    return result.mistakes


async def _resolve_mistake_type(db: AsyncSession, code: str) -> int:
    """Resolve a mistake type code to its ID. Falls back to 'other' if unknown."""
    result = await db.execute(select(MistakeType).where(MistakeType.code == code))
    mt = result.scalar_one_or_none()
    if mt:
        return mt.id

    # Fallback to 'other'
    result = await db.execute(select(MistakeType).where(MistakeType.code == "other"))
    mt = result.scalar_one_or_none()
    if mt:
        return mt.id

    # Shouldn't happen if seeds ran
    raise ValueError(f"MistakeType 'other' not found. Run seed first.")


async def analyze_transcript(
    db: AsyncSession,
    session_id: int,
    transcript_text: Optional[str] = None,
) -> list[Mistake]:
    """Run LLM analysis on a transcript and store mistakes in the DB."""
    session_result = await db.execute(select(Session).where(Session.id == session_id))
    session = session_result.scalar_one_or_none()
    if not session:
        raise ValueError(f"Session {session_id} not found")

    # Ensure there is a language profile link for this session (backward compatible).
    link_result = await db.execute(
        select(SessionLanguageProfile).where(SessionLanguageProfile.session_id == session_id)
    )
    if link_result.scalar_one_or_none() is None:
        logger.info("Session %s has no language profile link yet; proceeding with session.language", session_id)

    # Get transcript text
    if transcript_text is None:
        result = await db.execute(
            select(Transcript).where(Transcript.session_id == session_id)
        )
        transcript_obj = result.scalar_one_or_none()
        if not transcript_obj:
            raise ValueError(f"No transcript found for session {session_id}")
        transcript_text = transcript_obj.raw_text

    # Get prior mistake summary for personalization in the same language.
    prior_summary = await get_prior_mistake_summary(
        db,
        user_id=session.user_id or 1,
        language=session.language,
    )

    # Build prompts and call LLM
    user_prompt = _build_user_prompt(transcript_text, session.language, prior_summary)
    system_prompt = SYSTEM_PROMPT_BY_LANGUAGE.get(session.language, DEFAULT_SYSTEM_PROMPT)
    llm = get_llm_provider()
    logger.info(f"LLM System Prompt (Language: {session.language}): \"\"\"{system_prompt}\"\"\"") # ADD THIS LINE
    logger.info(f"LLM User Prompt: \"\"\"{user_prompt}\"\"\"") # ADD THIS LINE

    try:
        raw_response = await llm.complete(system_prompt, user_prompt)
        logger.info(f"LLM Raw Response: {raw_response[:500]}...") # ADD THIS LINE (truncated to 500 chars)
        llm_mistakes = _parse_llm_response(raw_response)
    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        session.status = "error" 
        llm_mistakes = []

    # Store mistakes
    db_mistakes = []
    for m in llm_mistakes:
        type_id = await _resolve_mistake_type(db, m.type)
        mistake = Mistake(
            session_id=session_id,
            mistake_type_id=type_id,
            transcript_span=m.span_text,
            start_char=m.start_char,
            end_char=m.end_char,
            suggested_correction=m.suggested_correction,
            explanation_short=m.explanation,
            confidence=m.confidence,
            stt_uncertain=m.stt_uncertain,
            uncertain=m.uncertain,
            uncertain_reason=m.uncertain_reason,
        )
        db.add(mistake)
        db_mistakes.append(mistake)

    # Update session status
    if session:
        session.status = "analyzed"

    await db.commit()

    # Refresh to get IDs and relationships
    for m in db_mistakes:
        await db.refresh(m)

    return db_mistakes
