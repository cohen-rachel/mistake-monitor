"""Analysis service — orchestrates LLM call, parses JSON, stores Mistakes."""

import json
import logging
import re
from typing import Optional

from langdetect import DetectorFactory, LangDetectException, detect_langs

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
    "Be highly conservative: only flag clear learner mistakes that genuinely require correction. "
    "Do not flag stylistic preferences, more formal alternatives, paraphrases, or wording that is merely less polished. "
    "If a phrase is grammatical and acceptable in spontaneous conversation, do not flag it. "
    "Do not rewrite utterances to sound more natural, more explicit, more standard, or more native-like unless the original is actually incorrect. "
    "Do not convert spoken phrasing into a different wording with the same meaning. "
    "Do not treat discourse markers, fillers, hedges, self-repairs, repetitions, restarts, or sentence fragments as mistakes unless they create a clear grammatical error. "
    "If the utterance appears incomplete or cut off, assume missing context and do not guess. "
    "If you are not highly confident that something is truly an error, omit it rather than guessing. "
    "Never include an item if the suggested correction is blank, identical to the original span, or does not actually change the wording. "
    "Never return a 'mistake' whose explanation says the original is grammatically correct, not an error, or only a style/formality/register preference. "
    "If a correction is required, explain the grammatical problem directly rather than describing it as a matter of polish or formality. "
    "Use canonical type codes: verb-tense, preposition, article, word-order, "
    "pronunciation, false-friend, pronoun, pluralization, vocabulary, "
    "subject-verb-agreement, other. If a token's STT confidence < 0.6, set "
    "\"stt_uncertain\": true. If you are uncertain whether something is an error, "
    "set \"uncertain\": true and provide a short reason. If the language of the transcript does not match the language you are aiming to correct, output the message 'Language mismatch: transcript is in {transcript_language} but you are aiming to correct {language}'. Output only valid JSON "
    "matching this schema: "
    '{"mistakes": [{"type": "str", "span_text": "str", "start_char": 0, '
    '"end_char": 0, "suggested_correction": "str", "explanation": "str", '
    '"confidence": 0.0, "stt_uncertain": false, "uncertain": false, '
    '"uncertain_reason": null}]}'
)

SYSTEM_PROMPT_BY_LANGUAGE = {
    "en": (
        "You are a English grammar and pedagogy assistant focused on learner English. "
        "Prioritize: verb conjugation and tense/aspect (present, past, future), "
        "subject-verb-agreement, article usage (a/an/the), preposition usage (in/on/at/to/from), "
        "clitic pronouns (I/me/my/mine, you/your/yours, he/him/his, she/her/hers, it/its/its, we/us/our/ours, they/them/their/theirs), "
        "negation (not/no/never), adjective placement, and false-friend vocabulary. "
        "This is conversational English, not formal writing. Be strict about avoiding false positives. "
        "Only flag a construction when it is clearly ungrammatical or clearly the wrong lexical choice for the intended meaning. "
        "Do not over-correct colloquial but acceptable spoken English. "
        "Do not rewrite for style, tone, register, or greater formality. "
        "Do not replace valid conversational expressions with more polished alternatives. "
        "Do not 'improve' fragments, interruptions, trailing-off sentences, hesitations, softened requests, or informal discourse markers. "
        "Assume the speaker may still be mid-sentence unless the grammar error is already unambiguous. "
        "If a careful native speaker could plausibly say it in conversation, prefer no correction. "
        "Your job is error detection, not rewriting."
        + COMMON_PROMPT_SUFFIX
    ),
    "fr": (
        "You are a French grammar and pedagogy assistant focused on learner French. "
        "Prioritize: agreement (gender/number), article usage (le/la/les/un/une/des), "
        "verb conjugation and tense selection (present, passe compose, imparfait), "
        "prepositions (a/de/en/dans/chez), clitics and pronouns (y/en, me/te/se), "
        "negation (ne...pas), adjective placement, and false-friend vocabulary. "
        "This is conversational French, not formal writing. Be strict about avoiding false positives. "
        "Do not over-correct colloquial but acceptable spoken French. "
        "Do not rewrite for style, tone, register, or greater formality. "
        "Do not replace valid conversational expressions with more polished alternatives. "
        "Do not 'improve' fragments, interruptions, trailing-off sentences, hesitations, or informal discourse markers. "
        "Assume the speaker may still be mid-sentence unless the grammar error is already unambiguous. "
        "If a careful native speaker could plausibly say it in conversation, prefer no correction."
        + COMMON_PROMPT_SUFFIX
    ),
    "es": (
        "You are a Spanish grammar and pedagogy assistant focused on learner Spanish. "
        "Prioritize: verb conjugation and tense/aspect (preterito vs imperfecto), "
        "ser vs estar, por vs para, gender/number agreement, article usage, clitic "
        "pronouns (lo/la/le/se), reflexive constructions, prepositions, and common "
        "false-friend vocabulary. This is conversational Spanish, not formal writing. "
        "Be strict about avoiding false positives. Do not over-correct regional but valid variants. "
        "Do not rewrite for style, tone, register, or greater formality. "
        "Do not replace valid conversational expressions with more polished alternatives. "
        "Do not 'improve' fragments, interruptions, trailing-off sentences, hesitations, or informal discourse markers. "
        "Assume the speaker may still be mid-sentence unless the grammar error is already unambiguous. "
        "If a careful native speaker could plausibly say it in conversation, prefer no correction."
        + COMMON_PROMPT_SUFFIX
    ),
    "ja": (
        "You are a Japanese grammar and pedagogy assistant focused on learner Japanese. "
        "Prioritize: particle errors (wa/ga/o/ni/de/e), politeness/register consistency "
        "(desu/masu vs plain form), word choice,"
        "word order,  and unnatural lexical choice. "
        "This is conversational Japanese, not formal writing. Be strict about avoiding false positives. "
        "Do not rewrite for style, tone, register, or greater formality unless the original is actually incorrect. "
        "Do not replace valid conversational expressions with more polished alternatives. "
        "Do not 'improve' fragments, interruptions, trailing-off sentences, hesitations, or informal discourse markers. "
        "Assume the speaker may still be mid-sentence unless the grammar error is already unambiguous. "
        "If a careful native speaker could plausibly say it in conversation, prefer no correction."
        + COMMON_PROMPT_SUFFIX
    ),
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a grammar and pedagogy assistant. " + COMMON_PROMPT_SUFFIX
)


# Stabilize langdetect randomness so repeated runs behave same.
DetectorFactory.seed = 0

__LANGUAGE_CONFIDENCE_THRESHOLD = 0.6


LANGUAGE_MISMATCH_PATTERN = re.compile(
    r"Language mismatch:\s*transcript is in (?P<transcript>.+?) but you are aiming to correct (?P<target>.+)",
    re.IGNORECASE,
)


class LanguageMismatchError(ValueError):
    def __init__(self, transcript_language: str, target_language: str, raw_response: str):
        self.transcript_language = transcript_language
        self.target_language = target_language
        self.raw_response = raw_response
        super().__init__(
            f"Language mismatch: transcript is in {transcript_language} but you are aiming to correct {target_language}"
        )


def _build_user_prompt(
    transcript: str,
    language: str,
    prior_summary: Optional[list[dict]] = None,
    allow_language_mismatch: bool = False,
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
    if allow_language_mismatch:
        parts.append(
            "\nThe user explicitly chose to continue even if the transcript may not match the selected language profile. "
            "Do not refuse due to language mismatch. Analyze the transcript against the selected target language as best you can."
        )
    return "\n".join(parts)


def _strip_code_fences(text: str) -> str:
    content = text.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        content = "\n".join(lines)
    return content


def _maybe_language_mismatch(text: str) -> Optional[LanguageMismatchError]:
    match = LANGUAGE_MISMATCH_PATTERN.search(text)
    if not match:
        return None
    transcript_language = match.group("transcript").strip()
    target_language = match.group("target").strip()
    return LanguageMismatchError(transcript_language, target_language, text)


def _detect_transcript_language(text: str) -> Optional[tuple[str, float]]:
    content = text.strip()
    if not content:
        return None

    try:
        detections = detect_langs(content)
    except LangDetectException as exc:
        logger.warning("Unable to detect transcript language: %s", exc)
        return None

    if not detections:
        return None

    best = detections[0]
    if best.prob < __LANGUAGE_CONFIDENCE_THRESHOLD:
        logger.debug(
            "Low-confidence language detection (%.2f) for transcript; treating as unknown.",
            best.prob,
        )
        return None

    return best.lang, best.prob


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
    text = _strip_code_fences(raw)
    mismatch = _maybe_language_mismatch(text)
    if mismatch:
        raise mismatch

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
    allow_language_mismatch: bool = False,
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

    detected_language_info = _detect_transcript_language(transcript_text)
    if detected_language_info:
        detected_language, confidence = detected_language_info
        if detected_language != session.language and not allow_language_mismatch:
            logger.warning(
                "Detected transcript language %s (confidence %.2f) != session target %s; short-circuiting analysis.",
                detected_language,
                confidence,
                session.language,
            )
            session.status = "error"
            await db.commit()
            mismatch_error = LanguageMismatchError(detected_language, session.language, f"detected confidence {confidence:.2f}")
            raise ValueError(str(mismatch_error))

    # Get prior mistake summary for personalization in the same language.
    prior_summary = await get_prior_mistake_summary(
        db,
        user_id=session.user_id or 1,
        language=session.language,
    )

    # Build prompts and call LLM
    user_prompt = _build_user_prompt(
        transcript_text,
        session.language,
        prior_summary,
        allow_language_mismatch=allow_language_mismatch,
    )
    system_prompt = SYSTEM_PROMPT_BY_LANGUAGE.get(session.language, DEFAULT_SYSTEM_PROMPT)
    llm = get_llm_provider()
    logger.info(f"LLM System Prompt (Language: {session.language}): \"\"\"{system_prompt}\"\"\"")
    logger.info(f"LLM User Prompt: \"\"\"{user_prompt}\"\"\"")

    raw_response = ""
    llm_mistakes: list[LLMMistake] = []
    language_mismatch_error: Optional[LanguageMismatchError] = None
    analysis_exception: Optional[Exception] = None
    analysis_failed = False

    try:
        raw_response = await llm.complete(system_prompt, user_prompt)
        logger.info(f"LLM Raw Response: {raw_response[:500]}...")
        llm_mistakes = _parse_llm_response(raw_response)
    except LanguageMismatchError as mismatch_error:
        if allow_language_mismatch:
            logger.warning(
                "LLM reported language mismatch despite override; retrying once with stronger instruction."
            )
            retry_user_prompt = (
                f"{user_prompt}\n\nOverride reminder: the user explicitly chose to proceed. "
                "You must return JSON analysis for the selected target language and must not reply with a language mismatch refusal."
            )
            try:
                raw_response = await llm.complete(system_prompt, retry_user_prompt)
                logger.info(f"LLM Raw Response (override retry): {raw_response[:500]}...")
                llm_mistakes = _parse_llm_response(raw_response)
            except LanguageMismatchError as retry_mismatch_error:
                analysis_failed = True
                language_mismatch_error = retry_mismatch_error
                logger.warning(
                    "LLM still reported language mismatch after override retry (transcript=%s target=%s).",
                    retry_mismatch_error.transcript_language,
                    retry_mismatch_error.target_language,
                )
        else:
            analysis_failed = True
            language_mismatch_error = mismatch_error
            logger.warning(
                "LLM reported language mismatch (transcript=%s target=%s); raw response snippet: %s",
                mismatch_error.transcript_language,
                mismatch_error.target_language,
                raw_response[:500],
            )
    except Exception as exc:
        analysis_failed = True
        analysis_exception = exc
        logger.exception("LLM analysis failed during completion/parsing; inspect prompts and response above.")
        logger.debug("LLM Raw Response (truncated): %s", raw_response[:500])

    db_mistakes: list[Mistake] = []
    if not analysis_failed:
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

    if session:
        session.status = "error" if analysis_failed else "analyzed"

    await db.commit()

    # Refresh to get IDs and relationships
    for m in db_mistakes:
        await db.refresh(m)

    if language_mismatch_error:
        raise ValueError(str(language_mismatch_error))

    if analysis_exception:
        raise analysis_exception

    return db_mistakes
