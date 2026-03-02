"""Rewrite training endpoints."""

from difflib import SequenceMatcher

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, cast, Integer
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Mistake, MistakeType, Session, Transcript, RewriteAttempt, User
from app.schemas import (
    RewriteExerciseResponse,
    RewriteSubmitRequest,
    RewriteSubmitResponse,
    RewriteStatsResponse,
    RewriteStatsItem,
)

router = APIRouter(prefix="/api/rewrite", tags=["rewrite"])


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


def _build_expected_sentence(original_sentence: str, wrong_span: str | None, expected: str | None) -> str:
    if not original_sentence:
        return ""
    if wrong_span and expected and wrong_span in original_sentence:
        return original_sentence.replace(wrong_span, expected, 1)
    if expected:
        return expected
    return original_sentence


def _normalize(s: str | None) -> str:
    if not s:
        return ""
    return " ".join(s.lower().strip().split())


def _evaluate_rewrite(
    original_sentence: str,
    wrong_span: str | None,
    expected_correction: str | None,
    user_rewrite: str,
) -> tuple[bool, float, str]:
    expected_sentence = _build_expected_sentence(original_sentence, wrong_span, expected_correction)
    rewrite_norm = _normalize(user_rewrite)
    expected_norm = _normalize(expected_sentence)
    wrong_norm = _normalize(wrong_span)
    corr_norm = _normalize(expected_correction)

    similarity = SequenceMatcher(None, rewrite_norm, expected_norm).ratio()
    has_correction = bool(corr_norm and corr_norm in rewrite_norm)
    removed_wrong = not wrong_norm or wrong_norm not in rewrite_norm

    score = similarity
    if has_correction:
        score = max(score, 0.85)
    if has_correction and removed_wrong:
        score = max(score, 0.92)

    is_correct = score >= 0.85
    if is_correct:
        feedback = "Nice correction. This rewrite looks correct."
    elif has_correction:
        feedback = "Good direction. You used the correction, but sentence form can improve."
    else:
        feedback = "Try replacing the incorrect span with the expected correction."

    return is_correct, round(score, 3), feedback


@router.get("/next", response_model=RewriteExerciseResponse)
async def next_exercise(
    language: str = Query("en"),
    user_id: int = Query(1, ge=1),
    db: AsyncSession = Depends(get_db),
):
    """Serve a recent mistake as a rewrite exercise."""
    user_result = await db.execute(select(User).where(User.id == user_id))
    if user_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Prefer mistakes that were answered less often in rewrite training.
    subq = (
        select(RewriteAttempt.source_mistake_id, func.count(RewriteAttempt.id).label("cnt"))
        .group_by(RewriteAttempt.source_mistake_id)
        .subquery()
    )

    query = (
        select(Mistake, MistakeType, Session, Transcript, subq.c.cnt)
        .join(MistakeType, Mistake.mistake_type_id == MistakeType.id)
        .join(Session, Mistake.session_id == Session.id)
        .join(Transcript, Transcript.session_id == Session.id)
        .outerjoin(subq, subq.c.source_mistake_id == Mistake.id)
        .where(Session.user_id == user_id, Session.language == language)
        .order_by(func.coalesce(subq.c.cnt, 0).asc(), Mistake.id.desc())
        .limit(1)
    )
    result = await db.execute(query)
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="No mistakes available for rewrite training yet")

    mistake, mtype, session, transcript, _ = row
    sentence = _extract_sentence(transcript.raw_text or "", mistake.start_char, mistake.end_char)

    return RewriteExerciseResponse(
        source_mistake_id=mistake.id,
        language=session.language,
        mistake_type_code=mtype.code,
        mistake_type_label=mtype.label,
        original_sentence=sentence,
        wrong_span=mistake.transcript_span,
        expected_correction=mistake.suggested_correction,
        explanation_short=mistake.explanation_short,
    )


@router.post("/submit", response_model=RewriteSubmitResponse)
async def submit_rewrite(req: RewriteSubmitRequest, db: AsyncSession = Depends(get_db)):
    """Submit a rewritten sentence and persist correctness stats."""
    mistake_result = await db.execute(select(Mistake).where(Mistake.id == req.source_mistake_id))
    source_mistake = mistake_result.scalar_one_or_none()
    if not source_mistake:
        raise HTTPException(status_code=404, detail="Source mistake not found")

    is_correct, score, feedback = _evaluate_rewrite(
        req.original_sentence,
        req.wrong_span,
        req.expected_correction,
        req.user_rewrite,
    )

    attempt = RewriteAttempt(
        user_id=req.user_id,
        language=req.language,
        source_mistake_id=req.source_mistake_id,
        original_sentence=req.original_sentence,
        wrong_span=req.wrong_span,
        expected_correction=req.expected_correction,
        user_rewrite=req.user_rewrite,
        is_correct=is_correct,
        score=score,
    )
    db.add(attempt)
    await db.commit()

    return RewriteSubmitResponse(
        is_correct=is_correct,
        score=score,
        feedback=feedback,
        expected_correction=req.expected_correction,
    )


@router.get("/stats", response_model=RewriteStatsResponse)
async def rewrite_stats(
    language: str = Query("en"),
    user_id: int = Query(1, ge=1),
    db: AsyncSession = Depends(get_db),
):
    """Return rewrite-training accuracy stats over time."""
    total_result = await db.execute(
        select(func.count(RewriteAttempt.id), func.coalesce(func.sum(cast(RewriteAttempt.is_correct, Integer)), 0))
        .where(RewriteAttempt.user_id == user_id, RewriteAttempt.language == language)
    )
    total_attempts, total_correct = total_result.one()
    total_attempts = int(total_attempts or 0)
    total_correct = int(total_correct or 0)
    overall_accuracy = (total_correct / total_attempts) if total_attempts else 0.0

    pair_query = await db.execute(
        select(
            RewriteAttempt.wrong_span,
            RewriteAttempt.expected_correction,
            func.count(RewriteAttempt.id).label("attempts"),
            func.coalesce(func.sum(cast(RewriteAttempt.is_correct, Integer)), 0).label("correct"),
            func.max(RewriteAttempt.created_at).label("latest_date"),
        )
        .where(RewriteAttempt.user_id == user_id, RewriteAttempt.language == language)
        .group_by(RewriteAttempt.wrong_span, RewriteAttempt.expected_correction)
        .order_by(func.max(RewriteAttempt.created_at).desc())
        .limit(20)
    )

    items: list[RewriteStatsItem] = []
    for wrong_span, expected, attempts, correct, latest_date in pair_query.all():
        latest_row = await db.execute(
            select(RewriteAttempt.is_correct)
            .where(
                RewriteAttempt.user_id == user_id,
                RewriteAttempt.language == language,
                RewriteAttempt.wrong_span == wrong_span,
                RewriteAttempt.expected_correction == expected,
            )
            .order_by(RewriteAttempt.created_at.desc())
            .limit(1)
        )
        latest_result = latest_row.scalar_one_or_none()

        attempts_int = int(attempts or 0)
        correct_int = int(correct or 0)
        items.append(
            RewriteStatsItem(
                wrong_span=wrong_span,
                expected_correction=expected,
                attempts=attempts_int,
                correct_attempts=correct_int,
                accuracy=(correct_int / attempts_int) if attempts_int else 0.0,
                latest_result=bool(latest_result) if latest_result is not None else None,
                latest_date=latest_date.strftime("%Y-%m-%d %H:%M") if latest_date else None,
            )
        )

    return RewriteStatsResponse(
        total_attempts=total_attempts,
        total_correct=total_correct,
        overall_accuracy=round(overall_accuracy, 3),
        recent_attempts=items,
    )
