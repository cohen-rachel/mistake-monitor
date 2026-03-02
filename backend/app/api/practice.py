"""Practice endpoint: generate practice prompts for a mistake type."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import PracticeRequest, PracticeResponse, PracticePrompt
from app.services.llm.factory import get_llm_provider

router = APIRouter(prefix="/api/practice", tags=["practice"])

PRACTICE_SYSTEM_PROMPT = (
    "You are a language practice assistant. Generate short practice exercises "
    "that help the learner correct a specific type of grammar mistake. "
    "For each exercise, provide a prompt sentence with a blank or error, "
    "and the expected correct answer. Output valid JSON matching this schema: "
    '{"prompts": [{"prompt": "str", "expected_answer": "str"}]}'
)


@router.post("", response_model=PracticeResponse)
async def generate_practice(
    req: PracticeRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate practice prompts for a specific mistake type.

    This endpoint is a placeholder — it will call the LLM to generate
    targeted practice prompts based on the user's most common mistakes.
    """
    # For now, return placeholder prompts
    placeholder_prompts = [
        PracticePrompt(
            prompt=f"Practice for '{req.mistake_type_code}' — coming soon!",
            expected_answer="This feature is under development.",
        )
    ]

    return PracticeResponse(
        mistake_type_code=req.mistake_type_code,
        prompts=placeholder_prompts,
    )
