"""Analysis endpoint: run LLM analysis on a session's transcript."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import AnalyzeRequest, AnalyzeResponse, MistakeOut
from app.services.analysis import analyze_transcript

router = APIRouter(prefix="/api/analyze", tags=["analyze"])


@router.post("", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, db: AsyncSession = Depends(get_db)):
    """Run LLM analysis on a session's transcript and return structured mistakes."""
    try:
        mistakes = await analyze_transcript(
            db,
            session_id=req.session_id,
            transcript_text=req.transcript_text,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    return AnalyzeResponse(
        session_id=req.session_id,
        mistakes=[MistakeOut.model_validate(m) for m in mistakes],
    )
