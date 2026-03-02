"""Seed canonical MistakeType rows."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import MistakeType

CANONICAL_TYPES = [
    ("verb-tense", "Verb Tense", "Incorrect use of verb tenses (past, present, future, perfect, etc.)"),
    ("preposition", "Preposition", "Wrong or missing preposition usage"),
    ("article", "Article", "Incorrect, missing, or unnecessary article (a, an, the)"),
    ("word-order", "Word Order", "Words placed in the wrong order in the sentence"),
    ("pronunciation", "Pronunciation", "Word pronounced or transcribed incorrectly due to pronunciation error"),
    ("false-friend", "False Friend", "Using a word that looks similar to a word in the native language but has a different meaning"),
    ("pronoun", "Pronoun", "Incorrect pronoun usage (he/she/it, subject/object, possessive)"),
    ("pluralization", "Pluralization", "Incorrect singular/plural form"),
    ("vocabulary", "Vocabulary", "Using a wrong or imprecise word; could use a better or more idiomatic word"),
    ("subject-verb-agreement", "Subject-Verb Agreement", "Subject and verb do not agree in number or person"),
    ("other", "Other", "Other grammatical or usage error not covered by specific categories"),
]


async def seed_mistake_types(db: AsyncSession) -> None:
    """Insert canonical mistake types if they don't exist."""
    for code, label, description in CANONICAL_TYPES:
        result = await db.execute(select(MistakeType).where(MistakeType.code == code))
        existing = result.scalar_one_or_none()
        if existing is None:
            db.add(MistakeType(code=code, label=label, description=description))
    await db.commit()
