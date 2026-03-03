from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.database import get_db
from app.models import User, UserLanguageProfile
from app.schemas import (
    UserLanguageProfileCreate,
    UserLanguageProfileOut,
    UserLanguageProfileSetCurrent,
)

router = APIRouter(prefix="/api", tags=["language_profiles"])


def _default_display_name(language_code: str) -> str:
    names = {
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "ja": "Japanese",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
    }
    return names.get(language_code, language_code.upper())


# Dummy function to get current user - replace with actual auth later
async def get_current_user(session: AsyncSession = Depends(get_db)) -> User:
    # For now, always return a dummy user with ID 1.
    # In a real app, this would involve authentication.
    user = await session.get(User, 1)
    if not user:
        # If user 1 doesn't exist, create it
        user = User()
        session.add(user)
        await session.commit()
        await session.refresh(user)

    # Ensure at least one language profile exists so the UI can initialize.
    profiles_result = await session.execute(
        select(UserLanguageProfile).where(UserLanguageProfile.user_id == user.id)
    )
    profiles = profiles_result.scalars().all()
    if not profiles:
        default_profile = UserLanguageProfile(
            user_id=user.id,
            language_code="en",
            display_name=_default_display_name("en"),
        )
        session.add(default_profile)
        await session.commit()
        await session.refresh(default_profile)
        user.current_language_profile_id = default_profile.id
        session.add(user)
        await session.commit()
        await session.refresh(user)
    elif not user.current_language_profile_id:
        user.current_language_profile_id = profiles[0].id
        session.add(user)
        await session.commit()
        await session.refresh(user)

    return user


@router.post(
    "/user/language_profiles",
    response_model=UserLanguageProfileOut,
    status_code=status.HTTP_201_CREATED,
)
async def create_language_profile(
    profile_in: UserLanguageProfileCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    # Check if a profile for this language already exists for the user
    existing_profile_stmt = select(UserLanguageProfile).where(
        UserLanguageProfile.user_id == current_user.id,
        UserLanguageProfile.language_code == profile_in.language_code,
    )
    existing_profile = (await session.execute(existing_profile_stmt)).scalar_one_or_none()

    if existing_profile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Language profile for {profile_in.language_code} already exists for this user.",
        )

    new_profile = UserLanguageProfile(**profile_in.model_dump(), user_id=current_user.id)
    session.add(new_profile)
    await session.commit()
    await session.refresh(new_profile)

    # If this is the first profile, set it as current
    if not current_user.current_language_profile_id:
        current_user.current_language_profile_id = new_profile.id
        session.add(current_user)
        await session.commit()
        await session.refresh(current_user)

    return new_profile


@router.get(
    "/user/language_profiles", response_model=List[UserLanguageProfileOut]
)
async def get_language_profiles(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    stmt = select(UserLanguageProfile).where(
        UserLanguageProfile.user_id == current_user.id
    )
    result = await session.execute(stmt)
    profiles = result.scalars().all()
    return profiles


@router.put(
    "/user/language_profiles/set_current",
    response_model=UserLanguageProfileOut,
)
async def set_current_language_profile(
    set_current_in: UserLanguageProfileSetCurrent,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    profile = await session.get(UserLanguageProfile, set_current_in.profile_id)
    if not profile or profile.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found or not owned by user")

    current_user.current_language_profile_id = profile.id
    session.add(current_user)
    await session.commit()
    await session.refresh(current_user, attribute_names=["current_language_profile"])
    return current_user.current_language_profile


@router.get(
    "/user/language_profiles/current", response_model=UserLanguageProfileOut, # Now returns current language profile
)
async def get_current_language_profile(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """Get the user's current language profile."""
    await session.refresh(current_user, attribute_names=["current_language_profile"])
    if not current_user.current_language_profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No current language profile set for this user.",
        )
    return current_user.current_language_profile
