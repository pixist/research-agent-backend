"""Request/response models shared across the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    request: str = Field(min_length=1, description="The user's research question.")


class UploadedFile(BaseModel):
    name: str
    size: int
    type: str


class UploadResponse(BaseModel):
    uploaded: list[UploadedFile]
