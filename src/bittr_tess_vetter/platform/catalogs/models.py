"""Local models for catalog clients.

Contains SourceRecord for provenance tracking. This is a standalone
implementation to avoid external bittr dependencies.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class SourceRecord(BaseModel):
    """Record of where a value came from.

    Captures metadata about a data source for provenance tracking.
    Used to record which catalog/API/file a value was retrieved from.

    Attributes:
        name: Source identifier (e.g., "gaia_dr3", "simbad")
        version: Optional version or data release (e.g., "dr3", "v8.2")
        retrieved_at: When the data was retrieved
        query: Optional query string or API call that produced the data
        raw_ref: Optional blob_ref to the raw response for full traceability
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Source identifier (e.g., 'gaia_dr3')")
    version: str | None = Field(
        default=None,
        description="Version or data release (e.g., 'dr3', 'v8.2')",
    )
    retrieved_at: datetime | None = Field(
        default=None,
        description="When the data was retrieved",
    )
    query: str | None = Field(
        default=None,
        description="Query string or API call that produced the data",
    )
    raw_ref: str | None = Field(
        default=None,
        description="blob_ref to raw response for full traceability",
    )


__all__ = ["SourceRecord"]
