"""Typed schema models for authored custom chart views."""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_CUSTOM_VIEWS = 100
MAX_SERIES_PER_VIEW = 12
MAX_VIEW_ID_LENGTH = 128
MAX_TITLE_LENGTH = 200
MAX_DESCRIPTION_LENGTH = 1000
MAX_NOTE_LENGTH = 1000
MAX_PATH_LENGTH = 256
MAX_ZOOM_PRESETS = 16


class ViewProducerSource(str, Enum):
    AGENT = "agent"
    USER = "user"
    SYSTEM = "system"


class ViewMode(str, Enum):
    DETERMINISTIC = "deterministic"
    AD_HOC = "ad_hoc"


class ChartType(str, Enum):
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"


class ViewQualityStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ViewProducerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: ViewProducerSource
    id: str | None = Field(default=None, min_length=1, max_length=128)


class PathRefModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # RFC 6901 JSON Pointer, e.g. "/plot_data/full_lc/time".
    path: str = Field(min_length=1, max_length=MAX_PATH_LENGTH)


class ZoomPresetModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=64)
    x_range: list[float] | None = None
    y_range: list[float] | None = None

    @field_validator("x_range", "y_range")
    @classmethod
    def _validate_range(cls, value: list[float] | None) -> list[float] | None:
        return _validate_optional_range(value)


class ChartOptionsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x_label: str | None = Field(default=None, max_length=128)
    y_label: str | None = Field(default=None, max_length=128)
    show_error_bars: bool = True
    legend: bool = True
    x_range: list[float] | None = None
    y_range: list[float] | None = None
    zoom_presets: list[ZoomPresetModel] = Field(default_factory=list)

    @field_validator("x_range", "y_range")
    @classmethod
    def _validate_range(cls, value: list[float] | None) -> list[float] | None:
        return _validate_optional_range(value)

    @field_validator("zoom_presets")
    @classmethod
    def _validate_zoom_presets(cls, value: list[ZoomPresetModel]) -> list[ZoomPresetModel]:
        if len(value) > MAX_ZOOM_PRESETS:
            raise ValueError(f"zoom_presets cannot exceed {MAX_ZOOM_PRESETS}")
        return value


class ChartSeriesModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str | None = Field(default=None, max_length=128)
    x: PathRefModel
    y: PathRefModel
    y_err: PathRefModel | None = None


class ChartSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: ChartType
    series: list[ChartSeriesModel] = Field(min_length=1, max_length=MAX_SERIES_PER_VIEW)
    options: ChartOptionsModel = Field(default_factory=ChartOptionsModel)


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    references: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("notes")
    @classmethod
    def _validate_note_lengths(cls, value: list[str]) -> list[str]:
        for note in value:
            if len(note) > MAX_NOTE_LENGTH:
                raise ValueError(f"note length cannot exceed {MAX_NOTE_LENGTH}")
        return value


class QualityModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_points_required: int = Field(default=3, ge=1)
    status: ViewQualityStatus = ViewQualityStatus.OK
    flags: list[str] = Field(default_factory=list)


class CustomViewModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=MAX_VIEW_ID_LENGTH)
    title: str = Field(min_length=1, max_length=MAX_TITLE_LENGTH)
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION_LENGTH)
    producer: ViewProducerModel
    mode: ViewMode
    chart: ChartSpecModel
    evidence: EvidenceModel = Field(default_factory=EvidenceModel)
    quality: QualityModel = Field(default_factory=QualityModel)


class CustomViewsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = Field(default="1", min_length=1, max_length=16)
    views: list[CustomViewModel] = Field(default_factory=list)

    @field_validator("views")
    @classmethod
    def _validate_view_count(cls, views: list[CustomViewModel]) -> list[CustomViewModel]:
        if len(views) > MAX_CUSTOM_VIEWS:
            raise ValueError(f"custom views cannot exceed {MAX_CUSTOM_VIEWS} items")
        return views

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> CustomViewsModel:
        seen: set[str] = set()
        for view in self.views:
            if view.id in seen:
                raise ValueError(f"custom view ids must be unique; duplicate: {view.id}")
            seen.add(view.id)
        return self


def _validate_optional_range(value: list[float] | None) -> list[float] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("axis ranges must be null or [min, max]")
    min_value, max_value = value
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        raise ValueError("axis ranges must be finite")
    if min_value >= max_value:
        raise ValueError("axis range min must be less than max")
    return [float(min_value), float(max_value)]
