from enum import Enum
from typing import List, Dict, Optional, Literal

from pydantic import BaseModel, Field, field_validator


class OperationMode(str, Enum):
    ARRIVAL = "arrival"
    DEPARTURE = "departure"


class RosterGenerationRequest(BaseModel):
    mode: OperationMode = Field(
        default=OperationMode.ARRIVAL,
        description="Operation mode: arrival or departure"
    )
    main_officers_reported: str = Field(
        ...,
        description="Officers who reported (e.g., '1-18' or '1,3,5-10')",
        min_length=1
    )
    report_gl_counters: str = Field(
        default="",
        description="Ground level counters (e.g., '4AC1, 8AC11, 12AC21')"
    )
    handwritten_counters: str = Field(
        default="",
        description="Takeover counters (e.g., '3AC12,5AC13')"
    )
    ot_counters: str = Field(
        default="",
        description="OT officer counters (e.g., '2,20,40')"
    )
    ro_ra_officers: str = Field(
        default="",
        description="Late arrival/early departure adjustments (e.g., '3RO2100, 11RO1700')"
    )
    sos_timings: str = Field(
        default="",
        description="SOS officer timings (e.g., '(AC22)1000-1300, 2000-2200')"
    )
    beam_width: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Beam width for optimization algorithm"
    )
    save_to_history: bool = Field(
        default=True,
        description="Whether to save this generation to history"
    )

    @field_validator("main_officers_reported")
    @classmethod
    def validate_officers(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("main_officers_reported cannot be empty")
        return v.strip()


class OfficerCounts(BaseModel):
    main: int = Field(description="Number of main officers")
    sos: int = Field(description="Number of SOS officers")
    ot: int = Field(description="Number of OT officers")
    total: int = Field(description="Total number of officers")


class StatisticsData(BaseModel):
    stats1: str = Field(description="Statistics before SOS optimization")
    stats2: str = Field(description="Statistics after SOS optimization")


class RosterGenerationResponse(BaseModel):
    success: bool = Field(default=True)
    data: Dict = Field(description="Generated roster data")
    officer_counts: OfficerCounts
    optimization_penalty: Optional[float] = Field(
        default=None,
        description="Optimization penalty score"
    )
    statistics: StatisticsData


class LastInputsResponse(BaseModel):
    success: bool = Field(default=True)
    data: Dict = Field(description="Last used input values")


class RosterHistoryItem(BaseModel):
    id: int
    timestamp: str
    main_officers: str
    gl_counters: Optional[str]
    handwritten_counters: Optional[str]
    ot_counters: Optional[str]
    ro_ra_officers: Optional[str]
    sos_timings: Optional[str]
    beam_width: int
    optimization_penalty: Optional[float]
    main_officer_count: Optional[int]
    sos_officer_count: Optional[int]
    ot_officer_count: Optional[int]
    total_officer_count: Optional[int]
    notes: Optional[str]


class RosterHistoryResponse(BaseModel):
    success: bool = Field(default=True)
    data: List[RosterHistoryItem]
    count: int


class RosterEditCreate(BaseModel):
    edit_type: Literal["delete", "swap", "add"] = Field(
        description="Type of edit operation"
    )
    officer_id: str = Field(description="Primary officer identifier")
    officer_id_2: Optional[str] = Field(
        default=None,
        description="Second officer for swap operations"
    )
    counter_no: Optional[int] = Field(
        default=None,
        description="Counter number for add operations"
    )
    slot_start: int = Field(ge=0, le=47, description="Starting time slot")
    slot_end: int = Field(ge=0, le=47, description="Ending time slot")
    time_start: str = Field(description="Start time in HHMM format")
    time_end: str = Field(description="End time in HHMM format")
    roster_history_id: Optional[int] = Field(
        default=None,
        description="Link to roster history record"
    )
    notes: Optional[str] = Field(default=None, description="Additional notes")

    @field_validator("slot_end")
    @classmethod
    def validate_slot_range(cls, v: int, info) -> int:
        if "slot_start" in info.data and v < info.data["slot_start"]:
            raise ValueError("slot_end must be >= slot_start")
        return v


class RosterEditResponse(BaseModel):
    success: bool = Field(default=True)
    edit_id: int
    message: str = Field(default="Edit saved successfully")


class RosterEditsListResponse(BaseModel):
    success: bool = Field(default=True)
    data: List[Dict]
    count: int


class ErrorResponse(BaseModel):
    success: bool = Field(default=False)
    error: Dict[str, any] = Field(description="Error details")

    class Config:
        arbitrary_types_allowed = True


class SaveInputsRequest(BaseModel):
    main_officers: str
    gl_counters: str = ""
    handwritten_counters: str = ""
    ot_counters: str = ""
    ro_ra_officers: str = ""
    sos_timings: str = ""
    raw_sos_text: str = ""
    beam_width: int = 20


class SuccessResponse(BaseModel):
    success: bool = Field(default=True)
    message: str
