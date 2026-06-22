from pydantic import BaseModel, Field, field_validator
from typing import Any, Literal, Optional
from enum import Enum


class FieldType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DATE = "date"
    DATETIME = "datetime"
    BOOLEAN = "boolean"


class FieldInfo(BaseModel):
    name: str                          # caption / display name (what the user & LLM see)
    type: FieldType = FieldType.STRING
    role: str = "dimension"
    local_name: Optional[str] = None   # physical/upstream column name = the sqlproxy
                                       # binding name; set only when it differs from the
                                       # caption (e.g. "Sub_Category" vs "Sub Category").


class ConnectionSummary(BaseModel):
    connection_type: Optional[str] = None
    server_address: Optional[str] = None
    server_port: Optional[str] = None


class DataSourceMetadata(BaseModel):
    datasource_name: str
    datasource_caption: str = ""
    fields: list[FieldInfo] = []
    connection: Optional[ConnectionSummary] = None
    logical_table_names: list[str] = []
    luid: Optional[str] = None  # Tableau Server datasource LUID (populated in Server mode)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str
    metadata: Optional[DataSourceMetadata] = None
    conversation_history: list[dict] = []
    workbook_name: Optional[str] = None




class FilterSpec(BaseModel):
    field: str
    op: str  # eq, in, gt, gte, lt, lte, between, year, quarter,
             # month, last_n_days, last_n_months, top_n, bottom_n, not_null
    value: Optional[Any] = None
    values: Optional[list] = None   # for op=in
    min: Optional[float] = None     # for op=between
    max: Optional[float] = None     # for op=between
    by: Optional[str] = None        # for top_n/bottom_n: rank by this field
    display_type: Optional[str] = None  # "single_value_list", "multi_value_list", "dropdown", "dropdown_search", "slider_range", "relative_date"
    show_filter_card: bool = True


class CalculatedField(BaseModel):
    name: str
    formula: str
    datatype: str = "real"
    role: str = "measure"


class VizIntent(BaseModel):
    viz_type: str
    title: str
    x_field: str
    y_field: str = ""  # empty for kpi viz type

    @field_validator("y_field", mode="before")
    @classmethod
    def coerce_y_field_none(cls, v):
        """LLM often returns y_field: null for KPI queries — coerce to empty string."""
        return v if v is not None else ""
    color_field: Optional[str] = None
    filters: list[FilterSpec] = []
    calculated_fields: list[CalculatedField] = []
    clarification_needed: Optional[str] = None
    sort: Optional[str] = None
    aggregation: str = "SUM"  # SUM, AVG, COUNT, MIN, MAX
    color_scheme: str = "tableau10"
    action: Optional[str] = None  # "new" | "modify" | "clarify"
    datasource_luid: Optional[str] = None
    secondary_datasource_luid: Optional[str] = None


class ConversationTurn(BaseModel):
    user_message: str
    resolved_intent: Optional[VizIntent] = None
    twb_path: Optional[str] = None
    kind: Optional[str] = None      # "answer" | "create" | "modify"
    assistant_text: str = ""        # readable outcome used as LLM memory (never raw JSON)


class SessionState(BaseModel):
    session_id: str
    turns: list[ConversationTurn] = []
    available_datasources: list[DataSourceMetadata] = []
    published_workbook_luid: Optional[str] = None
    session_workbook_path: Optional[str] = None
    session_workbook_name: Optional[str] = None
    summary: str = ""                  # rolling summary of older turns (filled by compaction)
    cart: list[VizIntent] = []         # accumulated charts for the multi-sheet download

    @property
    def last_intent(self) -> Optional[VizIntent]:
        for turn in reversed(self.turns):
            if turn.resolved_intent is not None:
                return turn.resolved_intent
        return None

    @property
    def last_twb(self) -> Optional[str]:
        for turn in reversed(self.turns):
            if turn.twb_path is not None:
                return turn.twb_path
        return None


class ChatResponse(BaseModel):
    session_id: str
    trace_id: str
    viz_intent: Optional[VizIntent] = None
    twb_filename: str = ""
    twb_download_url: str = ""
    message: str = ""
    warning: Optional[str] = None
    suggestion: Optional[str] = None
    mode: Literal["new_workbook", "sheet_added", "clarification", "conversation"] = "new_workbook"
    judge_score: Optional[float] = None
    judge_feedback: Optional[str] = None
    clarification_needed: Optional[str] = None
    view_url: Optional[str] = None


class FeedbackRequest(BaseModel):
    trace_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    comment: Optional[str] = None


class DownloadRequest(BaseModel):
    metadata: Optional[DataSourceMetadata] = None
    charts: Optional[list[dict]] = None  # if provided, use these instead of server cart


class HealthResponse(BaseModel):
    status: str
    openrouter_status: str          # status of the ACTIVE provider (legacy field name)
    model_id: str                   # model id of the ACTIVE provider
    provider: str = ""              # name of the active provider (groq/openrouter/google)
    version: str = "1.0.0"
    server_mode: bool = False
