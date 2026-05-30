from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.enums import ConversationState, MessageRole


TaskType = Literal["text", "image", "audio", "video", "document"]


class ConversationMetadata(BaseModel):
    task_type: TaskType | None = None
    prompt: str = ""
    ai_output: str = ""
    ai_output_file_url: str = ""
    is_locked: bool = False


class ConversationCreateResponse(BaseModel):
    conversation_id: int
    state: ConversationState
    assistant_message: str


class ChatRequest(BaseModel):
    conversation_id: int
    message: str = Field(min_length=1, max_length=4000)
    task_type: TaskType | None = None
    prompt: str | None = Field(default=None, max_length=12000)
    ai_output: str | None = Field(default=None, max_length=12000)


class FeedbackStartRequest(BaseModel):
    conversation_id: int
    task_type: TaskType | None = None
    prompt: str | None = Field(default=None, max_length=12000)
    ai_output: str | None = Field(default=None, max_length=12000)


class MessageResponse(BaseModel):
    id: int
    role: MessageRole
    content: str
    timestamp: datetime

    model_config = {"from_attributes": True}


class ConversationSummary(BaseModel):
    id: int
    title: str
    state: ConversationState
    created_at: datetime
    task_type: TaskType | None = None

    model_config = {"from_attributes": True}


class ChatResponse(BaseModel):
    conversation_id: int
    state: ConversationState
    assistant_message: str
    context: dict[str, Any]
    metadata: ConversationMetadata = Field(default_factory=ConversationMetadata)


class ConversationHistoryResponse(BaseModel):
    conversation_id: int
    state: ConversationState
    messages: list[MessageResponse]
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: ConversationMetadata = Field(default_factory=ConversationMetadata)
