from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationCreateResponse,
    FeedbackStartRequest,
    ConversationHistoryResponse,
    ConversationSummary,
)
from app.services.chat_service import ChatService
from app.services.file_storage import LocalFileStorage

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/conversations", response_model=ConversationCreateResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationCreateResponse:
    service = ChatService(db)
    conversation, assistant_message = service.create_conversation(current_user)
    return ConversationCreateResponse(
        conversation_id=conversation.id,
        state=conversation.state,
        assistant_message=assistant_message,
    )


@router.get("/conversations", response_model=list[ConversationSummary])
def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ConversationSummary]:
    service = ChatService(db)
    return service.list_conversations(current_user)


@router.post("/send", response_model=ChatResponse)
def send_message(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatResponse:
    service = ChatService(db)
    try:
        conversation, assistant_message = service.process_message(
            current_user,
            payload.conversation_id,
            payload.message,
            task_type=payload.task_type,
            prompt=payload.prompt,
            ai_output=payload.ai_output,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return ChatResponse(
        conversation_id=conversation.id,
        state=conversation.state,
        assistant_message=assistant_message,
        context=conversation.context or {},
        metadata=service.get_metadata(conversation),
    )


@router.post("/start", response_model=ChatResponse)
def start_feedback(
    payload: FeedbackStartRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatResponse:
    service = ChatService(db)
    try:
        conversation, assistant_message = service.start_feedback_session(
            current_user,
            payload.conversation_id,
            task_type=payload.task_type,
            prompt=payload.prompt,
            ai_output=payload.ai_output,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return ChatResponse(
        conversation_id=conversation.id,
        state=conversation.state,
        assistant_message=assistant_message,
        context=conversation.context or {},
        metadata=service.get_metadata(conversation),
    )


@router.post("/start-upload", response_model=ChatResponse)
def start_feedback_with_upload(
    conversation_id: int = Form(...),
    task_type: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    ai_output: str | None = Form(default=None),
    ai_output_file: UploadFile | None = File(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatResponse:
    service = ChatService(db)
    file_url = None
    if ai_output_file is not None and ai_output_file.filename:
        file_url = LocalFileStorage().save_chat_output(ai_output_file)

    try:
        conversation, assistant_message = service.start_feedback_session(
            current_user,
            conversation_id,
            task_type=task_type,
            prompt=prompt,
            ai_output=ai_output,
            ai_output_file_url=file_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return ChatResponse(
        conversation_id=conversation.id,
        state=conversation.state,
        assistant_message=assistant_message,
        context=conversation.context or {},
        metadata=service.get_metadata(conversation),
    )


@router.post("/send-upload", response_model=ChatResponse)
def send_message_with_upload(
    conversation_id: int = Form(...),
    message: str = Form(...),
    task_type: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    ai_output: str | None = Form(default=None),
    ai_output_file: UploadFile | None = File(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatResponse:
    service = ChatService(db)
    file_url = None
    if ai_output_file is not None and ai_output_file.filename:
        file_url = LocalFileStorage().save_chat_output(ai_output_file)

    try:
        conversation, assistant_message = service.process_message(
            current_user,
            conversation_id,
            message,
            task_type=task_type,
            prompt=prompt,
            ai_output=ai_output,
            ai_output_file_url=file_url,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return ChatResponse(
        conversation_id=conversation.id,
        state=conversation.state,
        assistant_message=assistant_message,
        context=conversation.context or {},
        metadata=service.get_metadata(conversation),
    )


@router.get("/history/{conversation_id}", response_model=ConversationHistoryResponse)
def get_history(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationHistoryResponse:
    service = ChatService(db)
    try:
        conversation = service.get_conversation(current_user, conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return ConversationHistoryResponse(
        conversation_id=conversation.id,
        state=conversation.state,
        messages=conversation.messages,
        context=conversation.context or {},
        metadata=service.get_metadata(conversation),
    )
