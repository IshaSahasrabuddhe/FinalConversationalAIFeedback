import api from "./client";
import type {
  ChatResponse,
  ConversationHistory,
  ConversationSummary,
  CreateConversationResponse,
  TaskType,
} from "../types";

export async function listConversations(): Promise<ConversationSummary[]> {
  const response = await api.get<ConversationSummary[]>("/chat/conversations");
  return response.data;
}

export async function createConversation(): Promise<CreateConversationResponse> {
  const response = await api.post<CreateConversationResponse>("/chat/conversations");
  return response.data;
}

export async function getHistory(conversationId: number): Promise<ConversationHistory> {
  const response = await api.get<ConversationHistory>(`/chat/history/${conversationId}`);
  return response.data;
}

export async function sendMessage(
  conversationId: number,
  message: string,
  metadata?: { task_type: TaskType | null; prompt: string; ai_output: string }
): Promise<ChatResponse> {
  const response = await api.post<ChatResponse>("/chat/send", {
    conversation_id: conversationId,
    message,
    task_type: metadata?.task_type ?? undefined,
    prompt: metadata?.prompt ?? undefined,
    ai_output: metadata?.ai_output ?? undefined,
  });
  return response.data;
}

export async function startFeedback(
  conversationId: number,
  metadata: { task_type: TaskType | null; prompt: string; ai_output: string }
): Promise<ChatResponse> {
  const response = await api.post<ChatResponse>("/chat/start", {
    conversation_id: conversationId,
    task_type: metadata.task_type ?? undefined,
    prompt: metadata.prompt || undefined,
    ai_output: metadata.ai_output || undefined,
  });
  return response.data;
}

export async function startFeedbackWithUpload(
  conversationId: number,
  metadata: { task_type: TaskType | null; prompt: string; ai_output: string },
  file?: File | null
): Promise<ChatResponse> {
  const formData = new FormData();
  formData.set("conversation_id", String(conversationId));
  if (metadata.task_type) {
    formData.set("task_type", metadata.task_type);
  }
  if (metadata.prompt) {
    formData.set("prompt", metadata.prompt);
  }
  if (metadata.ai_output) {
    formData.set("ai_output", metadata.ai_output);
  }
  if (file) {
    formData.set("ai_output_file", file);
  }

  const response = await api.post<ChatResponse>("/chat/start-upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
}

export async function sendMessageWithUpload(
  conversationId: number,
  message: string,
  metadata: { task_type: TaskType | null; prompt: string; ai_output: string },
  file?: File | null
): Promise<ChatResponse> {
  const formData = new FormData();
  formData.set("conversation_id", String(conversationId));
  formData.set("message", message);
  if (metadata.task_type) {
    formData.set("task_type", metadata.task_type);
  }
  if (metadata.prompt) {
    formData.set("prompt", metadata.prompt);
  }
  if (metadata.ai_output) {
    formData.set("ai_output", metadata.ai_output);
  }
  if (file) {
    formData.set("ai_output_file", file);
  }

  const response = await api.post<ChatResponse>("/chat/send-upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
}
