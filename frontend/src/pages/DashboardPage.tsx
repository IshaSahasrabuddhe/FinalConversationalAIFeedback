import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import {
  createConversation,
  getHistory,
  listConversations,
  sendMessage,
  sendMessageWithUpload,
  startFeedback,
  startFeedbackWithUpload,
} from "../api/chat";
import ChatWindow from "../components/ChatWindow";
import ConversationList from "../components/ConversationList";
import { useAuth } from "../context/AuthContext";
import type { ConversationHistory, ConversationMetadata, ConversationSummary, Message } from "../types";

const EMPTY_METADATA: ConversationMetadata = {
  task_type: "text",
  prompt: "",
  ai_output: "",
  ai_output_file_url: "",
  is_locked: false,
};

export default function DashboardPage() {
  const navigate = useNavigate();
  const { adminToken, logout } = useAuth();
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<number | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [metadata, setMetadata] = useState<ConversationMetadata>(EMPTY_METADATA);
  const [selectedOutputFile, setSelectedOutputFile] = useState<File | null>(null);
  const [currentState, setCurrentState] = useState<string>("START");
  const [loading, setLoading] = useState(false);
  const autoStartKeyRef = useRef("");

  useEffect(() => {
    void hydrate();
  }, []);

  useEffect(() => {
    if (!activeConversationId || loading || metadata.is_locked || currentState !== "START") {
      return;
    }

    const hasPrompt = Boolean(metadata.prompt.trim());
    const isTextTask = metadata.task_type === "text" || !metadata.task_type;
    const hasOutput = isTextTask ? Boolean(metadata.ai_output.trim()) : Boolean(selectedOutputFile);
    if (!hasPrompt || !hasOutput) {
      return;
    }

    const autoStartKey = [
      activeConversationId,
      metadata.task_type ?? "text",
      metadata.prompt,
      metadata.ai_output,
      selectedOutputFile?.name ?? "",
      selectedOutputFile?.size ?? 0,
    ].join("|");
    if (autoStartKeyRef.current === autoStartKey) {
      return;
    }
    autoStartKeyRef.current = autoStartKey;
    void handleStartFeedback();
  }, [activeConversationId, currentState, loading, metadata, selectedOutputFile]);

  async function hydrate() {
    setLoading(true);
    try {
      const conversationList = await listConversations();
      setConversations(conversationList);

      if (conversationList.length) {
        await selectConversation(conversationList[0].id);
      } else {
        setMessages([]);
        setMetadata(EMPTY_METADATA);
        setSelectedOutputFile(null);
      }
    } catch {
      logout();
    } finally {
      setLoading(false);
    }
  }

  async function handleCreateConversation() {
    setLoading(true);
    try {
      const created = await createConversation();
      const updatedConversations = await listConversations();
      setConversations(updatedConversations);
      setActiveConversationId(created.conversation_id);
      const history = await getHistory(created.conversation_id);
      applyHistory(history);
    } finally {
      setLoading(false);
    }
  }

  async function selectConversation(conversationId: number) {
    setLoading(true);
    try {
      const history = await getHistory(conversationId);
      setActiveConversationId(conversationId);
      applyHistory(history);
    } finally {
      setLoading(false);
    }
  }

  function applyHistory(history: ConversationHistory) {
    setMessages(history.messages);
    setCurrentState(history.state);
    setMetadata(history.metadata ?? EMPTY_METADATA);
    setSelectedOutputFile(null);
  }

  async function handleSend() {
    if (!activeConversationId || !draft.trim()) {
      return;
    }

    const userDraft = draft.trim();
    const payloadMetadata = metadata.is_locked ? undefined : metadata;
    const needsUploadRoute = !metadata.is_locked && metadata.task_type !== "text";
    setDraft("");

    const optimisticMessage: Message = {
      id: Date.now(),
      role: "user",
      content: userDraft,
      timestamp: new Date().toISOString(),
    };
    setMessages((current) => [...current, optimisticMessage]);
    setLoading(true);

    try {
      const response =
        needsUploadRoute && payloadMetadata
          ? await sendMessageWithUpload(activeConversationId, userDraft, payloadMetadata, selectedOutputFile)
          : await sendMessage(activeConversationId, userDraft, payloadMetadata);
      const history = await getHistory(response.conversation_id);
      applyHistory(history);
      setCurrentState(response.state);
      setConversations(await listConversations());
    } finally {
      setLoading(false);
    }
  }

  async function handleStartFeedback() {
    if (!activeConversationId || metadata.is_locked) {
      return;
    }

    const needsUploadRoute = metadata.task_type !== "text" && Boolean(selectedOutputFile);
    setLoading(true);

    try {
      const response = needsUploadRoute
        ? await startFeedbackWithUpload(activeConversationId, metadata, selectedOutputFile)
        : await startFeedback(activeConversationId, metadata);
      const history = await getHistory(response.conversation_id);
      applyHistory(history);
      setCurrentState(response.state);
      setConversations(await listConversations());
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="flex h-screen min-h-screen w-full overflow-hidden bg-[#081A35] text-slate-100">
      <div className="flex h-full min-h-0 w-full flex-1 flex-col overflow-hidden">
        <header className="flex shrink-0 items-center justify-between border-b border-cyan-300/10 bg-[#0A1A35] px-4 py-3 sm:px-6">
          <div>
            <p className="text-sm uppercase tracking-[0.2em] text-cyan-300">AI Feedback App</p>
            <h1 className="mt-1 text-lg font-semibold text-white">Conversational Feedback Dashboard</h1>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => navigate(adminToken ? "/admin/dashboard" : "/login?mode=admin")}
              className="rounded-xl border border-cyan-300/20 bg-cyan-300/10 px-3 py-2 text-base font-medium text-cyan-100 transition hover:bg-cyan-300/20"
            >
              Admin Dashboard
            </button>
            <button
              onClick={logout}
              className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-base font-medium text-slate-200 transition hover:bg-white/10"
            >
              Log out
            </button>
          </div>
        </header>

        <div className="grid h-full min-h-0 w-full flex-1 overflow-hidden lg:grid-cols-[300px,minmax(0,1fr)]">
          <ConversationList
            conversations={conversations}
            activeConversationId={activeConversationId}
            onCreateConversation={() => void handleCreateConversation()}
            onSelectConversation={(conversationId) => void selectConversation(conversationId)}
          />
          <ChatWindow
            messages={messages}
            draft={draft}
            metadata={metadata}
            selectedOutputFile={selectedOutputFile}
            onDraftChange={setDraft}
            onMetadataChange={setMetadata}
            onOutputFileChange={setSelectedOutputFile}
            onSend={() => void handleSend()}
            disabled={loading || !activeConversationId}
            currentState={currentState}
          />
        </div>
      </div>
    </main>
  );
}
