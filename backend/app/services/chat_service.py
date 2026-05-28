from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.conversation import Conversation
from app.models.enums import ConversationState, MessageRole
from app.models.feedback import Feedback
from app.models.message import Message
from app.models.user import User
from app.schemas.chat import ConversationMetadata
from app.schemas.llm import ConversationTurnAnalysis, FeedbackExtraction, RatingExtraction
from app.services.chains.llm_service import FeedbackLLMService

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.llm_service = FeedbackLLMService()

    def create_conversation(self, user: User) -> tuple[Conversation, str]:
        conversation = Conversation(
            user_id=user.id,
            state=ConversationState.START,
            context=self._default_context(),
        )
        self.db.add(conversation)
        self.db.flush()
        assistant_message = self._advance_without_user_message(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation, assistant_message

    def list_conversations(self, user: User) -> list[Conversation]:
        return list(
            self.db.scalars(
                select(Conversation)
                .where(Conversation.user_id == user.id)
                .order_by(Conversation.created_at.desc())
            )
        )

    def get_conversation(self, user: User, conversation_id: int) -> Conversation:
        conversation = self.db.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user.id,
            )
        )
        if not conversation:
            raise ValueError("Conversation not found")
        return conversation

    def get_metadata(self, conversation: Conversation) -> ConversationMetadata:
        return ConversationMetadata(
            task_type=conversation.task_type,
            prompt=conversation.prompt or "",
            ai_output=conversation.ai_output or "",
            ai_output_file_url=conversation.ai_output_file_url or "",
            is_locked=self._metadata_locked(conversation) or any(message.role == MessageRole.USER for message in conversation.messages),
        )

    def process_message(
        self,
        user: User,
        conversation_id: int,
        content: str,
        *,
        task_type: str | None = None,
        prompt: str | None = None,
        ai_output: str | None = None,
        ai_output_file_url: str | None = None,
    ) -> tuple[Conversation, str]:
        conversation = self.get_conversation(user, conversation_id)
        cleaned_content = content.strip()

        if not cleaned_content:
            assistant_message = self._respond(
                conversation,
                "I am still here. Share anything that stood out, even if it is just one quick detail.",
            )
            self.db.commit()
            self.db.refresh(conversation)
            return conversation, assistant_message

        self._store_metadata_on_first_message(
            conversation,
            task_type=task_type,
            prompt=prompt,
            ai_output=ai_output,
            ai_output_file_url=ai_output_file_url,
        )

        self.db.add(
            Message(
                conversation_id=conversation.id,
                role=MessageRole.USER,
                content=cleaned_content,
            )
        )

        assistant_message = self._advance_state(conversation, cleaned_content)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation, assistant_message

    def _advance_without_user_message(self, conversation: Conversation) -> str:
        if conversation.state == ConversationState.START:
            conversation.state = ConversationState.ASK_FEEDBACK
            assistant_message = "I would love to hear how the AI experience went. What worked well, and what did not?"
            self._add_assistant_message(conversation, assistant_message)
            return assistant_message
        return "How can I help with your feedback?"

    def _advance_state(self, conversation: Conversation, user_input: str) -> str:
        previous_state = conversation.state
        self._log_state("before", conversation, user_input)

        if previous_state == ConversationState.END and not self._is_user_declining(user_input):
            conversation.state = ConversationState.FEEDBACK_CONTINUE

        if self._is_user_declining(user_input):
            conversation.state = ConversationState.END
            self._log_state("end", conversation, user_input)
            return self._respond(
                conversation,
                "Understood. I will pause this feedback for now. Let me know if you would like to add anything later.",
            )

        if self._is_confirmation_signal(user_input) and self._has_feedback_context(conversation):
            self._mark_active_thread_completed_if_present(conversation)
            conversation.state = ConversationState.FEEDBACK_CONTINUE
            self._log_state("confirmation", conversation, user_input)
            return self._respond(conversation, self._build_confirmation_continue_response(conversation))

        if previous_state == ConversationState.START:
            return self._advance_without_user_message(conversation)

        if self._is_off_topic_question(user_input):
            conversation.state = ConversationState.FEEDBACK_CONTINUE
            self._log_state("off_topic", conversation, user_input)
            return self._respond(conversation, self._build_off_track_response(conversation, question=True))

        if self._is_vague_acknowledgement(user_input):
            conversation.state = ConversationState.FEEDBACK_CONTINUE
            self._log_state("vague_ack", conversation, user_input)
            return self._respond(conversation, self._build_off_track_response(conversation, question=False))

        rating_already_captured = self._get_context_value(conversation, "rating") is not None
        conversation.state = ConversationState.PRE_FEEDBACK_ANALYSIS
        analysis = self.llm_service.analyze_turn(user_input)

        if analysis.is_feedback_present:
            reply = self._handle_feedback_turn(conversation, user_input, analysis, previous_state)
            self._log_state("after_feedback", conversation, user_input)
            return reply

        if not rating_already_captured:
            rating_result = self.llm_service.extract_rating(user_input)
            if rating_result.rating is not None or rating_result.is_vague:
                reply = self._handle_rating_turn(conversation, rating_result)
                self._log_state("after_rating", conversation, user_input)
                return reply

        if previous_state == ConversationState.ISSUE_HANDLING:
            conversation.state = ConversationState.FEEDBACK_CONTINUE
            reply = self._build_off_track_response(conversation, question=False)
            self._log_state("off_track", conversation, user_input)
            return self._respond(conversation, reply)

        conversation.state = ConversationState.FEEDBACK_CONTINUE
        self._log_state("redirect", conversation, user_input)
        return self._respond(conversation, self._build_off_track_response(conversation, question=False))

    def _handle_feedback_turn(
        self,
        conversation: Conversation,
        user_input: str,
        analysis: ConversationTurnAnalysis,
        previous_state: ConversationState,
    ) -> str:
        extracted = self.llm_service.extract_feedback(user_input)
        issue = self.llm_service.classify_issue(user_input)
        positive_only = self._is_positive_only_feedback(extracted)
        issue_type_for_storage = "none" if positive_only else issue.issue_type
        rating_already_captured = self._get_context_value(conversation, "rating") is not None

        if not rating_already_captured:
            embedded_rating = self._extract_embedded_rating_if_present(user_input)
            if embedded_rating is not None:
                self._set_rating(conversation, embedded_rating)

        if self._has_extractable_feedback(extracted):
            conversation.state = ConversationState.STORE_FEEDBACK
            self._store_feedback_entry(
                conversation=conversation,
                user_input=user_input,
                extracted=extracted,
                sentiment=extracted.sentiment,
                issue_type=issue_type_for_storage,
            )

        self._set_context_value(conversation, "active_issue_type", issue_type_for_storage)
        self._set_context_value(conversation, "last_issue_tags", extracted.issue_tags)
        self._set_context_value(conversation, "last_sentiment", extracted.sentiment)

        if positive_only:
            self._reset_off_track_count(conversation)
            conversation.state = ConversationState.FEEDBACK_CONTINUE
            return self._respond(conversation, self._build_positive_feedback_response(conversation, extracted))

        self._update_grounding_memory(conversation, user_input, extracted)
        self._reset_off_track_count(conversation)

        issue_summary = self._build_contextual_issue_summary(
            conversation=conversation,
            user_input=user_input,
            extracted=extracted,
            issue_type=issue_type_for_storage,
        )
        if issue_summary and self._should_complete_capture_after_summary(extracted, issue_type_for_storage):
            if self._get_context_value(conversation, "rating") is None:
                conversation.state = ConversationState.ASK_RATING
                self._set_context_value(conversation, "rating_prompt_asked", True)
            else:
                conversation.state = ConversationState.FEEDBACK_CONTINUE
            return self._respond(
                conversation,
                self._build_captured_issue_response(conversation, issue_summary),
            )

        completion_reply = self._maybe_complete_active_thread(
            conversation=conversation,
            user_input=user_input,
            extracted=extracted,
            previous_state=previous_state,
        )
        if completion_reply:
            conversation.state = ConversationState.FEEDBACK_CONTINUE
            return self._respond(conversation, completion_reply)

        contextual_follow_up = self._maybe_build_contextual_follow_up(
            conversation=conversation,
            extracted=extracted,
            issue_type=issue_type_for_storage,
            previous_state=previous_state,
        )

        if contextual_follow_up:
            conversation.state = ConversationState.ISSUE_HANDLING
            base_reply = contextual_follow_up
        elif self._get_context_value(conversation, "rating") is None:
            if self._get_context_value(conversation, "rating_prompt_asked", False):
                conversation.state = ConversationState.FEEDBACK_CONTINUE
                base_reply = self._build_progressive_acknowledgement(conversation)
            else:
                conversation.state = ConversationState.ASK_RATING
                self._set_context_value(conversation, "rating_prompt_asked", True)
                base_reply = self._build_rating_prompt()
        else:
            conversation.state = ConversationState.FEEDBACK_CONTINUE
            base_reply = self._build_continue_prompt(conversation, extracted)

        if issue_summary:
            base_reply = self._combine_issue_summary_with_reply(issue_summary, base_reply)

        if (
            conversation.state != ConversationState.ASK_RATING
            and "?" in base_reply
            and not self._question_has_high_value(
                conversation=conversation,
                question=base_reply,
                extracted=extracted,
                user_feedback=user_input,
                issue_type=issue_type_for_storage,
            )
        ):
            base_reply = self._build_progressive_acknowledgement(conversation)

        reply = self._append_human_followup_if_needed(
            conversation=conversation,
            base_reply=base_reply,
            user_feedback=user_input,
            extracted=extracted,
            issue_type=issue_type_for_storage,
        )
        return self._respond(conversation, reply)

    def _handle_rating_turn(self, conversation: Conversation, rating_result: RatingExtraction) -> str:
        if self._get_context_value(conversation, "rating") is not None:
            conversation.state = ConversationState.FEEDBACK_CONTINUE
            return self._respond(conversation, "I already have your rating. If there is more feedback, I can add it.")

        if rating_result.is_vague or rating_result.rating is None:
            conversation.state = ConversationState.HANDLE_VAGUE_RATING
            return self._respond(conversation, rating_result.clarification_needed)

        self._set_rating(conversation, rating_result.rating)

        if self._feedback_count(conversation) > 0:
            conversation.state = ConversationState.FEEDBACK_CONTINUE
            return self._respond(conversation, self._build_post_rating_response(conversation, rating_result.rating))

        conversation.state = ConversationState.FEEDBACK_CONTINUE
        return self._respond(conversation, self._build_post_rating_prompt(rating_result.rating))

    def _set_rating(self, conversation: Conversation, rating: int) -> None:
        self._set_context_value(conversation, "rating", rating)
        self._update_feedback_ratings(conversation, rating)

    def _update_feedback_ratings(self, conversation: Conversation, rating: int) -> None:
        feedback = self._get_or_create_feedback_entry(conversation)
        feedback.rating = rating
        feedback.summary = self._build_feedback_summary(feedback)

    def _store_feedback_entry(
        self,
        conversation: Conversation,
        user_input: str,
        extracted: FeedbackExtraction,
        sentiment: str,
        issue_type: str,
    ) -> None:
        feedback = self._get_or_create_feedback_entry(conversation)
        feedback.rating = self._get_context_value(conversation, "rating")
        feedback.sentiment = self._resolve_conversation_sentiment(conversation, sentiment)
        feedback.positives = self._merge_lists(feedback.positives, extracted.positives)
        feedback.negatives = self._merge_lists(feedback.negatives, extracted.negatives)
        feedback.suggestions = self._merge_lists(feedback.suggestions, extracted.suggestions)
        feedback.issue_tags = self._merge_lists(feedback.issue_tags, extracted.issue_tags)
        feedback.raw_text = self._merge_raw_text(feedback.raw_text, user_input)
        feedback.summary = self._build_feedback_summary(feedback)
        self._append_feedback_memory(conversation, extracted, sentiment, issue_type)
        feedback.issue_type = self._resolve_issue_type(conversation)

    def _append_feedback_memory(
        self,
        conversation: Conversation,
        extracted: FeedbackExtraction,
        sentiment: str,
        issue_type: str,
    ) -> None:
        context = self._conversation_context(conversation)
        memory = context.setdefault(
            "feedback_memory",
            {
                "positives": [],
                "negatives": [],
                "suggestions": [],
                "issue_types": [],
                "issue_tags": [],
                "sentiments": [],
                "entry_count": 0,
                "follow_ups_asked": [],
                "human_followups_asked": [],
                "stated_preferences": [],
                "contextual_mismatches": [],
                "unresolved_mismatches": [],
                "active_thread": "",
                "completed_threads": [],
                "thread_evidence": [],
                "thread_turn_counts": {},
                "repeated_completed_thread": "",
                "invalidated_threads": [],
                "latest_correction": "",
                "current_domain": "",
                "lightweight_continuations_used": 0,
                "off_track_count": 0,
            },
        )

        memory["positives"] = self._merge_lists(memory.get("positives"), extracted.positives)
        memory["negatives"] = self._merge_lists(memory.get("negatives"), extracted.negatives)
        memory["suggestions"] = self._merge_lists(memory.get("suggestions"), extracted.suggestions)
        memory["issue_tags"] = self._merge_lists(memory.get("issue_tags"), extracted.issue_tags)
        if issue_type != "none":
            memory["issue_types"].append(issue_type)
        memory["sentiments"].append(sentiment)
        memory["entry_count"] += 1
        memory.setdefault("completed_threads", [])
        memory.setdefault("thread_evidence", [])
        memory.setdefault("thread_turn_counts", {})
        memory.setdefault("repeated_completed_thread", "")
        memory.setdefault("invalidated_threads", [])
        memory.setdefault("latest_correction", "")

        context["feedback_captured"] = memory["entry_count"] > 0
        context["issue_type"] = self._resolve_issue_type_from_memory(memory)
        context["sentiment"] = self._resolve_sentiment_from_memory(memory)
        conversation.context = context

    def _conversation_context(self, conversation: Conversation) -> dict:
        context = dict(conversation.context or {})
        default_context = self._default_context()
        merged = {**default_context, **context}
        merged["feedback_memory"] = {**default_context["feedback_memory"], **context.get("feedback_memory", {})}
        return merged

    def _set_context_value(self, conversation: Conversation, key: str, value) -> None:
        context = self._conversation_context(conversation)
        context[key] = value
        conversation.context = context

    def _get_context_value(self, conversation: Conversation, key: str, default=None):
        return self._conversation_context(conversation).get(key, default)

    def _default_context(self) -> dict:
        return {
            "rating": None,
            "active_issue_type": "none",
            "last_issue_tags": [],
            "last_sentiment": "mixed",
            "feedback_captured": False,
            "metadata_locked": False,
            "rating_prompt_asked": False,
            "feedback_memory": {
                "positives": [],
                "negatives": [],
                "suggestions": [],
                "issue_types": [],
                "issue_tags": [],
                "sentiments": [],
                "entry_count": 0,
                "follow_ups_asked": [],
                "human_followups_asked": [],
                "stated_preferences": [],
                "contextual_mismatches": [],
                "unresolved_mismatches": [],
                "active_thread": "",
                "completed_threads": [],
                "thread_evidence": [],
                "thread_turn_counts": {},
                "repeated_completed_thread": "",
                "invalidated_threads": [],
                "latest_correction": "",
                "current_domain": "",
                "lightweight_continuations_used": 0,
                "off_track_count": 0,
            },
        }

    def _update_grounding_memory(
        self,
        conversation: Conversation,
        user_input: str,
        extracted: FeedbackExtraction,
    ) -> None:
        context = self._conversation_context(conversation)
        memory = context.setdefault("feedback_memory", self._default_context()["feedback_memory"])
        memory.setdefault("completed_threads", [])
        memory.setdefault("thread_evidence", [])
        memory.setdefault("thread_turn_counts", {})
        memory["repeated_completed_thread"] = ""
        memory.setdefault("invalidated_threads", [])
        memory.setdefault("latest_correction", "")
        memory.setdefault("current_domain", "")
        memory.setdefault("lightweight_continuations_used", 0)
        memory.setdefault("off_track_count", 0)

        current_domain = self._detect_conversation_domain(user_input)
        if not current_domain:
            current_domain = self._detect_conversation_domain(" ".join([conversation.prompt or "", conversation.ai_output or ""]))
        if current_domain:
            self._reset_stale_thread_context_if_domain_changed(memory, current_domain)

        correction = self._detect_user_correction(user_input)
        if correction:
            self._apply_user_correction_to_memory(memory, correction)

        intent_traits = self._extract_context_traits(" ".join([conversation.prompt or "", user_input]))
        output_traits = self._extract_context_traits(conversation.ai_output or "")
        stated_preferences = self._format_traits(intent_traits)
        if stated_preferences:
            memory["stated_preferences"] = self._merge_limited(
                memory.get("stated_preferences", []),
                stated_preferences,
                limit=8,
            )

        mismatches = self._detect_context_mismatches(intent_traits, output_traits, extracted)
        if mismatches:
            open_mismatches = [item for item in mismatches if item not in memory.get("completed_threads", [])]
            memory["contextual_mismatches"] = self._merge_limited(
                memory.get("contextual_mismatches", []),
                mismatches,
                limit=6,
            )
            if open_mismatches:
                memory["unresolved_mismatches"] = self._merge_limited(
                    memory.get("unresolved_mismatches", []),
                    open_mismatches,
                    limit=4,
                )
                memory["active_thread"] = open_mismatches[-1]
            else:
                memory["repeated_completed_thread"] = mismatches[-1]
        elif self._is_dissatisfaction_about_active_thread(user_input, memory):
            memory["active_thread"] = list(memory.get("unresolved_mismatches", []))[-1]

        active_thread = memory.get("active_thread")
        if active_thread:
            thread_counts = dict(memory.get("thread_turn_counts", {}))
            thread_counts[active_thread] = int(thread_counts.get(active_thread, 0)) + 1
            memory["thread_turn_counts"] = thread_counts

        issue_terms = self._extract_issue_terms(user_input, extracted)
        if issue_terms:
            memory["thread_evidence"] = self._merge_limited(
                memory.get("thread_evidence", []),
                issue_terms,
                limit=16,
            )

        context["feedback_memory"] = memory
        conversation.context = context

    def _detect_conversation_domain(self, text: str) -> str:
        normalized = text.lower()
        domain_tokens = {
            "environmental_aerial_realism": {
                "island",
                "tropical",
                "aerial",
                "drone",
                "water",
                "ocean",
                "coast",
                "coastline",
                "environmental",
                "scale",
                "perspective",
                "terrain",
                "texture",
                "textures",
            },
            "technical_product_realism": {
                "watch",
                "product",
                "luxury",
                "advertisement",
                "metal",
                "metallic",
                "material",
                "materials",
                "reflection",
                "reflections",
                "sharpness",
                "proportion",
                "proportions",
            },
            "emotional_scene_realism": {
                "cafe",
                "cinematic",
                "movie",
                "character",
                "woman",
                "emotion",
                "emotional",
                "warm",
                "cozy",
                "atmosphere",
                "immersive",
                "alive",
            },
        }

        best_domain = ""
        best_score = 0
        for domain, tokens in domain_tokens.items():
            score = sum(1 for token in tokens if token in normalized)
            if score > best_score:
                best_domain = domain
                best_score = score
        return best_domain if best_score >= 2 else ""

    def _reset_stale_thread_context_if_domain_changed(self, memory: dict, current_domain: str) -> None:
        previous_domain = str(memory.get("current_domain") or "")
        if previous_domain == current_domain:
            return

        memory["current_domain"] = current_domain
        if not previous_domain:
            return

        memory["active_thread"] = ""
        memory["unresolved_mismatches"] = []
        memory["completed_threads"] = []
        memory["contextual_mismatches"] = []
        memory["thread_evidence"] = []
        memory["thread_turn_counts"] = {}
        memory["repeated_completed_thread"] = ""
        memory["invalidated_threads"] = []
        memory["latest_correction"] = ""
        memory["human_followups_asked"] = []
        memory["follow_ups_asked"] = []
        memory["stated_preferences"] = []

    def _build_grounding_context(
        self,
        conversation: Conversation,
        user_feedback: str,
        extracted: FeedbackExtraction,
    ) -> str:
        prompt = conversation.prompt or ""
        ai_output = conversation.ai_output or ""
        if not prompt and not ai_output:
            return ""

        context = self._conversation_context(conversation)
        memory = context.get("feedback_memory", {})
        intent_traits = self._extract_context_traits(" ".join([prompt, user_feedback]))
        output_traits = self._extract_context_traits(ai_output)
        current_mismatches = self._detect_context_mismatches(intent_traits, output_traits, extracted)
        stored_mismatches = list(memory.get("contextual_mismatches", []))

        sections: list[str] = []
        if intent_traits:
            sections.append(f"intent_traits={self._format_traits(intent_traits)}")
        if output_traits:
            sections.append(f"output_traits={self._format_traits(output_traits)}")

        mismatches = self._merge_limited(stored_mismatches, current_mismatches, limit=4)
        if mismatches:
            sections.append(f"likely_mismatches={mismatches}")

        unresolved = list(memory.get("unresolved_mismatches", []))[-4:]
        if unresolved:
            sections.append(f"unresolved_mismatches={unresolved}")

        active_thread = memory.get("active_thread")
        if active_thread:
            sections.append(f"active_thread={active_thread}")

        latest_correction = memory.get("latest_correction")
        if latest_correction:
            sections.append(f"latest_user_correction={latest_correction}")

        current_domain = memory.get("current_domain")
        if current_domain:
            sections.append(f"current_domain={current_domain}")

        invalidated = list(memory.get("invalidated_threads", []))[-4:]
        if invalidated:
            sections.append(f"do_not_reuse_invalidated_threads={invalidated}")

        preferences = list(memory.get("stated_preferences", []))[-4:]
        if preferences:
            sections.append(f"already_stated_preferences={preferences}")

        return " | ".join(sections)

    def _extract_context_traits(self, text: str) -> dict[str, list[str]]:
        normalized = text.lower()
        trait_map = {
            "style": [
                (("realistic", "photorealistic", "photo-realistic", "realism"), "realistic"),
                (("cinematic", "film still", "movie-like"), "cinematic"),
                (("anime", "cartoon", "cartoonish", "animated"), "anime/cartoon"),
                (("illustration", "sketch", "drawing"), "illustrative"),
            ],
            "lighting": [
                (("natural lighting", "natural light"), "natural lighting"),
                (("soft lighting", "soft light"), "soft lighting"),
                (("flat lighting", "flat light"), "flat lighting"),
                (("harsh lighting", "harsh light"), "harsh lighting"),
                (("overexposed", "too bright"), "overexposed"),
                (("dark", "underexposed"), "too dark"),
            ],
            "tone": [
                (("serious", "professional", "sober"), "serious"),
                (("playful", "flashy", "exaggerated"), "playful/exaggerated"),
            ],
            "emotional_tone": [
                (("warm", "cozy", "cosy", "alive", "emotional warmth"), "warm/cozy/alive"),
                (("cinematic", "movie", "film", "real cafe"), "cinematic realism"),
                (("flat", "cold", "lifeless", "not emotionally warm", "not warm", "not cozy", "missed the vibe"), "emotionally flat"),
            ],
            "quality": [
                (("detailed", "sharp", "high quality"), "high detail"),
                (("blurry", "low quality", "distorted", "unrealistic"), "low quality"),
            ],
            "composition": [
                (("portrait", "close-up", "close up"), "portrait"),
                (("full body", "wide shot"), "wide composition"),
                (("centered", "symmetrical"), "centered"),
            ],
            "color": [
                (("warm", "golden"), "warm color"),
                (("cool", "blue tone"), "cool color"),
                (("muted", "subtle"), "muted color"),
                (("vibrant", "saturated"), "vibrant color"),
            ],
        }

        traits: dict[str, list[str]] = {}
        for category, patterns in trait_map.items():
            for keywords, label in patterns:
                if any(keyword in normalized for keyword in keywords):
                    traits.setdefault(category, [])
                    if label not in traits[category]:
                        traits[category].append(label)
        return traits

    def _detect_context_mismatches(
        self,
        intent_traits: dict[str, list[str]],
        output_traits: dict[str, list[str]],
        extracted: FeedbackExtraction,
    ) -> list[str]:
        if not intent_traits:
            return []

        feedback_text = " ".join(extracted.negatives + extracted.suggestions + extracted.issue_tags).lower()
        mismatches: list[str] = []
        for category, expected_values in intent_traits.items():
            observed_values = output_traits.get(category, [])
            if not observed_values and category == "emotional_tone":
                observed_values = self._infer_emotional_observed_values(feedback_text)
            if not observed_values:
                continue

            expected = expected_values[0]
            observed = observed_values[0]
            if expected == observed:
                continue

            if self._is_meaningful_mismatch(category, expected, observed, feedback_text):
                mismatches.append(f"{category}: wanted {expected}, got {observed}")

        return mismatches[:4]

    def _is_meaningful_mismatch(self, category: str, expected: str, observed: str, feedback_text: str) -> bool:
        if expected in feedback_text or observed in feedback_text:
            return True
        if category == "style":
            return ("realistic" in expected or "cinematic" in expected) and "anime/cartoon" in observed
        if category == "lighting":
            return expected in {"natural lighting", "soft lighting"} and observed in {
                "flat lighting",
                "harsh lighting",
                "overexposed",
                "too dark",
            }
        if category == "tone":
            return expected == "serious" and observed == "playful/exaggerated"
        if category == "emotional_tone":
            return observed == "emotionally flat" and expected in {"warm/cozy/alive", "cinematic realism"}
        if category == "quality":
            return expected == "high detail" and observed == "low quality"
        if category == "color":
            return expected != observed
        return False

    def _infer_emotional_observed_values(self, feedback_text: str) -> list[str]:
        if any(
            token in feedback_text
            for token in {
                "missed the vibe",
                "not emotionally warm",
                "not warm",
                "not cozy",
                "not cosy",
                "felt flat",
                "emotionally it missed",
                "lacked warmth",
            }
        ):
            return ["emotionally flat"]
        return []

    def _is_dissatisfaction_about_active_thread(self, user_input: str, memory: dict) -> bool:
        if not memory.get("unresolved_mismatches"):
            return False
        normalized = user_input.lower()
        return any(
            token in normalized
            for token in {
                "missed",
                "not",
                "flat",
                "wanted",
                "because",
                "but",
                "vibe",
                "warm",
                "cozy",
                "cinematic",
                "alive",
                "atmosphere",
            }
        )

    def _detect_user_correction(self, user_input: str) -> dict[str, str] | None:
        normalized = user_input.lower()
        correction_markers = {
            "actually",
            "rather than",
            "instead",
            "not emotional",
            "not about emotion",
            "not atmosphere",
            "more technical",
            "technical in this case",
            "product quality",
        }
        if not any(marker in normalized for marker in correction_markers):
            return None

        if any(
            marker in normalized
            for marker in {
                "technical",
                "sharpness",
                "blurry",
                "material",
                "materials",
                "reflection",
                "reflections",
                "proportion",
                "proportions",
                "product quality",
                "rendering quality",
                "luxury product",
            }
        ):
            return {
                "invalidates": "emotional_tone",
                "replacement": "technical_product_realism",
                "summary": "technical realism and product rendering quality",
            }

        if any(marker in normalized for marker in {"emotional", "atmosphere", "mood", "vibe", "warmth"}):
            return {
                "invalidates": "technical_product_realism",
                "replacement": "emotional_tone",
                "summary": "emotional tone and atmosphere",
            }

        return {"invalidates": "", "replacement": "", "summary": ""}

    def _apply_user_correction_to_memory(self, memory: dict, correction: dict[str, str]) -> None:
        invalidates = correction.get("invalidates", "")
        replacement = correction.get("replacement", "")
        summary = correction.get("summary", "")

        invalidated_threads = list(memory.get("invalidated_threads", []))
        active_thread = str(memory.get("active_thread") or "")
        completed_threads = list(memory.get("completed_threads", []))
        unresolved = list(memory.get("unresolved_mismatches", []))

        def is_invalidated(thread: str) -> bool:
            return bool(invalidates and invalidates in thread)

        stale_threads = [thread for thread in [active_thread, *completed_threads, *unresolved] if is_invalidated(thread)]
        memory["invalidated_threads"] = self._merge_limited(invalidated_threads, stale_threads, limit=8)
        memory["completed_threads"] = [thread for thread in completed_threads if not is_invalidated(thread)]
        memory["unresolved_mismatches"] = [thread for thread in unresolved if not is_invalidated(thread)]
        if is_invalidated(active_thread):
            memory["active_thread"] = ""

        if replacement:
            replacement_thread = f"{replacement}: {summary}"
            memory["active_thread"] = replacement_thread
            memory["unresolved_mismatches"] = self._merge_limited(
                memory.get("unresolved_mismatches", []),
                [replacement_thread],
                limit=4,
            )

        memory["latest_correction"] = summary or replacement

    def _maybe_complete_active_thread(
        self,
        *,
        conversation: Conversation,
        user_input: str,
        extracted: FeedbackExtraction,
        previous_state: ConversationState,
    ) -> str | None:
        memory = self._conversation_context(conversation).get("feedback_memory", {})
        active_thread = str(memory.get("active_thread") or "")
        if not active_thread:
            repeated_completed_thread = str(memory.get("repeated_completed_thread") or "")
            if repeated_completed_thread and self._get_context_value(conversation, "rating") is not None:
                self._clear_repeated_completed_thread(conversation)
                return self._build_thread_summary_response(repeated_completed_thread)
            return None

        if self._get_context_value(conversation, "rating") is None and previous_state != ConversationState.ISSUE_HANDLING:
            return None

        if active_thread in memory.get("completed_threads", []):
            self._mark_thread_completed(conversation, active_thread)
            return self._build_thread_summary_response(active_thread)

        completion_signal = self._is_completion_signal(user_input)
        saturated = self._is_thread_saturated(active_thread, user_input, extracted, memory)
        if not completion_signal and not saturated:
            return None

        self._mark_thread_completed(conversation, active_thread)
        return self._build_thread_summary_response(active_thread)

    def _is_completion_signal(self, user_input: str) -> bool:
        normalized = user_input.strip().lower().strip(".!")
        return normalized in {
            "yes",
            "yeah",
            "yep",
            "exactly",
            "correct",
            "right",
            "that's it",
            "thats it",
            "that is it",
            "i am done",
            "i'm done",
            "done",
            "nothing else",
            "no more",
        }

    def _is_off_topic_question(self, user_input: str) -> bool:
        raw = user_input.strip().lower()
        is_question = raw.endswith("?")
        normalized = raw.strip("?.! ")
        if not normalized:
            return False
        off_topic_patterns = {
            "what is your name",
            "whats your name",
            "what's your name",
            "who are you",
            "how are you",
            "are you an ai",
            "are you ai",
            "who made you",
            "what can you do",
        }
        if normalized in off_topic_patterns:
            return True
        return is_question and not any(
            token in normalized
            for token in {
                "rate",
                "rating",
                "feedback",
                "image",
                "output",
                "result",
                "improve",
                "issue",
                "problem",
                "look",
                "feel",
            }
        )

    def _is_vague_acknowledgement(self, user_input: str) -> bool:
        normalized = user_input.strip().lower().strip(".! ")
        return normalized in {
            "ok",
            "okay",
            "alright",
            "fine",
            "cool",
            "hmm",
            "hmmm",
            "got it",
            "i see",
            "sure",
        }

    def _build_off_track_response(self, conversation: Conversation, *, question: bool) -> str:
        count = self._increment_off_track_count(conversation)
        if count >= 2:
            conversation.state = ConversationState.END
            return "Understood. I will pause this feedback for now. Let me know if you would like to add anything later."

        if question:
            return "My name is HeuriSense. I am here to help collect feedback about your AI experience."

        if self._has_feedback_context(conversation):
            return "Noted. If there is anything else you would like to improve or highlight about the output, feel free to mention it."
        return "I can add feedback about the AI experience whenever you are ready."

    def _increment_off_track_count(self, conversation: Conversation) -> int:
        context = self._conversation_context(conversation)
        memory = context.setdefault("feedback_memory", self._default_context()["feedback_memory"])
        count = int(memory.get("off_track_count", 0)) + 1
        memory["off_track_count"] = count
        context["feedback_memory"] = memory
        conversation.context = context
        return count

    def _reset_off_track_count(self, conversation: Conversation) -> None:
        context = self._conversation_context(conversation)
        memory = context.setdefault("feedback_memory", self._default_context()["feedback_memory"])
        memory["off_track_count"] = 0
        context["feedback_memory"] = memory
        conversation.context = context

    def _is_confirmation_signal(self, user_input: str) -> bool:
        normalized = user_input.strip().lower()
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in normalized)
        tokens = [token for token in cleaned.split() if token]
        if not tokens or len(tokens) > 5:
            return False

        confirmation_tokens = {
            "yes",
            "yeah",
            "yep",
            "correct",
            "exactly",
            "right",
            "true",
            "sure",
            "ok",
            "okay",
            "thanks",
            "thank",
            "you",
            "that",
            "s",
            "is",
            "thats",
        }
        if not set(tokens).issubset(confirmation_tokens):
            return False
        return any(token in tokens for token in {"yes", "yeah", "yep", "correct", "exactly", "right", "true", "sure", "ok", "okay"})

    def _has_feedback_context(self, conversation: Conversation) -> bool:
        memory = self._conversation_context(conversation).get("feedback_memory", {})
        return bool(
            self._get_context_value(conversation, "feedback_captured", False)
            or self._feedback_count(conversation) > 0
            or memory.get("negatives")
            or memory.get("suggestions")
            or memory.get("issue_tags")
            or memory.get("active_thread")
            or memory.get("completed_threads")
        )

    def _is_thread_saturated(
        self,
        active_thread: str,
        user_input: str,
        extracted: FeedbackExtraction,
        memory: dict,
    ) -> bool:
        issue_terms = self._extract_issue_terms(user_input, extracted)
        evidence_terms = set(memory.get("thread_evidence", []))
        thread_turn_count = int(dict(memory.get("thread_turn_counts", {})).get(active_thread, 0))
        detail_count = len(evidence_terms)
        repeated_terms = len(set(issue_terms) & evidence_terms)
        word_count = len(user_input.split())

        if word_count >= 24 and detail_count >= 5:
            return True
        if thread_turn_count >= 2 and detail_count >= 4:
            return True
        if thread_turn_count >= 2 and issue_terms and repeated_terms >= max(2, len(set(issue_terms)) - 1):
            return True
        return False

    def _extract_issue_terms(self, user_input: str, extracted: FeedbackExtraction) -> list[str]:
        normalized = " ".join(
            [user_input, *extracted.negatives, *extracted.suggestions, *extracted.issue_tags]
        ).lower()
        term_map = {
            "emotional_warmth": {"warm", "cozy", "cosy", "emotional warmth"},
            "cinematic_atmosphere": {"cinematic", "movie", "film", "atmosphere", "immersive", "lived-in"},
            "emotional_flatness": {"flat", "cold", "empty", "lifeless", "missed the vibe"},
            "natural_human_feeling": {"natural", "real", "realism", "realistic", "human"},
            "artificial_posing": {"artificial", "posed", "posing", "staged"},
            "composition_worked": {"composition", "setting", "scene worked", "looked pretty", "visually"},
            "missing_reflections": {"reflection", "reflections"},
            "missing_steam": {"steam"},
            "image_sharpness": {"sharpness", "sharp", "blurry", "blurred"},
            "material_quality": {"material", "materials", "metallic", "metal", "texture"},
            "distorted_proportions": {"proportion", "proportions", "distorted", "warped"},
            "product_rendering_quality": {"product quality", "rendering quality", "luxury product", "advertisement"},
            "environmental_scale": {"scale", "environmental scale", "island scale", "terrain"},
            "aerial_perspective": {"aerial", "drone", "perspective", "drone photography"},
            "water_texture_realism": {"water texture", "water textures", "ocean texture", "water"},
            "island_realism": {"island", "tropical", "coast", "coastline"},
            "lighting_mood": {"lighting", "rainy", "mood"},
            "style_mismatch": {"style", "cartoon", "anime"},
            "quality_gap": {"quality", "blurry", "distorted", "incorrect"},
        }

        terms: list[str] = []
        for label, tokens in term_map.items():
            if any(token in normalized for token in tokens):
                terms.append(label)
        return terms

    def _mark_thread_completed(self, conversation: Conversation, active_thread: str) -> None:
        context = self._conversation_context(conversation)
        memory = context.setdefault("feedback_memory", self._default_context()["feedback_memory"])
        memory["completed_threads"] = self._merge_limited(
            memory.get("completed_threads", []),
            [active_thread],
            limit=8,
        )
        memory["unresolved_mismatches"] = [
            item for item in list(memory.get("unresolved_mismatches", [])) if item != active_thread
        ]
        if memory.get("active_thread") == active_thread:
            memory["active_thread"] = ""
        memory["repeated_completed_thread"] = ""
        context["feedback_memory"] = memory
        conversation.context = context

    def _clear_repeated_completed_thread(self, conversation: Conversation) -> None:
        context = self._conversation_context(conversation)
        memory = context.setdefault("feedback_memory", self._default_context()["feedback_memory"])
        memory["repeated_completed_thread"] = ""
        context["feedback_memory"] = memory
        conversation.context = context

    def _mark_active_thread_completed_if_present(self, conversation: Conversation) -> None:
        memory = self._conversation_context(conversation).get("feedback_memory", {})
        active_thread = str(memory.get("active_thread") or "")
        if active_thread:
            self._mark_thread_completed(conversation, active_thread)

    def _build_thread_summary_response(self, active_thread: str) -> str:
        if "technical_product_realism:" in active_thread:
            return "Understood. This was a technical realism issue: sharpness, materials, reflections, and proportions mattered most."
        if "emotional_tone:" in active_thread:
            return "That makes sense. The scene needed stronger tone consistency, warmth, and lived-in detail."
        if "style:" in active_thread:
            return "Got it. The main gap was that the visual style did not match what you had in mind."
        if "lighting:" in active_thread:
            return "Understood. The lighting missed the mood you were aiming for."
        if "tone:" in active_thread:
            return "Got it. The tone shifted away from the feeling you wanted."
        return "I think I understand the gap much more clearly now. This feedback is helpful."

    def _format_traits(self, traits: dict[str, list[str]]) -> list[str]:
        return [f"{category}: {', '.join(values)}" for category, values in traits.items() if values]

    def _merge_limited(self, old_list, new_list, *, limit: int) -> list[str]:
        merged: list[str] = []
        for item in list(old_list or []) + list(new_list or []):
            if item and item not in merged:
                merged.append(item)
        return merged[-limit:]

    def _feedback_count(self, conversation: Conversation) -> int:
        memory = self._conversation_context(conversation).get("feedback_memory", {})
        return int(memory.get("entry_count", 0))

    def _merge_lists(self, old_list, new_list) -> list[str]:
        if not old_list:
            old_list = []
        if not new_list:
            new_list = []
        merged: list[str] = []
        for item in old_list + new_list:
            if item and item not in merged:
                merged.append(item)
        return merged

    def _merge_raw_text(self, existing_text: str | None, new_text: str) -> str:
        existing_parts = [part for part in (existing_text or "").split("\n---\n") if part.strip()]
        if new_text not in existing_parts:
            existing_parts.append(new_text)
        return "\n---\n".join(existing_parts)

    def _build_feedback_summary(self, feedback: Feedback) -> str:
        summary_parts: list[str] = []
        if feedback.rating is not None:
            summary_parts.append(f"Rating {feedback.rating}/5")

        positive = self._summary_phrase(feedback.positives, fallback="")
        issue_dimensions = self._summary_issue_dimensions(feedback.issue_tags, feedback.negatives, feedback.suggestions)
        complaint = self._summary_phrase(feedback.negatives, fallback="")
        request = self._summary_phrase(feedback.suggestions, fallback="")

        if positive and issue_dimensions:
            summary_parts.append(f"{positive.capitalize()}, but {issue_dimensions} weakened the result")
        elif positive and complaint:
            summary_parts.append(f"{positive.capitalize()}, but {complaint}")
        elif issue_dimensions:
            summary_parts.append(f"{issue_dimensions.capitalize()} weakened the result")
        elif complaint:
            summary_parts.append(f"Key issue: {complaint}")
        elif request:
            summary_parts.append(f"Top request: {request}")
        elif positive:
            summary_parts.append(f"Highlight: {positive}")
        return " | ".join(summary_parts)

    def _summary_phrase(self, values: list, *, fallback: str) -> str:
        for value in values or []:
            text = " ".join(str(value).strip().split())
            if text:
                return text
        return fallback

    def _summary_issue_dimensions(self, issue_tags: list, negatives: list, suggestions: list) -> str:
        labels = [self._humanize_issue_tag(tag) for tag in issue_tags or [] if tag]
        labels = [label for label in labels if label]
        if not labels:
            combined = " ".join(str(item) for item in list(negatives or []) + list(suggestions or [])).lower()
            inferred: list[str] = []
            if "density" in combined or "empty" in combined:
                inferred.append("environmental density")
            if "motion" in combined or "static" in combined:
                inferred.append("motion realism")
            if "lighting" in combined or "falloff" in combined:
                inferred.append("lighting consistency")
            if "texture" in combined:
                inferred.append("texture realism")
            if "scale" in combined or "massive" in combined:
                inferred.append("scale consistency")
            labels = inferred
        return self._format_summary_labels(labels[:3])

    def _humanize_issue_tag(self, tag: str) -> str:
        label_map = {
            "environmental_density": "environmental density",
            "environmental_realism": "environmental realism",
            "motion_realism": "motion realism",
            "lighting_consistency": "lighting consistency",
            "texture_realism": "texture realism",
            "scale_consistency": "scale consistency",
            "atmospheric_depth": "atmospheric depth",
            "material_realism": "material realism",
            "reflection_realism": "reflection realism",
            "interaction_realism": "interaction realism",
            "composition_balance": "composition balance",
            "perspective_consistency": "perspective consistency",
            "anatomy_accuracy": "anatomy accuracy",
            "cinematic_alignment": "cinematic alignment",
            "prompt_alignment": "prompt alignment",
            "detail_sharpness": "detail sharpness",
            "environmental_believability": "environmental believability",
        }
        return label_map.get(str(tag), str(tag).replace("_", " "))

    def _format_summary_labels(self, labels: list[str]) -> str:
        cleaned: list[str] = []
        for label in labels:
            if label and label not in cleaned:
                cleaned.append(label)
        if len(cleaned) <= 1:
            return cleaned[0] if cleaned else ""
        if len(cleaned) == 2:
            return " and ".join(cleaned)
        return f"{cleaned[0]}, {cleaned[1]}, and {cleaned[2]}"

    def _get_feedback_entries(self, conversation: Conversation) -> list[Feedback]:
        return list(self.db.scalars(select(Feedback).where(Feedback.conversation_id == conversation.id).order_by(Feedback.id)))

    def _get_or_create_feedback_entry(self, conversation: Conversation) -> Feedback:
        feedback_entries = self._get_feedback_entries(conversation)
        if not feedback_entries:
            feedback = Feedback(
                conversation_id=conversation.id,
                rating=self._get_context_value(conversation, "rating"),
                sentiment=self._resolve_conversation_sentiment(conversation, self._get_context_value(conversation, "last_sentiment", "mixed")),
                positives=[],
                negatives=[],
                suggestions=[],
                issue_type=self._resolve_issue_type(conversation),
                issue_tags=[],
                raw_text="",
                summary="",
            )
            self.db.add(feedback)
            self.db.flush()
            return feedback

        primary = feedback_entries[0]
        if len(feedback_entries) > 1:
            for extra in feedback_entries[1:]:
                primary.rating = primary.rating if primary.rating is not None else extra.rating
                primary.positives = self._merge_lists(primary.positives, extra.positives)
                primary.negatives = self._merge_lists(primary.negatives, extra.negatives)
                primary.suggestions = self._merge_lists(primary.suggestions, extra.suggestions)
                primary.issue_tags = self._merge_lists(primary.issue_tags, extra.issue_tags)
                primary.raw_text = self._merge_raw_text(primary.raw_text, extra.raw_text)
                primary.summary = primary.summary or extra.summary
                self.db.delete(extra)
        return primary

    def _resolve_issue_type(self, conversation: Conversation) -> str:
        memory = self._conversation_context(conversation).get("feedback_memory", {})
        return self._resolve_issue_type_from_memory(memory)

    def _resolve_issue_type_from_memory(self, memory: dict) -> str:
        issue_types = [issue for issue in memory.get("issue_types", []) if issue and issue != "none"]
        if not issue_types:
            return "none"

        counts: dict[str, int] = {}
        for issue_type in issue_types:
            counts[issue_type] = counts.get(issue_type, 0) + 1

        priority = {"technical": 3, "quality": 2, "usability": 1, "none": 0}
        return max(counts, key=lambda issue: (counts[issue], priority.get(issue, 0)))

    def _resolve_conversation_sentiment(self, conversation: Conversation, latest_sentiment: str) -> str:
        memory = self._conversation_context(conversation).get("feedback_memory", {})
        sentiments = list(memory.get("sentiments", []))
        if latest_sentiment:
            sentiments.append(latest_sentiment)
        return self._resolve_sentiment_from_memory({"sentiments": sentiments})

    def _resolve_sentiment_from_memory(self, memory: dict) -> str:
        sentiments = [sentiment for sentiment in memory.get("sentiments", []) if sentiment]
        if not sentiments:
            return "mixed"
        unique = set(sentiments)
        if len(unique) > 1:
            return "mixed"
        return sentiments[-1]

    def _has_extractable_feedback(self, extracted: FeedbackExtraction) -> bool:
        return bool(extracted.positives or extracted.negatives or extracted.suggestions or extracted.issue_tags)

    def _extract_embedded_rating_if_present(self, user_input: str) -> int | None:
        result = self.llm_service.extract_rating(user_input)
        return result.rating

    def _is_user_declining(self, user_input: str) -> bool:
        normalized = user_input.strip().lower()
        return normalized in {
            "no",
            "nope",
            "nah",
            "not now",
            "nothing else",
            "nothing more",
            "no other issues",
            "that's all",
            "thats all",
            "all good",
            "no thanks",
            "no more",
            "stop",
            "that's it",
            "thats it",
            "done",
            "i'm done",
            "i am done",
        }

    def _maybe_build_contextual_follow_up(
        self,
        conversation: Conversation,
        extracted: FeedbackExtraction,
        issue_type: str,
        previous_state: ConversationState,
    ) -> str | None:
        if previous_state == ConversationState.ISSUE_HANDLING and not self._has_unresolved_grounding_thread(conversation):
            return None

        if self._get_context_value(conversation, "rating") is not None and self._has_unresolved_grounding_thread(conversation):
            humanized = self._generate_human_follow_up(
                conversation=conversation,
                user_feedback=" ".join(extracted.negatives + extracted.suggestions + extracted.positives),
                extracted=extracted,
                issue_type=issue_type,
            )
            if humanized:
                return humanized
            grounded = self._build_grounded_thread_follow_up(conversation)
            if grounded:
                return grounded

        follow_up_key = self._choose_follow_up_key(extracted, issue_type)
        if not follow_up_key or self._was_follow_up_asked(conversation, follow_up_key):
            return None

        self._mark_follow_up_asked(conversation, follow_up_key)
        fallback = self._build_follow_up_from_key(follow_up_key)
        if not self._question_has_high_value(
            conversation=conversation,
            question=fallback,
            extracted=extracted,
            user_feedback=" ".join(extracted.negatives + extracted.suggestions + extracted.positives),
            issue_type=issue_type,
        ):
            return None

        humanized = self._generate_human_follow_up(
            conversation=conversation,
            user_feedback=" ".join(extracted.negatives + extracted.suggestions + extracted.positives),
            extracted=extracted,
            issue_type=issue_type,
        )
        return humanized or fallback

    def _choose_follow_up_key(self, extracted: FeedbackExtraction, issue_type: str) -> str | None:
        tags = set(extracted.issue_tags)
        if "slow_response_time" in tags:
            return "performance_detail"
        if extracted.suggestions:
            return "feature_request"
        if issue_type in {"technical", "usability"} and extracted.negatives:
            return issue_type
        if extracted.sentiment == "positive" and extracted.positives:
            return "positive_detail"
        if not extracted.positives and not extracted.negatives and not extracted.suggestions:
            return "clarify"
        return None

    def _was_follow_up_asked(self, conversation: Conversation, key: str) -> bool:
        memory = self._conversation_context(conversation).get("feedback_memory", {})
        return key in memory.get("follow_ups_asked", [])

    def _mark_follow_up_asked(self, conversation: Conversation, key: str) -> None:
        context = self._conversation_context(conversation)
        memory = context.setdefault("feedback_memory", self._default_context()["feedback_memory"])
        if key not in memory["follow_ups_asked"]:
            memory["follow_ups_asked"].append(key)
        conversation.context = context

    def _build_follow_up_from_key(self, key: str) -> str:
        prompts = {
            "performance_detail": "That helps. About how long did the delay take, and what were you trying to do when it felt slow?",
            "feature_request": "That is helpful. What would you expect the system to do differently in the ideal version?",
            "technical": "That sounds frustrating. What exactly happened, and what were you trying to do at the time?",
            "usability": "I can work with that. Which part felt hardest to use, and what would make it clearer?",
            "quality": "I have noted the quality concern.",
            "positive_detail": "That is good to hear. What specifically worked well for you?",
            "clarify": "Can you say a little more so I capture the useful part of that feedback?",
        }
        return prompts[key]

    def _has_unresolved_grounding_thread(self, conversation: Conversation) -> bool:
        memory = self._conversation_context(conversation).get("feedback_memory", {})
        return bool(memory.get("unresolved_mismatches") or memory.get("active_thread"))

    def _build_grounded_thread_follow_up(self, conversation: Conversation) -> str | None:
        memory = self._conversation_context(conversation).get("feedback_memory", {})
        active_thread = str(memory.get("active_thread") or "")
        if not active_thread:
            unresolved = list(memory.get("unresolved_mismatches", []))
            active_thread = unresolved[-1] if unresolved else ""

        if "emotional_tone:" in active_thread:
            return "So the issue was more about emotional tone than image quality itself?"
        if "style:" in active_thread:
            return "So the main gap was the style not matching what you imagined?"
        if "lighting:" in active_thread:
            return "Was the lighting the biggest part that missed the intended mood?"
        if "tone:" in active_thread:
            return "So the tone shifted away from what you wanted?"
        return None

    def _question_has_high_value(
        self,
        *,
        conversation: Conversation,
        question: str,
        extracted: FeedbackExtraction,
        user_feedback: str,
        issue_type: str,
    ) -> bool:
        if not question.strip() or "?" not in question:
            return True

        memory = self._conversation_context(conversation).get("feedback_memory", {})
        normalized = question.strip().lower()
        previous_questions = [item.lower().strip() for item in memory.get("human_followups_asked", [])]
        if normalized in previous_questions:
            return False

        if self._is_generic_survey_question(normalized) and self._has_rich_grounded_context(conversation, extracted, user_feedback):
            return False

        if "confusing" in normalized or "step" in normalized:
            combined = " ".join([user_feedback, *extracted.negatives, *extracted.suggestions, *extracted.issue_tags]).lower()
            if not any(token in combined for token in {"confusing", "confused", "step", "navigation", "hard to use", "unclear"}):
                return False

        active_thread = str(memory.get("active_thread") or "")
        if active_thread and self._is_thread_saturated(active_thread, user_feedback, extracted, memory):
            return False

        completed_threads = list(memory.get("completed_threads", []))
        if completed_threads and self._is_generic_survey_question(normalized):
            return False

        return True

    def _is_generic_survey_question(self, normalized_question: str) -> bool:
        generic_patterns = {
            "what felt most off",
            "what stood out as the main problem",
            "which step felt",
            "which part felt",
            "what should have looked",
            "what exactly happened",
            "can you say a little more",
            "could you share a little more",
            "anything else",
            "one more detail",
            "another improvement",
            "another change",
            "another strong point",
            "what would you expect the system to do differently",
            "what one detail should we capture next",
        }
        return any(pattern in normalized_question for pattern in generic_patterns)

    def _has_rich_grounded_context(
        self,
        conversation: Conversation,
        extracted: FeedbackExtraction,
        user_feedback: str,
    ) -> bool:
        memory = self._conversation_context(conversation).get("feedback_memory", {})
        issue_terms = set(memory.get("thread_evidence", [])) | set(self._extract_issue_terms(user_feedback, extracted))
        feedback_text = " ".join(
            [
                user_feedback,
                " ".join(memory.get("negatives", [])),
                " ".join(memory.get("suggestions", [])),
                " ".join(memory.get("issue_tags", [])),
            ]
        )
        return bool(
            len(issue_terms) >= 4
            or len(feedback_text.split()) >= 35
            or memory.get("completed_threads")
            or (memory.get("contextual_mismatches") and len(issue_terms) >= 3)
        )

    def _build_progressive_acknowledgement(self, conversation: Conversation) -> str:
        memory = self._conversation_context(conversation).get("feedback_memory", {})
        current_domain = str(memory.get("current_domain") or "")
        active_thread = str(memory.get("active_thread") or "")
        if not active_thread:
            completed = list(memory.get("completed_threads", []))
            active_thread = completed[-1] if completed else ""

        if active_thread:
            return self._build_thread_summary_response(active_thread)

        if current_domain == "environmental_aerial_realism":
            return "Understood. The scene had the right vacation idea, but the scale, perspective, and environmental realism felt artificial."

        latest_correction = str(memory.get("latest_correction") or "")
        if "technical realism" in latest_correction or "product rendering" in latest_correction:
            return "Got it. This was more of a technical realism failure than a mood or atmosphere issue."

        evidence = set(memory.get("thread_evidence", []))
        if {"environmental_scale", "aerial_perspective", "water_texture_realism", "island_realism"} & evidence:
            return "Understood. The realism issue was environmental believability: scale, aerial perspective, and natural water texture."
        if current_domain == "technical_product_realism" and {"image_sharpness", "material_quality", "missing_reflections", "distorted_proportions"} & evidence:
            return "Got it. The product needed sharper detail, stronger materials, cleaner reflections, and more realistic proportions."
        invalidated = " ".join(memory.get("invalidated_threads", []))
        if current_domain != "technical_product_realism" and "emotional_tone" not in invalidated and {"cinematic_atmosphere", "emotional_flatness"} & evidence:
            return "Understood. The realism problem seems more about tone, warmth, and lived-in scene detail."
        if {"natural_human_feeling", "artificial_posing"} & evidence:
            return "Got it. The scene needed more natural posing and lived-in interaction detail."
        return "I think I understand the gap clearly now. This feedback is helpful."

    def _should_complete_capture_after_summary(self, extracted: FeedbackExtraction, issue_type: str) -> bool:
        if not self._has_extractable_feedback(extracted):
            return False
        if extracted.sentiment == "positive" and not extracted.negatives and not extracted.suggestions:
            return False
        return issue_type in {"quality", "technical", "usability", "none"}

    def _is_positive_only_feedback(self, extracted: FeedbackExtraction) -> bool:
        return bool(extracted.positives) and not extracted.negatives and not extracted.suggestions

    def _build_positive_feedback_response(self, conversation: Conversation, extracted: FeedbackExtraction) -> str:
        positive_focus = self._positive_focus_phrase(extracted.positives)
        acknowledgement = (
            f"Glad to hear that - I have noted {positive_focus} positively."
            if positive_focus
            else "Glad to hear that - I have noted the positive feedback."
        )

        if self._get_context_value(conversation, "rating") is None and not self._get_context_value(conversation, "rating_prompt_asked", False):
            return acknowledgement

        continuation = self._maybe_lightweight_continuation(conversation)
        if continuation:
            return f"{acknowledgement} {continuation}"
        return acknowledgement

    def _positive_focus_phrase(self, positives: list[str]) -> str:
        combined = " ".join(positives or []).lower()
        focus: list[str] = []
        if any(token in combined for token in {"water", "underwater", "ocean"}):
            focus.append("the underwater atmosphere")
        if any(token in combined for token in {"color", "colors", "palette", "blue"}):
            focus.append("the color palette")
        if any(token in combined for token in {"whale", "character", "design"}):
            focus.append("the whale design")
        if any(token in combined for token in {"lighting", "light", "mood"}):
            focus.append("the lighting mood")
        if any(token in combined for token in {"composition", "framing"}):
            focus.append("the composition")
        if any(token in combined for token in {"atmosphere", "mood", "style", "cinematic"}):
            focus.append("the atmosphere")

        if focus:
            return self._format_summary_labels(focus[:2])
        if positives:
            return positives[0]
        return ""

    def _build_captured_issue_response(self, conversation: Conversation, issue_summary: str) -> str:
        if self._get_context_value(conversation, "rating") is None:
            return f"{issue_summary}\n\n{self._build_rating_prompt()}"

        acknowledgement = "I have noted this feedback down."
        continuation = self._maybe_lightweight_continuation(conversation)
        if continuation:
            return f"{issue_summary} {acknowledgement} {continuation}"
        return f"{issue_summary} {acknowledgement}"

    def _build_confirmation_continue_response(self, conversation: Conversation) -> str:
        continuation = self._maybe_lightweight_continuation(conversation, force=True)
        if continuation:
            return f"Great. I have noted this feedback down. {continuation}"
        return "Great. I have noted this feedback down."

    def _maybe_lightweight_continuation(self, conversation: Conversation, *, force: bool = False) -> str:
        context = self._conversation_context(conversation)
        memory = context.setdefault("feedback_memory", self._default_context()["feedback_memory"])
        used = int(memory.get("lightweight_continuations_used", 0))
        count = self._feedback_count(conversation)

        if not force and used > 0 and count % 3 != 1:
            return ""

        variants = [
            "Anything else you would like to add?",
            "Is there anything else that needs improvement?",
            "Would you like to add any other feedback?",
        ]
        prompt = variants[used % len(variants)]
        memory["lightweight_continuations_used"] = used + 1
        context["feedback_memory"] = memory
        conversation.context = context
        return prompt

    def _build_contextual_issue_summary(
        self,
        *,
        conversation: Conversation,
        user_input: str,
        extracted: FeedbackExtraction,
        issue_type: str,
    ) -> str:
        if extracted.sentiment == "positive" and not extracted.negatives and not extracted.suggestions:
            return ""

        dimensions = self._infer_issue_dimensions(conversation, user_input, extracted, issue_type)
        if not dimensions:
            return ""

        phrase = self._format_issue_dimensions(dimensions)

        realism_dimensions = {
            "scale",
            "environmental",
            "material",
            "texture",
            "motion",
            "body",
            "lighting",
            "posing",
            "sharpness",
            "finish",
        }
        if any(token in phrase for token in realism_dimensions) or issue_type == "quality":
            return f"I see - the realism mainly breaks in {phrase}."
        if issue_type == "technical":
            return f"I see - the main problem is {phrase}."
        return f"I see - the feedback points mostly to {phrase}."

    def _format_issue_dimensions(self, dimensions: list[str]) -> str:
        selected = dimensions[:3]
        if len(selected) <= 1:
            return selected[0] if selected else ""
        if len(selected) == 2:
            return " and ".join(selected)
        return f"{selected[0]}, {selected[1]}, and {selected[2]}"

    def _infer_issue_dimensions(
        self,
        conversation: Conversation,
        user_input: str,
        extracted: FeedbackExtraction,
        issue_type: str,
    ) -> list[str]:
        normalized = " ".join(
            [user_input, *extracted.negatives, *extracted.suggestions, *extracted.issue_tags]
        ).lower()
        memory = self._conversation_context(conversation).get("feedback_memory", {})
        evidence = set(memory.get("thread_evidence", [])) | set(self._extract_issue_terms(user_input, extracted))

        dimensions: list[str] = []

        def add(dimension: str) -> None:
            if any(dimension in existing or existing in dimension for existing in dimensions):
                return
            if dimension not in dimensions:
                dimensions.append(dimension)

        if any(token in normalized for token in {"scale", "massive", "huge", "size", "proportion", "proportions"}):
            add("scale consistency")
        if any(token in normalized for token in {"depth", "depth cues", "underwater depth", "underwater"}):
            add("underwater depth cues")
        if any(token in normalized for token in {"falloff", "fall off", "lighting falloff", "light falloff"}):
            add("lighting falloff")
        elif any(token in normalized for token in {"lighting", "artificial light", "shadow", "shadows", "mood"}):
            add("lighting consistency")
        if any(token in normalized for token in {"integrated", "integration", "naturally integrated"}):
            add("environmental integration")
        elif any(token in normalized for token in {"environment", "scene", "water", "ocean"}):
            add("environmental realism")
        if any(token in normalized for token in {"syrup", "butter", "melt", "melting", "pour", "flow", "fluid", "liquid", "behave"}):
            add("material behavior")
        if any(token in normalized for token in {"texture", "textures", "material", "materials", "surface", "syrup", "butter"}):
            add("texture consistency")
        if any(token in normalized for token in {"frozen", "stiff", "static", "motion", "action", "movement"}):
            add("motion and body dynamics")
        if any(token in normalized for token in {"player", "athlete", "pose", "body"}):
            add("body dynamics")
        if any(token in normalized for token in {"flat", "cold", "emotion", "emotional", "vibe", "atmosphere", "warm", "cozy"}):
            add("tone consistency")
        if any(token in normalized for token in {"style", "cartoon", "anime", "cinematic"}):
            add("style alignment")
        if any(token in normalized for token in {"blurry", "sharpness", "sharp", "detail", "details"}):
            add("detail sharpness")
        if any(token in normalized for token in {"reflection", "reflections", "metal", "metallic"}):
            add("material finish")

        if {"environmental_scale", "aerial_perspective", "water_texture_realism", "island_realism"} & evidence:
            add("environmental believability")
        if {"image_sharpness", "missing_reflections", "material_quality", "distorted_proportions"} & evidence:
            add("technical realism")
        if {"cinematic_atmosphere", "emotional_flatness"} & evidence:
            add("tone consistency")
        if {"artificial_posing", "natural_human_feeling"} & evidence:
            add("natural posing")

        if not dimensions and issue_type == "quality":
            add("output quality")
        elif not dimensions and issue_type == "usability":
            add("usability clarity")

        return dimensions[:4]

    def _combine_issue_summary_with_reply(self, issue_summary: str, base_reply: str) -> str:
        if not issue_summary or issue_summary.lower() in base_reply.lower():
            return base_reply
        if base_reply.startswith(("I see -", "Got it.", "Understood.", "That makes sense.")):
            return base_reply
        return f"{issue_summary}\n\n{base_reply}"

    def _build_issue_follow_up(self, issue_type: str, issue_tags: list[str], detailed: bool) -> str:
        if "slow_response_time" in issue_tags:
            return "About how long did the delay take, and did it happen every time or only on that step?"
        if "multiple_output_request" in issue_tags:
            return "How many outputs would feel right to you, and when would you want that option?"
        if issue_type == "technical":
            if detailed:
                return "Did it crash, freeze, or fail in some other way? A quick step-by-step description would help."
            return "What exactly happened and when?"
        if issue_type == "usability":
            if detailed:
                return "Which part was hardest to use, and what would make that flow feel more obvious?"
            return "Which part was difficult to use?"
        if issue_type == "quality":
            if detailed:
                return "What should have looked or behaved differently so the result felt right to you?"
            return "What felt incorrect or low quality?"
        return "Could you share a little more detail so I can capture the issue accurately?"

    def _build_rating_prompt(self) -> str:
        return (
            "On a scale of 1-5, how would you rate your experience?\n\n"
            "1 = Very poor (unusable, major issues)\n"
            "2 = Poor (many problems)\n"
            "3 = Average (some issues)\n"
            "4 = Good (minor improvements needed)\n"
            "5 = Excellent (worked very well)"
        )

    def _build_continue_prompt(self, conversation: Conversation, extracted: FeedbackExtraction) -> str:
        count = self._feedback_count(conversation)
        if extracted.sentiment == "positive":
            variants = [
                "I have noted the positive feedback. Was there anything else that worked especially well?",
                "That is helpful. If there was another strong point, I would love to capture it too.",
                "Great, I have added that. Anything else you want the team to keep doing?",
            ]
        elif extracted.suggestions:
            variants = [
                "I have captured that idea. Any other improvement you would like to add?",
                "That suggestion is noted. Is there another change you would want most?",
                "Helpful direction. Anything else you want the product to do differently?",
            ]
        else:
            variants = [
                "I have noted that. Anything else you would like to add?",
                "Thanks, I have captured that. Is there one more detail you want the team to know?",
                "That is recorded. If anything else comes to mind, you can add it now.",
            ]
        return variants[count % len(variants)]

    def _build_post_rating_response(self, conversation: Conversation, rating: int) -> str:
        follow_up = self._build_context_aware_rating_follow_up(conversation, rating)
        if follow_up:
            return f"Thanks, I have added the rating. {follow_up}"
        return "Thanks, I have added the rating. I have captured the key feedback areas. Is there anything else you would like to add?"

    def _build_context_aware_rating_follow_up(self, conversation: Conversation, rating: int) -> str:
        covered = self._captured_feedback_dimensions(conversation)
        if self._has_enough_feedback_after_rating(conversation, covered, rating):
            return ""

        missing_options = self._missing_rating_follow_up_options(covered)
        if len(missing_options) >= 2:
            return (
                "What aspect would improve the experience most now - "
                f"{', '.join(missing_options[:2])}, or something else?"
            )

        if rating <= 3:
            return "What part of the result still feels furthest from your expectation?"

        return "If one thing could improve the score most, what would it be?"

    def _has_enough_feedback_after_rating(self, conversation: Conversation, covered: set[str], rating: int) -> bool:
        count = self._feedback_count(conversation)
        if rating >= 4 and count > 0:
            return True
        return count >= 2 or len(covered) >= 4

    def _missing_rating_follow_up_options(self, covered: set[str]) -> list[str]:
        priority = [
            ("realism", {"scale", "environment", "lighting", "texture", "motion", "anatomy", "integration"}),
            ("scene density", {"scene_density", "composition"}),
            ("motion", {"motion"}),
            ("prompt accuracy", {"prompt_alignment", "accuracy"}),
            ("surface detail", {"texture", "material", "detail"}),
            ("composition", {"composition"}),
        ]
        options: list[str] = []
        for label, overlaps in priority:
            if not overlaps & covered and label not in options:
                options.append(label)
        return options[:3]

    def _captured_feedback_dimensions(self, conversation: Conversation) -> set[str]:
        memory = self._conversation_context(conversation).get("feedback_memory", {})
        combined = " ".join(
            [
                " ".join(memory.get("negatives", [])),
                " ".join(memory.get("suggestions", [])),
                " ".join(memory.get("issue_tags", [])),
                " ".join(memory.get("thread_evidence", [])),
                " ".join(memory.get("contextual_mismatches", [])),
            ]
        ).lower()

        dimensions: set[str] = set()

        def add_if(tokens: set[str], dimension: str) -> None:
            if any(token in combined for token in tokens):
                dimensions.add(dimension)

        add_if({"light", "lighting", "shadow", "falloff", "mood"}, "lighting")
        add_if({"scale", "massive", "size", "proportion", "environmental_scale"}, "scale")
        add_if({"environment", "integrated", "integration", "water", "ocean", "underwater", "scene"}, "environment")
        add_if({"texture", "surface", "material", "water_texture", "materials"}, "texture")
        add_if({"motion", "movement", "static", "frozen", "body dynamics"}, "motion")
        add_if({"anatomy", "body", "pose", "posing", "proportions"}, "anatomy")
        add_if({"composition", "framing", "perspective", "aerial"}, "composition")
        add_if({"accuracy", "incorrect", "wrong", "missed prompt", "prompt"}, "prompt_alignment")
        add_if({"detail", "sharpness", "sharp", "blurry"}, "detail")
        add_if({"density", "empty", "sparse"}, "scene_density")
        add_if({"reflection", "metal", "finish"}, "material")
        add_if({"interaction", "interact", "lived-in", "natural"}, "integration")
        return dimensions

    def _build_post_rating_prompt(self, rating: int) -> str:
        if rating <= 2:
            return "Thanks for being candid. Tell me what did not work, and I will make sure it gets fixed."
        if rating == 3:
            return "Thanks. What felt average or inconsistent, and what would improve the experience?"
        return "Thanks. What worked especially well, and is there anything you would still improve?"

    def _append_human_followup_if_needed(
        self,
        *,
        conversation: Conversation,
        base_reply: str,
        user_feedback: str,
        extracted: FeedbackExtraction,
        issue_type: str,
    ) -> str:
        if "\n" in base_reply or base_reply.count("?") >= 1:
            return base_reply

        follow_up = self._generate_human_follow_up(
            conversation=conversation,
            user_feedback=user_feedback,
            extracted=extracted,
            issue_type=issue_type,
        )
        if not follow_up or follow_up in base_reply:
            return base_reply
        return f"{base_reply}\n\n{follow_up}"

    def _generate_human_follow_up(
        self,
        *,
        conversation: Conversation,
        user_feedback: str,
        extracted: FeedbackExtraction,
        issue_type: str,
    ) -> str | None:
        if not self._has_extractable_feedback(extracted):
            return None

        context = self._conversation_context(conversation)
        memory = context.get("feedback_memory", {})
        previous_followups = list(memory.get("human_followups_asked", []))
        grounding_context = self._build_grounding_context(conversation, user_feedback, extracted)
        result = self.llm_service.generate_human_followup_question(
            task_type=conversation.task_type or "text",
            prompt=conversation.prompt or "",
            ai_output=conversation.ai_output or "",
            user_feedback=user_feedback,
            existing_negatives=memory.get("negatives", []),
            existing_suggestions=memory.get("suggestions", []),
            previous_followups=previous_followups,
            detected_issue_type=issue_type,
            grounding_context=grounding_context,
        )
        question = next((item.strip() for item in result.questions if item.strip()), "")
        if not question or question in previous_followups:
            return None
        if not self._question_has_high_value(
            conversation=conversation,
            question=question,
            extracted=extracted,
            user_feedback=user_feedback,
            issue_type=issue_type,
        ):
            return None

        memory["human_followups_asked"] = previous_followups + [question]
        context["feedback_memory"] = memory
        conversation.context = context
        return question

    def _store_metadata_on_first_message(
        self,
        conversation: Conversation,
        *,
        task_type: str | None,
        prompt: str | None,
        ai_output: str | None,
        ai_output_file_url: str | None,
    ) -> None:
        if self._metadata_locked(conversation):
            return

        conversation.task_type = task_type or conversation.task_type or "text"
        conversation.prompt = (prompt or conversation.prompt or "").strip()
        conversation.ai_output = (ai_output or conversation.ai_output or "").strip()
        conversation.ai_output_file_url = (ai_output_file_url or conversation.ai_output_file_url or "").strip() or None
        self._set_context_value(conversation, "metadata_locked", True)

    def _metadata_locked(self, conversation: Conversation) -> bool:
        return bool(
            self._get_context_value(conversation, "metadata_locked", False)
            or conversation.task_type
            or conversation.prompt
            or conversation.ai_output
            or conversation.ai_output_file_url
        )

    def _respond(self, conversation: Conversation, content: str) -> str:
        self._add_assistant_message(conversation, content)
        if conversation.title == "New feedback session":
            conversation.title = content[:60]
        return content

    def _add_assistant_message(self, conversation: Conversation, content: str) -> None:
        self.db.add(
            Message(
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT,
                content=content,
            )
        )

    def _log_state(self, label: str, conversation: Conversation, user_input: str) -> None:
        logger.info(
            "chat_transition label=%s state=%s context=%s user_input=%r",
            label,
            conversation.state,
            conversation.context,
            user_input,
        )
