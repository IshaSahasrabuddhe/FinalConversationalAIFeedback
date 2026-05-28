from __future__ import annotations

from typing import Type, TypeVar

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_groq import ChatGroq
from pydantic import BaseModel

from app.core.config import get_settings
from app.schemas.llm import (
    ConversationTurnAnalysis,
    FeedbackExtraction,
    FeedbackInsightsResult,
    HumanFollowupQuestionResult,
    IntentClassification,
    IssueClassification,
    RatingExtraction,
    SentimentAnalysis,
)
from app.services.chains.prompts import (
    FEEDBACK_EXTRACTION_PROMPT,
    FEEDBACK_INSIGHTS_PROMPT,
    HUMAN_FOLLOWUP_PROMPT,
    INTENT_PROMPT,
    ISSUE_CLASSIFICATION_PROMPT,
    ISSUE_TAG_PROMPT,
    RATING_PROMPT,
    SENTIMENT_PROMPT,
    TURN_ANALYSIS_PROMPT,
)

T = TypeVar("T", bound=BaseModel)


class IssueTagResult(BaseModel):
    issue_tags: list[str]


class StructuredChainFactory:
    def __init__(self) -> None:
        settings = get_settings()
        self._llm = None
        if settings.groq_api_key:
            self._llm = ChatGroq(
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                temperature=0,
            )

    @property
    def enabled(self) -> bool:
        return self._llm is not None

    def build_chain(self, prompt_template: str, schema: Type[T]) -> RunnableSerializable:
        if not self._llm:
            raise RuntimeError("Groq LLM is not configured")
        prompt = ChatPromptTemplate.from_template(prompt_template)
        return prompt | self._llm.with_structured_output(schema)


class FallbackClassifier:
    POSITIVE_HINTS = {
        "good",
        "great",
        "helpful",
        "fast",
        "clear",
        "excellent",
        "love",
        "smooth",
        "useful",
        "like",
        "nice",
    }
    NEGATIVE_HINTS = {
        "bad",
        "slow",
        "broken",
        "confusing",
        "error",
        "issue",
        "bug",
        "wrong",
        "poor",
        "missed",
        "missing",
        "lacked",
        "lacking",
        "flat",
        "cold",
        "not warm",
        "not cozy",
        "vibe",
        "atmosphere",
        "emotionally",
        "sharpness",
        "material",
        "materials",
        "reflection",
        "reflections",
        "proportion",
        "proportions",
        "product quality",
        "crash",
        "crashed",
        "freeze",
        "frozen",
        "confused",
        "difficult",
        "incorrect",
        "inaccurate",
        "missing",
        "hard",
        "unrealistic",
        "delay",
    }
    VAGUE_RATING_HINTS = {"okay", "ok", "fine", "average", "decent", "not bad", "so so", "its okay", "it's okay"}
    STOP_HINTS = {"no", "no more", "stop", "that's it", "thats it", "nothing else", "no thanks"}
    TECHNICAL_HINTS = {
        "bug",
        "error",
        "crash",
        "crashed",
        "freeze",
        "frozen",
        "latency",
        "slow",
        "broken",
        "failed",
        "delay",
        "minutes",
    }
    QUALITY_HINTS = {
        "quality",
        "accuracy",
        "accurate",
        "wrong",
        "hallucination",
        "incorrect",
        "inaccurate",
        "missing",
        "realistic",
        "unrealistic",
        "sober",
        "vibe",
        "atmosphere",
        "emotionally",
        "warm",
        "cozy",
        "cinematic",
        "alive",
        "flat",
        "sharpness",
        "material",
        "materials",
        "reflection",
        "reflections",
        "proportion",
        "proportions",
        "product quality",
        "luxury product",
    }
    USABILITY_HINTS = {"hard", "confusing", "confused", "ui", "ux", "difficult", "unclear", "find", "navigation", "use"}

    @classmethod
    def analyze_turn(cls, message: str) -> ConversationTurnAnalysis:
        rating_result = cls.extract_rating(message)
        extracted = cls.extract_feedback(message)
        issue = cls.classify_issue(message)
        sentiment = cls.analyze_sentiment(message)
        has_feedback = bool(extracted.positives or extracted.negatives or extracted.suggestions)

        if has_feedback:
            intent = "feedback"
        elif rating_result.rating is not None or rating_result.is_vague:
            intent = "rating"
        else:
            intent = "off_topic"

        return ConversationTurnAnalysis(
            intent=intent,
            issue_type=issue.issue_type,
            sentiment=sentiment.sentiment,
            is_feedback_present=has_feedback,
        )

    @classmethod
    def classify_intent(cls, message: str) -> IntentClassification:
        normalized = message.strip().lower()
        if any(token in normalized for token in {"yes", "sure", "okay", "ok", "yeah", "yep"}):
            return IntentClassification(intent="YES", confidence=0.82, reasoning="Affirmative response detected.")
        if any(token in normalized for token in {"no", "nope", "nah", "not now"}):
            return IntentClassification(intent="NO", confidence=0.85, reasoning="Negative response detected.")
        return IntentClassification(intent="OFF_TOPIC", confidence=0.55, reasoning="Message does not clearly accept or reject feedback.")

    @classmethod
    def extract_rating(cls, message: str) -> RatingExtraction:
        normalized = message.strip().lower()
        rating_cues = {"rate", "rating", "score", "stars", "/5", "out of 5"}

        if normalized in cls.STOP_HINTS:
            return RatingExtraction(rating=None, is_vague=False, clarification_needed="")
        if any(phrase in normalized for phrase in cls.VAGUE_RATING_HINTS):
            return RatingExtraction(
                rating=None,
                is_vague=True,
                clarification_needed="Please share a rating from 1 to 5, where 1 is very poor and 5 is excellent.",
            )

        tokens = message.replace("/", " ").split()
        for token in tokens:
            cleaned = "".join(ch for ch in token if ch.isdigit())
            if cleaned.isdigit():
                rating = int(cleaned)
                if 1 <= rating <= 5:
                    return RatingExtraction(rating=rating, is_vague=False, clarification_needed="")

        if any(cue in normalized for cue in rating_cues):
            return RatingExtraction(
                rating=None,
                is_vague=True,
                clarification_needed="Please share a rating from 1 to 5, where 1 is very poor and 5 is excellent.",
            )

        return RatingExtraction(rating=None, is_vague=False, clarification_needed="")

    @classmethod
    def analyze_sentiment(cls, message: str) -> SentimentAnalysis:
        normalized = message.lower()
        positive = any(word in normalized for word in cls.POSITIVE_HINTS)
        negative = any(word in normalized for word in cls.NEGATIVE_HINTS)
        if "okay" in normalized or "fine" in normalized:
            return SentimentAnalysis(sentiment="mixed", confidence=0.73)
        if positive and negative:
            return SentimentAnalysis(sentiment="mixed", confidence=0.78)
        if negative:
            return SentimentAnalysis(sentiment="negative", confidence=0.76)
        if positive:
            return SentimentAnalysis(sentiment="positive", confidence=0.76)
        return SentimentAnalysis(sentiment="mixed", confidence=0.5)

    @classmethod
    def extract_feedback(cls, message: str) -> FeedbackExtraction:
        lower = message.lower()
        parts = [part.strip() for part in message.replace("\n", ". ").split(".") if part.strip()]
        positives = [part for part in parts if any(word in part.lower() for word in cls.POSITIVE_HINTS)]
        negatives = [part for part in parts if any(word in part.lower() for word in cls.NEGATIVE_HINTS)]
        suggestions = [
            part
            for part in parts
            if any(token in part.lower() for token in {"should", "could", "improve", "better", "easier", "add", "make"})
        ]

        if "realistic" in lower and any(token in lower for token in {"more realistic", "not realistic", "unrealistic"}):
            tag_hint = "lack_of_realism"
        elif any(token in lower for token in {"missed the vibe", "not emotionally warm", "not warm", "not cozy", "felt flat"}):
            tag_hint = "emotional_tone_mismatch"
        else:
            tag_hint = None

        issue_tags = cls.generate_issue_tags(message)
        if tag_hint and tag_hint not in issue_tags:
            issue_tags.insert(0, tag_hint)

        if not positives and not negatives and not suggestions and parts and len(parts[0].split()) >= 3:
            negatives = [parts[0]]

        sentiment = cls.analyze_sentiment(message).sentiment
        return FeedbackExtraction(
            sentiment=sentiment,
            positives=positives[:5],
            negatives=negatives[:5],
            suggestions=suggestions[:5],
            issue_tags=issue_tags[:8],
        )

    @classmethod
    def classify_issue(cls, message: str) -> IssueClassification:
        normalized = message.lower()
        if any(word in normalized for word in cls.TECHNICAL_HINTS | {"froze"}):
            return IssueClassification(issue_type="technical", rationale="Technical keywords detected.")
        if any(word in normalized for word in cls.USABILITY_HINTS):
            return IssueClassification(issue_type="usability", rationale="Usability-related language detected.")
        if any(word in normalized for word in cls.QUALITY_HINTS):
            return IssueClassification(issue_type="quality", rationale="Output quality concerns detected.")
        return IssueClassification(issue_type="none", rationale="No clear issue category was detected.")

    @classmethod
    def generate_issue_tags(cls, message: str) -> list[str]:
        normalized = message.lower()
        tags: list[str] = []

        def add(tag: str) -> None:
            if tag not in tags:
                tags.append(tag)

        if any(token in normalized for token in {"navigation", "hard to find", "difficult to find"}):
            add("navigation_difficulty")
        if any(token in normalized for token in {"slow", "minutes", "delay", "took"}):
            add("slow_response_time")
        if any(token in normalized for token in {"realistic", "unrealistic"}):
            add("lack_of_realism")
        if any(token in normalized for token in {"add more", "nature elements", "more nature"}):
            add("nature_element_request")
        if any(token in normalized for token in {"generate 2 images", "two images", "multiple images"}):
            add("multiple_output_request")
        if any(token in normalized for token in {"crash", "error", "freeze", "broken"}):
            add("runtime_failure")
        if any(token in normalized for token in {"incorrect", "inaccurate", "wrong"}):
            add("accuracy_problem")
        if any(token in normalized for token in {"color theme", "colour scheme", "color scheme"}):
            add("visual_style_feedback")
        if any(token in normalized for token in {"sober", "tone down", "less flashy"}):
            add("visual_tone_adjustment")
        if any(token in normalized for token in {"missed the vibe", "not emotionally warm", "not warm", "not cozy", "felt flat"}):
            add("emotional_tone_mismatch")
        if any(token in normalized for token in {"cinematic and alive", "real cafe", "movie"}):
            add("cinematic_atmosphere_request")
        if any(token in normalized for token in {"sharpness", "blurry", "material", "materials", "reflections", "proportions", "product quality"}):
            add("technical_product_realism")
        if any(token in normalized for token in {"tropical island", "aerial", "drone", "environmental scale", "water texture", "water textures", "perspective"}):
            add("environmental_aerial_realism")

        return tags[:8]

    @classmethod
    def generate_human_followup_question(
        cls,
        *,
        task_type: str,
        prompt: str,
        ai_output: str,
        user_feedback: str,
        existing_negatives: list[str],
        existing_suggestions: list[str],
        previous_followups: list[str],
        detected_issue_type: str,
        grounding_context: str = "",
    ) -> HumanFollowupQuestionResult:
        normalized_feedback = user_feedback.lower()
        style_mismatch = cls._grounding_mismatch(grounding_context, "style")
        lighting_mismatch = cls._grounding_mismatch(grounding_context, "lighting")
        tone_mismatch = cls._grounding_mismatch(grounding_context, "tone")
        emotional_mismatch = cls._grounding_mismatch(grounding_context, "emotional_tone")
        quality_mismatch = cls._grounding_mismatch(grounding_context, "quality")
        if "latest_user_correction=technical realism" in grounding_context.lower():
            question = "So this was a technical realism issue rather than an atmosphere issue?"
        elif emotional_mismatch:
            expected, _ = emotional_mismatch
            question = f"So the emotional tone needed to feel more {expected}?"
        elif style_mismatch:
            expected, _ = style_mismatch
            question = f"So the style felt wrong because you wanted {expected}?"
        elif lighting_mismatch:
            expected, _ = lighting_mismatch
            question = f"You wanted the lighting closer to {expected}?"
        elif tone_mismatch:
            expected, _ = tone_mismatch
            question = f"So the tone shifted away from {expected}?"
        elif quality_mismatch:
            expected, _ = quality_mismatch
            question = f"Was the main issue that it needed {expected}?"
        elif detected_issue_type == "technical":
            question = "What happened right before it failed?"
        elif detected_issue_type == "quality":
            question = f"What felt most off about the {task_type or 'result'}?"
        elif detected_issue_type == "usability":
            question = "Which step felt the most confusing?"
        elif prompt and ai_output and not cls._looks_aligned(prompt, ai_output):
            question = "What part of the result missed your prompt?"
        elif existing_suggestions:
            question = "Which change would help the most first?"
        elif existing_negatives or any(token in normalized_feedback for token in {"not", "didn't", "wrong", "off"}):
            question = "What stood out as the main problem?"
        else:
            question = f"What worked best in the {task_type or 'output'}?"

        if question in previous_followups:
            question = "What one detail should we capture next?"

        return HumanFollowupQuestionResult(questions=[question])

    @staticmethod
    def _grounding_mismatch(grounding_context: str, category: str) -> tuple[str, str] | None:
        marker = f"{category}: wanted "
        normalized = grounding_context.lower()
        if marker not in normalized or ", got " not in normalized:
            return None

        rest = normalized.split(marker, 1)[1]
        expected, observed_part = rest.split(", got ", 1)
        observed = observed_part.split("'", 1)[0].split("]", 1)[0].strip()
        return expected.strip(), observed

    @classmethod
    def generate_feedback_insights(
        cls,
        *,
        negatives: list[str],
        suggestions: list[str],
        issue_tags: list[str],
    ) -> FeedbackInsightsResult:
        if not negatives and not suggestions and not issue_tags:
            return FeedbackInsightsResult(
                summary="There is not enough feedback yet to identify meaningful product patterns.",
                top_problems=[],
                improvement_suggestions=[],
            )

        top_problems = _top_items(issue_tags or negatives, limit=3)
        improvements = _top_items(suggestions or issue_tags, limit=3)
        summary = (
            f"Users most often mention {', '.join(top_problems)}."
            if top_problems
            else "Users are sharing early signals, but patterns are still emerging."
        )
        return FeedbackInsightsResult(
            summary=summary,
            top_problems=top_problems,
            improvement_suggestions=improvements,
        )

    @staticmethod
    def _looks_aligned(prompt: str, ai_output: str) -> bool:
        prompt_words = {word for word in prompt.lower().split() if len(word) > 3}
        output_words = {word for word in ai_output.lower().split() if len(word) > 3}
        if not prompt_words or not output_words:
            return True
        overlap = prompt_words & output_words
        return len(overlap) >= min(3, max(1, len(prompt_words) // 5))


def _top_items(items: list[str], limit: int) -> list[str]:
    counts: dict[str, int] = {}
    for item in items:
        cleaned = item.strip()
        if not cleaned:
            continue
        counts[cleaned] = counts.get(cleaned, 0) + 1
    return [item for item, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]


class FeedbackLLMService:
    def __init__(self) -> None:
        self.factory = StructuredChainFactory()
        self._turn_analysis_chain = self._build_optional_chain(TURN_ANALYSIS_PROMPT, ConversationTurnAnalysis)
        self._intent_chain = self._build_optional_chain(INTENT_PROMPT, IntentClassification)
        self._rating_chain = self._build_optional_chain(RATING_PROMPT, RatingExtraction)
        self._sentiment_chain = self._build_optional_chain(SENTIMENT_PROMPT, SentimentAnalysis)
        self._extraction_chain = self._build_optional_chain(FEEDBACK_EXTRACTION_PROMPT, FeedbackExtraction)
        self._issue_chain = self._build_optional_chain(ISSUE_CLASSIFICATION_PROMPT, IssueClassification)
        self._tag_chain = self._build_optional_chain(ISSUE_TAG_PROMPT, IssueTagResult)
        self._human_followup_chain = self._build_optional_chain(HUMAN_FOLLOWUP_PROMPT, HumanFollowupQuestionResult)
        self._feedback_insights_chain = self._build_optional_chain(FEEDBACK_INSIGHTS_PROMPT, FeedbackInsightsResult)

    def _build_optional_chain(self, prompt: str, schema: Type[T]) -> RunnableSerializable | None:
        if not self.factory.enabled:
            return None
        return self.factory.build_chain(prompt, schema)

    def analyze_turn(self, message: str) -> ConversationTurnAnalysis:
        if self._turn_analysis_chain:
            return self._turn_analysis_chain.invoke({"message": message})
        return FallbackClassifier.analyze_turn(message)

    def classify_intent(self, message: str) -> IntentClassification:
        if self._intent_chain:
            return self._intent_chain.invoke({"message": message})
        return FallbackClassifier.classify_intent(message)

    def extract_rating(self, message: str) -> RatingExtraction:
        if self._rating_chain:
            return self._rating_chain.invoke({"message": message})
        return FallbackClassifier.extract_rating(message)

    def analyze_sentiment(self, message: str) -> SentimentAnalysis:
        if self._sentiment_chain:
            return self._sentiment_chain.invoke({"message": message})
        return FallbackClassifier.analyze_sentiment(message)

    def extract_feedback(self, message: str) -> FeedbackExtraction:
        if self._extraction_chain:
            extraction = self._extraction_chain.invoke({"message": message})
            if not extraction.issue_tags:
                extraction.issue_tags = self.generate_issue_tags(message)
            return extraction
        return FallbackClassifier.extract_feedback(message)

    def classify_issue(self, message: str) -> IssueClassification:
        if self._issue_chain:
            return self._issue_chain.invoke({"message": message})
        return FallbackClassifier.classify_issue(message)

    def generate_issue_tags(self, message: str) -> list[str]:
        if self._tag_chain:
            return self._tag_chain.invoke({"message": message}).issue_tags
        return FallbackClassifier.generate_issue_tags(message)

    def generate_human_followup_question(self, **payload) -> HumanFollowupQuestionResult:
        if self._human_followup_chain:
            return self._human_followup_chain.invoke(payload)
        return FallbackClassifier.generate_human_followup_question(**payload)

    def generate_feedback_insights(self, **payload) -> FeedbackInsightsResult:
        if self._feedback_insights_chain:
            return self._feedback_insights_chain.invoke(payload)
        return FallbackClassifier.generate_feedback_insights(**payload)
