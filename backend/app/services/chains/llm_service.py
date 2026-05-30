from __future__ import annotations

import re
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
        "beautiful",
        "strong",
        "worked",
        "appealing",
        "professional",
        "cinematic",
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
        "empty",
        "lacked",
        "lacks",
        "density",
        "static",
        "artificial",
        "fake",
        "staged",
        "unnatural",
        "weak",
        "hollow",
        "unreadable",
        "illegible",
        "label",
        "labels",
        "chart",
        "graph",
        "logo",
        "typography",
        "ignored",
        "instruction",
        "instructions",
        "monitor",
        "monitors",
        "headphones",
        "sticky notes",
        "desk plant",
        "objects",
        "elements",
        "contained",
    }
    VAGUE_RATING_HINTS = {"okay", "ok", "fine", "average", "decent", "not bad", "so so", "its okay", "it's okay"}
    STOP_HINTS = {"no", "no more", "stop", "that's it", "thats it", "nothing else", "no thanks"}
    OFF_TOPIC_HINTS = {
        "what should i wear",
        "what should i eat",
        "do you like",
        "tell me a joke",
        "recommend a restaurant",
        "restaurant",
        "outfit",
        "denim",
        "jeans",
        "fashion",
        "how are you",
        "what is your name",
        "what's your name",
        "who are you",
    }
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
        "realism",
        "empty",
        "lacked",
        "lacks",
        "density",
        "static",
        "artificial",
        "fake",
        "staged",
        "unnatural",
        "unreadable",
        "illegible",
        "label",
        "labels",
        "chart",
        "graph",
        "logo",
        "typography",
        "readability",
        "ignored",
        "instruction",
        "instructions",
        "monitor",
        "monitors",
        "headphones",
        "sticky notes",
        "desk plant",
        "objects",
        "elements",
        "contained",
        "followed",
    }
    USABILITY_HINTS = {"hard", "confusing", "confused", "ui", "ux", "difficult", "unclear", "find", "navigation", "use"}

    @classmethod
    def analyze_turn(cls, message: str) -> ConversationTurnAnalysis:
        if cls._is_off_topic_message(message):
            return ConversationTurnAnalysis(
                intent="off_topic",
                issue_type="none",
                sentiment="mixed",
                is_feedback_present=False,
            )

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
        if any(phrase in normalized for phrase in {"generate 2 images", "two images", "multiple images"}):
            return RatingExtraction(rating=None, is_vague=False, clarification_needed="")
        rating = cls._extract_explicit_rating(normalized)
        if rating is not None:
            return RatingExtraction(rating=rating, is_vague=False, clarification_needed="")
        if cls._has_vague_rating_hint(normalized):
            return RatingExtraction(
                rating=None,
                is_vague=True,
                clarification_needed="Please share a rating from 1 to 5, where 1 is very poor and 5 is excellent.",
            )

        if cls._has_rating_cue(normalized):
            return RatingExtraction(
                rating=None,
                is_vague=True,
                clarification_needed="Please share a rating from 1 to 5, where 1 is very poor and 5 is excellent.",
            )

        return RatingExtraction(rating=None, is_vague=False, clarification_needed="")

    @staticmethod
    def _has_rating_cue(normalized: str) -> bool:
        return bool(
            re.search(r"\b(?:rate|rating|score|stars?)\b|/5|\bout\s+of\s+5\b", normalized)
        )

    @classmethod
    def _has_vague_rating_hint(cls, normalized: str) -> bool:
        tokens = {"".join(ch for ch in token if ch.isalnum() or ch == "'") for token in normalized.split()}
        if tokens & {"okay", "ok", "fine", "average", "decent"}:
            return True
        return bool(re.search(r"\bnot\s+bad\b|\bso\s+so\b|\bits\s+okay\b|\bit's\s+okay\b", normalized))

    @staticmethod
    def _extract_explicit_rating(normalized: str) -> int | None:
        patterns = [
            r"^\s*(?:probably|maybe|around|about)?\s*(?:a\s+)?([1-5])(?:\b|(?=\s*/\s*5)|(?=\s*[-.:]))",
            r"\b(?:rate|rating|score|stars?|give|gave|giving)\s+(?:it\s+)?(?:a\s+)?([1-5])(?:\b|(?=\s*/\s*5))",
            r"\b(?:would|would probably|probably|maybe)\s+(?:give|rate)\s+(?:it\s+)?(?:a\s+)?([1-5])(?:\b|(?=\s*/\s*5))",
            r"\b([1-5])\s*/\s*5\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                return int(match.group(1))
        return None

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
        if cls._is_off_topic_message(message):
            return FeedbackExtraction(sentiment="mixed", positives=[], negatives=[], suggestions=[], issue_tags=[])

        clauses = cls._feedback_clauses(message)
        positives: list[str] = []
        negatives: list[str] = []
        suggestions: list[str] = []

        for clause in clauses:
            normalized = clause.lower()
            cleaned = cls._clean_feedback_phrase(clause)
            if not cleaned:
                continue

            is_suggestion = cls._has_suggestion_signal(normalized)
            is_negative = cls._has_negative_signal(normalized)
            is_positive = cls._has_positive_signal(normalized)

            if is_suggestion:
                suggestions.append(cleaned)
            if is_negative or (is_suggestion and any(token in normalized for token in {"missing", "lacked", "lacks", "needed", "needs"})):
                negatives.append(cleaned)
            elif is_positive:
                positives.append(cleaned)

        if (
            not positives
            and not negatives
            and not suggestions
            and clauses
            and len(clauses[0].split()) >= 3
            and cls._has_feedback_signal(message)
        ):
            negatives = [cls._clean_feedback_phrase(clauses[0])]

        return cls.refine_feedback(
            message,
            FeedbackExtraction(
                sentiment=cls.analyze_sentiment(message).sentiment,
                positives=positives[:5],
                negatives=negatives[:5],
                suggestions=suggestions[:5],
                issue_tags=cls.generate_issue_tags(message),
            ),
        )

    @classmethod
    def refine_feedback(cls, message: str, extraction: FeedbackExtraction) -> FeedbackExtraction:
        positives: list[str] = []
        negatives: list[str] = []
        suggestions: list[str] = []

        for item in extraction.positives:
            cleaned = cls._clean_feedback_phrase(item)
            lowered = cleaned.lower()
            if not cleaned:
                continue
            if cls._has_negative_signal(lowered) or cls._has_suggestion_signal(lowered):
                if cls._has_suggestion_signal(lowered):
                    suggestions.append(cleaned)
                negatives.append(cleaned)
            else:
                positives.append(cleaned)

        for item in extraction.negatives:
            cleaned = cls._clean_feedback_phrase(item)
            if cleaned:
                negatives.append(cleaned)

        for item in extraction.suggestions:
            cleaned = cls._clean_feedback_phrase(item)
            if cleaned:
                suggestions.append(cleaned)

        inferred = cls._infer_structured_items(message)
        positives = cls._dedupe_preserve([*positives, *inferred["positives"]])[:5]
        negatives = cls._dedupe_preserve([*negatives, *inferred["negatives"]])[:6]
        suggestions = cls._dedupe_preserve([*suggestions, *inferred["suggestions"]])[:5]
        issue_tags = (
            cls._best_issue_tags(message, positives, negatives, suggestions, extraction.issue_tags)
            if negatives or suggestions
            else []
        )
        sentiment = cls._structured_sentiment(positives, negatives, suggestions)

        return FeedbackExtraction(
            sentiment=sentiment,
            positives=positives,
            negatives=negatives,
            suggestions=suggestions,
            issue_tags=issue_tags,
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
        if cls._is_off_topic_message(message):
            return []
        return cls._best_issue_tags(message, [], [], [], [])

    @classmethod
    def _is_off_topic_message(cls, message: str) -> bool:
        normalized = message.strip().lower().strip("?.! ")
        if not normalized:
            return False
        feedback_terms = {
            "ai",
            "app",
            "application",
            "feedback",
            "generated",
            "image",
            "output",
            "prompt",
            "result",
            "quality",
            "realism",
            "realistic",
            "unrealistic",
            "rating",
            "score",
            "color",
            "colors",
            "lighting",
            "composition",
            "style",
            "texture",
            "subject",
            "person",
            "scene",
            "readability",
        }
        if any(term in normalized for term in feedback_terms):
            return False
        if re.search(r"\bi\s+(?:like|love|prefer|enjoy|hate|dislike)\b", normalized):
            return True
        return any(hint in normalized for hint in cls.OFF_TOPIC_HINTS)

    @classmethod
    def _best_issue_tags(
        cls,
        message: str,
        positives: list[str],
        negatives: list[str],
        suggestions: list[str],
        existing_tags: list[str],
    ) -> list[str]:
        normalized = " ".join([message, *negatives, *suggestions]).lower()
        tags: list[str] = []

        def add(tag: str) -> None:
            if tag not in tags:
                tags.append(tag)

        def has_any(tokens: set[str]) -> bool:
            return any(cls._evidence_contains(normalized, token) for token in tokens)

        blocked_generic = {
            "lack_of_realism",
            "realism_issue",
            "visual_quality",
            "quality_problem",
            "technical_product_realism",
            "environmental_aerial_realism",
            "emotional_tone_mismatch",
            "cinematic_atmosphere_request",
            "visual_style_feedback",
        }
        for tag in existing_tags:
            if tag and tag not in blocked_generic and cls._tag_supported_by_evidence(tag, normalized):
                add(tag)

        if has_any({"unreadable", "illegible", "hard to read", "could not read", "can't read", "text"}):
            add("text_rendering")
            add("readability")
            add("typography_fidelity")
        if has_any({"label", "labels", "caption", "captions", "annotation", "annotations"}):
            add("label_fidelity")
            if has_any({"incorrect", "inaccurate", "wrong", "mislabeled", "mislabelled"}):
                add("information_accuracy")
                add("content_correctness")
        if has_any({"chart", "charts", "graph", "graphs", "axis", "axes", "legend", "plot"}):
            add("data_visualization_quality")
            add("chart_readability")
        if any(token in normalized for token in {"logo", "brand", "branding", "wordmark"}):
            add("branding_fidelity")
            add("logo_accuracy")
        if any(token in normalized for token in {"subject", "main subject", "not what i asked", "not what i asked for", "did not match", "does not match"}):
            add("subject_mismatch")
        if any(token in normalized for token in {"person", "character"}) and any(token in normalized for token in {"did not match", "does not match", "wrong"}):
            add("character_mismatch")
        if "japanese" in normalized and any(token in normalized for token in {"person did not match", "did not match", "wrong person"}):
            add("identity_mismatch")
        if "elderly" in normalized and any(token in normalized for token in {"person did not match", "did not match", "wrong person", "not what i asked"}):
            add("age_mismatch")
        if any(token in normalized for token in {"description", "attribute", "attributes", "did not match", "does not match"}):
            add("attribute_mismatch")
        if any(token in normalized for token in {"missing", "missed", "ignored", "left out", "left-out", "omitted", "not included"}):
            if any(token in normalized for token in {"object", "objects", "element", "elements", "monitor", "monitors", "headphones", "sticky notes", "desk plant"}):
                add("missing_objects")
                add("missing_elements")
            add("prompt_adherence")
            add("instruction_following")
        if any(token in normalized for token in {"instruction", "instructions", "followed", "prompt", "should have contained", "asked for", "requested"}):
            add("prompt_adherence")
            add("instruction_following")
        if any(token in normalized for token in {"object count", "count", "three monitors", "two monitors", "too few", "too many"}):
            add("object_count_errors")
        if any(token in normalized for token in {"navigation", "hard to find", "difficult to find"}):
            add("navigation_difficulty")
        if any(token in normalized for token in {"slow", "minutes", "delay", "took"}):
            add("slow_response_time")
        if any(token in normalized for token in {"generate 2 images", "two images", "multiple images"}):
            add("multiple_output_request")
        if any(token in normalized for token in {"crash", "error", "freeze", "broken"}):
            add("runtime_failure")
        if any(token in normalized for token in {"missed prompt", "prompt", "did not follow", "didn't follow"}):
            add("prompt_alignment")
        if any(token in normalized for token in {"empty", "density", "crowd", "crowds", "busy", "alive", "activity", "drones", "traffic"}):
            add("environmental_density")
        realism_complaint = (
            any(token in normalized for token in {"unrealistic", "not realistic", "fake", "artificial"})
            or ("realism" in normalized and any(token in normalized for token in {"missing", "lacked", "weak", "breaks", "issue", "problem"}))
        )
        if any(token in normalized for token in {"environment", "city", "street", "scene", "underwater", "water", "ocean", "believable"}) or realism_complaint:
            add("environmental_realism")
        if any(token in normalized for token in {"motion", "movement", "moving", "static", "frozen", "action"}):
            add("motion_realism")
        if any(token in normalized for token in {"lighting", "light", "shadow", "falloff", "glow", "neon"}):
            add("lighting_consistency")
        if any(token in normalized for token in {"texture", "surface", "grain", "artificial texture"}):
            add("texture_realism")
        if any(token in normalized for token in {"material", "metal", "metallic", "fabric", "skin", "plastic"}):
            add("material_realism")
        if any(token in normalized for token in {"reflection", "reflections", "reflective", "mirror"}):
            add("reflection_realism")
        if any(token in normalized for token in {"scale", "massive", "size", "proportion", "proportions"}):
            add("scale_consistency")
        if any(token in normalized for token in {"depth", "depth cues", "atmospheric", "haze", "distance", "falloff"}):
            add("atmospheric_depth")
        if any(token in normalized for token in {"interaction", "integrated", "integration", "natural behavior", "unnatural"}):
            add("interaction_realism")
        if any(token in normalized for token in {"composition", "framing", "balance", "layout"}):
            add("composition_balance")
        if any(token in normalized for token in {"perspective", "aerial", "angle", "vanishing"}):
            add("perspective_consistency")
        if any(token in normalized for token in {"anatomy", "body", "limb", "face", "hand", "pose"}):
            add("anatomy_accuracy")
        if any(token in normalized for token in {"cinematic", "film", "mood", "atmosphere", "cyberpunk", "neon", "visual style", "style looked strong"}):
            add("cinematic_alignment")
        if any(token in normalized for token in {"sharp", "sharpness", "blurry", "detail", "details"}):
            add("detail_sharpness")

        if not tags and realism_complaint:
            add("environmental_believability")

        return tags[:5]

    @staticmethod
    def _tag_supported_by_evidence(tag: str, normalized: str) -> bool:
        evidence_by_tag = {
            "text_rendering": {"text", "unreadable", "illegible", "read", "font", "type", "typography"},
            "typography_fidelity": {"text", "font", "type", "typography", "unreadable", "illegible"},
            "readability": {"read", "readable", "unreadable", "illegible", "text", "label"},
            "information_accuracy": {"incorrect", "inaccurate", "wrong", "label", "labels", "content", "data"},
            "label_fidelity": {"label", "labels", "caption", "annotation", "mislabeled", "mislabelled"},
            "content_correctness": {"incorrect", "inaccurate", "wrong", "content", "label", "labels"},
            "data_visualization_quality": {"chart", "charts", "graph", "graphs", "axis", "axes", "legend", "plot", "data"},
            "chart_readability": {"chart", "charts", "graph", "graphs", "axis", "axes", "legend"},
            "branding_fidelity": {"logo", "brand", "branding", "wordmark"},
            "logo_accuracy": {"logo", "brand", "branding", "wordmark"},
            "subject_mismatch": {"subject", "main subject", "not what i asked", "not what i asked for", "did not match", "does not match"},
            "character_mismatch": {"person", "character", "man", "woman", "did not match", "description"},
            "identity_mismatch": {"identity", "japanese", "wrong person", "person did not match"},
            "age_mismatch": {"elderly", "old", "young", "age", "aged"},
            "attribute_mismatch": {"description", "attribute", "attributes", "did not match"},
            "prompt_adherence": {"missing", "missed", "ignored", "instruction", "instructions", "followed", "prompt", "requested", "asked for", "should have contained"},
            "missing_objects": {"missing", "missed", "ignored", "left out", "omitted", "object", "objects", "monitor", "monitors", "headphones", "sticky notes", "desk plant"},
            "missing_elements": {"missing", "missed", "ignored", "left out", "omitted", "element", "elements", "object", "objects"},
            "instruction_following": {"ignored", "instruction", "instructions", "followed", "prompt", "requested", "asked for", "should have contained"},
            "object_count_errors": {"object count", "count", "three monitors", "two monitors", "too few", "too many"},
            "environmental_density": {"empty", "density", "crowd", "crowds", "busy", "alive", "activity", "drones", "traffic"},
            "environmental_realism": {"environment", "city", "street", "scene", "underwater", "water", "ocean", "believable", "realism", "realistic", "unrealistic"},
            "environmental_believability": {"environment", "city", "street", "scene", "believable", "realism", "realistic", "unrealistic"},
            "motion_realism": {"motion", "movement", "moving", "static", "frozen", "action"},
            "lighting_consistency": {"lighting", "light", "shadow", "falloff", "glow", "neon"},
            "texture_realism": {"texture", "surface", "grain"},
            "material_realism": {"material", "metal", "metallic", "fabric", "skin", "plastic"},
            "reflection_realism": {"reflection", "reflections", "reflective", "mirror"},
            "scale_consistency": {"scale", "massive", "size", "proportion", "proportions"},
            "atmospheric_depth": {"depth", "atmospheric", "haze", "distance", "falloff"},
            "interaction_realism": {"interaction", "integrated", "integration", "natural behavior", "unnatural"},
            "composition_balance": {"composition", "framing", "balance", "layout"},
            "perspective_consistency": {"perspective", "aerial", "angle", "vanishing"},
            "anatomy_accuracy": {"anatomy", "body", "limb", "face", "hand", "pose"},
            "cinematic_alignment": {"cinematic", "film", "mood", "atmosphere", "cyberpunk", "neon", "visual style", "style"},
            "detail_sharpness": {"sharp", "sharpness", "blurry", "detail", "details"},
            "prompt_alignment": {"prompt", "missed prompt", "did not follow", "didn't follow", "ignored"},
            "navigation_difficulty": {"navigation", "hard to find", "difficult to find"},
            "slow_response_time": {"slow", "minutes", "delay", "took"},
            "runtime_failure": {"crash", "error", "freeze", "broken"},
            "multiple_output_request": {"generate 2 images", "two images", "multiple images"},
        }
        allowed_terms = evidence_by_tag.get(tag)
        if not allowed_terms:
            return tag == "other"
        return any(FallbackClassifier._evidence_contains(normalized, term) for term in allowed_terms)

    @staticmethod
    def _evidence_contains(evidence_text: str, token: str) -> bool:
        if " " in token or "-" in token:
            return token in evidence_text
        return bool(re.search(rf"\b{re.escape(token)}\b", evidence_text))

    @classmethod
    def _feedback_clauses(cls, message: str) -> list[str]:
        text = message.replace("\n", ". ")
        for marker in [" but ", " however ", " though ", " although ", " while "]:
            text = text.replace(marker, ". ")
        return [part.strip(" ,.;:-") for part in text.split(".") if part.strip(" ,.;:-")]

    @classmethod
    def _has_positive_signal(cls, text: str) -> bool:
        return any(word in text for word in cls.POSITIVE_HINTS) or any(
            phrase in text
            for phrase in {
                "worked well",
                "looked strong",
                "looked beautiful",
                "was appealing",
                "felt professional",
            }
        )

    @classmethod
    def _has_negative_signal(cls, text: str) -> bool:
        return text.startswith("not ") or " not " in text or any(word in text for word in cls.NEGATIVE_HINTS) or any(
            phrase in text
            for phrase in {
                "did not feel",
                "didn't feel",
                "does not feel",
                "felt off",
                "felt empty",
                "felt static",
                "not integrated",
                "not massive",
            }
        )

    @classmethod
    def _has_suggestion_signal(cls, text: str) -> bool:
        if cls._is_observation_with_modal(text):
            return False
        return any(
            token in text
            for token in {
                "improve",
                "better",
                "add",
                "make",
                "more ",
                "stronger",
                "improved",
                "needed",
                "needs",
                "moving crowds",
                "active drones",
                "city activity",
                "environmental density",
            }
        ) or bool(
            re.search(
                r"\b(?:model|system|app|tool|generator|ai)\s+(?:should|could|would need to|needs to)\b",
                text,
            )
        )

    @staticmethod
    def _is_observation_with_modal(text: str) -> bool:
        observation_patterns = {
            "should have contained",
            "should have included",
            "should include",
            "would have contained",
            "could have contained",
            "was supposed to",
            "were supposed to",
            "needed to show",
            "needed to include",
        }
        return any(pattern in text for pattern in observation_patterns)

    @classmethod
    def _has_feedback_signal(cls, text: str) -> bool:
        normalized = text.lower()
        return (
            cls._has_positive_signal(normalized)
            or cls._has_negative_signal(normalized)
            or cls._has_suggestion_signal(normalized)
            or any(
                token in normalized
                for token in {
                    "image",
                    "output",
                    "result",
                    "scene",
                    "looked",
                    "felt",
                    "experience",
                    "feedback",
                    "ignored",
                    "instruction",
                    "instructions",
                    "contained",
                    "missing",
                    "objects",
                    "elements",
                }
            )
        )

    @staticmethod
    def _clean_feedback_phrase(text: str) -> str:
        cleaned = " ".join(text.strip(" ,.;:-").split())
        cleaned = re.sub(r"^(?:probably\s+|maybe\s+|around\s+|about\s+)?(?:a\s+)?[1-5](?:\s*/\s*5)?\s*(?:because|[-.:])?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        for prefix in ("but ", "and ", "also ", "the "):
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        return cleaned

    @classmethod
    def _infer_structured_items(cls, message: str) -> dict[str, list[str]]:
        normalized = message.lower()
        positives: list[str] = []
        negatives: list[str] = []
        suggestions: list[str] = []

        if any(token in normalized for token in {"visual style looked strong", "style looked strong", "strong visual style"}):
            positives.append("strong visual style")
        if any(token in normalized for token in {"neon", "cyberpunk"}) and any(token in normalized for token in {"strong", "worked", "beautiful", "appealing"}):
            positives.append("neon atmosphere worked well")
        if "composition" in normalized and any(token in normalized for token in {"beautiful", "worked", "cinematic", "professional"}):
            positives.append("composition worked well")
        if any(token in normalized for token in {"colors looked beautiful", "color palette", "underwater colors"}):
            positives.append("color palette worked well")

        if "empty" in normalized:
            negatives.append("scene felt empty")
        if any(token in normalized for token in {"unreadable", "illegible", "hard to read", "could not read", "can't read"}):
            negatives.append("text was unreadable")
        if "text" in normalized and any(token in normalized for token in {"wrong", "incorrect", "inaccurate"}):
            negatives.append("text content was incorrect")
        if any(token in normalized for token in {"label", "labels"}) and any(token in normalized for token in {"wrong", "incorrect", "inaccurate", "mislabeled", "mislabelled"}):
            negatives.append("labels were incorrect")
        if any(token in normalized for token in {"chart", "charts", "graph", "graphs"}) and any(token in normalized for token in {"unclear", "unreadable", "confusing", "wrong", "incorrect"}):
            negatives.append("chart was hard to interpret")
        if any(token in normalized for token in {"logo", "brand", "branding"}) and any(token in normalized for token in {"wrong", "incorrect", "off", "inaccurate"}):
            negatives.append("branding was inaccurate")
        if any(token in normalized for token in {"main subject", "subject", "person"}) and any(token in normalized for token in {"not what i asked", "not what i asked for", "did not match", "does not match"}):
            negatives.append("subject did not match the prompt")
        if "person did not match" in normalized or "person does not match" in normalized:
            negatives.append("person did not match the prompt description")
        if any(token in normalized for token in {"missing", "missed", "ignored", "left out", "omitted"}):
            if any(token in normalized for token in {"monitor", "monitors", "headphones", "sticky notes", "desk plant", "object", "objects", "element", "elements"}):
                negatives.append("requested objects were missing")
            if any(token in normalized for token in {"instruction", "instructions", "prompt", "requested", "asked for"}):
                negatives.append("prompt instructions were not fully followed")
        if "should have contained" in normalized:
            negatives.append(cls._clean_feedback_phrase(message))
        if "density" in normalized or "crowd" in normalized or "crowds" in normalized:
            negatives.append("lacked environmental density")
        if any(token in normalized for token in {"motion", "movement", "moving crowds", "active drones", "static"}):
            negatives.append("motion felt static")
        if any(token in normalized for token in {"unrealistic", "not realistic"}) or (
            "realism" in normalized and any(token in normalized for token in {"weak", "issue", "problem", "breaks", "missing"})
        ):
            negatives.append("realism weakened")
        if "texture" in normalized:
            negatives.append("textures looked artificial")
        if "scale" in normalized or "massive" in normalized:
            negatives.append("scale felt unrealistic")
        if "falloff" in normalized:
            negatives.append("lighting lacked falloff")
        if "reflection" in normalized:
            negatives.append("reflections felt fake")
        if "integrated" in normalized or "integration" in normalized:
            negatives.append("environmental integration felt unnatural")

        if any(token in normalized for token in {"moving crowds", "active drones", "more crowds", "more activity"}):
            suggestions.append("more visible city activity")
        if "motion" in normalized and "density" in normalized:
            suggestions.append("more motion and environmental density")
        if "density" in normalized:
            suggestions.append("better environmental density")
        if "falloff" in normalized:
            suggestions.append("stronger lighting falloff")
        if "reflection" in normalized:
            suggestions.append("improved reflections")
        if "texture" in normalized:
            suggestions.append("better texture realism")
        if "depth" in normalized:
            suggestions.append("stronger depth cues")

        return {"positives": positives, "negatives": negatives, "suggestions": suggestions}

    @staticmethod
    def _dedupe_preserve(items: list[str]) -> list[str]:
        deduped: list[str] = []
        for item in items:
            cleaned = " ".join(str(item).strip().split())
            if cleaned and cleaned.lower() not in {existing.lower() for existing in deduped}:
                deduped.append(cleaned)
        return deduped

    @staticmethod
    def _structured_sentiment(positives: list[str], negatives: list[str], suggestions: list[str]) -> str:
        if positives and (negatives or suggestions):
            return "mixed"
        if negatives or suggestions:
            return "negative"
        if positives:
            return "positive"
        return "mixed"

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
            question = "Which concrete detail should change first?"
        elif detected_issue_type == "usability":
            question = "Which step felt the most confusing?"
        elif prompt and ai_output and not cls._looks_aligned(prompt, ai_output):
            question = "What part of the result missed your prompt?"
        elif existing_suggestions:
            question = "Which change would help the most first?"
        elif existing_negatives or any(token in normalized_feedback for token in {"not", "didn't", "wrong", "off"}):
            question = "Which concrete detail matters most here?"
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
            return FallbackClassifier.refine_feedback(message, extraction)
        return FallbackClassifier.extract_feedback(message)

    def classify_issue(self, message: str) -> IssueClassification:
        if self._issue_chain:
            return self._issue_chain.invoke({"message": message})
        return FallbackClassifier.classify_issue(message)

    def generate_issue_tags(self, message: str) -> list[str]:
        if self._tag_chain:
            return FallbackClassifier._best_issue_tags(message, [], [], [], self._tag_chain.invoke({"message": message}).issue_tags)
        return FallbackClassifier.generate_issue_tags(message)

    def generate_human_followup_question(self, **payload) -> HumanFollowupQuestionResult:
        if self._human_followup_chain:
            return self._human_followup_chain.invoke(payload)
        return FallbackClassifier.generate_human_followup_question(**payload)

    def generate_feedback_insights(self, **payload) -> FeedbackInsightsResult:
        if self._feedback_insights_chain:
            return self._feedback_insights_chain.invoke(payload)
        return FallbackClassifier.generate_feedback_insights(**payload)
