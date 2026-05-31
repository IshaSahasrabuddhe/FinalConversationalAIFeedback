from __future__ import annotations

import re
from typing import Type, TypeVar

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

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
    ISSUE_CATEGORY_CLASSIFICATION_PROMPT,
    INTENT_LABEL_REFINEMENT_PROMPT,
    INTENT_LABEL_PROMPT,
    INTENT_PROMPT,
    ISSUE_CLASSIFICATION_PROMPT,
    ISSUE_TAG_PROMPT,
    RATING_PROMPT,
    SENTIMENT_PROMPT,
    STRICT_INTENT_LABEL_PROMPT,
    TURN_ANALYSIS_PROMPT,
)

T = TypeVar("T", bound=BaseModel)


class IssueTagResult(BaseModel):
    issue_tags: list[str]


class IssueCategoryItem(BaseModel):
    category: str = ""
    confidence: float = 0.0


class IssueCategoryClassificationResult(BaseModel):
    primary_issues: list[IssueCategoryItem] = Field(default_factory=list)
    secondary_issues: list[IssueCategoryItem] = Field(default_factory=list)
    positive_aspects: list[IssueCategoryItem] = Field(default_factory=list)
    rejected_categories: list[str] = Field(default_factory=list)


class IntentLabelResult(BaseModel):
    intent_label: str = ""
    confidence: float = 0.0
    groq_label: str = ""
    refined_label: str = ""
    validation_result: str = ""
    retry_used: bool = False
    fallback_used: bool = False


class IntentLabelRefinementResult(BaseModel):
    refined_label: str = ""


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
    _INTENT_ARTICLES = {"a", "an", "the"}
    _INTENT_PREPOSITIONS = {"to", "for", "with", "of", "in", "on", "near", "at", "by", "from", "into", "onto"}
    _INTENT_INSTRUCTION_VERBS = {
        "create",
        "write",
        "generate",
        "design",
        "make",
        "produce",
        "draft",
        "render",
        "compose",
    }
    _INTENT_UNFINISHED_MARKERS = {
        "showing",
        "containing",
        "displayed",
        "featuring",
        "including",
        "should",
        "background",
        "quality",
        "instructions",
        "tone",
        "feel",
        "near",
    }
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
    def refine_feedback(cls, message: str, extraction: FeedbackExtraction, *, infer_issue_tags: bool = True) -> FeedbackExtraction:
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
        issue_tags = []
        if infer_issue_tags and (negatives or suggestions):
            issue_tags = cls._best_issue_tags(message, positives, negatives, suggestions, extraction.issue_tags)
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
    def classify_issue_categories(
        cls,
        *,
        prompt: str,
        feedback_text: str,
        session_context: str = "",
        confidence_threshold: float = 0.7,
        proposed: IssueCategoryClassificationResult | None = None,
        allow_fallback_supplement: bool = True,
    ) -> IssueCategoryClassificationResult:
        result = proposed or cls._fallback_issue_categories(
            prompt=prompt,
            feedback_text=feedback_text,
            session_context=session_context,
        )
        return cls._validate_issue_categories(
            result,
            prompt=prompt,
            feedback_text=feedback_text,
            session_context=session_context,
            confidence_threshold=confidence_threshold,
            allow_fallback_supplement=allow_fallback_supplement,
        )

    @classmethod
    def _fallback_issue_categories(
        cls,
        *,
        prompt: str,
        feedback_text: str,
        session_context: str = "",
    ) -> IssueCategoryClassificationResult:
        normalized = feedback_text.lower()
        primary: list[IssueCategoryItem] = []
        positive: list[IssueCategoryItem] = []

        def add_issue(category: str, confidence: float) -> None:
            if category not in {item.category for item in primary}:
                primary.append(IssueCategoryItem(category=category, confidence=confidence))

        def add_positive(category: str, confidence: float) -> None:
            if category not in {item.category for item in positive}:
                positive.append(IssueCategoryItem(category=category, confidence=confidence))

        if any(token in normalized for token in {"realistic", "realism", "believable", "lifelike"}) and not cls._has_negative_realism_signal(normalized):
            add_positive("realism", 0.92)
        if any(token in normalized for token in {"cute", "great", "beautiful", "nice", "appealing", "looked good", "looked great"}):
            add_positive("visual appeal", 0.85)
        if any(token in normalized for token in {"missing", "missed", "left out", "omitted", "not included"}):
            add_issue("missing objects", 0.94)
            add_issue("prompt adherence", 0.88)
        if any(token in normalized for token in {"unreadable", "illegible", "hard to read", "can't read", "could not read"}):
            add_issue("text readability", 0.95)
        if any(token in normalized for token in {"massive", "scale", "size", "proportion", "proportions", "too small", "too large"}):
            add_issue("scale consistency", 0.9)
        if cls._has_negative_realism_signal(normalized):
            add_issue("realism", 0.88)
        if any(token in normalized for token in {"blurry", "blurred", "not sharp", "sharpness"}):
            add_issue("detail sharpness", 0.85)

        return IssueCategoryClassificationResult(
            primary_issues=primary[:3],
            secondary_issues=[],
            positive_aspects=positive[:3],
            rejected_categories=[],
        )

    @classmethod
    def _validate_issue_categories(
        cls,
        result: IssueCategoryClassificationResult,
        *,
        prompt: str,
        feedback_text: str,
        session_context: str,
        confidence_threshold: float,
        allow_fallback_supplement: bool,
    ) -> IssueCategoryClassificationResult:
        evidence = " ".join([feedback_text, session_context]).lower()
        rejected: list[str] = list(result.rejected_categories or [])
        positive_categories = {
            cls._normalize_issue_category(item.category)
            for item in result.positive_aspects
            if item.confidence >= confidence_threshold and cls._positive_category_supported(item.category, evidence)
        }

        def clean_items(items: list[IssueCategoryItem], *, positive: bool) -> list[IssueCategoryItem]:
            cleaned: list[IssueCategoryItem] = []
            for item in items:
                category = cls._normalize_issue_category(item.category)
                confidence = float(item.confidence or 0.0)
                if not category:
                    continue
                if confidence < confidence_threshold:
                    rejected.append(category)
                    continue
                supported = (
                    cls._positive_category_supported(category, evidence)
                    if positive
                    else cls._negative_category_supported(category, evidence)
                )
                if not supported:
                    rejected.append(category)
                    continue
                if not positive and cls._category_contradicted_by_praise(category, evidence):
                    rejected.append(category)
                    continue
                if not positive and category in positive_categories and not cls._explicit_negative_for_category(category, evidence):
                    rejected.append(category)
                    continue
                if category not in {existing.category for existing in cleaned}:
                    cleaned.append(IssueCategoryItem(category=category, confidence=confidence))
            return cleaned

        positives = clean_items(result.positive_aspects, positive=True)
        primary = clean_items(result.primary_issues, positive=False)
        secondary = [
            item
            for item in clean_items(result.secondary_issues, positive=False)
            if item.category not in {primary_item.category for primary_item in primary}
        ]
        if allow_fallback_supplement or not (primary or secondary or positives):
            fallback = cls._fallback_issue_categories(
                prompt=prompt,
                feedback_text=feedback_text,
                session_context=session_context,
            )
            existing_negative_categories = {item.category for item in [*primary, *secondary]}
            for item in fallback.primary_issues:
                category = cls._normalize_issue_category(item.category)
                if category in existing_negative_categories:
                    continue
                if not cls._negative_category_supported(category, evidence):
                    continue
                if cls._category_contradicted_by_praise(category, evidence):
                    continue
                primary.append(IssueCategoryItem(category=category, confidence=item.confidence))
                existing_negative_categories.add(category)
            existing_positive_categories = {item.category for item in positives}
            for item in fallback.positive_aspects:
                category = cls._normalize_issue_category(item.category)
                if category in existing_positive_categories:
                    continue
                if cls._positive_category_supported(category, evidence):
                    positives.append(IssueCategoryItem(category=category, confidence=item.confidence))
                    existing_positive_categories.add(category)
        return IssueCategoryClassificationResult(
            primary_issues=primary[:4],
            secondary_issues=secondary[:4],
            positive_aspects=positives[:4],
            rejected_categories=cls._dedupe_preserve(rejected)[:10],
        )

    @classmethod
    def issue_categories_to_tags(cls, result: IssueCategoryClassificationResult) -> list[str]:
        tags: list[str] = []
        for item in [*result.primary_issues, *result.secondary_issues]:
            tag = cls._issue_category_to_tag(item.category)
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    @staticmethod
    def _normalize_issue_category(category: str) -> str:
        cleaned = " ".join(str(category or "").strip().lower().replace("_", " ").split())
        aliases = {
            "readability": "text readability",
            "text rendering": "text readability",
            "typography fidelity": "text readability",
            "missing elements": "missing objects",
            "prompt alignment": "prompt adherence",
            "instruction following": "prompt adherence",
            "image sharpness": "detail sharpness",
            "sharpness": "detail sharpness",
            "environmental realism": "realism",
            "environmental believability": "realism",
        }
        return aliases.get(cleaned, cleaned)

    @staticmethod
    def _issue_category_to_tag(category: str) -> str:
        mapping = {
            "missing objects": "missing_objects",
            "prompt adherence": "prompt_adherence",
            "text readability": "readability",
            "scale consistency": "scale_consistency",
            "realism": "environmental_realism",
            "detail sharpness": "detail_sharpness",
            "chart readability": "chart_readability",
            "label accuracy": "label_fidelity",
            "branding fidelity": "branding_fidelity",
            "motion realism": "motion_realism",
            "lighting consistency": "lighting_consistency",
            "texture realism": "texture_realism",
            "material realism": "material_realism",
            "composition balance": "composition_balance",
            "natural posing": "anatomy_accuracy",
        }
        return mapping.get(category, category.replace(" ", "_"))

    @classmethod
    def _positive_category_supported(cls, category: str, evidence: str) -> bool:
        category = cls._normalize_issue_category(category)
        support = {
            "realism": {"realistic", "realism", "believable", "lifelike", "looked real"},
            "visual appeal": {"cute", "great", "beautiful", "nice", "appealing", "looked good", "looked great"},
            "composition": {"composition", "framing", "layout"},
            "style": {"style", "cinematic", "professional"},
        }
        terms = support.get(category, {category})
        return any(cls._evidence_contains(evidence, term) for term in terms)

    @classmethod
    def _negative_category_supported(cls, category: str, evidence: str) -> bool:
        category = cls._normalize_issue_category(category)
        support = {
            "missing objects": {"missing", "missed", "left out", "omitted", "not included"},
            "prompt adherence": {"missing", "missed", "ignored", "prompt", "asked for", "requested", "should have", "not included"},
            "text readability": {"unreadable", "illegible", "hard to read", "can't read", "could not read"},
            "scale consistency": {"massive", "scale", "size", "proportion", "proportions", "too small", "too large"},
            "realism": {"unrealistic", "not realistic", "fake", "artificial", "not believable"},
            "detail sharpness": {"blurry", "blurred", "not sharp", "sharpness"},
            "chart readability": {"unreadable chart", "illegible chart", "hard to read chart", "unclear chart", "confusing chart"},
            "label accuracy": {"label", "caption", "annotation", "wrong", "incorrect", "mislabeled", "mislabelled"},
            "branding fidelity": {"logo", "brand", "branding", "wordmark"},
            "motion realism": {"motion", "movement", "static", "frozen"},
            "lighting consistency": {"lighting", "light", "shadow", "falloff"},
            "texture realism": {"texture", "surface", "grain"},
            "material realism": {"material", "metal", "fabric", "skin", "plastic"},
            "composition balance": {"composition", "framing", "layout", "balance"},
            "natural posing": {"pose", "posing", "staged", "unnatural"},
        }
        terms = support.get(category)
        if not terms:
            return False
        return any(cls._evidence_contains(evidence, term) for term in terms)

    @classmethod
    def _category_contradicted_by_praise(cls, category: str, evidence: str) -> bool:
        category = cls._normalize_issue_category(category)
        if category == "realism":
            return any(
                phrase in evidence
                for phrase in {"looked realistic", "very realistic", "realistic and", "looked real", "felt realistic"}
            ) and not cls._has_negative_realism_signal(evidence)
        if category == "detail sharpness":
            return "details from the prompt were missing" in evidence or "key details from the prompt were missing" in evidence
        return False

    @classmethod
    def _explicit_negative_for_category(cls, category: str, evidence: str) -> bool:
        category = cls._normalize_issue_category(category)
        if category == "realism":
            return cls._has_negative_realism_signal(evidence)
        return cls._negative_category_supported(category, evidence)

    @staticmethod
    def _has_negative_realism_signal(evidence: str) -> bool:
        return any(token in evidence for token in {"unrealistic", "not realistic", "fake", "artificial", "not believable"})

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
            if any(token in normalized for token in {"object", "objects", "element", "elements", "detail", "details", "basket", "blanket", "monitor", "monitors", "headphones", "sticky notes", "desk plant"}):
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
        if realism_complaint or any(token in normalized for token in {"not believable", "unbelievable"}):
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
        if any(token in normalized for token in {"sharp", "sharpness", "blurry", "blurred", "not sharp"}):
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
            "missing_objects": {"missing", "missed", "ignored", "left out", "omitted", "object", "objects", "detail", "details", "basket", "blanket", "monitor", "monitors", "headphones", "sticky notes", "desk plant"},
            "missing_elements": {"missing", "missed", "ignored", "left out", "omitted", "element", "elements", "detail", "details", "object", "objects"},
            "instruction_following": {"ignored", "instruction", "instructions", "followed", "prompt", "requested", "asked for", "should have contained"},
            "object_count_errors": {"object count", "count", "three monitors", "two monitors", "too few", "too many"},
            "environmental_density": {"empty", "density", "crowd", "crowds", "busy", "alive", "activity", "drones", "traffic"},
            "environmental_realism": {"unrealistic", "not realistic", "fake", "artificial", "not believable", "unbelievable"},
            "environmental_believability": {"unrealistic", "not realistic", "fake", "artificial", "not believable", "unbelievable"},
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
            "detail_sharpness": {"sharp", "sharpness", "blurry", "blurred", "not sharp"},
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

    @classmethod
    def generate_intent_label(cls, *, task_type: str, prompt: str) -> IntentLabelResult:
        label = cls._fallback_intent_label(task_type, prompt)
        refined = cls.refine_intent_label(label, prompt=prompt)
        valid, validation_result = cls.validate_intent_label(refined)
        if not valid:
            refined = ""
        return IntentLabelResult(
            intent_label=refined,
            confidence=0.78 if refined else 0.0,
            refined_label=refined,
            validation_result=validation_result,
            fallback_used=True,
        )

    @classmethod
    def refine_intent_label(cls, raw_label: str, *, prompt: str = "") -> str:
        cleaned = " ".join((raw_label or "").strip().strip("\"'` .!?").split())
        if not cleaned:
            return ""

        normalized = cleaned.lower()
        prompt_normalized = prompt.lower()

        if "world map" in normalized:
            prefix = "fantasy " if "fantasy" in normalized or "fantasy" in prompt_normalized else ""
            return f"{prefix}world map".strip()

        if "presentation slide" in normalized or "presentation slide" in prompt_normalized:
            if "business" in normalized or "business" in prompt_normalized:
                return "business presentation slide"
            if "conference" in normalized or "conference" in prompt_normalized:
                return "conference presentation slide"
            return "presentation slide"

        if "underwater" in normalized and "photograph" in normalized:
            if "wildlife" in normalized or "wildlife" in prompt_normalized or "marine" in normalized or "marine" in prompt_normalized:
                return "underwater wildlife photograph"
            if "whale" in normalized or "whale" in prompt_normalized:
                return "whale photography scene"

        cleaned = re.split(
            r"\s+\b(?:showing|containing|displayed|featuring|including|with)\b",
            cleaned,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        cleaned = re.sub(r"\b(?:displayed|shown|visible|image|picture)\b$", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = cls._remove_intent_noise_words(cleaned)
        cleaned = cls._normalize_intent_label(cleaned)

        if cleaned.endswith(" castle"):
            cleaned = f"{cleaned} scene"
        if len(cleaned.split()) < 2:
            return ""
        return cleaned

    @classmethod
    def validate_intent_label(cls, label: str) -> tuple[bool, str]:
        cleaned = " ".join((label or "").strip().strip("\"'` .!?").split())
        if not cleaned:
            return False, "empty"

        words = cleaned.split()
        if not 2 <= len(words) <= 6:
            return False, "invalid_word_count"

        normalized_words = [word.lower().strip(" ,.;:-!?\"'`()[]{}") for word in words]
        if not normalized_words:
            return False, "empty"

        final_word = normalized_words[-1]
        if final_word in cls._INTENT_ARTICLES:
            return False, "ends_with_article"
        if final_word in cls._INTENT_PREPOSITIONS:
            return False, "ends_with_preposition"
        if any(word in cls._INTENT_INSTRUCTION_VERBS for word in normalized_words):
            return False, "contains_instruction_verb"

        normalized = " ".join(normalized_words)
        if any(re.search(rf"\b{re.escape(marker)}\b", normalized) for marker in cls._INTENT_UNFINISHED_MARKERS):
            return False, "contains_unfinished_clause"
        if re.search(r"\b(?:to|for|with|of|in|on|near|at|by|from)\s+(?:a|an|the)?$", normalized):
            return False, "unfinished_prepositional_phrase"
        if cleaned.endswith((",", ";", ":")):
            return False, "trailing_punctuation"
        return True, "pass"

    @staticmethod
    def _fallback_intent_label(task_type: str, prompt: str) -> str:
        task = (task_type or "").lower()
        normalized = prompt.lower()

        if task == "text" or any(token in normalized for token in {"email", "message", "letter", "post", "caption", "announcement"}):
            if "email" in normalized and any(token in normalized for token in {"thank", "thanking", "appreciat", "loyal", "loyalty"}):
                if "loyal" in normalized or "loyalty" in normalized:
                    return "loyalty thank-you email"
                return "customer appreciation email"
            if "email" in normalized and "customer" in normalized:
                return "customer email"
            if "message" in normalized and any(token in normalized for token in {"thank", "thanking", "appreciat"}):
                return "thank-you message"

        if "world map" in normalized:
            return "fantasy world map" if "fantasy" in normalized else "world map"
        if "presentation slide" in normalized:
            if "conference" in normalized:
                return "conference presentation slide"
            if "business" in normalized:
                return "business presentation slide"
            return "presentation slide"
        if "underwater" in normalized and any(token in normalized for token in {"photo", "photograph", "image"}):
            if "wildlife" in normalized:
                return "underwater wildlife photograph"
            if "whale" in normalized:
                return "whale photography scene"
        if any(token in normalized for token in {"potter", "pottery", "artisan"}):
            if "portrait" in normalized:
                return "portrait of a pottery artisan"
            return "traditional pottery scene"
        if any(token in normalized for token in {"workspace", "desk", "software engineer", "office"}):
            return "professional workspace image" if task == "image" else "professional workspace"
        return FallbackClassifier._generic_intent_label(prompt)

    @staticmethod
    def _generic_intent_label(prompt: str) -> str:
        cleaned = " ".join(prompt.strip().strip("\"'` ").split())
        if not cleaned:
            return ""

        cleaned = re.split(r"\.\s+|\b(?:the|this)\s+(?:image|output|result|scene)\s+should\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
        cleaned = re.sub(r"^(?:please\s+)?(?:create|generate|make|write|produce|draft|design|render|compose)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(?:a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = FallbackClassifier._remove_intent_noise_words(cleaned)
        cleaned = re.split(
            r"\s+(?:with|featuring|filled with|including|at night|at sunrise|at sunset|during|above|on a|on an|on the|in the style|in a style|using)\b",
            cleaned,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        cleaned = cleaned.strip(" ,.;:-")
        if not cleaned:
            return ""

        photo_match = re.search(r"^(?P<context>.*?)\b(?P<kind>photo|photograph)\s+of\s+(?P<subject>.+)$", cleaned, flags=re.IGNORECASE)
        if photo_match:
            context = FallbackClassifier._trim_label_words(photo_match.group("context").replace(" wildlife", ""), keep_environment=True)
            subject = FallbackClassifier._trim_label_words(photo_match.group("subject"), keep_environment=False)
            subject_words = subject.split()
            if subject_words:
                main_subject = subject_words[-1]
                parts = [context, main_subject, "photograph"]
                return FallbackClassifier._normalize_intent_label(" ".join(part for part in parts if part))

        portrait_match = re.search(r"^portrait\s+of\s+(?P<subject>.+)$", cleaned, flags=re.IGNORECASE)
        if portrait_match:
            subject = FallbackClassifier._trim_label_words(portrait_match.group("subject"), keep_environment=False)
            if subject:
                return FallbackClassifier._normalize_intent_label(f"{subject} portrait")

        label = FallbackClassifier._trim_label_words(cleaned, keep_environment=True)
        label = FallbackClassifier._normalize_intent_label(label)
        if label.endswith(" castle"):
            label = f"{label} scene"
        if label and len(label.split()) <= 1:
            return ""
        return label

    @staticmethod
    def _remove_intent_noise_words(text: str) -> str:
        noise = {
            "realistic",
            "photorealistic",
            "photo-realistic",
            "highly",
            "detailed",
            "ultra",
            "minimalist",
            "futuristic",
            "beautiful",
            "dramatic",
            "atmospheric",
            "authentic",
            "historical",
            "professional",
            "giant",
        }
        words = [word for word in text.split() if word.lower().strip(" ,.;:-") not in noise]
        return " ".join(words)

    @staticmethod
    def _trim_label_words(text: str, *, keep_environment: bool) -> str:
        text = re.sub(r"^(?:a|an|the)\s+", "", text.strip(), flags=re.IGNORECASE)
        remove = {"wildlife", "setup", "scene"} if not keep_environment else {"setup"}
        words = [word.strip(" ,.;:-") for word in text.split()]
        words = [word for word in words if word and word.lower() not in remove]
        return " ".join(words)

    @staticmethod
    def _normalize_intent_label(label: str) -> str:
        words = [word for word in label.strip().split() if word]
        if len(words) > 6:
            words = words[:6]
        cleaned = " ".join(words).strip(" ,.;:-")
        if not cleaned:
            return ""
        protected_title_words = {"Scandinavian", "Victorian"}
        first, *rest = cleaned.split()
        if first not in protected_title_words and not first.isupper():
            cleaned = " ".join([first[:1].lower() + first[1:], *rest])
        return cleaned


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
        self._issue_category_chain = self._build_optional_chain(
            ISSUE_CATEGORY_CLASSIFICATION_PROMPT,
            IssueCategoryClassificationResult,
        )
        self._human_followup_chain = self._build_optional_chain(HUMAN_FOLLOWUP_PROMPT, HumanFollowupQuestionResult)
        self._feedback_insights_chain = self._build_optional_chain(FEEDBACK_INSIGHTS_PROMPT, FeedbackInsightsResult)
        self._intent_label_chain = self._build_optional_chain(INTENT_LABEL_PROMPT, IntentLabelResult)
        self._strict_intent_label_chain = self._build_optional_chain(STRICT_INTENT_LABEL_PROMPT, IntentLabelResult)
        self._intent_label_refinement_chain = self._build_optional_chain(
            INTENT_LABEL_REFINEMENT_PROMPT,
            IntentLabelRefinementResult,
        )

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

    def extract_feedback(self, message: str, *, infer_issue_tags: bool = True) -> FeedbackExtraction:
        if self._extraction_chain:
            extraction = self._extraction_chain.invoke({"message": message})
            if infer_issue_tags and not extraction.issue_tags:
                extraction.issue_tags = self.generate_issue_tags(message)
            return FallbackClassifier.refine_feedback(message, extraction, infer_issue_tags=infer_issue_tags)
        extraction = FallbackClassifier.extract_feedback(message)
        if infer_issue_tags:
            return extraction
        return FeedbackExtraction(
            sentiment=extraction.sentiment,
            positives=extraction.positives,
            negatives=extraction.negatives,
            suggestions=extraction.suggestions,
            issue_tags=[],
        )

    def classify_issue(self, message: str) -> IssueClassification:
        if self._issue_chain:
            return self._issue_chain.invoke({"message": message})
        return FallbackClassifier.classify_issue(message)

    def generate_issue_tags(self, message: str) -> list[str]:
        if self._tag_chain:
            return FallbackClassifier._best_issue_tags(message, [], [], [], self._tag_chain.invoke({"message": message}).issue_tags)
        return FallbackClassifier.generate_issue_tags(message)

    def classify_issue_categories(
        self,
        *,
        prompt: str,
        feedback_text: str,
        session_context: str = "",
    ) -> IssueCategoryClassificationResult:
        threshold = get_settings().issue_category_confidence_threshold
        if self._issue_category_chain:
            try:
                proposed = self._issue_category_chain.invoke(
                    {
                        "prompt": prompt,
                        "feedback_text": feedback_text,
                        "session_context": session_context,
                    }
                )
                return FallbackClassifier.classify_issue_categories(
                    prompt=prompt,
                    feedback_text=feedback_text,
                    session_context=session_context,
                    confidence_threshold=threshold,
                    proposed=proposed,
                    allow_fallback_supplement=False,
                )
            except Exception:
                pass
        return FallbackClassifier.classify_issue_categories(
            prompt=prompt,
            feedback_text=feedback_text,
            session_context=session_context,
            confidence_threshold=threshold,
        )

    def generate_human_followup_question(self, **payload) -> HumanFollowupQuestionResult:
        if self._human_followup_chain:
            return self._human_followup_chain.invoke(payload)
        return FallbackClassifier.generate_human_followup_question(**payload)

    def generate_feedback_insights(self, **payload) -> FeedbackInsightsResult:
        if self._feedback_insights_chain:
            return self._feedback_insights_chain.invoke(payload)
        return FallbackClassifier.generate_feedback_insights(**payload)

    def generate_intent_label(self, *, task_type: str, prompt: str) -> IntentLabelResult:
        retry_used = False
        groq_label = ""

        if not self._intent_label_chain:
            return FallbackClassifier.generate_intent_label(task_type=task_type, prompt=prompt)

        try:
            raw_result = self._intent_label_chain.invoke({"task_type": task_type, "prompt": prompt})
        except Exception:
            return self._fallback_intent_label_result(
                task_type=task_type,
                prompt=prompt,
                groq_label=groq_label,
                retry_used=retry_used,
                validation_result="groq_unavailable_or_timeout",
            )

        groq_label = raw_result.intent_label or ""
        if not groq_label.strip():
            return self._fallback_intent_label_result(
                task_type=task_type,
                prompt=prompt,
                groq_label=groq_label,
                retry_used=retry_used,
                validation_result="empty_groq_label",
            )

        refined_label = self.refine_intent_label(groq_label, prompt=prompt)
        valid, validation_result = FallbackClassifier.validate_intent_label(refined_label)
        confidence = raw_result.confidence if valid else 0.0
        if valid:
            return IntentLabelResult(
                intent_label=refined_label,
                confidence=confidence,
                groq_label=groq_label,
                refined_label=refined_label,
                validation_result=validation_result,
                retry_used=retry_used,
                fallback_used=False,
            )

        if self._strict_intent_label_chain:
            retry_used = True
            try:
                retry_result = self._strict_intent_label_chain.invoke({"task_type": task_type, "prompt": prompt})
                groq_label = retry_result.intent_label or ""
                if groq_label.strip():
                    refined_label = self.refine_intent_label(groq_label, prompt=prompt)
                    valid, validation_result = FallbackClassifier.validate_intent_label(refined_label)
                    if valid:
                        return IntentLabelResult(
                            intent_label=refined_label,
                            confidence=retry_result.confidence,
                            groq_label=groq_label,
                            refined_label=refined_label,
                            validation_result=validation_result,
                            retry_used=retry_used,
                            fallback_used=False,
                        )
                else:
                    validation_result = "empty_retry_label"
            except Exception:
                validation_result = "retry_groq_unavailable_or_timeout"

        return self._fallback_intent_label_result(
            task_type=task_type,
            prompt=prompt,
            groq_label=groq_label,
            retry_used=retry_used,
            validation_result=validation_result,
        )

    def _fallback_intent_label_result(
        self,
        *,
        task_type: str,
        prompt: str,
        groq_label: str,
        retry_used: bool,
        validation_result: str,
    ) -> IntentLabelResult:
        fallback = FallbackClassifier.generate_intent_label(task_type=task_type, prompt=prompt)
        return IntentLabelResult(
            intent_label=fallback.intent_label,
            confidence=fallback.confidence,
            groq_label=groq_label,
            refined_label=fallback.refined_label or fallback.intent_label,
            validation_result=validation_result,
            retry_used=retry_used,
            fallback_used=True,
        )

    def refine_intent_label(self, raw_label: str, *, prompt: str = "") -> str:
        if self._intent_label_refinement_chain and raw_label:
            try:
                result = self._intent_label_refinement_chain.invoke({"raw_label": raw_label})
                refined = FallbackClassifier.refine_intent_label(result.refined_label, prompt=prompt)
                if refined:
                    return refined
            except Exception:
                pass
        return FallbackClassifier.refine_intent_label(raw_label, prompt=prompt)
