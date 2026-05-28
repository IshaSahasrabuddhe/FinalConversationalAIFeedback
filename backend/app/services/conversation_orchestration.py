from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.schemas.llm import FeedbackExtraction


@dataclass(frozen=True)
class SemanticTurn:
    observations: list[str]
    complaints: list[str]
    expected_outcomes: list[str]
    inferred_intent: str
    implied_fields: dict[str, Any]
    confidence: float
    evidence_terms: list[str]
    domain: str
    topic_tokens: list[str]


@dataclass(frozen=True)
class ResponseCandidate:
    kind: str
    content: str
    state: str
    base_score: int
    advancement_score: int = 0
    asks_question: bool = False
    target_field: str = ""
    question_key: str = ""
    blocks_next_action: bool = False


class SemanticMemoryExtractor:
    DOMAIN_KEYWORDS = {
        "food": {
            "food",
            "pancake",
            "pancakes",
            "syrup",
            "butter",
            "honey",
            "toast",
            "breakfast",
            "cafe",
            "chocolate",
            "sauce",
            "melt",
            "melting",
        },
        "sports": {
            "sport",
            "sports",
            "athlete",
            "player",
            "football",
            "soccer",
            "basketball",
            "tennis",
            "cricket",
            "baseball",
            "running",
            "action",
            "stadium",
            "jersey",
            "goal",
            "motion blur",
        },
        "coding": {"code", "coding", "bug", "api", "function", "typescript", "python", "backend", "frontend"},
        "medical": {"medical", "doctor", "patient", "diagnosis", "symptom", "treatment", "health"},
        "creative_feedback": {"image", "photo", "render", "scene", "style", "realistic", "lighting", "composition"},
    }
    FLOW_TERMS = {
        "flow",
        "flowing",
        "fluid",
        "pour",
        "pouring",
        "melt",
        "melting",
        "move",
        "moving",
        "motion",
        "animate",
        "animation",
    }
    STATIC_TERMS = {"frozen", "static", "stiff", "rigid", "stuck", "still", "sat there"}
    LIQUID_OR_FOOD_TERMS = {
        "syrup",
        "honey",
        "sauce",
        "liquid",
        "fluid",
        "pour",
        "pouring",
        "melt",
        "melting",
        "butter",
        "pancake",
        "pancakes",
    }
    EXPECTATION_MARKERS = {
        "should",
        "supposed to",
        "wanted",
        "expected",
        "instead of",
        "rather than",
        "needed to",
        "meant to",
    }

    def extract(self, message: str, extracted: FeedbackExtraction, issue_type: str) -> SemanticTurn:
        normalized = _normalize(message)
        topic_tokens = sorted(_meaningful_tokens(message) | set(extracted.issue_tags))
        domain = self.detect_domain(message, extracted)
        observations = self._dedupe([*extracted.negatives, *extracted.positives])
        complaints = self._dedupe([*extracted.negatives, *extracted.issue_tags])
        expected_outcomes = self._extract_expectations(message)
        evidence_terms = self._evidence_terms(normalized, extracted)

        inferred_intent = "provide_feedback" if observations or complaints or extracted.suggestions else "continue_conversation"
        implied_fields: dict[str, Any] = {}
        if observations or complaints:
            implied_fields["issue_description"] = observations[0] if observations else complaints[0]
            implied_fields["observed_behavior"] = observations[0] if observations else message.strip()
            implied_fields["issue_type"] = issue_type
        if expected_outcomes:
            implied_fields["expected_behavior"] = expected_outcomes[0]
        if extracted.suggestions:
            implied_fields["suggested_fix"] = extracted.suggestions[0]

        if self._has_fluid_motion_mismatch(normalized):
            implied_fields.setdefault("issue_description", "motion looked rigid instead of fluid")
            implied_fields.setdefault("observed_behavior", "the output looked frozen or static")
            implied_fields.setdefault("expected_behavior", "the motion should flow naturally")
            implied_fields["probable_cause"] = "rigid motion physics or weak temporal fluid continuity"
            if "motion_physics" not in evidence_terms:
                evidence_terms.append("motion_physics")

        confidence = self._confidence(
            message=message,
            extracted=extracted,
            implied_fields=implied_fields,
            evidence_terms=evidence_terms,
        )
        return SemanticTurn(
            observations=observations,
            complaints=complaints,
            expected_outcomes=expected_outcomes,
            inferred_intent=inferred_intent,
            implied_fields=implied_fields,
            confidence=confidence,
            evidence_terms=evidence_terms,
            domain=domain,
            topic_tokens=topic_tokens,
        )

    def detect_domain(self, message: str, extracted: FeedbackExtraction) -> str:
        normalized = _normalize(" ".join([message, " ".join(extracted.issue_tags)]))
        best_domain = "general"
        best_score = 0
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in normalized)
            if score > best_score:
                best_domain = domain
                best_score = score
        return best_domain if best_score > 0 else "general"

    def _extract_expectations(self, message: str) -> list[str]:
        normalized = _normalize(message)
        expectations: list[str] = []
        for marker in self.EXPECTATION_MARKERS:
            if marker not in normalized:
                continue
            tail = normalized.split(marker, 1)[1].strip(" .,!?:;")
            if tail:
                expectations.append(tail[:160])
        return self._dedupe(expectations)

    def _evidence_terms(self, normalized: str, extracted: FeedbackExtraction) -> list[str]:
        terms = list(extracted.issue_tags)
        if any(term in normalized for term in self.FLOW_TERMS):
            terms.append("fluid_motion")
        if any(term in normalized for term in self.STATIC_TERMS):
            terms.append("static_or_frozen_output")
        if "realistic" in normalized or "unrealistic" in normalized:
            terms.append("realism_gap")
        if any(token in normalized for token in {"saturated", "oversaturated", "over-saturated", "color grading", "colour grading"}):
            terms.append("over_saturated_color")
        if any(token in normalized for token in {"sunset", "colors", "colours", "color", "colour"}):
            terms.append("color_grading")
        if any(token in normalized for token in {"scale", "aerial", "perspective", "island", "water texture", "water"}):
            terms.append("environmental_believability")
        if any(token in normalized for token in {"reflection", "reflections", "metallic", "metal", "premium", "blurry", "soft"}):
            terms.append("product_material_finish")
        if any(token in normalized for token in {"athlete", "player", "jersey", "stadium", "pose", "motion blur"}):
            terms.append("sports_action_realism")
        return self._dedupe(terms)

    def _has_fluid_motion_mismatch(self, normalized: str) -> bool:
        return (
            any(term in normalized for term in self.LIQUID_OR_FOOD_TERMS)
            and any(term in normalized for term in self.FLOW_TERMS)
            and any(term in normalized for term in self.STATIC_TERMS)
        )

    def _confidence(
        self,
        *,
        message: str,
        extracted: FeedbackExtraction,
        implied_fields: dict[str, Any],
        evidence_terms: list[str],
    ) -> float:
        score = 0.35
        if extracted.negatives or extracted.suggestions:
            score += 0.25
        if implied_fields.get("observed_behavior"):
            score += 0.15
        if implied_fields.get("expected_behavior"):
            score += 0.12
        if len(evidence_terms) >= 2:
            score += 0.08
        if len(message.split()) >= 6:
            score += 0.05
        return min(score, 0.95)

    def _dedupe(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            cleaned = " ".join(str(value).strip().split())
            if cleaned and cleaned not in deduped:
                deduped.append(cleaned)
        return deduped


class QuestionGate:
    RECENT_TURN_WINDOW = 5

    def should_ask(
        self,
        *,
        context: dict[str, Any],
        question: str,
        semantic_turn: SemanticTurn,
        target_field: str,
        question_key: str,
        current_turn: int,
        blocks_next_action: bool,
    ) -> bool:
        if not question.strip() or "?" not in question:
            return True
        if not blocks_next_action:
            return False
        if self._survey_loop_active(context):
            return False
        if self._field_known(context, target_field):
            return False
        if self._semantically_answered(semantic_turn, target_field):
            return False
        if self._inferable(semantic_turn, target_field):
            return False
        if self._recently_asked(context, question, question_key, current_turn):
            return False
        return True

    def record_question(
        self,
        *,
        context: dict[str, Any],
        question: str,
        question_key: str,
        target_field: str,
        current_turn: int,
    ) -> dict[str, Any]:
        intelligence = _intelligence(context)
        questions = list(intelligence.get("question_intents", []))
        questions.append(
            {
                "key": question_key or _question_key(question),
                "text": question,
                "target_field": target_field,
                "turn": current_turn,
                "answered": False,
                "tokens": sorted(_meaningful_tokens(question)),
            }
        )
        intelligence["question_intents"] = questions[-20:]
        intelligence["assistant_actions"] = [*list(intelligence.get("assistant_actions", [])), {"type": "asked_question", "turn": current_turn}][-12:]
        intelligence["response_history"] = _append_response_history(
            intelligence.get("response_history", []),
            question,
            "asked_question",
            current_turn,
        )
        context["conversation_intelligence"] = intelligence
        return context

    def record_action(
        self,
        context: dict[str, Any],
        action_type: str,
        current_turn: int,
        content: str = "",
    ) -> dict[str, Any]:
        intelligence = _intelligence(context)
        intelligence["assistant_actions"] = [*list(intelligence.get("assistant_actions", [])), {"type": action_type, "turn": current_turn}][-12:]
        if content:
            intelligence["response_history"] = _append_response_history(
                intelligence.get("response_history", []),
                content,
                action_type,
                current_turn,
            )
        context["conversation_intelligence"] = intelligence
        return context

    def mark_answered(self, context: dict[str, Any], semantic_turn: SemanticTurn) -> dict[str, Any]:
        intelligence = _intelligence(context)
        questions = list(intelligence.get("question_intents", []))
        answered_fields = set(semantic_turn.implied_fields)
        turn_tokens = set(semantic_turn.evidence_terms)
        for question in questions:
            target = str(question.get("target_field") or "")
            token_overlap = turn_tokens & set(question.get("tokens", []))
            if target in answered_fields or token_overlap:
                question["answered"] = True
        intelligence["question_intents"] = questions
        context["conversation_intelligence"] = intelligence
        return context

    def _field_known(self, context: dict[str, Any], target_field: str) -> bool:
        if not target_field:
            return False
        memory = relevant_semantic_memory(context)
        facts = memory.get("facts", {})
        inferred = memory.get("inferred", {})
        return bool(facts.get(target_field) or inferred.get(target_field))

    def _semantically_answered(self, semantic_turn: SemanticTurn, target_field: str) -> bool:
        if not target_field:
            return False
        if target_field in semantic_turn.implied_fields:
            return True
        if target_field == "issue_description" and (semantic_turn.complaints or semantic_turn.observations):
            return True
        if target_field == "observed_behavior" and semantic_turn.observations:
            return True
        if target_field == "expected_behavior" and semantic_turn.expected_outcomes:
            return True
        return False

    def _inferable(self, semantic_turn: SemanticTurn, target_field: str) -> bool:
        return semantic_turn.confidence >= 0.78 and target_field in {
            "issue_description",
            "observed_behavior",
            "expected_behavior",
            "issue_type",
        }

    def _recently_asked(self, context: dict[str, Any], question: str, question_key: str, current_turn: int) -> bool:
        intelligence = _intelligence(context)
        question_tokens = _meaningful_tokens(question)
        for previous in intelligence.get("question_intents", []):
            if current_turn - int(previous.get("turn", 0)) > self.RECENT_TURN_WINDOW:
                continue
            if question_key and question_key == previous.get("key"):
                return True
            previous_tokens = set(previous.get("tokens", []))
            if previous.get("answered") and _jaccard(question_tokens, previous_tokens) >= 0.35:
                return True
            if _jaccard(question_tokens, previous_tokens) >= 0.55:
                return True
        return False

    def _survey_loop_active(self, context: dict[str, Any]) -> bool:
        actions = list(_intelligence(context).get("assistant_actions", []))[-4:]
        return sum(1 for action in actions if action.get("type") == "asked_question") >= 2


class ResponseArbiter:
    def choose(
        self,
        *,
        candidates: list[ResponseCandidate],
        context: dict[str, Any],
        semantic_turn: SemanticTurn,
        gate: QuestionGate,
        current_turn: int,
    ) -> ResponseCandidate:
        scored: list[tuple[int, ResponseCandidate]] = []
        for candidate in candidates:
            if candidate.asks_question and not gate.should_ask(
                context=context,
                question=candidate.content,
                semantic_turn=semantic_turn,
                target_field=candidate.target_field,
                question_key=candidate.question_key,
                current_turn=current_turn,
                blocks_next_action=candidate.blocks_next_action,
            ):
                score = -1000
            else:
                score = self._score(candidate, context, semantic_turn)
            scored.append((score, candidate))
        return max(scored, key=lambda item: item[0])[1]

    def _score(self, candidate: ResponseCandidate, context: dict[str, Any], semantic_turn: SemanticTurn) -> int:
        score = candidate.base_score
        memory = relevant_semantic_memory(context, semantic_turn)
        if semantic_turn.domain != "general" and memory.get("active_domain") not in {"", semantic_turn.domain}:
            score -= 40
        score += candidate.advancement_score
        score += self._reasoning_advancement_score(candidate, context, semantic_turn)
        score -= self._repetition_penalty(candidate, context)
        score -= self._stagnant_acknowledgement_penalty(candidate, semantic_turn)
        score -= self._meta_analysis_penalty(candidate)
        if candidate.asks_question:
            score -= 25
            if semantic_turn.confidence >= 0.72:
                score -= 30
        else:
            if semantic_turn.implied_fields:
                score += 12
            if candidate.kind == "issue_categorization":
                score += 26
            if candidate.kind in {"acknowledge_infer", "diagnose"}:
                score += 6
            if candidate.kind == "summarize":
                score -= 8
        if self._recent_question_count(context) >= 1 and candidate.kind in {"issue_categorization", "acknowledge_infer"}:
            score += 12
        return score

    def _recent_question_count(self, context: dict[str, Any]) -> int:
        actions = list(_intelligence(context).get("assistant_actions", []))[-3:]
        return sum(1 for action in actions if action.get("type") == "asked_question")

    def _reasoning_advancement_score(
        self,
        candidate: ResponseCandidate,
        context: dict[str, Any],
        semantic_turn: SemanticTurn,
    ) -> int:
        candidate_tokens = _meaningful_tokens(candidate.content)
        current_tokens = set(semantic_turn.topic_tokens) | set(semantic_turn.evidence_terms)
        if not current_tokens:
            return 0

        score = 0
        current_overlap = _jaccard(candidate_tokens, current_tokens)
        if current_overlap >= 0.12:
            score += 22

        memory = relevant_semantic_memory(context, semantic_turn)
        recent_turns = list(memory.get("recent_turns", []))
        previous_tokens: set[str] = set()
        for turn in recent_turns[:-1]:
            previous_tokens |= set(turn.get("topic_tokens", []))
            previous_tokens |= set(turn.get("evidence_terms", []))

        new_tokens = current_tokens - previous_tokens
        if new_tokens and candidate_tokens & new_tokens:
            score += 28
        if previous_tokens and current_tokens and candidate_tokens & previous_tokens and candidate_tokens & current_tokens:
            score += 14
        if candidate.kind in {"cumulative_reasoning", "diagnose"}:
            score += 12
        if candidate.kind == "issue_categorization":
            score += 18
        return score

    def _repetition_penalty(self, candidate: ResponseCandidate, context: dict[str, Any]) -> int:
        history = list(_intelligence(context).get("response_history", []))[-4:]
        candidate_tokens = _meaningful_tokens(candidate.content)
        penalty = 0
        for item in history:
            similarity = _jaccard(candidate_tokens, set(item.get("tokens", [])))
            if similarity >= 0.72:
                penalty += 55
            elif similarity >= 0.5:
                penalty += 28
        return penalty

    def _stagnant_acknowledgement_penalty(self, candidate: ResponseCandidate, semantic_turn: SemanticTurn) -> int:
        normalized = candidate.content.strip().lower()
        starts_ack = normalized.startswith(("understood.", "got it.", "got it -", "i think i understand"))
        if not starts_ack:
            return 0
        candidate_tokens = _meaningful_tokens(candidate.content)
        current_tokens = set(semantic_turn.topic_tokens) | set(semantic_turn.evidence_terms)
        if current_tokens and candidate_tokens & current_tokens:
            return 0
        return 35

    def _meta_analysis_penalty(self, candidate: ResponseCandidate) -> int:
        normalized = candidate.content.lower()
        meta_phrases = {
            "the issue is clear",
            "captured the main issue",
            "that adds a new layer",
            "that adds a useful detail",
            "the latest detail changes",
            "the diagnosis",
            "folding that into",
            "part of the issue",
            "realism issue is clear",
            "i have captured",
            "i have noted",
        }
        return sum(18 for phrase in meta_phrases if phrase in normalized)


def update_semantic_memory(context: dict[str, Any], semantic_turn: SemanticTurn) -> dict[str, Any]:
    intelligence = _intelligence(context)
    memory = dict(intelligence.get("semantic_memory", {}))
    facts = dict(memory.get("facts", {}))
    inferred = dict(memory.get("inferred", {}))
    current_turn = int(intelligence.get("turn_count", 0)) + 1
    previous_domain = str(memory.get("active_domain") or "general")
    active_domain = semantic_turn.domain or "general"
    previous_topic_tokens = set(memory.get("active_topic_tokens", []))
    current_topic_tokens = set(semantic_turn.topic_tokens)

    topic_similarity = _jaccard(previous_topic_tokens, current_topic_tokens)
    domain_changed = (
        previous_domain not in {"", "general", active_domain}
        and active_domain != "general"
        and topic_similarity < 0.25
    )
    if domain_changed:
        facts = _invalidate_mismatched_entries(facts, active_domain, current_turn)
        inferred = _invalidate_mismatched_entries(inferred, active_domain, current_turn)
        memory["invalidated_domains"] = _merge_limited(
            memory.get("invalidated_domains", []),
            [previous_domain],
            limit=8,
        )

    for key, value in semantic_turn.implied_fields.items():
        if not value:
            continue
        target = inferred if key == "probable_cause" else facts
        existing = target.get(key)
        if not existing or _entry_is_stale_or_mismatched(existing, semantic_turn, current_turn):
            target[key] = {
                "value": value,
                "confidence": semantic_turn.confidence,
                "domain": active_domain,
                "topic_tokens": semantic_turn.topic_tokens,
                "turn": current_turn,
            }

    memory["facts"] = facts
    memory["inferred"] = inferred
    memory["evidence_terms"] = _merge_limited(memory.get("evidence_terms", []), semantic_turn.evidence_terms, limit=30)
    memory["recent_turns"] = _merge_recent_turns(
        memory.get("recent_turns", []),
        {
            "turn": current_turn,
            "domain": active_domain,
            "topic_tokens": semantic_turn.topic_tokens,
            "evidence_terms": semantic_turn.evidence_terms,
            "implied_fields": list(semantic_turn.implied_fields),
        },
    )
    memory["active_domain"] = active_domain if active_domain != "general" else previous_domain
    memory["active_topic_tokens"] = sorted((previous_topic_tokens & current_topic_tokens) | current_topic_tokens)[-30:]
    intelligence["semantic_memory"] = memory
    intelligence["turn_count"] = current_turn
    context["conversation_intelligence"] = intelligence
    return context


def relevant_semantic_memory(context: dict[str, Any], semantic_turn: SemanticTurn | None = None) -> dict[str, Any]:
    memory = dict(_intelligence(context).get("semantic_memory", {}))
    current_turn = int(_intelligence(context).get("turn_count", 0))
    active_domain = semantic_turn.domain if semantic_turn and semantic_turn.domain != "general" else str(memory.get("active_domain") or "general")
    topic_tokens = set(semantic_turn.topic_tokens if semantic_turn else memory.get("active_topic_tokens", []))

    def is_relevant(entry: dict[str, Any]) -> bool:
        turn = int(entry.get("turn", 0))
        if current_turn - turn > 4:
            return False
        entry_domain = str(entry.get("domain") or "general")
        if active_domain != "general" and entry_domain not in {"general", active_domain}:
            return False
        entry_tokens = set(entry.get("topic_tokens", []))
        return not topic_tokens or not entry_tokens or _jaccard(topic_tokens, entry_tokens) >= 0.12

    facts = {
        key: value
        for key, value in dict(memory.get("facts", {})).items()
        if isinstance(value, dict) and is_relevant(value)
    }
    inferred = {
        key: value
        for key, value in dict(memory.get("inferred", {})).items()
        if isinstance(value, dict) and is_relevant(value)
    }
    return {
        **memory,
        "facts": facts,
        "inferred": inferred,
        "active_domain": active_domain,
    }


def _intelligence(context: dict[str, Any]) -> dict[str, Any]:
    return dict(
        context.get(
            "conversation_intelligence",
            {
                "semantic_memory": {"facts": {}, "inferred": {}, "evidence_terms": []},
                "question_intents": [],
                "assistant_actions": [],
                "response_history": [],
                "turn_count": 0,
            },
        )
    )


def _entry_is_stale_or_mismatched(entry: dict[str, Any], semantic_turn: SemanticTurn, current_turn: int) -> bool:
    entry_domain = str(entry.get("domain") or "general")
    if semantic_turn.domain != "general" and entry_domain not in {"general", semantic_turn.domain}:
        return True
    if current_turn - int(entry.get("turn", 0)) > 4:
        return True
    entry_tokens = set(entry.get("topic_tokens", []))
    return bool(entry_tokens and _jaccard(entry_tokens, set(semantic_turn.topic_tokens)) < 0.12)


def _invalidate_mismatched_entries(entries: dict[str, Any], active_domain: str, current_turn: int) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in entries.items():
        if not isinstance(value, dict):
            continue
        if str(value.get("domain") or "general") in {"general", active_domain} and current_turn - int(value.get("turn", 0)) <= 4:
            cleaned[key] = value
    return cleaned


def _merge_recent_turns(old_turns: list[dict[str, Any]], new_turn: dict[str, Any]) -> list[dict[str, Any]]:
    return [*list(old_turns or []), new_turn][-6:]


def _append_response_history(
    history: list[dict[str, Any]],
    content: str,
    action_type: str,
    turn: int,
) -> list[dict[str, Any]]:
    return [
        *list(history or []),
        {
            "content": content,
            "type": action_type,
            "turn": turn,
            "tokens": sorted(_meaningful_tokens(content)),
        },
    ][-8:]


def _normalize(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _question_key(question: str) -> str:
    return "_".join(sorted(_meaningful_tokens(question))[:5])


def _meaningful_tokens(text: str) -> set[str]:
    stopwords = {
        "the",
        "and",
        "you",
        "your",
        "what",
        "which",
        "would",
        "could",
        "should",
        "that",
        "this",
        "there",
        "about",
        "with",
        "felt",
        "feel",
        "share",
        "detail",
        "little",
    }
    return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) > 2 and token not in stopwords}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _merge_limited(old_list: list[str], new_list: list[str], *, limit: int) -> list[str]:
    merged: list[str] = []
    for item in [*list(old_list or []), *list(new_list or [])]:
        if item and item not in merged:
            merged.append(item)
    return merged[-limit:]
