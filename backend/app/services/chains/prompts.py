TURN_ANALYSIS_PROMPT = """
You analyze a user message inside a conversational AI feedback interview.
Return structured JSON only.

Required output format:
{{
  "intent": "feedback | rating | off_topic",
  "issue_type": "technical | usability | quality | none",
  "sentiment": "positive | negative | mixed",
  "is_feedback_present": true/false
}}

Rules:
- Mark intent as "feedback" when the user gives any opinion, complaint, suggestion, praise, or problem report.
- Mark intent as "rating" only when the message is mainly a score such as "3", "I'd rate it 4", or "2/5".
- Mark intent as "off_topic" when it does not contribute feedback or rating.
- If the message contains both feedback and rating, prefer intent="feedback".
- Treat emotional mismatch, missing vibe, warmth, mood, tone, realism, or atmosphere as feedback.
- Choose issue_type="technical" for crashes, freezing, delays, failed actions, or broken behavior.
- Choose issue_type="usability" for confusing flows, difficult UI, unclear steps, navigation problems, or hard-to-use interactions.
- Choose issue_type="quality" for unrealistic, inaccurate, low-quality, incomplete, irrelevant, incorrect output, or missed emotional tone.
- Choose issue_type="none" if no clear issue type is present.

User message: {message}
"""

INTENT_PROMPT = """
You classify whether a user wants to give feedback.
Return structured JSON.

User message: {message}

Possible intents:
- YES
- NO
- OFF_TOPIC
"""

RATING_PROMPT = """
Extract a 1-5 rating from the user's message if present.
Support formats like:
- "3"
- "I would rate it 3"
- "2/5"

If the answer is vague, such as "it's okay", "fine", or "not bad", set is_vague=true.
When vague, do not guess a score. Ask for clarification in one short sentence that reminds the user of the 1-5 scale.
If the message is clearly a stop/closure message such as "no", "stop", or "that's it", do not treat it as a rating.

User message: {message}
"""

SENTIMENT_PROMPT = """
Analyze the sentiment of the user's feedback semantically.
Infer tone even when it is implicit.
Return positive, negative, or mixed as structured JSON.

Examples:
- "I like the color theme" -> positive
- "navigation is hard" -> negative
- "it's okay" -> mixed

Feedback text: {message}
"""

FEEDBACK_EXTRACTION_PROMPT = """
You extract structured product feedback from a natural user message.
Return strict JSON only in this format:
{{
  "sentiment": "positive | negative | mixed",
  "positives": [],
  "negatives": [],
  "suggestions": [],
  "issue_tags": []
}}

Extraction rules:
- Capture multiple insights from one message.
- Ignore noise or irrelevant filler.
- Use short concrete phrases grounded in the user's wording.
- Treat "wanted", "missed the vibe", "not warm/cozy", "felt flat", and similar emotional/tone gaps as negatives or suggestions even when phrased softly.
- issue_tags must be snake_case, specific, reusable, 2 to 4 words when possible, and non-duplicated.
- issue_tags should describe root issues or requested changes, not emotions.
- Tags must align with negatives and suggestions.

Examples:
- "navigation is hard" -> negatives=["navigation is hard"], issue_tags=["navigation_difficulty"]
- "it took 7 minutes" -> negatives=["it took 7 minutes"], issue_tags=["slow_image_generation"]
- "puppies were not realistic" -> negatives=["puppies were not realistic"], issue_tags=["lack_of_realism"]
- "generate 2 images" -> suggestions=["generate 2 images"], issue_tags=["multiple_output_request"]

Feedback text: {message}
"""

ISSUE_CLASSIFICATION_PROMPT = """
Classify the feedback into one issue category:
- technical
- quality
- usability
- none

Feedback text: {message}
"""

ISSUE_TAG_PROMPT = """
Generate dynamic issue tags from the user's feedback.
Return strict JSON only:
{{
  "issue_tags": []
}}

Rules:
- Use snake_case
- Make tags specific but reusable
- Avoid duplicates
- Focus on root issue or requested behavior
- 2 to 4 words when possible

Feedback text: {message}
"""

HUMAN_FOLLOWUP_PROMPT = """
You write exactly one short, natural follow-up question for a feedback chat.
Return strict JSON only:
{{
  "questions": ["..."]
}}

Context:
- task_type: {task_type}
- prompt: {prompt}
- ai_output: {ai_output}
- user_feedback: {user_feedback}
- existing_negatives: {existing_negatives}
- existing_suggestions: {existing_suggestions}
- previous_followups: {previous_followups}
- detected_issue_type: {detected_issue_type}
- grounding_context: {grounding_context}

Rules:
- Return exactly 1 question.
- Keep it under 18 words.
- Sound human and conversational.
- Do not repeat or closely paraphrase previous_followups.
- Adapt to task_type and any mismatch between prompt and output.
- If grounding_context names a likely mismatch, ask about that concrete mismatch first.
- If grounding_context includes an active_thread or unresolved_mismatches, continue that same thread.
- If grounding_context includes latest_user_correction, treat it as higher priority than earlier context.
- If grounding_context includes current_domain, keep the response inside that domain.
- Do not reuse anything listed in do_not_reuse_invalidated_threads.
- Prefer confirming a specific mismatch over generic questions like "what felt off?"
- Do not ask usability/confusion questions unless the user actually mentioned confusion, navigation, or steps.
- Do not ask another question just to reconfirm something already clear from context.
- Ask for one useful missing detail, not multiple questions.
- Avoid greetings, preambles, and lists.
"""

FEEDBACK_INSIGHTS_PROMPT = """
You are an AI product analyst.

Analyze all feedback and return strict JSON only:
{{
  "summary": "...",
  "top_problems": [],
  "improvement_suggestions": []
}}

Focus on:
- recurring issues
- system weaknesses
- actionable improvements

Collected negatives: {negatives}
Collected suggestions: {suggestions}
Collected issue_tags: {issue_tags}
"""
