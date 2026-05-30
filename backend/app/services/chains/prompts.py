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
- Mark intent as "off_topic" when it does not contribute feedback or rating, including casual questions like "what is your name?" or "how are you?"
- Mark personal preferences or general statements as off_topic when they are not tied to the generated output, prompt, or application experience, e.g. "I like matcha".
- If the message contains both feedback and rating, prefer intent="feedback".
- Pure praise is feedback with positive sentiment and issue_type="none" unless it also includes a problem or requested improvement.
- Treat emotional mismatch, missing vibe, warmth, mood, tone, realism, or atmosphere as feedback.
- Treat unrelated questions, casual chat, jokes, restaurant/fashion/lifestyle questions, or assistant identity questions as off_topic.
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
- "2 because it ignored several objects"
- "4 - image quality was good but text was hard to read"

Extract a valid 1-5 rating even when the user adds explanation in the same message.
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
- Ignore personal preferences or vague conversation that are not about the generated output, prompt, or application experience, e.g. "I like matcha".
- Use short concrete phrases grounded in the user's wording.
- Split mixed statements carefully. Text before "but/however/though" may be positive while text after it may be negative.
- positives: only things the user explicitly liked, praised, or said worked.
- negatives: concrete failures, missing qualities, realism gaps, or weak areas.
- suggestions: only proposed improvements to the model, system, workflow, or generation process.
- Do not mark observations like "the workspace should have contained three monitors" as suggestions; capture them as negatives/prompt adherence issues.
- Never put missing/needed improvements in positives.
- Treat "wanted", "missed the vibe", "not warm/cozy", "felt flat", and similar emotional/tone gaps as negatives or suggestions even when phrased softly.
- issue_tags must be 3 to 5 snake_case tags, specific, reusable, and non-duplicated.
- issue_tags should describe root issues or requested changes, not emotions.
- Tags must align with negatives and suggestions.
- Tags must be grounded only in the current feedback text. Never infer realism, posing, anatomy, environment, or motion unless the user actually mentions that evidence.
- For unreadable text use text_rendering, typography_fidelity, readability.
- For incorrect labels use label_fidelity, information_accuracy, content_correctness.
- For bad charts use data_visualization_quality, chart_readability.
- For bad logos or branding use branding_fidelity, logo_accuracy.
- For missed requested objects use prompt_adherence, missing_objects, instruction_following.
- If confidence is low, use other rather than forcing realism, environment, anatomy, or posing categories.
- Prefer concrete tags such as environmental_density, environmental_realism, lighting_consistency, motion_realism, texture_realism, scale_consistency, atmospheric_depth, material_realism, reflection_realism, interaction_realism, composition_balance, perspective_consistency, anatomy_accuracy, cinematic_alignment, prompt_alignment, detail_sharpness.
- Avoid vague tags such as realism_issue, visual_quality, quality_problem, lack_of_realism, technical_product_realism, environmental_aerial_realism.

Examples:
- "navigation is hard" -> negatives=["navigation is hard"], issue_tags=["navigation_difficulty"]
- "it took 7 minutes" -> negatives=["it took 7 minutes"], issue_tags=["slow_image_generation"]
- "The visual style looked strong, but the city felt empty and lacked motion, density, and realism." -> positives=["visual style looked strong"], negatives=["city felt empty", "lacked motion, density, and realism"], suggestions=["more motion and environmental density"], issue_tags=["environmental_density", "environmental_realism", "motion_realism", "cinematic_alignment"]
- "puppies were not realistic" -> negatives=["puppies were not realistic"], issue_tags=["environmental_realism"]
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
- Return 3 to 5 tags when enough signal exists.
- Make tags specific but reusable.
- Avoid duplicates
- Focus on root issue or requested behavior
- Prefer concrete issue dimensions over broad labels.
- Use only categories supported by the current feedback text. Do not inject common image categories from other sessions.
- For unreadable text use text_rendering, typography_fidelity, readability.
- For incorrect labels use label_fidelity, information_accuracy, content_correctness.
- For bad charts use data_visualization_quality, chart_readability.
- For bad logos or branding use branding_fidelity, logo_accuracy.
- For missed requested objects use prompt_adherence, missing_objects, instruction_following.
- If confidence is low, use other rather than forcing previous or common categories.
- Good tags include environmental_density, environmental_realism, lighting_consistency, motion_realism, texture_realism, scale_consistency, atmospheric_depth, material_realism, reflection_realism, interaction_realism, composition_balance, perspective_consistency, anatomy_accuracy, cinematic_alignment, prompt_alignment, detail_sharpness.
- Avoid vague tags such as realism_issue, visual_quality, quality_problem, lack_of_realism, technical_product_realism, environmental_aerial_realism.

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
- Never ask generic collection prompts like "what felt off?", "tell me more", or "what looked unrealistic?"
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

INTENT_LABEL_PROMPT = """
Extract the user's primary creation intent from the prompt.
Return strict JSON only:
{{
  "intent_label": "...",
  "confidence": 0.0
}}

Context:
- task_type: {task_type}
- prompt: {prompt}

Rules:
- Maximum 2 to 6 words.
- Use natural vocabulary.
- Focus on the main subject or asset being generated.
- Extract the primary subject, scene, environment, or object even for unseen categories.
- Paraphrase when helpful.
- Do not copy full prompt fragments.
- Ignore secondary instructions, mood notes, camera directions, background details, and quality requirements.
- Ignore quality adjectives such as realistic, highly detailed, photorealistic, and minimalist unless essential to the subject.
- Do not include verbs like create, generate, make, write, produce, or design.
- Do not include trailing incomplete phrases.
- Set confidence from 0 to 1. Use >=0.75 for clear prompts.

Examples:
- "Create a realistic wedding photograph featuring a bride and groom at sunset." -> "wedding photograph"
- "Create a realistic medieval castle on a misty hill at sunrise." -> "medieval castle scene"
- "Create a realistic portrait of an elderly Japanese potter." -> "portrait of a pottery artisan"
- "Create a workspace setup for a software engineer." -> "professional workspace image"
- "Create a minimalist Scandinavian living room." -> "Scandinavian living room"
- "Create a highly detailed fantasy world map." -> "fantasy world map"
- "Create a realistic underwater wildlife photograph of a giant whale." -> "underwater whale photograph"
- "Create a cyberpunk street market at night." -> "cyberpunk street market"
- "Create a Victorian library filled with ancient books." -> "Victorian library"
- "Create a futuristic lunar research station." -> "lunar research station"
"""

INTENT_LABEL_REFINEMENT_PROMPT = """
Refine a raw intent label into a short natural English phrase.
Return strict JSON only:
{{
  "refined_label": "..."
}}

Input:
- raw_label: {raw_label}

Rules:
- Keep only the primary subject, scene, environment, or object.
- Remove trailing prompt fragments such as showing, containing, displayed, featuring, with, including.
- Prefer the subject over the environment when both are present.
- Keep 2 to 6 words when possible.
- Do not add instruction text.
- Do not return a sentence.

Examples:
- "fantasy world map showing kingdoms forests" -> "fantasy world map"
- "conference presentation slide displayed" -> "conference presentation slide"
- "underwater ocean photograph" -> "marine wildlife photograph"
- "castle image realistic sunrise atmospheric" -> "castle scene"
"""
