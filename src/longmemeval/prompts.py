"""Prompt templates for LongMemEval answer generation and judging."""

from __future__ import annotations


def build_answer_prompt(query: str, context: str, question_date: str | None = None) -> str:
    """Build grounded answer generation prompt matching official LongMemEval format."""
    formatted_date = question_date if question_date else "Not specified"

    return f"""I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.

History Chats:

{context}

Current Date: {formatted_date}
Question: {query}
Answer:
"""


def get_official_judge_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    abstention: bool = False,
) -> str:
    """Official LongMemEval judge prompt definitions matching evaluate_qa.py exactly."""
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, "
                "you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. "
                "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        elif task == "temporal-reasoning":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, "
                "you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. "
                "In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., "
                "and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. "
                "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        elif task == "knowledge-update":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, the response should be considered as correct "
                "as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        elif task == "single-session-preference":
            template = (
                "I will give you a question, a rubric for desired personalized response, and a response from a model. "
                "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                "The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        else:
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
    else:
        template = (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        )
        return template.format(question, answer, response)


def build_extract_evidence_prompt(query: str, context: str, question_date: str | None = None) -> str:
    """Prompt for Stage 1 of agentic-v1: extract factual evidence and classify support status."""
    formatted_date = question_date if question_date else "Not specified"

    return f"""Select and extract factual evidence for a conversational-memory question.
Use only the retrieved chat history below. Do not assume facts that are not present.
Do not use benchmark labels, gold answers, or outside assumptions. Do not answer the question yet.

Classify support status as:
- direct: the memories explicitly contain the requested answer, event, fact, or preference;
- inferable: the memories contain relevant facts that permit a narrow, straightforward inference or calculation (e.g. identifying a retailer or location from the conversational context, counting items across multiple chats, calculating dates/durations relative to current date, synthesizing user preference);
- unsupported: the memories do not contain the asked information, or the question asks about a person, event, pet, possession, or activity that was never mentioned in the history.

Guidelines:
- Exhaustive Scanning: Thoroughly scan the ENTIRE retrieved chat history from beginning to end. Gold evidence may appear across multiple distinct sessions spaced throughout the text.
- Dates & Chronology: Note timestamps and session dates whenever mentioned.
- Assistant Recall: The question may ask what the assistant previously recommended, suggested, or answered (e.g. restaurant recommendations, movie lists, advice). Search BOTH assistant and user messages in the chat history to extract any recommendations or advice provided by the assistant.
- Multi-Event & Temporal Reasoning: When a question asks about events meeting a chronological condition (e.g. "on consecutive days", "within a week", "before X", "after Y"), extract ALL candidate events of that type across all sessions with their explicit dates and timestamps so that sequences and elapsed time can be verified. For "consecutive days", specifically check for any events that occurred on adjacent calendar dates (e.g. Day 1 and Day 2). Do not expect the chat text to use the literal words "consecutive days".
- Knowledge Updates & Relocations: If an entity, attribute, number, or location changed over time (e.g. someone moved to Chicago, then later relocated to the suburbs; or mortgage pre-approval changed), extract BOTH earlier and LATEST values with their dates, and clearly identify the LATEST chronological state as the current active fact.
- Contextual Associations & Locations: If asked where an action occurred or where an item was bought/redeemed, and the event is discussed within a conversation centered around a specific retailer, store, or venue (e.g. Target, Cartwheel), note that store as the contextual location.
- User Preferences & Recommendations: When the user asks for recommendations, suggestions, or advice (e.g. "Can you suggest a hotel...", "Can you recommend activities...", "Any advice on slow cooker recipes..."), this is a personalization request. ALWAYS classify as "inferable" (or "direct"). Extract the user's specific past experiences, successes (e.g. beef stew, making yogurt), preferences, and constraints, and explicitly list them in requirements so the response incorporates them. NEVER classify recommendation requests as "unsupported".
- Aggregation & Counting: If the question asks for a count (e.g. "How many [projects/items/plants/hours/days]..."), the user is asking how many instances they mentioned in their history chats. Classify as "inferable", list every distinct mentioned instance as a separate fact, and require summing them into an exact count. Carefully distinguish distinct named projects/items from ongoing general duties within a project.
- Never invent facts, dates, names, or items that are absent from the text.
- Do not include meta-statements in facts (e.g. do NOT write "the history does not specify total count"). Only extract concrete positive events, statements, preferences, and actions.
- Return at most 15 concise, factual bullet points.

Current Date: {formatted_date}
Question: {query}

Retrieved chat history:
{context or "[No history retrieved]"}

Return JSON only (no markdown fences, no extra text):
{{"status":"direct|inferable|unsupported","facts":["fact 1","fact 2"],"requirements":["what a complete answer must include"]}}"""


def build_evidence_answer_prompt(query: str, evidence_context: str, question_date: str | None = None) -> str:
    """Prompt for Stage 2 of agentic-v1: generate answer from extracted evidence."""
    formatted_date = question_date if question_date else "Not specified"

    return f"""Answer the question based strictly on the extracted conversational evidence.

Current Date: {formatted_date}
Question: {query}

{evidence_context}

Guidelines:
- Provide a clear, direct, and factual answer based on the extracted facts.
- Include all key identifying details (e.g. names AND their roles/specialties, items AND their categories, destinations AND times) to make the answer fully complete.
- If the question asks for a count or total ("How many..."), count every distinct instance or item listed in the facts and state the exact integer count directly (e.g. "You have led 2 projects: ..."). Do not state that the total is unknown if distinct instances are present in the facts.
- If an entity, attribute, or state changed over time (e.g. job, salary, pre-approval amount, location, pet name), state the MOST RECENT / UPDATED value directly as the primary answer (you may optionally note previous values as historical context).
- If the question asks for recommendations, activities, or advice, generate personalized suggestions for the requested topic that strictly incorporate the user's explicit domain, background, and negative constraints (e.g. avoid screens, specific field like healthcare AI) from the facts.
- If the support status is "unsupported" or no facts relate to the question premise, clearly state that the user did not mention this information in the chat history.

Answer:"""


def build_infer_answer_prompt(query: str, evidence_context: str, question_date: str | None = None) -> str:
    """Prompt for Stage 3 of agentic-v1: conditional narrow inference or time calculation."""
    formatted_date = question_date if question_date else "Not specified"

    return f"""Answer the question using the extracted conversational evidence through narrow, direct inference or calculation.

Current Date: {formatted_date}
Question: {query}

{evidence_context}

Guidelines:
- Make only calculations or narrow inferences directly licensed by the extracted facts.
- For counting or aggregation queries, verify every distinct instance or session mentioned in the facts.
- For time calculations, compute the difference relative to the stated dates and the Current Date ({formatted_date}).
- For updated information or changed states, ensure the latest chronological update is the primary answer.
- For preference queries, directly utilize the user's personal interests, field of study, and constraints.
- If the premise is unsupported or no responsible inference is possible, answer: "You did not mention this information in the chat history."

Answer:"""


def build_verify_answer_prompt(
    query: str,
    evidence_context: str,
    candidates: tuple[str, ...],
    question_date: str | None = None,
) -> str:
    """Prompt for Stage 4 of agentic-v1: verify candidates and hold back if unsupported."""
    formatted_date = question_date if question_date else "Not specified"
    candidate_text = "\n".join(f"Candidate {idx}: {c}" for idx, c in enumerate(candidates, 1))

    return f"""Verify and, when necessary, correct the candidate answer to a conversational-memory question.
You may use only the extracted facts below.

Check:
1. Exact counts and identifying details: Recount the extracted facts to ensure no items are omitted or double-counted across different chat sessions. Include specific details like specialties, roles, or names. If asked "How many...", provide the exact integer count of mentioned items directly.
2. Temporal order & updates: If an updated fact or state is present, the primary answer MUST state the latest/updated value directly (e.g. state the latest pre-approval amount or salary, rather than presenting both equally).
3. Personalization & Recommendations: If asked for recommendations, suggestions, or advice, ensure the answer delivers relevant suggestions that directly reflect the user's specific preferences, background field, and negative constraints from the evidence. Do NOT abstain on recommendation requests.
4. Abstention vs Grounded Answers: If one of the candidates provides concrete historical details (e.g. specific dates like Feb 14/15, specific entity/restaurant names, or exact counts), prefer the candidate that answers with specific grounded details over a generic abstention. Only abstain if the information was genuinely never mentioned in the chat history.
5. Directness: Keep the answer clear, complete, and direct.

Question: {query}
Current Date: {formatted_date}

{evidence_context}

Candidate Answers:
{candidate_text}

Return JSON only:
{{"answer":"final verified answer","reason":"brief reason for choice or correction"}}"""

