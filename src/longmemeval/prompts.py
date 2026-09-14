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


def build_extract_evidence_prompt(
    query: str,
    context: str,
    question_date: str | None = None,
    window: tuple[int, int] | None = None,
) -> str:
    """Prompt for Stage 1 of agentic-v1: extract factual evidence and classify support status.

    ``window=(i, n)`` marks this call as reading slice i of n of a longer history; the
    model is told other slices exist so it reports only what THIS slice contains.
    """
    formatted_date = question_date if question_date else "Not specified"
    window_note = ""
    if window and window[1] > 1:
        window_note = (
            f"\nNOTE: This is window {window[0]} of {window[1]} of a longer chat history. Other windows are read separately "
            f"and merged later. Extract only what THIS window contains. If this window has nothing relevant, return "
            f'{{"status":"unsupported","facts":[],"requirements":[]}} - do NOT write facts saying the history lacks information.\n'
        )

    return f"""Select and extract factual evidence for a conversational-memory question.
Use only the retrieved chat history below. Do not assume facts that are not present.
Do not use benchmark labels, gold answers, or outside assumptions. Do not answer the question yet.
{window_note}
Classify support status as:
- direct: the memories explicitly contain the requested answer, event, fact, or preference;
- inferable: the memories contain relevant facts that permit a narrow, straightforward inference or calculation (identifying a place or vendor from the surrounding conversation, counting instances across several chats, computing dates or durations relative to the current date, or combining stated preferences);
- unsupported: the memories do not contain the asked information, or the question asks about a person, event, pet, possession, or activity that was never mentioned in the history.

Guidelines:
- Exhaustive Scanning: Thoroughly scan the ENTIRE retrieved chat history from beginning to end. Relevant evidence may appear across multiple distinct sessions spaced throughout the text.
- Dates & Chronology: Note timestamps and session dates whenever mentioned.
- Assistant Recall: The question may ask what the assistant previously recommended, suggested, listed, or answered. Search BOTH assistant and user messages in the chat history to extract any recommendations or advice provided by the assistant.
- Multi-Event & Temporal Reasoning: When a question asks about events meeting a chronological condition (adjacent dates, within some period, before or after another event), extract ALL candidate events of that type across all sessions with their explicit dates and timestamps so that ordering and elapsed time can be verified. Do not expect the chat text to use the question's own wording for the condition.
- Knowledge Updates: If an entity, attribute, number, or location changed over time (for example a user switches phone plans twice, or a quoted price is later revised), extract BOTH earlier and LATEST values with their dates, and clearly identify the LATEST chronological state as the current active fact.
- Contextual Associations & Locations: If asked where an action occurred or where an item was bought or redeemed, and the event is discussed within a conversation centred on a specific store, app, or venue, note that place as the contextual location.
- User Preferences & Recommendations: When the user asks for recommendations, suggestions, or advice (for example "Can you recommend a board game for...", "Any tips for my first pottery class..."), this is a personalization request. If the history contains the user's relevant experiences, preferences, or constraints, classify as "inferable" (or "direct"), extract them, and list them in requirements so the response incorporates them. Classify as "unsupported" only if the history contains nothing about the user that bears on the request.
- Aggregation & Counting: If the question asks for a count ("How many ..."), the user is asking how many instances they mentioned in their history chats. Classify as "inferable", list every distinct mentioned instance as a separate fact, and require summing them into an exact count. Carefully distinguish distinct named instances from ongoing general activity within one of them.
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
- Explicit recall requests: ONLY when the question literally asks you to recall a stored name or value (phrasings like "remind me of the name of...", "what was the name of the...", "do you remember what ... was called"), answer in a single sentence stating that value in full (keep its complete title, link/URL, or identifier if one was given) and nothing else; do not add descriptions, features, menus, or other unrequested details. This rule does NOT apply to advice, recommendation, suggestion, yes/no, "how many", "how long", or "when" questions - answer those normally and completely.
- Multi-part questions: include all key identifying details (e.g. names AND their roles, items AND their categories, destinations AND times) to make the answer fully complete.
- If the question asks for a count or total ("How many..."), count every distinct instance or item listed in the facts and state the exact integer count directly (e.g. "You mentioned 3 woodworking projects: ..."). Do not state that the total is unknown if distinct instances are present in the facts.
- If an entity, attribute, or state changed over time (a job, a price, a plan, a location, a name), state the MOST RECENT / UPDATED value directly as the primary answer (you may optionally note previous values as historical context).
- If the question asks for recommendations, activities, or advice, generate personalized suggestions for the requested topic that strictly incorporate the user's explicit interests, background, and negative constraints from the facts (for example a stated dislike of crowds, or a hobby such as kayaking).
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
- For preference queries, directly utilize the user's stated interests, background, and constraints.
- Only if the question literally asks to recall a stored name or value ("remind me of the name of...", "what was the name of..."), answer in a single sentence with that value only. Never apply this to advice, recommendation, yes/no, or counting/time questions.
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
1. Exact counts and identifying details: Recount the extracted facts to ensure no items are omitted or double-counted across different chat sessions. Include specific details like roles, categories, or names. If asked "How many...", provide the exact integer count of mentioned items directly.
2. Temporal order & updates: If an updated fact or state is present, the primary answer MUST state the latest/updated value directly (the current plan, price, or role, rather than presenting old and new equally).
3. Personalization & Recommendations: If asked for recommendations, suggestions, or advice, and the evidence contains the user's relevant preferences, background, or constraints, ensure the answer delivers suggestions that directly reflect them rather than abstaining. If the evidence contains nothing about the user that bears on the request, say so.
4. Abstention vs Grounded Answers: If one of the candidates provides concrete details drawn from the extracted facts (specific dates, named entities, or exact counts), prefer that candidate over a generic abstention. Only abstain if the information was genuinely never mentioned in the chat history.
5. Directness: Keep the answer clear, complete, and direct.
6. Explicit recall requests only: if the question literally asks to recall a stored name or value ("remind me of the name of...", "what was the name of the..."), the final answer should be a single sentence stating that value in full (never drop an associated title, link/URL, or identifier); strip unrequested descriptions and never list multiple alternative values to hedge. Do NOT shorten answers to advice, recommendation, suggestion, yes/no, counting, or time-calculation questions - those must stay complete.

Question: {query}
Current Date: {formatted_date}

{evidence_context}

Candidate Answers:
{candidate_text}

Return JSON only:
{{"answer":"final verified answer","reason":"brief reason for choice or correction"}}"""

