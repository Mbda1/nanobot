"""LLM-as-judge evaluation utilities.

Sends (question, response, criterion) triples to the local Mistral model and
returns a PASS/FAIL verdict. Zero cloud tokens — all inference is local.

Usage:
    from nanobot.agent.eval import judge_response

    passed, reason = await judge_response(
        question="What is 2+2?",
        response="The answer is 4.",
        criterion="Directly states the correct numerical answer",
    )
"""

from __future__ import annotations

from nanobot.agent.local_llm import ollama_chat
from nanobot.config.constants import JUDGE_MODEL_DEFAULT

_JUDGE_PROMPT = """\
You are a strict AI response evaluator. Read the user question and the AI response below, \
then decide if the response meets the criterion.

User question: {question}

AI response: {response}

Criterion: {criterion}

Reply with exactly one word on the first line — PASS or FAIL — then one brief sentence explaining why.\
"""

_DEFAULT_MODEL = JUDGE_MODEL_DEFAULT  # "ollama/mistral-nemo" from constants
_DEFAULT_TIMEOUT = 30.0


async def judge_response(
    question: str,
    response: str,
    criterion: str,
    *,
    model: str = _DEFAULT_MODEL,
    timeout: float = _DEFAULT_TIMEOUT,
) -> tuple[bool, str]:
    """Judge a (question, response) pair against a criterion using local Mistral.

    Returns:
        (passed, reason) — passed is True if the judge says PASS, reason is the
        first sentence of the judge's explanation.
    """
    prompt = _JUDGE_PROMPT.format(
        question=question.strip(),
        response=response.strip(),
        criterion=criterion.strip(),
    )

    # ollama_chat takes the bare model name (strip "ollama/" prefix if present)
    model_name = model.split("/")[-1]
    content, _ = await ollama_chat(
        model_name,
        [{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.0,
        timeout=timeout,
    )

    if not response.strip():
        return False, "Agent response is empty — nothing to evaluate"

    if not content:
        return False, "Judge returned empty response (Ollama may be offline)"

    # Parse first word of response — tolerant of punctuation / mixed case.
    first_word = content.strip().split()[0].upper().rstrip(".,:")
    passed = first_word == "PASS"
    return passed, content.strip()
