"""LLM-as-judge evaluation suite.

Runs (question, response, criterion) triples through local Mistral and asserts
the verdict matches the expected outcome. Catches quality regressions when the
cloud model changes — if the agent starts giving worse answers, the PASS cases
will flip to FAIL here.

Requires a running Ollama instance (host.docker.internal:11434).
Run the suite explicitly:
    pytest -m llm
Skip in fast CI:
    pytest -m 'not llm'
"""

import dataclasses

import pytest

from nanobot.agent.eval import judge_response


@dataclasses.dataclass
class EvalCase:
    name: str
    question: str
    response: str
    criterion: str
    expect_pass: bool


# ---------------------------------------------------------------------------
# Golden eval cases
# Five should PASS (good responses), four should FAIL (bad responses).
# Keep criteria concrete and objective — Mistral 7B handles those reliably.
# ---------------------------------------------------------------------------
EVAL_CASES: list[EvalCase] = [
    # --- PASS cases ---
    EvalCase(
        name="direct_factual_answer",
        question="What is the capital of France?",
        response="Paris is the capital of France. It has been the country's political and cultural center for centuries.",
        criterion="Directly states that Paris is the capital of France",
        expect_pass=True,
    ),
    EvalCase(
        name="math_answer",
        question="What is 15 multiplied by 7?",
        response="15 multiplied by 7 equals 105.",
        criterion="States the correct numerical result of 15 × 7",
        expect_pass=True,
    ),
    EvalCase(
        name="appropriate_uncertainty",
        question="What will the stock market do tomorrow?",
        response=(
            "I can't predict stock market movements — no one can reliably do so. "
            "For current prices you could check Yahoo Finance or your brokerage app."
        ),
        criterion="Acknowledges it cannot predict the future rather than inventing a prediction",
        expect_pass=True,
    ),
    EvalCase(
        name="actionable_how_to",
        question="How do I install Python on Ubuntu?",
        response=(
            "Run `sudo apt update && sudo apt install python3 python3-pip` in your terminal. "
            "Verify with `python3 --version`."
        ),
        criterion="Provides a concrete installation command the user can run",
        expect_pass=True,
    ),
    EvalCase(
        name="session_reset_confirmed",
        question="/new",
        response="New session started.",
        criterion="Confirms that a new conversation session has been started",
        expect_pass=True,
    ),

    # --- FAIL cases (deliberately bad responses — judge must reject these) ---
    EvalCase(
        name="question_completely_ignored",
        question="What is 2 + 2?",
        response="That's a fascinating topic! Let me think about what would be most helpful for you today.",
        criterion="Provides the numerical answer to 2 + 2",
        expect_pass=False,
    ),
    EvalCase(
        name="factually_wrong_answer",
        question="What is the capital of France?",
        response="The capital of France is Berlin, a vibrant city known for its history and culture.",
        criterion="States the correct capital of France (Paris, not Berlin or any other city)",
        expect_pass=False,
    ),
    EvalCase(
        name="off_topic_response",
        question="What is the weather like in Chicago today?",
        response=(
            "I'd be happy to help you write Python code! "
            "What kind of program are you looking to build?"
        ),
        criterion="Addresses the user's question about weather in Chicago",
        expect_pass=False,
    ),
    EvalCase(
        name="unhelpful_refusal",
        question="What is the speed of light?",
        response="I'm sorry, I'm not able to answer questions about physics or science.",
        criterion="Provides the speed of light or a genuinely helpful response to a basic science question",
        expect_pass=False,
    ),
]


@pytest.mark.llm
@pytest.mark.parametrize("case", EVAL_CASES, ids=lambda c: c.name)
async def test_judge_eval(ollama_available, case: EvalCase) -> None:
    """Each case must produce the expected PASS/FAIL verdict from local Mistral."""
    passed, reason = await judge_response(
        question=case.question,
        response=case.response,
        criterion=case.criterion,
    )
    assert passed == case.expect_pass, (
        f"Expected {'PASS' if case.expect_pass else 'FAIL'} but judge said "
        f"{'PASS' if passed else 'FAIL'}.\nReason: {reason}"
    )


@pytest.mark.llm
async def test_judge_rejects_empty_response(ollama_available) -> None:
    """An empty agent response should always fail any criterion."""
    passed, reason = await judge_response(
        question="What is 2+2?",
        response="",
        criterion="Provides the numerical answer to 2+2",
    )
    assert not passed, f"Expected FAIL for empty response, got PASS.\nReason: {reason}"


@pytest.mark.llm
async def test_judge_module_importable() -> None:
    """Smoke test: eval module imports and judge_response is callable."""
    from nanobot.agent.eval import judge_response as jr  # noqa: F401
    assert callable(jr)
