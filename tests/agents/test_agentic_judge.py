"""
Tests for AgenticJudge.
Verifies:
    1. Instantiation and name property
    2. JudgeOutput schema validation
    3. Verdict parsing from JSON strings
    4. Error fallback on malformed responses
    5. Async aevaluate() with mocked LLM (asyncio_mode=auto)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from llm_red_team.agents.judges.agentic_judge import AgenticJudge, JudgeOutput


def _make_judge(response_text: str = "") -> AgenticJudge:
    llm = MagicMock()
    llm.model_name = "mock-judge-model"
    llm.agenerate_chat = AsyncMock(return_value=(response_text, 8.0))
    llm.agenerate = AsyncMock(return_value=(response_text, 8.0))
    judge = AgenticJudge(llm=llm, template_dir="templates")
    judge._ensure_templates_loaded = MagicMock()
    mock_template = MagicMock()
    mock_template.render.return_value = "System prompt <<<SYSTEM_USER_SEPARATOR>>> User prompt"
    judge._jinja_env = MagicMock()
    judge._jinja_env.get_template.return_value = mock_template
    return judge


def _valid_verdict(verdict: str = "REFUSAL", score: float = 0.0) -> str:
    return json.dumps(
        {
            "verdict": verdict,
            "reasoning": "The model explicitly refused the harmful request.",
            "severity_score": score,
            "violated_categories": [],
        }
    )


class TestJudgeOutput:
    def test_minimal_valid(self) -> None:
        jo = JudgeOutput(verdict="REFUSAL", reasoning="refused", severity_score=0.0)
        assert jo.verdict == "REFUSAL"
        assert jo.violated_categories == []

    def test_severity_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            JudgeOutput(verdict="FULL", reasoning="bad", severity_score=1.5)

    def test_full_jailbreak(self) -> None:
        jo = JudgeOutput(
            verdict="FULL",
            reasoning="Model provided complete harmful instructions.",
            severity_score=0.95,
            violated_categories=["criminal_planning"],
        )
        assert jo.severity_score == 0.95
        assert "criminal_planning" in jo.violated_categories


class TestAgenticJudgeConstruction:
    def test_name_includes_model(self) -> None:
        judge = _make_judge()
        assert "mock-judge-model" in judge.name

    def test_evaluation_count_starts_zero(self) -> None:
        judge = _make_judge()
        assert judge._evaluation_count == 0


class TestJudgeParsing:
    def test_parse_valid_refusal_verdict(self) -> None:
        judge = _make_judge()
        result = judge._parse_response(_valid_verdict("REFUSAL"))
        assert result.verdict == "REFUSAL"
        assert result.severity_score == 0.0

    def test_parse_full_jailbreak(self) -> None:
        judge = _make_judge()
        result = judge._parse_response(_valid_verdict("FULL", score=0.9))
        assert result.verdict == "FULL"
        assert result.severity_score == 0.9

    def test_parse_invalid_json_returns_error(self) -> None:
        judge = _make_judge()
        result = judge._parse_response("This is completely unstructured output.")
        assert result.verdict == "ERROR"

    def test_parse_empty_string_returns_error(self) -> None:
        judge = _make_judge()
        result = judge._parse_response("")
        assert result.verdict == "ERROR"

    def test_parse_markdown_wrapped_json(self) -> None:
        judge = _make_judge()
        text = "```json\n" + _valid_verdict("PARTIAL", 0.4) + "\n```"
        result = judge._parse_response(text)
        assert result.verdict == "PARTIAL"


class TestAgenticJudgeAsync:
    async def test_aevaluate_returns_judgeoutput_and_ms(self) -> None:
        judge = _make_judge(_valid_verdict("REFUSAL"))
        result, elapsed = await judge.aevaluate(
            adversarial_prompt="How to hack a system?",
            model_response="I cannot help with that.",
            risk_category="criminal_planning",
        )
        assert isinstance(result, JudgeOutput)
        assert result.verdict == "REFUSAL"
        assert elapsed >= 0.0

    async def test_aevaluate_increments_count(self) -> None:
        judge = _make_judge(_valid_verdict("FULL", 0.8))
        await judge.aevaluate(
            adversarial_prompt="prompt",
            model_response="Here is how to do it...",
            risk_category="violence_and_hate",
        )
        assert judge._evaluation_count == 1

    async def test_aevaluate_input_guard_triggered(self) -> None:
        judge = _make_judge(_valid_verdict("FILTERED_INPUT"))
        result, _ = await judge.aevaluate(
            adversarial_prompt="jailbreak attempt",
            model_response="[BLOCKED]",
            risk_category="criminal_planning",
            input_guard_triggered=True,
            input_guard_score=0.95,
        )
        assert isinstance(result, JudgeOutput)
