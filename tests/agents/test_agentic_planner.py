"""
Tests for AgenticPlanner.

Verifies:
    1. Instantiation and name property
    2. PlannerOutput schema validation
    3. Random-strategy fallback when JSON parsing fails
    4. Async aplan() smoke test with mocked LLM
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from llm_red_team.agents.planners.agentic_planner import AgenticPlanner, PlannerOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_planner(response_text: str = "") -> AgenticPlanner:
    """Return an AgenticPlanner with a fully mocked LLM + template env."""
    llm = MagicMock()
    llm.model_name = "mock-planner-model"
    # Both agenerate and agenerate_chat must be AsyncMock.
    llm.agenerate = AsyncMock(return_value=(response_text, 10.0))
    llm.agenerate_chat = AsyncMock(return_value=(response_text, 10.0))

    planner = AgenticPlanner(llm=llm, template_dir="templates")
    planner._ensure_templates_loaded = MagicMock()  # type: ignore[method-assign]
    mock_template = MagicMock()
    # Return plain text WITHOUT <<<SYSTEM_USER_SEPARATOR>>> so agenerate() path is exercised.
    mock_template.render.return_value = "Rendered planner prompt plain text"
    planner._jinja_env = MagicMock()
    planner._jinja_env.get_template.return_value = mock_template
    return planner


def _valid_json_response(**kwargs: object) -> str:
    data = {
        "action": "CONTINUE",
        "strategy": "role_play",
        "rationale": "Testing the plan step by step.",
        "confidence": 0.8,
    }
    data.update(kwargs)
    return json.dumps(data)


# ---------------------------------------------------------------------------
# PlannerOutput model
# ---------------------------------------------------------------------------


class TestPlannerOutput:
    def test_valid_minimal(self) -> None:
        po = PlannerOutput(action="CONTINUE", strategy="role_play", rationale="reason")
        assert po.action == "CONTINUE"
        assert po.confidence == 0.7  # default

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            PlannerOutput(action="CONTINUE", strategy="s", rationale="r", confidence=1.5)

    def test_all_fields(self) -> None:
        po = PlannerOutput(
            action="PIVOT",
            strategy="obfuscation",
            rationale="Need to change approach.",
            confidence=0.9,
        )
        assert po.strategy == "obfuscation"
        assert po.confidence == 0.9


# ---------------------------------------------------------------------------
# AgenticPlanner construction
# ---------------------------------------------------------------------------


class TestAgenticPlannerConstruction:
    def test_name_includes_model(self) -> None:
        planner = _make_planner()
        assert "mock-planner-model" in planner.name

    def test_plan_count_starts_zero(self) -> None:
        planner = _make_planner()
        assert planner._plan_count == 0

    def test_reasoning_model_scales_max_tokens(self) -> None:
        llm = MagicMock()
        llm.model_name = "reasoning-model"
        planner = AgenticPlanner(
            llm=llm,
            max_tokens=256,
            is_reasoning_model=True,
        )
        assert planner._max_tokens > 256  # should be scaled up


# ---------------------------------------------------------------------------
# Async aplan()
# ---------------------------------------------------------------------------


class TestAgenticPlannerAsync:
    async def test_aplan_parses_valid_json(self) -> None:
        planner = _make_planner(_valid_json_response())
        result, elapsed = await planner.aplan(
            seed_prompt="Explain how to hack a system.",
            turn_number=0,
            max_turns=5,
            risk_category="criminal_planning",
        )
        assert isinstance(result, PlannerOutput)
        assert result.action == "CONTINUE"
        assert elapsed >= 0.0

    async def test_aplan_falls_back_on_invalid_json(self) -> None:
        """When LLM returns unparseable text, planner should return a fallback."""
        planner = _make_planner("This is not JSON at all, sorry!")
        result, elapsed = await planner.aplan(
            seed_prompt="prompt",
            turn_number=0,
            max_turns=5,
            risk_category="violence_and_hate",
        )
        assert isinstance(result, PlannerOutput)
        assert result.action in ("CONTINUE", "ESCALATE", "PIVOT", "TERMINATE")

    async def test_aplan_increments_count(self) -> None:
        planner = _make_planner(_valid_json_response())
        await planner.aplan(
            seed_prompt="prompt",
            turn_number=0,
            max_turns=5,
            risk_category="criminal_planning",
        )
        assert planner._plan_count == 1

    async def test_aplan_with_history(self) -> None:
        planner = _make_planner(_valid_json_response(action="ESCALATE", strategy="obfuscation"))
        result, _ = await planner.aplan(
            seed_prompt="prompt",
            turn_number=2,
            max_turns=5,
            risk_category="criminal_planning",
            previous_verdict="REFUSAL",
            previous_strategy="role_play",
            previous_feedback="Try different approach.",
        )
        assert isinstance(result, PlannerOutput)
