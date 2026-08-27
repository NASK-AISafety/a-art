"""
A-ART Agentic Judge: LLM-based Safety Evaluator.

This judge uses a shared LLM with structured output to evaluate
whether target model responses violate safety policies.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

from llm_red_team.schemas.llm_red_team_schema import RiskCategory
from llm_red_team.utils.json_parsing import extract_json_object as _extract_json_object

if TYPE_CHECKING:
    from llm_red_team.models.shared_llm import SharedLLM

logger = logging.getLogger(__name__)


class JudgeOutput(BaseModel):
    """Structured output from the Judge."""

    verdict: str = Field(..., description="Safety verdict code")
    reasoning: str = Field(..., description="Analysis of the response")
    severity_score: float = Field(0.0, ge=0.0, le=1.0)
    violated_categories: list[str] = Field(default_factory=list)


class AgenticJudge:
    """
    LLM-based safety evaluator using structured output.

    Uses the shared LLM to evaluate target responses against
    safety policies and return structured verdicts.
    """

    # Reasoning judges spend much of their budget on chain-of-thought before the
    # JSON verdict; scale the cap so the verdict isn't truncated (mirrors the
    # planner/mutator). Without this a reasoning judge emits malformed JSON.
    _REASONING_TOKEN_MULTIPLIER: int = 4

    def __init__(
        self,
        llm: SharedLLM,
        template_dir: str = "templates",
        template_name: str = "judge_structured.j2",
        temperature: float = 0.1,
        max_tokens: int = 512,
        is_reasoning_model: bool = False,
    ):
        self._llm = llm
        self._template_dir = template_dir
        self._template_name = template_name
        self._temperature = temperature
        self._is_reasoning_model = is_reasoning_model
        self._max_tokens = (
            max_tokens * self._REASONING_TOKEN_MULTIPLIER if is_reasoning_model else max_tokens
        )
        if is_reasoning_model:
            logger.info(
                "Judge reasoning model: max_tokens scaled %d → %d (×%d)",
                max_tokens,
                self._max_tokens,
                self._REASONING_TOKEN_MULTIPLIER,
            )
        self._jinja_env: Environment | None = None
        self._evaluation_count = 0

    @property
    def name(self) -> str:
        """Return the judge's identifier."""
        return f"AgenticJudge({self._llm.model_name})"

    def _ensure_templates_loaded(self) -> None:
        """Lazy-load Jinja2 templates."""
        if self._jinja_env is not None:
            return

        template_path = Path(self._template_dir)
        if not template_path.exists():
            template_path = Path(__file__).parent.parent.parent.parent.parent / self._template_dir

        self._jinja_env = Environment(
            loader=FileSystemLoader(str(template_path)),
            autoescape=False,
        )

    def evaluate(
        self,
        adversarial_prompt: str,
        model_response: str,
        risk_category: str,
        input_guard_triggered: bool = False,
        input_guard_score: float = 0.0,
        image_reference: str | None = None,
    ) -> tuple[JudgeOutput, float]:
        """
        Evaluate if a model response violates safety policies.

        Args:
            adversarial_prompt: The attack prompt that was sent
            model_response: The target model's response
            risk_category: The risk category being tested
            input_guard_triggered: Whether input guard detected the attack
            input_guard_score: Input guard's unsafe probability score
            image_reference: Optional image path/URL for multimodal attacks

        Returns:
            Tuple of (JudgeOutput, elapsed_ms)
        """
        self._ensure_templates_loaded()
        self._evaluation_count += 1

        risk_categories_context = RiskCategory.to_prompt_context()

        template = self._jinja_env.get_template(self._template_name)
        rendered = template.render(
            adversarial_prompt=adversarial_prompt,
            model_response=model_response,
            risk_category=risk_category,
            risk_categories_context=risk_categories_context,
            input_guard_triggered=input_guard_triggered,
            input_guard_score=input_guard_score,
            image_reference=image_reference,
        )

        # Model-agnostic chat templates use <<<SYSTEM_USER_SEPARATOR>>>
        if "<<<SYSTEM_USER_SEPARATOR>>>" in rendered:
            parts = rendered.split("<<<SYSTEM_USER_SEPARATOR>>>", 1)
            messages = [
                {"role": "system", "content": parts[0].strip()},
                {"role": "user", "content": parts[1].strip()},
            ]
            response, elapsed_ms = self._llm.generate_chat(
                messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                do_sample=False,
            )
        else:
            # Legacy templates with hardcoded chat tokens (e.g. <|im_start|>)
            response, elapsed_ms = self._llm.generate(
                rendered,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                do_sample=False,
            )

        output = self._parse_response(response)
        return output, elapsed_ms

    async def aevaluate(
        self,
        adversarial_prompt: str,
        model_response: str,
        risk_category: str,
        input_guard_triggered: bool = False,
        input_guard_score: float = 0.0,
        image_reference: str | None = None,
    ) -> tuple[JudgeOutput, float]:
        """Async version of evaluate() — uses SharedLLM async methods."""
        self._ensure_templates_loaded()
        self._evaluation_count += 1

        risk_categories_context = RiskCategory.to_prompt_context()

        template = self._jinja_env.get_template(self._template_name)
        rendered = template.render(
            adversarial_prompt=adversarial_prompt,
            model_response=model_response,
            risk_category=risk_category,
            risk_categories_context=risk_categories_context,
            input_guard_triggered=input_guard_triggered,
            input_guard_score=input_guard_score,
            image_reference=image_reference,
        )

        if "<<<SYSTEM_USER_SEPARATOR>>>" in rendered:
            parts = rendered.split("<<<SYSTEM_USER_SEPARATOR>>>", 1)
            messages = [
                {"role": "system", "content": parts[0].strip()},
                {"role": "user", "content": parts[1].strip()},
            ]
            response, elapsed_ms = await self._llm.agenerate_chat(
                messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                do_sample=False,
            )
            output = self._parse_response(response)
            # Single retry: if the model returned malformed JSON, append its
            # bad response and a correction request, then re-query once.
            # Cheap at temperature=0.0 (greedy); avoids ERROR verdicts in logs.
            if output.verdict == "ERROR":
                logger.warning("Judge returned malformed JSON — retrying with correction prompt")
                retry_messages = messages + [
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": (
                            "Your response was not valid JSON. "
                            "Please output ONLY a JSON object with keys: "
                            "verdict, reasoning, severity_score, violated_categories. "
                            "No markdown, no prose — just the JSON object."
                        ),
                    },
                ]
                retry_response, retry_ms = await self._llm.agenerate_chat(
                    retry_messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    do_sample=False,
                )
                elapsed_ms += retry_ms
                output = self._parse_response(retry_response)
        else:
            response, elapsed_ms = await self._llm.agenerate(
                rendered,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                do_sample=False,
            )
            output = self._parse_response(response)

        return output, elapsed_ms

    def _parse_response(self, response: str) -> JudgeOutput:
        """Parse the LLM response into structured JudgeOutput."""
        logger.debug(f"Raw judge response: {response!r}")
        try:
            data = _extract_json_object(response)
            if data is None:
                return JudgeOutput(
                    verdict="ERROR",
                    reasoning="Failed to parse judge output - no JSON found.",
                    severity_score=0.0,
                )

            # Fix: Handle None values and lists (model sometimes returns malformed data)
            if isinstance(data.get("reasoning"), list):
                data["reasoning"] = " ".join(str(x) for x in data["reasoning"])

            # Normalize None values to defaults before Pydantic validation
            if not data.get("verdict"):
                data["verdict"] = "ERROR"
            if not data.get("reasoning"):
                data["reasoning"] = "No reasoning provided"
            if data.get("severity_score") is None:
                data["severity_score"] = 0.0
            if "violated_categories" not in data or data["violated_categories"] is None:
                data["violated_categories"] = []

            return JudgeOutput(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse judge output: {e}")
            return JudgeOutput(
                verdict="ERROR",
                reasoning=f"Parse error: {str(e)[:100]}",
                severity_score=0.0,
            )

    def validate_template(self) -> None:
        """
        Eagerly load the Jinja2 environment and validate the template exists.

        Call this from pipeline.initialize() so a missing template raises
        TemplateNotFound immediately at startup, not hours into an HPC run.
        """
        self._ensure_templates_loaded()
        assert self._jinja_env is not None
        self._jinja_env.get_template(self._template_name)

    def reset(self) -> None:
        """Reset judge state for a new session."""
        self._evaluation_count = 0
