"""
A-ART Judge Protocol: Interface Definition for Safety Evaluators.

This module defines the Protocol interface for all Judge implementations.
Judges evaluate target model responses for safety violations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from llm_red_team.agents.judges.agentic_judge import JudgeOutput


@runtime_checkable
class JudgeProtocol(Protocol):
    """
    Protocol for safety evaluation judges.

    The Judge evaluates target model responses to determine if
    the attack was successful in bypassing safety measures.
    """

    @property
    def name(self) -> str:
        """Return the judge's identifier."""
        ...

    def evaluate(
        self,
        adversarial_prompt: str,
        model_response: str,
        risk_category: str,
        input_guard_triggered: bool = False,
        input_guard_score: float = 0.0,
    ) -> tuple[JudgeOutput, float]:
        """
        Evaluate if a model response violates safety policies.

        Args:
            adversarial_prompt: The attack prompt that was sent
            model_response: The target model's response
            risk_category: The risk category being tested
            input_guard_triggered: Whether input guard detected the attack
            input_guard_score: Input guard's unsafe probability score

        Returns:
            Tuple of (JudgeOutput, elapsed_ms)
        """
        ...

    def reset(self) -> None:
        """Reset judge state for a new session."""
        ...
