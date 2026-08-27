"""Memory Keeper agent: conservative short-term memory updates from judge output."""

from __future__ import annotations

from dataclasses import dataclass

from llm_red_team.core.short_term_memory import ShortTermMemory


@dataclass
class MemoryKeeperConfig:
    """Conservative thresholds for promoting patterns into short-term memory."""

    min_samples_for_pattern: int = 3
    min_win_rate_for_winning_pattern: float = 0.45


class MemoryKeeper:
    """Assess turn outcomes and update short-term memory conservatively."""

    def __init__(self, memory: ShortTermMemory, config: MemoryKeeperConfig | None = None):
        self._memory = memory
        self._config = config or MemoryKeeperConfig()

    async def aassess_and_update(
        self,
        strategy: str,
        risk_category: str,
        judge_verdict: str,
        success: bool,
        severity_score: float,
        input_guard_triggered: bool,
        output_guard_triggered: bool,
    ) -> bool:
        """Update memory from one judged turn; returns True when memory changed."""
        return await self._memory.record_outcome(
            strategy=strategy,
            risk_category=risk_category,
            verdict=judge_verdict,
            success=success,
            severity_score=severity_score,
            input_guard_triggered=input_guard_triggered,
            output_guard_triggered=output_guard_triggered,
            min_samples_for_pattern=self._config.min_samples_for_pattern,
            min_win_rate_for_winning_pattern=self._config.min_win_rate_for_winning_pattern,
        )

    async def asnapshot_for_prompt(self) -> str:
        """Expose memory snapshot for planner/mutator prompt injection."""
        return await self._memory.snapshot_for_prompt()
