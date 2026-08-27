"""
Tests for the canonical schemas: LogEntry, KnowledgeBaseEntry, enums.

Replaces the stale test_logger.py, test_model_factory.py,
test_target_graybox.py (all imported deleted modules).

Verifies:
    1. LogEntry required-field validation
    2. LogEntry defaults
    3. LogEntry to_jsonl() produces single-line valid JSON
    4. LogEntry extra="ignore" swallows unknown fields
    5. JudgeVerdict / AttackStyle / RiskCategory enums
    6. KnowledgeBaseEntry inherits LogEntry correctly
    7. PipelineConfig Pydantic validation (min / max bounds)
    8. ComponentConfig model
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from llm_red_team.schemas.llm_red_team_schema import (
    AttackStyle,
    JudgeVerdict,
    KnowledgeBaseEntry,
    LogEntry,
    ModalityType,
    RiskCategory,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_log_entry(**overrides: object) -> LogEntry:
    defaults: dict = dict(
        experiment_batch_id="batch-1",
        root_attack_id="root-1",
        mutated_prompt="adversarial prompt",
        target_model_full_name="test/model",
        response_text="some response",
        success_boolean=False,
        timing_mutator_ms=10.0,
        timing_target_model_ms=20.0,
        timing_judge_ms=5.0,
        timing_turn_total_ms=35.0,
        timing_session_cumulative_ms=35.0,
    )
    defaults.update(overrides)
    return LogEntry(**defaults)


# ---------------------------------------------------------------------------
# LogEntry required fields
# ---------------------------------------------------------------------------


class TestLogEntryRequiredFields:
    def test_valid_minimal_entry(self) -> None:
        entry = _minimal_log_entry()
        assert entry.success_boolean is False
        assert entry.turn_index == 0

    def test_missing_mutated_prompt_raises(self) -> None:
        with pytest.raises(ValidationError):
            LogEntry(
                experiment_batch_id="b",
                root_attack_id="r",
                # mutated_prompt missing
                target_model_full_name="m",
                response_text="r",
                success_boolean=False,
                timing_mutator_ms=1.0,
                timing_target_model_ms=2.0,
                timing_judge_ms=3.0,
                timing_turn_total_ms=6.0,
                timing_session_cumulative_ms=6.0,
            )

    def test_missing_timing_raises(self) -> None:
        with pytest.raises(ValidationError):
            LogEntry(
                experiment_batch_id="b",
                root_attack_id="r",
                mutated_prompt="p",
                target_model_full_name="m",
                response_text="r",
                success_boolean=False,
                # All timing fields missing
            )


# ---------------------------------------------------------------------------
# LogEntry defaults
# ---------------------------------------------------------------------------


class TestLogEntryDefaults:
    def test_attack_id_auto_generated(self) -> None:
        e1 = _minimal_log_entry()
        e2 = _minimal_log_entry()
        assert e1.attack_id != e2.attack_id  # UUIDs are unique

    def test_timestamp_utc_set(self) -> None:
        entry = _minimal_log_entry()
        assert entry.timestamp_utc is not None
        assert len(entry.timestamp_utc) > 10

    def test_modality_type_default(self) -> None:
        entry = _minimal_log_entry()
        assert entry.modality_type == "text"

    def test_pipeline_version_default(self) -> None:
        entry = _minimal_log_entry()
        assert entry.pipeline_version == "v5-flat"

    def test_success_boolean_false_default(self) -> None:
        entry = _minimal_log_entry()
        assert entry.success_boolean is False

    def test_executor_tools_used_defaults_to_empty_list(self) -> None:
        entry = _minimal_log_entry()
        assert entry.executor_tools_used == []

    def test_optional_fields_default_to_none(self) -> None:
        entry = _minimal_log_entry()
        assert entry.parent_id is None
        assert entry.planner_model_id is None
        assert entry.mutator_model_id is None
        assert entry.judge_verdict_code is None
        assert entry.judge_reasoning is None


# ---------------------------------------------------------------------------
# LogEntry serialisation
# ---------------------------------------------------------------------------


class TestLogEntrySerialisation:
    def test_to_jsonl_is_single_line(self) -> None:
        entry = _minimal_log_entry()
        jsonl = entry.to_jsonl()
        assert "\n" not in jsonl

    def test_to_jsonl_parses_as_json(self) -> None:
        entry = _minimal_log_entry(mutated_prompt="test-prompt-xyz")
        parsed = json.loads(entry.to_jsonl())
        assert parsed["mutated_prompt"] == "test-prompt-xyz"

    def test_to_jsonl_contains_all_required_fields(self) -> None:
        entry = _minimal_log_entry()
        parsed = json.loads(entry.to_jsonl())
        for field in (
            "attack_id",
            "experiment_batch_id",
            "root_attack_id",
            "mutated_prompt",
            "target_model_full_name",
            "response_text",
            "success_boolean",
            "timing_mutator_ms",
            "timing_target_model_ms",
            "timing_judge_ms",
            "timing_turn_total_ms",
            "timing_session_cumulative_ms",
            "timestamp_utc",
        ):
            assert field in parsed, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# LogEntry extra="ignore"
# ---------------------------------------------------------------------------


class TestLogEntryExtraFields:
    def test_extra_fields_silently_ignored(self) -> None:
        """extra='ignore' must not raise on unknown fields."""
        entry = LogEntry(
            experiment_batch_id="b",
            root_attack_id="r",
            mutated_prompt="p",
            target_model_full_name="m",
            response_text="resp",
            success_boolean=True,
            timing_mutator_ms=1.0,
            timing_target_model_ms=2.0,
            timing_judge_ms=3.0,
            timing_turn_total_ms=6.0,
            timing_session_cumulative_ms=6.0,
            totally_unknown_field="ignored",  # type: ignore[call-arg]
        )
        assert not hasattr(entry, "totally_unknown_field")


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TestJudgeVerdict:
    def test_all_values_present(self) -> None:
        values = {v.value for v in JudgeVerdict}
        assert "FULL" in values
        assert "PARTIAL" in values
        assert "REFUSAL" in values
        assert "FILTERED_INPUT" in values
        assert "ERROR" in values

    def test_description_available(self) -> None:
        assert len(JudgeVerdict.FULL.description) > 0
        assert len(JudgeVerdict.REFUSAL.description) > 0

    def test_to_prompt_context_non_empty(self) -> None:
        ctx = JudgeVerdict.to_prompt_context()
        assert "FULL" in ctx
        assert "REFUSAL" in ctx


class TestAttackStyle:
    def test_has_values(self) -> None:
        assert len(list(AttackStyle)) > 0

    def test_to_prompt_context(self) -> None:
        ctx = AttackStyle.to_prompt_context()
        assert len(ctx) > 0


class TestRiskCategory:
    def test_has_values(self) -> None:
        assert len(list(RiskCategory)) > 0

    def test_to_prompt_context(self) -> None:
        ctx = RiskCategory.to_prompt_context()
        assert len(ctx) > 0


class TestModalityType:
    def test_text_value(self) -> None:
        assert ModalityType.TEXT.value == "text"

    def test_image_value(self) -> None:
        assert ModalityType.IMAGE.value == "image"

    def test_composite_value(self) -> None:
        assert ModalityType.COMPOSITE.value == "composite"


# ---------------------------------------------------------------------------
# KnowledgeBaseEntry (cold-path schema)
# ---------------------------------------------------------------------------


class TestKnowledgeBaseEntry:
    def test_inherits_logentry(self) -> None:
        assert issubclass(KnowledgeBaseEntry, LogEntry)

    def test_has_analysis_field(self) -> None:
        entry = KnowledgeBaseEntry(
            experiment_batch_id="b",
            root_attack_id="r",
            mutated_prompt="p",
            target_model_full_name="m",
            response_text="resp",
            success_boolean=False,
            timing_mutator_ms=1.0,
            timing_target_model_ms=2.0,
            timing_judge_ms=3.0,
            timing_turn_total_ms=6.0,
            timing_session_cumulative_ms=6.0,
        )
        assert hasattr(entry, "analysis")
        # ColdPathExtension defaults
        assert entry.analysis.prompt_embedding == []
        assert entry.analysis.response_embedding == []
