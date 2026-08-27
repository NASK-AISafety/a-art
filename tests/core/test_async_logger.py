"""
Tests for AsyncJSONLLogger.

Verifies:
    1. Basic async write / file creation
    2. Parent directory auto-creation
    3. Buffer batching (flush on buffer_size threshold)
    4. Context manager (__aenter__ / __aexit__) drains queue
    5. Log entry validated through LogEntry schema
    6. Partial / corrupted lines do not crash the consumer
    7. entries_written counter
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from llm_red_team.core.async_logger import AsyncJSONLLogger
from llm_red_team.schemas.llm_red_team_schema import LogEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_entry(**kwargs: object) -> LogEntry:
    """Build the smallest valid LogEntry (only required fields)."""
    defaults: dict = dict(
        experiment_batch_id="batch-test",
        root_attack_id="root-test",
        mutated_prompt="test adversarial prompt",
        target_model_full_name="test-model",
        response_text="test response",
        success_boolean=False,
        timing_mutator_ms=1.0,
        timing_target_model_ms=2.0,
        timing_judge_ms=3.0,
        timing_turn_total_ms=6.0,
        timing_session_cumulative_ms=6.0,
    )
    defaults.update(kwargs)
    return LogEntry(**defaults)


# ---------------------------------------------------------------------------
# Basic write
# ---------------------------------------------------------------------------


class TestAsyncJSONLLoggerBasic:
    async def test_creates_file_on_write(self, tmp_path: Path) -> None:
        filepath = tmp_path / "out.jsonl"
        async with AsyncJSONLLogger(filepath, buffer_size=1) as log:
            await log.log(_minimal_entry())
        assert filepath.exists()

    async def test_creates_parent_directories(self, tmp_path: Path) -> None:
        filepath = tmp_path / "nested" / "deep" / "out.jsonl"
        async with AsyncJSONLLogger(filepath, buffer_size=1) as log:
            await log.log(_minimal_entry())
        assert filepath.exists()

    async def test_writes_valid_jsonl(self, tmp_path: Path) -> None:
        filepath = tmp_path / "out.jsonl"
        async with AsyncJSONLLogger(filepath, buffer_size=1) as log:
            await log.log(_minimal_entry())
            await log.log(_minimal_entry())
            await log.log(_minimal_entry())

        lines = filepath.read_text().splitlines()
        assert len(lines) == 3
        for line in lines:
            parsed = json.loads(line)
            assert "attack_id" in parsed
            assert "timestamp_utc" in parsed

    async def test_appends_not_overwrites(self, tmp_path: Path) -> None:
        filepath = tmp_path / "out.jsonl"
        async with AsyncJSONLLogger(filepath, buffer_size=1) as log:
            await log.log(_minimal_entry())

        async with AsyncJSONLLogger(filepath, buffer_size=1) as log:
            await log.log(_minimal_entry())

        lines = filepath.read_text().splitlines()
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# entries_written counter
# ---------------------------------------------------------------------------


class TestAsyncJSONLLoggerCounter:
    async def test_entries_written_counter(self, tmp_path: Path) -> None:
        filepath = tmp_path / "out.jsonl"
        async with AsyncJSONLLogger(filepath, buffer_size=1) as log:
            assert log.entries_written == 0
            await log.log(_minimal_entry())
            await log.log(_minimal_entry())
        # After __aexit__ everything is flushed
        assert log.entries_written == 2

    async def test_filepath_property(self, tmp_path: Path) -> None:
        filepath = tmp_path / "out.jsonl"
        async with AsyncJSONLLogger(filepath, buffer_size=1) as log:
            assert log.filepath == filepath


# ---------------------------------------------------------------------------
# Buffering
# ---------------------------------------------------------------------------


class TestAsyncJSONLLoggerBuffering:
    async def test_buffer_size_one_writes_immediately(self, tmp_path: Path) -> None:
        filepath = tmp_path / "out.jsonl"
        log = AsyncJSONLLogger(filepath, buffer_size=1)
        await log.start()
        await log.log(_minimal_entry())
        # Give the consumer a moment to flush
        await asyncio.sleep(0.1)
        lines = filepath.read_text().splitlines() if filepath.exists() else []
        assert len(lines) == 1
        await log.stop()

    async def test_log_batch_writes_all(self, tmp_path: Path) -> None:
        filepath = tmp_path / "out.jsonl"
        entries = [_minimal_entry() for _ in range(5)]
        async with AsyncJSONLLogger(filepath, buffer_size=10) as log:
            await log.log_batch(entries)

        lines = filepath.read_text().splitlines()
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# LogEntry validation at write time
# ---------------------------------------------------------------------------


class TestAsyncJSONLLoggerValidation:
    async def test_pydantic_entry_written_correctly(self, tmp_path: Path) -> None:
        filepath = tmp_path / "out.jsonl"
        entry = _minimal_entry(mutated_prompt="unique-prompt-abc123")
        async with AsyncJSONLLogger(filepath, buffer_size=1) as log:
            await log.log(entry)

        line = filepath.read_text().strip()
        parsed = json.loads(line)
        assert parsed["mutated_prompt"] == "unique-prompt-abc123"

    async def test_dict_entry_validated_through_logentry(self, tmp_path: Path) -> None:
        """A plain dict matching LogEntry schema should be accepted and written."""
        filepath = tmp_path / "out.jsonl"
        entry_dict = {
            "experiment_batch_id": "batch-dict",
            "root_attack_id": "root-dict",
            "mutated_prompt": "dict prompt",
            "target_model_full_name": "test-model",
            "response_text": "resp",
            "success_boolean": False,
            "timing_mutator_ms": 1.0,
            "timing_target_model_ms": 2.0,
            "timing_judge_ms": 3.0,
            "timing_turn_total_ms": 6.0,
            "timing_session_cumulative_ms": 6.0,
        }
        async with AsyncJSONLLogger(filepath, buffer_size=1) as log:
            await log.log(entry_dict)

        line = filepath.read_text().strip()
        parsed = json.loads(line)
        assert parsed["mutated_prompt"] == "dict prompt"


# ---------------------------------------------------------------------------
# Serialise helper (unit tests)
# ---------------------------------------------------------------------------


class TestSerialise:
    def test_serialise_pydantic(self) -> None:
        entry = _minimal_entry(mutated_prompt="pydantic test")
        line = AsyncJSONLLogger._serialise(entry)
        parsed = json.loads(line)
        assert parsed["mutated_prompt"] == "pydantic test"

    def test_serialise_dict(self) -> None:
        d = {
            "experiment_batch_id": "b",
            "root_attack_id": "r",
            "mutated_prompt": "dict test",
            "target_model_full_name": "m",
            "response_text": "resp",
            "success_boolean": True,
            "timing_mutator_ms": 1.0,
            "timing_target_model_ms": 2.0,
            "timing_judge_ms": 3.0,
            "timing_turn_total_ms": 6.0,
            "timing_session_cumulative_ms": 6.0,
        }
        line = AsyncJSONLLogger._serialise(d)
        assert "\n" not in line
        assert json.loads(line)["mutated_prompt"] == "dict test"

    def test_serialise_bad_dict_falls_back_to_raw(self) -> None:
        """A dict that fails LogEntry validation should still be serialised."""
        d = {"only_key": "no required fields here at all"}
        line = AsyncJSONLLogger._serialise(d)
        assert "\n" not in line
        parsed = json.loads(line)
        assert parsed["only_key"] == "no required fields here at all"
