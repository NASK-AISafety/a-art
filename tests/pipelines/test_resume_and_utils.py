"""
Tests for pipeline utilities and JSON parsing.

Covers:
    1. extract_json_object: clean JSON
    2. extract_json_object: markdown code block
    3. extract_json_object: extra text after JSON
    4. extract_json_object: no JSON → None
    5. SeedLoader CSV round-trip
    6. SeedLoader JSONL round-trip
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from llm_red_team.data.seed_loader import SeedLoader
from llm_red_team.utils.json_parsing import extract_json_object

# ---------------------------------------------------------------------------
# extract_json_object
# ---------------------------------------------------------------------------


class TestExtractJsonObject:
    def test_clean_json(self) -> None:
        result = extract_json_object('{"verdict": "REFUSAL", "score": 0.0}')
        assert result == {"verdict": "REFUSAL", "score": 0.0}

    def test_json_in_markdown_code_block(self) -> None:
        text = '```json\n{"action": "CONTINUE", "strategy": "role_play"}\n```'
        result = extract_json_object(text)
        assert result is not None
        assert result["action"] == "CONTINUE"

    def test_json_with_preamble_text(self) -> None:
        text = 'Here is my analysis:\n\n{"verdict": "FULL", "severity_score": 0.9}'
        result = extract_json_object(text)
        assert result is not None
        assert result["verdict"] == "FULL"

    def test_json_with_trailing_text(self) -> None:
        text = '{"action": "PIVOT"} additional text the model emitted'
        result = extract_json_object(text)
        assert result is not None
        assert result["action"] == "PIVOT"

    def test_no_json_returns_none(self) -> None:
        result = extract_json_object("This is plain text with no JSON at all.")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = extract_json_object("")
        assert result is None

    def test_nested_json(self) -> None:
        text = '{"outer": {"inner": "value"}, "count": 3}'
        result = extract_json_object(text)
        assert result is not None
        assert result["outer"]["inner"] == "value"
        assert result["count"] == 3


# ---------------------------------------------------------------------------
# SeedLoader
# ---------------------------------------------------------------------------


class TestSeedLoaderCSV:
    def test_load_csv_basic(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "seeds.csv"
        df = pd.DataFrame(
            {
                "prompt": ["prompt A", "prompt B"],
                "risk_category": ["criminal_planning", "violence_and_hate"],
                "attack_style": ["role_play", None],
            }
        )
        df.to_csv(csv_path, index=False)

        loader = SeedLoader()
        seeds = loader.load(csv_path)

        assert len(seeds) == 2
        assert seeds[0].prompt == "prompt A"
        assert seeds[0].risk_category == "criminal_planning"
        assert seeds[0].attack_style == "role_play"
        assert seeds[1].attack_style is None

    def test_load_csv_max_rows(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "seeds.csv"
        df = pd.DataFrame(
            {
                "prompt": [f"prompt {i}" for i in range(20)],
                "risk_category": ["criminal_planning"] * 20,
            }
        )
        df.to_csv(csv_path, index=False)

        loader = SeedLoader()
        seeds = loader.load(csv_path, max_rows=5)
        assert len(seeds) == 5

    def test_load_missing_file_raises(self) -> None:
        loader = SeedLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/seeds.csv")

    def test_load_unsupported_format_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "seeds.parquet"
        p.write_text("dummy")
        loader = SeedLoader()
        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load(p)


class TestSeedLoaderJSONL:
    def test_load_jsonl_basic(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "attacks.jsonl"
        entries = [
            {
                "mutated_prompt": "attack 1",
                "risk_category_tag": "criminal_planning",
                "attack_style_tag": "role_play",
                "attack_id": "id1",
            },
            {
                "mutated_prompt": "attack 2",
                "risk_category_tag": "violence_and_hate",
                "attack_id": "id2",
            },
        ]
        jsonl_path.write_text("\n".join(json.dumps(e) for e in entries))

        loader = SeedLoader()
        seeds = loader.load(jsonl_path)

        assert len(seeds) == 2
        assert seeds[0].prompt == "attack 1"
        assert seeds[0].risk_category == "criminal_planning"
        assert seeds[0].source_attack_id == "id1"

    def test_load_jsonl_max_rows(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "attacks.jsonl"
        entries = [{"mutated_prompt": f"p{i}", "risk_category_tag": "c"} for i in range(10)]
        jsonl_path.write_text("\n".join(json.dumps(e) for e in entries))

        loader = SeedLoader()
        seeds = loader.load(jsonl_path, max_rows=3)
        assert len(seeds) == 3

    def test_load_jsonl_skips_malformed_lines(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "attacks.jsonl"
        jsonl_path.write_text(
            json.dumps({"mutated_prompt": "good", "risk_category_tag": "c"})
            + "\n"
            + "not json at all\n"
            + json.dumps({"mutated_prompt": "also good", "risk_category_tag": "c"})
            + "\n"
        )

        loader = SeedLoader()
        seeds = loader.load(jsonl_path)
        assert len(seeds) == 2

    def test_iter_load_csv_yields_batches(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "seeds.csv"
        df = pd.DataFrame(
            {
                "prompt": [f"p{i}" for i in range(15)],
                "risk_category": ["c"] * 15,
            }
        )
        df.to_csv(csv_path, index=False)

        loader = SeedLoader()
        batches = list(loader.iter_load(csv_path, batch_size=5))
        assert len(batches) == 3
        assert all(len(b) == 5 for b in batches)
