"""
A-ART Seed Loader: Unified input abstraction for CSV and JSONL.

Supports loading seed prompts from:
- CSV files (initial dataset)
- JSONL files (re-feeding successful attacks from knowledge base)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SeedPrompt:
    """A single seed prompt for attack generation."""

    prompt: str
    risk_category: str
    attack_style: str | None = None
    source_file: str | None = None
    source_attack_id: str | None = None


class SeedLoader:
    """
    Unified loader for seed prompts from multiple formats.

    Supports:
    - CSV: Traditional seed dataset with columns (prompt, risk_category, attack_style)
    - JSONL: Re-feeding successful attacks from knowledge base

    Example:
        loader = SeedLoader()
        seeds = loader.load("data/seeds.csv", max_rows=100)
        seeds = loader.load("logs/successful_attacks.jsonl", max_rows=50)
    """

    SUPPORTED_FORMATS = {".csv", ".jsonl"}

    def load(
        self,
        file_path: str | Path,
        max_rows: int | None = None,
    ) -> list[SeedPrompt]:
        """
        Load seed prompts from a file.

        Args:
            file_path: Path to CSV or JSONL file
            max_rows: Maximum number of rows to load (None = all)

        Returns:
            List of SeedPrompt objects
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Seed file not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {suffix}. Use {self.SUPPORTED_FORMATS}")

        if suffix == ".csv":
            return self._load_csv(path, max_rows)
        elif suffix == ".jsonl":
            return self._load_jsonl(path, max_rows)

        return []

    def _load_csv(self, path: Path, max_rows: int | None) -> list[SeedPrompt]:
        """Load seeds from CSV file."""
        df = pd.read_csv(path, nrows=max_rows)

        seeds = []
        for _, row in df.iterrows():
            seeds.append(
                SeedPrompt(
                    prompt=str(row.get("prompt", "")),
                    risk_category=str(row.get("risk_category", "criminal_planning")),
                    attack_style=row.get("attack_style")
                    if pd.notna(row.get("attack_style"))
                    else None,
                    source_file=str(path),
                )
            )

        logger.info(f"Loaded {len(seeds)} seeds from CSV: {path}")
        return seeds

    def _load_jsonl(self, path: Path, max_rows: int | None) -> list[SeedPrompt]:
        """Load seeds from JSONL file (successful attacks knowledge base)."""
        seeds = []
        count = 0

        with open(path) as f:
            for line in f:
                if max_rows is not None and count >= max_rows:
                    break

                try:
                    entry = json.loads(line.strip())

                    # Extract from LogEntry schema structure (v5-flat uses risk_category_tag)
                    prompt = entry.get("prompt_text") or entry.get("mutated_prompt", "")
                    risk_category = entry.get("risk_category_tag") or entry.get(
                        "original_risk_category", "criminal_planning"
                    )
                    attack_styles = entry.get("attack_style_tags") or []
                    primary_style = attack_styles[0] if isinstance(attack_styles, list) and attack_styles else None
                    attack_style = (
                        entry.get("attack_style_tag")
                        or primary_style
                        or entry.get("original_attack_style")
                    )
                    attack_id = entry.get("attack_id")

                    if prompt:
                        seeds.append(
                            SeedPrompt(
                                prompt=prompt,
                                risk_category=risk_category,
                                attack_style=attack_style,
                                source_file=str(path),
                                source_attack_id=attack_id,
                            )
                        )
                        count += 1

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSONL line: {e}")
                    continue

        logger.info(f"Loaded {len(seeds)} seeds from JSONL: {path}")
        return seeds

    def iter_load(
        self,
        file_path: str | Path,
        batch_size: int = 100,
    ) -> Iterator[list[SeedPrompt]]:
        """
        Lazily load seeds in batches (memory-efficient for large files).

        Args:
            file_path: Path to seed file
            batch_size: Number of seeds per batch

        Yields:
            Batches of SeedPrompt objects
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            for chunk in pd.read_csv(path, chunksize=batch_size):
                batch = []
                for _, row in chunk.iterrows():
                    batch.append(
                        SeedPrompt(
                            prompt=str(row.get("prompt", "")),
                            risk_category=str(row.get("risk_category", "criminal_planning")),
                            attack_style=row.get("attack_style")
                            if pd.notna(row.get("attack_style"))
                            else None,
                            source_file=str(path),
                        )
                    )
                yield batch

        elif suffix == ".jsonl":
            batch = []
            with open(path) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        prompt = entry.get("prompt_text") or entry.get("mutated_prompt", "")
                        if prompt:
                            attack_styles = entry.get("attack_style_tags") or []
                            primary_style = (
                                attack_styles[0]
                                if isinstance(attack_styles, list) and attack_styles
                                else None
                            )
                            batch.append(
                                SeedPrompt(
                                    prompt=prompt,
                                    risk_category=entry.get("risk_category_tag")
                                    or entry.get("original_risk_category", "criminal_planning"),
                                    attack_style=entry.get("attack_style_tag")
                                    or primary_style
                                    or entry.get("original_attack_style"),
                                    source_file=str(path),
                                    source_attack_id=entry.get("attack_id"),
                                )
                            )

                        if len(batch) >= batch_size:
                            yield batch
                            batch = []

                    except json.JSONDecodeError:
                        continue

            if batch:
                yield batch
