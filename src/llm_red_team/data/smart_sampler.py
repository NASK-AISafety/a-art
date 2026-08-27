"""Smart Sampler: Adaptive Bayesian seed selection for attack generation.

Implements the adaptive baseline approach from the smart retrieval research,
adapted for online use within the attack pipeline. Instead of random sampling,
seeds are selected in batches using posterior-updated scores based on
which seeds lead to successful jailbreaks on the target model.

Protocol:
  1. Load seeds from a JSONL file that contains per-model transferability data.
  2. Compute priors from cross-model success rates (excluding target model).
  3. Select first batch by deterministic top-prior seeds.
  4. After each batch completes, receive feedback (success/failure per seed).
  5. Update posteriors and select the next batch with updated scores.

The JSONL input must contain:
  - attack_id: unique identifier
  - mutated_prompt or attack_text: the attack text (used as seed prompt)
  - risk_category_tag: risk category
  - attack_style_tag: attack style
  - planner_action: planner action tag (optional)
  - source_target_model: model this attack was originally crafted for
  - harmful_success_rate: fraction of evaluated models this attack jailbreaks
  - harmful_success_count: number of models jailbroken
  - evaluated_model_count: number of models evaluated on
  - harmful_success__<model_name>: per-model binary success (True/False or 1/0)

Optional fields (for enhanced consensus_weighted mode):
  - strict_success_rate, strict_success_count
  - judge_label__<model_name>: per-model judge labels (SAFE/PARTIAL/UNSAFE)
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Categorical features used for posterior grouping
BASE_FEATURES = [
    "source_target_model",
    "attack_style_tag",
    "risk_category_tag",
    "planner_action",
]


def _posterior_mean(
    successes: int, total: int, prior_mean: float, pseudo_count: float = 10.0
) -> float:
    """Bayesian posterior mean with conjugate Beta prior."""
    return (successes + pseudo_count * prior_mean) / (total + pseudo_count)


@dataclass
class SeedRecord:
    """A single seed with its metadata and transferability features."""

    attack_id: str
    prompt: str
    risk_category: str
    attack_style: str | None
    planner_action: str | None
    source_target_model: str | None
    # Aggregate transferability
    harmful_success_rate: float
    harmful_success_count: int
    evaluated_model_count: int
    # Per-model success (model_name -> bool)
    model_successes: dict[str, bool] = field(default_factory=dict)
    # Computed
    prior_mean: float = 0.0


class SmartSampler:
    """Adaptive Bayesian seed sampler for attack generation.

    Selects seeds in batches, updating posteriors after each batch
    based on whether the attack succeeded on the target model.

    Args:
        seeds_path: Path to JSONL file with enriched seed data.
        target_model: Name of the target model (excluded from prior computation).
        batch_size: Number of seeds per sampling batch.
        mode: Sampling mode ('adaptive_baseline' or 'random').
        rng_seed: Random seed for reproducibility.
        max_seeds: Maximum seeds to load from file (None = all).
    """

    def __init__(
        self,
        seeds_path: str | Path,
        target_model: str,
        batch_size: int = 32,
        mode: str = "adaptive_baseline",
        rng_seed: int | None = None,
        max_seeds: int | None = None,
        recycle: bool = False,
    ):
        self.seeds_path = Path(seeds_path)
        self.target_model = target_model
        self.batch_size = batch_size
        self.mode = mode
        self.rng = random.Random(rng_seed)
        self.max_seeds = max_seeds
        self.recycle = recycle

        # State
        self._pool: list[SeedRecord] = []
        self._used_ids: set[str] = set()
        self._model_cols: list[str] = []
        self._global_prior: float = 0.0
        self._feature_priors: dict[str, dict] = {}
        self._feature_stats: dict[str, dict] = {}
        self._model_priors: dict[tuple[str, int], float] = {}
        self._model_stats: dict[tuple[str, int], list[int]] = {}
        self._batches_completed: int = 0
        self._total_selected: int = 0
        self._total_successes: int = 0

        self._load_seeds()
        self._compute_priors()

    def _load_seeds(self) -> None:
        """Load and parse seed records from JSONL."""
        path = self.seeds_path
        if not path.exists():
            raise FileNotFoundError(f"Smart sampler seeds not found: {path}")

        records: list[SeedRecord] = []
        with open(path) as f:
            for i, line in enumerate(f):
                if self.max_seeds is not None and i >= self.max_seeds:
                    break
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                prompt = entry.get("mutated_prompt") or entry.get("attack_text", "")
                if not prompt:
                    continue

                # Extract per-model successes (exclude target model)
                model_successes: dict[str, bool] = {}
                for key, val in entry.items():
                    if key.startswith("harmful_success__"):
                        model_name = key[len("harmful_success__"):]
                        if model_name != self.target_model:
                            model_successes[model_name] = bool(val)

                attack_style = entry.get("attack_style_tag")

                records.append(
                    SeedRecord(
                        attack_id=entry.get("attack_id", f"seed_{i}"),
                        prompt=prompt,
                        risk_category=entry.get("risk_category_tag", "unknown"),
                        attack_style=attack_style if attack_style else None,
                        planner_action=entry.get("planner_action"),
                        source_target_model=entry.get("source_target_model"),
                        harmful_success_rate=float(
                            entry.get("harmful_success_rate", 0.0)
                        ),
                        harmful_success_count=int(
                            entry.get("harmful_success_count", 0)
                        ),
                        evaluated_model_count=int(
                            entry.get("evaluated_model_count", 0)
                        ),
                        model_successes=model_successes,
                    )
                )

        self._pool = records
        self._all_records = list(records)  # Keep original for recycling

        # Discover model columns (models present in the data, excluding target)
        if records:
            self._model_cols = sorted(records[0].model_successes.keys())

        logger.info(
            f"SmartSampler loaded {len(records)} seeds from {path.name}, "
            f"target={self.target_model}, "
            f"reference_models={len(self._model_cols)}, "
            f"mode={self.mode}"
        )

    def _compute_priors(self) -> None:
        """Compute initial priors from cross-model transferability data."""
        if not self._pool:
            return

        # Compute prior_mean per seed (Beta-smoothed success rate on non-target models)
        for seed in self._pool:
            n_models = len(seed.model_successes)
            if n_models > 0:
                successes = sum(1 for v in seed.model_successes.values() if v)
                # Laplace-smoothed: (successes + 1) / (models + 2)
                seed.prior_mean = (successes + 1.0) / (n_models + 2.0)
            else:
                # Fallback to reported rate
                seed.prior_mean = seed.harmful_success_rate

        self._global_prior = sum(s.prior_mean for s in self._pool) / len(self._pool)

        # Feature priors: mean prior_mean per feature value
        self._feature_priors = {}
        for feat in BASE_FEATURES:
            groups: dict[str, list[float]] = {}
            for seed in self._pool:
                val = getattr(seed, feat.replace("_tag", "").replace("risk_category", "risk_category"), None)
                # Map feature name to seed attribute
                if feat == "source_target_model":
                    val = seed.source_target_model
                elif feat == "attack_style_tag":
                    val = seed.attack_style
                elif feat == "risk_category_tag":
                    val = seed.risk_category
                elif feat == "planner_action":
                    val = seed.planner_action
                val_str = str(val) if val is not None else "None"
                groups.setdefault(val_str, []).append(seed.prior_mean)
            self._feature_priors[feat] = {
                k: sum(v) / len(v) for k, v in groups.items()
            }

        # Model priors: mean prior_mean for seeds where model success = 0 vs 1
        for col in self._model_cols:
            for binary_val in (0, 1):
                matching = [
                    s.prior_mean
                    for s in self._pool
                    if int(s.model_successes.get(col, False)) == binary_val
                ]
                self._model_priors[(col, binary_val)] = (
                    sum(matching) / len(matching) if matching else self._global_prior
                )

        # Initialize stats trackers
        self._feature_stats = {f: {} for f in BASE_FEATURES}
        self._model_stats = {
            (c, v): [0, 0] for c in self._model_cols for v in (0, 1)
        }

    def _get_feature_value(self, seed: SeedRecord, feat: str) -> str:
        """Extract a feature value from a seed record."""
        if feat == "source_target_model":
            return str(seed.source_target_model)
        elif feat == "attack_style_tag":
            return str(seed.attack_style)
        elif feat == "risk_category_tag":
            return str(seed.risk_category)
        elif feat == "planner_action":
            return str(seed.planner_action)
        return "None"

    @property
    def pool_size(self) -> int:
        """Number of seeds remaining in the pool."""
        return len(self._pool)

    def _recycle_pool(self) -> None:
        """Reset pool to all original records for another pass.

        Preserves learned posteriors so subsequent passes benefit from
        accumulated feedback.
        """
        self._pool = list(self._all_records)
        self._used_ids.clear()
        logger.info(
            f"SmartSampler recycled pool: {len(self._pool)} seeds available again "
            f"(posteriors preserved, cycle {self._batches_completed // max(1, len(self._all_records) // self.batch_size) + 1})"
        )

    @property
    def stats(self) -> dict:
        """Return current sampling statistics."""
        return {
            "batches_completed": self._batches_completed,
            "total_selected": self._total_selected,
            "total_successes": self._total_successes,
            "pool_remaining": len(self._pool),
            "success_rate": (
                self._total_successes / self._total_selected
                if self._total_selected > 0
                else 0.0
            ),
        }

    def select_batch(self, n: int | None = None) -> list[dict[str, str]]:
        """Select the next batch of seeds for attack generation.

        Returns a list of dicts compatible with the pipeline's seed_prompts format:
        [{"prompt": ..., "risk_category": ..., "attack_style": ..., "_attack_id": ...}]

        Args:
            n: Override batch size for this selection. Defaults to self.batch_size.

        Returns:
            List of seed prompt dicts ready for the pipeline.
        """
        batch_size = n if n is not None else self.batch_size
        batch_size = min(batch_size, len(self._pool))

        if batch_size == 0:
            if self.recycle:
                self._recycle_pool()
                batch_size = min(
                    n if n is not None else self.batch_size,
                    len(self._pool),
                )
                if batch_size == 0:
                    return []
            else:
                logger.warning("SmartSampler pool exhausted, no seeds available")
                return []

        if self.mode == "random":
            selected = self._select_random(batch_size)
        elif self.mode == "adaptive_baseline":
            selected = self._select_adaptive(batch_size)
        else:
            raise ValueError(f"Unknown smart sampling mode: {self.mode}")

        # Remove selected seeds from pool
        selected_ids = {s.attack_id for s in selected}
        self._pool = [s for s in self._pool if s.attack_id not in selected_ids]
        self._used_ids.update(selected_ids)
        self._total_selected += len(selected)

        # Convert to pipeline format
        result = []
        for seed in selected:
            result.append(
                {
                    "prompt": seed.prompt,
                    "risk_category": seed.risk_category,
                    "attack_style": seed.attack_style,
                    "_attack_id": seed.attack_id,
                    "_source_target_model": seed.source_target_model,
                    "_prior_mean": seed.prior_mean,
                    "_model_successes": dict(seed.model_successes),
                }
            )

        logger.info(
            f"SmartSampler batch {self._batches_completed + 1}: "
            f"selected {len(selected)} seeds "
            f"(pool remaining: {len(self._pool)}, "
            f"mode={self.mode})"
        )
        return result

    def update(self, feedback: list[dict]) -> None:
        """Update posteriors with batch feedback.

        Args:
            feedback: List of dicts with at minimum:
                - '_attack_id': seed attack_id used
                - 'success_boolean': whether the attack succeeded
                Optional:
                - 'risk_category_tag', 'attack_style_tag' (for feature updates)
        """
        if self.mode == "random":
            # No learning for random mode
            self._batches_completed += 1
            return

        successes_this_batch = 0
        for entry in feedback:
            attack_id = entry.get("_attack_id") or entry.get("attack_id", "")
            success = bool(entry.get("success_boolean", False))
            label = int(success)
            if success:
                successes_this_batch += 1

            # Find the original seed record for feature values
            # (we stored them in _used_ids context, but let's use entry metadata)
            risk_cat = str(entry.get("risk_category_tag", entry.get("risk_category", "None")))
            attack_style = str(entry.get("attack_style_tag", entry.get("attack_style", "None")))
            planner_action = str(entry.get("planner_action", "None"))
            source_model = str(entry.get("_source_target_model", "None"))

            # Update feature stats
            feat_vals = {
                "source_target_model": source_model,
                "attack_style_tag": attack_style,
                "risk_category_tag": risk_cat,
                "planner_action": planner_action,
            }
            for feat, val in feat_vals.items():
                stats = self._feature_stats[feat].setdefault(val, [0, 0])
                stats[0] += label
                stats[1] += 1

            # Update model stats (using the model_successes from the seed)
            # We need to look up the seed's model data - stored via _attack_id
            model_data = entry.get("_model_successes")
            if model_data:
                for col in self._model_cols:
                    k = (col, int(model_data.get(col, False)))
                    self._model_stats[k][0] += label
                    self._model_stats[k][1] += 1

        self._total_successes += successes_this_batch
        self._batches_completed += 1

        logger.info(
            f"SmartSampler update batch {self._batches_completed}: "
            f"{successes_this_batch}/{len(feedback)} succeeded "
            f"(cumulative: {self._total_successes}/{self._total_selected} = "
            f"{100*self._total_successes/max(1,self._total_selected):.1f}% ASR)"
        )

    def _select_random(self, n: int) -> list[SeedRecord]:
        """Uniform random selection without replacement."""
        return self.rng.sample(self._pool, min(n, len(self._pool)))

    def _select_adaptive(self, n: int) -> list[SeedRecord]:
        """Adaptive posterior-based selection."""
        if self._batches_completed == 0:
            # First batch: deterministic top by prior_mean
            sorted_pool = sorted(
                self._pool,
                key=lambda s: (-s.prior_mean, s.attack_id),
            )
            return sorted_pool[:n]

        # Subsequent batches: score using updated posteriors
        scores: list[tuple[float, str, SeedRecord]] = []
        for seed in self._pool:
            # Feature score
            feat_score = 0.0
            for feat in BASE_FEATURES:
                val = self._get_feature_value(seed, feat)
                stats = self._feature_stats[feat].get(val, [0, 0])
                prior = self._feature_priors.get(feat, {}).get(
                    val, self._global_prior
                )
                feat_score += _posterior_mean(stats[0], stats[1], prior, 12.0)
            feat_score /= len(BASE_FEATURES)

            # Model score
            model_score = 0.0
            if self._model_cols:
                for col in self._model_cols:
                    binary_val = int(seed.model_successes.get(col, False))
                    stats = self._model_stats[(col, binary_val)]
                    prior = self._model_priors[(col, binary_val)]
                    model_score += _posterior_mean(stats[0], stats[1], prior, 10.0)
                model_score /= len(self._model_cols)

            # Combined score (same weights as adaptive_baseline in smart retrieval)
            combined = (
                0.45 * seed.prior_mean + 0.25 * feat_score + 0.30 * model_score
            )
            scores.append((combined, seed.attack_id, seed))

        # Sort by score descending, break ties by attack_id
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [s[2] for s in scores[:n]]

    def get_seed_model_data(self, attack_id: str) -> dict[str, bool] | None:
        """Retrieve model success data for a seed (for feedback updates).

        This is called by the pipeline to attach model data to feedback entries
        so the update() method can update model_stats.
        """
        # Check pool first
        for seed in self._pool:
            if seed.attack_id == attack_id:
                return dict(seed.model_successes)
        # It might have been already selected (check isn't needed since
        # we remove from pool, but provide graceful fallback)
        return None


def load_smart_seeds_jsonl(
    path: str | Path,
    max_rows: int | None = None,
) -> list[dict]:
    """Load raw JSONL records for use with SmartSampler or inspection.

    This is a utility function; SmartSampler loads internally.
    """
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return records
