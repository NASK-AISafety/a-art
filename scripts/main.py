#!/usr/bin/env python3
"""
A-ART Agentic Attack Generation Runner

Executes the agentic attack pipeline with:
- System 2 Planner for strategic decisions
- System 1 Mutator for attack generation
- Llama-Prompt-Guard-2 input filter
- Shared model for all components
- Structured outputs via instructor
- LogEntry schema for JSONL output
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_red_team.pipelines.agentic_attack_pipeline import (
    AgenticAttackPipeline,
    PipelineConfig,
    load_config_from_yaml,
    load_seed_prompts,
)


def setup_logging(log_level: str = "INFO", debug: bool = False) -> None:
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    if debug:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _detect_vlm_in_config(config: PipelineConfig) -> bool:
    """Return True if the target component is explicitly or heuristically a VLM."""
    if config.target is not None:
        if config.target.is_vlm is True:
            return True
        vlm_indicators = ["-vl-", "-vl", "vl-", "vision", "multimodal"]
        return any(ind in config.target.model.lower() for ind in vlm_indicators)
    return False


def main():
    """Main entry point for the agentic attack generation runner."""
    parser = argparse.ArgumentParser(
        description="A-ART Agentic Attack Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run (sync, single model)
  uv run python scripts/main.py --config configs/attack_generation/debug.yaml

  # Multi-model vLLM run (requires vLLM server)
  uv run python scripts/main.py --config configs/attack_generation/multi_model_vllm.yaml \\
    --num-attacks 100

  # Limit dataset rows for quick testing
  uv run python scripts/main.py --config configs/attack_generation/debug.yaml \\
    --max_rows 10

  # Run without input guard (faster local testing)
  uv run python scripts/main.py --config configs/attack_generation/debug.yaml \\
    --no-input-guard
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--num-attacks",
        type=int,
        default=None,
        help="Override number of attacks to generate",
    )
    parser.add_argument(
        "--no-input-guard",
        action="store_true",
        help="Disable input guard (faster but less realistic)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values (e.g., --override num_iterations=10 --override shared_model_name=model/name)",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Limit number of seed prompts to load from dataset",
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help=(
            "Use async pipeline (asyncio.gather over seeds) for concurrent seed processing. "
            "RECOMMENDED for all production runs — delivers 9× speedup over sync mode. "
            "REQUIRED for VLM targets (sync VLM is highly inefficient)."
        ),
    )

    args = parser.parse_args()

    # Load raw YAML config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Apply CLI overrides to raw config
    for override in args.override:
        if "=" not in override:
            print(f"Error: Invalid override format '{override}'. Use KEY=VALUE")
            sys.exit(1)

        key, value = override.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Auto-detect value type
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        elif value.isdigit():
            value = int(value)
        elif value.replace(".", "", 1).isdigit():
            value = float(value)

        raw_config[key] = value
        print(f"Override: {key} = {value}")

    # Setup logging
    log_level = raw_config.get("log_level", "INFO")
    debug = raw_config.get("debug", False) or args.verbose
    setup_logging(log_level, debug)

    logger = logging.getLogger(__name__)

    print("\n" + "=" * 70)
    print("A-ART AGENTIC ATTACK GENERATION")
    print("=" * 70)

    # Load pipeline config
    try:
        config = load_config_from_yaml(args.config)

        # Apply CLI overrides
        if args.num_attacks is not None:
            num_attacks = args.num_attacks
        else:
            num_attacks = raw_config.get("num_attacks", 3)

        if args.no_input_guard:
            config.input_guard_enabled = False

        logger.info(f"Config: {args.config}")

        # Log model configuration
        logger.info("Separate models per component:")
        if config.planner:
            logger.info(f"  Planner: {config.planner.model}")
        if config.mutator:
            logger.info(f"  Mutator: {config.mutator.model}")
        if config.target:
            logger.info(f"  Target: {config.target.model}")
        if config.judge:
            logger.info(f"  Judge: {config.judge.model}")

        input_guard_label = (
            config.input_guard.model
            if (config.input_guard_enabled and config.input_guard is not None)
            else "DISABLED"
        )
        output_guard_label = (
            config.output_guard.model
            if (config.output_guard_enabled and config.output_guard is not None)
            else "DISABLED"
        )
        logger.info(f"Input Guard: {input_guard_label}")
        logger.info(f"Output Guard: {output_guard_label}")
        logger.info(f"Attacks to generate: {num_attacks}")

        # Warn if a VLM target is configured but async mode is not enabled.
        # Sync VLM inference processes seeds sequentially, wasting most of the
        # available GPU concurrency and running ~9× slower than async.
        if not args.use_async and _detect_vlm_in_config(config):
            print(
                "\nWARNING: VLM target detected but --async flag is NOT set.\n"
                "  Synchronous VLM inference processes seeds one at a time (no concurrency).\n"
                "  Add --async for 9× faster throughput via concurrent seed processing.\n"
                "  Example: uv run python scripts/main.py --config ... --async\n",
                file=sys.stderr,
            )

    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Load seed prompts
    try:
        data_config = raw_config.get("data", {})
        input_path = data_config.get("input_file", "datasets/example_seeds.jsonl")

        # Use CLI --max_rows if provided, else config value, else num_attacks
        if args.max_rows is not None:
            max_rows = args.max_rows
        else:
            max_rows = data_config.get("max_rows", num_attacks)

        # For smart sampling, max_seeds should only limit if explicitly configured
        # (not the default fallback to num_attacks)
        explicit_max_rows = args.max_rows or data_config.get("max_rows")

        # For smart sampling, we skip CSV loading (sampler loads JSONL internally)
        if config.smart_sampling_enabled:
            seed_prompts = [{"prompt": "_smart_sampler_placeholder", "risk_category": "n/a"}]
            logger.info("Smart sampling enabled — seeds loaded from JSONL by sampler")
        else:
            language_filter = data_config.get("language_filter", "EN")
            seed_prompts = load_seed_prompts(
                input_path, max_rows, language_filter=language_filter
            )
            logger.info(f"Loaded {len(seed_prompts)} seed prompts from {input_path}")

    except Exception as e:
        logger.error(f"Failed to load seed prompts: {e}")
        sys.exit(1)

    # Run pipeline
    pipeline = AgenticAttackPipeline(config)

    # Attach smart sampler if configured
    smart_sampler = None
    if config.smart_sampling_enabled:
        from llm_red_team.data.smart_sampler import SmartSampler

        target_model = ""
        if config.target is not None:
            target_model = config.target.model

        if not input_path.endswith(".jsonl"):
            logger.error(
                "Smart sampling requires a JSONL input file with transferability data. "
                f"Got: {input_path}"
            )
            sys.exit(1)

        smart_sampler = SmartSampler(
            seeds_path=input_path,
            target_model=target_model,
            batch_size=config.smart_sampling_batch_size,
            mode=config.smart_sampling_mode,
            rng_seed=config.seed,
            max_seeds=explicit_max_rows,
            recycle=config.smart_sampling_recycle,
        )
        pipeline.set_smart_sampler(smart_sampler)
        logger.info(
            f"Smart sampling enabled: mode={config.smart_sampling_mode}, "
            f"batch_size={config.smart_sampling_batch_size}, "
            f"pool_size={smart_sampler.pool_size}"
        )

    try:
        print("\n" + "-" * 70)
        print("GENERATING ATTACKS...")
        print("-" * 70 + "\n")

        if args.use_async:
            import asyncio

            entries = asyncio.run(pipeline.arun(seed_prompts, num_attacks=num_attacks))
        else:
            entries = pipeline.run(seed_prompts, num_attacks=num_attacks)

        # Print summary
        print("\n" + "-" * 70)
        print("RESULTS SUMMARY")
        print("-" * 70)

        successes = sum(1 for e in entries if e.get("success_boolean"))
        filtered = sum(1 for e in entries if e.get("judge_verdict_code") == "FILTERED_INPUT")

        print(f"\nTotal attacks: {len(entries)}")
        print(f"Successful jailbreaks: {successes}")
        print(f"Filtered by input guard: {filtered}")
        print(f"Attack Success Rate: {100 * successes / len(entries):.1f}%" if entries else "N/A")

        print("\n--- Attack Details ---")
        for i, entry in enumerate(entries):
            print(f"\n[Attack {i + 1}]")
            print(f"  Risk Category: {entry.get('risk_category_tag')}")
            print(f"  Attack Style: {entry.get('attack_style_tag')}")
            print(f"  Verdict: {entry.get('judge_verdict_code')}")
            print(f"  Severity: {entry.get('risk_severity_score', 0):.2f}")
            print(f"  Planner Strategy: {entry.get('planner_strategy')}")
            print(f"  Planner Reasoning: {entry.get('planner_reasoning', '')[:60]}...")
            print(f"  Prompt: {entry.get('mutated_prompt', '')[:80]}...")
            print(f"  Response: {entry.get('response_text', '')[:80]}...")

        print(f"\nOutputs saved to: {config.output_dir}/")

        print("\n" + "=" * 70)
        print("ATTACK GENERATION COMPLETE")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        logger.info("\nAttack generation interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        if debug:
            logger.exception("Full traceback:")
        sys.exit(1)

    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
