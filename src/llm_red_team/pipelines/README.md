# MVP Attack Generation Pipeline

This directory contains the framework for attack generation alongside the existing batch evaluation features.

## Architecture

It implements the agentic loop pattern: `Database -> Mutator -> Target -> Judge`

### Components

- **TargetComponent**: Wraps existing `models/huggingface.py` for target model inference
- **MutatorComponent**: Uses a HF model to mutate/reformulate prompts 
- **JudgeComponent**: Wraps existing `evaluators/llama_guard.py` for safety evaluation
- **AttackPipeline**: Orchestrates the loop and handles logging with the database data schema

## Quick Start

### Prerequisites

- HuggingFace token set as `HF_TOKEN` environment variable
- CUDA GPU recommended (CPU fallback available)
- Required packages: transformers, pydantic, torch

### Running Attack Generation

```bash
# Basic usage (first 10 prompts)
python scripts/run_attack_generation.py --max-rows 10

# With specific models
python scripts/run_attack_generation.py \
  --mutator-model "meta-llama/Llama-3.1-8B-Instruct" \
  --target-model "meta-llama/Llama-3.1-8B-Instruct" \
  --max-rows 50

# CPU testing
python scripts/run_attack_generation.py --device cpu --max-rows 5
```

### Command Line Options

- `--input-file`: seed prompts file, `.jsonl`/`.parquet`/`.csv` (default: `datasets/example_seeds.jsonl`)
- `--mutator-model`: Model for prompt mutation
- `--target-model`: Target model to attack  
- `--judge-model`: Safety evaluator model
- `--device`: cuda/cpu (auto-detected)
- `--max-rows`: Limit number of prompts processed
- `--output-dir`: Directory for logs (default: `outputs/runs`)
- `--attack-styles`: List of attack styles to try

## Output Format

The pipeline generates:

1. **JSONL Log File**: Structured logs using A-ART schema in `outputs/runs/{timestamp}/log.jsonl`
2. **Statistics Summary**: Attack Success Rate (ASR) and processing stats

### Log Schema

Each log entry contains:
- Attack metadata (IDs, timing, configuration)
- Payload (mutated prompt, attack style, risk category)
- Target response and safety evaluation
- Performance metrics and operational metadata

## Attack Styles

The MVP supports several mutation strategies:
- `rewrite`: General prompt rewriting
- `slang`: Using informal language and slang
- `role_play`: Framing as fictional scenarios
- `technical_terms`: Using scientific jargon
- `misspellings`: Intentional typos and alternative spellings

## File Structure

```
src/llm_red_team/
├── schemas/                 # A-ART data schema
│   └── llm_red_team_schema.py
├── components/              # Pipeline components
│   ├── target.py           # Target model wrapper
│   ├── mutator.py          # Prompt mutator
│   └── judge.py            # Safety evaluator
└── pipelines/               # Pipeline orchestration
    └── attack_pipeline.py  # Main attack generation loop

scripts/
└── run_attack_generation.py  # CLI runner script
```

## Integration with Existing Framework

The MVP is designed to coexist with the existing batch evaluation system:

- **No Breaking Changes**: Existing `run_experiment.py` continues to work
- **Shared Components**: Reuses `models/` and `evaluators/` infrastructure
- **Unified Logging**: Both systems can output to the same analysis pipeline

## Next Steps

Future enhancements may include:
- Integration with LiteLLM for multiple providers
- Structured output using Instructor
- Logit masking with Outlines for local models
- Multi-turn conversation support
- Advanced planning strategies

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use smaller models or CPU mode
2. **HuggingFace Token Errors**: Set `HF_TOKEN` environment variable
3. **Model Loading Failures**: Check model names and internet connectivity

### Debug Mode

Run with small dataset and CPU mode for debugging:
```bash
python scripts/run_attack_generation.py --device cpu --max-rows 1
```
