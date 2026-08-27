#!/usr/bin/env python3
"""Fetch a larger set of seed prompts from the published A-ART dataset.

The A-ART framework reuses existing adversarial attacks as seeds (retrieving and
mutating known-working prompts). A small example set already ships in
`datasets/example_seeds.jsonl`; use this script to pull a larger sample from the
gated Hugging Face dataset, written as JSONL (`prompt`, `risk_category`,
`attack_style`) that the pipeline loads directly.

Access is gated: request access to the dataset and authenticate first
(`huggingface-cli login`, or set the HF_TOKEN environment variable).

Usage:
    python scripts/fetch_example_seeds.py --n 500 --out datasets/seeds.jsonl
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset

DATASET = "NASK-PIB/a-art-jailbreaks"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="number of seed prompts to fetch")
    ap.add_argument("--out", default="datasets/seeds.jsonl", help="output JSONL path")
    ap.add_argument("--category", default=None, help="optional risk_category_tag filter")
    args = ap.parse_args()

    ds = load_dataset(DATASET, "attacks", split="train", streaming=True)
    rows = []
    for row in ds:
        if args.category and row.get("risk_category_tag") != args.category:
            continue
        rows.append(
            {
                "prompt": row["prompt"],
                "risk_category": row.get("risk_category_tag", ""),
                "attack_style": row.get("attack_style_tag", ""),
            }
        )
        if len(rows) >= args.n:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} seed prompts to {out}")


if __name__ == "__main__":
    main()
