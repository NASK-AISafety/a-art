"""
Robust JSON extraction utilities for parsing LLM outputs.

Reasoning models (DeepSeek-R1, Ministral-Reasoning, etc.) often emit
chain-of-thought text before/after the JSON object, causing standard
json.loads() to fail with "Extra data" errors. These utilities handle:

- Markdown code block wrappers (```json ... ```)
- Multiple JSON objects in output (takes the first valid one)
- Extra text before/after JSON
- Nested braces inside JSON strings
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


def extract_json_object(text: str) -> dict | None:
    """
    Extract the first valid JSON object from LLM output text.

    Handles common issues with reasoning model outputs:
    1. Markdown code blocks: ```json { ... } ```
    2. Extra data after JSON (common with DeepSeek-R1)
    3. Text/reasoning before JSON
    4. Multiple JSON objects (returns first valid one)

    Args:
        text: Raw LLM output that may contain a JSON object

    Returns:
        Parsed dict if a valid JSON object was found, None otherwise
    """
    if not text or not text.strip():
        return None

    # Step 1: Try to extract from markdown code blocks first.
    # Greedy (\{.*\}) so the capture spans nested objects up to the LAST closing
    # brace inside the fence; a non-greedy match stops at the first '}' and drops
    # any JSON that contains a nested object.
    code_block_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Step 2: Try raw_decode which handles extra data after JSON
    text_stripped = text.strip()
    for i, char in enumerate(text_stripped):
        if char == "{":
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(text_stripped, i)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

    # Step 3: Fallback - try find/rfind with progressively smaller windows
    json_start = text.find("{")
    if json_start == -1:
        return None

    # Try each closing brace from innermost to outermost
    search_from = json_start
    while True:
        json_end = text.find("}", search_from)
        if json_end == -1:
            break
        try:
            candidate = text[json_start : json_end + 1]
            return json.loads(candidate)
        except json.JSONDecodeError:
            search_from = json_end + 1

    return None
