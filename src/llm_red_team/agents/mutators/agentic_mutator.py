"""
A-ART Agentic Mutator: Template-based Attack Generator.

This mutator uses a shared LLM with Jinja2 templates to generate
adversarial prompts based on planner decisions.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader

if TYPE_CHECKING:
    from llm_red_team.models.shared_llm import SharedLLM

logger = logging.getLogger(__name__)

# Patterns that indicate the mutator model refused to generate an adversarial prompt
_REFUSAL_PATTERNS = [
    re.compile(r"I('m| am) sorry,? but I can.?t", re.IGNORECASE),
    re.compile(r"I cannot (help|assist|comply|provide|generate)", re.IGNORECASE),
    re.compile(r"I must refuse", re.IGNORECASE),
    re.compile(r"I will not (help|assist|comply|provide|generate)", re.IGNORECASE),
    re.compile(r"(is|are) (disallowed|not allowed|prohibited)", re.IGNORECASE),
    re.compile(
        r"violates? (my |the )?(safety|content|usage) (policy|policies|guidelines)", re.IGNORECASE
    ),
]

# Phrases that indicate the mutator is outputting reasoning/meta-text instead of a clean attack prompt
_META_TEXT_INDICATORS = [
    "we need to",
    "we must",
    "we should",
    "let's produce",
    "let us produce",
    "we have to",
    "according to policy",
    "the policy",
    "we are told",
    "the user is asking",
    "let me",
    "the instructions",
    "we can",
    "so we",
    "thus we",
    "check the rules",
    "ok produce",
    "ok let's",
    "the prompt should",
    "output only the",
    "new adversarial prompt",
    "my approach",
    "i'll create",
    "i will create",
    "here is my",
    "let's think",
    "step 1:",
    "first, i need to",
]


class AgenticMutator:
    """
    Template-based attack generator using shared LLM.

    Uses Jinja2 templates for prompt generation and produces
    adversarial prompts following the planner's strategy.
    """

    # Reasoning models spend a large share of their token budget on chain-of-
    # thought before emitting the mutated prompt; scale the cap so the prompt is
    # not truncated mid-thought (mirrors AgenticPlanner).
    _REASONING_TOKEN_MULTIPLIER: int = 4

    def __init__(
        self,
        llm: SharedLLM,
        template_dir: str = "templates",
        template_name: str = "mutator_general_llm.j2",
        temperature: float = 1.0,
        max_tokens: int = 512,
        max_retries: int = 2,
        few_shot_examples_file: str | None = None,
        few_shot_examples_per_style: int = 2,
        is_reasoning_model: bool = False,
    ):
        self._llm = llm
        self._template_dir = template_dir
        self._template_name = template_name
        self._temperature = temperature
        self._is_reasoning_model = is_reasoning_model
        self._max_tokens = (
            max_tokens * self._REASONING_TOKEN_MULTIPLIER if is_reasoning_model else max_tokens
        )
        if is_reasoning_model:
            logger.info(
                "Mutator reasoning model: max_tokens scaled %d → %d (×%d)",
                max_tokens,
                self._max_tokens,
                self._REASONING_TOKEN_MULTIPLIER,
            )
        self._max_retries = max_retries
        self._few_shot_examples_per_style = max(0, few_shot_examples_per_style)
        self._jinja_env: Environment | None = None
        self._mutation_count = 0
        self._last_raw_output: str = ""  # For debugging / logging
        self._few_shot_examples: dict[str, list[dict]] = self._load_few_shot_examples(
            few_shot_examples_file
        )

    @property
    def name(self) -> str:
        """Return the mutator's identifier."""
        return f"AgenticMutator({self._llm.model_name})"

    @staticmethod
    def _load_few_shot_examples(file_path: str | None) -> dict[str, list[dict]]:
        """Load few-shot attack examples from YAML file."""
        if not file_path:
            return {}
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"Few-shot examples file not found: {file_path}")
            return {}
        try:
            import yaml

            with open(path) as f:
                data = yaml.safe_load(f)
            examples = data.get("attack_style_examples", {})
            count = sum(len(v) for v in examples.values())
            logger.info(f"Loaded {count} few-shot examples for {len(examples)} attack styles")
            return examples
        except Exception as e:
            logger.warning(f"Failed to load few-shot examples: {e}")
            return {}

    def _get_examples_for_styles(
        self,
        primary_style: str,
        secondary_style: str | None = None,
    ) -> list[dict]:
        """Get few-shot examples for one or two attack styles.

        Returns up to ``few_shot_examples_per_style`` examples per style.
        For composite attacks, examples from both styles are merged and
        deduplicated by attack text while preserving style provenance.
        """
        # Normalize style name: try exact match, then lowercase, then with underscores
        styles = [primary_style]
        if secondary_style:
            styles.append(secondary_style)

        merged: OrderedDict[str, dict] = OrderedDict()
        for style in styles:
            style_key = style.lower().replace(" ", "_")
            examples = self._few_shot_examples.get(style_key, [])
            if self._few_shot_examples_per_style >= 0:
                examples = examples[: self._few_shot_examples_per_style]
            for ex in examples:
                attack_text = str(ex.get("attack", "")).strip()
                if not attack_text:
                    continue
                if attack_text in merged:
                    continue
                item = dict(ex)
                item.setdefault("style", style_key)
                merged[attack_text] = item

        return list(merged.values())

    def _ensure_templates_loaded(self) -> None:
        """Lazy-load Jinja2 templates."""
        if self._jinja_env is not None:
            return

        template_path = Path(self._template_dir)
        if not template_path.exists():
            template_path = Path(__file__).parent.parent.parent.parent.parent / self._template_dir

        self._jinja_env = Environment(
            loader=FileSystemLoader(str(template_path)),
            autoescape=False,
        )

    def mutate(
        self,
        original_prompt: str,
        attack_style: str,
        risk_category: str,
        feedback: str | None = None,
        previous_attempt: str | None = None,
        feedback_template: str | None = None,
        image_reference: str | None = None,
        tactical_instructions: str | None = None,
        secondary_style: str | None = None,
        planner_action: str | None = None,
        planner_rationale: str | None = None,
        planner_confidence: float | None = None,
        short_term_memory: str | None = None,
    ) -> tuple[str, float]:
        """
        Generate an adversarial prompt based on the attack strategy.

        Supports three modes for fair ablation:
        - Mode A (System 2): Strategy from planner with tactical instructions
        - Mode B (System 1 + Reflection): Feedback passed for tactical correction
        - Mode C (Blind): Random mutation, no feedback

        Args:
            original_prompt: The seed prompt to mutate
            attack_style: The attack style to apply
            risk_category: The target risk category
            feedback: Judge's reasoning from previous turn
            previous_attempt: The failed prompt from previous turn
            feedback_template: Template to use for feedback mode
            image_reference: Optional image reference for VLM attacks
            tactical_instructions: Detailed planner guidance for the mutator
            secondary_style: Optional secondary attack style for composite attacks

        Returns:
            Tuple of (mutated_prompt, elapsed_ms)
        """
        self._ensure_templates_loaded()
        self._mutation_count += 1

        # Select template based on mode
        if feedback is not None and feedback_template is not None:
            # Feedback mode: tactical correction with planner guidance
            try:
                template = self._jinja_env.get_template(feedback_template)
                prompt = template.render(
                    original_prompt=original_prompt,
                    previous_attempt=previous_attempt or original_prompt,
                    feedback=feedback,
                    attack_style=attack_style,
                    risk_category=risk_category,
                    image_reference=image_reference,
                    tactical_instructions=tactical_instructions or "",
                    secondary_style=secondary_style or "",
                    planner_action=planner_action or "",
                    planner_rationale=planner_rationale or "",
                    planner_confidence=planner_confidence,
                    short_term_memory=short_term_memory or "",
                )
            except Exception as e:
                logger.warning(f"Failed to load feedback template: {e}, using default")
                template = self._jinja_env.get_template(self._template_name)
                prompt = template.render(
                    original_prompt=original_prompt,
                    attack_style=attack_style,
                    risk_category=risk_category,
                    few_shot_examples=self._get_examples_for_styles(attack_style, secondary_style),
                    image_reference=image_reference,
                    tactical_instructions=tactical_instructions or "",
                    secondary_style=secondary_style or "",
                    planner_action=planner_action or "",
                    planner_rationale=planner_rationale or "",
                    planner_confidence=planner_confidence,
                    short_term_memory=short_term_memory or "",
                )
        else:
            # Standard mutation with few-shot examples + tactical instructions
            template = self._jinja_env.get_template(self._template_name)
            prompt = template.render(
                original_prompt=original_prompt,
                attack_style=attack_style,
                risk_category=risk_category,
                few_shot_examples=self._get_examples_for_styles(attack_style, secondary_style),
                image_reference=image_reference,
                tactical_instructions=tactical_instructions or "",
                secondary_style=secondary_style or "",
                planner_action=planner_action or "",
                planner_rationale=planner_rationale or "",
                planner_confidence=planner_confidence,
                short_term_memory=short_term_memory or "",
            )

        total_ms = 0.0
        best_candidate: str | None = None  # Track best output even if meta-text

        for attempt in range(self._max_retries + 1):
            temp = min(self._temperature + (attempt * 0.2), 1.5)
            response, elapsed_ms = self._generate(prompt, temp)
            total_ms += elapsed_ms
            self._last_raw_output = response

            mutated = self._clean_response(response)

            if self._is_refusal(mutated):
                logger.warning(
                    f"Mutator produced refusal (attempt {attempt + 1}/{self._max_retries + 1})"
                )
                continue

            # Check for meta-text leakage
            if self._has_meta_text(mutated):
                logger.warning(
                    f"Mutator output contains meta-text (attempt {attempt + 1}), "
                    f"will retry: {mutated[:80]}..."
                )
                # Keep as fallback — better than nothing
                if best_candidate is None or len(mutated) > len(best_candidate):
                    best_candidate = mutated
                continue

            # Clean output — return immediately
            return mutated, total_ms

        # All retries exhausted — use best candidate or original prompt
        if best_candidate and len(best_candidate) > 20:
            logger.warning("Returning best meta-text candidate after all retries exhausted")
            return best_candidate, total_ms

        logger.warning("Mutator refused after all retries, falling back to original prompt")
        return original_prompt, total_ms

    async def amutate(
        self,
        original_prompt: str,
        attack_style: str,
        risk_category: str,
        feedback: str | None = None,
        previous_attempt: str | None = None,
        feedback_template: str | None = None,
        image_reference: str | None = None,
        tactical_instructions: str | None = None,
        secondary_style: str | None = None,
        planner_action: str | None = None,
        planner_rationale: str | None = None,
        planner_confidence: float | None = None,
        short_term_memory: str | None = None,
    ) -> tuple[str, float]:
        """Async version of mutate() — uses SharedLLM async methods."""
        self._ensure_templates_loaded()
        self._mutation_count += 1

        # Select template based on mode
        if feedback is not None and feedback_template is not None:
            try:
                template = self._jinja_env.get_template(feedback_template)
                prompt = template.render(
                    original_prompt=original_prompt,
                    previous_attempt=previous_attempt or original_prompt,
                    feedback=feedback,
                    attack_style=attack_style,
                    risk_category=risk_category,
                    image_reference=image_reference,
                    tactical_instructions=tactical_instructions or "",
                    secondary_style=secondary_style or "",
                    planner_action=planner_action or "",
                    planner_rationale=planner_rationale or "",
                    planner_confidence=planner_confidence,
                    short_term_memory=short_term_memory or "",
                )
            except Exception as e:
                logger.warning(f"Failed to load feedback template: {e}, using default")
                template = self._jinja_env.get_template(self._template_name)
                prompt = template.render(
                    original_prompt=original_prompt,
                    attack_style=attack_style,
                    risk_category=risk_category,
                    few_shot_examples=self._get_examples_for_styles(attack_style, secondary_style),
                    image_reference=image_reference,
                    tactical_instructions=tactical_instructions or "",
                    secondary_style=secondary_style or "",
                    planner_action=planner_action or "",
                    planner_rationale=planner_rationale or "",
                    planner_confidence=planner_confidence,
                    short_term_memory=short_term_memory or "",
                )
        else:
            template = self._jinja_env.get_template(self._template_name)
            prompt = template.render(
                original_prompt=original_prompt,
                attack_style=attack_style,
                risk_category=risk_category,
                few_shot_examples=self._get_examples_for_styles(attack_style, secondary_style),
                image_reference=image_reference,
                tactical_instructions=tactical_instructions or "",
                secondary_style=secondary_style or "",
                planner_action=planner_action or "",
                planner_rationale=planner_rationale or "",
                planner_confidence=planner_confidence,
                short_term_memory=short_term_memory or "",
            )

        total_ms = 0.0
        best_candidate: str | None = None

        for attempt in range(self._max_retries + 1):
            temp = min(self._temperature + (attempt * 0.2), 1.5)
            response, elapsed_ms = await self._agenerate(prompt, temp)
            total_ms += elapsed_ms
            self._last_raw_output = response

            mutated = self._clean_response(response)

            if self._is_refusal(mutated):
                logger.warning(
                    f"Mutator produced refusal (attempt {attempt + 1}/{self._max_retries + 1})"
                )
                continue

            if self._has_meta_text(mutated):
                logger.warning(
                    f"Mutator output contains meta-text (attempt {attempt + 1}), "
                    f"will retry: {mutated[:80]}..."
                )
                if best_candidate is None or len(mutated) > len(best_candidate):
                    best_candidate = mutated
                continue

            return mutated, total_ms

        if best_candidate and len(best_candidate) > 20:
            logger.warning("Returning best meta-text candidate after all retries exhausted")
            return best_candidate, total_ms

        logger.warning("Mutator refused after all retries, falling back to original prompt")
        return original_prompt, total_ms

    async def _agenerate(self, prompt: str, temperature: float) -> tuple[str, float]:
        """Async version of _generate() — uses SharedLLM async methods."""
        if "<<<SYSTEM_USER_SEPARATOR>>>" in prompt:
            parts = prompt.split("<<<SYSTEM_USER_SEPARATOR>>>", 1)
            messages = [
                {"role": "system", "content": parts[0].strip()},
                {"role": "user", "content": parts[1].strip()},
            ]
            return await self._llm.agenerate_chat(
                messages,
                max_tokens=self._max_tokens,
                temperature=temperature,
            )
        else:
            return await self._llm.agenerate(
                prompt,
                max_tokens=self._max_tokens,
                temperature=temperature,
            )

    def _generate(self, prompt: str, temperature: float) -> tuple[str, float]:
        """
        Generate text from the mutator prompt.

        Detects chat templates (<<<SYSTEM_USER_SEPARATOR>>>) and uses
        generate_chat() for proper model-agnostic formatting. Falls back
        to generate() for legacy templates.
        """
        if "<<<SYSTEM_USER_SEPARATOR>>>" in prompt:
            parts = prompt.split("<<<SYSTEM_USER_SEPARATOR>>>", 1)
            messages = [
                {"role": "system", "content": parts[0].strip()},
                {"role": "user", "content": parts[1].strip()},
            ]
            return self._llm.generate_chat(
                messages,
                max_tokens=self._max_tokens,
                temperature=temperature,
            )
        else:
            return self._llm.generate(
                prompt,
                max_tokens=self._max_tokens,
                temperature=temperature,
            )

    @property
    def last_raw_output(self) -> str:
        """Return the raw model output from the last mutation (for debugging)."""
        return self._last_raw_output

    @staticmethod
    def _is_refusal(text: str) -> bool:
        """Check if the mutator output is a refusal to generate adversarial content."""
        if not text or len(text.strip()) < 10:
            return True
        for pattern in _REFUSAL_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _clean_response(self, response: str) -> str:
        """
        Extract the actual adversarial prompt from the mutator model output.

        Handles several common patterns:
        1. Model outputs reasoning then 'assistantfinal<actual prompt>'
        2. Model outputs reasoning then the actual prompt as last paragraph
        3. Model wraps output in quotes
        4. Model outputs clean prompt directly (ideal case)
        """
        if not response:
            return ""

        text = response.strip()

        # Pattern 1: Look for 'assistantfinal' marker (seen in gpt-oss-20b output)
        # The model outputs "...reasoning...assistantfinal<actual answer>"
        marker_patterns = ["assistantfinal", "assistant\nfinal", "FINAL ANSWER:", "Final answer:"]
        for marker in marker_patterns:
            idx = text.rfind(marker)
            if idx != -1:
                extracted = text[idx + len(marker) :].strip()
                # Remove leading '>' or other formatting artifacts
                extracted = extracted.lstrip(">").strip()
                if extracted and len(extracted) > 5:
                    return self._strip_quotes(extracted)

        # Pattern 2: Look for template output markers.
        # Only specific, unambiguous lead-ins belong here. Generic markers like
        # "Output:" / "PROMPT:" occur inside legitimate adversarial prompts
        # (e.g. "Format your Output: ...") and would truncate the payload to its
        # trailing fragment, so they are intentionally excluded.
        for marker in [
            "New adversarial prompt:",
            "new adversarial prompt:",
            "NEW ADVERSARIAL PROMPT:",
            "Rewritten prompt:",
            "rewritten prompt:",
            "REWRITTEN PROMPT:",
            "Here is the prompt:",
            "here is the prompt:",
        ]:
            idx = text.rfind(marker)
            if idx != -1:
                extracted = text[idx + len(marker) :].strip()
                if extracted and len(extracted) > 10:
                    return self._strip_quotes(extracted)

        # Pattern 3: If response contains reasoning paragraphs, take the last
        # substantial paragraph (likely the actual prompt)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            # Check if the response looks like reasoning (contains meta-language)
            first_para_lower = paragraphs[0].lower()
            if any(ind in first_para_lower for ind in _META_TEXT_INDICATORS):
                # Output contains reasoning — take the last substantial paragraph
                for para in reversed(paragraphs):
                    # Skip very short paragraphs and refusals
                    if len(para) > 20 and not self._is_refusal(para):
                        return self._strip_quotes(para)

        # Pattern 4: Entire output is meta-text — try to extract any quoted block
        text_lower = text.lower()
        if any(ind in text_lower[:200] for ind in _META_TEXT_INDICATORS):
            # Look for content in quotes as a last resort
            import re

            quoted = re.findall(r'"([^"]{20,})"', text)
            if quoted:
                # Return the longest quoted block
                return max(quoted, key=len)

        # Pattern 5: Clean output — strip quotes and return
        return self._strip_quotes(text)

    @staticmethod
    def _has_meta_text(text: str) -> bool:
        """Detect if the output contains mutator reasoning/meta-text instead of a clean attack prompt."""
        if not text or len(text.strip()) < 10:
            return False
        lower = text[:200].lower()
        return any(ind in lower for ind in _META_TEXT_INDICATORS)

    @staticmethod
    def _strip_quotes(text: str) -> str:
        """Remove surrounding quotes from text."""
        text = text.strip()
        if (text.startswith('"') and text.endswith('"')) or (
            text.startswith("'") and text.endswith("'")
        ):
            text = text[1:-1].strip()
        return text

    def validate_template(self) -> None:
        """
        Eagerly load the Jinja2 environment and validate the template exists.

        Call this from pipeline.initialize() so a missing template raises
        TemplateNotFound immediately at startup, not hours into an HPC run.
        """
        self._ensure_templates_loaded()
        assert self._jinja_env is not None
        self._jinja_env.get_template(self._template_name)

    def reset(self) -> None:
        """Reset mutator state for a new session."""
        self._mutation_count = 0
