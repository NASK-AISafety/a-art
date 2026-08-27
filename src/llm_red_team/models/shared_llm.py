"""
A-ART Shared LLM: Single Model Instance for Multi-Component Usage.

All generative LLM inference (planner, mutator, target, judge) routes through an
OpenAI-compatible API via ``litellm.acompletion()``.  Three backends are supported:

1. **vLLM** — local/HPC self-hosted models with continuous batching.
2. **Azure AI Foundry** — cloud-hosted models behind Azure API Management.
3. **OpenRouter** — cloud-hosted models exposed through an OpenAI-compatible API.

Both backends use the ``openai/`` litellm provider prefix, which appends
``/v1`` to the base URL automatically.  Azure Foundry additionally requires
an ``api-key`` HTTP header for APIM gateway authentication.

Local HuggingFace inference is intentionally NOT supported for generative models.
The only HuggingFace usage in the framework is the safety guards (PromptGuard,
OutputGuard) which are classifier models incompatible with vLLM's chat API.

Setup (vLLM):
    vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 0.0.0.0 --port 8000
    export VLLM_BASE_URL=http://localhost:8000

Setup (Azure AI Foundry):
    export AZURE_AI_ENDPOINT=https://your-apim-gateway.azure-api.net
    export AZURE_AI_API_KEY=your-subscription-key

Setup (OpenRouter):
    export OPENROUTER_API_KEY=your-openrouter-key
    export OPENROUTER_BASE_URL=https://openrouter.ai/api
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from enum import Enum
from typing import Any, Protocol, cast, runtime_checkable

from llm_red_team.types import ChatHistory

logger = logging.getLogger(__name__)

# Patterns for reasoning tokens from various models
_REASONING_PATTERNS = [
    # DeepSeek-R1 style: <think>...</think>
    re.compile(r"<think>.*?</think>", re.DOTALL),
    # Alternative: <reasoning>...</reasoning>
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL),
    # Mistral/Ministral style: [THINKING]...[/THINKING]
    re.compile(r"\[THINKING\].*?\[/THINKING\]", re.DOTALL),
    # Generic: <|think|>...<|/think|>
    re.compile(r"<\|think\|>.*?<\|/think\|>", re.DOTALL),
    # QwQ / Qwen-thinking style: <thought>...</thought>
    re.compile(r"<thought>.*?</thought>", re.DOTALL),
    # <output>...</output> wrappers (some models wrap final answer)
    re.compile(r"<output>(.*?)</output>", re.DOTALL),
]

# Plain-text markers that thinking models emit before the final answer.
# Content AFTER the last occurrence of these markers is the actual response.
#
# NOTE: only *unambiguous* channel artifacts belong here. Generic markers like
# "Final Answer:" or a markdown horizontal rule ("\n---\n") occur naturally
# inside legitimate (including harmful step-by-step) target responses, so
# splitting on them silently discards the real answer body — do NOT add them.
_FINAL_ANSWER_MARKERS = [
    "assistantfinal",  # gpt-oss-20b channel artifact
    "assistant\nfinal",  # gpt-oss-20b variant
]

# Leaked chat-template tokens that sometimes survive skip_special_tokens
_CHAT_TEMPLATE_LEAKS = re.compile(
    r"<\|(?:im_start|im_end|assistant|user|system|end|eot_id|start_header_id"
    r"|end_header_id|sep|pad|eos)\|>",
    re.IGNORECASE,
)


def strip_reasoning_tokens(text: str) -> str:
    """
    Strip reasoning/thinking blocks from model output.

    Reasoning models (DeepSeek-R1, Ministral-Reasoning, gpt-oss-20b, etc.)
    emit internal chain-of-thought in various formats. This function removes
    them so downstream components receive only the final answer.

    Handles:
        - <think>...</think>           (DeepSeek-R1, QwQ)
        - <reasoning>...</reasoning>   (generic)
        - [THINKING]...[/THINKING]     (Mistral/Ministral variants)
        - <|think|>...<|/think|>       (generic)
        - <thought>...</thought>       (QwQ/Qwen variants)
        - Plain-text markers like 'assistantfinal'  (gpt-oss-20b)
        - Leaked chat template tokens  (<|im_start|>, <|assistant|>, etc.)

    Also handles incomplete/unclosed thinking blocks (model ran out of tokens
    mid-thought).
    """
    result = text

    # Phase 1: Strip tagged reasoning blocks
    for pattern in _REASONING_PATTERNS:
        result = pattern.sub(lambda m: m.group(1) if m.lastindex else "", result)

    # Phase 2: Handle unclosed thinking blocks (model hit max_tokens mid-reasoning)
    # Keep only the content BEFORE the (unclosed) reasoning tag. If the whole
    # output was an unfinished chain-of-thought, the answer is genuinely empty —
    # return "" rather than fabricating an answer out of raw reasoning text
    # (returning the CoT would have the judge grade thinking, not a response).
    for open_tag in ["<think>", "<reasoning>", "[THINKING]", "<|think|>", "<thought>"]:
        if open_tag in result:
            idx = result.find(open_tag)
            result = result[:idx].strip()
            break

    # Phase 3: Handle plain-text final-answer markers (gpt-oss-20b, etc.)
    # Take content AFTER the last occurrence of any marker
    for marker in _FINAL_ANSWER_MARKERS:
        idx = result.rfind(marker)
        if idx != -1:
            extracted = result[idx + len(marker) :].strip()
            if extracted and len(extracted) > 5:
                result = extracted
                break

    # Phase 4: Strip leaked chat-template tokens
    result = _CHAT_TEMPLATE_LEAKS.sub("", result)

    return result.strip()


class InferenceBackend(str, Enum):
    """Inference backend for generative LLM calls."""

    VLLM = "vllm"
    AZURE_FOUNDRY = "azure_foundry"
    OPENROUTER = "openrouter"


@runtime_checkable
class GenerativeModelProtocol(Protocol):
    """Protocol for generative models used by agents."""

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    async def agenerate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> tuple[str, float]:
        """Async generate text from prompt. Returns (text, elapsed_ms)."""
        ...

    async def agenerate_chat(
        self,
        messages: ChatHistory,
        max_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> tuple[str, float]:
        """Async chat generation. Returns (text, elapsed_ms)."""
        ...


class SharedLLM:
    """
    Single LLM instance shared across planner/mutator/target/judge.

    Routes all inference through an OpenAI-compatible API via
    ``litellm.acompletion()``.  Supports two backends:

    - **vLLM**: self-hosted models on local GPU or HPC Slurm cluster.
      Multiple concurrent calls are batched server-side (continuous batching).
    - **Azure AI Foundry**: cloud-hosted models behind Azure APIM gateway.
      Requires ``api-key`` header for subscription-key authentication.

    Instance deduplication: components sharing (model_name, base_url, backend)
    are resolved to a single SharedLLM inside _build_components, so connection
    objects are reused across agents.

    Implements GenerativeModelProtocol for type safety.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        vllm_base_url: str | None = None,
        vllm_api_key: str = "EMPTY",
        is_vlm: bool | None = None,
        # Legacy parameters kept for config-file compatibility; silently ignored.
        torch_dtype: str = "auto",
        device_map: str = "auto",
        backend: str | InferenceBackend = InferenceBackend.VLLM,
    ):
        # Normalise backend to enum
        if isinstance(backend, str):
            try:
                backend = InferenceBackend(backend)
            except ValueError:
                raise ValueError(
                    f"Unknown inference backend '{backend}'. "
                    f"Supported: {[b.value for b in InferenceBackend]}"
                )

        self._backend = backend

        # Strip the "hosted_vllm/" prefix that _build_components may pass.
        if model_name.startswith("hosted_vllm/"):
            model_name = model_name[len("hosted_vllm/") :]
            if vllm_base_url is None:
                vllm_base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")

        # ── Backend-specific URL / key resolution ─────────────────────────
        if self._backend == InferenceBackend.AZURE_FOUNDRY:
            # Azure AI Foundry: resolve from AZURE_AI_* env vars
            if vllm_base_url is None:
                vllm_base_url = os.environ.get("AZURE_AI_ENDPOINT")
            if vllm_api_key == "EMPTY":
                vllm_api_key = os.environ.get("AZURE_AI_API_KEY", "EMPTY")
            if vllm_base_url is None:
                raise ValueError(
                    f"No Azure AI Foundry endpoint configured for model '{model_name}'. "
                    "Set the AZURE_AI_ENDPOINT environment variable or provide "
                    "base_url in the component YAML config block.\n"
                    "Example: export AZURE_AI_ENDPOINT=https://your-gateway.azure-api.net"
                )
            if vllm_api_key == "EMPTY":
                raise ValueError(
                    f"No Azure AI API key configured for model '{model_name}'. "
                    "Set the AZURE_AI_API_KEY environment variable.\n"
                    "Example: export AZURE_AI_API_KEY=your-subscription-key"
                )
            logger.info(f"Azure Foundry backend: {model_name} @ {vllm_base_url}")
        elif self._backend == InferenceBackend.OPENROUTER:
            if vllm_base_url is None:
                vllm_base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api")
            if vllm_api_key == "EMPTY":
                vllm_api_key = os.environ.get("OPENROUTER_API_KEY", "EMPTY")
            if vllm_base_url is None:
                raise ValueError(
                    f"No OpenRouter base URL configured for model '{model_name}'. "
                    "Set OPENROUTER_BASE_URL or provide base_url in the component YAML config block.\n"
                    "Example: export OPENROUTER_BASE_URL=https://openrouter.ai/api"
                )
            if vllm_api_key == "EMPTY":
                raise ValueError(
                    f"No OpenRouter API key configured for model '{model_name}'. "
                    "Set the OPENROUTER_API_KEY environment variable.\n"
                    "Example: export OPENROUTER_API_KEY=your-openrouter-key"
                )
            logger.info(f"OpenRouter backend: {model_name} @ {vllm_base_url}")
        else:
            # vLLM: resolve from VLLM_BASE_URL env var
            if vllm_base_url is None:
                vllm_base_url = os.environ.get("VLLM_BASE_URL")
            if vllm_base_url is None:
                raise ValueError(
                    f"No vLLM base URL configured for model '{model_name}'. "
                    "Set the VLLM_BASE_URL environment variable or provide "
                    "base_url in the component YAML config block.\n"
                    "Example: export VLLM_BASE_URL=http://localhost:8000"
                )

        self._model_name = model_name
        self._base_url = vllm_base_url
        self._vllm_base_url = vllm_base_url
        self._api_key = vllm_api_key
        # Explicit is_vlm overrides the name-matching heuristic.
        self._is_vlm = is_vlm if is_vlm is not None else self._detect_vlm(model_name)

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model_name

    @property
    def is_vlm(self) -> bool:
        """Return whether this is a Vision-Language Model."""
        return self._is_vlm

    @property
    def backend(self) -> InferenceBackend:
        """Return the inference backend."""
        return self._backend

    @property
    def is_loaded(self) -> bool:
        """Always True — models are served remotely; no local loading."""
        return True

    def _detect_vlm(self, model_name: str) -> bool:
        """Detect if model is a Vision-Language Model from its name."""
        vlm_indicators = ["-VL-", "-VL", "VL-", "vision", "multimodal"]
        return any(indicator.lower() in model_name.lower() for indicator in vlm_indicators)

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def agenerate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> tuple[str, float]:
        """
        Async text generation via vLLM server (litellm.acompletion).

        Multiple concurrent calls are batched server-side for maximum throughput.
        ``do_sample=False`` maps to ``temperature=0.0`` (greedy decoding) because
        the OpenAI-compatible API does not expose a ``do_sample`` flag.

        Returns:
            Tuple of (generated_text, elapsed_ms).
        """
        effective_temperature = 0.0 if not do_sample else temperature
        return await self._agenerate_vllm(prompt, max_tokens, effective_temperature)

    async def agenerate_chat(
        self,
        messages: ChatHistory,
        max_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> tuple[str, float]:
        """
        Async chat generation via vLLM server (litellm.acompletion).

        Accepts standard OpenAI message format including multimodal content parts
        (image_url, base64 images) forwarded natively through the
        OpenAI-compatible API.

        Returns:
            Tuple of (generated_text, elapsed_ms).
        """
        effective_temperature = 0.0 if not do_sample else temperature
        return await self._agenerate_chat_vllm(messages, max_tokens, effective_temperature)

    # ------------------------------------------------------------------
    # vLLM backend helpers (litellm.acompletion)
    # ------------------------------------------------------------------

    def _litellm_model_string(self) -> str:
        """
        Build the litellm model string.

        Uses the ``openai/`` provider prefix so litellm automatically appends
        ``/v1`` to the base URL and uses OpenAI-compatible headers.  This works
        for both vLLM (which mounts ``/v1/*`` routes) and Azure AI Foundry
        (whose APIM gateway expects ``/v1/chat/completions``).

        Returns:
            ``openai/<model_name>``
        """
        return f"openai/{self._model_name}"

    async def _acompletion_with_retry(
        self,
        *,
        messages: list[Any],
        max_tokens: int,
        temperature: float,
        max_retries: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 30.0,
    ) -> Any:
        """
        Call litellm.acompletion with exponential-backoff retry for transient errors.

        Critical for Slurm jobs where the vLLM server and the Python client are
        launched at the same time: the server may need 30–120 s to load weights
        before it starts accepting requests.

        Retryable conditions (exception type AND message keyword must both match):
            - ``APIConnectionError``     — connection refused / network unreachable
            - ``ServiceUnavailableError`` (HTTP 503) — vLLM weights still loading
            - ``InternalServerError``    — litellm wraps hosted_vllm connection
              failures as this type (e.g. "Hosted_vllmException - Connection error.")
              rather than as ``APIConnectionError``; only retried when the message
              contains a connection-related keyword so genuine InternalServerErrors
              (bad model output, etc.) are NOT retried.

        Non-retryable errors (AuthenticationError, RateLimitError, etc.) are
        re-raised immediately.

        Args:
            messages: OpenAI-format message list passed to ``litellm.acompletion``.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            max_retries: Maximum number of retry attempts (default 5 → up to ~60 s wait).
            base_delay: Initial back-off in seconds, doubles each attempt (default 2.0).
            max_delay: Cap on per-attempt sleep (default 30.0 s).

        Returns:
            litellm ``ModelResponse`` object.

        Raises:
            The last exception if all retries are exhausted without success.
        """
        import litellm

        _retryable_exc: tuple[type[Exception], ...] = (
            litellm.exceptions.APIConnectionError,
            litellm.exceptions.ServiceUnavailableError,
            # litellm wraps hosted_vllm connection failures as InternalServerError
            # (e.g. "Hosted_vllmException - Connection error.") rather than as
            # APIConnectionError.  Include it here so the keyword check below
            # can still trigger retries on connection-related messages.
            litellm.exceptions.InternalServerError,
        )

        last_exc: BaseException | None = None
        # Ensure base_url ends with /v1 for the OpenAI-compatible endpoint.
        # litellm's openai/ provider requires base_url (not api_base) and doesn't
        # auto-append /v1, so we must append it explicitly.
        base_url = (
            f"{self._base_url}/v1"
            if self._base_url and not self._base_url.endswith("/v1")
            else self._base_url
        )

        # Azure APIM requires the subscription key in the 'api-key' HTTP header.
        # The openai/ provider sends api_key only as 'Authorization: Bearer'
        # which APIM does not accept — so we must add the header explicitly.
        extra_headers: dict[str, str] | None = None
        if self._backend == InferenceBackend.AZURE_FOUNDRY:
            extra_headers = {"api-key": self._api_key}

        backend_label = (
            "vLLM"
            if self._backend == InferenceBackend.VLLM
            else "Azure Foundry"
            if self._backend == InferenceBackend.AZURE_FOUNDRY
            else "OpenRouter"
        )

        for attempt in range(max_retries + 1):
            try:
                return await litellm.acompletion(
                    model=self._litellm_model_string(),
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    base_url=base_url,
                    api_key=self._api_key,
                    extra_headers=extra_headers,
                )
            except Exception as exc:
                exc_str = str(exc).lower()
                is_retryable = isinstance(exc, _retryable_exc) and any(
                    kw in exc_str
                    for kw in (
                        "connection refused",
                        "connection error",  # litellm hosted_vllm wrapping
                        "503",
                        "service unavailable",
                        "not ready",
                        "econnrefused",  # node.js-style kept for completeness
                    )
                )
                if not is_retryable or attempt >= max_retries:
                    raise
                delay = min(base_delay * (2**attempt), max_delay)
                logger.warning(
                    f"{backend_label} endpoint not ready "
                    f"(attempt {attempt + 1}/{max_retries}, retry in {delay:.1f}s): "
                    f"{type(exc).__name__}: {exc}"
                )
                last_exc = exc
                await asyncio.sleep(delay)

        # Unreachable — loop always raises or returns; satisfies type-checker.
        assert last_exc is not None
        raise last_exc

    async def _agenerate_vllm(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, float]:
        """Generate via vLLM server using litellm.acompletion() with retry."""
        start_time = time.perf_counter()

        messages: list[Any] = [{"role": "user", "content": prompt}]
        response = await self._acompletion_with_retry(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        text = response.choices[0].message.content or ""
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        self._warn_if_truncated(response, max_tokens)
        cleaned = strip_reasoning_tokens(text)
        return cleaned.strip(), elapsed_ms

    async def _agenerate_chat_vllm(
        self,
        messages: ChatHistory,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, float]:
        """Chat generation via vLLM server using litellm.acompletion() with retry."""
        start_time = time.perf_counter()

        # cast: litellm accepts list[Any]; ChatHistory (list[ChatMessage]) is
        # structurally compatible but invariant list typing requires explicit cast.
        response = await self._acompletion_with_retry(
            messages=cast(list[Any], messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )

        text = response.choices[0].message.content or ""
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        self._warn_if_truncated(response, max_tokens)
        cleaned = strip_reasoning_tokens(text)
        return cleaned.strip(), elapsed_ms

    def _warn_if_truncated(self, response: Any, max_tokens: int) -> None:
        """Log a warning when the server stopped generation at the token cap.

        vLLM sets ``finish_reason == "length"`` when output is cut off at
        ``max_tokens``. Truncated text is otherwise silently recorded and graded
        as if complete (the historical source of the truncation bug), so surface
        it loudly and point at the fix.
        """
        try:
            finish_reason = response.choices[0].finish_reason
        except (AttributeError, IndexError):
            return
        if finish_reason == "length":
            logger.warning(
                "Output truncated at max_tokens=%d (finish_reason='length') for "
                "model %s; the recorded response is incomplete. Raise the "
                "corresponding *_max_tokens to capture the full response.",
                max_tokens,
                self._model_name,
            )

    # ------------------------------------------------------------------
    # Resource Management
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """No-op: vLLM serves models from the server; no local memory to free."""
        logger.debug(f"unload() called (no-op for vLLM backend): {self._model_name}")
