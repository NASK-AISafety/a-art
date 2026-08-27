"""
Tests for SharedLLM backends.

Verifies:
    1. Raises ValueError when no base URL is configured
    2. Resolves base_url from VLLM_BASE_URL env var
    3. Properties: model_name, backend, is_loaded, is_vlm
    4. VLM auto-detection from model name heuristic
    5. hosted_vllm/ prefix stripping
    6. agenerate() delegates to vLLM path (mocked litellm)
    7. agenerate_chat() delegates to vLLM path (mocked litellm)
    8. strip_reasoning_tokens removes thinking blocks
    9. OpenRouter resolves base URL and API key from env
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_red_team.models.shared_llm import InferenceBackend, SharedLLM, strip_reasoning_tokens

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(base_url: str = "http://localhost:8000", **kwargs: object) -> SharedLLM:
    return SharedLLM(model_name="Qwen/Qwen2.5-1.5B-Instruct", vllm_base_url=base_url, **kwargs)


def _mock_litellm_response(text: str) -> MagicMock:
    """Build a minimal litellm ModelResponse mock."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Construction / URL resolution
# ---------------------------------------------------------------------------


class TestSharedLLMConstruction:
    def test_raises_without_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SharedLLM must raise ValueError when no base URL is provided."""
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        with pytest.raises(ValueError, match="VLLM_BASE_URL"):
            SharedLLM(model_name="Qwen/Qwen2.5-1.5B-Instruct")

    def test_resolves_base_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_BASE_URL", "http://vllm-server:8000")
        llm = SharedLLM(model_name="Qwen/Qwen2.5-1.5B-Instruct")
        assert llm._vllm_base_url == "http://vllm-server:8000"

    def test_explicit_base_url_takes_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_BASE_URL", "http://env-server:8000")
        llm = SharedLLM(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            vllm_base_url="http://explicit:9000",
        )
        assert llm._vllm_base_url == "http://explicit:9000"

    def test_hosted_vllm_prefix_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000")
        llm = SharedLLM(model_name="hosted_vllm/Qwen/Qwen2.5-1.5B-Instruct")
        assert llm.model_name == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_openrouter_resolves_url_and_key_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
        llm = SharedLLM(
            model_name="x-ai/grok-4.3",
            backend=InferenceBackend.OPENROUTER,
        )
        assert llm.backend == InferenceBackend.OPENROUTER
        assert llm._vllm_base_url == "https://openrouter.ai/api"
        assert llm._api_key == "or-key"


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestSharedLLMProperties:
    def test_model_name(self) -> None:
        llm = _make_llm()
        assert llm.model_name == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_backend_is_always_vllm(self) -> None:
        llm = _make_llm()
        assert llm.backend == InferenceBackend.VLLM

    def test_is_loaded_always_true(self) -> None:
        llm = _make_llm()
        assert llm.is_loaded is True

    def test_is_vlm_false_for_text_model(self) -> None:
        llm = _make_llm()
        assert llm.is_vlm is False

    def test_is_vlm_true_for_vl_model(self) -> None:
        llm = SharedLLM(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            vllm_base_url="http://localhost:8000",
        )
        assert llm.is_vlm is True

    def test_is_vlm_explicit_override(self) -> None:
        llm = SharedLLM(
            model_name="some-custom-model",
            vllm_base_url="http://localhost:8000",
            is_vlm=True,
        )
        assert llm.is_vlm is True

    def test_litellm_model_string(self) -> None:
        llm = _make_llm()
        assert llm._litellm_model_string() == "openai/Qwen/Qwen2.5-1.5B-Instruct"

    def test_unload_is_noop(self) -> None:
        """unload() must not raise; it's a no-op for vLLM."""
        llm = _make_llm()
        llm.unload()  # must not raise


# ---------------------------------------------------------------------------
# Async generation (litellm mocked)
# ---------------------------------------------------------------------------


class TestSharedLLMAsync:
    @patch("litellm.acompletion")
    async def test_agenerate_calls_vllm(self, mock_acompletion: AsyncMock) -> None:
        mock_acompletion.return_value = _mock_litellm_response("Generated text.")
        llm = _make_llm()
        text, elapsed = await llm.agenerate("Hello world", max_tokens=64)
        assert text == "Generated text."
        assert elapsed >= 0.0
        mock_acompletion.assert_called_once()

    @patch("litellm.acompletion")
    async def test_agenerate_chat_calls_vllm(self, mock_acompletion: AsyncMock) -> None:
        mock_acompletion.return_value = _mock_litellm_response("Chat response.")
        llm = _make_llm()
        messages = [{"role": "user", "content": "test"}]
        text, elapsed = await llm.agenerate_chat(messages, max_tokens=64)
        assert text == "Chat response."
        assert elapsed >= 0.0

    @patch("litellm.acompletion")
    async def test_do_sample_false_uses_zero_temperature(self, mock_acompletion: AsyncMock) -> None:
        mock_acompletion.return_value = _mock_litellm_response("greedy output")
        llm = _make_llm()
        await llm.agenerate("prompt", temperature=0.9, do_sample=False)
        call_kwargs = mock_acompletion.call_args
        assert call_kwargs.kwargs["temperature"] == 0.0

    @patch("litellm.acompletion")
    async def test_base_url_gets_v1_appended(self, mock_acompletion: AsyncMock) -> None:
        mock_acompletion.return_value = _mock_litellm_response("ok")
        llm = _make_llm(base_url="http://localhost:8000")
        await llm.agenerate("prompt")
        call_kwargs = mock_acompletion.call_args
        assert call_kwargs.kwargs["base_url"] == "http://localhost:8000/v1"

    @patch("litellm.acompletion")
    async def test_base_url_v1_not_doubled(self, mock_acompletion: AsyncMock) -> None:
        mock_acompletion.return_value = _mock_litellm_response("ok")
        llm = _make_llm(base_url="http://localhost:8000/v1")
        await llm.agenerate("prompt")
        call_kwargs = mock_acompletion.call_args
        assert call_kwargs.kwargs["base_url"] == "http://localhost:8000/v1"

    @patch("litellm.acompletion")
    async def test_openrouter_uses_openai_compatible_call(
        self, mock_acompletion: AsyncMock
    ) -> None:
        mock_acompletion.return_value = _mock_litellm_response("ok")
        llm = SharedLLM(
            model_name="deepseek/deepseek-v4-pro",
            vllm_base_url="https://openrouter.ai/api",
            vllm_api_key="or-key",
            backend=InferenceBackend.OPENROUTER,
        )
        await llm.agenerate("prompt")
        call_kwargs = mock_acompletion.call_args
        assert call_kwargs.kwargs["model"] == "openai/deepseek/deepseek-v4-pro"
        assert call_kwargs.kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert call_kwargs.kwargs["api_key"] == "or-key"
        assert call_kwargs.kwargs["extra_headers"] is None


# ---------------------------------------------------------------------------
# strip_reasoning_tokens
# ---------------------------------------------------------------------------


class TestStripReasoningTokens:
    def test_strips_think_tags(self) -> None:
        text = "<think>internal reasoning step</think>Final answer."
        assert strip_reasoning_tokens(text) == "Final answer."

    def test_strips_reasoning_tags(self) -> None:
        text = "<reasoning>lots of work</reasoning>Done."
        assert strip_reasoning_tokens(text) == "Done."

    def test_strips_thinking_brackets(self) -> None:
        text = "[THINKING]some thoughts[/THINKING]The result."
        assert strip_reasoning_tokens(text) == "The result."

    def test_clean_text_unchanged(self) -> None:
        text = "This is a plain response with no thinking tags."
        assert strip_reasoning_tokens(text) == text

    def test_unclosed_think_block_returns_content_before(self) -> None:
        text = "Answer text.<think>started thinking but never finished"
        result = strip_reasoning_tokens(text)
        assert "Answer text." in result
        assert "<think>" not in result

    def test_truncated_mid_think_returns_empty_not_cot(self) -> None:
        # Reasoning target cut off at max_tokens mid-<think>: the answer is
        # genuinely absent, so we must NOT surface the raw chain-of-thought as
        # the response (the judge would otherwise grade reasoning text).
        text = "<think>step 1 figure out how to... step 2 the payload would be"
        result = strip_reasoning_tokens(text)
        assert result == ""

    def test_markdown_rule_is_not_treated_as_answer_marker(self) -> None:
        # A markdown horizontal rule inside a normal (step-by-step) response must
        # be preserved, not used to discard everything before it.
        text = "Here is how:\n\n1. first step\n2. second step\n\n---\n\nNote: caveat"
        result = strip_reasoning_tokens(text)
        assert "first step" in result
        assert "second step" in result

    def test_strips_chat_template_leaks(self) -> None:
        text = "<|im_start|>Final answer without template tokens."
        result = strip_reasoning_tokens(text)
        assert "<|im_start|>" not in result
        assert "Final answer" in result
