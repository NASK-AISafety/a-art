"""
A-ART Prompt Guard: Input Safety Filter using Llama-Prompt-Guard-2.

This module provides input filtering to detect prompt injection and jailbreak attempts
before they reach the target model.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    """Result from a guard evaluation."""

    is_safe: bool
    score: float
    label: str
    elapsed_ms: float


@runtime_checkable
class GuardProtocol(Protocol):
    """Protocol for input/output guards."""

    def evaluate(self, text: str) -> GuardResult:
        """Evaluate text for safety. Returns GuardResult."""
        ...

    def load(self) -> None:
        """Load the guard model."""
        ...

    def unload(self) -> None:
        """Unload the guard model."""
        ...


class PromptGuard:
    """
    Llama-Prompt-Guard-2-86M input filter.

    Classifies prompts as safe/unsafe before sending to target.
    Uses a DeBERTa-based sequence classification model.

    Labels:
        - 0: benign (safe)
        - 1: injection (unsafe)
        - 2: jailbreak (unsafe)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-Prompt-Guard-2-86M",
        threshold: float = 0.5,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        if device:
            self.device = device
        elif torch.cuda.is_available():
            # Check if installed PyTorch supports the GPU architecture
            try:
                cap = torch.cuda.get_device_capability(0)
                # PyTorch compiled CUDA arch list (sm_50..sm_90 typical)
                # If sm_100+ (Blackwell) and not supported, fall back to CPU
                if cap[0] >= 10:
                    # Quick probe: try a small tensor op on CUDA
                    _t = torch.zeros(1, device="cuda")
                    del _t
                self.device = "cuda"
            except RuntimeError:
                self.device = "cpu"
        else:
            self.device = "cpu"
        self._model = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        """Return whether the model is loaded."""
        return self._model is not None

    def load(self) -> None:
        """Load the prompt guard model (DeBERTa-based classifier)."""
        if self._model is not None:
            return

        logger.info(f"Loading Prompt Guard: {self.model_name}")

        # Use the device configured in __init__ (respects explicit override)
        device = torch.device(self.device)
        # Use float16 on GPU for speed, float32 on CPU for compatibility
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
        )

        # Manually move to device (DeBERTa doesn't support device_map='auto' in transformers 4.46.0)
        self._model = self._model.to(device)
        self._model.eval()
        logger.info(f"Prompt Guard loaded successfully on {device} with dtype={dtype}")

    def evaluate(self, prompt: str) -> GuardResult:
        """
        Evaluate if a prompt is safe using classification.

        Args:
            prompt: The text to evaluate

        Returns:
            GuardResult with safety verdict
        """
        if self._model is None:
            self.load()

        start_time = time.perf_counter()

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            unsafe_prob = (
                probs[0, 1].item() + probs[0, 2].item()
                if probs.shape[1] > 2
                else probs[0, 1].item()
            )
            predicted_class = torch.argmax(logits, dim=-1).item()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        class_labels = {0: "benign", 1: "injection", 2: "jailbreak"}
        label = class_labels.get(predicted_class, "unknown")
        is_safe = predicted_class == 0

        logger.debug(f"Prompt Guard: {label} (unsafe_prob={unsafe_prob:.2f}, {elapsed_ms:.1f}ms)")

        return GuardResult(
            is_safe=is_safe,
            score=unsafe_prob,
            label=label,
            elapsed_ms=elapsed_ms,
        )

    async def aevaluate(self, prompt: str) -> GuardResult:
        """Async wrapper — offloads blocking HF inference to a thread."""
        return await asyncio.to_thread(self.evaluate, prompt)

    def unload(self) -> None:
        """Unload model to free VRAM."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            torch.cuda.empty_cache()
            logger.info(f"Unloaded guard: {self.model_name}")
