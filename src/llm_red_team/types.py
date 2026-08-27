"""
A-ART Shared Type Definitions.

Central home for TypedDicts that describe OpenAI-format chat messages.
Using TypedDict instead of ``dict[str, Any]`` gives mypy the information it needs
to catch malformed message construction at type-check time.

Python 3.9 compat:
    ``TypeAlias`` is 3.10+.  We use bare variable assignments here; mypy
    treats ``Var = Union[...]`` and ``Var = list[...]`` as implicit type aliases
    without needing the explicit ``TypeAlias`` marker.
"""

from __future__ import annotations

from typing import Literal, TypedDict, Union


class TextPart(TypedDict):
    """A plain-text content part in a multimodal message."""

    type: Literal["text"]
    text: str


class _ImageURLInner(TypedDict, total=False):
    """Inner dict for image_url content parts."""

    url: str
    detail: str  # "auto" | "low" | "high" — optional per OpenAI spec


class ImageURLPart(TypedDict):
    """An image-URL content part in a multimodal message."""

    type: Literal["image_url"]
    image_url: _ImageURLInner


# A single content part — either plain text or an image reference.
ContentPart = Union[TextPart, ImageURLPart]

# Message content is either a plain string (text-only) or a list of content parts.
MessageContent = Union[str, list[ContentPart]]


class ChatMessage(TypedDict):
    """A single message in an OpenAI-format chat history."""

    role: str
    content: MessageContent


# Full conversation history passed to chat-completion endpoints.
ChatHistory = list[ChatMessage]
