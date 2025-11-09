"""Common types and data structures for the agent framework.

All functions and classes must be fully typed with Google-style docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple, TypedDict


class ToolInput(TypedDict, total=False):
    """TypedDict representing a tool input payload.

    Attributes:
        name: The name of the tool to invoke.
        args: The arguments to pass into the tool call.
    """

    name: str
    args: Mapping[str, Any]


class Message(TypedDict):
    """TypedDict representing a conversational message."""

    role: Literal["user", "assistant", "system"]
    content: str


class AgentState(TypedDict, total=False):
    """State carried through the LangGraph execution."""

    messages: List[Message]
    plan: Optional[str]
    tool_call: Optional[ToolInput]
    result: Optional[str]
    tool_names: Optional[List[str]]
    tool_docs: Optional[Dict[str, str]]


class Tool(Protocol):
    """Protocol for a synchronous tool function.

    Implementations should be pure or idempotent whenever possible.
    """

    def __call__(self, **kwargs: Any) -> str:
        """Invoke the tool and return a string result.

        Args:
            **kwargs: Arbitrary keyword arguments validated upstream.

        Returns:
            Result string produced by the tool.
        """


ToolRegistry = Mapping[str, Tool]




