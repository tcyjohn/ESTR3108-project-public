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


class PlanStep(TypedDict, total=False):
    """A single step inside a multi-step long-term plan.

    Attributes:
        description: Human-readable description of the step.
        tool: Optional tool name to execute for this step.
        args: Optional arguments mapping for the tool call.
        status: Current status of the step.
        notes: Optional notes or observations attached to this step.
    """

    description: str
    tool: Optional[str]
    args: Optional[Mapping[str, Any]]
    status: Literal["todo", "doing", "done", "blocked"]
    notes: Optional[str]

class ActionRecord(TypedDict, total=False):
    tool: str
    args: Mapping[str, Any]
    observation: str
    
class AgentState(TypedDict, total=False):
    """State carried through the LangGraph execution."""

    messages: List[Message]
    plan: Optional[str]
    tool_call: Optional[ToolInput]
    result: Optional[str]
    tool_names: Optional[List[str]]
    tool_docs: Optional[Dict[str, str]]

    # Structured facts accumulated from tool observations during the session.
    # facts stores key -> value pairs (strings or numbers); facts_text is a
    # compact, stable textual summary injected into planner prompts.
    facts: Dict[str, Any]
    facts_text: Optional[str]

    # Recent action history for continuity (graph-maintained, planner-readable)
    # Each record should be compact: tool name, args, and a truncated observation

    action_history: List[ActionRecord]

    # Long-horizon planning
    plan_steps: List[PlanStep]
    current_step: int
    plan_needs_revision: bool

    # Supervision & loop-avoidance (optional; written by graph layer)
    last_tool: Optional[str]
    last_tool_args: Optional[Mapping[str, Any]]
    last_observation: Optional[str]
    tool_attempts: Dict[str, int]
    turn_count: int
    no_progress_count: int
    end_reason: Optional[str]

    # Optional budgets (can be injected via env in graph layer)
    max_turns: Optional[int]
    tool_attempt_limit: Optional[int]
    no_progress_limit: Optional[int]

    # Whether the last `act` actually executed a tool (set by graph layer)
    executed_tool: bool

    # Force replan if the planner emitted meta-instruction instead of final answer
    force_replan: bool


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




