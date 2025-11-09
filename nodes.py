"""Core nodes: planner, actor, router.

These are intentionally simple but fully typed so users can replace them.
"""

from __future__ import annotations

from typing import Literal

from goob_ai.types_new import AgentState, ToolRegistry

def simple_actor(state: AgentState, tools: ToolRegistry) -> AgentState:
    """Execute the tool call if present and append result to messages.

    Args:
        state: Current agent state with optional tool_call.
        tools: Registered tool implementations.

    Returns:
        Updated state with result and assistant message.
    """

    tool_call = state.get("tool_call")
    if not tool_call:
        content = state.get("plan") or ""
        messages = state.get("messages", []) + [{"role": "assistant", "content": content}]
        state["messages"] = messages
        state["result"] = content
        return state

    name = tool_call.get("name")
    args = tool_call.get("args", {})
    tool = tools.get(name)
    if tool is None:
        content = f"Tool not found: {name}"
    else:
        content = tool(**args)

    messages = state.get("messages", []) + [{"role": "assistant", "content": content}]
    state["messages"] = messages
    state["result"] = content
    # Clear tool_call so that next planning cycle can decide afresh
    state["tool_call"] = None
    return state


def simple_router(state: AgentState) -> Literal["act", "end"]:
    """Route to act if a tool_call exists; otherwise end.

    Args:
        state: Current agent state.

    Returns:
        Next step label.
    """

    return "act" if state.get("tool_call") else "end"


