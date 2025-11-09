"""LangGraph graph construction for the minimal agent.

This module wires the planner -> router -> actor flow using LangGraph.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

from langgraph.graph import END, StateGraph

from goob_ai.nodes import simple_actor, simple_router
from goob_ai.toolkit import default_tools
from goob_ai.types_new import AgentState, ToolRegistry
from goob_ai.planners import select_planner


PlannerCallable = Callable[[AgentState], AgentState]


def build_graph(tools: ToolRegistry | None = None, planner: PlannerCallable | None = None) -> StateGraph[AgentState]:
    """Build and return the agent graph.

    Args:
        tools: Optional tool registry. If None, uses built-in defaults.

    Returns:
        A configured LangGraph `StateGraph` for the agent.
    """

    registry = tools or default_tools()     #默认工具箱（toolkit里的所有工具）
    graph: StateGraph[AgentState] = StateGraph(AgentState)

    # Define nodes
    if planner is None:
        # Default to environment-selected planner (simple by default)
        import os

        planner_name = os.environ.get("PLANNER", "simple")
        planner = select_planner(planner_name)
    else:
        planner = select_planner(planner) if isinstance(planner, str) else planner
    
    # Inject available tool names and docstrings into state before planning
    tool_names = list(registry.keys())
    tool_docs = {}
    for name, fn in registry.items():
        doc = (getattr(fn, "__doc__", None) or "").strip()
        # compact first paragraph as one line
        if doc:
            first = doc.splitlines()[0].strip()
            tool_docs[name] = first if first else doc
        else:
            tool_docs[name] = ""

    def planner_with_tools(state: AgentState) -> AgentState:
        state["tool_names"] = tool_names
        state["tool_docs"] = tool_docs
        return planner(state)

    graph.add_node("plan", planner_with_tools)

    def act_wrapper(state: AgentState) -> AgentState:
        # Bind registry via closure
        return simple_actor(state, registry)

    graph.add_node("act", act_wrapper)

    # Entry
    graph.set_entry_point("plan")

    # Edges
    graph.add_conditional_edges("plan", simple_router, {"act": "act", "end": END})
    # After acting (tool execution), return to planner for another decision.
    graph.add_edge("act", "plan")

    return graph


def build_agent(tools: ToolRegistry | None = None, planner: PlannerCallable | None = None) -> Any:
    """Return a compiled agent (callable with `.invoke(state)`) with tools bound.

    Args:
        tools: Optional tool registry; defaults to `default_tools()`.
        planner: Optional planner function; defaults to env `PLANNER`.

    Returns:
        A compiled LangGraph object exposing `.invoke(AgentState)`.
    """

    graph = build_graph(tools=tools, planner=planner)
    return graph.compile()


