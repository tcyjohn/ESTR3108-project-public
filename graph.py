"""LangGraph graph construction for the minimal agent.

This module wires the planner -> router -> actor flow using LangGraph.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

from langgraph.graph import END, StateGraph
import os

from goob_ai.nodes import simple_actor, simple_router, assess_step
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
    
    # Single hard limit: per-tool attempt limit (soft constraints live in planners)
    try:
        tool_attempt_limit_default = int(os.environ.get("TOOL_ATTEMPT_LIMIT", "3"))
    except Exception:
        tool_attempt_limit_default = 3
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
        # initialize supervision counters if absent (no hard stop except per-tool attempts)
        state.setdefault("turn_count", 0)
        state.setdefault("tool_attempts", {})
        state.setdefault("tool_attempt_limit", tool_attempt_limit_default)
        return planner(state)

    graph.add_node("plan", planner_with_tools)

    def act_wrapper(state: AgentState) -> AgentState:
        # capture pending tool call
        pending = state.get("tool_call") or {}
        tname = pending.get("name")
        targs = dict(pending.get("args") or {})
        # execute
        new_state = simple_actor(state, registry)
        # update supervision if a tool executed
        if tname:
            attempts = new_state.get("tool_attempts") or {}
            attempts[tname] = int(attempts.get(tname, 0)) + 1
            new_state["tool_attempts"] = attempts
            new_state["turn_count"] = int(new_state.get("turn_count") or 0) + 1
            new_state["last_tool"] = tname
            new_state["last_tool_args"] = targs
            new_state["last_observation"] = new_state.get("result") or ""
            new_state["executed_tool"] = True
            # append compact action record (keep last 4)
            hist = list(new_state.get("action_history") or [])
            hist.append({
                "tool": tname,
                "args": targs,
                "observation": (new_state.get("result") or "")[:600],
            })
            if len(hist) > 4:
                hist = hist[-4:]
            new_state["action_history"] = hist
        else:
            # no tool; this is a final reply emission path
            new_state["executed_tool"] = False
            # detect meta-instruction instead of final answer; if so, remove last message and force replan
            content = (new_state.get("result") or "").strip().lower()
            meta_signals = (
                "combine the two facts",
                "combine the facts",
                "final answer",
                "answer the question",
                "respond to the question",
                "use these facts",
                "conclude",
                "总结",
                "结合",
                "输出最终答案",
                "回答问题",
                "给出最终答案",
            )
            if content and any(sig in content for sig in meta_signals):
                msgs = list(new_state.get("messages") or [])
                if msgs and isinstance(msgs[-1], dict) and (msgs[-1].get("content", "").strip().lower() == content):
                    msgs = msgs[:-1]
                    new_state["messages"] = msgs
                new_state["result"] = ""
                new_state["force_replan"] = True
        return new_state

    graph.add_node("act", act_wrapper)
    graph.add_node("assess", assess_step)

    # Entry
    graph.set_entry_point("plan")

    # Edges
    def router_guard(state: AgentState) -> str:
        """Route from plan to act always, except when per-tool attempts exceeded.

        We intentionally avoid early 'end' here even if tool_call is None,
        so that 'act' can emit the final answer and the post-act router decides to END.
        """
        # per-tool attempt limit only (single hard constraint)
        tc = state.get("tool_call") or {}
        name = tc.get("name")
        if name:
            attempts = state.get("tool_attempts") or {}
            attempt_limit = int(state.get("tool_attempt_limit") or tool_attempt_limit_default)
            if int(attempts.get(name, 0)) >= attempt_limit:
                state["end_reason"] = f"tool_attempts_exceeded:{name}"
                # Clear tool_call and let 'act' emit the final assistant message from 'plan'
                state["plan"] = f"工具 {name} 尝试次数达上限，结束。"
                state["tool_call"] = None
                return "act"
        return "act"

    graph.add_conditional_edges("plan", router_guard, {"act": "act", "end": END})
    # After acting: if a tool executed -> assess; else -> end (final answer path).
    def post_act_router(state: AgentState) -> str:
        if bool(state.get("executed_tool")):
            return "assess"
        # If no tool executed and we detected meta instruction, replan instead of ending
        if bool(state.get("force_replan")):
            state["force_replan"] = False
            # also hint planner to revise
            state["plan_needs_revision"] = True
            return "plan"
        return "end"
    graph.add_conditional_edges("act", post_act_router, {"assess": "assess", "end": END})
    graph.add_edge("assess", "plan")

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


