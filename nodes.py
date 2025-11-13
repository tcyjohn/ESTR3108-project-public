"""Core nodes: planner, actor, router.

These are intentionally simple but fully typed so users can replace them.
"""

from __future__ import annotations

from typing import Literal, Any

from goob_ai.types_new import AgentState, ToolRegistry, PlanStep

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


def assess_step(state: AgentState) -> AgentState:
    """Evaluate the last action result, extract tool-specific facts, and update progress.

    Retention policy:
        - Full retention (web_search, calculator, analyze_table): store entire result up to 6000 chars.
        - Partial retention (other tools): store truncated summary up to 1000 chars.
        - Facts are accumulated across steps and made available to planner via facts_text.

    Progress heuristics:
        - If there is a non-empty result without obvious error signals,
          mark current step as done and advance `current_step`.
        - Otherwise mark as blocked and set `plan_needs_revision=True`.
        - If planner produced direct response (no tool_call), also treat as success.

    Args:
        state: Current agent state.

    Returns:
        Updated state with facts extracted and plan progress updated.
    """

    # 1) Tool-specific fact extraction based on retention policy
    try:
        result_text = state.get("result") or ""
        # Get the actual tool name from last_tool (executed_tool is just a boolean flag)
        tool_name = state.get("last_tool") or ""
        
        if result_text and tool_name:
            current_facts = dict(state.get("facts") or {})
            
            # Define retention policy per tool type
            # Full retention: store entire result (useful for calculators and search results)
            FULL_RETENTION_TOOLS = {"web_search", "calculator", "analyze_table"}
            # Partial retention: store truncated summary only
            PARTIAL_RETENTION_LENGTH = 1000
            
            if tool_name in FULL_RETENTION_TOOLS:
                # Store full result for high-value tools (limited to prevent bloat)
                fact_key = f"{tool_name}_result"
                value_to_store = result_text[:6000]
                
                # Special handling for different tool types
                if tool_name == "calculator":
                    # Extract just the numeric result from "result=123.45" format
                    import re
                    match = re.search(r"result\s*=\s*([\d.e+-]+)", value_to_store, re.I)
                    if match:
                        value_to_store = match.group(1)
                elif tool_name == "web_search":
                    # Extract only the answer line if present
                    import re
                    answer_match = re.search(r"(?:^|\n)answer:\s*([^\n]+)", value_to_store, re.I)
                    if answer_match:
                        value_to_store = answer_match.group(1).strip()
                
                current_facts[fact_key] = value_to_store
            else:
                # For other tools (wiki_retrieve, arxiv_search, etc.), store truncated summary
                fact_key = f"{tool_name}_summary"
                current_facts[fact_key] = result_text[:PARTIAL_RETENTION_LENGTH]
            
            # Build stable facts_text from all accumulated facts
            if current_facts:
                items = []
                for k in sorted(current_facts.keys()):
                    v = current_facts[k]
                    # Truncate individual fact values for display in facts_text
                    v_str = str(v)
                    if len(v_str) > 400:
                        v_str = v_str[:400] + "..."
                    items.append(f"{k}={v_str}")
                
                facts_text = "; ".join(items)
                # Overall facts_text limit to keep planner prompt manageable
                if len(facts_text) > 3000:
                    facts_text = facts_text[:3000] + "..."
                
                state["facts"] = current_facts
                state["facts_text"] = facts_text
                print(f"facts_text: {facts_text}")
    except Exception:
        # Facts extraction is best-effort; never block the main flow.
        pass

    # 2) Update long-term plan progress (existing heuristic)
    steps = state.get("plan_steps") or []
    if not steps:
        return state

    idx = int(state.get("current_step") or 0)
    if idx < 0 or idx >= len(steps):
        return state

    step = steps[idx]
    # Last observed result
    result_text = (state.get("result") or "").strip()

    # Simple heuristic to detect failure
    error_signals = ("error", "failed", "exception", "traceback", "失败", "错误")
    success = bool(result_text) and not any(sig in result_text.lower() for sig in error_signals)

    # If no tool was invoked and planner output directly, also consider success
    # Use executed_tool flag from graph layer instead of transient tool_call
    used_tool = bool(state.get("executed_tool"))
    if not used_tool and result_text:
        success = True

    if success:
        step["status"] = "done"
        steps[idx] = step
        state["plan_steps"] = steps
        state["current_step"] = idx + 1
        state["plan_needs_revision"] = False
    else:
        step["status"] = "blocked"
        step["notes"] = (step.get("notes") or "") + (" | " if step.get("notes") else "") + "blocked by last result"
        steps[idx] = step
        state["plan_steps"] = steps
        state["plan_needs_revision"] = True

    return state

