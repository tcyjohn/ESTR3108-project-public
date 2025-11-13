"""Optional planners for different backends (Ollama, custom Azure API).

Each planner reads the conversation state and produces a plan and an
optional tool_call. They are drop-in replacements for `simple_planner`.
"""

from __future__ import annotations

import os
from typing import Callable, List
from ollama import Client

import requests
from goob_ai.nodes import simple_actor, simple_router
from langgraph.graph import END, StateGraph
from goob_ai.types_new import AgentState, Message, PlanStep
from goob_ai.toolkit import default_tools

def simple_planner(state: AgentState) -> AgentState:
    """A deterministic planner without external LLMs.

    If the last user message以 "echo" 开头，则调用 echo 工具；否则直接回复。

    Args:
        state: Current agent state.

    Returns:
        Updated agent state with plan and optional tool_call.
    """

    messages = state.get("messages", [])
    last = messages[-1]["content"].strip() if messages else ""
    lower = last.lower()
    if lower.startswith("echo"):
        payload = last[len("echo"):].strip()
        state["plan"] = "Use echo tool"
        state["tool_call"] = {"name": "echo", "args": {"text": payload}}
    else:
        state["plan"] = "Respond directly without tools"
        state["tool_call"] = None
    # print(state)
    return state


def _build_supervised_prompt(messages: List[Message]) -> str:
    """Construct a compact supervision prompt from messages.

    Args:
        messages: Conversation messages in the current state.

    Returns:
        A single string joining roles and contents to guide the planner.
    """

    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _parse_plan_and_tool(output: str) -> tuple[str, dict | None]:
    """Parse planner text into a `(plan, tool_call)` pair.

    Args:
        output: Raw text output from the LLM backend.

    Returns:
        A tuple of plan string and optional tool_call mapping.
    """

    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    plan_line = next((ln for ln in lines if ln.lower().startswith("plan:")), "")
    tool_line = next((ln for ln in lines if ln.lower().startswith("tool:")), "")
    plan = plan_line.split(":", 1)[1].strip() if plan_line else "Respond directly"

    if not tool_line:
        return plan, None

    rhs = tool_line.split(":", 1)[1].strip()
    # Remove the "args=" prefix if present anywhere before parsing.
    if "args=" in rhs:
        rhs = rhs.replace("args=", "", 1).strip()
    if not rhs or rhs.lower().startswith("none"):
        return plan, None
    try:
        import shlex
        import platform

        if platform.system()=='Windows':
            def arg_split(args, platform=os.name):
                return [a[1:-1].replace('""', '"') if a[0] == a[-1] == '"' else a
                        for a in (shlex.split(args, posix=False) if platform == 'nt' else shlex.split(args))]
            parts = arg_split(rhs)
        else:
            parts = shlex.split(rhs)
    except Exception:
        parts = [p.strip() for p in rhs.replace(",", " ").split() if p.strip()]
        
    if not parts:
        return plan, None
    
    name = parts[0]
    args: dict[str, str] = {}
    last_k = ""
    for token in parts[1:]:
        if "=" in token:
            k, v = token.split("=", 1)
            k = k.lstrip("[(")
            last_k = k
            # Only strip quotes and commas/semicolons, NOT parentheses (needed for expressions)
            v = v.strip().strip("\"'").rstrip(",;")
            args[k.strip()] = v
        else:  
            if last_k:
                # Only strip quotes and commas/semicolons, NOT parentheses
                v = token.strip().strip("\"'").rstrip(",;")
                args[last_k] = args.get(last_k, "") + " " + v
    return plan, {"name": name, "args": args}


def ollama_planner(state: AgentState) -> AgentState:
    """Planner that calls a local Ollama model via HTTP.

    Environment variables:
        OLLAMA_HOST: Host for Ollama (default: http://localhost:11434)
        OLLAMA_MODEL: Model tag (e.g., qwen2.5:7b-instruct-q4_K_M)

    Args:
        state: Current agent state.

    Returns:
        Updated agent state with plan and optional tool_call.
    """

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    #model = os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b")
    # host = os.environ.get("OLLAMA_HOST", "").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "deepseek-v3.1:671b-cloud")
    print(host, model)
    messages = state.get("messages", [])
    user_context = _build_supervised_prompt(messages)
    tools = state.get("tool_names", []) or []
    tools_hint = ", ".join(sorted(tools)) if tools else "echo, none"
    docs = state.get("tool_docs", {}) or {}
    docs_items = [f"- {k}: {v}".strip() for k, v in sorted(docs.items()) if k in tools and v]
    docs_block = "\n".join(docs_items) if docs_items else ""

    # Known facts accumulated by assessor (if any)
    facts_text = state.get("facts_text") or ""
    facts_block = f"Known facts: {facts_text}\n" if facts_text else ""
    # recent action context to discourage repetition
    last_name = state.get("last_tool") or ""
    last_args = state.get("last_tool_args") or {}
    last_obs = (state.get("last_observation") or "")[:600]
    attempts = state.get("tool_attempts") or {}
    last_attempts = int(attempts.get(last_name, 0)) if last_name else 0

    # Build recent actions block (last up to 3 actions for continuity)
    recent_block = ""
    try:
        history = state.get("action_history") or []
        if history:
            tail = history[-3:]
            lines = ["Recent actions:"]
            for i, rec in enumerate(tail, start=max(1, len(history)-len(tail)+1)):
                t = rec.get("tool", "")
                a = rec.get("args", {})
                o = (rec.get("observation", "") or "")[:280]
                lines.append(f"- #{i}: tool={t} args={a} obs={o}")
            lines.append("Guidance: Do NOT restart completed subgoals; build on these observations. If unhelpful, choose a different tool.")
            recent_block = "\n".join(lines) + "\n"
        elif last_name:
            recent_block = (
                f"Recent action: {last_name} args={last_args}\n"
                f"Observation (truncated): {last_obs}\n"
                f"Attempts for {last_name}: {last_attempts}\n"
                "Guidance: Do NOT repeat the same tool with identical args. If unhelpful, choose a different tool.\n"
            )
    except Exception:
        recent_block = ""
    # Build recent actions block (last up to 3 actions)
    

    system_prompt = (
        "You are the hierarchical planner of a Reason+Act agent.\n"
        "Long-horizon policy:\n"
        "- Silently draft a multi-step plan (do NOT output it). Keep and adapt it internally.\n"
        "- On each turn, observe the latest messages and known facts and choose the NEXT step.\n"
        "- If the last step failed or is blocked, REVISE the remaining plan silently and choose a better next step.\n"
        "- Continue step-by-step until the goal is achieved, then output the final user-facing answer.\n"
        "Action policy:\n"
        "- Unless you are outputting the FINAL answer, you MUST select exactly ONE tool for the next step.\n"
        "- If tool=none, the 'plan' line MUST be the exact final answer text (do NOT write meta instructions like 'combine the facts' or 'answer the question').\n"
        "- Prefer using the Known facts; if the known facts already include the required variables to solve, you MUST set tool=none and output the final answer.\n"
        "- If you see 'calculator_result=<number>' in Known facts, that IS the final calculation result; do NOT call calculator again.\n"
        "- If you see 'web_search_result=<text>' in Known facts, that IS the search answer; do NOT call web_search again.\n"
        "- You are encouraged to use web_search before using wiki_retrieve.\n"
        "Output EXACTLY two lines (no extra text):\n"
        "plan: <final answer if done, otherwise one-sentence next action>\n"
        f"tool: <one of {tools_hint} or none> key=value, space-separated\n"
        + ("STRICTLY follow the tool guidance and provide tool name followed by arguments ONLY:\n" + docs_block + "\n" if docs_block else "")
        + "You are encouraged to set max_results more than 1 when using web_search, wiki_retrieve, arxiv_search tools.\n"
    )
    prompt = f"{system_prompt}\n{facts_block}{recent_block}{user_context}\n"

    # Ollama client library usage (if available) with fallback to requests
    out_text = ""
    try:
        try:
            client = Client(base_url=host)  
        except TypeError:
            client = Client(host)  # fallback constructor style

        # try to call a generate helper; adapt to client API variations
        try:
            resp = client.generate(model=model, prompt=prompt, stream=False, options={"temperature": 0.0, "num_ctx": 4096})
            if isinstance(resp, dict):
                out_text = resp.get("response") or resp.get("text") or resp.get("content", "") or ""
            else:
                out_text = getattr(resp, "response", "") or getattr(resp, "text", "") or ""
        except Exception:
            # some client versions expose a completions-like API
            try:
                resp = client.create_completion(model=model, prompt=prompt, temperature=0.0, max_tokens=256)
                if isinstance(resp, dict):
                    out_text = resp.get("response") or resp.get("text") or ""
                else:
                    out_text = getattr(resp, "text", "") or ""
            except Exception:
                out_text = ""
    except Exception:
        out_text = ""

    # If client didn't produce text, fall back to the HTTP endpoint
    if not out_text:
        try:
            resp = requests.post(
                f"{host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_ctx": 4096},
                },
                timeout=60,
            )
            resp.raise_for_status()
            j = resp.json()
            # flexible extraction from possible response shapes
            out_text = j.get("response") or j.get("text") or j.get("content", "") or ""
            if not out_text and "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                # choices may contain dicts with 'message' or 'content' or 'text'
                c = j["choices"][0]
                if isinstance(c, dict):
                    out_text = c.get("message", {}).get("content") or c.get("text") or c.get("content", "") or ""
        except Exception:
            print("Ollama planner request failed")
            state["plan"] = "Respond directly (ollama planner error)"
            state["tool_call"] = None
            return state
    print("Ollama planner output:", out_text)
    plan, tool_call = _parse_plan_and_tool(out_text)
    state["plan"] = plan
    state["tool_call"] = tool_call
    # print(state)
    return state


def azure_sdk_planner(state: AgentState) -> AgentState:
    """Planner using OpenAI v1 AzureOpenAI SDK (chat.completions).

    Required environment variables:
        AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_API_KEY,
        AZURE_OPENAI_DEPLOYMENT (used as model name)

    Returns:
        Updated agent state with plan and optional tool_call.
    """

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "")
    api_key = os.environ.get("AZURE_API_KEY", "")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    if not (endpoint and api_version and api_key):
        state["plan"] = "Respond directly (planner backend not configured)"
        state["tool_call"] = None
        return state

    # Import locally to avoid hard dependency when unused
    try:
        from openai import AzureOpenAI  # type: ignore
    except Exception:
        state["plan"] = "OpenAI SDK not installed. Fail to compute plan."
        state["tool_call"] = None
        return state

    messages = state.get("messages", [])
    user_context = _build_supervised_prompt(messages)
    tools = state.get("tool_names", []) or []
    tools_hint = ", ".join(sorted(tools)) if tools else "echo, none"
    docs = state.get("tool_docs", {}) or {}
    docs_items = [f"- {k}: {v}".strip() for k, v in sorted(docs.items()) if k in tools and v]
    docs_block = "\n".join(docs_items) if docs_items else ""
    # recent action context to discourage repetition
    last_name = state.get("last_tool") or ""
    last_args = state.get("last_tool_args") or {}
    last_obs = (state.get("last_observation") or "")[:600]
    attempts = state.get("tool_attempts") or {}
    last_attempts = int(attempts.get(last_name, 0)) if last_name else 0

    recent_block = ""
    try:
        history = state.get("action_history") or []
        if history:
            tail = history[-3:]
            lines = ["Recent actions:"]
            for i, rec in enumerate(tail, start=max(1, len(history)-len(tail)+1)):
                t = rec.get("tool", "")
                a = rec.get("args", {})
                o = (rec.get("observation", "") or "")[:280]
                lines.append(f"- #{i}: tool={t} args={a} obs={o}")
            lines.append("Guidance: Do NOT restart completed subgoals; build on these observations.")
            recent_block = "\n".join(lines) + "\n"
        elif last_name:
            recent_block = (
                f"Recent action: {last_name} args={last_args}\n"
                f"Observation (truncated): {last_obs}\n"
                f"Attempts for {last_name}: {last_attempts}\n"
                "Guidance: Do NOT repeat the same tool with identical args. If unhelpful, modify args (e.g., refine query) or choose a different tool.\n"
            )
    except Exception:
        recent_block = ""

    sys_prompt = (
        "You are the hierarchical planner of an agentic Reason+Act system.\n"
        "Long-horizon policy:\n"
        "- Silently form a multi-step plan from the user's goal, keep it internal, and adapt after each observation.\n"
        "- On each turn, pick the NEXT step only; if blocked, silently revise remaining steps and choose a better next one.\n"
        "- Stop when the goal is accomplished and output the final user-facing answer.\n"
        "Behavioral rules:\n"
        "- Unless you are outputting the FINAL answer, you MUST choose exactly ONE tool for the next step.\n"
        "- If tool=none, the 'plan' line MUST be the exact final answer text (no meta like 'combine facts').\n"
        "- Prefer using the Known facts (provided in the user message). If the known facts already include the required variables to solve, you MUST set tool=none and output the final answer.\n"
        "- If you see 'calculator_result=<number>' in Known facts, that IS the final calculation result; do NOT call calculator again.\n"
        "- If you see 'web_search_result=<text>' in Known facts, that IS the search answer; do NOT call web_search again.\n"
        "Subgoal policy:\n"
        "- Decompose into subgoals and complete them in order. E.g., (1) identify the entity; (2) get the attribute (capital).\n"
        "- Do NOT proceed to later subgoals until prior ones are satisfied with credible evidence.\n"
        "- You do NOT need to double check or confirm the result of the tool call, just use it as is.\n"
        "Output format (EXACTLY TWO lines):\n"
        "plan: <final answer if done OR a one-sentence next action>\n"
        f"tool: <one of {tools_hint} or none> key=value, space-separated\n"
        + ("Tool guidance:\n" + docs_block + "\n" if docs_block else "")
        + "Constraints: at most one tool; if tool=none then 'plan' MUST be the final user reply.\n"
    )
    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=api_key,
        )
        # Build user payload with optional facts
        ft = state.get("facts_text") or ""
        user_payload = (f"Known facts: {ft}\n{recent_block}{user_context}") if ft else f"{recent_block}{user_context}"
        response = client.chat.completions.create(
            model=deployment,  # deployment name
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.0,
            max_tokens=4096,
        )
        out_text = ""
        try:
            out_text = (response.choices[0].message.content or "") if response and response.choices else ""
        except Exception as e:
            print("Azure planner response parsing failed ", e)
            out_text = ""
    except Exception:
        try: # Fallback to direct HTTP call
            response = requests.post(
                url=f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}",
                headers={
                    "api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={"messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_payload},
                ],
                    "temperature": 0.0, "max_tokens": 256},
                timeout=60
            )
            response.raise_for_status()
            j = response.json()
            if isinstance(j, dict):
                if "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                    c = j["choices"][0]
                    if isinstance(c, dict):
                        out_text = (
                            c.get("message", {}).get("content")
                            or c.get("text")
                            or c.get("content", "")
                        ) or ""
                else:
                    out_text = j.get("response") or j.get("text") or j.get("content", "") or ""
        except Exception as e:
            print(e)
            out_text = ""
    if not out_text:
        state["plan"] = "Respond directly (azure planner error)"
        state["tool_call"] = None
        return state
    
    print("Azure planner output:", out_text)
    plan, tool_call = _parse_plan_and_tool(out_text)
    state["plan"] = plan
    state["tool_call"] = tool_call
    return state
    
def select_planner(name: str) -> Callable[[AgentState], AgentState]:
    """Return a planner function by name.

    Args:
        name: One of "simple", "ollama", "azure_sdk" (also accepts alias "azure").

    Returns:
        A callable planner function.

    Raises:
        ValueError: If the name is not a supported planner.
    """

    key = name.strip().lower()
    if key == "simple":
        return simple_planner
    if key == "ollama":
        return ollama_planner
    if key in ("azure_sdk", "azure"):
        return azure_sdk_planner
    raise ValueError(f"Unsupported planner: {name}")




