"""Optional planners for different backends (Ollama, custom Azure API).

Each planner reads the conversation state and produces a plan and an
optional tool_call. They are drop-in replacements for `simple_planner`.
"""

from __future__ import annotations

import os
from typing import Callable, List
import copy
from ollama import Client

import requests
from goob_ai.nodes import simple_actor, simple_router
from langgraph.graph import END, StateGraph
from goob_ai.types_new import AgentState, Message
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
    if not rhs or rhs.lower().startswith("none"):
        return plan, None
    try:
        import shlex
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
            # 去除可能的尾随标点
            last_k = k
            v = v.rstrip(",;)]")
            args[k.strip()] = v.strip()
        else:  
            if last_k:
                v = token.rstrip(",;)]")
                args[last_k] = args.get(last_k, "") + " " + v.strip()
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
    model = os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b")
    # host = os.environ.get("OLLAMA_HOST", "").rstrip("/")
    # model = os.environ.get("OLLAMA_MODEL", "deepseek-v3.1:671b-cloud")
    print(host, model)
    messages = state.get("messages", [])
    user_context = _build_supervised_prompt(messages)
    tools = state.get("tool_names", []) or []
    tools_hint = ", ".join(sorted(tools)) if tools else "echo, none"
    docs = state.get("tool_docs", {}) or {}
    docs_items = [f"- {k}: {v}".strip() for k, v in sorted(docs.items()) if k in tools and v]
    docs_block = "\n".join(docs_items) if docs_items else ""
    system_prompt = (
        "You are the planner of a Reason+Act agent.\n" +
        "Policy:\n"+
        "- It is limit to up to 25 Actions per conversation. Always respond be exceed the limit.\n"+
        "- Think silently to decide whether a tool is needed; never reveal thoughts.\n"+
        "- Do not use your own dataset or knowledge; rely solely on tools when needed.\n"+
        "- If a tool is needed, output the next action succinctly; the system will execute the tool and return results later.\n"+
        "- If no tool is needed, output the final user-facing answer directly.\n"+
        "- When invoking arxiv_search, ALWAYS include max_results and set max_results=10 unless the user specified a different number.\n"+
        "Output exactly two lines ONLY following the format below:\n"+
        "plan: <final answer if you have a response to say to the Human, or if you do not need to use a tool; otherwise one-sentence next action>\n"+
        f"tool: <one of {tools_hint} or none> [args as key=value, space-separated]\n"+
        ("Tool guidance:\n" + docs_block + "\n" if docs_block else "")+
        "Constraints: no extra lines or markdown; at most one tool; if tool=none, 'plan' must be the exact final answer.\n"
    )
    prompt = f"{system_prompt}\n{user_context}\n"

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
    sys_prompt = (
        "You are the planner component of an agentic Reason+Act system.\n"
        "Role: Observe the conversation history below, decide whether a tool is required, "
        "and output either a final user-facing response or a single next tool action.\n"
        "Behavioral rules:\n"
        "- Do not use your own dataset or knowledge; rely solely on tools when needed.\n"
        "- This is an agentic planner: perform deliberation internally (DO NOT reveal chain-of-thought).\n"
        "- If a tool is required, produce a concise one-sentence plan that describes the next action, then a tool line.\n"
        "- If no tool is required, produce the exact final answer to send to the user on the 'plan' line and set tool=none.\n"
        "- Never output extra explanation, analysis, or markdown — only the two required lines in the exact format.\n"
        "- When invoking arxiv_search, ALWAYS include max_results and set max_results=10 unless the user specified a different number.\n"
        "Output format (exactly TWO lines):\n"
        "plan: <final answer if no tool is needed OR a one-sentence next action if you want the system to call a tool>\n"
        f"tool: <one of {tools_hint} or none> [args as key=value, space-separated]\n"
        + ("Tool guidance:\n" + docs_block + "\n" if docs_block else "")
        + "Constraints: at most one tool; if tool=none then 'plan' must be the exact final reply to the user.\n"
    )
    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model=deployment,  # deployment name
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_context},
            ],
            temperature=0.0,
            max_tokens=4096,
        )
        out_text = ""
        try:
            out_text = (response.choices[0].message.content or "") if response and response.choices else ""
        except Exception:
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
                    {"role": "user", "content": user_context},
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
        name: One of "ollama" or "azure" (case-insensitive).

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
    if key == "azure_sdk":
        return azure_sdk_planner
    raise ValueError(f"Unsupported planner: {name}")




