# Goob AI - An Extensible Agent Framework with LangGraph

Goob AI is a minimal, yet powerful, scaffolding for building and prototyping autonomous agents using [LangGraph](https://github.com/langchain-ai/langgraph). It provides a clean, modular architecture that separates planning, acting, and tool use, making it easy to extend and customize.

## Features

- **Modular Agent Architecture**: Built on a `plan -> act -> assess` graph that cleanly separates agent reasoning from execution.
- **Pluggable Planners**: Easily switch between different planning backends by setting an environment variable.
  - `simple`: A basic, deterministic planner for simple use cases (no LLM required).
  - `ollama`: Integrates with local LLMs via Ollama.
  - `azure_sdk`: Uses the official OpenAI SDK to connect to Azure OpenAI models.
- **Rich Built-in Toolkit**: Comes with a comprehensive set of tools for various tasks:
  - **Web & Academic Search**: `web_search` (Tavily), `arxiv_search`, and `wiki_retrieve`.
  - **Data Analysis**: `calculator`, `analyze_table` (CSVs via Pandas).
  - **Multimedia Analysis**: `analyze_image`, `analyze_remote_image` (with OCR via EasyOCR), `analyze_video`, `analyze_video_by_chapter` (via yt-dlp).
- **Interactive Web UI**: A [Gradio](https://www.gradio.app/) application (`app.py`) for easy interaction and evaluation.
- **Extensible by Design**: Add new tools or planners with minimal effort.

## Project Structure

The project is organized into distinct modules for clarity and maintainability:

- `src/goob_ai/app.py`: The main Gradio application entry point.
- `src/goob_ai/graph.py`: Defines the agent's structure and flow using LangGraph.
- `src/goob_ai/nodes.py`: Contains the core logic for the agent's nodes (actor, assessor).
- `src/goob_ai/planners.py`: Implements the various planner backends (Ollama, Azure, etc.).
- `src/goob_ai/toolkit.py`: The registry for all available tools. Add your custom tools here.
- `src/goob_ai/types_new.py`: Defines the core data structures and types, including the agent's state.

## Getting Started

### 1. Prerequisites

- Python 3.10+
- A virtual environment manager like `venv` or `uv`.

### 2. Installation

Clone the repository and install the required dependencies. Using `pip` with the editable flag (`-e`) is recommended.

```bash
# It is recommended to create a virtual environment first
# python -m venv .venv && source .venv/bin/activate

pip install -e .
```

### 3. Configuration

The agent's behavior and tool access are configured via environment variables.

1.  **Copy the Example Environment File**:
    Create a `.env` file from the example (if one is provided) or create a new one.

    ```bash
    cp .env.example .env
    ```

2.  **Set API Keys**:
    Add the necessary API keys for the tools you intend to use to your `.env` file.

    ```env
    # For web_search tool
    TAVILY_API_KEY="your-tavily-api-key"

    # For azure_sdk planner
    AZURE_OPENAI_ENDPOINT="your-azure-openai-endpoint"
    AZURE_OPENAI_API_VERSION="2024-02-01"
    AZURE_API_KEY="your-azure-api-key"
    AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
    ```

3.  **Choose a Planner**:
    Set the `PLANNER` environment variable to select a planning strategy.

    ```bash
    # For a simple, deterministic planner (no LLM needed)
    export PLANNER=simple

    # For using a local Ollama model
    export PLANNER=ollama
    export OLLAMA_MODEL="qwen2:7b" # Specify your desired model
    export OLLAMA_HOST="http://localhost:11434" # Optional: if not default

    # For using Azure OpenAI
    export PLANNER=azure_sdk
    ```

### 4. Running the Application

Launch the Gradio web interface to start interacting with the agent.

```bash
python src/goob_ai/app.py
```

Navigate to the local URL provided in your terminal (usually `http://127.0.0.1:7860`).

## Usage

The Gradio interface provides two main tabs:

1.  **Single Input**: A chat interface for direct, interactive conversations with the agent. You can ask questions and upload images for analysis.
2.  **Full Evaluation & Submission**: A runner designed to evaluate the agent against a benchmark test suite from a Hugging Face Space. It fetches questions, runs the agent on all of them, and submits the results for scoring.

## How It Works

The agent operates on a cyclical graph defined in `graph.py`. Each cycle involves several steps:

1.  **Plan**: The selected planner (`simple`, `ollama`, or `azure_sdk`) analyzes the user request and the current state to decide the next action. It produces a short-term plan and, optionally, a tool to execute.
2.  **Act**: The `simple_actor` node executes the chosen tool with the specified arguments. If no tool is chosen, it prepares the final response.
3.  **Assess**: The `assess_step` node evaluates the result of the tool execution. It extracts key facts from the output and updates the agent's state and long-term plan progress.
4.  **Re-plan**: The graph loops back to the planner, which uses the updated state and assessment to decide the next step. This loop continues until the user's goal is met.

## How to Extend the Agent

### Adding a New Tool

1.  **Define the Tool Function**: Create a new function in `src/goob_ai/toolkit.py`. It should accept arguments and have a clear docstring, as this is used by the planner.
    ```python
    def my_new_tool(*, parameter: str) -> str:
        """My new tool; args: parameter. Example: tool: my_new_tool parameter="value" """
        # ... tool logic ...
        return f"Result for {parameter}"
    ```
2.  **Register the Tool**: Add the function to the `default_tools()` registry at the bottom of `src/goob_ai/toolkit.py`.
    ```python
    def default_tools() -> ToolRegistry:
        registry: Mapping[str, Tool] = {
            # ... existing tools ...
            "my_new_tool": my_new_tool,
        }
        return registry
    ```
The planner will automatically have access to the new tool on the next run.

### Adding a New Planner

1.  **Create the Planner Function**: In `src/goob_ai/planners.py`, create a new function that follows the `(state: AgentState) -> AgentState` signature.
2.  **Implement the Logic**: The function should process the `state` (especially `state["messages"]`) and return an updated state with a `plan` (string) and an optional `tool_call` (dictionary).
3.  **Register the Planner**: Add your new planner to the `select_planner` function at the bottom of `src/goob_ai/planners.py`.

## Running Tests

To run the test suite, use `pytest`.

```bash
pytest
```
