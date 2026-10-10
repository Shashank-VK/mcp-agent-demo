# MCP Agent Demo (`mcp-agent-demo`)

An end-to-end Python implementation demonstrating the **Model Context Protocol (MCP)** using the official `mcp` SDK and a stateful **LangGraph** agent client. The agent communicates with the MCP server over standard input/output (`stdio`), dynamically discovers available tools at runtime without hardcoded schemas, and executes multi-step analytical tasks through an explicit state graph.

---

## Architecture Overview

The system consists of an MCP Server exposing data and text analysis capabilities and an autonomous LangGraph Agent Client communicating across bidirectional JSON-RPC 2.0 pipes:

```text
+-----------------------------------------------------------------------------------+
|                                 LANGGRAPH AGENT                                    |
|                                                                                   |
|   +-------------------+      +-------------------+      +---------------------+   |
|   |     PLAN NODE     | ---> |     ACT NODE      | ---> |   SYNTHESIZE NODE   |   |
|   |  Task breakdown   |      | MCP tool executor |      | Multi-tool synthesis|   |
|   +-------------------+      +-------------------+      +---------------------+   |
|                                        |                                          |
+----------------------------------------|------------------------------------------+
                                         |  JSON-RPC 2.0 (stdio)
                                         |  - tools/list (Runtime Introspection)
                                         |  - tools/call (Typed Tool Execution)
+----------------------------------------|------------------------------------------+
|                                        v                                          |
|                                 MCP SERVER (stdio)                                |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   | Exposed MCP Tools:                                                        |   |
|   |   * search_docs    : Pure-Python BM25 search over data/docs/*.md          |   |
|   |   * fetch_json     : Validated HTTP/HTTPS JSON client with timeout        |   |
|   |   * read_csv       : Sandboxed local CSV reader for data/sales.csv        |   |
|   |   * summarize      : Extractive sentence TF summarizer                    |   |
|   |   * word_frequency : Term frequency Counter analyzer                      |   |
|   +---------------------------------------------------------------------------+   |
|                                        |                                          |
|                                        v                                          |
|                       +---------------------------------+                         |
|                       |  Local Corpus & Sandboxed Data  |                         |
|                       |  - data/docs/*.md (7 documents) |                         |
|                       |  - data/sales.csv (24 rows)     |                         |
|                       +---------------------------------+                         |
+-----------------------------------------------------------------------------------+
```

---

## Exposed MCP Server Tools

All tools are exposed over `stdio` transport using `FastMCP` from the official Python MCP SDK. Errors are handled gracefully and returned as structured error strings rather than crashing the protocol session.

| Tool Name | Signature | Description |
| :--- | :--- | :--- |
| `search_docs` | `(query: str, top_k: int = 3) -> str` | Performs pure-Python BM25 keyword relevance scoring over markdown documents in `data/docs/`. Returns ranked snippets with titles and scores. |
| `fetch_json` | `(url: str) -> str` | Queries an HTTP(S) endpoint with a 10s timeout, verifies valid JSON payload, formats response, and truncates large payloads to 2000 characters. Rejects non-HTTP(S) schemas. |
| `read_csv` | `(path: str, max_rows: int = 20) -> str` | Reads local tabular data from the project `data/` directory. Enforces path sandboxing to block directory traversal attacks (e.g., `../../etc/passwd`). |
| `summarize` | `(text: str) -> str` | Extractive text summarizer: splits input into sentences, scores each sentence using word frequency (TF) after stopword filtering, and returns the top 3 sentences. |
| `word_frequency` | `(text: str, top_n: int = 10) -> str` | Computes term frequency statistics from text input, returning a sorted JSON array of word/count pairs. |

---

## Installation & Setup

### Requirements
- Linux or macOS
- Python 3.12+

### Install Dependencies
Install dependencies directly using `pip`:

```bash
pip install --user --break-system-packages -r requirements.txt
```

Or within a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### 1. Running the MCP Server Standalone
You can start the MCP server directly via stdio. It listens on `stdin` and writes JSON-RPC 2.0 messages to `stdout`:

```bash
python3 server.py
```

### 2. Running the End-to-End Demo
The demo script (`demo.py`) initializes the `MCPAgentClient`, connects to `server.py` as a subprocess over stdio, discovers all tools dynamically, and executes a multi-step task across a 3-node LangGraph:

```bash
python3 demo.py
```

#### Demo Output Highlights:
- **Runtime Discovery**: Finds all 5 tools (`search_docs`, `fetch_json`, `read_csv`, `summarize`, `word_frequency`).
- **Tool Invocations**:
  1. `search_docs` finds relevant architecture documentation for `"MCP tool-calling"`.
  2. `fetch_json` fetches and validates JSON from `https://jsonplaceholder.typicode.com/posts/1`.
  3. `summarize` distills an architectural summary of MCP agent workflows.
- **Synthesized Result**: Consolidates all 3 tool outputs into a coherent markdown report.

### 3. Execution Modes (LLM vs Deterministic)
The agent dynamically inspects the environment at runtime:
- **LLM Mode**: If `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is detected, the agent uses `ChatOpenAI` (`gpt-4o-mini`) or `ChatAnthropic` (`claude-3-5-sonnet-20241022`) to drive planning and synthesis.
- **Deterministic Mode**: If no API keys are configured, the agent logs an informative notice and runs in deterministic execution mode through the exact same LangGraph `StateGraph` and MCP stdio tools. No simulated or mock LLM claims are made.

---

## Running the Test Suite

The project includes unit, protocol integration, and agent orchestration tests via `pytest`:

```bash
pytest
```

Or using `python3 -m pytest`:

```bash
python3 -m pytest -v
```

### Test Coverage Breakdown
- `tests/test_tools.py` (23 tests): Unit tests verifying tool logic directly (BM25 ranking order, `fetch_json` protocol rejection and timeout handling, `read_csv` sandbox enforcement and column parsing, `summarize` top sentence selection, `word_frequency` counts).
- `tests/test_server.py` (6 tests): Integration tests that launch the real `server.py` as a child process, negotiate protocol initialization over stdio, verify tool listing, and invoke each tool over MCP.
- `tests/test_agent.py` (4 tests): Verifies dynamic tool discovery, ensures the deterministic agent demo executes $\ge 3$ distinct MCP tools, tests custom CSV routing, and validates LLM path orchestration with mock models.

---

## Scope & Limitations

This project is intentionally honest about its capabilities and constraints:
1. **Simplified BM25 Implementation**: The document search engine implements pure-Python BM25 scoring with standard Robertson-Spärck Jones IDF and stopword filtering. It does not include Porter stemming or lemmatization, so morphological variations (e.g., `connect` vs `connecting`) are scored as distinct tokens.
2. **Deterministic Fallback**: When API keys are not supplied, the agent uses an intent-matching planner to build the tool call sequence. It executes real MCP tools and real graph nodes, but does not simulate non-existent generative model behavior.
3. **Data Directory Sandboxing**: `read_csv` restricts all access strictly to the project's `data/` directory using `Path.resolve()` and `Path.relative_to()`. Any attempt to access files outside this boundary is rejected with an explicit security error.
4. **Subprocess Environment Propagation**: Stdio subprocesses inherit parent proxy environment variables (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`) to support sandboxed enterprise proxy environments.
