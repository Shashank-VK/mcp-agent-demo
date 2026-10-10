# Evaluation and Testing Strategies for MCP Tools and Agents

Ensuring reliability in agentic systems requires systematic testing across multiple abstraction boundaries.

## Three-Tier Testing Hierarchy

1. **Unit Testing Tool Functions Directly**:
   Individual tool functions should be isolated from network latency and transport overhead. Test core scoring algorithms (e.g., BM25 keyword matching), error responses for illegal file paths, and extractive summarization logic against synthetic fixtures.
2. **Protocol Integration Testing Over Stdio**:
   Spin up the MCP server as a real subprocess and issue JSON-RPC requests across `stdin`/`stdout`. This validates message serialization, schema compatibility, and initialization handshakes as specified by the MCP protocol.
3. **End-to-End Agent Verification**:
   Execute the full LangGraph state machine from user intent to synthesized output. Verify that the agent transitions across all expected nodes, executes at least the planned number of tools, and produces a validated output structure. Deterministic execution modes ensure CI/CD suites run reliably without external API dependencies.
