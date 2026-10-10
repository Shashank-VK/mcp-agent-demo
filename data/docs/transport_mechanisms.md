# MCP Transport Mechanisms: Stdio vs SSE and HTTP

The Model Context Protocol decouples communication semantics from the underlying transport layer. MCP defines standard wire transports to support diverse deployment environments, ranging from isolated local CLI tools to distributed enterprise cloud microservices.

## Stdio Transport

The Standard Input/Output (`stdio`) transport is the standard for local agent execution. In this model:
- The MCP client spawns the MCP server process as a child subprocess.
- Messages are formatted as JSON-RPC 2.0 payloads delimited by newlines.
- The client writes JSON-RPC requests to the subprocess's standard input (`stdin`) and reads responses from standard output (`stdout`).
- Diagnostic and debug messages from the server are routed to standard error (`stderr`) to prevent corrupting the JSON-RPC data stream.

Key advantages of stdio transport include process isolation, zero network port binding, instant lifecycle cleanup when the parent agent exits, and complete local security without network exposure.

## Server-Sent Events (SSE) and Streamable HTTP

For remote or networked server architectures, MCP specifies HTTP-based transports:
- **Server-Sent Events (SSE)**: The client establishes a persistent HTTP GET connection to receive real-time streaming events and notifications from the server, while sending requests via HTTP POST endpoints.
- **Streamable HTTP**: Modern microservices use bidirectional streaming HTTP/2 or HTTP/3 channels to maintain high-throughput tool execution pipelines.

When choosing a transport, local agent tools that interact with local file systems or deterministic algorithms should favor stdio for simplicity and speed.
