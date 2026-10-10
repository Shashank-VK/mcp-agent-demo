# Model Context Protocol (MCP) Overview

The Model Context Protocol (MCP) is an open standard designed to enable secure, two-way communication between AI applications (clients) and external data sources or tools (servers). Rather than building proprietary, one-off integrations for every database, API, and file system, MCP establishes a unified interface based on JSON-RPC 2.0.

## Core Architecture

An MCP system consists of two primary participants:
1. **MCP Client**: An AI application, agent runtime, or IDE (such as Claude Desktop, Antigravity, or LangGraph agents) that requests data, executes actions, and controls agentic reasoning loops.
2. **MCP Server**: A lightweight service that exposes specific capabilities to clients. These capabilities fall into three primary categories:
   - **Tools**: Executable functions that allow the model to take actions or query external systems (e.g., executing SQL, querying search engines, or calling REST APIs).
   - **Resources**: Structured, readable data artifacts such as files, database tables, or real-time application logs that provide context to the LLM.
   - **Prompts**: Pre-engineered interaction templates and workflows provided directly by the server to guide user-model interactions.

## Protocol Lifecycle

The client and server begin with an initialization handshake where protocol versions, client capabilities, and server features are negotiated. Once connected, the client dynamically discovers available tools and resources via standard listing RPC calls (`tools/list`, `resources/list`). The agent runtime can then invoke tools using `tools/call`, receiving structured results and error codes without hardcoding tool schemas into the client code.
