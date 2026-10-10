# Resource Management and Context Strategies in MCP

While tools represent active operations with side effects, resources in the Model Context Protocol represent passive data sources that can be read to provide rich context to the language model.

## URI-Based Resource Addressing

MCP uses standard Uniform Resource Identifiers (URIs) to identify resources across systems. Examples include:
- `file:///workspace/data/sales.csv` for local file system references.
- `postgres://analytics/tables/monthly_revenue` for relational database tables.
- `docs://mcp/spec/v1.0` for in-memory documentation trees.

## Subscriptions and Real-Time Updates

Servers can notify clients when resource contents change via the `notifications/resources/updated` method. This allows reactive agent workflows where the model recalculates answers or updates summaries whenever underlying files or database records are modified.

## Context Window Optimization

Because modern LLMs operate with bounded context windows, transferring entire databases or file systems into the model prompt is wasteful and slow. MCP servers employ search algorithms (such as BM25 and vector embeddings) or extraction tools (such as sentence summarizers) to distill high-value context before delivering it to the agent.
