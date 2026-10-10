"""End-to-end demonstration script running an autonomous LangGraph agent connected to an MCP server.

This script executes a multi-step task end-to-end:
1. Connects to the local MCP server over stdio transport.
2. Dynamically discovers available tools via the Model Context Protocol.
3. Executes a multi-step plan through a LangGraph StateGraph (plan -> act -> synthesize).
4. Invokes at least 3 distinct MCP tools (search_docs, fetch_json, summarize).
5. Prints the final synthesized answer.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from agent import MCPAgentClient


async def run_demo():
    print("=" * 78)
    print("  MCP AGENT DEMO: Model Context Protocol + LangGraph Orchestration")
    print("=" * 78)

    task_description = (
        "Search the docs for MCP tool-calling, fetch the sample JSONPlaceholder post #1 "
        "via fetch_json, summarize a paragraph with the summarize tool, and produce a combined answer."
    )

    print(f"\n[DEMO OBJECTIVE]:\n  {task_description}\n")

    t0 = time.perf_counter()
    agent = MCPAgentClient()

    print("[STEP 1/3] Connecting to MCP Server via stdio and discovering tools...")
    await agent.initialize()
    print(f"  -> Discovered {len(agent.tools)} MCP tools: {list(agent.tools_by_name.keys())}")
    print(f"  -> Active Execution Mode: {agent.mode.upper()}")

    print("\n[STEP 2/3] Executing LangGraph StateGraph (plan -> act -> synthesize)...")
    final_state = await agent.run(task_description)
    elapsed = time.perf_counter() - t0

    print(f"  -> Graph execution completed in {elapsed:.2f} seconds.")
    print(f"  -> Completed Steps ({len(final_state['steps_completed'])}): {final_state['steps_completed']}")
    print(f"  -> Tools Invoked ({len(final_state['tool_results'])}): {list(final_state['tool_results'].keys())}")

    print("\n[STEP 3/3] Final Synthesized Answer Produced by LangGraph:")
    print("=" * 78)
    print(final_state["final_answer"])
    print("=" * 78)


def main():
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
