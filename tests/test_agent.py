"""Agent client tests verifying multi-step LangGraph orchestration and MCP tool execution."""

import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from langchain_core.messages import AIMessage

from agent import MCPAgentClient, detect_llm_mode


@pytest.mark.asyncio
async def test_agent_dynamic_tool_discovery():
    """Verify that the agent dynamically discovers tools from the MCP server at runtime."""
    agent = MCPAgentClient()
    await agent.initialize()

    # Must discover tools dynamically, not hardcoded
    assert len(agent.tools) >= 5
    discovered_names = list(agent.tools_by_name.keys())
    assert "search_docs" in discovered_names
    assert "fetch_json" in discovered_names
    assert "read_csv" in discovered_names
    assert "summarize" in discovered_names
    assert "word_frequency" in discovered_names


@pytest.mark.asyncio
async def test_agent_deterministic_demo_calls_at_least_three_tools():
    """Test the deterministic agent demo calls >= 3 tools and produces a non-empty final answer."""
    agent = MCPAgentClient()
    task = (
        "Search the docs for MCP tool-calling, fetch the sample JSONPlaceholder post #1 "
        "via fetch_json, summarize a paragraph with the summarize tool, and produce a combined answer."
    )

    state = await agent.run(task)

    # 1. Assert it calls >= 3 tools
    assert len(state["steps_completed"]) >= 3, f"Expected >= 3 steps, got {state['steps_completed']}"
    assert len(state["tool_results"]) >= 3, f"Expected >= 3 tool results, got {list(state['tool_results'].keys())}"

    # 2. Check specific tools were invoked
    assert "search_docs" in state["tool_results"]
    assert "fetch_json" in state["tool_results"]
    assert "summarize" in state["tool_results"]

    # 3. Assert all executed tools succeeded
    for tool_name, result in state["tool_results"].items():
        assert result["status"] == "success"
        assert len(result["output"].strip()) > 0

    # 4. Assert it produces a non-empty final answer
    assert state["final_answer"] is not None
    assert len(state["final_answer"].strip()) > 150
    assert "Multi-Step Task Execution Report" in state["final_answer"]
    assert "search_docs" in state["final_answer"]
    assert "summarize" in state["final_answer"]


@pytest.mark.asyncio
async def test_agent_executes_csv_and_frequency_tools():
    """Test that the agent can route tasks specifying other discovered MCP tools (read_csv, word_frequency)."""
    agent = MCPAgentClient()
    task = "Read the sales csv data and calculate word frequency of keywords."

    state = await agent.run(task)

    assert "read_csv" in state["tool_results"]
    assert "word_frequency" in state["tool_results"]
    assert len(state["final_answer"]) > 100
    assert "sales.csv" in state["tool_results"]["read_csv"]["output"]


@pytest.mark.asyncio
async def test_agent_graph_with_mock_llm():
    """Test that the LangGraph agent routes correctly when an LLM is configured."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock()
    # Mock planning response
    mock_llm.ainvoke.side_effect = [
        AIMessage(content="Step 1: search docs. Step 2: fetch JSON. Step 3: summarize context."),
        AIMessage(content="Synthesized LLM Answer: MCP tools successfully orchestrated through LangGraph.")
    ]

    agent = MCPAgentClient()
    with patch("agent.detect_llm_mode", return_value=("llm_openai", mock_llm)):
        await agent.initialize()
        assert agent.mode == "llm_openai"

        task = "Search docs for MCP, fetch json from https://jsonplaceholder.typicode.com/posts/1, and summarize."
        state = await agent.run(task)

        assert len(state["steps_completed"]) >= 3
        assert state["execution_mode"] == "llm_openai"
        assert "Synthesized LLM Answer" in state["final_answer"]
        assert mock_llm.ainvoke.call_count == 2
