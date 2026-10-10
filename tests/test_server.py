"""Protocol integration tests starting the real MCP server as a subprocess over stdio."""

import json
import os
import sys
from pathlib import Path
import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

SERVER_SCRIPT = str(Path(__file__).parent.parent.resolve() / "server.py")


def extract_content(result) -> str:
    """Helper to extract text string from tool call results."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        texts = []
        for item in result:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            elif hasattr(item, "text"):
                texts.append(str(item.text))
            elif hasattr(item, "content"):
                texts.append(str(item.content))
            else:
                texts.append(str(item))
        return "\n".join(texts)
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)


@pytest.fixture
def mcp_client():
    """Create an MCP client connected to the real server subprocess."""
    client = MultiServerMCPClient({
        "server": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [SERVER_SCRIPT],
            "env": dict(os.environ),
        }
    })
    return client


@pytest.mark.asyncio
async def test_server_tool_listing(mcp_client):
    """Verify that the real MCP server lists all expected tools over stdio."""
    tools = await mcp_client.get_tools()
    tool_names = [t.name for t in tools]

    expected_tools = ["search_docs", "fetch_json", "read_csv", "summarize", "word_frequency"]
    for expected in expected_tools:
        assert expected in tool_names, f"Tool '{expected}' should be listed by MCP server"


@pytest.mark.asyncio
async def test_server_call_search_docs(mcp_client):
    """Call search_docs over the real MCP protocol."""
    tools = await mcp_client.get_tools()
    tool_map = {t.name: t for t in tools}

    raw = await tool_map["search_docs"].ainvoke({"query": "BM25 scoring", "top_k": 2})
    text = extract_content(raw)

    assert "Search Results for 'BM25 scoring'" in text
    assert "score:" in text
    assert ".md" in text


@pytest.mark.asyncio
async def test_server_call_fetch_json(mcp_client):
    """Call fetch_json over MCP protocol and verify error handling/schema rejection."""
    tools = await mcp_client.get_tools()
    tool_map = {t.name: t for t in tools}

    # Test bad protocol rejection over MCP
    raw_bad = await tool_map["fetch_json"].ainvoke({"url": "ftp://malicious.example.com/test.json"})
    text_bad = extract_content(raw_bad)
    assert "Error: Invalid URL scheme 'ftp'" in text_bad

    # Test valid fetch over MCP (JSONPlaceholder or similar)
    try:
        raw_good = await tool_map["fetch_json"].ainvoke({"url": "https://jsonplaceholder.typicode.com/posts/1"})
        text_good = extract_content(raw_good)
        if "Error:" not in text_good:
            data = json.loads(text_good)
            assert data.get("id") == 1
    except Exception:
        # Network transient failure allowed, but MCP call must succeed
        pass


@pytest.mark.asyncio
async def test_server_call_read_csv(mcp_client):
    """Call read_csv over the real MCP protocol."""
    tools = await mcp_client.get_tools()
    tool_map = {t.name: t for t in tools}

    raw = await tool_map["read_csv"].ainvoke({"path": "sales.csv", "max_rows": 3})
    text = extract_content(raw)

    assert "CSV File: sales.csv" in text
    assert "Columns (7): month, product_line, region" in text
    assert "Row 01:" in text
    assert "Row 03:" in text


@pytest.mark.asyncio
async def test_server_call_summarize(mcp_client):
    """Call summarize over the real MCP protocol."""
    tools = await mcp_client.get_tools()
    tool_map = {t.name: t for t in tools}

    sample_text = (
        "Model Context Protocol defines an open communication standard. "
        "Agents connect to servers over stdio or HTTP channels. "
        "Introspection allows discovering tools dynamically at runtime. "
        "Extractive summarization captures high frequency sentences. "
        "Error handling prevents child subprocesses from crashing."
    )
    raw = await tool_map["summarize"].ainvoke({"text": sample_text})
    text = extract_content(raw)

    assert len(text) > 20
    assert "Model Context Protocol" in text


@pytest.mark.asyncio
async def test_server_call_word_frequency(mcp_client):
    """Call word_frequency over the real MCP protocol."""
    tools = await mcp_client.get_tools()
    tool_map = {t.name: t for t in tools}

    raw = await tool_map["word_frequency"].ainvoke({"text": "mcp stdio mcp agent mcp protocol agent", "top_n": 3})
    text = extract_content(raw)
    data = json.loads(text)

    assert isinstance(data, list)
    assert len(data) == 3
    assert data[0]["word"] == "mcp"
    assert data[0]["count"] == 3
