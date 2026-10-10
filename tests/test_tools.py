"""Unit tests for MCP server tool functions directly."""

import json
from unittest.mock import MagicMock, patch
import pytest

from server import (
    search_docs,
    fetch_json,
    read_csv,
    summarize,
    word_frequency,
)


class TestSearchDocs:
    """Tests for pure-Python BM25 documentation search."""

    def test_search_docs_ranking_returns_relevant_doc_first(self):
        # Searching for transport mechanisms should rank transport_mechanisms.md first
        result = search_docs("stdio transport mechanisms sse", top_k=3)
        assert "Search Results for 'stdio transport mechanisms sse'" in result
        assert "1. [MCP Transport Mechanisms: Stdio vs SSE and HTTP]" in result
        assert "transport_mechanisms.md" in result
        assert "score:" in result

    def test_search_docs_ranking_security_topic(self):
        # Searching for path traversal security should rank security_sandboxing.md first
        result = search_docs("path traversal defense security sandboxing", top_k=3)
        assert "1. [Security Sandboxing and Defense in MCP Systems]" in result
        assert "security_sandboxing.md" in result

    def test_search_docs_respects_top_k(self):
        result = search_docs("MCP", top_k=2)
        # Should have results 1 and 2, but not 3
        assert "1. [" in result
        assert "2. [" in result
        assert "3. [" not in result

    def test_search_docs_empty_query(self):
        result = search_docs("")
        assert "Error: Search query cannot be empty." in result

    def test_search_docs_no_matches(self):
        result = search_docs("qwertyuiopasdfghjklzxcvbnm123456")
        assert "No relevant documents found" in result


class TestFetchJson:
    """Tests for HTTP/HTTPS JSON fetching and validation."""

    def test_fetch_json_rejects_file_scheme(self):
        result = fetch_json("file:///etc/passwd")
        assert "Error: Invalid URL scheme 'file'" in result

    def test_fetch_json_rejects_ftp_scheme(self):
        result = fetch_json("ftp://example.com/data.json")
        assert "Error: Invalid URL scheme 'ftp'" in result

    def test_fetch_json_rejects_missing_host(self):
        result = fetch_json("http://")
        assert "Error: Invalid URL format" in result

    def test_fetch_json_rejects_empty_url(self):
        result = fetch_json("")
        assert "Error: URL must be a non-empty string." in result

    @patch("requests.get")
    def test_fetch_json_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": 42, "status": "active", "item": "MCP"}
        mock_resp.headers = {"content-type": "application/json"}
        mock_get.return_value = mock_resp

        result = fetch_json("https://api.example.com/items/42")
        data = json.loads(result)
        assert data["id"] == 42
        assert data["item"] == "MCP"

    @patch("requests.get")
    def test_fetch_json_rejects_non_json_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/html"}
        mock_resp.json.side_effect = ValueError("No JSON object could be decoded")
        mock_get.return_value = mock_resp

        result = fetch_json("https://example.com/index.html")
        assert "Error: Response from https://example.com/index.html is not valid JSON" in result

    @patch("requests.get")
    def test_fetch_json_handles_http_errors(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.reason = "Not Found"
        mock_get.return_value = mock_resp

        result = fetch_json("https://api.example.com/missing")
        assert "Error: HTTP 404 Not Found" in result


class TestReadCsv:
    """Tests for sandboxed CSV reading."""

    def test_read_csv_reads_correct_columns_and_rows(self):
        result = read_csv("sales.csv", max_rows=5)
        assert "CSV File: sales.csv" in result
        assert "Columns (7): month, product_line, region, units_sold, unit_price, revenue, profit_margin" in result
        assert "Rows Displayed: 5" in result
        assert "Row 01:" in result
        assert "Row 05:" in result
        assert "Row 06:" not in result

    def test_read_csv_path_traversal_rejected(self):
        result = read_csv("../../etc/passwd")
        assert "Error: Access denied. Path '../../etc/passwd' is outside the authorized data directory" in result

    def test_read_csv_absolute_outside_path_rejected(self):
        result = read_csv("/etc/passwd")
        assert "Error: Access denied" in result

    def test_read_csv_nonexistent_file(self):
        result = read_csv("nonexistent_records.csv")
        assert "Error: File not found at 'nonexistent_records.csv'." in result

    def test_read_csv_non_csv_extension_rejected(self):
        result = read_csv("docs/mcp_overview.md")
        assert "Error: Path 'docs/mcp_overview.md' must be an existing .csv file." in result


class TestSummarize:
    """Tests for extractive text summarization."""

    def test_summarize_returns_top_sentences(self):
        text = (
            "Model Context Protocol creates an open standard for tool integration. "
            "Clients connect over stdio to communicate securely. "
            "Tools are discovered dynamically at runtime without static schemas. "
            "Extractive summarization identifies high frequency terms. "
            "Agents coordinate multi-step workflows with ease. "
            "Logging output is directed to stderr to keep stdout clean."
        )
        result = summarize(text)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should return top 3 sentences
        sentences = [s.strip() for s in result.split(".") if s.strip()]
        assert len(sentences) == 3

    def test_summarize_short_text_preserves_content(self):
        text = "First sentence here. Second sentence follows."
        result = summarize(text)
        assert "First sentence here." in result
        assert "Second sentence follows." in result

    def test_summarize_empty_text(self):
        result = summarize("")
        assert "Error: Input text is empty." in result


class TestWordFrequency:
    """Tests for word frequency analysis."""

    def test_word_frequency_counts_and_sorting(self):
        text = "protocol tool tool agent tool protocol agent"
        result = word_frequency(text, top_n=5)
        parsed = json.loads(result)

        assert isinstance(parsed, list)
        assert len(parsed) == 3
        # tool: 3, agent: 2, protocol: 2
        assert parsed[0]["word"] == "tool"
        assert parsed[0]["count"] == 3
        assert parsed[1]["count"] == 2
        assert parsed[2]["count"] == 2

    def test_word_frequency_respects_top_n(self):
        text = "alpha beta gamma delta epsilon zeta"
        result = word_frequency(text, top_n=2)
        parsed = json.loads(result)
        assert len(parsed) == 2

    def test_word_frequency_empty_text(self):
        result = word_frequency("")
        assert "Error: Input text is empty." in result
