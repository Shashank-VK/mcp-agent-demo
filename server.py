"""MCP Server exposing search, network, CSV, and text processing tools over stdio."""

import csv
import json
import math
import os
import re
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from mcp.server.fastmcp import FastMCP

# Define project directories
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = (BASE_DIR / "data").resolve()
DOCS_DIR = (DATA_DIR / "docs").resolve()

# Initialize FastMCP Server
mcp = FastMCP("mcp-agent-demo")

# Stopwords for pure-Python BM25 scoring and extractive summarization
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
}


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", text)]


def _load_documents() -> List[Dict[str, Any]]:
    """Load local markdown documents from the docs directory."""
    documents = []
    if not DOCS_DIR.exists():
        return documents

    for file_path in sorted(DOCS_DIR.glob("*.md")):
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            continue

        # Extract title from the first header or fall back to file stem
        title = file_path.stem.replace("_", " ").title()
        for line in content.splitlines():
            line_str = line.strip()
            if line_str.startswith("# "):
                title = line_str[2:].strip()
                break

        tokens = _tokenize(content)
        documents.append({
            "path": file_path.name,
            "title": title,
            "content": content,
            "tokens": tokens,
            "length": len(tokens),
        })

    return documents


def _find_best_snippet(content: str, query_terms: List[str], max_len: int = 220) -> str:
    """Find the sentence or snippet with highest query term density."""
    sentences = re.split(r"(?<=[.!?\n])\s+", content.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return content[:max_len].strip() + ("..." if len(content) > max_len else "")

    scored_sentences = []
    query_set = set(query_terms)
    for sent in sentences:
        tokens = set(_tokenize(sent))
        matches = len(tokens.intersection(query_set))
        scored_sentences.append((matches, sent))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    best_sent = scored_sentences[0][1]
    if len(best_sent) > max_len:
        return best_sent[:max_len].strip() + "..."
    return best_sent


@mcp.tool()
def search_docs(query: str, top_k: int = 3) -> str:
    """Search local documentation files using BM25 relevance scoring.

    Args:
        query: Search keywords or question.
        top_k: Maximum number of ranked results to return (default: 3).

    Returns:
        Formatted string of ranked document snippets with scores.
    """
    try:
        if not query or not query.strip():
            return "Error: Search query cannot be empty."

        top_k = max(1, int(top_k))
        documents = _load_documents()
        if not documents:
            return "Error: No documentation files found in data/docs."

        query_tokens = [w for w in _tokenize(query) if w not in STOPWORDS]
        if not query_tokens:
            # Fall back to unfiltered tokens if all were stopwords
            query_tokens = _tokenize(query)

        n_docs = len(documents)
        avg_doc_len = sum(doc["length"] for doc in documents) / max(1, n_docs)

        # Calculate document frequency n(q) for each query term
        doc_freqs: Dict[str, int] = {}
        for token in query_tokens:
            doc_freqs[token] = sum(1 for doc in documents if token in doc["tokens"])

        # BM25 scoring parameters
        k1 = 1.5
        b = 0.75

        scored_results: List[Tuple[float, Dict[str, Any]]] = []
        for doc in documents:
            doc_len = doc["length"]
            term_counts = Counter(doc["tokens"])
            score = 0.0

            for token in query_tokens:
                n_q = doc_freqs.get(token, 0)
                if n_q == 0:
                    continue

                # Standard Robertson-Spärck Jones IDF formula with +1.0 floor
                idf = math.log(((n_docs - n_q + 0.5) / (n_q + 0.5)) + 1.0)
                tf = term_counts.get(token, 0)

                # Term frequency saturation component
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
                term_score = idf * (numerator / denominator)
                score += term_score

            if score > 0.0:
                scored_results.append((score, doc))

        # Sort descending by BM25 score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        top_results = scored_results[:top_k]

        if not top_results:
            return f"No relevant documents found for query: '{query}'."

        output_lines = [f"Search Results for '{query}' (top {len(top_results)} of {len(documents)} docs):"]
        for rank, (score, doc) in enumerate(top_results, start=1):
            snippet = _find_best_snippet(doc["content"], query_tokens)
            output_lines.append(
                f"\n{rank}. [{doc['title']}] (file: {doc['path']}, score: {score:.4f})\n"
                f"   Snippet: \"{snippet}\""
            )

        return "\n".join(output_lines)

    except Exception as exc:
        return f"Error during doc search: {str(exc)}"


@mcp.tool()
def fetch_json(url: str) -> str:
    """Fetch and validate a JSON payload from an HTTP or HTTPS endpoint with a 10s timeout.

    Args:
        url: Full HTTP or HTTPS URL to query.

    Returns:
        Formatted JSON string (truncated to 2000 chars if larger) or error message.
    """
    try:
        if not url or not isinstance(url, str):
            return "Error: URL must be a non-empty string."

        parsed = urllib.parse.urlparse(url.strip())
        if parsed.scheme.lower() not in ("http", "https"):
            return f"Error: Invalid URL scheme '{parsed.scheme}'. Only http:// and https:// URLs are allowed."

        if not parsed.netloc:
            return f"Error: Invalid URL format '{url}'. Missing host domain."

        import requests

        headers = {"Accept": "application/json", "User-Agent": "mcp-agent-demo/1.0"}
        response = requests.get(url, headers=headers, timeout=10.0, allow_redirects=True)

        if response.status_code >= 400:
            return f"Error: HTTP {response.status_code} {response.reason} fetching {url}."

        content_type = response.headers.get("content-type", "").lower()
        # Verify JSON payload
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError):
            return (
                f"Error: Response from {url} is not valid JSON "
                f"(content-type: '{content_type}')."
            )

        formatted = json.dumps(data, indent=2)
        if len(formatted) > 2000:
            return formatted[:2000] + "\n... [truncated to 2000 characters]"
        return formatted

    except requests.exceptions.Timeout:
        return f"Error: Request timed out after 10.0s for URL: {url}"
    except requests.exceptions.ConnectionError as exc:
        return f"Error: Network connection failed for URL: {url} ({str(exc)})"
    except Exception as exc:
        return f"Error fetching JSON: {str(exc)}"


@mcp.tool()
def read_csv(path: str, max_rows: int = 20) -> str:
    """Read local CSV data from the project data directory.

    Rejects paths outside the project data folder to prevent directory traversal.

    Args:
        path: Path to the CSV file (relative to project data directory or project root).
        max_rows: Maximum rows to return (default: 20).

    Returns:
        Formatted tabular summary of CSV columns and initial rows.
    """
    try:
        if not path or not isinstance(path, str):
            return "Error: Path must be a non-empty string."

        max_rows = max(1, int(max_rows))
        target_path = Path(path).expanduser()

        # Resolve candidate path: preference is DATA_DIR, but allow BASE_DIR relative paths
        if not target_path.is_absolute():
            cand_data = (DATA_DIR / target_path).resolve()
            cand_base = (BASE_DIR / target_path).resolve()
            if cand_data.exists():
                candidate = cand_data
            elif cand_base.exists():
                candidate = cand_base
            else:
                # Default to DATA_DIR target for clear "file not found" message
                candidate = cand_data
        else:
            candidate = target_path.resolve()

        # Enforce sandbox security constraint: must reside strictly inside DATA_DIR
        try:
            candidate.relative_to(DATA_DIR)
        except ValueError:
            return (
                f"Error: Access denied. Path '{path}' is outside the authorized "
                f"data directory ({DATA_DIR})."
            )

        if not candidate.exists():
            return f"Error: File not found at '{path}'."

        if not candidate.is_file() or candidate.suffix.lower() != ".csv":
            return f"Error: Path '{path}' must be an existing .csv file."

        with open(candidate, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
            except StopIteration:
                return f"Error: CSV file '{candidate.name}' is empty."

            rows = []
            for row in reader:
                rows.append(row)
                if len(rows) >= max_rows:
                    break

        output = [
            f"CSV File: {candidate.name}",
            f"Columns ({len(headers)}): {', '.join(headers)}",
            f"Rows Displayed: {len(rows)} (max limit: {max_rows})",
            "---",
        ]

        for i, row in enumerate(rows, start=1):
            row_items = [f"{col}={val}" for col, val in zip(headers, row)]
            output.append(f"Row {i:02d}: " + ", ".join(row_items))

        return "\n".join(output)

    except Exception as exc:
        return f"Error reading CSV: {str(exc)}"


@mcp.tool()
def summarize(text: str) -> str:
    """Extractive text summarizer returning the top 3 most salient sentences.

    Scores sentences using word frequency (TF) analysis with stopword removal.

    Args:
        text: Raw text to summarize.

    Returns:
        Summary consisting of the top sentences in chronological order.
    """
    try:
        if not text or not text.strip():
            return "Error: Input text is empty."

        clean_text = text.strip()
        # Split sentences on punctuation boundaries
        raw_sentences = re.split(r"(?<=[.!?])\s+", clean_text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]

        if not sentences:
            return clean_text

        if len(sentences) <= 3:
            return " ".join(sentences)

        # Calculate word frequency across document
        all_words = _tokenize(clean_text)
        content_words = [w for w in all_words if w not in STOPWORDS]
        if not content_words:
            content_words = all_words

        word_counts = Counter(content_words)

        # Score sentences
        scored_sentences = []
        for idx, sentence in enumerate(sentences):
            sent_words = [w for w in _tokenize(sentence) if w not in STOPWORDS]
            if not sent_words:
                sent_words = _tokenize(sentence)
            if not sent_words:
                continue

            # Sum TF scores, normalized by square root of length to avoid long sentence bias
            raw_score = sum(word_counts.get(w, 0) for w in sent_words)
            norm_score = raw_score / math.sqrt(len(sent_words))
            scored_sentences.append((norm_score, idx, sentence))

        if not scored_sentences:
            return " ".join(sentences[:3])

        # Select top 3 sentences by score
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_selected = scored_sentences[:3]

        # Re-sort selected sentences by original chronological index
        top_selected.sort(key=lambda x: x[1])

        return " ".join(s[2] for s in top_selected)

    except Exception as exc:
        return f"Error during summarization: {str(exc)}"


@mcp.tool()
def word_frequency(text: str, top_n: int = 10) -> str:
    """Compute word frequency statistics from input text.

    Args:
        text: Raw text to analyze.
        top_n: Number of most frequent terms to return (default: 10).

    Returns:
        JSON string containing sorted list of word and count objects.
    """
    try:
        if not text or not text.strip():
            return "Error: Input text is empty."

        top_n = max(1, int(top_n))
        tokens = [w for w in _tokenize(text) if len(w) >= 2 and w not in STOPWORDS]
        if not tokens:
            tokens = [w for w in _tokenize(text) if len(w) >= 1]

        counts = Counter(tokens)
        # Sort descending by count, then alphabetically
        sorted_pairs = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_n]

        results = [{"word": word, "count": count} for word, count in sorted_pairs]
        return json.dumps(results, indent=2)

    except Exception as exc:
        return f"Error computing word frequency: {str(exc)}"


def main():
    """Run the MCP server over stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
