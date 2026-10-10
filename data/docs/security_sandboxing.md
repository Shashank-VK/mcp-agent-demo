# Security Sandboxing and Defense in MCP Systems

Because MCP servers expose tool execution and data retrieval capabilities to autonomous language models, robust defense-in-depth principles are mandatory.

## Defending Against Path Traversal

File-reading tools must never blindly trust input file paths supplied by clients or language models. Common attack vectors include directory traversal sequences such as `../../etc/passwd` or symbolic links pointing outside designated root directories.

Safe implementations enforce three defensive checks:
1. Canonical path resolution: Call `Path.resolve()` to eliminate relative dot segments and symlinks.
2. Boundary verification: Check that the resolved target path is strictly relative to the allowed data directory (e.g., using `Path.is_relative_to(base_dir)`).
3. File type whitelisting: Reject hidden dotfiles or non-standard file extensions.

## Safe Network Fetching and SSRF Protection

Tools that retrieve content over HTTP must defend against Server-Side Request Forgery (SSRF) and protocol manipulation:
- Enforce protocol restrictions: Explicitly allow only `http` and `https` schemes; reject `file://`, `gopher://`, or `data://` URIs.
- Response size limiting: Bounded character truncation prevents buffer overflow and denial of service when processing large payloads.
- Strict response validation: Confirm that the payload contains expected content types (e.g., JSON) before returning data to the reasoning agent.
