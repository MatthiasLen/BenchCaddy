# BenchCaddy MCP

<img src="benchcaddy_mcp_logo.png" alt="BenchCaddy MCP logo" width="220"></img>

BenchCaddy ships an MCP server so coding agents can inspect benchmark data through tools instead of shelling out to `benchcaddy ... --json`.

Start the server with:

```bash
benchcaddy-mcp
```

For stdio-based MCP clients, the minimal config looks like this:

```json
{
    "mcpServers": {
        "benchcaddy": {
            "command": "benchcaddy-mcp"
        }
    }
}
```

The main tools are `list_suites`, `get_suite`, `get_run`, `compare_suite`, `compare_runs`, `trend_suite`, `get_baseline_history`, and `pin_baseline`.
Most tools are read-only and accept an optional `database_path`; if omitted they read `./benchcaddy.db`.

## Claude Desktop

Add this to your `claude_desktop_config.json`, then restart Claude Desktop in the same environment where BenchCaddy is installed.

```json
{
    "mcpServers": {
        "benchcaddy": {
            "command": "benchcaddy-mcp"
        }
    }
}
```

## VS Code And GitHub Copilot

In VS Code, MCP servers are configured in `mcp.json`. For a workspace-local setup, create `.vscode/mcp.json`:

```json
{
    "servers": {
        "benchcaddy": {
            "type": "stdio",
            "command": "benchcaddy-mcp"
        }
    }
}
```

You can also use the `MCP: Add Server` command and choose Workspace or Global.
After VS Code detects the server, confirm the trust prompt. BenchCaddy tools then become available to GitHub Copilot in chat and agent mode.

## Typical Flow

- call `list_suites` to discover suites
- call `get_suite` or `get_run` to inspect the relevant context
- call `compare_suite`, `compare_runs`, or `trend_suite` for analysis
- call `pin_baseline` only when you want to update the suite baseline

## Abridged Transcript

```text
Client: Call list_suites with {"database_path": "./benchcaddy.db"}

BenchCaddy MCP:
{
    "schema_version": "1.0",
    "command": "list_suites",
    "status": "pass",
    "reason": "suite_inventory_available",
    "error_code": null,
    "suggested_action": "Use get_suite or compare_suite for the next step.",
    "confidence": null,
    "result": {
        "database_path": "./benchcaddy.db",
        "suite_count": 2,
        "suites": [
            {"suite_name": "nonlinear-transform"},
            {"suite_name": "pi-approximations"}
        ]
    }
}
```