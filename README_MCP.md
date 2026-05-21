# BenchCaddy MCP

<img src="benchcaddy_mcp_logo.png" alt="BenchCaddy MCP logo" width="220"></img>

BenchCaddy ships a standalone MCP server that exposes benchmark inspection and baseline-management tools directly to coding agents.
Instead of constructing CLI commands and parsing `--json` output, an MCP client can call named tools against a BenchCaddy database.

## What It Is

BenchCaddy MCP is a thin tool surface over the existing BenchCaddy persistence and analysis API.
It is designed for agent workflows such as:

- discovering which benchmark suites exist
- inspecting suite history or a specific recorded run
- comparing runs or suite configurations
- reviewing trend data over time
- reading or updating the pinned suite baseline

The server is read-first. All tools are read-only except `pin_baseline`.

## Requirements

- install BenchCaddy in the same Python environment where the MCP server will run
- ensure the `benchcaddy-mcp` command is available on `PATH`, or use its full path in client configuration
- keep the benchmark database accessible to the client; tools default to `./benchcaddy.db` when `database_path` is omitted

You can start the server directly with:

```bash
benchcaddy-mcp
```

## Setup By Environment

### Generic Stdio MCP Clients

Many MCP clients use a generic `mcpServers` configuration shape.

```json
{
    "mcpServers": {
        "benchcaddy": {
            "command": "benchcaddy-mcp"
        }
    }
}
```

### Claude Desktop

Add the server to `claude_desktop_config.json`, then restart Claude Desktop in the same environment where BenchCaddy is installed.

```json
{
    "mcpServers": {
        "benchcaddy": {
            "command": "benchcaddy-mcp"
        }
    }
}
```

### VS Code And GitHub Copilot

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

You can also use `MCP: Add Server` from the Command Palette and choose Workspace or Global.
When VS Code first detects the server, review the configuration and accept the trust prompt.
After that, BenchCaddy tools become available to GitHub Copilot in chat and agent mode.

## Available Functionality

BenchCaddy MCP currently exposes these tools:

- `list_suites`: list recorded suites in the database
- `get_suite`: inspect one suite, including recent runs, environment data, and baseline context
- `get_run`: inspect one recorded run in detail
- `compare_suite`: compare runs within a suite against the best, a baseline, or a reference run
- `compare_runs`: compare two explicit runs directly
- `trend_suite`: inspect how one suite or one configuration changes over time
- `get_baseline_history`: inspect the baseline pin history for a suite
- `pin_baseline`: update the pinned baseline for a suite

Most tools accept an optional `database_path`. If omitted, they read `./benchcaddy.db`.

## Recommended Agent Workflow

For most benchmark analysis tasks, the clean flow is:

1. Call `list_suites` to discover available suites.
2. Call `get_suite` or `get_run` to load the relevant context.
3. Call `compare_suite`, `compare_runs`, or `trend_suite` for the actual analysis.
4. Call `get_baseline_history` before changing baseline policy.
5. Call `pin_baseline` only when you intentionally want to update the suite baseline.

## Response Shape

BenchCaddy MCP responses use a consistent envelope so agents can branch on outcome first and inspect payloads second.

- `status`: primary control signal, one of `pass`, `fail`, or `inconclusive`
- `reason`: stable snake_case classifier for the outcome
- `error_code`: machine-readable error identifier when the request failed
- `suggested_action`: useful next step for the caller
- `result`: tool-specific payload

Example based on a real local `compare_runs` response, with the database path normalized for documentation:

```text
Client: Call compare_runs with {"left_run_id": "4.2", "right_run_id": "4.3", "database_path": "/home/bench/benchcaddy/benchcaddy.db"}

BenchCaddy MCP:
{
    "schema_version": "1.0",
    "command": "compare_runs",
    "status": "pass",
    "reason": "comparison_complete",
    "error_code": null,
    "suggested_action": "Use the result payload to inspect classifications and candidate deltas.",
    "confidence": "high",
    "result": {
        "baseline": {
            "display_id": "4.2",
            "configuration": {
                "size": 512,
                "variant": "stabilized"
            },
            "median_seconds": 0.00041360000614076853
        },
        "candidate": {
            "display_id": "4.3",
            "configuration": {
                "size": 1024,
                "variant": "baseline"
            },
            "median_seconds": 0.0009896999690681696
        },
        "delta_seconds": 0.0005760999629274011,
        "percent_change": 139.28915724709293,
        "comparison_analysis": {
            "classification": "stable",
            "regression_probability": 1,
            "statistically_significant": false,
            "exceeds_practical_threshold": true
        },
        "observation_rows": [
            {
                "label": "benchmark_case",
                "delta_seconds": 0.0005789999966509641
            },
            {
                "label": "nonlinear_iteration",
                "delta_seconds": 0.0005210599745623767
            }
        ],
        "comparison_mode": "direct"
    }
}
```