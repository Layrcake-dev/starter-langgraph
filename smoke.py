"""Free setup check. Runs the agent's tools in dry-run mode against the live Layrcake API, with no model.

Every call returns a plan and charges nothing.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from decimal import Decimal
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool
from layrcake import AsyncLayrcake

from layrcake_tools import Run, create_campaign_tool, enrich_tool, find_leads_tool, verify_email_tool

SAMPLE = {"email": "sample.person@example.com", "first_name": "Sample"}
CALLS: list[tuple[BaseTool, dict[str, Any]]] = [
    (
        find_leads_tool,
        {
            "audience": "Heads of sales at London software companies with 20 to 200 staff",
            "titles": ["Head of Sales"],
            "locations": ["London, United Kingdom"],
            "industries": ["Software Development"],
            "company_headcount": {"min": 20, "max": 200},
            "limit": 10,
        },
    ),
    (enrich_tool, {"full_name": "Sample Person", "company_domain": "example.com"}),
    (verify_email_tool, {"email": SAMPLE["email"]}),
    (
        create_campaign_tool,
        {"name": "Smoke test", "leads": [SAMPLE], "steps": [{"subject": "Hello", "body": "Hi {{first_name}}"}]},
    ),
]


async def call(tool: BaseTool, run: Run, args: dict[str, Any]) -> Any:
    """Invoke a tool the way the agent's ToolNode does: a tool call with the run as runtime context."""
    output = await tool.ainvoke(
        {"type": "tool_call", "name": tool.name, "id": "smoke", "args": {**args, "runtime": runtime(run)}}
    )
    try:
        return json.loads(output.content)
    except (TypeError, ValueError):
        raise SystemExit(f"{tool.name} failed: {output.content}") from None


def runtime(run: Run) -> ToolRuntime[Run]:
    return ToolRuntime(state={}, context=run, config={}, stream_writer=lambda _: None, tool_call_id="smoke", store=None)


async def main() -> None:
    async with AsyncLayrcake() as client:  # reads LAYRCAKE_API_KEY
        me = await client.get_me()
        print(f'Key works for workspace "{me.result["workspace"]["name"]}".')
        run = Run(client=client, budget=Decimal(os.environ.get("BUDGET", "100")), dry_run=True, run_id=uuid.uuid4().hex)
        for tool, args in CALLS:
            plan = await call(tool, run, args)
            if not plan.get("dry_run"):
                raise SystemExit(f"{tool.name}: expected a plan, got {plan}")
            trips = f", would trip: {', '.join(plan['would_exceed'])}" if plan["would_exceed"] else ""
            print(f"{tool.name}: would cost up to {plan['would_cost_up_to']} credits{trips}")
    print(f"\nCharged: {run.spent} credits. Your key and the Layrcake API work.")


if __name__ == "__main__":
    asyncio.run(main())
