from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool

from layrcake_tools import Run
from layrcake_tools import create_campaign_tool as create_campaign
from layrcake_tools import enrich_tool as enrich
from layrcake_tools import find_leads_tool as find_leads
from layrcake_tools import launch_campaign_tool as launch_campaign
from layrcake_tools import verify_email_tool as verify_email
from tests.conftest import APPROVAL_URL, CAMPAIGN_ID, MockLayrcake, meta


async def call(tool: BaseTool, run: Run, **args: Any) -> Any:
    """Invoke the tool the way the ToolNode does: a tool call in, a ToolMessage out."""
    runtime: ToolRuntime[Run] = ToolRuntime(
        state={}, context=run, config={}, stream_writer=lambda _: None, tool_call_id="t1", store=None
    )
    message = await tool.ainvoke(
        {"type": "tool_call", "name": tool.name, "id": "t1", "args": {**args, "runtime": runtime}}
    )
    try:
        return json.loads(message.content)
    except ValueError:
        return message.content  # a tool error, as the model would read it


async def test_paid_calls_carry_the_budget_left_and_an_idempotency_key(run: Run, api: MockLayrcake) -> None:
    found = await call(find_leads, run, audience="Sales leaders", titles=["Head of Sales"])
    assert found["leads"] == [
        {
            "full_name": "Ada Lovelace",
            "title": "Head of Sales",
            "company_name": "Acme",
            "company_domain": "acme.com",
            "location": "London",
        }
    ]
    path, body, key = api.requests[0]
    assert (path, body["budget_credits"], body["dry_run"], body["limit"]) == ("/v1/find_leads", 100.0, False, 10)
    assert key is not None
    assert key.startswith("run-test-find_leads-")

    await call(enrich, run, full_name="Ada Lovelace", company_domain="acme.com")
    assert api.requests[1][1]["want"] == ["work_email"]
    assert api.requests[1][1]["budget_credits"] == 97.0
    assert run.spent == Decimal(8)


async def test_find_leads_needs_a_filter(run: Run, api: MockLayrcake) -> None:
    output = await call(find_leads, run, audience="Anyone")
    assert "at least one of titles" in output
    assert api.requests == []


async def test_campaigns_only_take_verified_emails(run: Run) -> None:
    await call(verify_email, run, email="ada@acme.com")
    await call(verify_email, run, email="bob@acme.com")
    steps = [{"subject": "Hi", "body": "Hi {{first_name}}"}]
    refused = await call(create_campaign, run, name="Q4", leads=[{"email": "bob@acme.com"}], steps=steps)
    assert "bob@acme.com" in refused
    created = await call(create_campaign, run, name="Q4", leads=[{"email": "ada@acme.com"}], steps=steps)
    assert created["campaign_id"] == CAMPAIGN_ID
    assert (run.kept, run.spent) == (1, Decimal("2.5"))


async def test_launch_stops_at_the_approval_link(run: Run) -> None:
    preview = await call(launch_campaign, run, campaign_id=CAMPAIGN_ID)
    assert preview["approval_url"] == APPROVAL_URL
    assert run.approval_url == APPROVAL_URL
    assert "waiting for approval" in await call(verify_email, run, email="x@acme.com")


async def test_paid_calls_stop_when_the_budget_is_spent(run: Run) -> None:
    run.budget = Decimal(5)
    await call(enrich, run, full_name="Ada Lovelace", company_domain="acme.com")
    assert "budget" in await call(enrich, run, full_name="Bob Smith", company_domain="acme.com")


async def test_dry_run_returns_the_plan(run: Run, api: MockLayrcake) -> None:
    run.dry_run = True
    plan = {"verb": "verify_email", "credits_estimate": "1.25", "would_exceed": [], "warnings": []}
    api.answers["/v1/verify_email"] = lambda body: {
        "result": {"dry_run": True, "advisory": True, "evaluated_at": "2026-10-20T00:00:00Z", "plan": plan},
        "layrcake": meta("0"),
    }
    assert await call(verify_email, run, email="ada@acme.com") == {
        "dry_run": True,
        "would_cost_up_to": "1.25",
        "would_exceed": [],
    }
    assert api.requests[0][1]["dry_run"] is True


async def test_launch_campaign_has_no_confirmation_token_argument() -> None:
    assert set(launch_campaign.args) == {"campaign_id"}
    assert launch_campaign.return_direct
