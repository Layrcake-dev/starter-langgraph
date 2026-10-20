"""Runs the real LangGraph agent with a scripted chat model, so no model provider is called."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any, Optional

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from layrcake.agent_tools import AsyncLayrcakeSession

from agent import build_agent, stream
from instructions import instructions
from tests.conftest import APPROVAL_URL, CAMPAIGN_ID, MockLayrcake


class ScriptedModel(GenericFakeChatModel):
    """Answers with the next scripted message. Tools are bound by the agent; this model ignores them."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, Any]:
        return self


def tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


async def test_agent_stops_at_the_approval_link(run: AsyncLayrcakeSession, api: MockLayrcake) -> None:
    lead = {"email": "ada@acme.com", "first_name": "Ada", "company_name": "Acme"}
    script = [
        tool_call("find_leads", {"audience": "Sales leaders in London", "titles": ["Head of Sales"]}, "c1"),
        tool_call("enrich", {"full_name": "Ada Lovelace", "company_domain": "acme.com"}, "c2"),
        tool_call("verify_email", {"email": "ada@acme.com"}, "c3"),
        tool_call(
            "create_campaign",
            {"name": "Q4", "leads": [lead], "steps": [{"subject": "Hi", "body": "Hi {{first_name}}"}]},
            "c4",
        ),
        tool_call("launch_campaign", {"campaign_id": CAMPAIGN_ID}, "c5"),
        AIMessage(content="This turn must never happen."),
    ]
    messages = iter(script)
    agent = build_agent(Decimal(100), dry_run=False, model=ScriptedModel(messages=messages))

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Find heads of sales in London"}]}, context=run
    )

    assert api.paths() == ["/v1/find_leads", "/v1/enrich", "/v1/verify_email", "/v1/campaigns", "/v1/launch_campaign"]
    assert run.approval_url == APPROVAL_URL
    assert next(messages).content == "This turn must never happen."  # the graph ended at launch_campaign
    assert result["messages"][-1].type == "tool"
    assert run.spent == Decimal("9.25")
    assert "confirmation_token" not in api.requests[-1][1]


async def test_a_refusal_reaches_the_model(run: AsyncLayrcakeSession, api: MockLayrcake) -> None:
    script = [
        tool_call(
            "create_campaign", {"name": "Q4", "leads": [{"email": "bob@acme.com"}], "steps": [{"body": "Hi"}]}, "c1"
        ),
        AIMessage(content="I need to verify that address first."),
    ]
    agent = build_agent(Decimal(100), dry_run=False, model=ScriptedModel(messages=iter(script)))

    result = await agent.ainvoke({"messages": [{"role": "user", "content": "Draft a campaign"}]}, context=run)

    refusal = result["messages"][2]
    assert refusal.type == "tool"
    assert "Verify these first" in refusal.content
    assert api.requests == []


async def test_stream_prints_each_tool_call(run: AsyncLayrcakeSession, capsys: pytest.CaptureFixture[str]) -> None:
    script = [
        tool_call("verify_email", {"email": "ada@acme.com"}, "c1"),
        AIMessage(content="Kept 1 address."),
    ]
    agent = build_agent(Decimal(100), dry_run=False, model=ScriptedModel(messages=iter(script)))

    final = await stream(agent, "Check ada@acme.com", run)

    assert final == "Kept 1 address."
    assert capsys.readouterr().out == "-> verify_email\n"


def test_with_sender_name_emails_are_signed_with_it() -> None:
    prompt = instructions(Decimal(100), dry_run=False, sender_name="Sam Carter")

    assert "End every email with this sign-off line, exactly as written: Sam Carter\n" in prompt
    assert "never invent a name, title or company for the sender" in prompt
    assert "[Your name]" not in prompt


@pytest.mark.parametrize("sender_name", [None, "", "   "])
def test_without_sender_name_emails_are_signed_your_name(sender_name: Optional[str]) -> None:
    prompt = instructions(Decimal(100), dry_run=False, sender_name=sender_name)

    assert "End every email with this sign-off line, exactly as written: [Your name]\n" in prompt
    assert "never invent a name, title or company for the sender" in prompt
    assert prompt.endswith(
        "Also say that every email is signed [Your name] and the sign-off must be filled in before the campaign\n"
        "is approved."
    )
