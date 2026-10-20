"""Find people, verify their work emails and draft a campaign that waits for a person to approve."""

from __future__ import annotations

import argparse
import asyncio
import os
from decimal import Decimal
from typing import Any, Optional, Union

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from layrcake import AsyncLayrcake
from layrcake.agent_tools import AsyncLayrcakeSession

from instructions import DEFAULT_BRIEF, instructions
from layrcake_tools import TOOLS

DEFAULT_MODEL = "anthropic:claude-sonnet-5-5"
MODEL_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY"}


def build_agent(
    budget: Decimal, dry_run: bool, model: Union[str, BaseChatModel], sender_name: Optional[str] = None
) -> Any:
    """A LangGraph agent. launch_campaign is return_direct, so the graph ends at the approval link."""
    if isinstance(model, str):
        model = init_chat_model(model)  # "provider:model", e.g. anthropic:claude-sonnet-5-5
    # Paid calls take the session's lock, so parallel tool calls still run one at a time.
    return create_agent(
        model, TOOLS, system_prompt=instructions(budget, dry_run, sender_name), context_schema=AsyncLayrcakeSession
    )


async def stream(agent: Any, brief: str, run: AsyncLayrcakeSession) -> str:
    """Run the graph, print each tool call as it happens, and return the model's last text."""
    final = ""
    async for update in agent.astream(
        {"messages": [{"role": "user", "content": brief}]},
        context=run,
        config={"recursion_limit": 80},
        stream_mode="updates",
    ):
        for step in update.values():
            for message in (step or {}).get("messages", []):
                for call in getattr(message, "tool_calls", None) or []:
                    print(f"-> {call['name']}")
                if message.type == "ai" and not getattr(message, "tool_calls", None):
                    final = message.text
    return final


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("brief", nargs="*", help="one sentence: who to find and what to say")
    parser.add_argument(
        "--dry-run", action="store_true", help="every Layrcake call returns its plan and charges nothing"
    )
    args = parser.parse_args()
    brief = " ".join(args.brief) or DEFAULT_BRIEF
    budget = Decimal(os.environ.get("BUDGET", "100"))
    model = os.environ.get("MODEL", DEFAULT_MODEL)
    key = MODEL_KEYS.get(model.split(":", 1)[0])
    if not os.environ.get("LAYRCAKE_API_KEY"):
        raise SystemExit("Set LAYRCAKE_API_KEY. Get a key at https://layrcake.dev/connect")
    if key and not os.environ.get(key):
        raise SystemExit(f"Set {key} for {model}.")
    if budget <= 0:
        raise SystemExit("BUDGET must be a positive number of credits.")

    print(
        f"{'Dry run, nothing is charged' if args.dry_run else 'Live run'}. Budget: {budget} credits.\nBrief: {brief}\n"
    )
    async with AsyncLayrcake() as client:  # reads LAYRCAKE_API_KEY
        run = AsyncLayrcakeSession(client, budget_credits=budget, dry_run=args.dry_run)
        agent = build_agent(budget, args.dry_run, model, os.environ.get("SENDER_NAME"))
        final = await stream(agent, brief, run)

    if final and not run.approval_url and not run.launch_error:
        print(f"\n{final}")
    print(f"\nFound {run.found} people. Kept {len(run.verified)} verified emails ({len(run.catch_all)} catch-all).")
    print(f"Layrcake credits spent: {run.spent} of {budget}.")
    if run.approval_url:
        print(f"Nothing has been sent. A person approves or declines here: {run.approval_url}")
    if run.launch_error:
        print(f"No approval link. Layrcake said: {run.launch_error}")


if __name__ == "__main__":
    asyncio.run(main())
