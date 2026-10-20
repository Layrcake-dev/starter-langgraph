"""The Layrcake tools for this agent, as LangChain tools.

The tools, their prices and every rule from https://layrcake.dev/agents.md live
in the official SDK's `layrcake.agent_tools`, the one implementation all
Layrcake agent integrations share. This file only hands five of them to
LangChain. The run state is an `AsyncLayrcakeSession`, passed to the agent as
its runtime context: each paid call carries the budget that is left, --dry-run
reaches every call, only verified emails go into a campaign, and nothing runs
past the approval link.
"""

from __future__ import annotations

from typing import Any, Union

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool
from layrcake.agent_tools import SPECS, AsyncLayrcakeSession, invalid_arguments
from pydantic import ValidationError
from pydantic.v1 import ValidationError as ValidationErrorV1


def _readable(error: Union[ValidationError, ValidationErrorV1]) -> str:
    """Bad arguments go back to the model as text, so it can fix the call."""
    if isinstance(error, ValidationError):
        return invalid_arguments(error)
    return f"Refused: invalid arguments: {error}"


def layrcake_tool(name: str, *, return_direct: bool = False) -> BaseTool:
    """One Layrcake tool: the SDK's name, description and argument model, run on the session in the context."""
    spec = SPECS[name]

    async def invoke(runtime: ToolRuntime[AsyncLayrcakeSession], **arguments: Any) -> str:
        return await runtime.context.call(spec.name, arguments)  # never raises: refusals come back as text

    return StructuredTool.from_function(
        coroutine=invoke,
        name=spec.name,
        description=spec.description,
        args_schema=spec.args,
        return_direct=return_direct,
        handle_validation_error=_readable,
    )


find_leads_tool = layrcake_tool("find_leads")
enrich_tool = layrcake_tool("enrich")
verify_email_tool = layrcake_tool("verify_email")
create_campaign_tool = layrcake_tool("create_campaign")
launch_campaign_tool = layrcake_tool("launch_campaign", return_direct=True)  # the run ends here, at the approval link

TOOLS = [find_leads_tool, enrich_tool, verify_email_tool, create_campaign_tool, launch_campaign_tool]
