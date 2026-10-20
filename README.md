# Layrcake starter for LangGraph

A [LangGraph](https://langchain-ai.github.io/langgraph/) agent that turns a one-sentence brief into about 10 people with verified work emails and a draft campaign, then stops at the approval link.

It uses LangChain's `create_agent`, which builds a LangGraph graph, with five `@tool` functions over the official [`layrcake`](https://github.com/Layrcake-dev/layrcake-python) Python SDK. Nothing is sent. A person approves the campaign in the Layrcake console before any email goes out.

This kit is in Python because most LangGraph users work in Python.

## What it does

1. `find_leads` finds people who match your brief.
2. `enrich` gets each person's work email.
3. `verify_email` checks every address. Only `valid` and `catch_all_valid` addresses are kept.
4. `create_campaign` saves a short draft sequence for the kept people. Drafts never send.
5. `launch_campaign` returns a preview and an approval link. The graph ends there.

## Why tools over the SDK for this kit

LangChain tools are typed Python functions, so wrapping the `layrcake` SDK in five of them keeps the rules in code:

- Each tool takes only the arguments this agent needs. `launch_campaign` has no `confirmation_token` argument, so the model cannot approve anything.
- The run state travels as LangGraph runtime context (`context_schema=Run`), and each tool reads it through `ToolRuntime`. The model never sees it.
- Every paid call gets `budget_credits` set to what is left of `BUDGET`. `--dry-run` adds `dry_run: true` to every call. `create_campaign` refuses any address that did not pass `verify_email` in this run. Once the approval link appears, every paid tool refuses.
- Paid calls take a lock, so the running total stays exact even when the model calls tools in parallel. Each call carries an idempotency key, so a repeated call returns the first result instead of charging again.
- `launch_campaign` is a `return_direct` tool, so the graph ends as soon as it returns the approval link.
- A refused call comes back to the model as the tool result ("Refused: ..."), so it can fix the call or stop.

The hosted MCP server at `https://mcp.layrcake.dev/mcp` also works with LangGraph through `langchain-mcp-adapters`. Plain tools keep the schemas small and the guardrails in your code.

## Prerequisites

- Python 3.10 or later and [uv](https://docs.astral.sh/uv/).
- A Layrcake API key. Create one at [layrcake.dev/connect](https://layrcake.dev/connect). New accounts start with 300 free credits.
- An Anthropic API key from [platform.claude.com](https://platform.claude.com), or another model provider (see `MODEL` below).
- Sending mailboxes in your Layrcake workspace, for the launch preview. Without them, `launch_campaign` returns an error that says what is missing, and the run reports it.

## Setup

```sh
uv sync
cp .env.example .env   # then add LAYRCAKE_API_KEY and ANTHROPIC_API_KEY
uv run --env-file .env smoke.py   # free check of your Layrcake key, no model call
```

`smoke.py` runs the agent's own tools in dry-run mode against the live API, the same way the graph's tool node calls them. It charges nothing and calls no model. It prints something like this:

```text
Key works for workspace "My workspace".
find_leads: would cost up to 3 credits
enrich: would cost up to 5 credits
verify_email: would cost up to 1.25 credits
create_campaign: would cost up to 0 credits

Charged: 0 credits. Your key and the Layrcake API work.
```

If `create_campaign` says it `would trip: sending_capacity`, connect mailboxes in the [Layrcake console](https://app.layrcake.dev) before a live run.

## Run

Start with a dry run. Every Layrcake call returns a plan with its cost instead of data, so it is free. It still uses your model key.

```sh
uv run --env-file .env agent.py --dry-run
```

Then run it for real with your own brief:

```sh
uv run --env-file .env agent.py "Find 10 heads of sales at software companies in London with 20 to 200 staff and draft a short campaign about cutting time spent on lead research."
```

With no brief, it uses the one above.

## What you will see

Each tool call prints as it runs. The run ends with a short report from the tools themselves:

```text
Live run. Budget: 100 credits.
Brief: Find 10 heads of sales at software companies in London ...

-> find_leads
-> enrich
-> verify_email
...
-> create_campaign
-> launch_campaign

Found 10 people. Kept 8 verified emails (2 catch-all).
Layrcake credits spent: 53 of 100.
Nothing has been sent. A person approves or declines here: <approval link>
```

The numbers above show the shape of a run. Your run will differ. A dry run ends with the model's cost estimate instead of a link.

## How approval works

`launch_campaign` never sends on its own. Its first answer is a preview, a `confirmation_token` and an `approval_url`. The tool returns the link and the graph ends.

A signed-in person in your workspace opens the link, reads the preview and approves or declines. Layrcake can also email an approver in your workspace. After approval, Layrcake starts the campaign itself when the approval says `execution: server`. Otherwise you repeat the call with the token. This agent never does either: its `launch_campaign` tool has no token argument.

## Cost

A run of the default brief usually costs 45 to 55 Layrcake credits. A credit is $0.01. Finding people costs 0.25 credits each, an email costs 5 credits when found, a verification costs 1.25, and lookups that find nothing are free. Drafting the campaign is free. See [pricing](https://docs.layrcake.dev/pricing.html).

`BUDGET` in `.env` is the ceiling for the whole run (default 100). Model usage is billed separately by your model provider.

## Changing the model

`MODEL` takes a `provider:model` string for LangChain's `init_chat_model`. The default is `anthropic:claude-sonnet-5-5`, which reads `ANTHROPIC_API_KEY`. For OpenAI, run `uv add langchain-openai`, then set `MODEL=openai:<model>` and `OPENAI_API_KEY`.

## Using OpenRouter

The kit runs through [OpenRouter](https://openrouter.ai) with `langchain-openrouter`, which is already a dependency. Put these in `.env` instead of a direct model key:

```text
OPENROUTER_API_KEY=<your OpenRouter key>
MODEL=openrouter:anthropic/claude-sonnet-5.5
```

## Project layout

```text
agent.py              the agent: model, graph, stop rule, report
layrcake_tools.py     the five Layrcake tools and the run state
instructions.py       the system prompt
smoke.py              free setup check, no model call
tests/test_tools.py   tool tests with a mocked Layrcake API
tests/test_agent.py   the full graph with a scripted chat model
```

## Development

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

The tests give the `layrcake` client an `httpx.MockTransport`, and drive the graph with a scripted chat model. They need no keys and make no network calls.

## Links

- Layrcake docs: [docs.layrcake.dev](https://docs.layrcake.dev)
- Rules for agents: [layrcake.dev/agents.md](https://layrcake.dev/agents.md)
- Python SDK: [github.com/Layrcake-dev/layrcake-python](https://github.com/Layrcake-dev/layrcake-python)
- MCP server: `https://mcp.layrcake.dev/mcp`
- LangGraph: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)

## License

MIT. See [LICENSE](LICENSE).
