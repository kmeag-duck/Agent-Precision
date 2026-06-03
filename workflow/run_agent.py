"""Generic agent runner — the only place where agent calls hit the API.

run_agent(type, task) -> dict

Looks up `type` in the registry, forces the agent to call a `submit_result`
tool whose input schema matches the agent's declared output_schema, and
returns the parsed tool input.

Adding a new agent type = adding an entry to registry.AGENTS — no changes here.
"""

import anthropic

from .registry import AGENTS


def run_agent(type: str, task: str) -> dict:
    if type not in AGENTS:
        raise ValueError(
            f"Unknown agent type: {type!r}. Known: {list(AGENTS)}"
        )
    spec = AGENTS[type]

    submit_result_tool = {
        "name": "submit_result",
        "description": (
            "Submit your final structured result for this task. "
            "Calling this tool ends your turn."
        ),
        "input_schema": spec["output_schema"],
    }

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=spec["model"],
        max_tokens=8192,
        system=spec["system_prompt"],
        tools=[submit_result_tool],
        tool_choice={"type": "tool", "name": "submit_result"},
        messages=[{"role": "user", "content": task}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_result":
            return dict(block.input)

    raise RuntimeError(
        f"Agent {type!r} did not call submit_result. "
        f"stop_reason={response.stop_reason}, "
        f"blocks={[b.type for b in response.content]}"
    )
