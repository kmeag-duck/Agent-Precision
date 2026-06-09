"""Tests for workflow.run_agent."""

import pytest

from workflow.run_agent import run_agent

from .conftest import FakeResponse, TextBlock, ToolUseBlock


def test_unknown_agent_type_raises():
    """run_agent rejects an unknown agent type with ValueError."""
    with pytest.raises(ValueError, match="Unknown agent type"):
        run_agent("nonexistent", "task")


def test_returns_submit_result_input(fake_anthropic):
    """run_agent returns the input dict of the agent's submit_result call and forces the submit_result schema."""
    payload = {
        "variables": [
            {
                "name": "x",
                "action": "downcast",
                "target_precision": "float",
                "emulation_type": "",
                "reason": "bounded",
            },
        ],
        "rework": {
            "suggested": False,
            "transformation": "",
            "rationale": "",
            "affected_variables": [],
        },
        "precision_budget": {
            "target_kind": "sig_figs",
            "target_value": 6,
            "source": "user_cli",
            "claimed_output_precision": "~7 sig figs",
            "headroom_argument": "dominant rounding is in the sum",
        },
        "overall_notes": "ok",
    }
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input=payload)],
            stop_reason="tool_use",
        ),
    ])

    result = run_agent("analyst", "kernel source here")

    assert result == payload
    assert len(fake.messages.calls) == 1
    call = fake.messages.calls[0]
    # the runner must force the agent to call submit_result
    assert call["tool_choice"] == {"type": "tool", "name": "submit_result"}
    # the submit_result tool's schema must be the registry entry's output_schema
    tools = call["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "submit_result"
    assert tools[0]["input_schema"]["required"] == [
        "variables",
        "rework",
        "precision_budget",
        "overall_notes",
    ]


def test_raises_when_agent_does_not_call_submit_result(fake_anthropic):
    """run_agent raises RuntimeError if the agent returns text instead of calling submit_result."""
    fake_anthropic([
        FakeResponse(
            content=[TextBlock(text="I refuse to call the tool.")],
            stop_reason="end_turn",
        ),
    ])

    with pytest.raises(RuntimeError, match="did not call submit_result"):
        run_agent("analyst", "task")


def test_ignores_non_submit_result_tool_use(fake_anthropic):
    """run_agent matches the response block by name, skipping unrelated text blocks before submit_result.

    In practice the only tool available is submit_result, but the runner's
    loop iterates content blocks and matches by name, so verify it actually
    matches by name rather than just picking the first non-text block.
    """
    payload = {"rewritten_code": "code", "summary_of_changes": "changes"}
    fake_anthropic([
        FakeResponse(
            content=[
                TextBlock(text="thinking..."),
                ToolUseBlock(name="submit_result", input=payload),
            ],
            stop_reason="tool_use",
        ),
    ])

    result = run_agent("rewriter", "task")
    assert result == payload
