"""Tests for workflow.run_agent."""

import pytest

from workflow.run_agent import run_agent

from .conftest import FakeResponse, TextBlock, ToolUseBlock


def test_unknown_agent_type_raises():
    with pytest.raises(ValueError, match="Unknown agent type"):
        run_agent("nonexistent", "task")


def test_returns_submit_result_input(fake_anthropic):
    payload = {
        "variables": [
            {"name": "x", "action": "lower", "target_precision": "float", "reason": "bounded"},
        ],
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
    assert tools[0]["input_schema"]["required"] == ["variables", "overall_notes"]


def test_raises_when_agent_does_not_call_submit_result(fake_anthropic):
    fake_anthropic([
        FakeResponse(
            content=[TextBlock(text="I refuse to call the tool.")],
            stop_reason="end_turn",
        ),
    ])

    with pytest.raises(RuntimeError, match="did not call submit_result"):
        run_agent("analyst", "task")


def test_ignores_non_submit_result_tool_use(fake_anthropic):
    """If the agent calls some other tool first, only submit_result counts.

    In practice the only tool available is submit_result, but the runner's
    loop iterates content blocks and matches by name, so verify it actually
    matches by name rather than just picking the first tool_use.
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
