"""Tests for workflow.run_agent."""

import pytest

from workflow.run_agent import run_agent, run_agent_ensemble

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


def test_raises_when_response_content_is_none(fake_anthropic):
    """run_agent raises a useful RuntimeError naming stop_reason and response id when the SDK returns a response with content=None — guards against backend/proxy bugs that would otherwise surface as a cryptic TypeError on iteration."""
    fake_anthropic([
        FakeResponse(
            content=None,
            stop_reason="end_turn",
        ),
    ])

    with pytest.raises(RuntimeError, match="content=None"):
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


# ---------- tool_use.input coercion ----------
#
# Motivating incident (2026-07-01, K=3 nbody_force retry): one of three
# parallel analyst calls came back with `tool_use.input` as a JSON string
# instead of a decoded object (Argo proxy quirk), which flowed into
# aggregate_analyst_verdicts as `'str' object has no attribute 'get'`
# six frames deep. The coercion helper turns valid JSON strings into
# dicts silently, and turns everything else into a diagnosable
# RuntimeError naming the agent and response id.


def test_coerces_string_tool_input_via_json_decode(fake_anthropic):
    """run_agent silently JSON-decodes tool_use.input when the proxy forwards it as a string, so K>1 runs against Argo don't crash on a proxy quirk."""
    payload = {"rewritten_code": "code", "summary_of_changes": "changes"}
    import json as _json
    fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input=_json.dumps(payload))],
            stop_reason="tool_use",
        ),
    ])
    result = run_agent("rewriter", "task")
    assert result == payload


def test_raises_when_string_tool_input_is_not_valid_json(fake_anthropic):
    """A tool_use.input string that isn't valid JSON raises RuntimeError naming the agent and response id, not a bare JSONDecodeError."""
    fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input="{not json")],
            stop_reason="tool_use",
        ),
    ])
    with pytest.raises(RuntimeError, match="not valid JSON"):
        run_agent("rewriter", "task")


def test_raises_when_string_tool_input_decodes_to_non_dict(fake_anthropic):
    """A JSON-string tool_use.input that decodes to a list (or other non-object) raises RuntimeError — the aggregator and downstream tool-result plumbing require a dict."""
    fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input='["not", "an", "object"]')],
            stop_reason="tool_use",
        ),
    ])
    with pytest.raises(RuntimeError, match="JSON-encoded list"):
        run_agent("rewriter", "task")


def test_raises_when_tool_input_is_neither_dict_nor_string(fake_anthropic):
    """A tool_use.input of an unexpected type (e.g. int) raises RuntimeError — defensive guard against future SDK/proxy shape changes."""
    fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input=42)],
            stop_reason="tool_use",
        ),
    ])
    with pytest.raises(RuntimeError, match="expected dict or JSON-string"):
        run_agent("rewriter", "task")


# ---------- temperature plumbing ----------


@pytest.fixture
def reset_temperature_warning():
    """Clear the one-shot _TEMPERATURE_DROP_WARNED set before and after each test.

    The warning is process-global by design (we don't want to spam the
    operator), but that means tests that exercise the drop-warning path
    would interfere with each other in collection order. Reset it
    around every test that touches temperature handling.
    """
    from workflow.run_agent import _TEMPERATURE_DROP_WARNED
    _TEMPERATURE_DROP_WARNED.clear()
    yield
    _TEMPERATURE_DROP_WARNED.clear()


@pytest.fixture
def temperature_enabled_rewriter(monkeypatch):
    """Patch AGENTS['rewriter']['supports_temperature'] = True for the test.

    The registry default is False (Argo's claude-opus-4-7 rejects the
    kwarg). Tests that need to assert temperature reaches messages.create
    flip the flag for the rewriter entry specifically — picked because
    most tests in this file use the rewriter as their canary agent.
    """
    from workflow.registry import AGENTS
    monkeypatch.setitem(AGENTS["rewriter"], "supports_temperature", True)


@pytest.fixture
def temperature_enabled_verifier(monkeypatch):
    """Mirror of temperature_enabled_rewriter for verifier-targeted tests."""
    from workflow.registry import AGENTS
    monkeypatch.setitem(AGENTS["verifier"], "supports_temperature", True)


def test_temperature_dropped_by_default_when_unsupported(
    fake_anthropic, reset_temperature_warning, capsys
):
    """When the registry entry has supports_temperature=False (the default), run_agent does NOT forward `temperature` to messages.create — Argo's claude-opus-4-7 rejects the kwarg with HTTP 400, so the only safe default is to drop it."""
    payload = {"rewritten_code": "x", "summary_of_changes": "y"}
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input=payload)],
            stop_reason="tool_use",
        ),
    ])
    # No explicit temperature passed; the default-0.0 path must NOT
    # leak the kwarg when the model doesn't support it.
    run_agent("rewriter", "task")
    assert "temperature" not in fake.messages.calls[0]
    # And no warning because the caller didn't explicitly request a temperature.
    err = capsys.readouterr().err
    assert "temperature" not in err


def test_temperature_explicit_request_dropped_warns_once(
    fake_anthropic, reset_temperature_warning, capsys
):
    """When the caller explicitly requests a temperature for an agent whose registry entry sets supports_temperature=False, the kwarg is dropped (model would reject it) and a one-shot stderr warning is emitted naming the agent type and model — the operator's only signal that ensemble diversity is reduced."""
    payload = {"rewritten_code": "x", "summary_of_changes": "y"}
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input=payload)],
            stop_reason="tool_use",
        ),
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input=payload)],
            stop_reason="tool_use",
        ),
    ])
    run_agent("rewriter", "task", temperature=0.7)
    assert "temperature" not in fake.messages.calls[0]
    first = capsys.readouterr().err
    assert "temperature=0.7" in first
    assert "'rewriter'" in first
    assert "supports_temperature=False" in first
    # Second call with the same agent type does NOT re-warn.
    run_agent("rewriter", "task", temperature=0.5)
    assert "temperature" not in fake.messages.calls[1]
    second = capsys.readouterr().err
    assert "temperature" not in second


def test_temperature_defaults_to_single_shot_constant_when_supported(
    fake_anthropic, reset_temperature_warning, temperature_enabled_rewriter
):
    """When supports_temperature=True and no temperature is passed, DEFAULT_SINGLE_SHOT_TEMPERATURE is forwarded to messages.create — the single-shot path is deterministic by default; callers who want diversity must opt in by passing an explicit float."""
    from workflow.run_agent import DEFAULT_SINGLE_SHOT_TEMPERATURE

    payload = {"rewritten_code": "x", "summary_of_changes": "y"}
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input=payload)],
            stop_reason="tool_use",
        ),
    ])
    run_agent("rewriter", "task")
    assert (
        fake.messages.calls[0]["temperature"]
        == DEFAULT_SINGLE_SHOT_TEMPERATURE
    )
    # The constant itself is part of the contract: the whole point of
    # the change was to pin single-shot temperature at 0.0.
    assert DEFAULT_SINGLE_SHOT_TEMPERATURE == 0.0


def test_temperature_passes_through_when_set_and_supported(
    fake_anthropic, reset_temperature_warning, temperature_enabled_rewriter
):
    """When supports_temperature=True, a float temperature passed to run_agent reaches messages.create verbatim."""
    payload = {"rewritten_code": "x", "summary_of_changes": "y"}
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input=payload)],
            stop_reason="tool_use",
        ),
    ])
    run_agent("rewriter", "task", temperature=0.3)
    assert fake.messages.calls[0]["temperature"] == 0.3


# ---------- system_prompt_suffix plumbing ----------


def test_system_prompt_suffix_omitted_uses_base_prompt(fake_anthropic):
    """When run_agent is called without a system_prompt_suffix, the registry's base system prompt is passed verbatim to messages.create — no separator, no trailing whitespace shenanigans."""
    from workflow.registry import AGENTS

    payload = {"rewritten_code": "x", "summary_of_changes": "y"}
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input=payload)],
            stop_reason="tool_use",
        ),
    ])
    run_agent("rewriter", "task")
    assert fake.messages.calls[0]["system"] == AGENTS["rewriter"]["system_prompt"]


def test_system_prompt_suffix_appended_with_separator(fake_anthropic):
    """A non-None system_prompt_suffix is appended to the registry's base prompt with a blank-line separator — so the lens addendum reads as a distinct paragraph, not a run-on of the base instructions."""
    from workflow.registry import AGENTS

    payload = {"verdict": "accept", "per_variable": [], "concerns": []}
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input=payload)],
            stop_reason="tool_use",
        ),
    ])
    suffix = "LENS: faithfulness.\nFocus on per-variable boundary types."
    run_agent("verifier", "task", system_prompt_suffix=suffix)
    system = fake.messages.calls[0]["system"]
    assert system.startswith(AGENTS["verifier"]["system_prompt"])
    assert system.endswith(suffix)
    # Separator is exactly one blank line (two newlines) between base and suffix.
    assert f"{AGENTS['verifier']['system_prompt']}\n\n{suffix}" == system


def test_system_prompt_suffix_composes_with_temperature(
    fake_anthropic, reset_temperature_warning, temperature_enabled_verifier
):
    """system_prompt_suffix and temperature compose — when supports_temperature=True for the agent, both reach messages.create on the same call without interfering with each other (the verifier panel needs both at once)."""
    payload = {"verdict": "accept", "per_variable": [], "concerns": []}
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input=payload)],
            stop_reason="tool_use",
        ),
    ])
    run_agent(
        "verifier",
        "task",
        temperature=0.4,
        system_prompt_suffix="LENS: budget.",
    )
    call = fake.messages.calls[0]
    assert call["temperature"] == 0.4
    assert call["system"].endswith("LENS: budget.")


# ---------- run_agent_ensemble ----------


def test_ensemble_rejects_k_below_one():
    """run_agent_ensemble raises ValueError if k < 1; ensemble of size 0 is meaningless."""
    with pytest.raises(ValueError, match="k must be >= 1"):
        run_agent_ensemble("analyst", "task", k=0, temperature=0.7)


def test_ensemble_k_one_returns_single_result_and_threads_temperature(
    fake_anthropic, reset_temperature_warning, temperature_enabled_rewriter
):
    """K=1 takes the trivial single-call path (no thread pool); when supports_temperature=True, the call still receives the requested temperature."""
    payload = {"rewritten_code": "x", "summary_of_changes": "y"}
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input=payload)],
            stop_reason="tool_use",
        ),
    ])
    results = run_agent_ensemble("rewriter", "task", k=1, temperature=0.5)
    assert results == [payload]
    assert fake.messages.calls[0]["temperature"] == 0.5


def test_ensemble_k_three_returns_three_results_with_temperature(
    fake_anthropic, reset_temperature_warning, temperature_enabled_rewriter
):
    """K>1 returns one dict per call; when supports_temperature=True, each call sees the same temperature."""
    payloads = [
        {"rewritten_code": f"r{i}", "summary_of_changes": ""} for i in range(3)
    ]
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input=p)],
            stop_reason="tool_use",
        )
        for p in payloads
    ])
    results = run_agent_ensemble("rewriter", "task", k=3, temperature=0.7)
    assert len(results) == 3
    # FakeMessages pops responses in order regardless of which worker thread
    # got served first, so result identity-by-value (not by order) is what
    # we assert. Each call must carry the temperature.
    assert {r["rewritten_code"] for r in results} == {"r0", "r1", "r2"}
    assert all(c["temperature"] == 0.7 for c in fake.messages.calls)
    assert len(fake.messages.calls) == 3


def test_ensemble_k_three_drops_temperature_when_unsupported(
    fake_anthropic, reset_temperature_warning, capsys
):
    """When the registry entry has supports_temperature=False, K>1 ensemble calls each have temperature dropped on the wire — the operator gets a single-shot warning naming the agent type; ensemble runs proceed at the model's internal sampling instead of HTTP-400'ing."""
    payloads = [
        {"rewritten_code": f"r{i}", "summary_of_changes": ""} for i in range(3)
    ]
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input=p)],
            stop_reason="tool_use",
        )
        for p in payloads
    ])
    results = run_agent_ensemble("rewriter", "task", k=3, temperature=0.7)
    assert len(results) == 3
    assert all("temperature" not in c for c in fake.messages.calls)
    err = capsys.readouterr().err
    # Exactly one warning was emitted (one-shot per agent type).
    assert err.count("supports_temperature=False") == 1
