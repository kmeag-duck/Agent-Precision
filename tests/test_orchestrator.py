"""Tests for workflow.orchestrator.

Covers _hitl_pause, _execute_tool, and the run_orchestrator loop
(happy path, rejection, quit, stop-without-tool).
"""

import json

import pytest

from workflow import orchestrator
from workflow.orchestrator import (
    DEFAULT_TOLERANCE_ON_ADVISOR_UNKNOWN,
    ORCHESTRATOR_SYSTEM_PROMPT,
    ORCHESTRATOR_TOOLS,
    _execute_tool,
    _format_baseline_block,
    _format_tolerance_block,
    _hitl_pause,
    run_orchestrator,
)

from .conftest import FakeResponse, TextBlock, ToolUseBlock


# ---------- _hitl_pause ----------


def _scripted_input(monkeypatch, answers):
    """Make builtins.input return successive values from `answers`."""
    answers = list(answers)

    def fake_input(prompt=""):
        if not answers:
            raise AssertionError("input() called more times than scripted")
        return answers.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


@pytest.mark.parametrize("choice", ["y", "n", "q"])
def test_hitl_returns_each_choice(monkeypatch, choice):
    """_hitl_pause returns y, n, or q exactly as the user typed it."""
    _scripted_input(monkeypatch, [choice])
    assert _hitl_pause("spawn_analyst", {"kernel_source": "..."}) == choice


def test_hitl_loops_on_invalid_then_accepts(monkeypatch):
    """_hitl_pause re-prompts on invalid input and then accepts a valid choice."""
    _scripted_input(monkeypatch, ["maybe", "", "Y", "y"])
    # uppercase Y is accepted (the code lowercases input)
    assert _hitl_pause("spawn_rewriter", {"task_prompt": "..."}) == "y"


def test_hitl_accepts_uppercase(monkeypatch):
    """_hitl_pause accepts uppercase choices by lowercasing the input."""
    _scripted_input(monkeypatch, ["Q"])
    assert _hitl_pause("finish", {"rewritten_code": "x", "notes": "y"}) == "q"


# ---------- _execute_tool ----------


def test_execute_tool_unknown_raises(monkeypatch):
    """_execute_tool raises ValueError on an unknown tool name."""
    with pytest.raises(ValueError, match="Unknown tool"):
        _execute_tool("not_a_tool", {})


def test_execute_tool_dispatches_spawn_analyst(monkeypatch):
    """_execute_tool routes spawn_analyst to run_agent('analyst', kernel_source) and wraps the result."""
    calls = []

    def stub_run_agent(type_, task):
        calls.append((type_, task))
        return {"variables": [], "overall_notes": "stubbed"}

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    result = _execute_tool("spawn_analyst", {"kernel_source": "SOURCE"})

    assert result == {
        "status": "ok",
        "result": {"variables": [], "overall_notes": "stubbed"},
    }
    assert calls == [("analyst", "SOURCE")]


def test_execute_tool_dispatches_spawn_rewriter(monkeypatch):
    """_execute_tool routes spawn_rewriter to run_agent('rewriter', task_prompt) and wraps the result."""
    calls = []

    def stub_run_agent(type_, task):
        calls.append((type_, task))
        return {"rewritten_code": "code", "summary_of_changes": "..."}

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    result = _execute_tool("spawn_rewriter", {"task_prompt": "PROMPT"})

    assert result["status"] == "ok"
    assert result["result"]["rewritten_code"] == "code"
    assert calls == [("rewriter", "PROMPT")]


# ---------- run_orchestrator: happy path ----------


def test_run_orchestrator_happy_path(monkeypatch, fake_anthropic):
    """run_orchestrator drives analyst -> rewriter -> finish end-to-end with HITL approvals."""
    # Three orchestrator turns: spawn_analyst, spawn_rewriter, finish.
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_1",
                name="spawn_analyst",
                input={"kernel_source": "KSRC"},
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_2",
                name="spawn_rewriter",
                input={"task_prompt": "rewrite please"},
            )],
        ),
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_3",
                name="finish",
                input={"rewritten_code": "FINAL", "notes": "done"},
            )],
        ),
    ])

    # Stub the agents (called via _execute_tool).
    def stub_run_agent(type_, task):
        if type_ == "analyst":
            return {"variables": [], "overall_notes": "ok"}
        if type_ == "rewriter":
            return {"rewritten_code": "FINAL", "summary_of_changes": "ok"}
        raise AssertionError(f"unexpected agent: {type_}")

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    # Three HITL prompts, all approved.
    _scripted_input(monkeypatch, ["y", "y", "y"])

    result = run_orchestrator("path/to/kernel.cpp", "kernel source body")

    assert result == {"rewritten_code": "FINAL", "notes": "done"}
    assert len(fake.messages.calls) == 3

    # First call: just the user message.
    first_messages = fake.messages.calls[0]["messages"]
    assert len(first_messages) == 1
    assert first_messages[0]["role"] == "user"
    assert "kernel source body" in first_messages[0]["content"]

    # Second call: user + assistant + user(tool_result). Check the tool_result
    # carries the stubbed analyst output.
    second_messages = fake.messages.calls[1]["messages"]
    tool_result_msg = second_messages[-1]
    assert tool_result_msg["role"] == "user"
    tr_block = tool_result_msg["content"][0]
    assert tr_block["type"] == "tool_result"
    assert tr_block["tool_use_id"] == "tu_1"
    payload = json.loads(tr_block["content"])
    assert payload["status"] == "ok"
    assert payload["result"]["overall_notes"] == "ok"


# ---------- run_orchestrator: rejection feeds back sentinel ----------


def test_run_orchestrator_rejection_feeds_back_sentinel(
    monkeypatch, fake_anthropic
):
    """A HITL 'n' rejects the tool call without invoking run_agent and feeds {'status':'rejected_by_user'} back to the orchestrator."""
    fake = fake_anthropic([
        # Turn 1: orchestrator proposes spawn_analyst — user rejects.
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_1",
                name="spawn_analyst",
                input={"kernel_source": "KSRC"},
            )],
        ),
        # Turn 2: after seeing rejection, orchestrator just calls finish.
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_2",
                name="finish",
                input={"rewritten_code": "X", "notes": "gave up"},
            )],
        ),
    ])

    # If the analyst is actually invoked, this test should fail loudly.
    def fail_run_agent(type_, task):
        raise AssertionError(
            f"run_agent should not be called after rejection; got {type_!r}"
        )

    monkeypatch.setattr(orchestrator, "run_agent", fail_run_agent)
    _scripted_input(monkeypatch, ["n", "y"])  # reject, then approve finish

    result = run_orchestrator("k.cpp", "src")

    assert result == {"rewritten_code": "X", "notes": "gave up"}

    # Verify the rejection sentinel was passed back on turn 2.
    second_messages = fake.messages.calls[1]["messages"]
    tool_result_msg = second_messages[-1]
    tr_block = tool_result_msg["content"][0]
    assert tr_block["type"] == "tool_result"
    assert tr_block["tool_use_id"] == "tu_1"
    payload = json.loads(tr_block["content"])
    assert payload == {"status": "rejected_by_user"}


# ---------- run_orchestrator: quit ----------


def test_run_orchestrator_quit_returns_none(monkeypatch, fake_anthropic):
    """A HITL 'q' aborts the loop, skips run_agent, and returns None."""
    fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_1",
                name="spawn_analyst",
                input={"kernel_source": "KSRC"},
            )],
        ),
    ])
    monkeypatch.setattr(
        orchestrator,
        "run_agent",
        lambda *a, **kw: pytest.fail("run_agent must not be called after quit"),
    )
    _scripted_input(monkeypatch, ["q"])

    assert run_orchestrator("k.cpp", "src") is None


# ---------- run_orchestrator: stop without tool ----------


def test_run_orchestrator_stop_without_tool_returns_none(
    monkeypatch, fake_anthropic
):
    """If the orchestrator responds with text only (no tool_use), run_orchestrator returns None."""
    fake_anthropic([
        FakeResponse(
            content=[TextBlock(text="I am just going to say words.")],
            stop_reason="end_turn",
        ),
    ])
    # No HITL prompts should fire because there's no tool_use block.
    _scripted_input(monkeypatch, [])

    assert run_orchestrator("k.cpp", "src") is None


# ---------- Orchestrator prompt: vocabulary matches the agents it routes to ----------


def test_orchestrator_prompt_names_all_three_methods_and_rework():
    """The orchestrator prompt names downcast, emulate, keep, and the rework block, so its task-prompt translation step has the right vocabulary to pass to the rewriter."""
    for token in ("downcast", "emulate", "keep", "rework"):
        assert token in ORCHESTRATOR_SYSTEM_PROMPT, (
            f"orchestrator prompt missing {token!r}"
        )


def test_orchestrator_prompt_names_precision_advisor_and_tolerance_kinds():
    """The orchestrator prompt names spawn_precision_advisor, sig_figs, decimal_digits, and precision_budget so the LLM knows when to call the advisor and how to thread tolerance into downstream task prompts."""
    for token in (
        "spawn_precision_advisor",
        "sig_figs",
        "decimal_digits",
        "precision_budget",
    ):
        assert token in ORCHESTRATOR_SYSTEM_PROMPT, (
            f"orchestrator prompt missing {token!r}"
        )


def test_orchestrator_prompt_states_advisor_unknown_fallback():
    """The orchestrator prompt documents the fallback to {sig_figs, 6, advisor_unknown_defaulted} when the precision_advisor returns kind='unknown', matching DEFAULT_TOLERANCE_ON_ADVISOR_UNKNOWN."""
    assert "advisor_unknown_defaulted" in ORCHESTRATOR_SYSTEM_PROMPT
    # The constant and the prompt must agree on the fallback shape.
    assert DEFAULT_TOLERANCE_ON_ADVISOR_UNKNOWN == {
        "kind": "sig_figs",
        "value": 6,
        "source": "advisor_unknown_defaulted",
    }


def test_orchestrator_prompt_forbids_finish_without_accept():
    """The orchestrator prompt explicitly forbids calling finish unless the most recent verifier returned accept; this rule lives in the prompt, not in code."""
    text = ORCHESTRATOR_SYSTEM_PROMPT
    assert "may not call finish" in text or "may not call `finish`" in text
    assert "accept" in text


# ---------- _format_tolerance_block ----------


def test_format_tolerance_block_none_tells_orchestrator_to_call_advisor():
    """With tolerance=None, the rendered block instructs the orchestrator to call spawn_precision_advisor first and documents both the advisor-returns-tolerance and advisor-unknown-fallback paths."""
    block = _format_tolerance_block(None)
    assert "not specified" in block
    assert "spawn_precision_advisor" in block
    assert "advisor_unknown_defaulted" in block


def test_format_tolerance_block_user_cli_forbids_advisor_call():
    """With a user-supplied tolerance, the rendered block contains the kind/value/source verbatim and tells the orchestrator NOT to call spawn_precision_advisor."""
    block = _format_tolerance_block(
        {"kind": "sig_figs", "value": 7, "source": "user_cli"}
    )
    assert "sig_figs" in block
    assert "7" in block
    assert "user_cli" in block
    assert "do NOT call" in block or "do not call" in block.lower()


# ---------- _execute_tool: spawn_precision_advisor + spawn_verifier(tolerance_json) ----------


def test_execute_tool_dispatches_spawn_precision_advisor(monkeypatch):
    """_execute_tool routes spawn_precision_advisor to run_agent('precision_advisor', kernel_source) and wraps the result."""
    calls = []

    def stub_run_agent(type_, task):
        calls.append((type_, task))
        return {
            "kind": "sig_figs",
            "value": 6,
            "rationale": "stubbed",
            "confidence": "medium",
            "alternative": "",
        }

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    result = _execute_tool(
        "spawn_precision_advisor", {"kernel_source": "SOURCE"}
    )

    assert result["status"] == "ok"
    assert result["result"]["kind"] == "sig_figs"
    assert calls == [("precision_advisor", "SOURCE")]


def test_execute_tool_spawn_verifier_includes_tolerance_in_task(monkeypatch):
    """_execute_tool builds the verifier's task string from original_source, rewritten_source, analyst_verdict_json, AND tolerance_json — so the verifier sees the same tolerance the analyst saw."""
    captured = {}

    def stub_run_agent(type_, task):
        captured["type"] = type_
        captured["task"] = task
        return {"verdict": "accept", "per_variable": [], "concerns": []}

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    result = _execute_tool(
        "spawn_verifier",
        {
            "original_source": "ORIG",
            "rewritten_source": "REW",
            "analyst_verdict_json": '{"variables": []}',
            "tolerance_json": '{"kind":"sig_figs","value":6,"source":"user_cli"}',
        },
    )

    assert result["status"] == "ok"
    assert captured["type"] == "verifier"
    # the task must contain all four pieces
    assert "ORIG" in captured["task"]
    assert "REW" in captured["task"]
    assert "ANALYST VERDICT" in captured["task"]
    assert "TOLERANCE" in captured["task"]
    assert "user_cli" in captured["task"]


# ---------- run_orchestrator: tolerance plumbing in the initial user message ----------


def test_run_orchestrator_user_cli_tolerance_appears_in_first_user_message(
    monkeypatch, fake_anthropic
):
    """When a user-supplied tolerance is passed, the first user message embeds it verbatim and the orchestrator is told NOT to call spawn_precision_advisor."""
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_1",
                name="finish",
                input={"rewritten_code": "X", "notes": "Y"},
            )],
        ),
    ])
    _scripted_input(monkeypatch, ["y"])  # approve finish

    run_orchestrator(
        "k.cpp",
        "src",
        tolerance={
            "kind": "decimal_digits",
            "value": 4,
            "source": "user_cli",
        },
    )

    first_user = fake.messages.calls[0]["messages"][0]["content"]
    assert "decimal_digits" in first_user
    assert "4" in first_user
    assert "user_cli" in first_user
    # advisor must NOT be invited when tolerance is user-supplied
    assert "do NOT call" in first_user or "do not call" in first_user.lower()


def test_run_orchestrator_no_tolerance_invites_advisor_in_first_user_message(
    monkeypatch, fake_anthropic
):
    """When tolerance=None, the first user message instructs the orchestrator to call spawn_precision_advisor first."""
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_1",
                name="finish",
                input={"rewritten_code": "X", "notes": "Y"},
            )],
        ),
    ])
    _scripted_input(monkeypatch, ["y"])

    run_orchestrator("k.cpp", "src", tolerance=None)

    first_user = fake.messages.calls[0]["messages"][0]["content"]
    assert "spawn_precision_advisor" in first_user
    assert "not specified" in first_user


# ---------- Baseline harness: tool schema + prompt + dispatch + user message ----------


def test_orchestrator_tools_include_spawn_baseline_harness():
    """ORCHESTRATOR_TOOLS exposes spawn_baseline_harness with kernel_source and kernel_stem as required string inputs."""
    by_name = {t["name"]: t for t in ORCHESTRATOR_TOOLS}
    assert "spawn_baseline_harness" in by_name
    tool = by_name["spawn_baseline_harness"]
    props = tool["input_schema"]["properties"]
    assert "kernel_source" in props
    assert "kernel_stem" in props
    assert props["kernel_source"]["type"] == "string"
    assert props["kernel_stem"]["type"] == "string"
    assert set(tool["input_schema"]["required"]) == {
        "kernel_source",
        "kernel_stem",
    }


def test_orchestrator_prompt_mentions_baseline_harness_and_side_artifact():
    """The orchestrator prompt names baseline_harness, the BASELINE STEP block, and explicitly marks the baseline as a side artifact that is not a precondition for finish."""
    text = ORCHESTRATOR_SYSTEM_PROMPT
    assert "baseline_harness" in text
    assert "BASELINE STEP" in text
    lower = text.lower()
    assert "side artifact" in lower
    # The baseline must not gate finish; the analyst->rewriter->verifier
    # pipeline still does. Surface this as an explicit assertion so a
    # future prompt edit can't silently flip the semantics.
    assert "not a precondition for finish" in lower


def test_execute_tool_dispatches_spawn_baseline_harness(monkeypatch, tmp_path):
    """_execute_tool routes spawn_baseline_harness to run_agent('baseline_harness', kernel_source), writes the driver to baselines/<stem>/driver.cpp, and returns the driver_path alongside the result."""
    monkeypatch.chdir(tmp_path)
    calls = []

    driver_text = "// driver\nint main(){return 0;}\n"

    def stub_run_agent(type_, task):
        calls.append((type_, task))
        return {
            "driver_source": driver_text,
            "kernel_function_name": "vector_add",
            "inputs_summary": "N=16384, seed=42",
            "output_arrays": ["z"],
        }

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    result = _execute_tool(
        "spawn_baseline_harness",
        {"kernel_source": "KSRC", "kernel_stem": "vector_add"},
    )

    assert result["status"] == "ok"
    assert result["result"]["kernel_function_name"] == "vector_add"
    assert calls == [("baseline_harness", "KSRC")]

    # The driver must land at baselines/<stem>/driver.cpp under CWD
    # (the orchestrator writes via a *relative* Path; under
    # monkeypatch.chdir(tmp_path) that resolves to tmp_path/baselines/...).
    driver_path = tmp_path / "baselines" / "vector_add" / "driver.cpp"
    assert driver_path.exists()
    assert driver_path.read_text() == driver_text
    assert result["driver_path"] == "baselines/vector_add/driver.cpp"


def test_execute_tool_spawn_baseline_harness_overwrites_existing(
    monkeypatch, tmp_path
):
    """A second spawn_baseline_harness call for the same stem overwrites the previous driver.cpp (parents=True, exist_ok=True; write_text replaces)."""
    monkeypatch.chdir(tmp_path)
    # Pre-create an old driver to be overwritten.
    old_dir = tmp_path / "baselines" / "vector_add"
    old_dir.mkdir(parents=True)
    (old_dir / "driver.cpp").write_text("OLD CONTENT")

    def stub_run_agent(type_, task):
        return {
            "driver_source": "NEW CONTENT",
            "kernel_function_name": "vector_add",
            "inputs_summary": "...",
            "output_arrays": ["z"],
        }

    monkeypatch.setattr(orchestrator, "run_agent", stub_run_agent)

    _execute_tool(
        "spawn_baseline_harness",
        {"kernel_source": "KSRC", "kernel_stem": "vector_add"},
    )

    assert (tmp_path / "baselines" / "vector_add" / "driver.cpp").read_text() \
        == "NEW CONTENT"


# ---------- _format_baseline_block ----------


def test_format_baseline_block_cpp_no_kernel_name_invites_call():
    """For a .cpp kernel without an explicit kernel_name, the block invites spawn_baseline_harness, surfaces the file stem as KERNEL STEM, and emits no 'TARGET KERNEL: <name>' value line (the agent infers the function)."""
    block = _format_baseline_block("test-kernels/kokkos/mixed/nbody_force.cpp", None)
    assert "BASELINE STEP" in block
    assert "spawn_baseline_harness" in block
    assert "KERNEL STEM: nbody_force" in block
    # The boilerplate may mention 'TARGET KERNEL:' as a hint to the
    # orchestrator about what it MAY prepend; what must NOT appear is an
    # actual 'TARGET KERNEL: <name>' value line (it would be empty/wrong).
    for line in block.splitlines():
        assert not line.startswith("TARGET KERNEL:"), line


def test_format_baseline_block_cpp_with_kernel_name_includes_target_line():
    """When kernel_name is given, the block adds a TARGET KERNEL: <name> line so the orchestrator can prepend it to the harness's kernel_source argument."""
    block = _format_baseline_block(
        "test-kernels/kokkos/lowerable/vector_add.cpp", "vector_add"
    )
    assert "KERNEL STEM: vector_add" in block
    assert "TARGET KERNEL: vector_add" in block


def test_format_baseline_block_cu_tells_orchestrator_to_skip():
    """For a CUDA .cu kernel, the block explicitly tells the orchestrator NOT to call spawn_baseline_harness (v0 is Kokkos-only)."""
    block = _format_baseline_block(
        "test-kernels/cuda/lowerable/vector_add.cu", None
    )
    assert "skipped" in block.lower()
    assert "Do NOT call spawn_baseline_harness" in block
    # No KERNEL STEM line either; nothing to do.
    assert "spawn_baseline_harness" in block  # only in the negation


# ---------- run_orchestrator: baseline block in initial user message ----------


def test_run_orchestrator_cpp_kernel_invites_baseline_in_first_user_message(
    monkeypatch, fake_anthropic
):
    """For a .cpp kernel, the first user message embeds the BASELINE STEP block with the file stem so the orchestrator can decide whether to call spawn_baseline_harness."""
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_1",
                name="finish",
                input={"rewritten_code": "X", "notes": "Y"},
            )],
        ),
    ])
    _scripted_input(monkeypatch, ["y"])

    run_orchestrator("path/to/nbody_force.cpp", "src")

    first_user = fake.messages.calls[0]["messages"][0]["content"]
    assert "BASELINE STEP" in first_user
    assert "KERNEL STEM: nbody_force" in first_user
    assert "spawn_baseline_harness" in first_user


def test_run_orchestrator_cpp_with_kernel_name_includes_target_kernel_line(
    monkeypatch, fake_anthropic
):
    """When kernel_name is passed to run_orchestrator, the first user message adds a TARGET KERNEL: <name> line to the BASELINE STEP block."""
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_1",
                name="finish",
                input={"rewritten_code": "X", "notes": "Y"},
            )],
        ),
    ])
    _scripted_input(monkeypatch, ["y"])

    run_orchestrator(
        "path/to/vector_add.cpp", "src", kernel_name="vector_add"
    )

    first_user = fake.messages.calls[0]["messages"][0]["content"]
    assert "TARGET KERNEL: vector_add" in first_user
    assert "KERNEL STEM: vector_add" in first_user


def test_run_orchestrator_cu_kernel_skips_baseline_in_first_user_message(
    monkeypatch, fake_anthropic
):
    """For a CUDA .cu kernel, the first user message's BASELINE STEP block tells the orchestrator not to call spawn_baseline_harness (v0 is Kokkos-only)."""
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(
                id="tu_1",
                name="finish",
                input={"rewritten_code": "X", "notes": "Y"},
            )],
        ),
    ])
    _scripted_input(monkeypatch, ["y"])

    run_orchestrator("path/to/vector_add.cu", "src")

    first_user = fake.messages.calls[0]["messages"][0]["content"]
    assert "BASELINE STEP" in first_user
    assert "skipped" in first_user.lower()
    assert "Do NOT call spawn_baseline_harness" in first_user


# ---------- compile_baseline_driver: tool schema + prompt + dispatch ----------


def test_orchestrator_tools_include_compile_baseline_driver():
    """ORCHESTRATOR_TOOLS exposes compile_baseline_driver with kernel_stem as the only required string input."""
    by_name = {t["name"]: t for t in ORCHESTRATOR_TOOLS}
    assert "compile_baseline_driver" in by_name
    tool = by_name["compile_baseline_driver"]
    props = tool["input_schema"]["properties"]
    assert "kernel_stem" in props
    assert props["kernel_stem"]["type"] == "string"
    assert set(tool["input_schema"]["required"]) == {"kernel_stem"}


def test_orchestrator_prompt_mentions_compile_baseline_driver_and_env_var():
    """The orchestrator prompt names compile_baseline_driver and AGENT_PRECISION_KOKKOS_ROOT, and states the compile is a side artifact (not a precondition for finish)."""
    text = ORCHESTRATOR_SYSTEM_PROMPT
    assert "compile_baseline_driver" in text
    assert "AGENT_PRECISION_KOKKOS_ROOT" in text
    # Must be marked as a side artifact, same as the baseline itself.
    assert "side artifact" in text.lower() or "not a precondition for finish" in text.lower()


def test_format_baseline_block_cpp_mentions_compile_step():
    """For a .cpp kernel, the BASELINE STEP block tells the orchestrator to follow a successful spawn_baseline_harness with a single compile_baseline_driver call using the same KERNEL STEM."""
    block = _format_baseline_block(
        "test-kernels/kokkos/mixed/nbody_force.cpp", None
    )
    assert "compile_baseline_driver" in block
    # Must couple it to the harness call, not be a standalone instruction.
    assert "spawn_baseline_harness" in block


def test_format_baseline_block_cu_does_not_mention_compile_step():
    """For a CUDA .cu kernel, the (skipped) BASELINE STEP block must NOT mention compile_baseline_driver — the compile depends on the harness, which is forbidden for .cu inputs."""
    block = _format_baseline_block(
        "test-kernels/cuda/lowerable/vector_add.cu", None
    )
    assert "compile_baseline_driver" not in block


def test_execute_tool_dispatches_compile_baseline_driver(monkeypatch):
    """_execute_tool routes compile_baseline_driver to workflow.tools.compile_baseline_driver and returns its {status, stdout, stderr, artifacts} dict verbatim (no extra wrapping)."""
    captured = {}

    def stub_compile(kernel_stem):
        captured["kernel_stem"] = kernel_stem
        return {
            "status": "ok",
            "stdout": "compiled fine",
            "stderr": "",
            "artifacts": ["baselines/nbody_force/driver"],
        }

    monkeypatch.setattr(orchestrator, "compile_baseline_driver", stub_compile)

    result = _execute_tool(
        "compile_baseline_driver", {"kernel_stem": "nbody_force"}
    )

    assert captured == {"kernel_stem": "nbody_force"}
    # Verbatim pass-through — same shape future remote-batch tools share.
    assert result == {
        "status": "ok",
        "stdout": "compiled fine",
        "stderr": "",
        "artifacts": ["baselines/nbody_force/driver"],
    }


def test_execute_tool_compile_baseline_driver_error_passes_through(monkeypatch):
    """When the compile helper returns status='error', _execute_tool passes that result through unchanged so the orchestrator sees the same error shape as a successful tool result."""
    monkeypatch.setattr(
        orchestrator,
        "compile_baseline_driver",
        lambda stem: {
            "status": "error",
            "stdout": "",
            "stderr": "AGENT_PRECISION_KOKKOS_ROOT is not set.",
            "artifacts": [],
        },
    )
    result = _execute_tool(
        "compile_baseline_driver", {"kernel_stem": "x"}
    )
    assert result["status"] == "error"
    assert "AGENT_PRECISION_KOKKOS_ROOT" in result["stderr"]
    assert result["artifacts"] == []


# ---------- run_baseline_driver: tool schema + prompt + dispatch ----------


def test_orchestrator_tools_include_run_baseline_driver():
    """ORCHESTRATOR_TOOLS exposes run_baseline_driver with kernel_stem as the only required string input."""
    by_name = {t["name"]: t for t in ORCHESTRATOR_TOOLS}
    assert "run_baseline_driver" in by_name
    tool = by_name["run_baseline_driver"]
    props = tool["input_schema"]["properties"]
    assert "kernel_stem" in props
    assert props["kernel_stem"]["type"] == "string"
    assert set(tool["input_schema"]["required"]) == {"kernel_stem"}


def test_orchestrator_prompt_mentions_run_baseline_driver_and_env_var():
    """The orchestrator prompt names run_baseline_driver and AGENT_PRECISION_RUN_TIMEOUT_SEC, and states the run output is a side artifact (not a precondition for finish)."""
    text = ORCHESTRATOR_SYSTEM_PROMPT
    assert "run_baseline_driver" in text
    assert "AGENT_PRECISION_RUN_TIMEOUT_SEC" in text
    # Must be marked as a side artifact, same as the baseline + compile.
    assert "side artifact" in text.lower() or "not a precondition for finish" in text.lower()


def test_format_baseline_block_cpp_mentions_run_step():
    """For a .cpp kernel, the BASELINE STEP block tells the orchestrator to follow a successful compile_baseline_driver with a single run_baseline_driver call using the same KERNEL STEM."""
    block = _format_baseline_block(
        "test-kernels/kokkos/mixed/nbody_force.cpp", None
    )
    assert "run_baseline_driver" in block
    # Must be coupled to the compile call, not a standalone instruction.
    assert "compile_baseline_driver" in block


def test_format_baseline_block_cu_does_not_mention_run_step():
    """For a CUDA .cu kernel, the (skipped) BASELINE STEP block must NOT mention run_baseline_driver — the run depends on a compile, which depends on the harness, which is forbidden for .cu inputs."""
    block = _format_baseline_block(
        "test-kernels/cuda/lowerable/vector_add.cu", None
    )
    assert "run_baseline_driver" not in block


def test_execute_tool_dispatches_run_baseline_driver(monkeypatch):
    """_execute_tool routes run_baseline_driver to workflow.tools.run_baseline_driver and returns its {status, stdout, stderr, artifacts} dict verbatim (no extra wrapping)."""
    captured = {}

    def stub_run(kernel_stem):
        captured["kernel_stem"] = kernel_stem
        return {
            "status": "ok",
            "stdout": "driver ran cleanly",
            "stderr": "",
            "artifacts": ["baselines/nbody_force/reference.json"],
        }

    monkeypatch.setattr(orchestrator, "run_baseline_driver", stub_run)

    result = _execute_tool(
        "run_baseline_driver", {"kernel_stem": "nbody_force"}
    )

    assert captured == {"kernel_stem": "nbody_force"}
    # Verbatim pass-through — same shape future remote-batch tools share.
    assert result == {
        "status": "ok",
        "stdout": "driver ran cleanly",
        "stderr": "",
        "artifacts": ["baselines/nbody_force/reference.json"],
    }


def test_execute_tool_run_baseline_driver_error_passes_through(monkeypatch):
    """When the run helper returns status='error', _execute_tool passes that result through unchanged so the orchestrator sees the same error shape as a successful tool result."""
    monkeypatch.setattr(
        orchestrator,
        "run_baseline_driver",
        lambda stem: {
            "status": "error",
            "stdout": "",
            "stderr": "Driver exited with code 7.",
            "artifacts": [],
        },
    )
    result = _execute_tool(
        "run_baseline_driver", {"kernel_stem": "x"}
    )
    assert result["status"] == "error"
    assert "code 7" in result["stderr"]
    assert result["artifacts"] == []


# ---------- splice_rewritten_kernel: tool schema + prompt + dispatch ----------


def test_orchestrator_tools_include_splice_rewritten_kernel():
    """ORCHESTRATOR_TOOLS exposes splice_rewritten_kernel with kernel_stem and rewritten_kernel_source as the only required string inputs."""
    by_name = {t["name"]: t for t in ORCHESTRATOR_TOOLS}
    assert "splice_rewritten_kernel" in by_name
    tool = by_name["splice_rewritten_kernel"]
    props = tool["input_schema"]["properties"]
    assert "kernel_stem" in props
    assert props["kernel_stem"]["type"] == "string"
    assert "rewritten_kernel_source" in props
    assert props["rewritten_kernel_source"]["type"] == "string"
    assert set(tool["input_schema"]["required"]) == {
        "kernel_stem",
        "rewritten_kernel_source",
    }


def test_orchestrator_prompt_mentions_splice_rewritten_kernel():
    """The orchestrator prompt names splice_rewritten_kernel, ties it to a verifier accept after a successful run_baseline_driver, and states a splice error must not block finish."""
    text = ORCHESTRATOR_SYSTEM_PROMPT
    assert "splice_rewritten_kernel" in text
    # Must be conditioned on verifier accept, not on the baseline chain alone.
    assert "verdict='accept'" in text
    # Splice must be flagged as non-blocking for finish, mirroring the
    # rest of the baseline chain.
    assert "side artifact" in text.lower() or "not a precondition for finish" in text.lower() or "must NOT block finish" in text


def test_format_baseline_block_cpp_mentions_splice_step():
    """For a .cpp kernel, the BASELINE STEP block tells the orchestrator to call splice_rewritten_kernel after a verifier accept following a successful run_baseline_driver."""
    block = _format_baseline_block(
        "test-kernels/kokkos/mixed/nbody_force.cpp", None
    )
    assert "splice_rewritten_kernel" in block
    # Must be coupled to both the verifier accept and run_baseline_driver,
    # not a standalone instruction.
    assert "verdict='accept'" in block
    assert "run_baseline_driver" in block


def test_format_baseline_block_cu_does_not_mention_splice_step():
    """For a CUDA .cu kernel, the (skipped) BASELINE STEP block must NOT mention splice_rewritten_kernel — the splice depends on a baseline driver, which depends on the harness, which is forbidden for .cu inputs."""
    block = _format_baseline_block(
        "test-kernels/cuda/lowerable/vector_add.cu", None
    )
    assert "splice_rewritten_kernel" not in block


def test_execute_tool_dispatches_splice_rewritten_kernel(monkeypatch):
    """_execute_tool routes splice_rewritten_kernel to workflow.tools.splice_rewritten_kernel, forwards both kernel_stem and rewritten_kernel_source, and returns its {status, stdout, stderr, artifacts} dict verbatim (no extra wrapping)."""
    captured = {}

    def stub_splice(kernel_stem, rewritten_kernel_source):
        captured["kernel_stem"] = kernel_stem
        captured["rewritten_kernel_source"] = rewritten_kernel_source
        return {
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "artifacts": ["baselines/nbody_force/rewritten/driver.cpp"],
        }

    monkeypatch.setattr(orchestrator, "splice_rewritten_kernel", stub_splice)

    result = _execute_tool(
        "splice_rewritten_kernel",
        {
            "kernel_stem": "nbody_force",
            "rewritten_kernel_source": "void k() { /* rewritten */ }\n",
        },
    )

    assert captured == {
        "kernel_stem": "nbody_force",
        "rewritten_kernel_source": "void k() { /* rewritten */ }\n",
    }
    # Verbatim pass-through — same shape future remote-batch tools share.
    assert result == {
        "status": "ok",
        "stdout": "",
        "stderr": "",
        "artifacts": ["baselines/nbody_force/rewritten/driver.cpp"],
    }


def test_execute_tool_splice_rewritten_kernel_error_passes_through(monkeypatch):
    """When the splice helper returns status='error' (e.g. missing baseline, missing sentinels), _execute_tool passes that result through unchanged so the orchestrator sees the same error shape as a successful tool result."""
    monkeypatch.setattr(
        orchestrator,
        "splice_rewritten_kernel",
        lambda stem, src: {
            "status": "error",
            "stdout": "",
            "stderr": (
                "Baseline driver source not found at "
                "baselines/x/driver.cpp."
            ),
            "artifacts": [],
        },
    )
    result = _execute_tool(
        "splice_rewritten_kernel",
        {"kernel_stem": "x", "rewritten_kernel_source": "void k(){}"},
    )
    assert result["status"] == "error"
    assert "Baseline driver source not found" in result["stderr"]
    assert result["artifacts"] == []


# ---------- compile_rewritten_driver: tool schema + prompt + dispatch ----------


def test_orchestrator_tools_include_compile_rewritten_driver():
    """ORCHESTRATOR_TOOLS exposes compile_rewritten_driver with kernel_stem as its only required string input — same shape as compile_baseline_driver."""
    by_name = {t["name"]: t for t in ORCHESTRATOR_TOOLS}
    assert "compile_rewritten_driver" in by_name
    tool = by_name["compile_rewritten_driver"]
    props = tool["input_schema"]["properties"]
    assert "kernel_stem" in props
    assert props["kernel_stem"]["type"] == "string"
    assert tool["input_schema"]["required"] == ["kernel_stem"]


def test_orchestrator_prompt_mentions_compile_rewritten_driver():
    """The orchestrator prompt names compile_rewritten_driver, ties it to a preceding successful splice_rewritten_kernel, and states a rewritten-compile error must not block finish."""
    text = ORCHESTRATOR_SYSTEM_PROMPT
    assert "compile_rewritten_driver" in text
    # Must be conditioned on splice success, not standalone.
    assert "splice_rewritten_kernel" in text
    # Must be flagged as non-blocking for finish, mirroring the rest of
    # the baseline + dynamic-verification chain.
    assert (
        "must NOT block finish" in text
        or "not a precondition for finish" in text.lower()
    )


def test_format_baseline_block_cpp_mentions_compile_rewritten_step():
    """For a .cpp kernel, the BASELINE STEP block tells the orchestrator to call compile_rewritten_driver immediately after a successful splice_rewritten_kernel."""
    block = _format_baseline_block(
        "test-kernels/kokkos/mixed/nbody_force.cpp", None
    )
    assert "compile_rewritten_driver" in block
    # Must be coupled to splice success — never a standalone instruction.
    assert "splice_rewritten_kernel" in block


def test_format_baseline_block_cu_does_not_mention_compile_rewritten_step():
    """For a CUDA .cu kernel, the (skipped) BASELINE STEP block must NOT mention compile_rewritten_driver — it depends on splice, which depends on the baseline chain, which is forbidden for .cu inputs."""
    block = _format_baseline_block(
        "test-kernels/cuda/lowerable/vector_add.cu", None
    )
    assert "compile_rewritten_driver" not in block


def test_execute_tool_dispatches_compile_rewritten_driver(monkeypatch):
    """_execute_tool routes compile_rewritten_driver to workflow.tools.compile_rewritten_driver, forwards the kernel_stem argument, and returns its {status, stdout, stderr, artifacts} dict verbatim (no extra wrapping)."""
    captured = {}

    def stub_compile(kernel_stem):
        captured["kernel_stem"] = kernel_stem
        return {
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "artifacts": ["baselines/nbody_force/rewritten/driver"],
        }

    monkeypatch.setattr(
        orchestrator, "compile_rewritten_driver", stub_compile
    )

    result = _execute_tool(
        "compile_rewritten_driver", {"kernel_stem": "nbody_force"}
    )

    assert captured == {"kernel_stem": "nbody_force"}
    # Verbatim pass-through — same shape future remote-batch tools share.
    assert result == {
        "status": "ok",
        "stdout": "",
        "stderr": "",
        "artifacts": ["baselines/nbody_force/rewritten/driver"],
    }


def test_execute_tool_compile_rewritten_driver_error_passes_through(monkeypatch):
    """When the rewritten-compile helper returns status='error' (e.g. missing source, compile failure), _execute_tool passes that result through unchanged so the orchestrator sees the same error shape as a successful tool result."""
    monkeypatch.setattr(
        orchestrator,
        "compile_rewritten_driver",
        lambda stem: {
            "status": "error",
            "stdout": "",
            "stderr": (
                "Driver source not found at "
                "baselines/x/rewritten/driver.cpp. Did "
                "splice_rewritten_kernel run and get approved for this "
                "kernel_stem?"
            ),
            "artifacts": [],
        },
    )
    result = _execute_tool(
        "compile_rewritten_driver", {"kernel_stem": "x"}
    )
    assert result["status"] == "error"
    assert "rewritten/driver.cpp" in result["stderr"]
    assert "splice_rewritten_kernel" in result["stderr"]
    assert result["artifacts"] == []


# ---------- run_rewritten_driver: tool schema + prompt + dispatch ----------


def test_orchestrator_tools_include_run_rewritten_driver():
    """ORCHESTRATOR_TOOLS exposes run_rewritten_driver with kernel_stem as its only required string input — same shape as run_baseline_driver."""
    by_name = {t["name"]: t for t in ORCHESTRATOR_TOOLS}
    assert "run_rewritten_driver" in by_name
    tool = by_name["run_rewritten_driver"]
    props = tool["input_schema"]["properties"]
    assert "kernel_stem" in props
    assert props["kernel_stem"]["type"] == "string"
    assert tool["input_schema"]["required"] == ["kernel_stem"]


def test_orchestrator_prompt_mentions_run_rewritten_driver():
    """The orchestrator prompt names run_rewritten_driver, ties it to a preceding successful compile_rewritten_driver, and states a rewritten-run error must not block finish."""
    text = ORCHESTRATOR_SYSTEM_PROMPT
    assert "run_rewritten_driver" in text
    # Must be conditioned on the rewritten-compile step succeeding,
    # never a standalone instruction.
    assert "compile_rewritten_driver" in text
    # Must be flagged as non-blocking for finish, mirroring the rest of
    # the baseline + dynamic-verification chain. The wording matches
    # either of the two phrasings used elsewhere in the prompt.
    assert (
        "must NOT block finish" in text
        or "not a precondition for finish" in text.lower()
    )


def test_format_baseline_block_cpp_mentions_run_rewritten_step():
    """For a .cpp kernel, the BASELINE STEP block tells the orchestrator to call run_rewritten_driver immediately after a successful compile_rewritten_driver — and still mentions the upstream splice/compile_rewritten steps."""
    block = _format_baseline_block(
        "test-kernels/kokkos/mixed/nbody_force.cpp", None
    )
    assert "run_rewritten_driver" in block
    # Must be coupled to compile_rewritten success — never standalone.
    assert "compile_rewritten_driver" in block
    # The whole rewritten chain must still be visible in the block so
    # the orchestrator does not lose context of how it got here.
    assert "splice_rewritten_kernel" in block


def test_format_baseline_block_cu_does_not_mention_run_rewritten_step():
    """For a CUDA .cu kernel, the (skipped) BASELINE STEP block must NOT mention run_rewritten_driver — it depends on compile_rewritten, which depends on splice, which depends on the baseline chain, which is forbidden for .cu inputs."""
    block = _format_baseline_block(
        "test-kernels/cuda/lowerable/vector_add.cu", None
    )
    assert "run_rewritten_driver" not in block


def test_execute_tool_dispatches_run_rewritten_driver(monkeypatch):
    """_execute_tool routes run_rewritten_driver to workflow.tools.run_rewritten_driver, forwards the kernel_stem argument, and returns its {status, stdout, stderr, artifacts} dict verbatim (no extra wrapping)."""
    captured = {}

    def stub_run(kernel_stem):
        captured["kernel_stem"] = kernel_stem
        return {
            "status": "ok",
            "stdout": "rewritten driver ran",
            "stderr": "",
            "artifacts": [
                "baselines/nbody_force/rewritten/reference.json"
            ],
        }

    monkeypatch.setattr(orchestrator, "run_rewritten_driver", stub_run)

    result = _execute_tool(
        "run_rewritten_driver", {"kernel_stem": "nbody_force"}
    )

    assert captured == {"kernel_stem": "nbody_force"}
    # Verbatim pass-through — same shape future remote-batch tools share.
    assert result == {
        "status": "ok",
        "stdout": "rewritten driver ran",
        "stderr": "",
        "artifacts": [
            "baselines/nbody_force/rewritten/reference.json"
        ],
    }


def test_execute_tool_run_rewritten_driver_error_passes_through(monkeypatch):
    """When the rewritten-run helper returns status='error' (e.g. missing binary, non-zero exit, timeout, invalid JSON), _execute_tool passes that result through unchanged so the orchestrator sees the same error shape as a successful tool result."""
    monkeypatch.setattr(
        orchestrator,
        "run_rewritten_driver",
        lambda stem: {
            "status": "error",
            "stdout": "",
            "stderr": (
                "Driver binary not found at "
                "baselines/x/rewritten/driver. Did "
                "compile_rewritten_driver run and succeed for this "
                "kernel_stem?"
            ),
            "artifacts": [],
        },
    )
    result = _execute_tool(
        "run_rewritten_driver", {"kernel_stem": "x"}
    )
    assert result["status"] == "error"
    assert "rewritten/driver" in result["stderr"]
    assert "compile_rewritten_driver" in result["stderr"]
    assert result["artifacts"] == []
