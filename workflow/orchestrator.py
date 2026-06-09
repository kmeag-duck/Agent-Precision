"""LLM orchestrator with human-in-the-loop pause before every agent call.

The orchestrator is itself a Claude conversation. Its tools are one per
agent type (`spawn_analyst`, `spawn_rewriter`) plus a `finish` tool that
ends the workflow.

Before any tool is actually executed, this module prints the (tool_name,
arguments) and waits for the user to approve (y), reject (n), or quit (q).
That pause is the whole point of the v0 design: you see exactly what the
orchestrator decided to send to an agent before that call runs.

The orchestrator itself is a router + guardrail: it dispatches to
specialist agents and assembles their outputs. It does not make
per-variable precision decisions — that is the analyst's job.
"""

import json

import anthropic

from .run_agent import run_agent

ORCHESTRATOR_MODEL = "claude-opus-4-7"

# Hard upper bound on orchestrator API turns per run. The HITL pause is the
# primary safety net (the user can press 'q' at any time); this constant is a
# backstop so a misbehaving orchestrator loop cannot run indefinitely if
# left unattended.
MAX_TURNS = 20

ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator of a small
workflow whose goal is to rewrite a numerical kernel to reduce precision
cost where safe (typically double -> float, or a software-emulated wider
type on top of a narrower hardware type), while keeping the original
precision where it is not safe to reduce, so that the kernel's output
precision remains acceptable.

You are a router and guardrail, not a numerics expert. Do not decide
per-variable precision yourself — that is the analyst's job. Your job is
to call the right agent at the right time and assemble their outputs.

You have access to three specialist agents:
  - analyst: takes a kernel's source and returns a structured per-variable
    verdict plus an optional kernel-shape rework block and overall notes.
    Per-variable entries are
      {name, action, target_precision, emulation_type, reason}
    where action is one of:
      * 'downcast' — replace the declared type with a narrower one
        (target_precision says which, e.g. 'float'); emulation_type empty.
      * 'emulate'  — replace the declared type with a software-emulated
        pair type (emulation_type says which, e.g. 'float-float');
        target_precision empty.
      * 'keep'     — leave the variable unchanged; both target_precision
        and emulation_type empty.
    The rework block is
      {suggested, transformation, rationale, affected_variables}
    and, when suggested=true, names a single kernel-shape transformation
    (e.g. Kahan summation in an accumulator loop) that complements the
    per-variable verdict.

  - rewriter: takes a single task_prompt string and returns the rewritten
    kernel. It will only change variables the prompt tells it to change
    and only via the method the prompt specifies, so the prompt must
    contain both the kernel source and the analyst's full verdict in a
    form the rewriter can act on.

  - verifier: takes the original source, the rewritten source, and the
    analyst's verdict (as a JSON string), and returns
    {verdict: accept|reject, per_variable: [...], concerns: [...]}.
    It checks faithfulness of the rewrite to the verdict (including any
    suggested rework); it does not re-judge whether the verdict was
    numerically correct.

You also have a finish tool to emit the final answer.

Your job:
1. Read the kernel given to you in the user message.
2. Call spawn_analyst with the kernel source to get a verdict.
3. Translate the analyst's verdict into a self-contained task_prompt for
   the rewriter. The prompt must include the full kernel source and, for
   each variable, the analyst's chosen method (downcast / emulate / keep)
   together with target_precision or emulation_type as applicable. If
   the analyst's rework.suggested is true, include the transformation,
   rationale, and affected_variables verbatim and tell the rewriter to
   apply that transformation in addition to the per-variable changes.
   Do not editorialize — faithfully convey the analyst's calls and do
   not choose a method the analyst did not ask for.
4. Call spawn_rewriter with that task_prompt.
5. Call spawn_verifier with (original_source, rewritten_source from the
   rewriter, analyst_verdict_json). The analyst_verdict_json argument must
   be the analyst's full result object serialized as a JSON string.
6. If the verifier returns verdict='accept', call finish with the
   rewritten code. If verdict='reject', either call spawn_rewriter again
   with a task_prompt that incorporates the verifier's per-variable
   mismatches and concerns, or — if the verifier's `concerns` implicate
   the analyst's verdict itself — call spawn_analyst again. After any
   re-run, you must call spawn_verifier again on the new rewrite before
   calling finish.

Hard rule: you may not call finish unless the most recent spawn_verifier
call returned verdict='accept'.

Be deliberate. Each spawn_* call costs another model call and the user
will inspect every prompt before it runs. Prefer one well-crafted prompt
over several short ones."""

ORCHESTRATOR_TOOLS = [
    {
        "name": "spawn_analyst",
        "description": (
            "Run the analyst agent on a kernel source. "
            "Returns {variables: [{name, action, target_precision, reason}], "
            "overall_notes}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kernel_source": {
                    "type": "string",
                    "description": (
                        "The full kernel source to analyze. Do not include "
                        "file paths, framing hints, or any other text — the "
                        "analyst should see only the source."
                    ),
                },
            },
            "required": ["kernel_source"],
        },
    },
    {
        "name": "spawn_rewriter",
        "description": (
            "Run the rewriter agent with the given task prompt. "
            "Returns {rewritten_code, summary_of_changes}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_prompt": {
                    "type": "string",
                    "description": (
                        "The full prompt to send to the rewriter. Must "
                        "include the kernel source and a clear per-variable "
                        "precision instruction."
                    ),
                },
            },
            "required": ["task_prompt"],
        },
    },
    {
        "name": "spawn_verifier",
        "description": (
            "Run the verifier agent. It compares the rewritten source to "
            "the analyst's verdict and returns "
            "{verdict: accept|reject, per_variable: [...], concerns: [...]}. "
            "Must be called after spawn_rewriter and before finish."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "original_source": {
                    "type": "string",
                    "description": (
                        "The original kernel source, exactly as it was given "
                        "to the analyst."
                    ),
                },
                "rewritten_source": {
                    "type": "string",
                    "description": (
                        "The rewritten kernel source produced by the most "
                        "recent spawn_rewriter call."
                    ),
                },
                "analyst_verdict_json": {
                    "type": "string",
                    "description": (
                        "The analyst's full result object serialized as a "
                        "JSON string (i.e. json.dumps of the dict you got "
                        "back from spawn_analyst)."
                    ),
                },
            },
            "required": [
                "original_source",
                "rewritten_source",
                "analyst_verdict_json",
            ],
        },
    },
    {
        "name": "finish",
        "description": "Terminate the workflow with the final rewritten kernel.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rewritten_code": {
                    "type": "string",
                    "description": "The final rewritten kernel source.",
                },
                "notes": {
                    "type": "string",
                    "description": "Brief explanation of the rewrite decisions.",
                },
            },
            "required": ["rewritten_code", "notes"],
        },
    },
]


def _hitl_pause(tool_name: str, tool_input: dict) -> str:
    """Show the proposed tool call and ask y/n/q. Returns the choice."""
    print()
    print("=" * 72)
    print(f"=== Orchestrator wants to call: {tool_name} ===")
    print("=" * 72)
    for key, value in tool_input.items():
        print(f"\n--- argument: {key} ---")
        if isinstance(value, str):
            for line in (value.splitlines() or [""]):
                print(line)
        else:
            print(json.dumps(value, indent=2))
    print()
    while True:
        choice = input("Execute? [y]es / [n]o / [q]uit > ").strip().lower()
        if choice in ("y", "n", "q"):
            return choice
        print("Please answer y, n, or q.")


def _execute_tool(tool_name: str, tool_input: dict) -> dict:
    """Actually run the requested tool. Returns the result to feed back."""
    if tool_name == "spawn_analyst":
        result = run_agent("analyst", tool_input["kernel_source"])
        return {"status": "ok", "result": result}
    if tool_name == "spawn_rewriter":
        result = run_agent("rewriter", tool_input["task_prompt"])
        return {"status": "ok", "result": result}
    if tool_name == "spawn_verifier":
        task = (
            "ORIGINAL SOURCE:\n"
            f"{tool_input['original_source']}\n\n"
            "REWRITTEN SOURCE:\n"
            f"{tool_input['rewritten_source']}\n\n"
            "ANALYST VERDICT (JSON):\n"
            f"{tool_input['analyst_verdict_json']}\n"
        )
        result = run_agent("verifier", task)
        return {"status": "ok", "result": result}
    raise ValueError(f"Unknown tool: {tool_name}")


def run_orchestrator(
    kernel_path: str,
    kernel_source: str,
    max_turns: int = MAX_TURNS,
) -> dict | None:
    """Run the orchestrator loop.

    Returns the final finish() arguments dict, or None if the user quit,
    the orchestrator stopped without finishing, or max_turns was exhausted.
    """
    user_message = (
        f"Kernel file: {kernel_path}\n\n"
        f"Kernel source:\n```\n{kernel_source}\n```\n\n"
        "Rewrite this kernel to reduce precision cost where safe (via "
        "downcast, emulation, or — if warranted — a kernel-shape rework), "
        "preserving output precision where it is not."
    )
    messages: list[dict] = [{"role": "user", "content": user_message}]

    client = anthropic.Anthropic()

    turns = 0
    while True:
        if turns >= max_turns:
            print(
                f"\nOrchestrator hit max_turns={max_turns}. Stopping."
            )
            return None
        turns += 1
        response = client.messages.create(
            model=ORCHESTRATOR_MODEL,
            max_tokens=8192,
            system=ORCHESTRATOR_SYSTEM_PROMPT,
            tools=ORCHESTRATOR_TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        # surface any text the orchestrator emitted (its reasoning)
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print()
                print("--- Orchestrator reasoning ---")
                print(block.text)

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            print(
                f"\nOrchestrator stopped without calling a tool "
                f"(stop_reason={response.stop_reason}). Exiting."
            )
            return None

        tool_results: list[dict] = []
        finish_args: dict | None = None
        user_quit = False

        for tu in tool_use_blocks:
            choice = _hitl_pause(tu.name, dict(tu.input))
            if choice == "q":
                user_quit = True
                break
            if choice == "n":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps({"status": "rejected_by_user"}),
                })
                continue
            # choice == "y"
            if tu.name == "finish":
                finish_args = dict(tu.input)
                break
            exec_result = _execute_tool(tu.name, dict(tu.input))
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(exec_result),
            })

        if user_quit:
            print("\nUser quit. Stopping.")
            return None
        if finish_args is not None:
            return finish_args

        messages.append({"role": "user", "content": tool_results})
