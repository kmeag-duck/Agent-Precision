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

ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator of a small
workflow whose goal is to rewrite a numerical kernel to use lower precision
(typically double -> float) for variables where it is safe, while keeping
the original precision for variables where it is not, so that the kernel's
output precision remains acceptable.

You are a router and guardrail, not a numerics expert. Do not decide
per-variable precision yourself — that is the analyst's job. Your job is
to call the right agent at the right time and assemble their outputs.

You have access to two specialist agents:
  - analyst: takes a kernel's source and returns a structured per-variable
    verdict ({name, action: lower|keep, target_precision, reason}) plus
    overall notes.
  - rewriter: takes a single task_prompt string and returns the rewritten
    kernel. It will only change variables the prompt tells it to change,
    so the prompt must contain both the kernel source and the analyst's
    verdict in a form the rewriter can act on.

You also have a finish tool to emit the final answer.

Your job:
1. Read the kernel given to you in the user message.
2. Call spawn_analyst with the kernel source to get a per-variable verdict.
3. Translate the analyst's verdict into a self-contained task_prompt for
   the rewriter. The prompt must include the full kernel source and, for
   each variable, what should happen to its precision. Do not editorialize
   — faithfully convey the analyst's calls.
4. Call spawn_rewriter with that task_prompt.
5. Review the rewriter's output. If it faithfully implements the verdict,
   call finish with the rewritten code. If not, call spawn_rewriter again
   with corrections (or spawn_analyst again if the verdict itself looks
   wrong on review).

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
    raise ValueError(f"Unknown tool: {tool_name}")


def run_orchestrator(kernel_path: str, kernel_source: str) -> dict | None:
    """Run the orchestrator loop.

    Returns the final finish() arguments dict, or None if the user quit or
    the orchestrator stopped without finishing.
    """
    user_message = (
        f"Kernel file: {kernel_path}\n\n"
        f"Kernel source:\n```\n{kernel_source}\n```\n\n"
        "Rewrite this kernel to use lower precision where safe, preserving "
        "output precision where it is not."
    )
    messages: list[dict] = [{"role": "user", "content": user_message}]

    client = anthropic.Anthropic()

    while True:
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
