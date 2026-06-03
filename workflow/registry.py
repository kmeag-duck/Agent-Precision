"""Agent registry — single source of truth for what agents exist.

Each entry maps a type name to its system prompt, the JSON Schema its
structured output must conform to, and the model it runs on.

Adding a new agent type = adding a new entry here. run_agent.py never has to
change; the orchestrator only gains a new tool wrapping the new type.
"""

REWRITER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "rewritten_code": {
            "type": "string",
            "description": (
                "The full rewritten kernel source. Must be valid, compilable "
                "code in the same language as the input."
            ),
        },
        "summary_of_changes": {
            "type": "string",
            "description": (
                "Plain-text per-variable summary: for each variable in the "
                "kernel, state whether you lowered its precision (and to what) "
                "or left it unchanged, with a one-line reason."
            ),
        },
    },
    "required": ["rewritten_code", "summary_of_changes"],
}

REWRITER_SYSTEM_PROMPT = """You are a kernel-precision-rewriter agent.

You will be given a numerical kernel (Kokkos C++, CUDA, or similar) and a
precise instruction describing which variables in that kernel should have
their precision lowered (typically double -> float) and which must remain at
the original precision. Your job is to produce the rewritten kernel.

Hard requirements on your output:
1. Preserve the kernel's external interface — function name, argument list,
   and overall behavior — unless the task instruction explicitly says
   otherwise.
2. Only change precision on variables the task instruction tells you to
   change. Do not lower other variables unilaterally.
3. Insert explicit casts at boundaries where precisions meet, so the compiler
   does not silently promote or demote in surprising ways.
4. Preserve all comments unless they contradict the new precision; in that
   case, update the comment to match the rewritten code.
5. The rewritten code must compile as-is against a standard Kokkos / CUDA
   toolchain.

Return your result by calling the submit_result tool with:
- rewritten_code: the complete rewritten kernel source.
- summary_of_changes: a per-variable listing of decisions you made, including
  the variables you left unchanged and why."""

AGENTS = {
    "rewriter": {
        "system_prompt": REWRITER_SYSTEM_PROMPT,
        "output_schema": REWRITER_OUTPUT_SCHEMA,
        "model": "claude-opus-4.7",
    },
}
