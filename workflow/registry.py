"""Agent registry — single source of truth for what agents exist.

Each entry maps a type name to its system prompt, the JSON Schema its
structured output must conform to, and the model it runs on.

Adding a new agent type = adding a new entry here. run_agent.py never has to
change; the orchestrator only gains a new tool wrapping the new type.
"""

ANALYST_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "variables": {
            "type": "array",
            "description": (
                "One entry per named variable in the kernel that carries a "
                "floating-point value. Include every such variable; do not "
                "omit ones you decided to keep."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The variable's name as it appears in the source.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["lower", "keep"],
                        "description": (
                            "'lower' = reduce this variable's precision. "
                            "'keep' = leave at original precision."
                        ),
                    },
                    "target_precision": {
                        "type": "string",
                        "description": (
                            "Target precision when action='lower' (e.g. "
                            "'float', 'half'). Use the empty string when "
                            "action='keep'."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "One-line justification grounded in the kernel's "
                            "numerics (cancellation, accumulation, "
                            "long-time integration, bounded range, etc.)."
                        ),
                    },
                },
                "required": ["name", "action", "target_precision", "reason"],
            },
        },
        "overall_notes": {
            "type": "string",
            "description": (
                "Brief cross-cutting observations about the kernel's "
                "numerical behavior that influenced the per-variable calls."
            ),
        },
    },
    "required": ["variables", "overall_notes"],
}

ANALYST_SYSTEM_PROMPT = """You are a mixed-precision analyst agent for
numerical kernels (Kokkos C++, CUDA, or similar).

You will be given a kernel's source. Your job is to decide, on a
per-variable basis, which floating-point variables can have their precision
lowered (typically double -> float) and which must remain at the original
precision so that the kernel's output precision remains acceptable.

Reason explicitly about, for each variable:
- cancellation in subtractions of nearby values
- accumulation of many small terms (reduction-style roundoff)
- long-time integration roundoff (state that evolves over many steps)
- bounded-vs-unbounded magnitudes and known scale
- whether the variable feeds the kernel's final output directly or only
  appears in intermediate work

Coverage is mandatory and is the most common failure mode of this task.
Before you decide anything, enumerate every named floating-point entity
that appears in the kernel: scalar arguments, container/View arguments,
local variables, loop-carried accumulators, and any temporaries given a
name. Then produce one verdict entry per name on that list. Do not
invent variables that are not in the source. Do not omit variables that
are borderline or context-dependent — emit "keep" with a one-line reason
that explains the uncertainty. An entry missing from your variables
array is treated as a wrong answer.

Return your result by calling the submit_result tool with:
- variables: the per-variable list described above.
- overall_notes: brief cross-cutting observations that shaped your calls.

You do not rewrite code. Another agent will do that based on your verdict."""

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
1. Preserve the kernel's function name, the number and order of its
   arguments, and its overall behavior. Argument *types* may change to
   reflect the precision verdict — if an argument is named in the verdict
   as one that should be lowered, lower its declared type at the boundary
   (including container element types such as a Kokkos View's value type)
   so the precision change is end-to-end and not papered over with
   internal casts. Do not add, remove, or rename arguments. Lowering an
   argument's type is a callsite-breaking change for any code that calls
   this kernel; that is expected and acceptable here.
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

VERIFIER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["accept", "reject"],
            "description": (
                "'accept' = the rewritten kernel faithfully implements the "
                "analyst's verdict and is internally consistent. 'reject' = "
                "at least one per-variable entry has ok=false or the "
                "concerns list is non-empty in a way that warrants a redo."
            ),
        },
        "per_variable": {
            "type": "array",
            "description": (
                "One entry per variable named in the analyst's verdict. "
                "Each entry compares what the analyst asked for to what "
                "the rewritten source actually does."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "The variable's name, copied from the analyst "
                            "verdict."
                        ),
                    },
                    "expected_action": {
                        "type": "string",
                        "enum": ["lower", "keep"],
                        "description": (
                            "What the analyst said should happen to this "
                            "variable's precision."
                        ),
                    },
                    "observed_action": {
                        "type": "string",
                        "enum": ["lower", "keep", "unclear"],
                        "description": (
                            "What the rewritten source actually does, judged "
                            "from the declared type at the variable's boundary "
                            "(argument type, View element type, local declared "
                            "type), not from internal casts alone. Use "
                            "'unclear' only when the source is genuinely "
                            "ambiguous."
                        ),
                    },
                    "ok": {
                        "type": "boolean",
                        "description": (
                            "True iff expected_action == observed_action and "
                            "the rewrite is internally consistent for this "
                            "variable (e.g. lowering is end-to-end, not "
                            "papered over with internal casts)."
                        ),
                    },
                    "note": {
                        "type": "string",
                        "description": (
                            "One-line explanation. Required when ok=false "
                            "or observed_action='unclear'; otherwise a brief "
                            "confirmation is fine."
                        ),
                    },
                },
                "required": [
                    "name",
                    "expected_action",
                    "observed_action",
                    "ok",
                    "note",
                ],
            },
        },
        "concerns": {
            "type": "array",
            "description": (
                "Cross-cutting issues not tied to a single variable: renamed "
                "or reordered arguments, missing casts at boundaries, silent "
                "precision changes outside the analyst's verdict, broken "
                "control flow, comment/code mismatch, etc. Empty array is "
                "allowed and expected when the rewrite is clean."
            ),
            "items": {"type": "string"},
        },
    },
    "required": ["verdict", "per_variable", "concerns"],
}

VERIFIER_SYSTEM_PROMPT = """You are a static-review verifier agent for
mixed-precision kernel rewrites.

You will be given three things in your task message:
1. The original kernel source (all-double or all-long-double).
2. The rewritten kernel source produced by a separate rewriter agent.
3. The analyst agent's verdict, as JSON, describing what should have
   happened to each variable's precision.

Your job is to decide whether the rewritten source faithfully implements
the analyst's verdict and is internally consistent. You do not re-judge
whether the analyst's verdict itself was numerically correct — that is a
separate concern. If you suspect the analyst was wrong, surface it in the
`concerns` list rather than flipping per-variable verdicts.

For each variable in the analyst's verdict, decide observed_action by
inspecting the rewritten source at the variable's boundary:
- argument declared types (including container element types such as a
  Kokkos View's value type),
- local variable declared types,
- return types, struct/class field types if applicable.

A variable that is declared at the original precision but cast to a lower
precision only in internal expressions is NOT "lower" — that is "keep"
with an internal cast, and if the analyst asked for "lower" it is a
faithfulness failure (set ok=false). Lowering must be end-to-end at the
declared boundary.

Hard rules the rewriter was supposed to follow; flag violations in
`concerns`:
- Argument count, order, and names preserved.
- Argument types may change only when the analyst's verdict said so.
- Casts inserted at boundaries where precisions meet (so the compiler
  does not silently promote/demote).
- No precision changes outside what the analyst asked for.
- Comments either preserved or updated to match the new precision.

Set verdict='accept' only if every per_variable entry has ok=true AND
the concerns list is empty. Otherwise set verdict='reject'.

If the analyst-verdict JSON is malformed or absent, do the best you can
from source-vs-source inspection alone, leave per_variable empty, and
state the situation in `concerns`.

Borderline-numerics worries (e.g. "this variable was marked lower but I
think long runs might lose accuracy") belong in `concerns`, not in
per_variable ok=false. ok is strictly a faithfulness check.

Return your result by calling the submit_result tool with verdict,
per_variable, and concerns."""

AGENTS = {
    "analyst": {
        "system_prompt": ANALYST_SYSTEM_PROMPT,
        "output_schema": ANALYST_OUTPUT_SCHEMA,
        "model": "claude-opus-4-7",
    },
    "rewriter": {
        "system_prompt": REWRITER_SYSTEM_PROMPT,
        "output_schema": REWRITER_OUTPUT_SCHEMA,
        "model": "claude-opus-4-7",
    },
    "verifier": {
        "system_prompt": VERIFIER_SYSTEM_PROMPT,
        "output_schema": VERIFIER_OUTPUT_SCHEMA,
        "model": "claude-opus-4-7",
    },
}
