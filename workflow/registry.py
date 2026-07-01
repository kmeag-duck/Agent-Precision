"""Agent registry — single source of truth for what agents exist.

Each entry maps a type name to its system prompt, the JSON Schema its
structured output must conform to, and the model it runs on.

Adding a new agent type = adding a new entry here. run_agent.py never has to
change; the orchestrator only gains a new tool wrapping the new type.

Output-precision tolerance vocabulary used throughout this file:
  - "sig_figs"        — required correct *significant* figures of the
                        kernel's numerical output (relative tolerance:
                        N sig figs <=> relative error < 10^-N).
  - "decimal_digits"  — required correct *decimal places* after the point
                        (absolute tolerance: N digits <=> abs error < 10^-N).
These are deliberately distinct from floating-point storage precision
(float / double / float-float etc.); a kernel may need 6 sig figs of
output and achieve it with a mix of storage precisions internally. The
tolerance is always supplied by the operator on the command line
(--sig-figs or --decimal-digits); there is no inference path.
"""

# ---------------------------------------------------------------------------
# Analyst
# ---------------------------------------------------------------------------
#
# The analyst can recommend any of three precision-rewrite methods per
# variable, and (orthogonally) can suggest a single kernel-shape rework.
#
# Per-variable actions:
#   - "downcast" — replace the variable's declared type with a narrower one
#                  (e.g. double -> float). Cheapest and most portable;
#                  prefer this when the variable's numerics tolerate it.
#   - "emulate"  — keep the effective precision using a software-emulated
#                  wider type built from narrower hardware types (e.g. a
#                  float-float pair to approximate double). Use when the
#                  numerics demand the wider precision but the hardware
#                  type can be lowered (e.g. to free up double-precision
#                  units, or to run on a target without native doubles).
#   - "keep"     — leave the variable's precision unchanged.
#
# Kernel-shape rework is independent of the per-variable verdict. A kernel
# may have several variables downcast AND a rework suggestion (e.g. switch
# the accumulator loop to Kahan summation).

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
                        "enum": ["downcast", "emulate", "keep"],
                        "description": (
                            "'downcast' = replace this variable's declared "
                            "type with a narrower one (e.g. double->float). "
                            "'emulate' = keep effective precision via a "
                            "software-emulated wider type built from "
                            "narrower hardware types (e.g. float-float). "
                            "'keep' = leave at original precision."
                        ),
                    },
                    "target_precision": {
                        "type": "string",
                        "description": (
                            "Target hardware precision when action='downcast' "
                            "(e.g. 'float', 'half'). Use the empty string "
                            "when action='emulate' or 'keep'."
                        ),
                    },
                    "emulation_type": {
                        "type": "string",
                        "description": (
                            "Name of the emulated representation when "
                            "action='emulate' (e.g. 'float-float', "
                            "'double-double'). Use the empty string when "
                            "action='downcast' or 'keep'."
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
                "required": [
                    "name",
                    "action",
                    "target_precision",
                    "emulation_type",
                    "reason",
                ],
            },
        },
        "rework": {
            "type": "object",
            "description": (
                "Optional kernel-shape transformation orthogonal to the "
                "per-variable verdict. Suggested only when the kernel's "
                "structure itself is the numerical or performance problem "
                "(e.g. a long accumulation that would benefit from Kahan "
                "summation, an unstable subtraction that should be "
                "reformulated). Setting suggested=false with empty strings "
                "and an empty affected_variables list is the explicit "
                "'no rework' answer; do not omit this object."
            ),
            "properties": {
                "suggested": {
                    "type": "boolean",
                    "description": (
                        "True iff a kernel-shape rework is recommended."
                    ),
                },
                "transformation": {
                    "type": "string",
                    "description": (
                        "Name and brief description of the proposed "
                        "transformation (e.g. 'Kahan summation in the "
                        "accumulator loop'). Empty when suggested=false."
                    ),
                },
                "rationale": {
                    "type": "string",
                    "description": (
                        "Why this transformation helps the kernel's numerics "
                        "or performance. Empty when suggested=false."
                    ),
                },
                "affected_variables": {
                    "type": "array",
                    "description": (
                        "Names of variables (from the variables list above) "
                        "touched by the rework, for cross-reference. Empty "
                        "when suggested=false."
                    ),
                    "items": {"type": "string"},
                },
            },
            "required": [
                "suggested",
                "transformation",
                "rationale",
                "affected_variables",
            ],
        },
        "precision_budget": {
            "type": "object",
            "description": (
                "How the per-variable verdict relates to the output-"
                "precision tolerance you were given in the task. The "
                "tolerance is the constraint your verdict must satisfy; "
                "this block makes your reasoning about that constraint "
                "explicit so the verifier and the user can audit it."
            ),
            "properties": {
                "target_kind": {
                    "type": "string",
                    "enum": ["sig_figs", "decimal_digits"],
                    "description": (
                        "Copied verbatim from the tolerance in your task."
                    ),
                },
                "target_value": {
                    "type": "integer",
                    "description": (
                        "Copied verbatim from the tolerance in your task."
                    ),
                },
                "source": {
                    "type": "string",
                    "description": (
                        "Where the tolerance came from, copied from the "
                        "task (currently always 'user_cli')."
                    ),
                },
                "claimed_output_precision": {
                    "type": "string",
                    "description": (
                        "Your best estimate of the output precision the "
                        "rewritten kernel will actually deliver under "
                        "your per-variable verdict, in the same units as "
                        "target_kind (e.g. '~7 sig figs', '>=4 decimal "
                        "digits'). Be honest if your verdict is tight "
                        "against the target."
                    ),
                },
                "headroom_argument": {
                    "type": "string",
                    "description": (
                        "One or two sentences arguing why "
                        "claimed_output_precision meets target_value: "
                        "where the dominant rounding error in the "
                        "rewritten kernel comes from, and why it stays "
                        "below the tolerance. If you cannot make this "
                        "argument, your verdict is too aggressive and "
                        "you should mark more variables 'keep' or "
                        "'emulate' until you can."
                    ),
                },
            },
            "required": [
                "target_kind",
                "target_value",
                "source",
                "claimed_output_precision",
                "headroom_argument",
            ],
        },
        "overall_notes": {
            "type": "string",
            "description": (
                "Brief cross-cutting observations about the kernel's "
                "numerical behavior that influenced the per-variable calls."
            ),
        },
    },
    "required": ["variables", "rework", "precision_budget", "overall_notes"],
}

ANALYST_SYSTEM_PROMPT = """You are a mixed-precision analyst agent for
numerical kernels (Kokkos C++, CUDA, or similar).

You will be given a kernel's source AND an output-precision tolerance
(either N significant figures of the output, or N decimal digits after
the point). The tolerance is a hard constraint: your per-variable
verdict must produce a rewritten kernel whose output stays within that
tolerance of the original kernel's output.

Your job is to decide, per variable, which floating-point variables
should have their precision changed and how, so the rewritten kernel
meets that tolerance while reducing cost where safe.

You have three rewrite methods available per variable, in rough order of
preference:

1. downcast — replace the variable's declared type with a narrower one
   (e.g. double -> float). This is the throughput win: modern GPU and
   accelerator hardware runs fp32 (and narrower) at multiples of fp64
   throughput, and fp64 throughput has stagnated. Prefer downcast
   whenever the tolerance can absorb the narrower precision.

2. emulate — keep the variable's *effective* precision using a software-
   emulated wider type built from narrower hardware types (e.g. a
   float-float pair to approximate double). Emulation is
   throughput-NEGATIVE: a float-float operation costs several fp32 ops
   and is typically slower than a single native fp64 op. Only choose
   emulate when downcast would violate the tolerance AND the target
   hardware has weak or absent native double-precision support (so the
   emulated wider type is the only way to get the needed precision at
   any speed). If native doubles are available and meet the tolerance,
   'keep' beats 'emulate'. State the emulation_type explicitly (e.g.
   'float-float').

3. keep — leave the variable at its original precision. Choose this
   when downcast would violate the tolerance and native double already
   provides the needed precision at native throughput.

Orthogonal to the per-variable verdict, you may also suggest a single
kernel-shape rework: a structural transformation that improves the
kernel's numerics or performance (e.g. switching an accumulator to
Kahan summation, reformulating a cancellation-prone subtraction). A
rework is *not* a substitute for per-variable verdicts; it complements
them. Most kernels will not need one.

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

Field rules:
- action='downcast': set target_precision (e.g. 'float'); leave
  emulation_type as the empty string.
- action='emulate':  set emulation_type (e.g. 'float-float'); leave
  target_precision as the empty string.
- action='keep':     leave both target_precision and emulation_type as
  the empty string.

For the rework block: always submit it. If you have no rework to
suggest, set suggested=false and leave transformation, rationale, and
affected_variables empty.

For the precision_budget block: copy target_kind, target_value, and
source verbatim from the tolerance in your task. Then state your best
honest estimate of the output precision the rewritten kernel will
actually deliver under your per-variable verdict (claimed_output_
precision), and give a one- or two-sentence headroom_argument naming
where the dominant rounding error in the rewritten kernel will come
from and why it stays below the tolerance. If you cannot construct
that argument, your verdict is too aggressive — flip more variables to
'keep' (or 'emulate', subject to the throughput caveat above) until
you can.

Return your result by calling the submit_result tool with:
- variables: the per-variable list described above.
- rework: the rework object (suggested=false when none).
- precision_budget: the budget object linking your verdict to the
  tolerance.
- overall_notes: brief cross-cutting observations that shaped your calls.

You do not rewrite code. Another agent will do that based on your verdict.

PROBE EVIDENCE (when present in your task):
- If your task includes a 'PROBE EVIDENCE (JSON)' block at the end,
  the orchestrator already ran the kernel at several precisions
  (currently quad / double / float / mixed_io) and seeds (currently
  42 and 43) before invoking you, and is showing you per-output
  numerical stats from those runs against the quad/seed=42 ground
  truth. Use this evidence to corroborate or temper your verdict —
  do NOT let it override the source-level analysis you would
  otherwise do. Specifically:
  - A 'float_seed42' or 'mixed_io_seed42' cell whose per-output
    stats are well below the tolerance is evidence in favor of a
    'downcast' verdict for the kernel's float-storage variables;
    a cell whose stats are AT or ABOVE the tolerance is evidence
    against the same.
  - A large cross-seed delta (the same precision behaves very
    differently at seed=42 vs seed=43) is evidence of input-
    dependent rounding pain — usually a sign that downcast is risky
    even if the seed=42 cell looks fine. Lean toward 'keep' or
    'emulate' for the dominant accumulator in that case.
  - A 'no_quad_partner' or 'missing' or 'load_error' cell means no
    signal — NOT 'precision is safe'. Reason from source for those
    variables.
  - The evidence is per-output (e.g. the kernel's final array),
    not per-variable. Map outputs back to the variables that
    produced them using your source-level understanding of the
    kernel; if a variable feeds a clean output, its downcast is
    safer than a variable that feeds a noisy one.
  - The probe was run on the canonical seed (42) plus one adjacent
    seed (43); it is a sanity check, not a stress test. A clean
    probe does not license aggressive downcasting on edge-case
    inputs the probe never saw. Keep your precision_budget
    headroom_argument source-grounded."""

# ---------------------------------------------------------------------------
# Rewriter
# ---------------------------------------------------------------------------

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
                "kernel, state whether you downcast its precision (and to "
                "what), emulated it (and with what representation), or "
                "left it unchanged, with a one-line reason. If a kernel-"
                "shape rework was applied, describe it briefly."
            ),
        },
    },
    "required": ["rewritten_code", "summary_of_changes"],
}

REWRITER_SYSTEM_PROMPT = """You are a kernel-precision-rewriter agent.

You will be given a numerical kernel (Kokkos C++, CUDA, or similar) and a
precise instruction describing, per variable, which of three precision
methods to apply, plus an optional kernel-shape rework. Your job is to
produce the rewritten kernel.

You implement three per-variable methods:

1. downcast — replace the variable's declared type with the narrower
   type named in the instruction (e.g. double -> float). Lower the type
   end-to-end at the variable's boundary (argument type, container
   element type such as a Kokkos View's value type, local declaration);
   do not paper over the change with internal casts. Insert explicit
   casts where the lowered variable meets values of other precisions so
   the compiler does not silently promote or demote.

    Alias-driven downcast for kernel parameters. The kernel you are
   given may already declare its parameter types through `using`
   aliases (named `<ParamName>Type` — CamelCase of the parameter name
   plus the literal suffix `Type`) defined immediately above the
   function. When that pattern is present, downcast a kernel parameter
   by changing the corresponding `using` alias only — leave the
   function header and body untouched. For example, to downcast a
   parameter named `a` from double to float:

       // before
       using aType = Kokkos::View<const double*>;
       // after
       using aType = Kokkos::View<const float*>;

   This single edit propagates the precision change through the kernel
   header and through any caller (such as the baseline-harness driver's
   main()) that constructs `a`'s value through the same alias. Do not
   bypass the alias by writing `Kokkos::View<const float*>` directly
   in the function header — that breaks the contract the caller relies
   on. For parameters the analyst did NOT name, leave the alias
   unchanged.

   If the kernel you are given does not use the alias pattern, fall
   back to changing the parameter type directly in the function header
   as before.

2. emulate — replace the variable's declared type with a software-
   emulated pair type using the representation named in the instruction.
   For 'float-float', use the convention:

       struct ff_t {
           float hi;
           float lo;
       };

       // sum: Dekker / Knuth two-sum
       static inline ff_t ff_add(ff_t a, ff_t b) {
           float s = a.hi + b.hi;
           float bb = s - a.hi;
           float err = (a.hi - (s - bb)) + (b.hi - bb) + a.lo + b.lo;
           ff_t r;
           r.hi = s + err;
           r.lo = err - (r.hi - s);
           return r;
       }

       // product: Dekker two-prod (uses FMA when available)
       static inline ff_t ff_mul(ff_t a, ff_t b) {
           float p = a.hi * b.hi;
           float e = fmaf(a.hi, b.hi, -p) + a.hi * b.lo + a.lo * b.hi;
           ff_t r;
           r.hi = p + e;
           r.lo = e - (r.hi - p);
           return r;
       }

   Inline this struct and the helpers you need at the top of the file
   (or in an anonymous namespace for C++). Convert literals and
   single-precision values into ff_t with {value, 0.0f}. Convert ff_t
   back to a scalar at boundaries with `.hi` when the surrounding API
   takes a scalar. This is a deliberate v0 convention — it is not
   intended to be performant; it is intended to be correct and self-
   contained. Do not pull in an external library.

   For 'double-double' the same pattern applies with double/double
   instead of float/float and `fma` instead of `fmaf`.

3. keep — leave the variable unchanged.

Independently, the instruction may include a kernel-shape rework
(e.g. Kahan summation in an accumulator loop). If present, apply
exactly that transformation and no other algorithmic change. If
absent, do not invent one.

Hard requirements on your output:
1. Preserve the kernel's function name, the number and order of its
   arguments, and its overall behavior. Argument *types* may change to
   reflect the precision verdict — if an argument is named in the
   verdict as downcast or emulate, change its declared type at the
   boundary (including container element types such as a Kokkos View's
   value type) so the precision change is end-to-end and not papered
   over with internal casts. Do not add, remove, or rename arguments.
   Changing an argument's type is a callsite-breaking change for any
   code that calls this kernel; that is expected and acceptable here.
2. Only change precision on variables the task instruction tells you to
   change, and only via the method it specifies. Do not lower or
   emulate other variables unilaterally. Critically: do NOT silently
   substitute one method for another. If the instruction says
   'downcast' for a variable, do not 'emulate' it because you think
   emulate is safer; if it says 'emulate', do not 'downcast' it
   because you think emulate is too expensive. The analyst chose the
   method deliberately and the verifier will check method-by-method.
   If you genuinely believe a method choice is wrong, still apply it
   as instructed and say so in summary_of_changes — the orchestrator
   will route any disagreement back to the analyst.
3. Insert explicit casts at boundaries where precisions meet, so the
   compiler does not silently promote or demote.
4. Preserve all comments unless they contradict the new precision; in
   that case, update the comment to match the rewritten code.
5. The rewritten code must compile as-is against a standard Kokkos /
   CUDA toolchain. Emulation helpers must be defined in the same
   translation unit.

Return your result by calling the submit_result tool with:
- rewritten_code: the complete rewritten kernel source (including any
  inlined emulation helpers).
- summary_of_changes: a per-variable listing of decisions you made,
  including the variables you left unchanged and why, plus a brief
  description of any rework applied."""

# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

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
                        "enum": ["downcast", "emulate", "keep"],
                        "description": (
                            "What the analyst said should happen to this "
                            "variable's precision."
                        ),
                    },
                    "observed_action": {
                        "type": "string",
                        "enum": ["downcast", "emulate", "keep", "unclear"],
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
                            "variable (e.g. downcast is end-to-end, not "
                            "papered over with internal casts; emulate uses "
                            "the named pair representation at the boundary)."
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
                "control flow, comment/code mismatch, a rework block that "
                "was suggested by the analyst but not visibly applied in the "
                "rewrite, etc. Empty array is allowed and expected when the "
                "rewrite is clean."
            ),
            "items": {"type": "string"},
        },
    },
    "required": ["verdict", "per_variable", "concerns"],
}

VERIFIER_SYSTEM_PROMPT = """You are a static-review verifier agent for
mixed-precision kernel rewrites.

You will be given four things in your task message:
1. The original kernel source (all-double or all-long-double).
2. The rewritten kernel source produced by a separate rewriter agent.
3. The analyst agent's verdict, as JSON, describing what should have
   happened to each variable's precision and whether a kernel-shape
   rework was suggested.
4. The output-precision tolerance the analyst was working against
   (either N significant figures or N decimal digits, plus its source).
   You do not re-derive the tolerance; the analyst was given it as a
   constraint and you are given it for context.

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

observed_action interpretation:
- 'downcast' = the declared boundary type is a narrower scalar than the
  original (e.g. float in place of double).
- 'emulate'  = the declared boundary type is a software pair type (e.g.
  a struct named ff_t / dd_t, or similar) standing in for the wider
  hardware precision.
- 'keep'     = the declared boundary type matches the original.
- 'unclear'  = genuinely ambiguous; explain in `note`.

A variable that is declared at the original precision but cast to a
narrower precision only in internal expressions is NOT 'downcast' — that
is 'keep' with an internal cast, and if the analyst asked for 'downcast'
it is a faithfulness failure (set ok=false). Likewise, a variable
declared as a plain scalar but assigned from a pair-type's `.hi` field
is NOT 'emulate'. Boundary types are what count.

Hard rules the rewriter was supposed to follow; flag violations in
`concerns`:
- Argument count, order, and names preserved.
- Argument types may change only when the analyst's verdict said so,
  and only via the method (downcast vs emulate) it specified.
- Casts inserted at boundaries where precisions meet (so the compiler
  does not silently promote/demote).
- No precision changes outside what the analyst asked for.
- Comments either preserved or updated to match the new precision.
- If the analyst's rework.suggested was true, the rewrite should show
  evidence of the named transformation. If it does not, raise a
  concern naming the rework that was skipped.
- If the analyst's rework.suggested was false but the rewrite shows
  algorithmic restructuring beyond the per-variable precision changes,
  raise a concern about unauthorized rework.

Set verdict='accept' only if every per_variable entry has ok=true AND
the concerns list is empty. Otherwise set verdict='reject'.

If the analyst-verdict JSON is malformed or absent, do the best you can
from source-vs-source inspection alone, leave per_variable empty, and
state the situation in `concerns`.

Borderline-numerics worries (e.g. "this variable was marked downcast
but I think long runs might lose accuracy") belong in `concerns`, not
in per_variable ok=false. ok is strictly a faithfulness check.

If the analyst's precision_budget claims output precision that is
tight against the tolerance you were given, or if the analyst's
headroom_argument is missing or unconvincing, raise that in
`concerns`. Do not flip per_variable ok on that basis — faithfulness
is still a separate axis from whether the budget is realistic.

Return your result by calling the submit_result tool with verdict,
per_variable, and concerns."""

# ---------------------------------------------------------------------------
# Baseline harness
# ---------------------------------------------------------------------------
#
# The baseline-harness agent writes a self-contained driver program that,
# when later compiled and run, exercises the kernel on a fixed set of
# inputs and emits a reproducible reference output as JSON. The reference
# output is the baseline against which the dynamic-verification chain
# compares the rewritten kernel.
#
# Per-language driver templates, output schemas, and system prompts live
# in workflow.languages.<language>.BASELINE_HARNESS_SYSTEM_PROMPT etc.;
# this module just collects them into the AGENTS registry under
# `baseline_harness_<id>` keys (plus the legacy unsuffixed
# `baseline_harness` alias pointing at the Kokkos entry). The Kokkos
# prompt and schema are re-exported under their pre-refactor module-level
# names at the bottom of this file for back-compat with external imports.


_BASELINE_HARNESS_MODEL = "claude-opus-4-7"

# Per-language baseline-harness entries. The orchestrator's
# spawn_baseline_harness tool dispatches to `baseline_harness_<id>`
# where <id> matches the resolved LanguageProfile.id; the orchestrator
# never sees the per-language entry names directly. The plain
# `baseline_harness` alias is kept for back-compat with any caller that
# might still target the legacy single-language name; it points at the
# Kokkos entry because that was the only language the original prompt
# covered.

# Imported here (not at module top) so workflow.languages can in turn
# import workflow.registry helpers in the future without circling.
from .languages import PROFILES as _PROFILES  # noqa: E402

# Per-entry `supports_temperature` declares whether the model behind
# this agent will accept a `temperature` kwarg on messages.create.
# Argo's `claude-opus-4-7` snapshot rejects temperature outright
# (HTTP 400 `temperature is deprecated for this model`), so the
# default is False and `run_agent` drops the kwarg unless this flag
# is explicitly True. Flip to True per-entry when (and only when)
# you've verified the model accepts it; the ensemble path will still
# request K calls but they'll all run at the model's internal
# sampling, with a warning so the operator knows diversity is
# reduced. See AGENTS.md for the rationale and `run_agent` for the
# enforcement point.
AGENTS = {
    "analyst": {
        "system_prompt": ANALYST_SYSTEM_PROMPT,
        "output_schema": ANALYST_OUTPUT_SCHEMA,
        "model": "claude-opus-4-7",
        "supports_temperature": False,
    },
    "rewriter": {
        "system_prompt": REWRITER_SYSTEM_PROMPT,
        "output_schema": REWRITER_OUTPUT_SCHEMA,
        "model": "claude-opus-4-7",
        "supports_temperature": False,
    },
    "verifier": {
        "system_prompt": VERIFIER_SYSTEM_PROMPT,
        "output_schema": VERIFIER_OUTPUT_SCHEMA,
        "model": "claude-opus-4-7",
        "supports_temperature": False,
    },
}

for _profile in _PROFILES.values():
    AGENTS[f"baseline_harness_{_profile.id}"] = {
        "system_prompt": _profile.baseline_harness_system_prompt,
        "output_schema": _profile.baseline_harness_output_schema,
        "model": _BASELINE_HARNESS_MODEL,
        "supports_temperature": False,
    }

# Back-compat alias for callers that still target the unsuffixed name.
# Points at the Kokkos entry because that was the only language the
# pre-refactor `baseline_harness` covered.
AGENTS["baseline_harness"] = AGENTS["baseline_harness_kokkos"]

# Module-level back-compat: the module previously exported
# BASELINE_HARNESS_SYSTEM_PROMPT and BASELINE_HARNESS_OUTPUT_SCHEMA as
# the Kokkos prompt and schema. The canonical values now live on
# KOKKOS_PROFILE; expose them under the legacy names so external imports
# (and the test suite) keep working unchanged.
BASELINE_HARNESS_SYSTEM_PROMPT = _PROFILES["kokkos"].baseline_harness_system_prompt
BASELINE_HARNESS_OUTPUT_SCHEMA = _PROFILES["kokkos"].baseline_harness_output_schema
