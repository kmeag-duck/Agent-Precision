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
from pathlib import Path

import anthropic

from .run_agent import run_agent
from .tools import (
    compare_outputs,
    compile_baseline_driver,
    compile_rewritten_driver,
    run_baseline_driver,
    run_rewritten_driver,
    splice_rewritten_kernel,
)

ORCHESTRATOR_MODEL = "claude-opus-4-7"

# Hard upper bound on orchestrator API turns per run. The HITL pause is the
# primary safety net (the user can press 'q' at any time); this constant is a
# backstop so a misbehaving orchestrator loop cannot run indefinitely if
# left unattended. Raised from 20 to 40 to accommodate the dynamic-
# verification chain (splice -> compile_rewritten -> run_rewritten ->
# compare) appended after the analyst -> rewriter -> verifier loop.
MAX_TURNS = 40

# Fallback tolerance applied when the user did not pass --sig-figs or
# --decimal-digits AND the precision_advisor returned kind='unknown'. 6
# sig figs is the conventional working precision of single-precision
# scientific computation and is what most kernels coded in double are
# actually using in practice.
DEFAULT_TOLERANCE_ON_ADVISOR_UNKNOWN = {
    "kind": "sig_figs",
    "value": 6,
    "source": "advisor_unknown_defaulted",
}

ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator of a small
workflow whose goal is to rewrite a numerical kernel to reduce precision
cost where safe (typically double -> float, or a software-emulated wider
type on top of a narrower hardware type), while keeping the original
precision where it is not safe to reduce, so that the kernel's output
precision remains within a tolerance.

The tolerance is expressed either as 'sig_figs' (required correct
significant figures of the kernel's output — relative tolerance) or as
'decimal_digits' (required correct decimal places after the point —
absolute tolerance). These are output-precision targets and are
distinct from floating-point storage precision (float / double /
float-float etc.); a kernel can satisfy a 6-sig-fig tolerance with a
mix of storage precisions internally.

You are a router and guardrail, not a numerics expert. Do not decide
per-variable precision yourself — that is the analyst's job. Do not
decide the output-precision tolerance yourself — that is the user's job,
or, when the user did not specify one, the precision_advisor's job.
Your job is to call the right agent at the right time, thread the
tolerance and verdicts through their task prompts faithfully, and
assemble their outputs.

You have access to five specialist agents:
  - precision_advisor: called *only* when the user did not pass an
    output-precision tolerance on the command line. Takes the kernel
    source and returns
      {kind, value, rationale, confidence, alternative}
    where kind is one of 'sig_figs', 'decimal_digits', or 'unknown'.
    Call this exactly once and only at the start, before spawn_analyst,
    and only when the user message tells you no tolerance was provided.

  - analyst: takes the kernel source AND the agreed tolerance, and
    returns a structured per-variable verdict plus an optional
    kernel-shape rework block, a precision_budget block, and overall
    notes. Per-variable entries are
      {name, action, target_precision, emulation_type, reason}
    where action is one of:
      * 'downcast' — replace the declared type with a narrower one
        (target_precision says which, e.g. 'float'); emulation_type empty.
        This is the throughput win.
      * 'emulate'  — replace the declared type with a software-emulated
        pair type (emulation_type says which, e.g. 'float-float');
        target_precision empty. Note: emulate is throughput-NEGATIVE
        and is only justified when downcast violates tolerance AND
        native double is unavailable or weak on the target hardware.
      * 'keep'     — leave the variable unchanged; both target_precision
        and emulation_type empty.
    The rework block is
      {suggested, transformation, rationale, affected_variables}
    and, when suggested=true, names a single kernel-shape transformation
    (e.g. Kahan summation in an accumulator loop) that complements the
    per-variable verdict.
    The precision_budget block is
      {target_kind, target_value, source, claimed_output_precision,
       headroom_argument}
    and links the verdict back to the tolerance.

  - rewriter: takes a single task_prompt string and returns the rewritten
    kernel. It will only change variables the prompt tells it to change
    and only via the method the prompt specifies, so the prompt must
    contain both the kernel source and the analyst's full verdict in a
    form the rewriter can act on.

  - verifier: takes the original source, the rewritten source, the
    analyst's verdict (as a JSON string), and the tolerance, and returns
    {verdict: accept|reject, per_variable: [...], concerns: [...]}.
    It checks faithfulness of the rewrite to the verdict (including any
    suggested rework) and flags concerns when the analyst's precision
    budget looks tight against the tolerance. It does not re-judge
    whether the verdict was numerically correct.

  - baseline_harness: takes the original Kokkos C++ kernel source and a
    short kernel_stem string, and returns a self-contained C++ driver
    program that, when later compiled and run by the operator, exercises
    the kernel on fixed inputs and writes a reproducible reference
    output to ./reference.json. This is a SIDE ARTIFACT for a future
    mechanical comparator — it is not consumed by the analyst, rewriter,
    or verifier in this run, and it is not a precondition for finish.
    Call it at most once per run, and only when the user message's
    BASELINE STEP block invites you to (it only does so for Kokkos C++
    kernels). On approval, the orchestrator writes the driver to
    baselines/<kernel_stem>/driver.cpp; you do not need to manage that.

You also have two deterministic (non-LLM) tools:
  - compile_baseline_driver: takes a kernel_stem and compiles
    baselines/<kernel_stem>/driver.cpp into baselines/<kernel_stem>/
    driver using the local Kokkos install named by the
    AGENT_PRECISION_KOKKOS_ROOT environment variable. Returns
    {status, stdout, stderr, artifacts}. Call this exactly once,
    immediately after a successful spawn_baseline_harness call, and
    using the same kernel_stem. Do not call it if spawn_baseline_harness
    was skipped or rejected. Like the baseline itself, the compiled
    driver is a side artifact: it is not a precondition for finish, and
    a compile error there must NOT block the analyst -> rewriter ->
    verifier pipeline.

  - run_baseline_driver: takes a kernel_stem and executes
    baselines/<kernel_stem>/driver, then verifies that it produced a
    parseable baselines/<kernel_stem>/reference.json. Subject to a
    wall-clock timeout configured via the
    AGENT_PRECISION_RUN_TIMEOUT_SEC environment variable (default 60s).
    Returns {status, stdout, stderr, artifacts}. Call this exactly
    once, immediately after a successful compile_baseline_driver call,
    and using the same kernel_stem. Do not call it if
    compile_baseline_driver was skipped, rejected, or returned an error.
    The reference output is another side artifact: it is not a
    precondition for finish, and a run error there must NOT block the
    analyst -> rewriter -> verifier pipeline.

  - splice_rewritten_kernel: takes a kernel_stem and the rewriter's
    rewritten kernel source. Reads baselines/<kernel_stem>/driver.cpp
    (the baseline driver written by spawn_baseline_harness), replaces
    the text strictly between the '// ---- KERNEL BEGIN ----' and
    '// ---- KERNEL END ----' sentinel lines with the rewritten source,
    and writes the result to baselines/<kernel_stem>/rewritten/
    driver.cpp. Returns {status, stdout, stderr, artifacts}. Call this
    at most once per accepted verifier verdict, immediately after a
    successful run_baseline_driver call AND a successful spawn_verifier
    call with verdict='accept', using the same kernel_stem. Do not call
    it if any prior step in the baseline chain (spawn_baseline_harness,
    compile_baseline_driver, run_baseline_driver) was skipped, rejected,
    or returned an error. The spliced driver is a precursor for a
    future mechanical comparator; like the baseline chain, a splice
    error must NOT block finish on its own.

  - compile_rewritten_driver: takes a kernel_stem and compiles
    baselines/<kernel_stem>/rewritten/driver.cpp (produced by a prior
    splice_rewritten_kernel call) into baselines/<kernel_stem>/
    rewritten/driver, using the same AGENT_PRECISION_KOKKOS_ROOT
    install and the same flags as compile_baseline_driver. Returns
    {status, stdout, stderr, artifacts}. Call this exactly once per
    accepted verifier verdict, immediately after a successful
    splice_rewritten_kernel call, with the same kernel_stem. Do not
    call it if splice_rewritten_kernel was skipped, rejected, or
    returned an error. The compiled rewritten driver is a precursor
    for a future mechanical comparator; like the splice step, a
    compile error here must NOT block finish on its own.

  - run_rewritten_driver: takes a kernel_stem and executes
    baselines/<kernel_stem>/rewritten/driver (produced by a prior
    compile_rewritten_driver call) with cwd set to
    baselines/<kernel_stem>/rewritten/, then verifies that it
    produced a parseable baselines/<kernel_stem>/rewritten/reference.json.
    Subject to the same AGENT_PRECISION_RUN_TIMEOUT_SEC wall-clock
    timeout as run_baseline_driver. Returns
    {status, stdout, stderr, artifacts}. Call this exactly once per
    accepted verifier verdict, immediately after a successful
    compile_rewritten_driver call, with the same kernel_stem. Do not
    call it if compile_rewritten_driver was skipped, rejected, or
    returned an error. The rewritten reference output is the input to
    compare_outputs; a run error here means the comparator cannot
    proceed and finish will be blocked for .cpp kernels until
    compare_outputs has successfully run.

  - compare_outputs: takes a kernel_stem and a tolerance_json (a JSON
    string with the same {kind, value, source} shape that
    spawn_verifier received) and numerically compares
    baselines/<kernel_stem>/reference.json (baseline) against
    baselines/<kernel_stem>/rewritten/reference.json (rewritten)
    under the supplied tolerance. Writes a
    baselines/<kernel_stem>/rewritten/comparison.json artifact on
    both pass and fail paths. Returns
    {status, stdout, stderr, artifacts}, with status='ok' iff every
    compared value agrees under the tolerance and no shape mismatch
    was detected. Call exactly once per accepted verifier verdict,
    immediately after a successful run_rewritten_driver, with the
    same kernel_stem and the same tolerance_json that was passed to
    spawn_verifier. Unlike the other rewritten-chain tools, this IS a
    precondition for finish on .cpp (Kokkos) kernels: the
    orchestrator loop will refuse a finish call until compare_outputs
    has returned status='ok' for the current rewrite cycle. If
    compare_outputs returns an error, the numerical mismatch usually
    indicates the verifier's verdict was wrong rather than just the
    implementation; spawn_analyst (not spawn_rewriter) is typically
    the right retry. .cu (CUDA) kernels skip this whole chain and
    finish remains gated only on the verifier verdict.

You also have a finish tool to emit the final answer.

Tolerance handling:
- The user message will tell you either a concrete tolerance
  ({kind, value, source='user_cli'}) or that no tolerance was given.
- If no tolerance was given: call spawn_precision_advisor first. If the
  advisor returns kind='sig_figs' or 'decimal_digits', use that value
  with source='precision_advisor' as the agreed tolerance for the rest
  of the run. If the advisor returns kind='unknown', fall back to the
  default {kind:'sig_figs', value:6, source:'advisor_unknown_defaulted'}
  and use that as the agreed tolerance.
- Once an agreed tolerance is fixed, thread it verbatim into the
  task prompts of analyst, rewriter, and verifier. The analyst MUST
  see {target_kind, target_value, source}; the rewriter SHOULD see the
  tolerance for context; the verifier MUST see the same tolerance the
  analyst saw so it can audit the precision_budget block.

Your job after the tolerance is fixed:
1. Call spawn_analyst with a kernel_source argument that contains the
   kernel and a clearly-labeled tolerance block (target_kind,
   target_value, source). The analyst will fill precision_budget from
   that block.
2. Translate the analyst's verdict into a self-contained task_prompt for
   the rewriter. The prompt must include the full kernel source, the
   agreed tolerance, and, for each variable, the analyst's chosen
   method (downcast / emulate / keep) together with target_precision or
   emulation_type as applicable. If the analyst's rework.suggested is
   true, include the transformation, rationale, and affected_variables
   verbatim and tell the rewriter to apply that transformation in
   addition to the per-variable changes. Do not editorialize —
   faithfully convey the analyst's calls and do not choose a method the
   analyst did not ask for.
3. Call spawn_rewriter with that task_prompt.
4. Call spawn_verifier with (original_source, rewritten_source from the
   rewriter, analyst_verdict_json, tolerance_json). The
   analyst_verdict_json argument must be the analyst's full result
   object serialized as a JSON string. The tolerance_json argument
   must be the agreed tolerance serialized as a JSON string.
5. If the verifier returns verdict='accept', call finish with the
   rewritten code. If verdict='reject', either call spawn_rewriter again
   with a task_prompt that incorporates the verifier's per-variable
   mismatches and concerns, or — if the verifier's `concerns` implicate
   the analyst's verdict itself — call spawn_analyst again with the
   same tolerance. After any re-run, you must call spawn_verifier again
   on the new rewrite before calling finish.

Hard rules:
- You may not call finish unless the most recent spawn_verifier call
  returned verdict='accept'. On .cpp (Kokkos) inputs, you must ALSO
  have run compare_outputs after the most recent verifier-accept and
  received status='ok' from it; the orchestrator loop enforces this
  in code, not just in the prompt, and a premature finish call will
  be turned into a synthetic tool error telling you what is missing.
- You may not call spawn_precision_advisor if the user message
  provided a tolerance. You may not call spawn_precision_advisor more
  than once.

Be deliberate. Each spawn_* call costs another model call and the user
will inspect every prompt before it runs. Prefer one well-crafted prompt
over several short ones."""

ORCHESTRATOR_TOOLS = [
    {
        "name": "spawn_precision_advisor",
        "description": (
            "Run the precision_advisor agent on a kernel source to infer "
            "an output-precision tolerance. Use only when the user did not "
            "specify a tolerance on the command line. Returns "
            "{kind, value, rationale, confidence, alternative}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kernel_source": {
                    "type": "string",
                    "description": (
                        "The full kernel source. Do not include file paths, "
                        "framing hints, or any other text — the advisor "
                        "should see only the source."
                    ),
                },
            },
            "required": ["kernel_source"],
        },
    },
    {
        "name": "spawn_analyst",
        "description": (
            "Run the analyst agent on a kernel source AND a tolerance. "
            "The kernel_source argument must contain both the kernel and a "
            "clearly-labeled tolerance block (target_kind, target_value, "
            "source). Returns the analyst's per-variable verdict, rework "
            "block, precision_budget, and overall_notes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kernel_source": {
                    "type": "string",
                    "description": (
                        "The full kernel source with a clearly-labeled "
                        "tolerance block prepended or appended (containing "
                        "target_kind, target_value, source). The analyst "
                        "must see the tolerance so it can fill in "
                        "precision_budget. Do not include file paths or "
                        "other framing."
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
            "the analyst's verdict, audits the precision_budget against "
            "the tolerance, and returns "
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
                        "to the analyst (without the tolerance block; the "
                        "verifier reconstructs that from tolerance_json)."
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
                "tolerance_json": {
                    "type": "string",
                    "description": (
                        "The agreed output-precision tolerance as a JSON "
                        "string with keys {kind, value, source}, matching "
                        "what the analyst was given. Use kind='sig_figs' "
                        "or 'decimal_digits'; source is 'user_cli', "
                        "'precision_advisor', or 'advisor_unknown_defaulted'."
                    ),
                },
            },
            "required": [
                "original_source",
                "rewritten_source",
                "analyst_verdict_json",
                "tolerance_json",
            ],
        },
    },
    {
        "name": "spawn_baseline_harness",
        "description": (
            "Run the baseline_harness agent to generate a self-contained "
            "Kokkos C++ driver that, when later compiled and run by the "
            "operator, exercises the kernel on fixed inputs and writes a "
            "reproducible reference output to ./reference.json. Side "
            "artifact for a future mechanical comparator; not consumed by "
            "the other agents in this run, not a precondition for finish. "
            "Call at most once per run, and only when the user message's "
            "BASELINE STEP block invites you to. On HITL approval, the "
            "orchestrator writes the driver to "
            "baselines/<kernel_stem>/driver.cpp."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kernel_source": {
                    "type": "string",
                    "description": (
                        "The full original kernel source. Do not include "
                        "the tolerance block, file paths, or other "
                        "framing — the harness agent should see only the "
                        "kernel code (optionally preceded by a single "
                        "TARGET KERNEL: line if you need to disambiguate "
                        "which function to call)."
                    ),
                },
                "kernel_stem": {
                    "type": "string",
                    "description": (
                        "Filesystem-safe short name for this kernel "
                        "(typically the input file stem). The "
                        "orchestrator writes the approved driver to "
                        "baselines/<kernel_stem>/driver.cpp. Use "
                        "exactly the KERNEL STEM value given in the "
                        "user message."
                    ),
                },
            },
            "required": ["kernel_source", "kernel_stem"],
        },
    },
    {
        "name": "compile_baseline_driver",
        "description": (
            "Deterministic (non-LLM) tool. Compiles "
            "baselines/<kernel_stem>/driver.cpp (produced by a prior "
            "spawn_baseline_harness call) into baselines/<kernel_stem>/"
            "driver, linking against the local Kokkos install named by "
            "the AGENT_PRECISION_KOKKOS_ROOT environment variable. "
            "Returns {status, stdout, stderr, artifacts}. Call exactly "
            "once per run, immediately after a successful "
            "spawn_baseline_harness, with the same kernel_stem. The "
            "compiled driver is a side artifact and not a precondition "
            "for finish."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kernel_stem": {
                    "type": "string",
                    "description": (
                        "Filesystem-safe short name for this kernel; "
                        "MUST match the kernel_stem passed to the "
                        "preceding spawn_baseline_harness call."
                    ),
                },
            },
            "required": ["kernel_stem"],
        },
    },
    {
        "name": "run_baseline_driver",
        "description": (
            "Deterministic (non-LLM) tool. Executes "
            "baselines/<kernel_stem>/driver (produced by a prior "
            "compile_baseline_driver call) with cwd set to "
            "baselines/<kernel_stem>/, so the driver writes "
            "./reference.json next to itself. Validates that the "
            "resulting reference.json is parseable JSON. Subject to a "
            "wall-clock timeout from AGENT_PRECISION_RUN_TIMEOUT_SEC "
            "(default 60s). Returns {status, stdout, stderr, "
            "artifacts}. Call exactly once per run, immediately after "
            "a successful compile_baseline_driver, with the same "
            "kernel_stem. The reference output is a side artifact and "
            "not a precondition for finish."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kernel_stem": {
                    "type": "string",
                    "description": (
                        "Filesystem-safe short name for this kernel; "
                        "MUST match the kernel_stem passed to the "
                        "preceding compile_baseline_driver call."
                    ),
                },
            },
            "required": ["kernel_stem"],
        },
    },
    {
        "name": "splice_rewritten_kernel",
        "description": (
            "Deterministic (non-LLM) tool. Reads "
            "baselines/<kernel_stem>/driver.cpp (produced by a prior "
            "spawn_baseline_harness call), replaces the text strictly "
            "between the '// ---- KERNEL BEGIN ----' and "
            "'// ---- KERNEL END ----' sentinel lines with the supplied "
            "rewritten_kernel_source, and writes the spliced result to "
            "baselines/<kernel_stem>/rewritten/driver.cpp. Returns "
            "{status, stdout, stderr, artifacts}. Call at most once per "
            "accepted verifier verdict, immediately after a successful "
            "run_baseline_driver AND a successful spawn_verifier with "
            "verdict='accept', using the same kernel_stem. The spliced "
            "driver is a precursor for a future mechanical comparator "
            "and is not a precondition for finish."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kernel_stem": {
                    "type": "string",
                    "description": (
                        "Filesystem-safe short name for this kernel; "
                        "MUST match the kernel_stem passed to the "
                        "preceding spawn_baseline_harness call."
                    ),
                },
                "rewritten_kernel_source": {
                    "type": "string",
                    "description": (
                        "The rewritten kernel source produced by the "
                        "most recent spawn_rewriter call that the "
                        "verifier accepted. This text replaces the "
                        "baseline kernel body between the splice "
                        "sentinels; do not include the sentinel lines "
                        "themselves, and do not wrap in code fences."
                    ),
                },
            },
            "required": ["kernel_stem", "rewritten_kernel_source"],
        },
    },
    {
        "name": "compile_rewritten_driver",
        "description": (
            "Deterministic (non-LLM) tool. Compiles "
            "baselines/<kernel_stem>/rewritten/driver.cpp (produced by a "
            "prior splice_rewritten_kernel call) into "
            "baselines/<kernel_stem>/rewritten/driver, using the same "
            "AGENT_PRECISION_KOKKOS_ROOT install and the same compile "
            "flags as compile_baseline_driver. Returns "
            "{status, stdout, stderr, artifacts}. Call exactly once per "
            "accepted verifier verdict, immediately after a successful "
            "splice_rewritten_kernel, with the same kernel_stem. The "
            "compiled rewritten driver is a precursor for a future "
            "mechanical comparator and is not a precondition for finish."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kernel_stem": {
                    "type": "string",
                    "description": (
                        "Filesystem-safe short name for this kernel; "
                        "MUST match the kernel_stem passed to the "
                        "preceding splice_rewritten_kernel call."
                    ),
                },
            },
            "required": ["kernel_stem"],
        },
    },
    {
        "name": "run_rewritten_driver",
        "description": (
            "Deterministic (non-LLM) tool. Executes "
            "baselines/<kernel_stem>/rewritten/driver (produced by a "
            "prior compile_rewritten_driver call) with cwd set to "
            "baselines/<kernel_stem>/rewritten/, so the driver writes "
            "./reference.json next to itself. Validates that the "
            "resulting reference.json is parseable JSON. Subject to a "
            "wall-clock timeout from AGENT_PRECISION_RUN_TIMEOUT_SEC "
            "(default 60s) — same env contract as run_baseline_driver. "
            "Returns {status, stdout, stderr, artifacts}. Call exactly "
            "once per accepted verifier verdict, immediately after a "
            "successful compile_rewritten_driver, with the same "
            "kernel_stem. The rewritten reference output is the input "
            "to compare_outputs; without it, compare_outputs cannot "
            "run, so on .cpp inputs a run error here transitively "
            "blocks finish until the chain is repaired."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kernel_stem": {
                    "type": "string",
                    "description": (
                        "Filesystem-safe short name for this kernel; "
                        "MUST match the kernel_stem passed to the "
                        "preceding compile_rewritten_driver call."
                    ),
                },
            },
            "required": ["kernel_stem"],
        },
    },
    {
        "name": "compare_outputs",
        "description": (
            "Deterministic (non-LLM) tool. Numerically compares "
            "baselines/<kernel_stem>/reference.json (baseline) against "
            "baselines/<kernel_stem>/rewritten/reference.json "
            "(rewritten) under the agreed output-precision tolerance. "
            "Writes baselines/<kernel_stem>/rewritten/comparison.json "
            "on BOTH pass and fail paths so the operator always has a "
            "machine-readable record of the most recent decision. "
            "Returns {status, stdout, stderr, artifacts}, with "
            "status='ok' iff every compared value agrees under the "
            "tolerance and no shape mismatch was detected. Call "
            "exactly once per accepted verifier verdict, immediately "
            "after a successful run_rewritten_driver, with the same "
            "kernel_stem and the same tolerance_json that was passed "
            "to spawn_verifier. This IS a precondition for finish on "
            ".cpp (Kokkos) kernels: the orchestrator loop refuses a "
            "finish call until compare_outputs has returned "
            "status='ok' for the current rewrite cycle. On a "
            "comparator error, spawn_analyst (not spawn_rewriter) is "
            "typically the right retry, because a numerical mismatch "
            "usually indicates the verifier's verdict was wrong "
            "rather than just the implementation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kernel_stem": {
                    "type": "string",
                    "description": (
                        "Filesystem-safe short name for this kernel; "
                        "MUST match the kernel_stem passed to the "
                        "preceding run_rewritten_driver call."
                    ),
                },
                "tolerance_json": {
                    "type": "string",
                    "description": (
                        "The same {kind, value, source} JSON string "
                        "passed to spawn_verifier on this rewrite "
                        "cycle. The comparator parses it and uses "
                        "kind ('sig_figs' or 'decimal_digits') and "
                        "value (positive integer) to decide the "
                        "per-value pass/fail threshold."
                    ),
                },
            },
            "required": ["kernel_stem", "tolerance_json"],
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
    if tool_name == "spawn_precision_advisor":
        result = run_agent("precision_advisor", tool_input["kernel_source"])
        return {"status": "ok", "result": result}
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
            f"{tool_input['analyst_verdict_json']}\n\n"
            "TOLERANCE (JSON):\n"
            f"{tool_input['tolerance_json']}\n"
        )
        result = run_agent("verifier", task)
        return {"status": "ok", "result": result}
    if tool_name == "spawn_baseline_harness":
        result = run_agent("baseline_harness", tool_input["kernel_source"])
        # Side artifact: persist the driver next to its kernel stem so the
        # operator can `cd baselines/<stem>/` and compile it. This is the
        # first _execute_tool branch that touches the filesystem; the path
        # is computed from the orchestrator-supplied kernel_stem (not from
        # the agent's output) so a misbehaving agent cannot redirect the
        # write.
        kernel_stem = tool_input["kernel_stem"]
        driver_dir = Path("baselines") / kernel_stem
        driver_dir.mkdir(parents=True, exist_ok=True)
        driver_path = driver_dir / "driver.cpp"
        driver_path.write_text(result["driver_source"])
        return {
            "status": "ok",
            "result": result,
            "driver_path": str(driver_path),
        }
    if tool_name == "compile_baseline_driver":
        # Deterministic tool (no LLM call): shells out to g++ against
        # the Kokkos install named by AGENT_PRECISION_KOKKOS_ROOT and
        # returns a {status, stdout, stderr, artifacts} dict verbatim
        # (no extra wrapping), so the orchestrator sees the same shape
        # future remote-batch verifier tools will return.
        return compile_baseline_driver(tool_input["kernel_stem"])
    if tool_name == "run_baseline_driver":
        # Deterministic tool (no LLM call): executes the previously-
        # compiled baselines/<stem>/driver with cwd set to that
        # directory so ./reference.json lands beside it. Subject to
        # AGENT_PRECISION_RUN_TIMEOUT_SEC. Returns the same
        # {status, stdout, stderr, artifacts} shape verbatim.
        return run_baseline_driver(tool_input["kernel_stem"])
    if tool_name == "splice_rewritten_kernel":
        # Deterministic tool (no LLM call): pure text I/O. Splices the
        # rewriter's kernel source into a copy of the baseline driver
        # between the KERNEL BEGIN / KERNEL END sentinels and writes
        # baselines/<stem>/rewritten/driver.cpp. Returns the same
        # {status, stdout, stderr, artifacts} shape verbatim.
        return splice_rewritten_kernel(
            tool_input["kernel_stem"],
            tool_input["rewritten_kernel_source"],
        )
    if tool_name == "compile_rewritten_driver":
        # Deterministic tool (no LLM call): same g++ invocation as
        # compile_baseline_driver but targeting the spliced source at
        # baselines/<stem>/rewritten/driver.cpp -> .../rewritten/driver.
        # Returns the same {status, stdout, stderr, artifacts} shape
        # verbatim.
        return compile_rewritten_driver(tool_input["kernel_stem"])
    if tool_name == "run_rewritten_driver":
        # Deterministic tool (no LLM call): same subprocess shape as
        # run_baseline_driver but cwd=baselines/<stem>/rewritten/, so
        # the rewritten driver's ./reference.json lands inside the
        # rewritten subtree and the baseline tree is never touched.
        # Returns the same {status, stdout, stderr, artifacts} shape
        # verbatim.
        return run_rewritten_driver(tool_input["kernel_stem"])
    if tool_name == "compare_outputs":
        # Deterministic tool (no LLM call): pure file + arithmetic I/O,
        # no subprocess. Reads the two reference.json files, walks
        # outputs/ under the supplied tolerance, writes a
        # comparison.json artifact, returns the same
        # {status, stdout, stderr, artifacts} shape verbatim. The
        # status of this call is what the finish-gate guard in
        # run_orchestrator reads when deciding whether to honor a
        # finish call on a .cpp kernel.
        return compare_outputs(
            tool_input["kernel_stem"],
            tool_input["tolerance_json"],
        )
    raise ValueError(f"Unknown tool: {tool_name}")


def _format_tolerance_block(tolerance: dict | None) -> str:
    """Render the tolerance block embedded in the initial user message.

    `tolerance` is either None (no user-supplied tolerance; the
    orchestrator must call spawn_precision_advisor first) or a dict
    with keys {kind, value, source}.
    """
    if tolerance is None:
        return (
            "OUTPUT-PRECISION TOLERANCE: not specified by the user.\n"
            "You MUST call spawn_precision_advisor first with the kernel "
            "source. Then:\n"
            "  - if the advisor returns kind='sig_figs' or "
            "'decimal_digits', use {kind, value, source='precision_advisor'} "
            "as the agreed tolerance;\n"
            "  - if the advisor returns kind='unknown', use the documented "
            "fallback {kind:'sig_figs', value:6, "
            "source:'advisor_unknown_defaulted'} as the agreed tolerance.\n"
            "Thread the agreed tolerance verbatim into the analyst, "
            "rewriter, and verifier task prompts."
        )
    return (
        "OUTPUT-PRECISION TOLERANCE (user-supplied; do NOT call "
        "spawn_precision_advisor):\n"
        f"  kind:   {tolerance['kind']}\n"
        f"  value:  {tolerance['value']}\n"
        f"  source: {tolerance['source']}\n"
        "Thread this tolerance verbatim into the analyst, rewriter, and "
        "verifier task prompts."
    )


def _format_baseline_block(kernel_path: str, kernel_name: str | None) -> str:
    """Render the BASELINE STEP block embedded in the initial user message.

    The baseline_harness agent v0 is Kokkos-only: the orchestrator is
    invited to call spawn_baseline_harness only when the kernel file is
    a .cpp. For .cu (CUDA) inputs, the block tells the orchestrator
    explicitly not to call it. The block also surfaces the kernel_stem
    the orchestrator must pass to the tool (so the driver lands at
    baselines/<stem>/driver.cpp), and, if the caller provided one, the
    target kernel function name.
    """
    stem = Path(kernel_path).stem
    suffix = Path(kernel_path).suffix.lower()
    if suffix != ".cpp":
        return (
            "BASELINE STEP: skipped for this kernel (not a Kokkos .cpp "
            "file). Do NOT call spawn_baseline_harness."
        )
    target_line = (
        f"TARGET KERNEL: {kernel_name}\n" if kernel_name else ""
    )
    return (
        "BASELINE STEP: this is a Kokkos C++ kernel, so you MAY call "
        "spawn_baseline_harness exactly once to generate a reference "
        "driver. This is a side artifact for a future mechanical "
        "comparator; it is NOT consumed by the analyst, rewriter, or "
        "verifier in this run, and it is NOT a precondition for finish. "
        "Skip it if it seems unhelpful.\n"
        f"KERNEL STEM: {stem}\n"
        f"{target_line}"
        "When you call spawn_baseline_harness, pass the original kernel "
        "source as kernel_source (no tolerance block; you MAY prepend a "
        "single TARGET KERNEL: line if one is given above) and the "
        "KERNEL STEM verbatim as kernel_stem. If (and only if) "
        "spawn_baseline_harness succeeds, follow it immediately with a "
        "single call to compile_baseline_driver using the same "
        "KERNEL STEM. If (and only if) compile_baseline_driver returns "
        "status='ok', follow it immediately with a single call to "
        "run_baseline_driver using the same KERNEL STEM. A non-zero "
        "compile or run result must NOT block the rest of the pipeline.\n"
        "Later, after spawn_verifier returns verdict='accept' AND the "
        "preceding run_baseline_driver returned status='ok', call "
        "splice_rewritten_kernel exactly once with the same KERNEL STEM "
        "and the rewriter's accepted rewritten kernel source as "
        "rewritten_kernel_source. If (and only if) splice_rewritten_kernel "
        "returns status='ok', follow it immediately with a single call to "
        "compile_rewritten_driver using the same KERNEL STEM. If (and only "
        "if) compile_rewritten_driver returns status='ok', follow it "
        "immediately with a single call to run_rewritten_driver using the "
        "same KERNEL STEM. If (and only if) run_rewritten_driver returns "
        "status='ok', follow it immediately with a single call to "
        "compare_outputs using the same KERNEL STEM and the same "
        "tolerance_json string you passed to spawn_verifier. "
        "compare_outputs IS a precondition for finish on this .cpp "
        "kernel: a finish call before compare_outputs has returned "
        "status='ok' will be turned into a synthetic tool error. A "
        "splice, rewritten-compile, or rewritten-run error means the "
        "comparator cannot run, so the chain must be repaired before "
        "finish. If compare_outputs returns an error, prefer spawn_analyst "
        "(not spawn_rewriter) for the retry — a numerical mismatch "
        "usually indicates the verifier's verdict was wrong rather "
        "than just the implementation."
    )


class _FinishGateState:
    """Per-run state that decides whether a `finish` tool call is allowed.

    The orchestrator's hard rule (prompt-visible AND enforced here in
    code) is:

      - For .cpp (Kokkos) inputs, `finish` requires the most recent
        spawn_verifier call to have returned verdict='accept' AND the
        most recent compare_outputs call to have returned status='ok'
        for the CURRENT rewrite cycle.
      - For .cu (CUDA) inputs the dynamic-verification chain is
        skipped entirely; `finish` is gated only on the most recent
        spawn_verifier verdict being 'accept' (the same rule the
        system prompt has always carried, now backed by code).

    "Current rewrite cycle" is the key constraint: any spawn_rewriter
    call invalidates BOTH tracked statuses (a new rewrite obviously
    invalidates the verifier verdict that approved the previous
    rewrite, and equally obviously invalidates a comparator result
    that was computed against the previous rewritten reference.json).
    Any subsequent step in the dynamic-verification chain
    (splice_rewritten_kernel, compile_rewritten_driver,
    run_rewritten_driver) also invalidates the comparator status,
    because each of those steps overwrites a file the comparator
    later reads. Only compare_outputs writes last_compare_status, and
    only spawn_verifier writes last_verifier_verdict.

    On a gate violation the orchestrator loop turns the finish call
    into a synthetic {status:'error', stderr:<missing_steps>}
    tool_result that the orchestrator sees on its next turn and can
    self-correct from, instead of silently letting finish through.
    """

    def __init__(self, kernel_path: str) -> None:
        self.kernel_path = kernel_path
        self.requires_comparator = kernel_path.endswith(".cpp")
        self.last_verifier_verdict: str | None = None
        self.last_compare_status: str | None = None

    def observe(self, tool_name: str, exec_result: dict) -> None:
        """Update tracked state given the tool that just ran."""
        if tool_name == "spawn_rewriter":
            # New rewrite cycle starts here. Any prior verifier accept
            # was approving the PREVIOUS rewrite, not this one; any
            # prior comparator result was reading the PREVIOUS
            # rewritten reference.json.
            self.last_verifier_verdict = None
            self.last_compare_status = None
            return
        if tool_name in {
            "splice_rewritten_kernel",
            "compile_rewritten_driver",
            "run_rewritten_driver",
        }:
            # Each of these overwrites or invalidates a file the
            # comparator reads. The verifier verdict is unaffected
            # (the verifier read source text, not the run artifact).
            self.last_compare_status = None
            return
        if tool_name == "spawn_verifier":
            if exec_result.get("status") == "ok":
                verdict = exec_result.get("result", {}).get("verdict")
                self.last_verifier_verdict = verdict
            return
        if tool_name == "compare_outputs":
            self.last_compare_status = exec_result.get("status")
            return
        # Other tools (spawn_precision_advisor, spawn_analyst,
        # spawn_baseline_harness, compile_baseline_driver,
        # run_baseline_driver) do not affect the gate.
        return

    def check_finish(self) -> str | None:
        """Return None if `finish` is allowed, else a missing-steps message."""
        if self.last_verifier_verdict != "accept":
            return (
                "finish is not allowed: the most recent spawn_verifier "
                "call did not return verdict='accept' (current state: "
                f"verifier_verdict={self.last_verifier_verdict!r}). "
                "Call spawn_verifier on the current rewritten kernel "
                "(re-running spawn_rewriter and/or spawn_analyst first "
                "if needed) before calling finish."
            )
        if self.requires_comparator and self.last_compare_status != "ok":
            return (
                "finish is not allowed: on a .cpp (Kokkos) input, "
                "finish requires the dynamic-verification chain to "
                "have ended with compare_outputs returning "
                "status='ok' for the current rewrite cycle "
                "(current state: compare_status="
                f"{self.last_compare_status!r}). Run "
                "splice_rewritten_kernel -> compile_rewritten_driver "
                "-> run_rewritten_driver -> compare_outputs (with the "
                "same kernel_stem and the same tolerance_json passed "
                "to spawn_verifier) before calling finish. If "
                "compare_outputs has already returned an error, the "
                "numerical mismatch usually indicates the verifier's "
                "verdict was wrong rather than just the "
                "implementation, so spawn_analyst (not "
                "spawn_rewriter) is typically the right retry."
            )
        return None


def run_orchestrator(
    kernel_path: str,
    kernel_source: str,
    tolerance: dict | None = None,
    kernel_name: str | None = None,
    max_turns: int = MAX_TURNS,
) -> dict | None:
    """Run the orchestrator loop.

    `tolerance` is either None (no user-supplied tolerance; the
    orchestrator will be instructed to call spawn_precision_advisor
    first) or a dict {kind, value, source} where kind is one of
    'sig_figs' or 'decimal_digits', value is a small positive integer,
    and source is 'user_cli'.

    `kernel_name` is an optional explicit name of the kernel function
    inside `kernel_source` for the baseline_harness agent to target.
    When None (the common case in v0), the orchestrator tells the
    harness agent to infer the function from the source. CLI does not
    expose this yet.

    Returns the final finish() arguments dict, or None if the user quit,
    the orchestrator stopped without finishing, or max_turns was exhausted.
    """
    tolerance_block = _format_tolerance_block(tolerance)
    baseline_block = _format_baseline_block(kernel_path, kernel_name)
    user_message = (
        f"Kernel file: {kernel_path}\n\n"
        f"Kernel source:\n```\n{kernel_source}\n```\n\n"
        f"{tolerance_block}\n\n"
        f"{baseline_block}\n\n"
        "Rewrite this kernel to reduce precision cost where safe (via "
        "downcast, emulation, or — if warranted — a kernel-shape rework), "
        "so that the rewritten kernel's output stays within the agreed "
        "tolerance above."
    )
    messages: list[dict] = [{"role": "user", "content": user_message}]

    client = anthropic.Anthropic()
    gate = _FinishGateState(kernel_path)

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
                gate_error = gate.check_finish()
                if gate_error is None:
                    finish_args = dict(tu.input)
                    break
                # Gate violation: synthesize a tool_result the
                # orchestrator can self-correct from on its next turn,
                # instead of returning the finish args. This is the
                # source-of-truth enforcement for the rule the system
                # prompt describes; if the two ever disagree, this
                # code wins.
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps({
                        "status": "error",
                        "stdout": "",
                        "stderr": gate_error,
                        "artifacts": [],
                    }),
                    "is_error": True,
                })
                continue
            exec_result = _execute_tool(tu.name, dict(tu.input))
            gate.observe(tu.name, exec_result)
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
