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
import os
import sys
from pathlib import Path

import anthropic

from .aggregator import aggregate_analyst_verdicts
from .languages import LanguageProfile, detect_language
from .run_agent import run_agent, run_agent_ensemble
from .tools import (
    compare_outputs,
    compile_baseline_driver,
    compile_rewritten_driver,
    probe_compare,
    probe_step,
    run_baseline_driver,
    run_rewritten_driver,
    splice_rewritten_kernel,
    syntax_check_driver_source,
)
from .verifier_panel import (
    VERIFIER_LENSES,
    aggregate_verifier_verdicts,
    run_verifier_panel,
)

ORCHESTRATOR_MODEL = "claude-opus-4-7"

# Hard upper bound on orchestrator API turns per run. The HITL pause is the
# primary safety net (the user can press 'q' at any time); this constant is a
# backstop so a misbehaving orchestrator loop cannot run indefinitely if
# left unattended. Raised from 20 to 40 to accommodate the dynamic-
# verification chain (splice -> compile_rewritten -> run_rewritten ->
# compare) appended after the analyst -> rewriter -> verifier loop.
# Raised again from 40 to 60 in v1 to accommodate the precision probe
# (1 baseline harness + 1 baseline compile + 1 baseline run + up to 8
# probe_step cells + 1 probe_compare = up to 12 calls before the
# analyst even starts), plus headroom for a verifier-driven rewriter
# retry (analyst -> rewriter -> verifier -> rewriter -> verifier ->
# splice -> compile -> run -> compare).
MAX_TURNS = 60

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
per-variable precision yourself — that is the analyst's job. The
output-precision tolerance is fixed by the user on the command line
and provided to you verbatim in the initial user message; you do not
decide it and you cannot change it. Your job is to call the right
agent at the right time, thread the tolerance and verdicts through
their task prompts faithfully, and assemble their outputs.

You have access to four specialist agents:
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

  - baseline_harness: takes the original kernel source and a short
    kernel_stem string, and returns a self-contained driver program
    that, when later compiled and run by the operator, exercises the
    kernel on fixed inputs and writes a reproducible reference output
    to ./reference.json. The driver's language matches the kernel's
    language profile (Kokkos C++ -> driver.cpp built with g++; CUDA
    C++ -> driver.cu built with nvcc). This is the first link in the
    dynamic-verification chain that ends in compare_outputs and the
    code-side finish-gate. Call it at most once per run, and only when
    the user message's BASELINE STEP block invites you to (it does so
    for any kernel whose resolved language profile supports dynamic
    verification). On approval, the orchestrator writes the driver to
    baselines/<kernel_stem>/<profile driver filename>; you do not need
    to manage that.

You also have two deterministic (non-LLM) tools:
  - compile_baseline_driver: takes a kernel_stem and compiles the
    baseline driver source under baselines/<kernel_stem>/ into a
    baselines/<kernel_stem>/driver binary, using the toolchain dictated
    by the resolved language profile (g++ + AGENT_PRECISION_KOKKOS_ROOT
    for Kokkos; nvcc + AGENT_PRECISION_CUDA_ARCH for CUDA). Returns
    {status, stdout, stderr, artifacts}. Call this exactly once,
    immediately after a successful spawn_baseline_harness call, and
    using the same kernel_stem. Do not call it if spawn_baseline_harness
    was skipped or rejected. A compile error here transitively blocks
    the rest of the dynamic-verification chain (and therefore finish on
    profiles where the chain is required) but does NOT block the
    analyst -> rewriter -> verifier pipeline itself.

  - run_baseline_driver: takes a kernel_stem and executes
    baselines/<kernel_stem>/driver, then verifies that it produced a
    parseable baselines/<kernel_stem>/reference.json. Subject to a
    wall-clock timeout configured via the
    AGENT_PRECISION_RUN_TIMEOUT_SEC environment variable (default 60s).
    Returns {status, stdout, stderr, artifacts}. Call this exactly
    once, immediately after a successful compile_baseline_driver call,
    and using the same kernel_stem. Do not call it if
    compile_baseline_driver was skipped, rejected, or returned an error.
    Like the compile step, a run error here transitively blocks the
    rest of the dynamic-verification chain (and therefore finish on
    profiles where the chain is required) but does NOT block the
    analyst -> rewriter -> verifier pipeline itself.

  - probe_step: takes a kernel_stem, a precision, and a seed. Reads
    the per-precision driver template at
    baselines/<kernel_stem>/probe/<precision>/driver.cpp (written by
    the v1 baseline_harness alongside the canonical baseline),
    rewrites its 'static constexpr int RNG_SEED = ...;' line to the
    requested seed, then compiles and runs the rewritten driver under
    baselines/<kernel_stem>/probe/<precision>_seed<seed>/. The
    template directory is never touched. Returns
    {status, stdout, stderr, artifacts}. The probe is INFORMATIONAL:
    its output flows into the analyst as evidence (see probe_compare
    below); it is NOT a finish-gate precondition, and a failed cell
    does NOT block any downstream step. Call this exactly once per
    (precision, seed) cell, and only when the BASELINE STEP block in
    the user message tells you to. The user message lists the exact
    (precision, seed) matrix to drive; do not invent additional
    cells, and do not skip cells the matrix lists. Profiles whose
    BASELINE STEP block does not mention probe_step have no probe
    templates on disk and the tool will hard-error if called.

  - probe_compare: takes a kernel_stem and aggregates whatever
    per-cell probe_step results landed under baselines/<kernel_stem>/
    probe/ into a single baselines/<kernel_stem>/probe/evidence.json
    document. Returns {status, stdout, stderr, artifacts}. Hard-errors
    only when the canonical quad/seed=42 cell is missing (no ground
    truth -> no comparison); any other missing or failed cell is
    recorded with a non-ok status and skipped during the stats walk.
    The aggregated evidence is INFORMATIONAL: the orchestrator loop
    automatically attaches it to the next spawn_analyst call's task
    prompt as a PROBE EVIDENCE (JSON) block — you do NOT need to
    pass it through yourself. Call this exactly once, after all
    probe_step cells the matrix lists have been attempted (succeed or
    fail), and before spawn_analyst.

  - splice_rewritten_kernel: takes a kernel_stem and the rewriter's
    rewritten kernel source. Reads the baseline driver source under
    baselines/<kernel_stem>/ (written by spawn_baseline_harness),
    replaces the text strictly between the '// ---- KERNEL BEGIN ----'
    and '// ---- KERNEL END ----' sentinel lines with the rewritten
    source, and writes the result under baselines/<kernel_stem>/
    rewritten/ with the same driver filename. Returns
    {status, stdout, stderr, artifacts}. Call this at most once per
    accepted verifier verdict, immediately after a successful
    run_baseline_driver call AND a successful spawn_verifier call with
    verdict='accept', using the same kernel_stem. Do not call it if
    any prior step in the baseline chain (spawn_baseline_harness,
    compile_baseline_driver, run_baseline_driver) was skipped,
    rejected, or returned an error. The spliced driver feeds the
    rewritten compile/run/compare chain that the code-side finish-gate
    enforces on profiles with dynamic_verification=True.

  - compile_rewritten_driver: takes a kernel_stem and compiles the
    spliced driver source under baselines/<kernel_stem>/rewritten/
    (produced by a prior splice_rewritten_kernel call) into
    baselines/<kernel_stem>/rewritten/driver, using the same toolchain
    and flags as compile_baseline_driver (so the only intentional
    difference between the baseline and rewritten binaries is the
    kernel source between the sentinels). Returns
    {status, stdout, stderr, artifacts}. Call this exactly once per
    accepted verifier verdict, immediately after a successful
    splice_rewritten_kernel call, with the same kernel_stem. Do not
    call it if splice_rewritten_kernel was skipped, rejected, or
    returned an error. A compile error here transitively blocks the
    rest of the dynamic-verification chain and therefore finish on
    profiles where the chain is required.

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
    proceed and finish will be blocked on profiles with
    dynamic_verification=True until compare_outputs has successfully
    run.

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
    precondition for finish on profiles with dynamic_verification=True
    (currently Kokkos C++ and CUDA C++): the orchestrator loop will
    refuse a finish call until compare_outputs has returned
    status='ok' for the current rewrite cycle. If compare_outputs
    returns an error, the numerical mismatch usually indicates the
    verifier's verdict was wrong rather than just the implementation;
    spawn_analyst (not spawn_rewriter) is typically the right retry.
    Profiles with dynamic_verification=False (none today; reserved for
    languages registered before a baseline harness exists) skip this
    whole chain and finish remains gated only on the verifier verdict.

You also have a finish tool to emit the final answer.

Tolerance handling:
- The user message will tell you a concrete tolerance
  ({kind, value, source='user_cli'}). It is always present; the CLI
  requires the operator to pass --sig-figs or --decimal-digits and
  rejects a run with neither.
- Thread the tolerance verbatim into the task prompts of analyst,
  rewriter, and verifier. The analyst MUST see {target_kind,
  target_value, source}; the rewriter SHOULD see the tolerance for
  context; the verifier MUST see the same tolerance the analyst saw
  so it can audit the precision_budget block.

Your job after the tolerance is fixed:
0. If the user message's BASELINE STEP block invites it (Kokkos C++
   in v1; other languages today silently skip this whole step), run
   spawn_baseline_harness, then compile_baseline_driver, then
   run_baseline_driver. If the BASELINE STEP block additionally lists
   a probe matrix (precision/seed pairs), drive each cell with one
   probe_step call, then call probe_compare exactly once. Failures
   in any baseline or probe call are non-fatal to the analyst ->
   rewriter -> verifier loop; proceed to step 1 in either case.
1. Call spawn_analyst with a kernel_source argument that contains the
   kernel and a clearly-labeled tolerance block (target_kind,
   target_value, source). The analyst will fill precision_budget from
   that block. If probe_compare succeeded, the orchestrator will
   automatically attach the aggregated probe evidence to the task; you
   do not need to thread it through yourself.
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
  returned verdict='accept'. On inputs whose language profile carries
  dynamic_verification=True (currently any Kokkos C++ or CUDA C++
  kernel), you must ALSO have run compare_outputs after the most
  recent verifier-accept and received status='ok' from it; the
  orchestrator loop enforces this in code, not just in the prompt,
  and a premature finish call will be turned into a synthetic tool
  error telling you what is missing.

Be deliberate. Each spawn_* call costs another model call and the user
will inspect every prompt before it runs. Prefer one well-crafted prompt
over several short ones."""

ORCHESTRATOR_TOOLS = [
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
                        "or 'decimal_digits'; source is 'user_cli'."
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
        "name": "probe_step",
        "description": (
            "Deterministic (non-LLM) tool. Runs one (precision, seed) "
            "cell of the precision probe: rewrites the RNG_SEED line of "
            "baselines/<kernel_stem>/probe/<precision>/driver.cpp "
            "(written by spawn_baseline_harness as a template for "
            "seed=42), writes the rewritten source to "
            "baselines/<kernel_stem>/probe/<precision>_seed<seed>/"
            "driver.cpp, compiles it, runs it, and validates the "
            "resulting reference.json. The template directory is never "
            "touched. Returns {status, stdout, stderr, artifacts}. The "
            "probe is INFORMATIONAL for the analyst (it lets the "
            "analyst reason from numerical evidence, not just from "
            "source); it is NOT a finish-gate precondition. Available "
            "only on language profiles whose probe_precisions is "
            "non-empty (currently Kokkos: quad / double / float / "
            "mixed_io). Call once per (precision, seed) cell after a "
            "successful run_baseline_driver and before spawn_analyst; "
            "skip cleanly on profiles where the BASELINE STEP block "
            "does not mention probe_step."
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
                "precision": {
                    "type": "string",
                    "description": (
                        "Which probe driver template to use. Must be a "
                        "key the harness emitted under drivers.* "
                        "(Kokkos: 'quad', 'double', 'float', 'mixed_io')."
                    ),
                },
                "seed": {
                    "type": "integer",
                    "description": (
                        "The RNG seed to bake into the rewritten driver "
                        "source. v1 drives the canonical seed 42 and "
                        "the adjacent seed 43; vary one cell at a time."
                    ),
                },
            },
            "required": ["kernel_stem", "precision", "seed"],
        },
    },
    {
        "name": "probe_compare",
        "description": (
            "Deterministic (non-LLM) tool. Aggregates the per-cell "
            "probe runs under baselines/<kernel_stem>/probe/ into a "
            "single evidence.json document for the analyst: per-output "
            "stats for every non-quad cell against its same-seed quad "
            "ground truth, plus cross-seed deltas so the analyst can "
            "tell seed-correlated precision pain from seed-independent "
            "pain. Cells whose probe_step failed (or was never run) "
            "are recorded with a non-ok status and skipped in the "
            "stats walk; the analyst sees exactly which signals are "
            "real. Returns {status, stdout, stderr, artifacts}. "
            "Hard-errors only when the canonical quad_seed42 cell is "
            "missing — without it there is no ground truth. Call once "
            "after all probe_step cells have been attempted, and "
            "before spawn_analyst. The evidence is INFORMATIONAL: the "
            "analyst sees it but is told to treat it as one input "
            "among several, not as a verdict."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kernel_stem": {
                    "type": "string",
                    "description": (
                        "Filesystem-safe short name for this kernel; "
                        "MUST match the kernel_stem passed to the "
                        "preceding probe_step calls."
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


def _append_trace(
    trace_path: Path | None,
    turn: int,
    tool_name: str,
    tool_input: dict,
    exec_result: dict,
) -> None:
    """Append one JSONL record per executed tool when auto-mode is on.

    No-op when trace_path is None (interactive mode). The record schema is
    {turn, tool_name, tool_input, exec_result} — flat enough for jq filters
    and small enough that a 40-turn run produces a few hundred KiB max.
    Trace writes happen for every executed tool, for synthesized finish-gate
    errors, and for the honored finish call; user rejections cannot occur
    in auto mode, so they are not represented.
    """
    if trace_path is None:
        return
    record = {
        "turn": turn,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "exec_result": exec_result,
    }
    with trace_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


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


def _execute_tool(
    tool_name: str,
    tool_input: dict,
    profile: LanguageProfile,
    kernel_stem: str | None = None,
) -> dict:
    """Actually run the requested tool. Returns the result to feed back.

    `profile` is the LanguageProfile resolved once per run by
    run_orchestrator (via workflow.languages.detect_language). The
    deterministic-tool branches inject `profile.id` as the `language_id`
    arg into the tool wrappers, and the spawn_baseline_harness branch
    uses `profile.driver_filename` when persisting the harness output.
    The LLM never sees or chooses `language_id` — that would be a
    constant per run with no real choice — so this is the single
    chokepoint where the per-run profile meets the per-tool call.

    `kernel_stem` is the per-run kernel stem (Path(kernel_path).stem) so
    the spawn_analyst branch can locate the probe evidence written by
    a prior probe_compare call at baselines/<kernel_stem>/probe/
    evidence.json and inject it into the analyst's task prompt as a
    PROBE EVIDENCE (JSON) block. The LLM does not pass the evidence
    through itself (that would be both fragile and a large token
    duplicate of what is already on disk); the orchestrator attaches
    it on the analyst's behalf. None means "no probe evidence
    available" — kept optional for tests that exercise _execute_tool
    in isolation.
    """
    if tool_name == "spawn_analyst":
        # Probe evidence injection: if a prior probe_compare call wrote
        # baselines/<kernel_stem>/probe/evidence.json, append it to the
        # analyst task as a PROBE EVIDENCE (JSON) block. The block is
        # appended (not prepended) so it sits AFTER the kernel source
        # the LLM passed in -- mirroring the orchestrator's existing
        # convention of putting raw inputs first and metadata blocks
        # second. If evidence.json is absent (no probe was run, or
        # probe_compare failed before writing it) or unreadable, we
        # silently fall back to the un-augmented task: the probe is
        # informational and missing evidence must not block the
        # analyst.
        analyst_task = tool_input["kernel_source"]
        if kernel_stem is not None:
            evidence_path = (
                Path("baselines") / kernel_stem / "probe" / "evidence.json"
            )
            if evidence_path.is_file():
                try:
                    evidence_text = evidence_path.read_text()
                except OSError:
                    evidence_text = None
                if evidence_text is not None:
                    analyst_task = (
                        f"{analyst_task}\n\n"
                        "PROBE EVIDENCE (JSON): the orchestrator ran a "
                        "precision probe on this kernel before invoking "
                        "you. The aggregated evidence below shows, for "
                        "each (precision, seed) cell that succeeded, "
                        "per-output stats against the quad/seed=42 "
                        "ground truth, plus cross-seed deltas. Treat it "
                        "as ONE input among several: corroborate it "
                        "against the source you can see, do not let it "
                        "override your own analysis, and remember that "
                        "a 'no_quad_partner' or 'missing' cell means no "
                        "signal -- not 'precision is safe'.\n"
                        f"{evidence_text}"
                    )
        # Optional self-consistency ensemble: when AGENT_PRECISION_ANALYST_K
        # is > 1, run the analyst K times in parallel at
        # AGENT_PRECISION_ANALYST_T (default 0.7 for genuine diversity)
        # and fold the K verdicts through aggregate_analyst_verdicts.
        # Default K=1 preserves the single-shot behavior so existing runs
        # are unaffected unless the operator opts in. The aggregator
        # output conforms to the analyst schema, so the verifier (which
        # reads analyst_verdict_json downstream) needs no changes.
        k = int(os.environ.get("AGENT_PRECISION_ANALYST_K", "1"))
        if k > 1:
            temperature = float(
                os.environ.get("AGENT_PRECISION_ANALYST_T", "0.7")
            )
            verdicts = run_agent_ensemble(
                "analyst",
                analyst_task,
                k=k,
                temperature=temperature,
            )
            aggregated, report = aggregate_analyst_verdicts(verdicts)
            return {
                "status": "ok",
                "result": aggregated,
                "aggregator_metadata": report,
            }
        result = run_agent("analyst", analyst_task)
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
        # Optional perspective-diverse panel: when
        # AGENT_PRECISION_VERIFIER_K is > 1, run the verifier K times in
        # parallel under K different lenses (faithfulness, budget,
        # edge_cases) at AGENT_PRECISION_VERIFIER_T (default 0.7) and
        # fold the K verdicts through aggregate_verifier_verdicts. K
        # must be <= len(VERIFIER_LENSES); the lenses ARE the panel, so
        # asking for more is a configuration error. Default K=1
        # preserves the single-shot behavior so existing runs are
        # unaffected unless the operator opts in. The aggregator output
        # conforms to the verifier schema, so the finish-gate code that
        # reads result['verdict'] needs no changes.
        k = int(os.environ.get("AGENT_PRECISION_VERIFIER_K", "1"))
        if k > 1:
            if k > len(VERIFIER_LENSES):
                raise ValueError(
                    f"AGENT_PRECISION_VERIFIER_K={k} exceeds the number "
                    f"of defined verifier lenses ({len(VERIFIER_LENSES)}). "
                    "Lenses are the panel, not just a replication "
                    "multiplier; lower K or add a lens to "
                    "verifier_panel.VERIFIER_LENSES."
                )
            temperature = float(
                os.environ.get("AGENT_PRECISION_VERIFIER_T", "0.7")
            )
            lenses = VERIFIER_LENSES[:k]
            lens_names = [lens["name"] for lens in lenses]
            verdicts = run_verifier_panel(task, lenses, temperature)
            aggregated, report = aggregate_verifier_verdicts(
                verdicts, lens_names
            )
            return {
                "status": "ok",
                "result": aggregated,
                "verifier_aggregator_metadata": report,
            }
        result = run_agent("verifier", task)
        return {"status": "ok", "result": result}
    if tool_name == "spawn_baseline_harness":
        # Dispatch to the per-language baseline harness agent. The
        # registry auto-builds one AGENTS entry per registered
        # LanguageProfile under the key `baseline_harness_<id>`, so
        # adding a new language only means registering a profile;
        # nothing in this dispatch site changes. The legacy
        # `baseline_harness` alias still resolves to Kokkos for
        # back-compat with any caller that hits run_agent directly,
        # but the orchestrator itself always goes through the
        # per-language key so the prompt actually matches the
        # language.
        result = run_agent(
            f"baseline_harness_{profile.id}",
            tool_input["kernel_source"],
        )
        # Side artifact: persist the driver(s) next to its kernel stem so
        # the operator can `cd baselines/<stem>/` and compile it. This is
        # the first _execute_tool branch that touches the filesystem; the
        # path is computed from the orchestrator-supplied kernel_stem (not
        # from the agent's output) so a misbehaving agent cannot redirect
        # the write.
        kernel_stem = tool_input["kernel_stem"]
        driver_dir = Path("baselines") / kernel_stem
        # Per-language driver filename (driver.cpp for Kokkos,
        # driver.cu for CUDA). Owned by the LanguageProfile so the
        # orchestrator stays language-agnostic.
        # NOTE: driver_dir.mkdir is deferred to AFTER the syntax-check
        # gate so a gate failure leaves the filesystem untouched — the
        # all-or-nothing contract is what makes the gate safe to retry
        # (no half-written baselines/<stem>/ tree the next probe_step
        # call could silently reuse).
        driver_path = driver_dir / profile.driver_filename
        # Two output shapes coexist: the v0 single-driver schema
        # (`driver_source: str`) is still used by CUDA/HIP/SYCL/OMP-
        # offload profiles; the v1 multi-driver schema (`drivers:
        # {<precision>: str}`) is used by Kokkos (the only profile
        # whose probe_precisions is populated). When `drivers` is
        # present, the canonical splice scaffold at
        # `baselines/<stem>/driver.cpp` = `drivers["double"]`, NOT
        # `drivers[profile.baseline_precision]`. The role split
        # exists because Kokkos has no `__float128` math overloads
        # (`Kokkos::sqrt(__float128)` does not exist), so the quad
        # driver is plain C++ + quadmath (per the kokkos harness
        # prompt's quad bullet) — uncompilable as a splice target
        # for the rewriter's Kokkos kernels. The DOUBLE driver fills
        # the splice-scaffold role; the QUAD driver fills the
        # ground-truth-oracle role and its seed=42 reference.json is
        # promoted to `baselines/<stem>/reference.json` later in the
        # chain (see the probe_compare branch below) so the finish-
        # gate comparator measures against true quad ground truth.
        # The remaining drivers fan out into per-precision probe
        # subdirectories so the probe_step tool can reuse the existing
        # compile/run helpers per directory.
        probe_driver_paths: dict[str, str] = {}
        # TEMPORARY DIAGNOSTIC (remove after we identify why some
        # backends return a payload missing both `drivers` and
        # `driver_source`). Dumps the result keys + a short JSON
        # preview to stderr so the operator can see what the
        # baseline-harness agent actually submitted.
        _preview = {
            k: (v if not isinstance(v, str) else v[:200] + ("..." if len(v) > 200 else ""))
            for k, v in result.items()
        }
        print(
            f"[orchestrator] baseline_harness result keys={list(result.keys())} "
            f"preview={json.dumps(_preview)[:1000]}",
            file=sys.stderr,
        )
        if "drivers" in result:
            drivers = result["drivers"]
            # SPLICE-SCAFFOLD CHOICE: the canonical baseline is the
            # DOUBLE driver (a real Kokkos driver), not the
            # baseline_precision driver. baseline_precision="quad" is
            # the oracle role only — see the role-split comment above
            # for why these can't be the same file.
            canonical = drivers["double"]
            # SYNTAX-CHECK GATE. Validate every driver source in the
            # payload BEFORE we touch the filesystem. If any driver
            # fails to parse under g++ -fsyntax-only, return an
            # is_error tool_result and let the orchestrator's harness
            # re-run see the compiler diagnostic verbatim — the
            # motivating case is the "alias-naming drift" bug seen on
            # a real nbody_force run where the harness declared
            # `vxType vx(...); vyType_v vy(...);` (two different alias
            # conventions in one line), which compiled cleanly enough
            # under the write-first path to reach compile_baseline_
            # driver and only failed there, wasting a full HITL cycle.
            # The gate silently skips when the profile has no check
            # command or KOKKOS_ROOT is unset (see
            # syntax_check_driver_source in tools.py).
            for precision, source in drivers.items():
                gate_err = syntax_check_driver_source(
                    profile, source, f"drivers[{precision!r}]"
                )
                if gate_err is not None:
                    return {**gate_err, "is_error": True}
            driver_dir.mkdir(parents=True, exist_ok=True)
            driver_path.write_text(canonical)
            for precision, source in drivers.items():
                probe_dir = driver_dir / "probe" / precision
                probe_dir.mkdir(parents=True, exist_ok=True)
                probe_path = probe_dir / profile.driver_filename
                probe_path.write_text(source)
                probe_driver_paths[precision] = str(probe_path)
        elif "driver_source" in result:
            gate_err = syntax_check_driver_source(
                profile, result["driver_source"], "driver_source"
            )
            if gate_err is not None:
                return {**gate_err, "is_error": True}
            driver_dir.mkdir(parents=True, exist_ok=True)
            driver_path.write_text(result["driver_source"])
        else:
            raise RuntimeError(
                f"baseline_harness agent for profile {profile.id!r} "
                f"returned a submit_result payload that has neither "
                f"`drivers` (v1 multi-driver schema) nor "
                f"`driver_source` (v0 single-driver schema). "
                f"Keys present: {sorted(result.keys())}. "
                f"This is a schema-enforcement failure upstream of "
                f"this code: either the model returned a malformed "
                f"tool call that the SDK accepted, or the proxy "
                f"stripped a required field. Check the [run_agent] "
                f"stop_reason warning above this line — if it says "
                f"stop_reason='max_tokens', the model truncated "
                f"mid-tool-use and the input dict is empty by "
                f"definition (raise max_tokens in workflow/"
                f"run_agent.py beyond the current 32768). Otherwise "
                f"inspect the diagnostic print above for the full "
                f"payload preview."
            )
        response: dict = {
            "status": "ok",
            "result": result,
            "driver_path": str(driver_path),
        }
        if probe_driver_paths:
            response["probe_driver_paths"] = probe_driver_paths
        return response
    if tool_name == "compile_baseline_driver":
        # Deterministic tool (no LLM call): shells out to the
        # profile-selected compiler (g++ for Kokkos, eventually nvcc
        # for CUDA) against the install named by the profile's
        # env_required vars, and returns a {status, stdout, stderr,
        # artifacts} dict verbatim (no extra wrapping). The same shape
        # future remote-batch verifier tools will return. `profile.id`
        # is injected here so the LLM never has to pass it; the
        # per-run profile is a constant for the entire orchestrator
        # loop.
        return compile_baseline_driver(
            tool_input["kernel_stem"], profile.id
        )
    if tool_name == "run_baseline_driver":
        # Deterministic tool (no LLM call): executes the previously-
        # compiled baselines/<stem>/driver with cwd set to that
        # directory so ./reference.json lands beside it. Subject to
        # AGENT_PRECISION_RUN_TIMEOUT_SEC. Returns the same
        # {status, stdout, stderr, artifacts} shape verbatim.
        return run_baseline_driver(tool_input["kernel_stem"], profile.id)
    if tool_name == "splice_rewritten_kernel":
        # Deterministic tool (no LLM call): pure text I/O. Splices the
        # rewriter's kernel source into a copy of the baseline driver
        # between the profile's KERNEL BEGIN / KERNEL END sentinels and
        # writes baselines/<stem>/rewritten/<profile.driver_filename>.
        # Returns the same {status, stdout, stderr, artifacts} shape
        # verbatim.
        return splice_rewritten_kernel(
            tool_input["kernel_stem"],
            tool_input["rewritten_kernel_source"],
            profile.id,
        )
    if tool_name == "compile_rewritten_driver":
        # Deterministic tool (no LLM call): same compile invocation as
        # compile_baseline_driver but targeting the spliced source at
        # baselines/<stem>/rewritten/<profile.driver_filename> ->
        # .../rewritten/driver. Returns the same {status, stdout,
        # stderr, artifacts} shape verbatim.
        return compile_rewritten_driver(
            tool_input["kernel_stem"], profile.id
        )
    if tool_name == "run_rewritten_driver":
        # Deterministic tool (no LLM call): same subprocess shape as
        # run_baseline_driver but cwd=baselines/<stem>/rewritten/, so
        # the rewritten driver's ./reference.json lands inside the
        # rewritten subtree and the baseline tree is never touched.
        # Returns the same {status, stdout, stderr, artifacts} shape
        # verbatim.
        return run_rewritten_driver(
            tool_input["kernel_stem"], profile.id
        )
    if tool_name == "compare_outputs":
        # Deterministic tool (no LLM call): pure file + arithmetic I/O,
        # no subprocess. Reads the two reference.json files, walks
        # outputs/ under the supplied tolerance, writes a
        # comparison.json artifact, returns the same
        # {status, stdout, stderr, artifacts} shape verbatim. The
        # status of this call is what the finish-gate guard in
        # run_orchestrator reads when deciding whether to honor a
        # finish call on a kernel whose language profile carries
        # dynamic_verification=True.
        return compare_outputs(
            tool_input["kernel_stem"],
            tool_input["tolerance_json"],
            profile.id,
        )
    if tool_name == "probe_step":
        # Deterministic tool (no LLM call): seed-rewrites a per-precision
        # driver template that the v1 baseline_harness wrote under
        # baselines/<stem>/probe/<precision>/, then compiles and runs it
        # under baselines/<stem>/probe/<precision>_seed<seed>/. The
        # template directory is never touched. `profile.id` is injected
        # here so the LLM never has to pass it; only profiles whose
        # probe_precisions is non-empty have working templates to act
        # on (the system prompt's BASELINE STEP block silently omits
        # the probe instructions otherwise). Returns the same
        # {status, stdout, stderr, artifacts} shape verbatim.
        return probe_step(
            tool_input["kernel_stem"],
            tool_input["precision"],
            tool_input["seed"],
            profile.id,
        )
    if tool_name == "probe_compare":
        # Deterministic tool (no LLM call): aggregates the per-cell
        # probe runs into baselines/<stem>/probe/evidence.json. The
        # orchestrator does NOT pass this evidence into spawn_analyst
        # as a tool argument; instead, the spawn_analyst branch below
        # reads evidence.json off disk and appends it to the analyst
        # task prompt as a PROBE EVIDENCE block (mirroring the
        # tolerance / baseline blocks pattern). Returns the same
        # {status, stdout, stderr, artifacts} shape verbatim.
        compare_result = probe_compare(tool_input["kernel_stem"], profile.id)
        # ORACLE PROMOTION: when probe_compare succeeded for a profile
        # whose baseline_precision differs from the canonical splice-
        # scaffold precision (currently only Kokkos:
        # baseline_precision="quad", splice scaffold = "double"),
        # promote the higher-precision probe reference into the
        # canonical baseline slot so the finish-gate comparator
        # (compare_outputs) measures the rewritten kernel against true
        # ground truth instead of against the same-precision double
        # reference that run_baseline_driver wrote earlier in the
        # chain. The destination (baselines/<stem>/reference.json) is
        # the file compare_outputs reads as the baseline; the source
        # is the probe seed=42 cell for baseline_precision. For
        # profiles where baseline_precision == "double" (CUDA / HIP /
        # SYCL / OMP-offload today), this is a no-op. For profiles
        # whose probe pipeline is disabled (probe_precisions=()), the
        # probe_compare branch is never reached.
        if (
            compare_result.get("status") == "ok"
            and profile.baseline_precision != "double"
            and profile.baseline_precision in profile.probe_precisions
        ):
            stem = tool_input["kernel_stem"]
            oracle_src = (
                Path("baselines") / stem / "probe"
                / f"{profile.baseline_precision}_seed42" / "reference.json"
            )
            oracle_dst = Path("baselines") / stem / "reference.json"
            if oracle_src.is_file():
                try:
                    oracle_dst.write_text(oracle_src.read_text())
                    print(
                        f"[orchestrator] oracle promotion: copied "
                        f"{oracle_src} -> {oracle_dst} (profile "
                        f"{profile.id!r}, baseline_precision="
                        f"{profile.baseline_precision!r})",
                        file=sys.stderr,
                    )
                except OSError as exc:
                    # Non-fatal: leave the run_baseline_driver output
                    # in place. The comparator will still run, just
                    # against a lower-precision baseline. We surface
                    # the failure in stderr but don't change
                    # compare_result's status (probe_compare itself
                    # succeeded; this is a downstream artifact).
                    print(
                        f"[orchestrator] oracle promotion FAILED: "
                        f"{exc}. Finish-gate comparator will run "
                        f"against the double-precision baseline.",
                        file=sys.stderr,
                    )
            else:
                print(
                    f"[orchestrator] oracle promotion skipped: "
                    f"{oracle_src} does not exist (the "
                    f"{profile.baseline_precision}_seed42 probe cell "
                    f"may have failed). Finish-gate comparator will "
                    f"run against the double-precision baseline.",
                    file=sys.stderr,
                )
        return compare_result
    raise ValueError(f"Unknown tool: {tool_name}")


def _format_tolerance_block(tolerance: dict) -> str:
    """Render the tolerance block embedded in the initial user message.

    `tolerance` is a dict with keys {kind, value, source}. It is always
    present: the CLI requires --sig-figs or --decimal-digits and rejects
    a run with neither, so run_orchestrator never receives None here.
    """
    return (
        "OUTPUT-PRECISION TOLERANCE (user-supplied):\n"
        f"  kind:   {tolerance['kind']}\n"
        f"  value:  {tolerance['value']}\n"
        f"  source: {tolerance['source']}\n"
        "Thread this tolerance verbatim into the analyst, rewriter, and "
        "verifier task prompts."
    )


def _format_baseline_block(
    kernel_path: str,
    kernel_name: str | None,
    profile: LanguageProfile,
    run_probe: bool = True,
) -> str:
    """Render the BASELINE STEP block embedded in the initial user message.

    The baseline_harness agent v0 is Kokkos-only: the orchestrator is
    invited to call spawn_baseline_harness only when the resolved
    language profile carries `dynamic_verification=True`. Profiles that
    set that flag to False (none today; reserved for languages we add
    before a baseline harness exists for them) get a "skip baseline"
    block instead. Earlier revisions hard-coded this check to
    `suffix == ".cpp"` because the Kokkos harness was the only one
    written; with the CUDA profile (Phase B) the gate now follows the
    profile flag, so `.cu` inputs invite the full chain too. The block
    also surfaces the kernel_stem the orchestrator must pass to the
    tool (so the driver lands at
    baselines/<stem>/<profile.driver_filename>), and, if the caller
    provided one, the target kernel function name.

    `profile.display_name` and `profile.driver_filename` populate the
    prose so the block stays in sync with the resolved language profile
    when new languages land (eg. "CUDA C++" / "driver.cu" instead of
    "Kokkos C++" / "driver.cpp").
    """
    stem = Path(kernel_path).stem
    if not profile.dynamic_verification:
        return (
            f"BASELINE STEP: skipped for this kernel (the {profile.display_name} "
            "profile does not yet support dynamic verification). Do NOT "
            "call spawn_baseline_harness."
        )
    target_line = (
        f"TARGET KERNEL: {kernel_name}\n" if kernel_name else ""
    )
    # Probe matrix is silently omitted on profiles whose
    # probe_precisions tuple is empty (every profile other than
    # Kokkos in v1). The two probe tools still exist in
    # ORCHESTRATOR_TOOLS, but without explicit instructions naming
    # the matrix the orchestrator should not invoke them on a
    # non-probing profile; if it does, probe_step's preflight
    # ("template not found") cleanly errors.
    if profile.probe_precisions and run_probe:
        from .tools import _PROBE_SEEDS  # local import: private helper
        precisions_list = ", ".join(repr(p) for p in profile.probe_precisions)
        seeds_list = ", ".join(str(s) for s in _PROBE_SEEDS)
        cells = [
            f"({precision!r}, {seed})"
            for seed in _PROBE_SEEDS
            for precision in profile.probe_precisions
        ]
        probe_block = (
            "PROBE STEP: this kernel's language profile carries a "
            f"non-empty probe_precisions tuple ({precisions_list}). "
            "After run_baseline_driver returns status='ok', drive the "
            "precision probe before calling spawn_analyst. The probe "
            "matrix for this run is:\n"
            f"  precisions: {precisions_list}\n"
            f"  seeds:      {seeds_list}\n"
            f"  cells:      {len(cells)} total = "
            + " | ".join(cells) + "\n"
            "Call probe_step exactly once per cell (the order does not "
            "matter; canonical order is seeds outer, precisions inner). "
            "Each probe_step call takes kernel_stem (same KERNEL STEM "
            "as above), precision (one of the precisions listed), and "
            "seed (one of the seeds listed). Cell failures are "
            "non-fatal: continue to the next cell. After every cell has "
            "been attempted, call probe_compare exactly once with the "
            "same KERNEL STEM. probe_compare hard-errors only if the "
            "canonical quad/seed=42 cell is missing. The aggregated "
            "evidence is attached to spawn_analyst automatically; you "
            "do not pass it through yourself. If run_baseline_driver "
            "did NOT return status='ok' (so the probe templates may "
            "not be on disk in usable shape), skip the probe step "
            "entirely and proceed to spawn_analyst.\n"
        )
    else:
        probe_block = ""
    return (
        f"BASELINE STEP: this is a {profile.display_name} kernel, so "
        "you SHOULD call spawn_baseline_harness exactly once to generate "
        f"a reference driver (baselines/{stem}/{profile.driver_filename}). "
        "This driver is the first link in the dynamic-verification chain "
        "that ends in compare_outputs and the code-side finish-gate: it "
        "is not consumed by the analyst, rewriter, or verifier, but it "
        "IS a precondition for finish on this kernel's language profile.\n"
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
        f"{probe_block}"
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
        f"compare_outputs IS a precondition for finish on this "
        f"{profile.display_name} kernel: a finish call before "
        "compare_outputs has returned status='ok' will be turned into "
        "a synthetic tool error. A "
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

      - For inputs whose language profile carries
        dynamic_verification=True (currently Kokkos C++ and CUDA C++),
        `finish` requires the most recent spawn_verifier call to have
        returned verdict='accept' AND the most recent compare_outputs
        call to have returned status='ok' for the CURRENT rewrite
        cycle.
      - For profiles with dynamic_verification=False (none today;
        reserved for languages we register before a baseline harness
        exists for them), the dynamic-verification chain is skipped
        entirely; `finish` is gated only on the most recent
        spawn_verifier verdict being 'accept'.

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

    def __init__(self, kernel_path: str, profile: LanguageProfile) -> None:
        self.kernel_path = kernel_path
        self.profile = profile
        # Whether the dynamic-verification chain (and therefore the
        # comparator) is required before finish is gated by the
        # resolved language profile, not the file extension. Phase A
        # threaded the profile through the orchestrator for exactly
        # this purpose; Phase B added the second profile (CUDA) and
        # flipped this from a `.cpp` literal check to the flag below.
        # Future profiles register with dynamic_verification=False
        # when no baseline harness exists for them yet.
        self.requires_comparator = profile.dynamic_verification
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
        # Other tools (spawn_analyst,
        # spawn_baseline_harness, compile_baseline_driver,
        # run_baseline_driver, probe_step, probe_compare) do not affect
        # the gate. The probe tools in particular are INFORMATIONAL --
        # they exist to give the analyst evidence, not to gate finish;
        # a failed or skipped probe does not block finish, and a
        # successful probe does not earn finish without the regular
        # verifier+comparator chain.
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
                "finish is not allowed: on a "
                f"{self.profile.display_name} input, "
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
    *,
    tolerance: dict,
    kernel_name: str | None = None,
    max_turns: int = MAX_TURNS,
    auto: bool = False,
    run_probe: bool = True,
) -> dict | None:
    """Run the orchestrator loop.

    `tolerance` is a dict {kind, value, source} where kind is one of
    'sig_figs' or 'decimal_digits', value is a small positive integer,
    and source is 'user_cli'. It is required — the CLI enforces that
    the operator passes --sig-figs or --decimal-digits and rejects a
    run with neither, and there is no in-workflow fallback.

    `kernel_name` is an optional explicit name of the kernel function
    inside `kernel_source` for the baseline_harness agent to target.
    When None (the common case in v0), the orchestrator tells the
    harness agent to infer the function from the source. CLI does not
    expose this yet.

    `auto` toggles the human-in-the-loop pause. Default False preserves
    the interactive y/n/q gate before every tool call. When True, every
    tool call is approved automatically and the loop writes a JSONL
    trace of {turn, tool_name, tool_input, exec_result} to
    baselines/<kernel_stem>/orchestrator_trace.jsonl so the operator can
    inspect post-hoc what each agent saw and returned. The trace file
    is truncated at the start of every auto run; MAX_TURNS remains the
    only backstop against runaway loops in this mode.

    `run_probe` (default True) toggles whether the BASELINE STEP block
    includes the precision-probe matrix. When False, the probe matrix
    is silently omitted from the system-message prompt and the
    orchestrator should not invoke probe_step / probe_compare at all.
    This is the --no-probe escape hatch for batch runs where the
    operator wants to reproduce the v0 (pre-probe) behavior exactly,
    or for kernels where the probe's wall-clock cost (up to 8 cells
    on a quad-emulated build) is not worth its evidence value.
    Profiles whose probe_precisions tuple is empty (every profile
    other than Kokkos in v1) ignore this flag — they never had probe
    instructions in their BASELINE STEP block to begin with.

    Returns the final finish() arguments dict, or None if the user quit,
    the orchestrator stopped without finishing, or max_turns was exhausted.
    """
    # Resolve the language profile once per run. The profile is a
    # constant for the entire orchestrator loop — it determines the
    # driver filename, the compiler invocation, and (in Phase B) which
    # baseline_harness_<lang> agent ships. Threading it explicitly to
    # _format_baseline_block, _FinishGateState, and _execute_tool keeps
    # the orchestrator itself language-agnostic and means the LLM
    # never has to pass language_id as a tool argument (Phase A.5
    # Option B: language is per-run, not per-call).
    profile = detect_language(kernel_path, kernel_source)
    tolerance_block = _format_tolerance_block(tolerance)
    baseline_block = _format_baseline_block(
        kernel_path, kernel_name, profile, run_probe=run_probe
    )
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
    gate = _FinishGateState(kernel_path, profile)

    trace_path: Path | None = None
    if auto:
        stem = Path(kernel_path).stem
        trace_dir = Path("baselines") / stem
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "orchestrator_trace.jsonl"
        trace_path.write_text("")

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

        # Defensive guard: some backends (notably some Argo proxy
        # configurations) return HTTP 200 with a body the SDK accepts
        # but cannot fully unmarshal into content blocks, leaving
        # `response.content` as None instead of an empty list. Iterating
        # that produces a cryptic TypeError far from the source. Fail
        # loudly here with the response id and stop_reason so the
        # operator can correlate against the proxy logs.
        if response.content is None:
            print(
                f"\nOrchestrator received a response with content=None. "
                f"stop_reason={response.stop_reason}, "
                f"response_id={getattr(response, 'id', '<unknown>')}. "
                f"This usually indicates a backend/proxy returned a "
                f"malformed message body; retry, or inspect the proxy "
                f"logs. Exiting."
            )
            return None

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
            choice = "y" if auto else _hitl_pause(tu.name, dict(tu.input))
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
                    _append_trace(
                        trace_path,
                        turns,
                        tu.name,
                        dict(tu.input),
                        {"status": "ok", "honored": True},
                    )
                    finish_args = dict(tu.input)
                    break
                # Gate violation: synthesize a tool_result the
                # orchestrator can self-correct from on its next turn,
                # instead of returning the finish args. This is the
                # source-of-truth enforcement for the rule the system
                # prompt describes; if the two ever disagree, this
                # code wins.
                gate_payload = {
                    "status": "error",
                    "stdout": "",
                    "stderr": gate_error,
                    "artifacts": [],
                }
                _append_trace(
                    trace_path, turns, tu.name, dict(tu.input), gate_payload
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(gate_payload),
                    "is_error": True,
                })
                continue
            exec_result = _execute_tool(
                tu.name,
                dict(tu.input),
                profile,
                kernel_stem=Path(kernel_path).stem,
            )
            gate.observe(tu.name, exec_result)
            _append_trace(
                trace_path, turns, tu.name, dict(tu.input), exec_result
            )
            # If the tool itself flagged an error condition (e.g. the
            # baseline_harness syntax-check gate rejected the driver
            # source), surface that at the Anthropic tool_result block
            # level so the model treats it as a failed tool call and
            # self-corrects on its next turn. The `is_error` key is a
            # tool-side signal; it does not belong in the JSON payload
            # the model reads, so we pop it before serializing content.
            is_error_flag = bool(exec_result.pop("is_error", False))
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(exec_result),
            }
            if is_error_flag:
                tool_result_block["is_error"] = True
            tool_results.append(tool_result_block)

        if user_quit:
            print("\nUser quit. Stopping.")
            return None
        if finish_args is not None:
            return finish_args

        messages.append({"role": "user", "content": tool_results})
