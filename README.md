# Agent-Precision

An LLM orchestrator that rewrites numerical kernels (Kokkos C++ / CUDA) to
lower precision where it is numerically safe to do so.

This is a research prototype and a deliberate learning exercise in agent
orchestration and tool-calling. It is a single Python package (`workflow/`)
that talks to the Anthropic SDK. There is no build system and no CI. A
small Layer 1 test suite (`tests/`, run with `python -m pytest -q`) covers
the workflow plumbing with monkeypatched API calls; there is no Layer 2
evaluation of agent judgment yet. Features are added in small, observable
steps rather than as end-to-end machinery.

If you want to hack on the workflow itself, read `AGENTS.md` first — it
contains the gotchas (model id, env-loading, backend quirks) that will
otherwise burn you.

## Status

What works today:

- Two-agent pipeline: **analyst → rewriter**, driven by an LLM orchestrator.
- Human-in-the-loop pause before every agent call (`y` / `n` / `q`).
  Rejections are fed back to the orchestrator as
  `{"status": "rejected_by_user"}`, so it can self-correct without burning
  another agent call.
- Registry-driven agent definitions: adding a new agent type is one entry
  in `workflow/registry.py`; the generic runner does not change.
- Three execution backends: direct `api.anthropic.com`, Argo via local
  `argo-proxy`, and Argo via SSH tunnel + shim (fallback).

What is intentionally **not** here yet:

- No verifier agent. The orchestrator can `spawn_rewriter → finish` without
  anyone checking correctness.
- No compilation check, no accuracy check, no benchmark runner.
- No automated evaluation across the `test-kernels/` corpus.
- No framework (LangGraph etc.). See "Design notes" and "Roadmap" below.

## Run

Only dependency: `anthropic>=0.40.0`.

```bash
pip install -r requirements.txt
```

Stdout in every mode is the orchestrator's reasoning, then a HITL prompt
for each proposed tool call, then the final rewritten kernel and notes.

**Direct (Anthropic API).** `.env` is not auto-loaded.

```bash
set -a; source .env; set +a            # exports ANTHROPIC_API_KEY etc.
python -m workflow.run test-kernels/kokkos/mixed/nbody_force.cpp
```

**Argo via local `argo-proxy` (recommended on Argonne hosts).** Requires
`argo-proxy serve` already running; no per-session Duo prompt.

```bash
./scripts/run-argoproxy.sh test-kernels/kokkos/mixed/nbody_force.cpp
```

**Argo via SSH tunnel + shim (fallback).** Use when `argo-proxy` is not
installed on this host. Brings up its own tunnel and local Anthropic-
compatible shim.

```bash
./scripts/run-argo.sh test-kernels/kokkos/mixed/nbody_force.cpp
```

See `AGENTS.md` for the backend details (ports, model-id quirks, why both
Argo paths exist).

## Workflow

High-level flow. The orchestrator is itself a Claude conversation; its
tools are one per agent type plus a `finish` tool.

```mermaid
flowchart TD
  U["python -m workflow.run &lt;kernel&gt;"] --> O["Orchestrator (Claude)"]
  O -- "tool_use" --> H{"HITL pause<br/>y / n / q"}
  H -- "n" --> O
  H -- "q" --> X(["exit"])
  H -- "y" --> D{"which tool?"}
  D -- "spawn_analyst" --> A["Analyst agent"]
  D -- "spawn_rewriter" --> R["Rewriter agent"]
  D -- "finish" --> F(["print final kernel + notes"])
  A -- "{variables, overall_notes}" --> O
  R -- "{rewritten_code, summary_of_changes}" --> O
```

Intended happy path through the conversation:

```mermaid
sequenceDiagram
  participant User
  participant Orch as Orchestrator
  participant An as Analyst
  participant Rw as Rewriter
  User->>Orch: kernel source
  Orch->>User: HITL: spawn_analyst(kernel_source)?
  User->>Orch: y
  Orch->>An: kernel_source
  An-->>Orch: per-variable verdict
  Orch->>User: HITL: spawn_rewriter(task_prompt)?
  User->>Orch: y
  Orch->>Rw: task_prompt (source + verdict)
  Rw-->>Orch: rewritten_code
  Orch->>User: HITL: finish(rewritten_code, notes)?
  User->>Orch: y
  Orch-->>User: final kernel
```

## Agents

| Agent          | Model             | Input                                                | Output (schema keys)                                                       | Defined in                |
| -------------- | ----------------- | ---------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------- |
| `analyst`      | `claude-opus-4-7` | kernel source only                                   | `variables[{name, action, target_precision, reason}]`, `overall_notes`     | `workflow/registry.py`    |
| `rewriter`     | `claude-opus-4-7` | `task_prompt` (source + verdict, composed by orch.)  | `rewritten_code`, `summary_of_changes`                                     | `workflow/registry.py`    |
| `orchestrator` | `claude-opus-4-7` | kernel path + source                                 | one `finish(rewritten_code, notes)` call                                   | `workflow/orchestrator.py` |

The analyst receives the kernel source only — no file path, no orchestrator
hints — so that ground-truth labels encoded in directory names cannot leak
into the verdict.

## Orchestrator tools

| Tool             | Purpose             | Input                       | Returns to orchestrator                                       |
| ---------------- | ------------------- | --------------------------- | ------------------------------------------------------------- |
| `spawn_analyst`  | dispatch to analyst | `kernel_source: string`     | analyst's structured output, or `{"status":"rejected_by_user"}` |
| `spawn_rewriter` | dispatch to rewriter | `task_prompt: string`       | rewriter's structured output, or `{"status":"rejected_by_user"}` |
| `finish`         | end the workflow    | `rewritten_code`, `notes`   | (terminates; nothing fed back)                                |

## Repo layout

- `workflow/`
  - `registry.py` — agent definitions (`AGENTS` dict: system prompt, output
    schema, model). Single source of truth.
  - `run_agent.py` — generic agent runner. Forces structured output via
    `tool_choice={"type":"tool","name":"submit_result"}` whose input schema
    is the registry entry's `output_schema`. Never edited per-agent.
  - `orchestrator.py` — router + HITL loop. One tool per agent type plus
    `finish`.
  - `run.py` — CLI entrypoint (`python -m workflow.run <kernel_file>`).
- `test-kernels/` — 17 kernels under
  `{cuda,kokkos}/{lowerable,needs_precision,mixed}/`. Directory name is
  the ground-truth label and is **never** fed into agent prompts. Per-
  variable expected verdicts and test tolerances live in
  `test-kernels/SUMMARY.md` and are used for evaluating orchestrator
  output, not as input to it.
- `scripts/` — Argo backend wrappers (`run-argoproxy.sh`, `run-argo.sh`).
- `AGENTS.md` — instructions for coding agents working on this repo.
  Contains the gotchas you need before changing anything.
- `opencode.json` — opencode configuration. Unrelated to the workflow
  itself; only relevant if you run opencode against this repo.

## Design notes

**Why a separate analyst agent (vs a smarter rewriter).** Keeps the
rewriter mechanical — it transforms code to match an explicit verdict and
does not exercise numerical judgment. Makes the HITL pause informative,
because the verdict is human-readable structured data before any code is
touched. And it gives a future verifier a structured artifact to check
against, separate from the code diff.

**Why HITL on every tool call.** This is a learning-by-watching tool, not
a production system. Seeing the exact prompt that is about to be sent to
each agent is most of the value. Rejection-as-self-correction
(`{"status":"rejected_by_user"}`) is a deliberately cheap feedback loop:
you can steer the orchestrator without spending an agent API call.

**Why the registry pattern.** Adding an agent type is one entry in
`AGENTS`. The orchestrator gains one new `spawn_*` tool wrapping it;
`run_agent.py` does not change. This is the constraint that keeps the
prototype small as new agent types are added.

**Why no framework (LangGraph etc.) yet.** Not enough nodes or edges to
justify it. The orchestrator is a single Claude conversation with a HITL
gate; that is easier to read in ~230 lines of Python than as a graph
config. Revisit once a verifier exists and there are real cycles.

## Roadmap

1. **Verifier agent** — compile the rewritten kernel and check accuracy
   against a reference run, using the same registry pattern (one new entry
   in `AGENTS`, one new tool on the orchestrator).
2. **Corpus evaluation** — run the workflow across all 17 kernels in
   `test-kernels/` and compare its per-variable verdicts to
   `test-kernels/SUMMARY.md`.
3. **Reconsider framework choice** — once node count and edge complexity
   (retries, verifier feedback loops, branching) warrant it, evaluate
   LangGraph or similar. Likely direction if it happens: LangGraph for the
   runtime, raw Anthropic SDK inside nodes, no LangChain.
