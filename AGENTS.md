# AGENTS.md

Research prototype: an LLM orchestrator that rewrites numerical kernels (Kokkos C++ / CUDA) to lower-precision where safe. Single Python package (`workflow/`), Anthropic SDK, no build system, no CI.

**The author is building this in deliberate baby steps as a learning exercise** in agent orchestration and tool-calling. Resist the urge to add abstractions, frameworks, or features that haven't been explicitly requested. Prefer the smallest concrete change that answers the immediate question. When in doubt, ask before expanding scope.

## Run

```bash
set -a; source .env; set +a            # .env is NOT auto-loaded
pip install -r requirements.txt        # runtime: anthropic; dev: pytest
python -m workflow.run <kernel_file> [--sig-figs N | --decimal-digits N]
# e.g. python -m workflow.run test-kernels/kokkos/mixed/nbody_force.cpp --sig-figs 6
```

`ANTHROPIC_API_KEY` is read from the environment by the `anthropic` client. There is no other config file for the workflow.

`AGENT_PRECISION_KOKKOS_ROOT` (optional) is read by `workflow/tools.py:compile_baseline_driver` and must point at a Kokkos install prefix (the directory containing `include/` and `lib/`). It is only consulted when the orchestrator calls the `compile_baseline_driver` tool, i.e. only after a successful `spawn_baseline_harness` on a Kokkos `.cpp` input. When unset, the tool returns `status='error'` and the rest of the pipeline continues (the compiled driver is a side artifact). The repo has a local Kokkos install at `./kokkos/` for this purpose; set `AGENT_PRECISION_KOKKOS_ROOT=$PWD/kokkos` before running the workflow directly (`python -m workflow.run ...`) if you want the baseline driver to actually compile. Both Argo wrapper scripts (`scripts/run-argo.sh`, `scripts/run-argoproxy.sh`) auto-default this var to `${PWD}/kokkos` when it is unset and that directory looks like a Kokkos prefix (has `include/` and `lib/`), so the compile tool works out of the box from the repo root via either wrapper; an explicit value in the caller's environment is honored unchanged. The name is deliberately namespaced so it doesn't collide with Kokkos's own CMake convention (`Kokkos_ROOT`) or with a system-wide install.

`AGENT_PRECISION_RUN_TIMEOUT_SEC` (optional) is read by `workflow/tools.py:run_baseline_driver` and must be a positive integer (seconds). It bounds wall-clock time for the compiled baseline driver subprocess. Defaults to `60` when unset; an invalid value (non-int, ≤0) makes the tool return `status='error'` without invoking the subprocess. The `_SEC` suffix is deliberate to avoid ms/s ambiguity; the name is namespaced for the same reason as `AGENT_PRECISION_KOKKOS_ROOT`. Both Argo wrapper scripts auto-default it to `60`; an explicit value in the caller's environment is honored unchanged. Like the compile tool, a timeout or non-zero exit is non-fatal — the rest of the pipeline continues.

`--sig-figs` and `--decimal-digits` are mutually exclusive and both optional. When neither is given, the orchestrator calls the `precision_advisor` agent to infer one; if the advisor returns `kind='unknown'`, the orchestrator falls back to `{kind:'sig_figs', value:6, source:'advisor_unknown_defaulted'}` (the constant `DEFAULT_TOLERANCE_ON_ADVISOR_UNKNOWN` in `orchestrator.py`). Both Argo wrapper scripts (`run-argo.sh`, `run-argoproxy.sh`) forward `"$@"` so flags pass through unchanged.

### Running via Argo (Argonne) instead of api.anthropic.com

```bash
./scripts/run-argo.sh test-kernels/kokkos/mixed/nbody_force.cpp
```

`run-argo.sh` brings up an SSH tunnel (`:8082` → `apps.inside.anl.gov:443` via `homes.cels.anl.gov`) and a local Anthropic-compatible shim (`claude-argo-proxy.py` on `:8083`), then runs `python -m workflow.run` with:

- `ANTHROPIC_BASE_URL=http://127.0.0.1:8083/argoapi/`
- `ANTHROPIC_AUTH_TOKEN=$USER` (Argonne username, not an API key)

The `anthropic` SDK honors both env vars, so no workflow code changes are required to switch backends. If the tunnel and/or proxy are already running (e.g. started by another tool in the same session), the script reuses them and does not kill them on exit.

The shim script is expected at `~/argo-shim-lite/claude-argo-proxy.py` (override with `ARGO_PROXY_SCRIPT=...`); it depends on `aiohttp`. The reference implementation this was modeled on lives at `~/Agentic-Mixed-Precision-Demo/run-argo.sh`.

### Or: use the local argo-proxy on :52675 (no SSH tunnel, no Duo)

```bash
./scripts/run-argoproxy.sh test-kernels/kokkos/mixed/nbody_force.cpp
```

`argo-proxy serve` (the same daemon OpenCode's `provider.argo` depends on) natively exposes an Anthropic-compatible `/v1/messages` route at `http://127.0.0.1:52675/v1/`. The wrapper just `curl`s `/health`, then runs `python -m workflow.run` with `ANTHROPIC_BASE_URL=http://127.0.0.1:52675/v1/` and `ANTHROPIC_AUTH_TOKEN=$USER` (any non-empty string; argo-proxy ignores it).

Prereq: `argo-proxy serve` is already running. argo-proxy has persistent auth set up at pipx-install time, so there's no per-session Duo prompt.

Use `scripts/run-argo.sh` instead only when `argo-proxy` isn't installed on this host — that script's SSH tunnel + `claude-argo-proxy.py` shim is a self-contained fallback. Both paths terminate at the same upstream Argo service, but they are **not** interchangeable for model ids: `argo-proxy` normalizes model names before forwarding, while `claude-argo-proxy.py` is a transparent forwarder. The id Argo's `/v1/messages` upstream actually accepts is `claude-opus-4-7` (hyphen, no prefix); see "Model names look wrong but aren't" below.

## Architecture (don't redesign this)

- `workflow/registry.py` — `AGENTS` dict is the single source of truth for agent types. Each entry = `{system_prompt, output_schema, model}`. **Adding a new agent type = one new entry here, nothing else changes.**
- `workflow/run_agent.py` — generic `run_agent(type, task) -> dict`. Forces the agent to call a `submit_result` tool whose schema is the registry's `output_schema`. Never edit this when adding agent types.
- `workflow/orchestrator.py` — itself a Claude conversation. Tools are one-per-agent-type (`spawn_precision_advisor`, `spawn_analyst`, `spawn_rewriter`, `spawn_verifier`, `spawn_baseline_harness`) plus `finish`. Contains the **human-in-the-loop pause** (`_hitl_pause`): every tool call is shown and the user must approve `y/n/q` before it runs. This pause is the whole point of v0 — do not remove it or auto-approve. A `MAX_TURNS=20` backstop guards against runaway loops if the HITL is left unattended; don't raise it casually.
- `workflow/run.py` — thin CLI; takes a kernel path plus optional `--sig-figs N` / `--decimal-digits N`, normalizes either flag into `{kind, value, source:'user_cli'}` (or `None`), passes that as the `tolerance` kwarg to `run_orchestrator`, then prints final code + notes.

Rejecting (`n`) returns `{"status": "rejected_by_user"}` to the orchestrator so it can self-correct without burning an agent API call — that behavior is intentional.

The orchestrator system prompt enforces the pipeline `(precision_advisor ->)? analyst -> rewriter -> verifier -> finish`, and explicitly forbids calling `finish` unless the most recent `spawn_verifier` returned `verdict='accept'`. It also forbids calling `spawn_precision_advisor` when the user supplied a tolerance on the command line, and forbids calling it more than once. `spawn_baseline_harness` is orthogonal to this pipeline — it is a side artifact (see next section), never a precondition for `finish`, and called at most once per run, only for Kokkos `.cpp` inputs. `compile_baseline_driver` is similarly orthogonal: it is the deterministic follow-up to a successful `spawn_baseline_harness`, called at most once and only with the same `kernel_stem`, and its result (success or compile error) is never a precondition for `finish`. `run_baseline_driver` extends the chain one step further: the deterministic follow-up to a successful `compile_baseline_driver`, called at most once and only with the same `kernel_stem`, and its result (success, non-zero exit, timeout, or missing/invalid `reference.json`) is likewise never a precondition for `finish`. Those rules live in the prompt, not in code — the orchestrator is trusted to obey them. If you add a hard guard, do it in `orchestrator.py` and keep the prompt in sync.

### Baseline harness (side artifact)

`baseline_harness` is the first component of the planned dynamic verifier (see "Planned next steps"). It is a one-shot LLM agent that, given a Kokkos C++ kernel, writes a self-contained C++ driver source. When the operator later compiles and runs that driver, it exercises the kernel on fixed inputs (deterministic: `Kokkos::Serial`, fixed RNG seed 42 by default) and writes a reproducible reference output to `./reference.json` using `%.17g` formatting. A future mechanical comparator will diff this against the rewritten kernel's output.

v0 scope is intentionally narrow:

- Kokkos-only. The orchestrator's initial user message contains a `BASELINE STEP` block that invites the harness call for `.cpp` inputs and explicitly forbids it for `.cu` inputs (see `_format_baseline_block` in `orchestrator.py`). CUDA support is deferred.
- Driver source only. The agent never compiles, runs, or invents numerical output values. Its `submit_result` payload is `{driver_source, kernel_function_name, inputs_summary, output_arrays}` (see `BASELINE_HARNESS_OUTPUT_SCHEMA` in `registry.py`).
- One driver per kernel stem. On HITL approval, `_execute_tool` writes the driver to `baselines/<kernel_stem>/driver.cpp` (relative to the orchestrator's CWD; this is currently the only `_execute_tool` branch that touches the filesystem). `kernel_stem` is derived from `Path(kernel_path).stem` by `run_orchestrator` and passed to the tool via the user message; the orchestrator forwards it back as the `kernel_stem` tool argument. Re-running overwrites the previous driver.
- Splice sentinels. The harness prompt mandates that the inlined kernel be bracketed by the exact lines `// ---- KERNEL BEGIN ----` and `// ---- KERNEL END ----` (each on its own line, no indentation, no paraphrasing). These are a hard contract for a later mechanical-verification step that string-replaces the text between them to swap in a rewritten kernel against bit-identical inputs / RNG / JSON output. If you change the sentinel strings, update `BASELINE_HARNESS_SYSTEM_PROMPT` in `registry.py`, the future splice helper in `workflow/tools.py`, and the asserting test in `tests/test_registry.py` together.
- No downstream consumer. The harness output is not fed into the analyst, rewriter, or verifier in this run. It exists so that a later mechanical-verification tool has a reference to compare against. `finish` does not require the baseline.

`run_orchestrator(..., kernel_name: str | None = None)` accepts an optional explicit kernel function name to disambiguate which function the harness should target; when given, a `TARGET KERNEL: <name>` line is added to the `BASELINE STEP` block. The CLI does not expose this flag in v0 (the agent infers the function from a single-kernel source). When multi-kernel inputs eventually land, the per-stem directory will likely evolve to `baselines/<file_stem>__<kernel_name>/`.

Compilation of the driver is handled by a separate deterministic (non-LLM) orchestrator tool, `compile_baseline_driver`, defined in `workflow/tools.py`. The orchestrator prompt instructs the LLM to call it exactly once, immediately after a successful `spawn_baseline_harness`, passing the same `kernel_stem`. The tool reads `AGENT_PRECISION_KOKKOS_ROOT` (see the env-var note at the top of this file), shells out to `g++ -std=c++20 -O2 -fopenmp -I$ROOT/include -L$ROOT/lib baselines/<stem>/driver.cpp -lkokkoscore -lkokkoscontainers -lpthread -ldl -o baselines/<stem>/driver`, and returns `{status, stdout, stderr, artifacts}`. That shape is the same one the planned dynamic-verification tools will return (see "Planned next steps") so the orchestrator's tool-result handling doesn't change when "compile locally" becomes "submit compile job and poll". The repo's bundled Kokkos at `./kokkos/` was built as static archives with the OpenMP host backend, which is why the flag set is fixed — there is no rpath because the binary has no `libkokkos*` in its dynamic NEEDED list. Like the baseline itself, a failed compile is non-fatal: the analyst -> rewriter -> verifier pipeline still runs and `finish` is still reachable.

Execution of the compiled driver is a third deterministic (non-LLM) orchestrator tool, `run_baseline_driver`, also defined in `workflow/tools.py`. The orchestrator prompt instructs the LLM to call it exactly once, immediately after a successful `compile_baseline_driver`, passing the same `kernel_stem`. The tool checks that `baselines/<stem>/driver` exists and is executable, deletes any stale `baselines/<stem>/reference.json` from a prior run (so a failed run cannot leave a misleadingly-stale file in place), then `subprocess.run(["./driver"], cwd="baselines/<stem>", capture_output=True, text=True, check=False, timeout=AGENT_PRECISION_RUN_TIMEOUT_SEC)`. On a clean exit it requires the driver to have written `reference.json` and validates that it parses as JSON; the shape of that JSON (e.g. presence of an `outputs` key) is left to the future mechanical comparator. The tool returns the same `{status, stdout, stderr, artifacts}` dict as `compile_baseline_driver`, with `artifacts=["baselines/<stem>/reference.json"]` on success and `[]` on any error path (missing binary, non-executable binary, bad env value, non-zero exit, `TimeoutExpired`, `FileNotFoundError` at exec time, missing `reference.json`, invalid JSON). Like the compile step, a failed run is non-fatal: the analyst -> rewriter -> verifier pipeline still runs and `finish` is still reachable.

### Tolerance vocabulary and plumbing

There are three distinct precision concepts kept deliberately separate in this code, prompts, and tests:

- **`sig_figs` (significant figures)** — relative tolerance on the kernel's *output*.
- **`decimal_digits` (decimal places after the point)** — absolute tolerance on the output.
- **Floating-point storage precision** (`float`, `double`, `float-float`, …) — the *mechanism* the analyst chooses per variable. A kernel can hit a 6-sig-fig output target using a mix of internal storage precisions.

The tolerance is a single dict `{kind, value, source}` flowing through the run:

- `kind ∈ {sig_figs, decimal_digits}` once fixed; the advisor's `unknown` is never propagated downstream — it triggers the documented fallback.
- `value` is a small positive integer (`int > 0`; non-positive values are rejected by `workflow/run.py`).
- `source ∈ {user_cli, precision_advisor, advisor_unknown_defaulted}` — records which path set the tolerance, so logs and the analyst's `precision_budget` block can carry the provenance.

Plumbing: `run_orchestrator(kernel_path, kernel_source, tolerance=None)` embeds the tolerance verbatim into the first user message via `_format_tolerance_block()`. The orchestrator LLM is then responsible for (a) calling `spawn_precision_advisor` first if and only if `tolerance` was `None`, and (b) inlining the agreed tolerance into the `kernel_source` argument of `spawn_analyst` (as a labeled block the analyst echoes into its `precision_budget` output), into the `task_prompt` of `spawn_rewriter`, and as the separate `tolerance_json` argument of `spawn_verifier` (which `_execute_tool` joins into a labeled TOLERANCE (JSON) section of the verifier's task string).

The analyst is told, in its system prompt, that the tolerance is a hard constraint and that `emulate` is throughput-negative — preferred only when `downcast` would violate tolerance. The rewriter prompt forbids silently substituting one method for another. Both rules are LLM-side, not Python-enforced.

### Per-variable action enums

The analyst's per-variable `action` enum is `{downcast, emulate, keep}`. `downcast` replaces a type with a narrower hardware type (e.g. `double` -> `float`) and uses `target_precision`; `emulate` replaces a type with a software-emulated pair (currently float-float / Dekker, written inline as `struct ff_t {float hi, lo;}` per the rewriter prompt) and uses `emulation_type`; `keep` leaves the variable alone. The analyst's top-level result also includes a required `rework` object (`{suggested, transformation, rationale, affected_variables}`) for kernel-shape changes such as Kahan summation (when none is warranted, the analyst still returns the object with `suggested=false` and empty fields rather than omitting it), and a required `precision_budget` object (`{target_kind, target_value, source, claimed_output_precision, headroom_argument}`) that links the verdict back to the tolerance. The verifier's `expected_action` enum mirrors the analyst's; its `observed_action` enum adds `unclear`. If you rename any of these tokens, update `registry.py`, `orchestrator.py`'s system prompt, and the tests in `tests/test_registry.py` / `tests/test_orchestrator.py` together — they are deliberately coupled.

## Model names look wrong but aren't (verify before changing)

`registry.py` and `orchestrator.py` both hardcode `claude-opus-4-7` (hyphen, no prefix). This is what the upstream Argo `/v1/messages` endpoint actually accepts — verified empirically. Do not "fix" it to a more familiar id (e.g. `claude-opus-4.7`, `argo:claude-opus-4.7`, `claude-opus-4-20250514`) without testing against the real backend first. The `:52675` `argo-proxy` is forgiving and normalizes several variants; the `:8083` `claude-argo-proxy.py` shim is a transparent forwarder and will pass any string straight to Argo, which then rejects unknown ids with HTTP 400. The current id works on both paths.

## opencode.json has a trailing comment block

`opencode.json` ends with `}/*  ...notes... */` (lines ~596–618). It is technically not valid JSON; opencode tolerates it. If you edit this file, preserve or remove that block cleanly — do not blindly close it, and do not let an auto-formatter touch the file.

The `provider.argo` config in `opencode.json` points at `argo-proxy` on `:52675` and uses its OpenAI route. The workflow can share that same daemon via its Anthropic route (see "Or: use the local argo-proxy on :52675" above). `scripts/run-argo.sh` is a separate path entirely — it talks to `claude-argo-proxy.py` on `:8083` and only exists as a fallback for hosts where `argo-proxy` isn't installed. Don't conflate the two proxies; they speak different protocols on different ports.

## test-kernels/

17 kernels under `test-kernels/{cuda,kokkos}/{lowerable,needs_precision,mixed}/`. Directory name = ground-truth label. The workflow must decide from **file content only**; never feed the path verdict into prompts. `test-kernels/SUMMARY.md` has the per-variable expected verdicts and test tolerances — use it to evaluate orchestrator output, not as input to it.

## scripts/setup_argo_proxy.sh

Argonne-specific: opens an SSH tunnel through `logins.cels.anl.gov` → `homes.cels.anl.gov` and runs `~/lmtools-main/bin/apiproxy` remotely. Requires Duo. Only relevant when using OpenCode itself against Argo from JLSE; unrelated to running the Python workflow.

## Conventions

- No linter, formatter, or typecheck configured. Don't invent commands; ask before adding tooling.
- Layer 1 tests (workflow plumbing) live in `tests/`; run with `python -m pytest -q` for a terse pass/fail, or `python -m pytest -v` to see each test's docstring appended to its node id as a self-describing checklist (a `pytest_collection_modifyitems` hook in `tests/conftest.py` does the appending). They monkeypatch `anthropic.Anthropic` and make zero network calls. There is no Layer 2 (agent-judgment) evaluation harness yet.
- Python 3.10+ (uses `dict | None`).
- Keep agent system prompts in `registry.py`, not scattered. Keep orchestrator prompt in `orchestrator.py`.
- `flowchart.md` is a Mermaid view of the orchestrator loop (tools, HITL pause, rejection feedback). It is documentation only — nothing in `workflow/` reads it — but if you change the pipeline (add/remove a spawn tool, change the HITL contract, change what rejection returns), update `flowchart.md` in the same change so it doesn't drift from `orchestrator.py`. README's "Workflow" section has the companion sequence diagram.

## Planned next steps (not yet implemented)

Documented here so future sessions don't relitigate decisions or accidentally double-implement them. None of these are in the code today.

- **Dynamic verification stack.** The current `verifier` agent only checks faithfulness of the rewrite to the analyst's verdict (a static / textual check). The next step is *mechanical* verifiers — compile the driver from `baseline_harness` (compile and run of the baseline are already implemented as `compile_baseline_driver` and `run_baseline_driver`; see "Baseline harness" above), then splice the rewritten kernel between the sentinels, recompile, re-run, and compare outputs within tolerance — exposed to the orchestrator as new tools (not new agents), because they are deterministic and shouldn't burn an LLM call. Tool return shape should be `{status, stdout, stderr, artifacts}` from day one so the same interface survives migration from local Kokkos+CUDA to a remote batch system (see "JLSE/async toolchain" below). Landing the splice + compare steps will likely require raising `MAX_TURNS` past 20.
- **Kernel-extractor agent.** When inputs eventually contain more than one kernel function in a single file (or the kernel is buried in a larger translation unit), a one-shot LLM agent will identify the target kernel function and slice it out into a standalone source before the analyst and baseline-harness run. v0 sidesteps this by assuming one obvious kernel per file and letting the baseline-harness agent infer the function name; `run_orchestrator` already accepts an optional `kernel_name` argument that the extractor could populate. When this lands, the per-stem baseline directory likely becomes `baselines/<file_stem>__<kernel_name>/` so multiple kernels per file don't collide.
- **JLSE / async toolchain migration.** Current host has Kokkos+CUDA locally; future runs will submit compile/run jobs to a remote scheduler. Designing the verifier tool interfaces around `{status, stdout, stderr, artifacts}` now means the orchestrator loop doesn't change when "run" becomes "submit job, poll, fetch artifacts".
- **Emulation library upgrade.** v0 writes `struct ff_t {float hi, lo;}` plus Dekker two-sum/two-prod inline (per the rewriter prompt) rather than vendoring a real float-float library. Replacing this with a proper header is straightforward once we have mechanical verification to catch regressions.
- **Rejected: multiple per-method analysts.** Considered splitting the analyst into priority-ordered specialists (downcast-first, then emulate, then rework). Rejected: more moving parts, breaks the "one entry in `AGENTS` per agent type" invariant, and gives no clear win over a single analyst with a richer output schema. Revisit only if a single analyst proves to systematically under-explore one method.
- **Rejected: LangGraph (or similar framework).** Considered for state management as the pipeline grows. Rejected for v0: adds a dependency and an abstraction layer for a workflow that is still linear and synchronous. Revisit only if we get *parallel* branches (e.g. multiple rewrite candidates evaluated concurrently), async/durable state across long jobs, or a non-CLI frontend. Intermediate alternatives to try first: an explicit `OrchestratorState` dataclass, or pickled-message resume. The harness -> compile -> run trio specifically is a tempting subgraph candidate (shared `kernel_stem`, shared `baselines/<stem>/` paths, eventually a "which compiler" gate before compile), and was also rejected for v0 for the same reasons plus three more: (a) it would collapse three HITL approval points into one, undermining v0's whole point that every tool call is shown to the user before it runs; (b) the "which compiler" gate is a one-line `shutil.which("g++")` / `shutil.which("nvcc")` check (and a kernel-extension dispatch) *inside* `compile_baseline_driver`, not a graph node — same spirit as the existing env-var validation block; (c) compile and run are deterministic `subprocess.run` calls, not LLM nodes, so a graph framework here is mostly ceremony around two shell-outs. Revisit specifically for the trio if a *second* consumer of harness -> compile -> run appears outside the main orchestrator (e.g. a batch evaluator over `test-kernels/`), if rewrite candidates fan out in parallel and each needs its own compile+run, or if the JLSE migration forces async/durable state across long-running jobs.
