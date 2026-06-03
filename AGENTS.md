# AGENTS.md

Research prototype: an LLM orchestrator that rewrites numerical kernels (Kokkos C++ / CUDA) to lower-precision where safe. Single Python package (`workflow/`), Anthropic SDK, no build system, no tests, no CI.

## Run

```bash
set -a; source .env; set +a            # .env is NOT auto-loaded
pip install -r requirements.txt        # only dep: anthropic>=0.40.0
python -m workflow.run <kernel_file>   # e.g. test-kernels/kokkos/mixed/nbody_force.cpp
```

`ANTHROPIC_API_KEY` is read from the environment by the `anthropic` client. There is no other config file for the workflow.

### Running via Argo (Argonne) instead of api.anthropic.com

```bash
./scripts/run-argo.sh test-kernels/kokkos/mixed/nbody_force.cpp
```

`run-argo.sh` brings up an SSH tunnel (`:8082` → `apps.inside.anl.gov:443` via `homes.cels.anl.gov`) and a local Anthropic-compatible shim (`claude-argo-proxy.py` on `:8083`), then runs `python -m workflow.run` with:

- `ANTHROPIC_BASE_URL=http://127.0.0.1:8083/argoapi/`
- `ANTHROPIC_AUTH_TOKEN=$USER` (Argonne username, not an API key)

The `anthropic` SDK honors both env vars, so no workflow code changes are required to switch backends. If the tunnel and/or proxy are already running (e.g. started by another tool in the same session), the script reuses them and does not kill them on exit.

The shim script is expected at `~/argo-shim-lite/claude-argo-proxy.py` (override with `ARGO_PROXY_SCRIPT=...`); it depends on `aiohttp`. The reference implementation this was modeled on lives at `~/Agentic-Mixed-Precision-Demo/run-argo.sh`.

### Or: skip the SSH tunnel by using the local argo-proxy on :52675

`argo-proxy serve` (the same daemon OpenCode's `provider.argo` already depends on) natively serves `/v1/messages` at `http://127.0.0.1:52675/v1/`. The Anthropic SDK can talk to it directly — no SSH tunnel, no per-session Duo prompt:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:52675/v1/
export ANTHROPIC_AUTH_TOKEN=$USER   # any non-empty string; argo-proxy ignores it
python -m workflow.run test-kernels/kokkos/mixed/nbody_force.cpp
```

Prereq: `argo-proxy serve` is running (check: `curl -sf http://127.0.0.1:52675/health`).

Use `scripts/run-argo.sh` instead only when `argo-proxy` isn't installed on this host — that script's SSH tunnel + `claude-argo-proxy.py` shim is a self-contained fallback. Both paths terminate at the same upstream Argo service.

## Architecture (don't redesign this)

- `workflow/registry.py` — `AGENTS` dict is the single source of truth for agent types. Each entry = `{system_prompt, output_schema, model}`. **Adding a new agent type = one new entry here, nothing else changes.**
- `workflow/run_agent.py` — generic `run_agent(type, task) -> dict`. Forces the agent to call a `submit_result` tool whose schema is the registry's `output_schema`. Never edit this when adding agent types.
- `workflow/orchestrator.py` — itself a Claude conversation. Tools are one-per-agent-type (`spawn_rewriter`) plus `finish`. Contains the **human-in-the-loop pause** (`_hitl_pause`): every tool call is shown and the user must approve `y/n/q` before it runs. This pause is the whole point of v0 — do not remove it or auto-approve.
- `workflow/run.py` — thin CLI; takes one path argument, prints final code + notes.

Rejecting (`n`) returns `{"status": "rejected_by_user"}` to the orchestrator so it can self-correct without burning an agent API call — that behavior is intentional.

## Model names look wrong but aren't (verify before changing)

`registry.py` and `orchestrator.py` both hardcode `claude-opus-4-7`. See `orchestrator-model.txt` for context — the author deliberately picked this string. Do not "fix" it to a more familiar id without confirming the deployment actually exposes a different name.

## opencode.json has a trailing comment block

`opencode.json` ends with `}/*  ...notes... */` (lines ~596–618). It is technically not valid JSON; opencode tolerates it. If you edit this file, preserve or remove that block cleanly — do not blindly close it, and do not let an auto-formatter touch the file.

The `provider.argo` config in `opencode.json` points at `argo-proxy` on `:52675` and uses its OpenAI route. The workflow can share that same daemon via its Anthropic route (see "Or: skip the SSH tunnel" above). `scripts/run-argo.sh` is a separate path entirely — it talks to `claude-argo-proxy.py` on `:8083` and only exists as a fallback for hosts where `argo-proxy` isn't installed. Don't conflate the two proxies; they speak different protocols on different ports.

## test-kernels/

17 kernels under `test-kernels/{cuda,kokkos}/{lowerable,needs_precision,mixed}/`. Directory name = ground-truth label. The workflow must decide from **file content only**; never feed the path verdict into prompts. `test-kernels/SUMMARY.md` has the per-variable expected verdicts and test tolerances — use it to evaluate orchestrator output, not as input to it.

## scripts/setup_argo_proxy.sh

Argonne-specific: opens an SSH tunnel through `logins.cels.anl.gov` → `homes.cels.anl.gov` and runs `~/lmtools-main/bin/apiproxy` remotely. Requires Duo. Only relevant when using OpenCode itself against Argo from JLSE; unrelated to running the Python workflow.

## No verifier yet

The orchestrator can `spawn_rewriter` → `finish` without anyone checking correctness. Adding a verifier agent is the next planned step. If you add one, follow the registry pattern: one new entry in `AGENTS`, one new tool on the orchestrator, no edits to `run_agent`.

## Conventions

- No tests, linter, formatter, or typecheck configured. Don't invent commands; ask before adding tooling.
- Python 3.10+ (uses `dict | None`).
- Keep agent system prompts in `registry.py`, not scattered. Keep orchestrator prompt in `orchestrator.py`.
