# JLSE A100 bring-up runbook (Option A, interactive)

Operator recipe for bringing the workflow up on a JLSE (Joint Laboratory
for System Evaluation) A100 node via `qsub -I`. Complements README's
"Running on JLSE" section — that section gives the shape of the plan
and the env-var contract; this file gives the exact commands, the
paste-output checkpoints, and the branch logic when reality diverges
from the happy path.

This is Option A ("interactive") from the AGENTS.md roadmap. The whole
workflow — LLM orchestrator, compile, run, HITL — happens inside one
`qsub -I` allocation on the compute node. Option B ("async per-tool
scheduler submission") is a separate future story; not covered here.

Success criteria (referenced throughout as **S1..S5**):

- **S1** `pytest -q` passes inside the JLSE venv.
- **S2** `run_agent` succeeds from a JLSE Python REPL (LLM plumbing works).
- **S3** `saxpy.cu --sig-figs 6 --auto` runs end-to-end; `baselines/saxpy/rewritten/timing.json` exists.
- **S4** `nbody_force.cpp --sig-figs 6 --auto` runs end-to-end; `baselines/nbody_force/rewritten/timing.json` exists.
- **S5** README + AGENTS.md reflect what actually landed (done on laptop before rsync).

Prereqs: Argo access from JLSE (Duo may be required for the
`homes.cels.anl.gov` hop), Anthropic-compatible LLM plumbing decided
(this runbook uses the `claude-argo-proxy.py` shim on `:8083`; the
`argo-proxy` daemon on `:52675` is an alternative if it's pipx-installed
on JLSE — see README's "Or: use the local argo-proxy" section).

---

## Step 1: rsync from laptop (run on laptop, before session)

Confirm laptop-side state is clean:

```bash
cd ~/Agent-Precision
git status              # expect: nothing to commit, working tree clean
git log --oneline -3    # confirm the JLSE bring-up commit is present
python -m pytest -q     # sanity: 788 passed on laptop
```

Rsync the repo. Explicit excludes are needed because rsync does NOT
honor `.gitignore` by default, and several gitignored directories are
either huge regeneratable output (`poster-results/`, `evals/results/`)
or Python-version-specific (`.venv*/`) or laptop-only tooling state
(`.opencode/`, `error_dumps/`). `baselines/` and `kokkos/` ARE
gitignored but we DO want them on JLSE, so they are deliberately NOT
excluded:

```bash
rsync -avz --progress \
  --exclude='.git/' \
  --exclude='.venv*/' \
  --exclude='__pycache__/' \
  --exclude='evals/results/' \
  --exclude='poster-results/' \
  --exclude='error_dumps/' \
  --exclude='.opencode/' \
  ~/Agent-Precision/ <user>@<jlse-login>:~/Agent-Precision/
```

Total transfer ~4.5 GB with the exclude list above (baselines 4.3 GB
+ kokkos 28 MB + code + docs + tests ~5 MB). Without the excludes,
the laptop tree is ~8 GB (`poster-results/` adds 2.9 GB, `.git/` adds
900 MB, `error_dumps/` adds 8.5 MB). Either fits comfortably under
JLSE's 200 GB home quota, but the exclude list matters for rsync
diff-and-retransmit speed on subsequent updates.

Also rsync the Argo shim (needed if you take Path B in Step 4):

```bash
rsync -avz ~/argo-shim-lite/ <user>@<jlse-login>:~/argo-shim-lite/
```

**Checkpoint 1a:** on JLSE login node, verify the tree landed:

```bash
ssh <user>@<jlse-login>
du -sh ~/Agent-Precision   # expect ~4.5 GB with the exclude list above
ls ~/Agent-Precision/baselines/ | head -5   # expect kernel-stem dirs
ls ~/argo-shim-lite/claude-argo-proxy.py    # expect exists, ~2 KB
```

If you see ~7 GB instead of ~4.5 GB, you rsync'd without excluding
`poster-results/` (2.9 GB of archived poster runs) or `error_dumps/`
(8.5 MB of argo-proxy failure dumps). Both are safe to delete on JLSE
(regeneratable historical output, unused by the workflow):

```bash
rm -rf ~/Agent-Precision/poster-results ~/Agent-Precision/error_dumps
rm -rf ~/Agent-Precision/.opencode   # if present
du -sh ~/Agent-Precision             # should now show ~4.5 GB
```

If `baselines/` didn't come across, the rsync excluded it by mistake —
re-run and double-check your exclude list does NOT contain `baselines/`
or `kokkos/`.

---

## Step 2: qsub -I allocation

From the JLSE login node, find the A100 queue and project:

```bash
qstat -Q                         # list queues
pbsnodes -a | grep -B1 A100      # find A100-tagged nodes
```

Grab an allocation. 4 hours covers S1..S4 comfortably:

```bash
qsub -I -A <project> -q <a100-queue> -l select=1:ngpus=1,walltime=04:00:00
```

`<project>` is your CELS project name; `<a100-queue>` is whatever
`qstat -Q` surfaced for A100 hardware (varies per JLSE testbed —
common names include `arcticus`, `polaris-a100`, `gpu_a100`).

**Checkpoint 2a:** once the shell prompt returns (may take seconds to
minutes depending on queue depth), you're on a compute node. Sanity-check:

```bash
hostname                # expect a compute-node name, not the login host
nvidia-smi              # expect an A100 in the device list
echo $PBS_JOBID         # expect a job id — confirms you're inside the alloc
```

If `nvidia-smi` says "No devices were found", you got a non-GPU node;
`exit` and re-`qsub` with a corrected `select=` expression.

---

## Step 3: module tree init

The JLSE Spack + Lmod modules are NOT visible until you load the
Spack init module:

```bash
module load spack/linux-rhel7-x86_64
module avail kokkos 2>&1 | tee ~/jlse-modules-kokkos.txt
module avail cuda 2>&1 | tee ~/jlse-modules-cuda.txt
module avail python 2>&1 | tee ~/jlse-modules-python.txt
module avail gcc 2>&1 | tee ~/jlse-modules-gcc.txt
```

**Checkpoint 3a:** paste the four `.txt` files into the chat if the
next steps get confused about module names. What we're looking for:

- **CUDA:** any `cuda/11.x` or `cuda/12.x`; note the version.
- **Kokkos:** ideally a module with `+cuda` and `+openmp` variants in
  its name (Spack module names encode variants). If none exists,
  you'll build from source in Step 8.
- **Python:** `python/3.10` or later. If only 3.9 is available, the
  workflow's `dict | None` annotations will crash at import — use a
  newer Spack Python or a system `python3.10+` if installed.
- **GCC:** the workflow expects `g++` supporting `-std=c++20`; that's
  gcc 10+. On rhel7 hosts the system `g++` is often 4.8; you'll want
  a module like `gcc/11.x`.

Load what you need (adjust versions from the checkpoint output):

```bash
module load gcc/11.3.0
module load cuda/12.2.0
module load python/3.10.10
```

**Checkpoint 3b:** verify the compiler stack:

```bash
which g++ && g++ --version | head -1     # expect 10+ (workflow needs c++20)
which nvcc && nvcc --version | tail -1   # expect matching cuda version
which python3 && python3 --version       # expect 3.10+
```

---

## Step 4: probe LLM plumbing route (decides Step 5)

The workflow needs an Anthropic-compatible LLM endpoint reachable from
the compute node. Probe outbound HTTPS:

```bash
curl -v --max-time 10 https://apps.inside.anl.gov/ 2>&1 | head -30
curl -v --max-time 10 https://homes.cels.anl.gov/ 2>&1 | head -30
```

**Decision:**

- **Path A** — both probes return HTTP (any status, even 403) with an
  observable TLS handshake: the compute node has outbound HTTPS. Set
  up the shim locally on the compute node in Step 5A.
- **Path B** — probes hang or return "Connection refused" / "No route
  to host": the compute node is firewalled. Set up the shim on a JLSE
  login node and reverse-tunnel it in Step 5B.

Paste the two curl outputs into the chat if you're unsure which path
you're on.

---

## Step 5A: Argo tunnel (Path A — compute node has outbound)

Runs the same shape as `scripts/run-argo.sh` on the laptop, but
locally on the compute node:

```bash
# Terminal 1 (or backgrounded on the compute node):
ssh -f -N -L 8082:apps.inside.anl.gov:443 <user>@homes.cels.anl.gov

# Terminal 2 (same compute node):
cd ~/argo-shim-lite
pip install --user aiohttp   # if not already installed
python3 claude-argo-proxy.py --port 8083 --upstream https://127.0.0.1:8082 &
```

**Checkpoint 5A-a:** verify both are up:

```bash
ss -tlnp | grep -E '8082|8083'   # expect both LISTENs
curl -sS http://127.0.0.1:8083/health 2>&1 | head -5   # expect 200 or similar
```

If the SSH command prompts for Duo, complete the push; the tunnel
process detaches after auth.

Skip to Step 6.

---

## Step 5B: Argo tunnel (Path B — compute node firewalled)

Bring up the shim on a JLSE login node, then reverse-tunnel it back:

```bash
# On a JLSE login node (separate SSH session, keep the qsub alive):
ssh -f -N -L 8082:apps.inside.anl.gov:443 <user>@homes.cels.anl.gov
cd ~/argo-shim-lite
pip install --user aiohttp
python3 claude-argo-proxy.py --port 8083 --upstream https://127.0.0.1:8082 &
```

**On the compute node** (inside the qsub session), forward the shim's
port from the login node into the compute node's `127.0.0.1:8083`:

```bash
ssh -f -N -L 8083:127.0.0.1:8083 <user>@<jlse-login>
```

The `-L` direction here is compute-node -> login-node (local port
forward from the compute node's perspective). The compute node now
sees `127.0.0.1:8083` as the login node's shim; the workflow's
`ANTHROPIC_BASE_URL=http://127.0.0.1:8083/argoapi/` reaches through
the tunnel transparently.

**Checkpoint 5B-a:**

```bash
ss -tlnp | grep 8083           # expect a LISTEN on 8083 (local end of tunnel)
curl -sS http://127.0.0.1:8083/health 2>&1 | head -5
```

If Duo prompts appear at any of the SSH steps, complete them; expect
one prompt per ssh command.

---

## Step 5 (both paths): export Anthropic env vars

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8083/argoapi/
export ANTHROPIC_AUTH_TOKEN=$USER
```

**Checkpoint 5-a:** confirm the shim actually reaches Argo end-to-end
by asking it for a trivial completion:

```bash
curl -sS -X POST http://127.0.0.1:8083/argoapi/v1/messages \
  -H "x-api-key: $USER" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-opus-4-7","max_tokens":32,"messages":[{"role":"user","content":"say hi"}]}' \
  | head -30
```

Expect a JSON response with a `content` array containing `hi` or similar.
If you get a 404, the base URL is wrong (`/argoapi/` trailing slash matters).
If you get a 400 with `"unknown model"`, the model id needs updating
(see AGENTS.md's "Model names look wrong but aren't" section — do NOT
change `claude-opus-4-7` without verifying against the real backend).

---

## Step 6: Python env + S1

```bash
cd ~/Agent-Precision
python3 -m venv .venv-jlse
source .venv-jlse/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Checkpoint 6a — S1:**

```bash
python -m pytest -q
```

Expect: `788 passed` (or whatever the current count is; check with
`git log --oneline -1 tests/`). All tests are network-free and
monkeypatch the Anthropic SDK, so this must pass even with the shim
untested at this point.

If any test fails, STOP — do not proceed to S2. The most likely cause
is a Python-version mismatch (older than 3.10 chokes on `dict | None`
type annotations in `workflow/tools.py` and elsewhere).

**Checkpoint 6b — S2:** verify `run_agent` reaches the shim:

```bash
python3 - <<'EOF'
from workflow.run_agent import run_agent
result = run_agent(
    "candidate_finder",
    "kernel source:\n\nvoid f(double* x, int n) { for (int i=0;i<n;++i) x[i] *= 2.0; }",
    temperature=0.0,
)
print("OK:", result.get("candidates", "no candidates key")[:2] if isinstance(result.get("candidates"), list) else result)
EOF
```

Expect a printed dict with a `candidates` list containing the `x`
variable. This confirms the whole plumbing (SDK -> shim -> tunnel ->
Argo -> back) works end-to-end on the compute node.

If this hangs, one of the tunnels dropped. If it errors with an HTTP
400, the model id or shim protocol is off — recheck Checkpoint 5-a.

---

## Step 7: CUDA env + S3 (saxpy.cu smoke)

```bash
export AGENT_PRECISION_CUDA_ARCH=sm_80        # A100
export AGENT_PRECISION_RUN_TIMEOUT_SEC=120    # generous first-run
```

**Checkpoint 7a:**

```bash
env | grep AGENT_PRECISION_
```

Expect the two exports above plus (from your shell rc, maybe) the
ARGO ones. `AGENT_PRECISION_KOKKOS_ROOT` and `AGENT_PRECISION_KOKKOS_CXX`
are NOT needed for S3 (CUDA-only).

**Run S3:**

```bash
rm -rf baselines/saxpy   # start clean so probe pipeline reruns
python -m workflow.run test-kernels/cuda/lowerable/saxpy.cu --sig-figs 6 --auto
```

Expected timeline: ~5-15 minutes (dominated by LLM latency + 8 probe
cells). Watch for these landmarks in stdout:

- `spawn_baseline_harness` accepted (four drivers written).
- 8 `probe_step` calls succeed (`quad`/`double`/`float`/`original` x
  `seed=42`/`43`). The quad cell shells out to `g++ + libquadmath`,
  NOT `nvcc`.
- `probe_compare` reports per-precision max_absrel; expect
  `float_seed42` around 0.179 (SAXPY-shaped catastrophic-cancellation),
  `double_seed42` == 0.0.
- `spawn_candidate_finder`, then 3-4 `spawn_variable_analyst` calls
  (one per candidate), then 3-4 `test_variable_downcast` calls.
- Union test passes, `spawn_analyst_finalizer` runs, `spawn_rewriter`
  runs, `spawn_verifier` returns `verdict='accept'`.
- `splice_rewritten_kernel` -> `compile_rewritten_driver` ->
  `run_rewritten_driver` -> `compare_outputs` (`status='ok'`,
  1000000 outputs agreed) -> `measure_speedup` writes `timing.json`.
- `finish` called; run exits 0.

**Checkpoint 7b — S3:**

```bash
ls -la baselines/saxpy/rewritten/timing.json
cat baselines/saxpy/rewritten/timing.json
```

The file must exist and contain `baseline`, `rewritten`, `speedup`,
`speedup_stddev`, `trials_timed` keys. Speedup will be close to 1.0x
(SAXPY is memory-bandwidth-bound; the intended win is on a compute-
bound kernel).

If `compile_rewritten_driver` fails with "nvcc: command not found",
Step 3 didn't load the `cuda/*` module — reload and re-run S3 from
scratch (rm the baselines/saxpy tree first).

---

## Step 8: Kokkos (S4)

**8a: choose a Kokkos install.**

Ideal: a Spack module with both `+cuda` and `+openmp` variants for the
target CUDA + gcc versions. Check `~/jlse-modules-kokkos.txt`
(saved at Step 3). If a matching module exists:

```bash
module load kokkos/<version>-cuda-openmp   # exact name from module avail
echo $KOKKOS_ROOT                          # Spack sets this
ls $KOKKOS_ROOT/bin/nvcc_wrapper           # must exist
export AGENT_PRECISION_KOKKOS_ROOT=$KOKKOS_ROOT
export AGENT_PRECISION_KOKKOS_CXX=$KOKKOS_ROOT/bin/nvcc_wrapper
```

If NO such module exists (the JLSE Kokkos modules are all CPU-only, or
none carry `+cuda`), build from source. Rough shape:

```bash
cd ~
git clone --depth 1 https://github.com/kokkos/kokkos.git kokkos-src
cd kokkos-src && mkdir build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/kokkos-cuda-omp \
  -DCMAKE_CXX_COMPILER=$PWD/../bin/nvcc_wrapper \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE80=ON \
  -DKokkos_CXX_STANDARD=17
make -j 8 && make install
export AGENT_PRECISION_KOKKOS_ROOT=$HOME/kokkos-cuda-omp
export AGENT_PRECISION_KOKKOS_CXX=$HOME/kokkos-cuda-omp/bin/nvcc_wrapper
```

Build takes ~20-30 minutes on an A100 node's host CPUs. Do this in
the same allocation to avoid path issues.

**Checkpoint 8a:**

```bash
echo $AGENT_PRECISION_KOKKOS_ROOT
ls $AGENT_PRECISION_KOKKOS_ROOT/include/Kokkos_Core.hpp   # must exist
ls $AGENT_PRECISION_KOKKOS_ROOT/lib/libkokkoscore*        # must exist
$AGENT_PRECISION_KOKKOS_CXX --version | head -3           # nvcc_wrapper prints g++ underneath
```

**8b: run S4:**

```bash
rm -rf baselines/nbody_force
python -m workflow.run test-kernels/kokkos/mixed/nbody_force.cpp --sig-figs 6 --auto
```

Expected timeline: ~10-25 minutes (more variables than saxpy -> more
LLM calls in the per-variable pipeline). Landmarks:

- Baseline harness accepts, 8 probe cells run. The quad cell is
  plain-C++ + libquadmath (NOT Kokkos, since Kokkos ships no
  `__float128` math overloads). The other 7 cells compile via
  `nvcc_wrapper` and dispatch through Kokkos-CUDA on the A100.
- Per-variable pipeline runs candidate_finder -> ~10 variable_analyst
  calls -> ~10 test_variable_downcast calls -> union test -> possibly
  bisect_variable_downcast -> finalizer -> rewriter -> verifier ->
  splice chain -> compare_outputs -> measure_speedup.
- Uses `nbody_force.cpp.testconfig.json` for N=1024 + fixed seed +
  known input distributions. Verify by looking at `baselines/
  nbody_force/reference.json` — should have `seed: 42` and 1024-element
  output arrays.

**Checkpoint 8b — S4:**

```bash
ls -la baselines/nbody_force/rewritten/timing.json
cat baselines/nbody_force/rewritten/timing.json
grep -c '"status": "ok"' baselines/nbody_force/rewritten/comparison.json
```

`timing.json` exists, `comparison.json` has `status: ok`. Speedup may
be > 1x since nbody has more compute than saxpy and downcasts
propagate through the pairwise-force loop (but don't be surprised if
it's < 1x — the test tolerance and the specific downcast set the
analyst picked determine the outcome).

If Kokkos compilation fails with weird nvcc_wrapper errors,
double-check `AGENT_PRECISION_KOKKOS_CXX` points at the wrapper (not
plain g++) and that `nvcc` from Step 3 is on PATH. `nvcc_wrapper`
dispatches to nvcc for device code, so both must be findable.

---

## Step 9: capture results back to laptop

Before the qsub allocation expires, rsync results back:

```bash
# From the compute node (fastest — direct outbound if the qsub node has it),
# or from a JLSE login node after the alloc ends (data persists in home):
rsync -avz --progress \
  ~/Agent-Precision/baselines/saxpy/ \
  <you>@<laptop>:~/Agent-Precision/baselines/saxpy-jlse-a100/
rsync -avz --progress \
  ~/Agent-Precision/baselines/nbody_force/ \
  <you>@<laptop>:~/Agent-Precision/baselines/nbody_force-jlse-a100/
```

The `-jlse-a100` suffix keeps them side-by-side with the laptop's
existing baselines for comparison (speedups on A100 vs laptop CPU are
the interesting data point).

---

## Known failure modes

**"Streaming is required" ValueError from the SDK on Kokkos runs.**
The harness emits four full drivers -> a large `submit_result` payload.
Should be fine at `max_tokens=32768` + `timeout=600.0` (both set by
`run_agent.py`), but if a specific kernel is verbose enough to hit the
ceiling, you'll see `stop_reason='max_tokens'` in
`baselines/<stem>/llm_calls.jsonl`. Retry the run; the underlying issue
is model-side variance.

**Quad probe cell timeout.** On kernels with heavy transcendentals
(`sin`/`cos`/`exp` in tight loops at N > 1e5), the software-quad host
driver can exceed `AGENT_PRECISION_RUN_TIMEOUT_SEC=120`. Two options:
raise the timeout to 300 or 600 and re-run, OR accept the missing quad
cell (probe_compare hard-errors on `quad_seed42` absent, so raising
the timeout is usually right).

**Argo shim disconnects mid-run.** SSH tunnels sometimes drop after
~15-30 minutes of idle. If S3 or S4 hangs at an LLM call, check
`ss -tlnp | grep 8083` — if it's gone, re-run Step 5A/5B and restart
the workflow (the orchestrator does NOT resume from mid-trace in v0).

**Kokkos-CUDA compilation errors mentioning `long double`.** CUDA
devices don't support `long double`. If the analyst emits an
`emulate` action with `long double` (should not happen — the rewriter
prompt says float-float only), file it as a workflow bug and manually
edit the verdict for that run.

---

## Cleanup

At end of the qsub session, either `exit` or let walltime expire.
Nothing needs to be shut down explicitly (SSH tunnels die with the
session; background shim processes on the login node persist and
should be killed manually: `pkill -f claude-argo-proxy.py`).

The `.venv-jlse/` venv persists in home dir for the next session.
Modules do NOT persist — re-run Step 3 (module loads) at the start
of every new qsub allocation.
