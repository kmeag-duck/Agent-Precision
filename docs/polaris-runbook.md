# Polaris A100 bring-up runbook (Option A, interactive)

Operator recipe for bringing the workflow up on ALCF's Polaris (A100
production system) via `qsub -I`. Peer of `docs/jlse-runbook.md` — that
file covers JLSE testbed bring-up (P100 fallback, since JLSE's
`gpu_a100` queue is ACL-restricted to the `nvidia-testbed` group);
this file covers Polaris (unrestricted access, real A100s, production
allocation-tracked scheduling).

This is Option A ("interactive") from the AGENTS.md roadmap. The whole
workflow — LLM orchestrator, compile, run, HITL — happens inside one
`qsub -I` allocation on the compute node. Option B ("async per-tool
scheduler submission") is a separate future story; not covered here.

**Speculative runbook.** Written from ALCF public docs + prior
knowledge of Polaris conventions before empirical verification. Every
non-obvious assumption is marked **`[ASSUMED: verify]`** in bold at
the end of the relevant paragraph. `grep -n 'ASSUMED' docs/polaris-runbook.md`
enumerates the outstanding verification checklist. Correct as you go
through it; each verification lands as a small follow-up commit.

Success criteria (referenced throughout as **S1..S5**):

- **S1** `pytest -q` passes inside the Polaris venv.
- **S2** `run_agent` succeeds from a Polaris Python REPL (LLM plumbing works).
- **S3** `saxpy.cu --sig-figs 6 --auto` runs end-to-end; `baselines/saxpy/rewritten/timing.json` exists.
- **S4** `nbody_force.cpp --sig-figs 6 --auto` runs end-to-end; `baselines/nbody_force/rewritten/timing.json` exists.
- **S5** README + AGENTS.md reflect what actually landed on Polaris (deferred until after S3/S4 pass).

Prereqs: ALCF account, active Polaris allocation, membership in an
ALCF project. The example project name in this runbook is `UIC-HPC`
— **other operators must substitute their own project name in every
`-A` flag and every `/eagle/UIC-HPC/` path.**

---

## Step 1: rsync from laptop (run on laptop, before session)

Confirm laptop-side state is clean:

```bash
cd ~/Agent-Precision
git status              # expect: nothing to commit, working tree clean
git log --oneline -3    # confirm the JLSE bring-up + this runbook commit are present
python -m pytest -q     # sanity: all tests pass on laptop
```

Rsync the repo to Polaris. Target `/eagle/<project>/<user>/Agent-Precision/`,
NOT `$HOME` — Polaris `$HOME` is a small quota (~50 GB) and won't fit
`baselines/` (4.3 GB) alongside a venv, kokkos install, and rsync
history. Eagle is the standard project scratch. **`[ASSUMED: verify
/eagle/UIC-HPC/<user>/ exists and is writable via `ls -ld
/eagle/UIC-HPC/kmeagher` on Polaris login]`**

Same exclude list as the JLSE runbook (rsync does NOT honor
`.gitignore` by default):

```bash
rsync -avz --progress \
  --exclude='.git/' \
  --exclude='.venv*/' \
  --exclude='__pycache__/' \
  --exclude='evals/results/' \
  --exclude='poster-results/' \
  --exclude='error_dumps/' \
  --exclude='.opencode/' \
  ~/Agent-Precision/ <user>@polaris.alcf.anl.gov:/eagle/UIC-HPC/<user>/Agent-Precision/
```

Duo push will prompt during the SSH handshake. Total transfer ~4.5 GB
with the exclude list above; well under Eagle project quotas (typically
1-10 TB per project). **`[ASSUMED: verify UIC-HPC project's Eagle
quota with `myprojectquotas` or similar on Polaris login]`**

Also rsync the Argo shim (needed for Step 5):

```bash
rsync -avz ~/argo-shim-lite/ <user>@polaris.alcf.anl.gov:/eagle/UIC-HPC/<user>/argo-shim-lite/
```

**Checkpoint 1a:** on Polaris login node, verify the tree landed:

```bash
ssh <user>@polaris.alcf.anl.gov
du -sh /eagle/UIC-HPC/<user>/Agent-Precision   # expect ~4.5 GB
ls /eagle/UIC-HPC/<user>/Agent-Precision/baselines/ | head -5   # expect kernel-stem dirs
ls /eagle/UIC-HPC/<user>/argo-shim-lite/claude-argo-proxy.py    # expect exists, ~2 KB
```

If you accidentally rsync'd without excludes and see ~7 GB, the same
cleanup recipe from `jlse-runbook.md` Checkpoint 1a applies:

```bash
rm -rf /eagle/UIC-HPC/<user>/Agent-Precision/poster-results
rm -rf /eagle/UIC-HPC/<user>/Agent-Precision/error_dumps
rm -rf /eagle/UIC-HPC/<user>/Agent-Precision/.opencode   # if present
```

---

## Step 2: qsub -I allocation on Polaris

Polaris uses PBS Pro (same commands as JLSE, but different queue
semantics: `-A <project>` is REQUIRED because Polaris is
allocation-tracked, and `-l filesystems=` declaration is REQUIRED
because jobs get killed on their first access to an undeclared
filesystem).

**Queue choice:**

- **`debug`** — 2 nodes reserved, 1 hour max walltime, no allocation
  charge. Best for the initial bring-up (S1/S2 confirmations, first
  Argo probe). Fastest to start.
- **`preemptable`** — up to 10 nodes, 72 hour max walltime, but jobs
  can be preempted (killed mid-run) by higher-priority production
  jobs. Best for S3/S4 smoke runs where you need 4+ hours. Accept
  the preemption risk (the workflow doesn't checkpoint — a preempted
  run must restart from scratch, but `baselines/<stem>/` is
  incremental so a re-run reuses everything up to the failure point).
- **`prod`** — up to 496 nodes, 24 hour max. Overkill for a
  single-node smoke; charged against your allocation. Use only when
  you're running the full 17-kernel corpus sweep.

**`[ASSUMED: verify queue names are `debug` / `preemptable` / `prod`
on the current Polaris scheduler config with `qstat -Q` on Polaris
login]`**

Bring-up (Steps 3-7, all quick):

```bash
qsub -I -A UIC-HPC -q debug -l select=1,walltime=01:00:00,filesystems=home:eagle
```

Smoke runs (Steps 8-9, need more time):

```bash
qsub -I -A UIC-HPC -q preemptable -l select=1,walltime=04:00:00,filesystems=home:eagle
```

Notes:

- `select=1` requests one node = 4× A100 40GB + 1× EPYC 7543P + 512 GB
  DDR4. The workflow only uses one GPU; the other three sit idle.
  That's standard Polaris practice for single-GPU workloads.
- `filesystems=home:eagle` declares your job will touch `$HOME` and
  `/eagle/`. Add `:grand` if your project also uses `/lus/grand/`.
- No `ngpus=` needed — Polaris compute nodes ARE 4-GPU by definition,
  the scheduler doesn't do per-GPU subdivision.

**Checkpoint 2a:** once the shell prompt returns (seconds to minutes
for `debug`, minutes to hours for `preemptable` depending on load):

```bash
hostname                # expect a compute-node name like xNNNNcNNsNbN
nvidia-smi              # expect 4× A100 40GB, driver 5xx+, CUDA 12.x
echo $PBS_JOBID         # expect a job id — confirms you're inside the alloc
echo $PBS_NODEFILE      # path to a file listing the nodes in your alloc
cat $PBS_NODEFILE       # should be one line (a single hostname) for select=1
```

If `nvidia-smi` says "No devices were found", something's badly wrong
with the alloc — `exit` and re-`qsub`.

---

## Step 3: module env

Polaris uses Lmod. The default module set on login uses `PrgEnv-nvhpc`
(NVIDIA HPC SDK: nvc, nvc++, nvfortran), but the workflow needs `g++`
for the quad-oracle host driver (plain-C++ + libquadmath, no CUDA).
Swap to `PrgEnv-gnu` OR keep nvhpc and explicitly load `gcc`.

**`[ASSUMED: verify PrgEnv-nvhpc is the default with `module list` after
a fresh Polaris login]`**

Recommended module load sequence:

```bash
module swap PrgEnv-nvhpc PrgEnv-gnu     # gives g++, gcc, gfortran
module load cudatoolkit-standalone      # nvcc + CUDA runtime headers
module load cray-python                 # Python 3.10+
```

**`[ASSUMED: verify `cray-python` provides Python 3.10+ with `python3
--version` after module load; if it's older, look for a `python/3.10.*`
or similar module]`**

**Checkpoint 3a:** verify the compiler stack:

```bash
which g++ && g++ --version | head -1     # expect 10+ (workflow needs c++20)
which nvcc && nvcc --version | tail -1   # expect CUDA 12.x
which python3 && python3 --version       # expect 3.10+
```

If any of these are missing or too old, adjust module loads and retry.
On Polaris the exact module versions rotate periodically; `module
avail` finds the current set.

---

## Step 4: probe network + Argo plumbing (compute node)

Polaris compute nodes require an HTTP proxy for outbound network.
Per ALCF network docs the standard proxy is `proxy.alcf.anl.gov:3128`.

**`[ASSUMED: verify proxy hostname and port with the ALCF User Guide
network policy page — the value here is derived from ALCF public
docs, not empirical probing]`**

Probe from the compute node:

```bash
# 1. Does the ALCF proxy work at all?
curl -sSv --max-time 15 -x http://proxy.alcf.anl.gov:3128 https://api.anthropic.com/ 2>&1 | head -20

# 2. Does the proxy allow the ANL-internal Argo host?
curl -sSv --max-time 15 -x http://proxy.alcf.anl.gov:3128 https://apps.inside.anl.gov/ 2>&1 | head -20

# 3. Baseline: is direct (no-proxy) outbound blocked?
curl -sSv --max-time 10 https://api.anthropic.com/ 2>&1 | head -10
```

**Decision matrix:**

- **Path C (expected):** (1) and (2) both succeed with HTTP responses
  (any status, even 403). Proxy works, and it allows both external and
  ANL-internal destinations. Proceed to Step 5.
- **Path C-external-only:** (1) succeeds but (2) fails. Proxy allows
  external only. Argo won't work through it; you'd need to skip the
  shim and point the SDK directly at `api.anthropic.com` using an
  `ANTHROPIC_API_KEY` instead of the `$USER` Argo token.
  **`[ASSUMED: if this happens, we cross that bridge — for now the
  runbook assumes Path C is available]`**
- **Path C-auth-required:** curl returns 407 Proxy Authentication
  Required. Uncommon on Polaris but possible; would need Kerberos
  or basic auth. Check the `WWW-Proxy-Authenticate` header in the
  407 response for the exact scheme.
- **Path A (direct works, unlikely):** (3) succeeds. Compute node
  has direct outbound; skip the proxy entirely. Set no `HTTPS_PROXY`
  in Step 5.

Paste the three curl outputs in chat if you're unsure which branch
you're on.

---

## Step 5: Argo shim (Path C — the expected path)

The shim (`~/argo-shim-lite/claude-argo-proxy.py` from the laptop,
now at `/eagle/UIC-HPC/<user>/argo-shim-lite/` on Polaris) runs
locally on the compute node. Its `--upstream
https://apps.inside.anl.gov` requests route through `HTTPS_PROXY`
transparently — no SSH tunnel needed.

**Critical ordering:** aiohttp reads proxy env at import time. Set
proxy env BEFORE starting the shim; setting it after has no effect
and the shim will try direct outbound and hang.

```bash
# Set proxy env FIRST:
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export NO_PROXY=127.0.0.1,localhost

# NO_PROXY is CRITICAL — the workflow's SDK calls to the local shim
# on :8083 would otherwise get routed through the ALCF proxy, which
# would either infinite-loop or return 502 (proxy trying to fetch a
# 127.0.0.1 URL that only exists inside the local process).

# Start the shim:
cd /eagle/UIC-HPC/<user>/argo-shim-lite
pip install --user aiohttp   # if not already installed
python3 claude-argo-proxy.py --port 8083 --upstream https://apps.inside.anl.gov &
```

Then export the standard Anthropic env vars for the workflow:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8083/argoapi/
export ANTHROPIC_AUTH_TOKEN=$USER
```

**Checkpoint 5a:** verify the shim is up and reaches Argo end-to-end:

```bash
ss -tlnp | grep 8083           # expect a LISTEN on 8083
curl -sS http://127.0.0.1:8083/health 2>&1 | head -5   # expect 200 or similar
```

**Checkpoint 5b — end-to-end LLM plumbing test (S2 precursor):**

```bash
curl -sS -X POST http://127.0.0.1:8083/argoapi/v1/messages \
  -H "x-api-key: $USER" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-opus-4-7","max_tokens":32,"messages":[{"role":"user","content":"say hi"}]}' \
  | head -30
```

Expect a JSON response with a `content` array containing `hi` or
similar. If you get a 404, the base URL is wrong (`/argoapi/`
trailing slash matters). If you get a 400 with `"unknown model"`,
the model id needs updating (see AGENTS.md's "Model names look wrong
but aren't" section — do NOT change `claude-opus-4-7` without
verifying against the real backend).

---

## Step 6: Python env + S1

```bash
cd /eagle/UIC-HPC/<user>/Agent-Precision
python3 -m venv .venv-polaris
source .venv-polaris/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Checkpoint 6a — S1:**

```bash
python -m pytest -q
```

Expect all tests to pass (current count as of the JLSE bring-up
commit: 788). All tests are network-free and monkeypatch the
Anthropic SDK, so this must pass even with the shim untested at this
point.

If any test fails, STOP — do not proceed to S2. The most likely
cause is a Python-version mismatch (older than 3.10 chokes on `dict
| None` type annotations in `workflow/tools.py` and elsewhere).

**Checkpoint 6b — S2:** verify `run_agent` reaches the shim through
the proxy:

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
variable. This confirms the whole plumbing (SDK -> HTTPS_PROXY ->
ALCF proxy -> Argo -> back through the shim on `:8083`) works
end-to-end on the compute node.

If this hangs, check `HTTPS_PROXY` is set in the current shell (may
have been lost if you spawned a subshell). If it errors with an HTTP
400, the model id or shim protocol is off — recheck Checkpoint 5b.

---

## Step 7: CUDA env for A100

```bash
export AGENT_PRECISION_CUDA_ARCH=sm_80        # A100 = compute capability 8.0
export AGENT_PRECISION_RUN_TIMEOUT_SEC=120    # generous first-run
```

**Compute capability reference (for future ports):**

| Arch flag | GPU family | Example device |
|-----------|-----------|----------------|
| `sm_60`   | Pascal    | P100 (JLSE fallback) |
| `sm_70`   | Volta     | V100 |
| `sm_75`   | Turing    | T4, RTX 20xx |
| `sm_80`   | Ampere    | **A100 (Polaris)** |
| `sm_86`   | Ampere    | RTX 30xx |
| `sm_89`   | Ada       | RTX 40xx (laptop default) |
| `sm_90`   | Hopper    | H100 |
| `sm_100`  | Blackwell | B200 |

**Checkpoint 7a:**

```bash
env | grep AGENT_PRECISION_
```

Expect the two exports above plus (from Step 5) the ARGO ones and
the HTTPS_PROXY set. `AGENT_PRECISION_KOKKOS_ROOT` /
`AGENT_PRECISION_KOKKOS_CXX` are NOT needed for S3 (CUDA-only) —
those come in Step 8.

**Run S3:**

```bash
rm -rf baselines/saxpy   # start clean so probe pipeline reruns
python -m workflow.run test-kernels/cuda/lowerable/saxpy.cu --sig-figs 6 --auto
```

Expected timeline: ~5-15 minutes on Polaris (A100 compile+run is
fast; LLM latency dominates). Landmarks and interpretation are the
same as JLSE runbook Step 7 — see there for the full list. Key
differences on Polaris A100:

- Speedups on A100 will be more meaningful than the RTX 4060 laptop
  default (higher bandwidth, more FP32 throughput). Even for
  bandwidth-bound SAXPY expect a modest speedup rather than the
  ~0.88x seen on the laptop.

**Checkpoint 7b — S3:**

```bash
ls -la baselines/saxpy/rewritten/timing.json
cat baselines/saxpy/rewritten/timing.json
```

The file must exist and contain `baseline`, `rewritten`, `speedup`,
`speedup_stddev`, `trials_timed` keys. Speedup on A100 will differ
from laptop RTX 4060 numbers; that's expected and part of what makes
the Polaris runbook worthwhile.

---

## Step 8: Kokkos (S4 prep)

**8a: choose a Kokkos install.**

Ideal: a Polaris Spack module with both `+cuda` and `+openmp`
variants for the loaded CUDA + gcc versions.

```bash
module avail kokkos 2>&1 | tee ~/polaris-modules-kokkos.txt
```

**`[ASSUMED: verify whether Polaris ships a Kokkos+CUDA+OpenMP
module. If yes, prefer the module route; if no, build from source
below]`**

If a matching module exists:

```bash
module load kokkos/<version>-cuda-openmp   # exact name from module avail
echo $KOKKOS_ROOT                          # Spack sets this
ls $KOKKOS_ROOT/bin/nvcc_wrapper           # must exist
export AGENT_PRECISION_KOKKOS_ROOT=$KOKKOS_ROOT
export AGENT_PRECISION_KOKKOS_CXX=$KOKKOS_ROOT/bin/nvcc_wrapper
```

If NO such module exists, build from source into your Eagle
project space (don't build in `$HOME` — quota):

```bash
cd /eagle/UIC-HPC/<user>
git clone --depth 1 https://github.com/kokkos/kokkos.git kokkos-src
cd kokkos-src && mkdir build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/eagle/UIC-HPC/<user>/kokkos-cuda-omp \
  -DCMAKE_CXX_COMPILER=$PWD/../bin/nvcc_wrapper \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE80=ON \
  -DKokkos_CXX_STANDARD=17
make -j 16 && make install
export AGENT_PRECISION_KOKKOS_ROOT=/eagle/UIC-HPC/<user>/kokkos-cuda-omp
export AGENT_PRECISION_KOKKOS_CXX=/eagle/UIC-HPC/<user>/kokkos-cuda-omp/bin/nvcc_wrapper
```

Build takes ~15-25 minutes on Polaris's EPYC 7543P (32 cores).

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
LLM calls in the per-variable pipeline). Uses
`nbody_force.cpp.testconfig.json` for N=1024 + fixed seed + known
input distributions.

**Checkpoint 8b — S4:**

```bash
ls -la baselines/nbody_force/rewritten/timing.json
cat baselines/nbody_force/rewritten/timing.json
grep -c '"status": "ok"' baselines/nbody_force/rewritten/comparison.json
```

`timing.json` exists, `comparison.json` has `status: ok`.

---

## Step 9: capture results back to laptop

Before the qsub allocation expires (and before any preemption on
`preemptable`), rsync results back to the laptop:

```bash
# From the compute node (if outbound rsync via proxy works) OR from
# a Polaris login node after the alloc ends (data on Eagle persists):
rsync -avz --progress \
  /eagle/UIC-HPC/<user>/Agent-Precision/baselines/saxpy/ \
  <you>@<laptop>:~/Agent-Precision/baselines/saxpy-polaris-a100/
rsync -avz --progress \
  /eagle/UIC-HPC/<user>/Agent-Precision/baselines/nbody_force/ \
  <you>@<laptop>:~/Agent-Precision/baselines/nbody_force-polaris-a100/
```

The `-polaris-a100` suffix keeps them side-by-side with the laptop's
existing baselines (and any JLSE `-p100` baselines) for cross-machine
speedup comparison.

Alternative: rsync from laptop, pulling from Polaris login. Simpler
if outbound from the compute node is finicky:

```bash
# On laptop, after allocation ends:
rsync -avz --progress \
  <you>@polaris.alcf.anl.gov:/eagle/UIC-HPC/<user>/Agent-Precision/baselines/saxpy/ \
  ~/Agent-Precision/baselines/saxpy-polaris-a100/
```

---

## Known failure modes

**Filesystem declaration missing.** If you `qsub -I` without
`-l filesystems=home:eagle`, your job will run but be killed the
first time it touches `/eagle/`. Symptom: workflow starts, gets
partway through Step 6 (pytest), then the shell dies with no clear
error. Fix: exit, re-qsub with the `filesystems=` declaration.

**Preemption on `preemptable` queue.** Jobs on the `preemptable`
queue can be killed with 5 minutes' notice when a higher-priority
`prod` job needs the node. The workflow doesn't checkpoint, so a
preempted S3/S4 run must restart from Step 7 (compile+run). However,
`baselines/<stem>/` is incremental — a re-run reuses the probe cells,
compiled drivers, and comparator artifacts from the pre-preemption
state, so recovery is usually just a few minutes of re-running rather
than a full from-scratch retry. For the initial bring-up prefer
`debug` (never preempted, but 1h max walltime).

**HTTPS_PROXY set after shim start.** aiohttp reads proxy env at
process import. If you set `HTTPS_PROXY` after starting the shim, it
will keep trying direct outbound and hang. Kill the shim (`pkill -f
claude-argo-proxy.py`), re-export, restart.

**NO_PROXY not set → SDK loops through proxy.** The workflow's SDK
calls to `http://127.0.0.1:8083/...` will get routed through
`proxy.alcf.anl.gov:3128` without `NO_PROXY=127.0.0.1`, producing
either infinite loops or 502 from the proxy (which tries to fetch a
`127.0.0.1` URL that only exists inside the local process). Symptom:
first `run_agent` call in S2 hangs indefinitely.

**PrgEnv-nvhpc default lacks g++.** The quad-oracle host driver
(plain-C++ + libquadmath, one of the 8 probe cells) needs `g++`, not
`nvc++`. If you skipped Step 3's `module swap PrgEnv-nvhpc
PrgEnv-gnu`, `compile_baseline_driver` will fail on the quad cell
with "command not found: g++". Fix: swap PrgEnvs and re-run.

**Argo shim disconnects mid-run.** Long-running workflow calls
sometimes see the shim's upstream HTTPS session drop. Symptom: S3 or
S4 hangs at an LLM call for >5 minutes. Check `ss -tlnp | grep 8083`
— if the shim died, restart it and re-run the workflow (the
orchestrator does NOT resume from mid-trace in v0; you re-invoke
`python -m workflow.run ...` and the incremental `baselines/<stem>/`
state gets reused).

**Quad probe cell timeout.** On kernels with heavy transcendentals
(`sin`/`cos`/`exp` in tight loops at N > 1e5), the software-quad host
driver can exceed `AGENT_PRECISION_RUN_TIMEOUT_SEC=120`. Raise to 300
or 600 and re-run. probe_compare hard-errors on `quad_seed42` missing,
so raising the timeout is usually right rather than accepting the
gap.

---

## Cleanup

At end of the qsub session, either `exit` or let walltime expire.
Nothing needs explicit shutdown (background shim processes die with
the session). The `.venv-polaris/` venv persists on Eagle for the
next session. Modules do NOT persist — re-run Step 3 (module loads)
at the start of every new qsub allocation.

To free Eagle space between sessions:

```bash
# Remove failed run artifacts:
rm -rf /eagle/UIC-HPC/<user>/Agent-Precision/error_dumps
# Remove per-kernel baselines you no longer need:
rm -rf /eagle/UIC-HPC/<user>/Agent-Precision/baselines/<stem>
```

Do NOT delete `.venv-polaris/`, `kokkos-cuda-omp/`, or the
`argo-shim-lite/` sibling — those take minutes-to-hours to rebuild.
