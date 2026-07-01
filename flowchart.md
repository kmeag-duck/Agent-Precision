# Orchestrator flowchart

The orchestrator is a single Claude conversation that drives an
agent-and-deterministic-tool pipeline from kernel source to a
verified, lower-precision rewrite. Tool calls execute as the
orchestrator issues them; under `--auto` every executed tool is also
journaled to `baselines/<kernel_stem>/orchestrator_trace.jsonl`.

The diagram below shows the orchestrator as a subgraph whose interior
is the **happy path** — the 13-step chain for a Kokkos `.cpp` kernel
when `--no-probe` is not passed (probe pipeline runs between baseline
and analyst). Error and reject branches are summarized as side-loops
off the relevant nodes rather than redrawn for every step; the
deterministic finish-gate (verifier accept ∧ comparator ok) is the
single exit. The same diagram appears in `README.md` under
"Workflow"; this file adds the prose notes on each deviation branch
below. For pipeline rules (finish-gate semantics, probe-pipeline
gating, oracle promotion) see `AGENTS.md`.

The same shape applies to CUDA, HIP, SYCL, and OpenMP-offload — only
steps 4–5 (`probe_step ×8` and `probe_compare`) are skipped on
profiles with empty `probe_precisions` (currently all non-Kokkos
profiles), shortening the chain to 11 steps; the
`spawn_baseline_harness_<id>` agent variant, driver file extension
(`.cu` / `.hip` / `.cpp`), and compiler invoked in steps 2 / 10 are
the only other differences.

```mermaid
flowchart TD
  U["python -m workflow.run &lt;kernel&gt;<br/>[--sig-figs N | --decimal-digits N]<br/>[--auto] [--no-probe]"]
  F(["print final kernel + notes"])

  U ==> Orch

  subgraph Orch ["Orchestrator (Claude conversation loop)<br/>routes every step · gates finish · MAX_TURNS=60"]
    direction TB

    S1["<b>1. spawn_baseline_harness_&lt;id&gt;</b><br/><i>id ∈ {kokkos, cuda, hip, sycl, omp_offload};<br/>Kokkos emits 4 drivers, others emit 1</i><br/>→ baselines/&lt;stem&gt;/driver.&lt;ext&gt;<br/>(+ probe/&lt;precision&gt;/driver.cpp ×4 on Kokkos)"]:::agent
    S2["<b>2. compile_baseline_driver</b><br/>per-profile compiler + env vars<br/>(KOKKOS_ROOT / CUDA_ARCH / HIP_ARCH /<br/>SYCL_CXX / OMP_CXX / OMP_TARGET)<br/>→ baselines/&lt;stem&gt;/driver"]:::det
    S3["<b>3. run_baseline_driver</b><br/>RUN_TIMEOUT_SEC (default 60)<br/>→ baselines/&lt;stem&gt;/reference.json"]:::det
    S4["<b>4. probe_step ×8</b><br/><i>Kokkos only; 4 precisions × seeds {42, 43};<br/>fused compile+run per cell</i><br/>→ probe/&lt;precision&gt;_seed&lt;N&gt;/reference.json"]:::det
    S5["<b>5. probe_compare</b><br/><i>Kokkos only; aggregates vs quad_seed42;<br/>then ORACLE PROMOTION:<br/>quad_seed42/reference.json →<br/>baselines/&lt;stem&gt;/reference.json</i><br/>→ probe/evidence.json<br/>(appended to next analyst task)"]:::det
    S6["<b>6. spawn_analyst</b><br/><i>K-fold ensemble via AGENT_PRECISION_ANALYST_K;<br/>aggregator.aggregate_analyst_verdicts</i><br/>→ {variables, rework, precision_budget,<br/>overall_notes}"]:::agent
    S7["<b>7. spawn_rewriter</b><br/>→ {rewritten_code, summary_of_changes}"]:::agent
    S8["<b>8. spawn_verifier</b><br/><i>K-lens panel (faithfulness / budget / edge_cases)<br/>via AGENT_PRECISION_VERIFIER_K;<br/>STRICT accept (all lenses must accept)</i>"]:::agent
    S9["<b>9. splice_rewritten_kernel</b><br/><i>text I/O: replace between<br/>KERNEL BEGIN / END sentinels</i><br/>→ baselines/&lt;stem&gt;/rewritten/driver.&lt;ext&gt;"]:::det
    S10["<b>10. compile_rewritten_driver</b><br/>same per-profile compiler as step 2<br/>→ baselines/&lt;stem&gt;/rewritten/driver"]:::det
    S11["<b>11. run_rewritten_driver</b><br/>→ baselines/&lt;stem&gt;/rewritten/reference.json"]:::det
    S12["<b>12. compare_outputs</b><br/><i>baseline vs rewritten under tolerance_json;<br/>writes comparison.json on pass AND fail</i>"]:::det
    S13["<b>13. finish</b><br/><i>code-side finish-gate</i>"]:::gate

    S1 ==> S2;
    S2 ==> S3;
    S3 ==> S4;
    S4 ==> S5;
    S5 ==> S6;
    S3 -. "non-Kokkos or --no-probe" .-> S6;
    S6 ==> S7;
    S7 ==> S8;
    S8 == "accept" ==> S9;
    S9 ==> S10;
    S10 ==> S11;
    S11 ==> S12;
    S12 == "status=ok" ==> S13;

    %% deviations from the happy path, as compact side-loops
    S8 -. "reject" .-> S7;
    S12 -. "status=error" .-> S6;
    S1 -. "malformed payload" .-> S1;
    S2 -. "compile error (non-fatal)" .-> S2;
    S3 -. "run error (non-fatal)" .-> S3;
  end

  S13 == "verifier accept ∧ compare ok" ==> F
  S13 -. "blocked: synthetic<br/>{status:error,<br/>is_error:true}<br/>tool_result" .-> S6

  classDef agent fill:#dbeafe,stroke:#1e3a8a,stroke-width:1px,color:#0f172a
  classDef det fill:#dcfce7,stroke:#14532d,stroke-width:1px,color:#0f172a
  classDef gate fill:#fef3c7,stroke:#78350f,stroke-width:1px,color:#0f172a
```

Legend: **blue** = LLM agent call (one entry per agent type in
`workflow/registry.py`); **green** = deterministic non-LLM tool
defined in `workflow/tools.py`; **yellow** = decision / gate. Solid
thick arrows (`==>`) are the happy path; dotted thin arrows are the
documented deviations (verifier reject, comparator mismatch, non-
fatal compile / run errors on the baseline chain, and the finish-
gate's synthetic-error retry).

## Notes on the deviations

The dotted-arrow side-loops are deliberately schematic — they show
which node the orchestrator typically returns to on each failure
class, not every possible retry path. The full rules:

- **Step 8 reject → step 7 (rewriter)**. The verifier's `concerns`
  list is fed back into the next `spawn_rewriter` task so the
  rewriter can address them directly; the analyst's verdict from
  step 6 is reused. After one retry the orchestrator may instead
  loop back to step 6 (`spawn_analyst`) if the verifier's concerns
  suggest the verdict itself was wrong.
- **Step 12 mismatch → step 6 (analyst)**. By system-prompt
  convention and `_FinishGateState` design, a numerical mismatch
  after a verifier accept means the verifier was over-permissive;
  the right response is a fresh analyst verdict, not just another
  rewrite of the same verdict. The synthetic
  `{status:'error', is_error:true}` tool result that blocks a
  premature `finish` call uses the same routing.
- **Step 1 error**. If the baseline-harness payload is malformed
  (missing both `drivers` and `driver_source`) the orchestrator
  raises a `RuntimeError` and the run ends.
- **Steps 2 / 3 errors are non-fatal to the pipeline**. The chain
  stalls (downstream steps that need `reference.json` cannot run),
  but the orchestrator does not abort — `finish` is still reachable
  on profiles without `dynamic_verification`. On Kokkos /
  CUDA / HIP / SYCL / OMP-offload (all `dynamic_verification=True`
  today), the chain stall transitively blocks `finish` via
  `_FinishGateState`'s `requires_comparator` check.
- **MAX_TURNS=60**. The only hard backstop against runaway loops.
  Raised from 40 to leave headroom for the probe pipeline (up to 9
  extra tool calls per Kokkos run) plus one verifier-retry cycle.

## Keeping this in sync

Nothing in `workflow/` reads this file — it is documentation only —
but if you change the pipeline (add or remove a spawn tool, change
the finish-gate, add a language profile that alters step counts),
update `flowchart.md`, the copy of this diagram in `README.md`
under "Workflow", and the relevant sections of `AGENTS.md` in the
same change so they do not drift.
