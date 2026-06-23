# Orchestrator flowchart

High-level view of the orchestrator loop. The orchestrator is a Claude
conversation; its tools are one per agent type plus `finish`. Every
proposed tool call passes through a human-in-the-loop pause (`y`/`n`/`q`)
before it runs. Rejection (`n`) is fed back to the orchestrator as
`{"status": "rejected_by_user"}` so it can self-correct without burning
another agent API call; quit (`q`) exits the loop. `--auto` skips the
pause entirely (every tool is auto-approved) and writes a JSONL trace
of every executed tool to
`baselines/<kernel_stem>/orchestrator_trace.jsonl`; user rejections
cannot occur in that mode.

```mermaid
flowchart TD
  U["python -m workflow.run &lt;kernel&gt; [--sig-figs N | --decimal-digits N] [--auto]"] --> O["Orchestrator (Claude)"]
  O -- "tool_use" --> H{"HITL pause<br/>y / n / q<br/>(skipped under --auto;<br/>writes trace.jsonl)"}
  H -- "n: {status: rejected_by_user}" --> O
  H -- "q" --> X(["exit"])
  H -- "y / auto" --> D{"which tool?"}
  D -- "spawn_precision_advisor<br/>(only if no tolerance flag)" --> P["Precision-advisor agent"]
  D -- "spawn_analyst" --> A["Analyst agent<br/>(K-fold ensemble via<br/>AGENT_PRECISION_ANALYST_K;<br/>aggregator.aggregate_analyst_verdicts)"]
  D -- "spawn_rewriter" --> R["Rewriter agent"]
  D -- "spawn_verifier" --> V["Verifier agent<br/>(K-lens panel via<br/>AGENT_PRECISION_VERIFIER_K;<br/>verifier_panel.aggregate_verifier_verdicts)"]
  D -- "spawn_baseline_harness<br/>(Kokkos .cpp only)" --> B["Baseline-harness agent"]
  D -- "compile_baseline_driver<br/>(deterministic; after baseline_harness)" --> C["g++ subprocess<br/>(AGENT_PRECISION_KOKKOS_ROOT)"]
  D -- "run_baseline_driver<br/>(deterministic; after compile_baseline_driver)" --> N["./driver subprocess<br/>(AGENT_PRECISION_RUN_TIMEOUT_SEC)<br/>writes baselines/&lt;stem&gt;/reference.json"]
  D -- "splice_rewritten_kernel<br/>(deterministic; after verifier accept + run_baseline_driver)" --> S["text I/O: replace between<br/>KERNEL BEGIN / END sentinels<br/>writes baselines/&lt;stem&gt;/rewritten/driver.cpp"]
  D -- "compile_rewritten_driver<br/>(deterministic; after splice)" --> CR["g++ subprocess<br/>(AGENT_PRECISION_KOKKOS_ROOT)<br/>writes baselines/&lt;stem&gt;/rewritten/driver"]
  D -- "run_rewritten_driver<br/>(deterministic; after compile_rewritten_driver)" --> NR["./driver subprocess<br/>(AGENT_PRECISION_RUN_TIMEOUT_SEC)<br/>writes baselines/&lt;stem&gt;/rewritten/reference.json"]
  D -- "compare_outputs<br/>(deterministic; after run_rewritten_driver)" --> CMP["arithmetic + file I/O:<br/>diff baseline vs rewritten reference.json<br/>under tolerance_json<br/>writes baselines/&lt;stem&gt;/rewritten/comparison.json"]
  D -- "finish" --> G{"finish-gate<br/>(code-side)"}
  G -- ".cpp: verifier accept AND compare_outputs ok" --> F(["print final kernel + notes"])
  G -- ".cu: verifier accept" --> F
  G -- "blocked: synthetic {status:'error', is_error:true} tool_result<br/>(comparator error → prefer spawn_analyst on retry)" --> O
  P -- "{kind, value, rationale, confidence, alternative}" --> O
  A -- "{variables, rework, precision_budget, overall_notes}" --> O
  R -- "{rewritten_code, summary_of_changes}" --> O
  V -- "{verdict, per_variable, concerns}" --> O
  B -- "{driver_source, ...} -> baselines/&lt;stem&gt;/driver.cpp" --> O
  C -- "{status, stdout, stderr, artifacts}" --> O
  N -- "{status, stdout, stderr, artifacts}" --> O
  S -- "{status, stdout, stderr, artifacts}" --> O
  CR -- "{status, stdout, stderr, artifacts}" --> O
  NR -- "{status, stdout, stderr, artifacts}" --> O
  CMP -- "{status, stdout, stderr, artifacts}" --> O
```

For the intended happy-path *sequence* through the conversation (who
speaks when, including HITL turns), see the sequence diagram in
`README.md` under "Workflow". For pipeline rules (advisor at most once,
finish-gate semantics, etc.) see `AGENTS.md`.

## Happy path

The 12-step chain through the orchestrator for a Kokkos `.cpp` kernel
when no tolerance flag is passed (so the advisor runs once up front and
the full dynamic-verification chain runs at the end). Every step is
routed by the orchestrator loop — the agent or deterministic tool
returns its result to the orchestrator, which then issues the next
tool call. Errors, rejections, and verifier `reject` branches are
deliberately omitted here; see the top-level flowchart above for those.
For a CUDA `.cu` input, steps 2–4 and 8–11 are skipped (verifier-only
gate) and the chain reduces to advisor → analyst → rewriter →
verifier(accept) → finish.

```mermaid
flowchart TD
  Orch[("Orchestrator (Claude conversation loop)<br/>routes every step, gates finish")]

  Orch ==> S1
  S1["1. spawn_precision_advisor<br/><i>(only when no --sig-figs/--decimal-digits)</i><br/>→ tolerance {kind, value, source}"]
  S2["2. spawn_baseline_harness<br/><i>(Kokkos .cpp only)</i><br/>→ writes baselines/&lt;stem&gt;/driver.cpp"]
  S3["3. compile_baseline_driver<br/>→ baselines/&lt;stem&gt;/driver"]
  S4["4. run_baseline_driver<br/>→ baselines/&lt;stem&gt;/reference.json"]
  S5["5. spawn_analyst<br/>→ {variables, rework, precision_budget, overall_notes}"]
  S6["6. spawn_rewriter<br/>→ {rewritten_code, summary_of_changes}"]
  S7["7. spawn_verifier<br/>→ verdict='accept'"]
  S8["8. splice_rewritten_kernel<br/>→ baselines/&lt;stem&gt;/rewritten/driver.cpp"]
  S9["9. compile_rewritten_driver<br/>→ baselines/&lt;stem&gt;/rewritten/driver"]
  S10["10. run_rewritten_driver<br/>→ baselines/&lt;stem&gt;/rewritten/reference.json"]
  S11["11. compare_outputs<br/>→ status='ok'<br/>(+ baselines/&lt;stem&gt;/rewritten/comparison.json)"]
  S12(["12. finish<br/>(finish-gate opens:<br/>verifier accept ∧ compare ok)"])

  S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8 --> S9 --> S10 --> S11 --> S12
  S12 ==> Orch
```
