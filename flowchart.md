# Orchestrator flowchart

High-level view of the orchestrator loop. The orchestrator is a Claude
conversation; its tools are one per agent type plus `finish`. Every
proposed tool call passes through a human-in-the-loop pause (`y`/`n`/`q`)
before it runs. Rejection (`n`) is fed back to the orchestrator as
`{"status": "rejected_by_user"}` so it can self-correct without burning
another agent API call; quit (`q`) exits the loop.

```mermaid
flowchart TD
  U["python -m workflow.run &lt;kernel&gt; [--sig-figs N | --decimal-digits N]"] --> O["Orchestrator (Claude)"]
  O -- "tool_use" --> H{"HITL pause<br/>y / n / q"}
  H -- "n: {status: rejected_by_user}" --> O
  H -- "q" --> X(["exit"])
  H -- "y" --> D{"which tool?"}
  D -- "spawn_precision_advisor<br/>(only if no tolerance flag)" --> P["Precision-advisor agent"]
  D -- "spawn_analyst" --> A["Analyst agent"]
  D -- "spawn_rewriter" --> R["Rewriter agent"]
  D -- "spawn_verifier" --> V["Verifier agent"]
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
