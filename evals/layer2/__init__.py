"""Layer 2 evaluation: end-to-end scoring of the workflow over test-kernels/.

Scope (deliberately limited):
  - Score the analyst's per-variable verdict against the hand-transcribed
    ground truth in `expected.py` (which mirrors test-kernels/SUMMARY.md).
  - Score whether the run reached `finish` and whether the comparator
    accepted the rewrite in the same rewrite cycle.

Out of scope for this layer (see AGENTS.md "Planned next steps"):
  - Per-stage scoring of verifier or rewriter.
  - Rewriter code-quality judgments beyond the comparator's verdict.
  - Closing the loop back into the workflow (strictly read-only).
"""
