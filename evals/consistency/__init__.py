"""Run-to-run consistency scoring for the workflow.

Given N orchestrator traces from independent runs of the same kernel with
the same flags, score the per-variable verdict agreement, the
finish-gate outcome distribution, and the rewriter-retry distribution.
Distinct from evals/layer2/, which measures accuracy against ground
truth across the corpus.
"""
