"""Evaluation harnesses for the precision-rewrite workflow.

Layer 1 (workflow plumbing) lives under top-level `tests/`. Layer 2 lives
here. The two are deliberately separate: Layer 1 monkeypatches the
Anthropic SDK and asserts internal plumbing; Layer 2 actually runs the
workflow end-to-end against real kernels and scores the result against
hand-transcribed ground truth in `evals/layer2/expected.py`.
"""
