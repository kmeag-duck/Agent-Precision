"""Drift detector for the Layer 2 ground-truth registry.

The ground truth lives in two places — `test-kernels/SUMMARY.md` (human
narrative) and `evals/layer2/expected.py` (machine-readable). These
tests don't compare the two directly (SUMMARY.md is prose), but they
DO assert that:

  - every kernel source file in `test-kernels/{cuda,kokkos}/` has an
    entry in `EXPECTED` (no kernel is silently missed by the harness),
  - every `EXPECTED` entry points at a file that actually exists (no
    stale entries after a kernel rename),
  - each entry has a category matching its directory and a
    tolerance kind/value the workflow CLI will accept.

When a kernel is added or renamed under `test-kernels/`, one of these
tests fails until `expected.py` is updated. That's the whole point.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.layer2.expected import EXPECTED, ExpectedKernel


# Repo root: tests/ -> repo root is parent.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEST_KERNELS_ROOT = _REPO_ROOT / "test-kernels"

_VALID_CATEGORIES = frozenset({"lowerable", "needs_precision", "mixed"})
_VALID_TOLERANCE_KINDS = frozenset({"sig_figs", "decimal_digits"})
_VALID_ACTIONS = frozenset({"downcast", "emulate", "keep"})


def _discover_kernel_files() -> list[Path]:
    """All `.cpp` and `.cu` files under test-kernels/, sorted for
    deterministic test ordering. Excludes SUMMARY.md and any README.
    """
    found: list[Path] = []
    for ext in ("*.cpp", "*.cu"):
        found.extend(_TEST_KERNELS_ROOT.rglob(ext))
    return sorted(found)


def test_every_kernel_file_has_an_expected_entry():
    """Every kernel source under test-kernels/ appears in EXPECTED."""
    discovered = _discover_kernel_files()
    assert discovered, (
        f"No kernel files found under {_TEST_KERNELS_ROOT}; the discovery "
        "glob is broken or test-kernels/ moved."
    )
    discovered_rel = {
        str(p.relative_to(_REPO_ROOT)) for p in discovered
    }
    expected_paths = set(EXPECTED.keys())
    missing = discovered_rel - expected_paths
    assert not missing, (
        "Kernel files exist under test-kernels/ but have no entry in "
        f"evals/layer2/expected.py: {sorted(missing)}. Add them (and "
        "update SUMMARY.md) before the Layer 2 harness can score them."
    )


def test_every_expected_entry_points_at_a_real_file():
    """Every EXPECTED.path exists on disk (no stale entries)."""
    stale = [
        path for path in EXPECTED
        if not (_REPO_ROOT / path).is_file()
    ]
    assert not stale, (
        f"EXPECTED references files that no longer exist: {stale}. "
        "A kernel was renamed/deleted without updating expected.py."
    )


@pytest.mark.parametrize(
    "path,entry",
    sorted(EXPECTED.items()),
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_expected_entry_is_well_formed(path: str, entry: ExpectedKernel):
    """Each EXPECTED entry has a valid category, tolerance, and per-variable map."""
    assert entry.path == path, (
        f"EXPECTED key {path!r} does not match its entry's path "
        f"{entry.path!r} (registry constructed inconsistently)."
    )
    assert entry.category in _VALID_CATEGORIES, (
        f"{path}: category {entry.category!r} is not one of "
        f"{sorted(_VALID_CATEGORIES)}."
    )
    # Category should match the parent directory under test-kernels/.
    # path is e.g. "test-kernels/kokkos/lowerable/vector_add.cpp".
    parts = Path(path).parts
    assert len(parts) >= 4 and parts[0] == "test-kernels", (
        f"{path}: expected layout test-kernels/<lang>/<category>/<file>."
    )
    dir_category = parts[2]
    assert dir_category == entry.category, (
        f"{path}: declared category {entry.category!r} disagrees with "
        f"directory category {dir_category!r}."
    )
    assert entry.tolerance_kind in _VALID_TOLERANCE_KINDS, (
        f"{path}: tolerance_kind {entry.tolerance_kind!r} is not one of "
        f"{sorted(_VALID_TOLERANCE_KINDS)}."
    )
    assert isinstance(entry.tolerance_value, int) and entry.tolerance_value > 0, (
        f"{path}: tolerance_value must be a positive int, got "
        f"{entry.tolerance_value!r}."
    )
    for var_name, action in entry.per_variable.items():
        assert isinstance(var_name, str) and var_name, (
            f"{path}: per_variable contains an empty or non-string key."
        )
        assert action in _VALID_ACTIONS, (
            f"{path}: per_variable[{var_name!r}] = {action!r} is not one "
            f"of {sorted(_VALID_ACTIONS)}."
        )


# Kernels under test-kernels/*/lowerable/ that are EMPIRICALLY known to
# require some 'keep' verdicts despite the directory category. These are
# carve-outs from the "lowerable => all-downcast" sanity check, NOT a
# license for SUMMARY.md to drift silently. Adding an entry here is a
# deliberate decision documented at the EXPECTED entry; see expected.py
# module docstring ("Methodology note") and the per-entry comment for
# the empirical evidence.
_LOWERABLE_KEEP_CARVEOUTS: frozenset[str] = frozenset({
    # Catastrophic cancellation: ~1.4% of outputs at |z| < 1e-7 violate
    # rtol=1e-6. See baselines/vector_add/orchestrator_trace.jsonl.
    "test-kernels/kokkos/lowerable/vector_add.cpp",
    # Same argument by analogy (same kernel shape, same inputs).
    "test-kernels/cuda/lowerable/vector_add.cu",
    # Same cancellation pathology: y = a*x + y with a=2.5, x, y ~ U(-1, 1)
    # produces ~0.6% of outputs at |y_new| << 1 where fp32 input
    # quantization dominates. ~422/65536 mismatches empirically observed
    # at rtol=1e-6. See baselines/saxpy_bounded/orchestrator_trace.jsonl.
    "test-kernels/kokkos/lowerable/saxpy_bounded.cpp",
    # Same kernel shape as the Kokkos saxpy_bounded; CUDA differs only
    # in launch syntax. ~5865/1048576 mismatches empirically observed.
    # See baselines/saxpy/orchestrator_trace.jsonl.
    "test-kernels/cuda/lowerable/saxpy.cu",
    # Phase 1b (2026-07-22): honest quad oracle exposed that
    # sigmoid(x) with x ~ U(-10, 10) has float_seed42.max_absrel=0.9975
    # vs quad — every input pushes exp(-x) through a region where fp32
    # loses order-of-magnitude precision. The earlier Phase-1a smoke
    # run's "safely lowerable" verdict compared float against a double
    # baseline that gave 0 relative error, hiding the failure.
    # See baselines/sigmoid/probe/evidence.json.
    "test-kernels/cuda/lowerable/sigmoid.cu",
})


# Needs-precision-category kernels where a subset of variables is
# nevertheless expected to downcast. Same carve-out philosophy as
# _LOWERABLE_KEEP_CARVEOUTS: the directory category ("needs_precision"
# = long-trajectory or accumulation-dominated) is correct for the
# kernel's state carriers, but per-variable analysis shows that some
# thread-local intermediates can safely downcast. Adding an entry here
# is a deliberate empirical decision documented at the EXPECTED entry;
# it does NOT change the kernel's directory category.
_NEEDS_PRECISION_DOWNCAST_CARVEOUTS: frozenset[str] = frozenset({
    # Phase 1b (2026-07-22): the two thread-local force-computation
    # intermediates r2 = x^2 + y^2 and inv_r3 = 1/(r2*sqrt(r2)) are
    # computed and consumed within a single kernel step; they are
    # never accumulated across steps. Singleton + union downcast tests
    # both passed; comparator passed at sig_figs=3 on 262144 outputs.
    # State arrays (x/y/vx/vy) still expected "keep". See
    # baselines/orbit_integrator/probe/evidence.json and the entry
    # header comment.
    "test-kernels/cuda/needs_precision/orbit_integrator.cu",
})


def test_lowerable_entries_only_specify_downcast():
    """Lowerable category sanity check: every scored variable expects
    'downcast' UNLESS the kernel is on the documented carve-out list.

    Carve-outs exist for kernels where the directory category
    ('lowerable' = no per-element-rounding pathology) doesn't match the
    empirical reality at the configured tolerance (cancellation on a
    fraction of outputs pushes fp32 past rtol). The carve-out list is
    small and explicit on purpose: it forces a deliberate code change
    to add one, rather than letting a typo silently weaken the
    invariant.
    """
    for path, entry in EXPECTED.items():
        if entry.category != "lowerable":
            continue
        non_downcast = {
            name: action
            for name, action in entry.per_variable.items()
            if action != "downcast"
        }
        if path in _LOWERABLE_KEEP_CARVEOUTS:
            # Carved out: must have at least one non-downcast verdict
            # (otherwise the carve-out is stale and should be removed).
            assert non_downcast, (
                f"{path}: listed in _LOWERABLE_KEEP_CARVEOUTS but every "
                f"per_variable verdict is 'downcast'. Remove the carve-out."
            )
            continue
        assert not non_downcast, (
            f"{path}: lowerable kernels should expect 'downcast' for every "
            f"scored variable; found {non_downcast}. If this is a new "
            f"empirical finding, add the kernel to _LOWERABLE_KEEP_CARVEOUTS "
            f"and document the evidence at the EXPECTED entry."
        )


def test_needs_precision_entries_only_specify_keep():
    """Needs-precision category sanity check: no 'downcast' or 'emulate'
    expected UNLESS the kernel is on the documented carve-out list.

    Carve-outs exist for needs_precision kernels where some
    thread-local intermediates can safely downcast even though the
    kernel's state carriers require double. The carve-out list is
    small and explicit on purpose (same philosophy as
    _LOWERABLE_KEEP_CARVEOUTS): it forces a deliberate code change to
    add one, rather than letting a typo silently weaken the invariant.
    """
    for path, entry in EXPECTED.items():
        if entry.category != "needs_precision":
            continue
        non_keep = {
            name: action
            for name, action in entry.per_variable.items()
            if action != "keep"
        }
        if path in _NEEDS_PRECISION_DOWNCAST_CARVEOUTS:
            # Carved out: must have at least one non-keep verdict
            # (otherwise the carve-out is stale and should be removed).
            assert non_keep, (
                f"{path}: listed in _NEEDS_PRECISION_DOWNCAST_CARVEOUTS "
                f"but every per_variable verdict is 'keep'. Remove the "
                f"carve-out."
            )
            continue
        assert not non_keep, (
            f"{path}: needs_precision kernels should expect 'keep' for "
            f"every scored variable; found {non_keep}. If this is a new "
            f"empirical finding, add the kernel to "
            f"_NEEDS_PRECISION_DOWNCAST_CARVEOUTS and document the "
            f"evidence at the EXPECTED entry."
        )
