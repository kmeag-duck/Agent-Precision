"""Language profiles for the dynamic-verification chain.

The orchestrator's chain — baseline-harness -> compile -> run -> splice ->
compile-rewritten -> run-rewritten -> compare — used to be hardcoded for
Kokkos C++ kernels. Each language a kernel is written in (Kokkos C++,
CUDA, SYCL, HIP, OpenMP target offload, ...) has its own driver-file
extension, compiler invocation, splice-sentinel comment syntax, and
baseline-harness prompt; this package abstracts those differences behind
a `LanguageProfile` dataclass and a `PROFILES` registry.

Lookup happens in `run_orchestrator` once per run via
`detect_language(kernel_path, kernel_source)`. The resolved profile is
then threaded through `_format_baseline_block`, `_FinishGateState`, and
every deterministic tool (each grew a required `language_id` argument).
Adding a new language is one new profile module here plus an entry in
PROFILES; the orchestrator loop and the tools layer do not change.
"""

from __future__ import annotations

from pathlib import Path

from .base import LanguageProfile
from .kokkos import KOKKOS_PROFILE

# Ordered registry. Insertion order is the tie-break order in
# detect_language()'s content-probe pass for ambiguous .cpp inputs:
# Kokkos first preserves the historical default behavior, then SYCL /
# HIP-cpp / OpenMP-offload as those profiles land.
PROFILES: dict[str, LanguageProfile] = {
    KOKKOS_PROFILE.id: KOKKOS_PROFILE,
}


def get_profile_by_id(language_id: str) -> LanguageProfile:
    """Look up a profile by its `id` field. Raises on unknown id.

    Used by every deterministic tool's wrapper in workflow.tools when it
    receives a `language_id` argument from the orchestrator. The raise
    (not a fallback) is deliberate: a typo'd language_id from the
    orchestrator must surface as a `_error()` tool result, not silently
    route to Kokkos.
    """
    try:
        return PROFILES[language_id]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(
            f"Unknown language_id: {language_id!r}. Known: {known}."
        )


def detect_language(
    kernel_path: str, kernel_source: str
) -> LanguageProfile:
    """Resolve the LanguageProfile for a kernel file.

    Algorithm:
      1. Match by source suffix. Each profile declares its own suffixes
         (e.g. CUDA owns .cu; HIP owns .hip). When the suffix is
         unambiguous across PROFILES, return immediately.
      2. When the suffix is ambiguous (multiple profiles claim it —
         today this only happens for .cpp, claimed by Kokkos / SYCL /
         HIP-cpp / OpenMP-offload), run each candidate's
         `detect_from_source(kernel_source)` in insertion order. First
         True wins.
      3. If no probe matches, fall back to the first candidate (Kokkos
         today). This preserves the v0 behavior where any unrecognized
         .cpp was treated as Kokkos.
    """
    suffix = Path(kernel_path).suffix.lower()
    candidates = [p for p in PROFILES.values() if suffix in p.source_suffixes]
    if not candidates:
        # No registered profile claims this suffix. Default to Kokkos
        # to preserve v0 behavior; downstream preflight may still error
        # if the chosen toolchain cannot process the file.
        return KOKKOS_PROFILE
    if len(candidates) == 1:
        return candidates[0]
    for profile in candidates:
        if profile.detect_from_source(kernel_source):
            return profile
    return candidates[0]


__all__ = [
    "LanguageProfile",
    "PROFILES",
    "KOKKOS_PROFILE",
    "get_profile_by_id",
    "detect_language",
]
