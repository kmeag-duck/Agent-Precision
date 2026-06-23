"""Tests for workflow.languages — language-profile registry + detection.

Phase B added CUDA as the second language profile. Phase C-1 added HIP
as the third. The registry must expose all three profiles, detection by
source suffix must route `.cu` to CUDA, `.hip` to HIP, and `.cpp` to
Kokkos, and `get_profile_by_id` must raise on an unknown id (no silent
Kokkos fallback — a typo'd `language_id` from the orchestrator must
surface as an error).

These tests are pure data assertions on the in-memory PROFILES dict and
on detect_language() / get_profile_by_id() — no subprocess, no network,
no file I/O.
"""

import pytest

from workflow.languages import (
    CUDA_PROFILE,
    HIP_PROFILE,
    KOKKOS_PROFILE,
    PROFILES,
    detect_language,
    get_profile_by_id,
)
from workflow.languages.base import LanguageProfile


# ---------- PROFILES registry shape ----------


def test_profiles_contains_kokkos_and_cuda():
    """The PROFILES registry exposes the two Phase-B profiles under their canonical ids — `kokkos` and `cuda` — keyed by each profile's own `.id` field."""
    assert "kokkos" in PROFILES
    assert "cuda" in PROFILES
    assert PROFILES["kokkos"] is KOKKOS_PROFILE
    assert PROFILES["cuda"] is CUDA_PROFILE


def test_profiles_contains_hip():
    """The PROFILES registry exposes the Phase C-1 HIP profile under its canonical id `hip`, keyed by its own `.id` field — same invariant the kokkos/cuda check applies."""
    assert "hip" in PROFILES
    assert PROFILES["hip"] is HIP_PROFILE


def test_every_profile_is_a_language_profile_dataclass():
    """Every registered profile is a LanguageProfile instance with the structural fields the orchestrator and tools layer rely on (id, source_suffixes, driver_filename, dynamic_verification, baseline_harness_*, build_compile_command, preflight)."""
    required_attrs = (
        "id",
        "display_name",
        "source_suffixes",
        "driver_filename",
        "env_required",
        "dynamic_verification",
        "baseline_harness_system_prompt",
        "baseline_harness_output_schema",
        "build_compile_command",
        "preflight",
        "detect_from_source",
    )
    for name, profile in PROFILES.items():
        assert isinstance(profile, LanguageProfile), (
            f"PROFILES[{name!r}] is not a LanguageProfile"
        )
        for attr in required_attrs:
            assert hasattr(profile, attr), (
                f"PROFILES[{name!r}] missing attribute {attr!r}"
            )


def test_profiles_keys_match_their_id_field():
    """Each PROFILES key equals the corresponding profile's `.id`; without this invariant, get_profile_by_id (which looks up by id) would silently miss entries that were registered under a different key."""
    for key, profile in PROFILES.items():
        assert key == profile.id, (
            f"PROFILES key {key!r} does not match profile.id {profile.id!r}"
        )


# ---------- Source-suffix detection ----------


def test_detect_language_cu_returns_cuda():
    """A `.cu` source path resolves to CUDA_PROFILE via unambiguous suffix match — kernel_source content is not consulted (CUDA owns `.cu` exclusively)."""
    # Source content is deliberately Kokkos-shaped to prove the suffix
    # is what drives the decision, not the body.
    kokkos_shaped = "#include <Kokkos_Core.hpp>\nvoid kernel() {}\n"
    assert detect_language("path/to/kernel.cu", kokkos_shaped) is CUDA_PROFILE


def test_detect_language_hip_returns_hip():
    """A `.hip` source path resolves to HIP_PROFILE via unambiguous suffix match — kernel_source content is not consulted (HIP owns `.hip` exclusively in v0)."""
    # Source content is deliberately Kokkos-shaped to prove the suffix
    # is what drives the decision, not the body.
    kokkos_shaped = "#include <Kokkos_Core.hpp>\nvoid kernel() {}\n"
    assert detect_language("path/to/kernel.hip", kokkos_shaped) is HIP_PROFILE


def test_detect_language_hip_suffix_is_case_insensitive():
    """`.HIP` (uppercase) resolves to HIP — detect_language normalizes the suffix to lowercase before lookup, so a kernel path's case does not change the routing. Same invariant as the `.CU` case."""
    assert detect_language("PATH/KERNEL.HIP", "") is HIP_PROFILE


def test_detect_language_hip_does_not_inspect_source_for_dot_hip():
    """For `.hip` inputs HIP's `_detect_from_source` probe is NEVER consulted (the suffix is unambiguous). Empty source still routes to HIP — proves the probe is not on the `.hip` path."""
    assert detect_language("kernel.hip", "") is HIP_PROFILE


def test_detect_language_cpp_returns_kokkos():
    """A `.cpp` source path resolves to KOKKOS_PROFILE; Kokkos is the only profile that claims `.cpp` today, so the suffix alone disambiguates."""
    assert detect_language("path/to/kernel.cpp", "void kernel() {}\n") is KOKKOS_PROFILE


def test_detect_language_cu_suffix_is_case_insensitive():
    """`.CU` (uppercase) resolves to CUDA — detect_language normalizes the suffix to lowercase before lookup, so a kernel path's case does not change the routing."""
    assert detect_language("PATH/KERNEL.CU", "") is CUDA_PROFILE


def test_detect_language_unknown_suffix_falls_back_to_kokkos():
    """An unknown suffix falls back to KOKKOS_PROFILE; this preserves v0 behavior where any unrecognized file was assumed to be Kokkos. The downstream preflight may still error, but detect_language itself never raises."""
    assert detect_language("kernel.unknown", "") is KOKKOS_PROFILE
    assert detect_language("kernel_no_suffix", "") is KOKKOS_PROFILE


def test_detect_language_cuda_does_not_inspect_source_for_cu():
    """For `.cu` inputs CUDA's `_detect_from_source` probe is NEVER consulted (the suffix is unambiguous). Empty source still routes to CUDA — proves the probe is not on the .cu path."""
    assert detect_language("kernel.cu", "") is CUDA_PROFILE


# ---------- Profile attribute spot-checks ----------


def test_cuda_profile_claims_dot_cu_exclusively():
    """CUDA_PROFILE.source_suffixes contains exactly `.cu`; no other registered profile claims that suffix, which is what lets the suffix-match branch of detect_language resolve without a content probe."""
    assert ".cu" in CUDA_PROFILE.source_suffixes
    cu_claimants = [
        name for name, p in PROFILES.items() if ".cu" in p.source_suffixes
    ]
    assert cu_claimants == ["cuda"]


def test_cuda_profile_driver_filename_is_dot_cu():
    """CUDA_PROFILE.driver_filename is `driver.cu` so the compile and splice tools target the right extension; the Kokkos analogue is `driver.cpp`."""
    assert CUDA_PROFILE.driver_filename == "driver.cu"
    assert KOKKOS_PROFILE.driver_filename == "driver.cpp"


def test_cuda_profile_dynamic_verification_is_true():
    """CUDA_PROFILE enables the dynamic-verification chain (baseline-harness -> compile -> run -> splice -> compile-rewritten -> run-rewritten -> compare) — the Phase B finish-gate flip was specifically to make `.cu` honor this gate the same way `.cpp` already does."""
    assert CUDA_PROFILE.dynamic_verification is True
    assert KOKKOS_PROFILE.dynamic_verification is True


def test_hip_profile_claims_dot_hip_exclusively():
    """HIP_PROFILE.source_suffixes contains exactly `.hip`; no other registered profile claims that suffix. Mirrors the `.cu`-exclusivity check for CUDA — both are language-owned extensions that bypass the content-probe pass."""
    assert ".hip" in HIP_PROFILE.source_suffixes
    hip_claimants = [
        name for name, p in PROFILES.items() if ".hip" in p.source_suffixes
    ]
    assert hip_claimants == ["hip"]


def test_hip_profile_does_not_claim_cpp():
    """HIP_PROFILE deliberately does NOT claim `.cpp` in v0 — real ROCm codebases often use `.cpp` with `#include <hip/hip_runtime.h>`, but supporting that would force a content-probe disambiguation against Kokkos / SYCL / OpenMP-offload, which is deferred until a HIP toolchain is available for smoke testing."""
    assert ".cpp" not in HIP_PROFILE.source_suffixes


def test_hip_profile_driver_filename_is_dot_hip():
    """HIP_PROFILE.driver_filename is `driver.hip` so the compile and splice tools target the right extension; the Kokkos analogue is `driver.cpp`, CUDA's is `driver.cu`."""
    assert HIP_PROFILE.driver_filename == "driver.hip"


def test_hip_profile_dynamic_verification_is_true():
    """HIP_PROFILE enables the dynamic-verification chain — same finish-gate behavior as CUDA and Kokkos. The chain is language-agnostic; the profile only chooses the compile command and harness prompt."""
    assert HIP_PROFILE.dynamic_verification is True


def test_hip_profile_id_and_display_name():
    """HIP_PROFILE.id is the lowercase `hip` token used in PROFILES keys and the orchestrator's `baseline_harness_<id>` tool-name suffix; display_name is the human-readable `HIP C++` shown in logs."""
    assert HIP_PROFILE.id == "hip"
    assert HIP_PROFILE.display_name == "HIP C++"


# ---------- get_profile_by_id ----------


def test_get_profile_by_id_returns_registered_profile():
    """get_profile_by_id resolves each registered id to the corresponding LanguageProfile object."""
    assert get_profile_by_id("kokkos") is KOKKOS_PROFILE
    assert get_profile_by_id("cuda") is CUDA_PROFILE
    assert get_profile_by_id("hip") is HIP_PROFILE


def test_get_profile_by_id_raises_on_unknown_id():
    """A typo'd or unregistered id raises KeyError (no silent fallback). This is what makes a buggy `language_id` argument from the orchestrator surface as a `_error()` tool result instead of routing to Kokkos by default."""
    with pytest.raises(KeyError) as excinfo:
        get_profile_by_id("rust")
    # The error message should name the bad id AND list the known ids,
    # so an operator triaging the trace can immediately see what was
    # expected.
    msg = str(excinfo.value)
    assert "rust" in msg
    assert "kokkos" in msg
    assert "cuda" in msg
    assert "hip" in msg
