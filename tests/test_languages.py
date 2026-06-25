"""Tests for workflow.languages — language-profile registry + detection.

Phase B added CUDA as the second language profile. Phase C-1 added HIP
as the third. Phase C-2 added SYCL as the fourth. Phase C-3 added
OpenMP target-offload as the fifth — the second `.cpp`-claiming
profile after SYCL, so the content-probe tie-break in
detect_language() is now a 3-way decision among Kokkos / SYCL /
OMP-offload. The registry must expose all five profiles, detection by
source suffix must route `.cu` to CUDA, `.hip` to HIP, and `.cpp` to
Kokkos OR SYCL OR OMP-offload based on a content probe (insertion
order in PROFILES — Kokkos, then SYCL, then OMP-offload — breaks
multi-probe ties), and `get_profile_by_id` must raise on an unknown
id (no silent Kokkos fallback — a typo'd `language_id` from the
orchestrator must surface as an error).

These tests are pure data assertions on the in-memory PROFILES dict and
on detect_language() / get_profile_by_id() — no subprocess, no network,
no file I/O.
"""

import pytest

from workflow.languages import (
    CUDA_PROFILE,
    HIP_PROFILE,
    KOKKOS_PROFILE,
    OMP_OFFLOAD_PROFILE,
    PROFILES,
    SYCL_PROFILE,
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


def test_profiles_contains_sycl():
    """The PROFILES registry exposes the Phase C-2 SYCL profile under its canonical id `sycl`, keyed by its own `.id` field — same invariant the kokkos / cuda / hip checks apply."""
    assert "sycl" in PROFILES
    assert PROFILES["sycl"] is SYCL_PROFILE


def test_profiles_insertion_order_puts_kokkos_before_sycl():
    """In PROFILES, KOKKOS_PROFILE appears before SYCL_PROFILE; this insertion order is the tie-break order in detect_language()'s content-probe pass for `.cpp` inputs. Kokkos first preserves the historical default behavior (a `.cpp` with no SYCL or Kokkos markers still routes to Kokkos)."""
    ids = list(PROFILES.keys())
    assert ids.index("kokkos") < ids.index("sycl"), (
        f"PROFILES insertion order broken: kokkos must precede sycl, got {ids}"
    )


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
        "probe_precisions",
        "baseline_precision",
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


# ---------- Probe-pipeline fields (v1) ----------
#
# v1 added two new LanguageProfile fields driving the pre-analyst
# probe pipeline (commits 1-5 of the v1 series). Kokkos is the only
# profile populated in v1 — quad is available via libquadmath under
# Kokkos::Serial, but no equivalent shipped library exists on
# CUDA/HIP/SYCL/OMP-offload, so those profiles keep the defaults
# (`probe_precisions=()`, `baseline_precision="double"`) until the
# deferred Commit 6 lands. The defaults make non-Kokkos profiles
# behave exactly as they did in v0: empty probe_precisions means the
# orchestrator skips the probe step entirely, and `double` matches
# the historical baseline behavior.


def test_kokkos_profile_baseline_precision_is_quad():
    """KOKKOS_PROFILE.baseline_precision == 'quad' so the probe measures float/double drift against true ground truth rather than against a same-precision reference; without this the float-vs-double signal collapses to zero on kernels whose true output IS the double answer."""
    assert KOKKOS_PROFILE.baseline_precision == "quad"


def test_kokkos_profile_probe_precisions_v1_set():
    """KOKKOS_PROFILE.probe_precisions == ('quad', 'double', 'float', 'mixed_io'); these are the four probe configurations the orchestrator runs (per seed) before invoking the analyst. The `quad` entry runs the baseline configuration as a self-consistency check; `double` and `float` measure drift; `mixed_io` (outputs at baseline precision, intermediates at float) gives the analyst a coarse signal on output-vs-intermediate sensitivity without per-variable instrumentation."""
    assert KOKKOS_PROFILE.probe_precisions == (
        "quad",
        "double",
        "float",
        "mixed_io",
    )


def test_non_kokkos_profiles_have_default_probe_fields():
    """CUDA / HIP / SYCL / OMP-offload all keep the v0-compatible defaults: empty `probe_precisions` (orchestrator skips the probe entirely) and `baseline_precision='double'` (matches the historical baseline). Deferred Commit 6 will lift these to populated probe sets, but v1 ships Kokkos-only probe support."""
    for profile in (CUDA_PROFILE, HIP_PROFILE, SYCL_PROFILE, OMP_OFFLOAD_PROFILE):
        assert profile.probe_precisions == (), (
            f"{profile.id} has non-empty probe_precisions={profile.probe_precisions!r}; "
            f"v1 only populates Kokkos"
        )
        assert profile.baseline_precision == "double", (
            f"{profile.id} has baseline_precision={profile.baseline_precision!r}; "
            f"v1 only changes Kokkos's baseline"
        )


def test_language_profile_probe_field_defaults():
    """The LanguageProfile dataclass itself provides safe defaults (`probe_precisions=()`, `baseline_precision='double'`) so a future profile that does not opt into the probe pipeline can omit both fields entirely; without this default a new profile would have to know about probe machinery just to be defined."""
    import dataclasses

    fields = {f.name: f for f in dataclasses.fields(LanguageProfile)}
    assert fields["probe_precisions"].default == ()
    assert fields["baseline_precision"].default == "double"


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


def test_detect_language_cpp_with_no_markers_returns_kokkos():
    """A `.cpp` source with neither Kokkos nor SYCL markers falls back to KOKKOS_PROFILE via insertion-order tie-break in PROFILES. This preserves the v0 behavior where any unrecognized `.cpp` was treated as Kokkos — important because SYCL joining the `.cpp` pool in Phase C-2 must not silently re-route legacy Kokkos kernels that happen to lack a `Kokkos::` token in the snippet shown to the probe."""
    assert detect_language("path/to/kernel.cpp", "void kernel() {}\n") is KOKKOS_PROFILE


def test_detect_language_cpp_with_kokkos_markers_returns_kokkos():
    """A `.cpp` source carrying canonical Kokkos markers (`#include <Kokkos_Core.hpp>` or `Kokkos::` namespace use or KOKKOS_LAMBDA) routes to KOKKOS_PROFILE. The Kokkos probe is consulted first because Kokkos precedes SYCL in PROFILES."""
    assert detect_language("k.cpp", "#include <Kokkos_Core.hpp>\nvoid f() {}\n") is KOKKOS_PROFILE
    assert detect_language("k.cpp", "void f() { Kokkos::parallel_for(...); }\n") is KOKKOS_PROFILE
    assert detect_language("k.cpp", "void f() { auto g = KOKKOS_LAMBDA(int i){}; }\n") is KOKKOS_PROFILE


def test_detect_language_cpp_with_sycl_markers_returns_sycl():
    """A `.cpp` source carrying canonical SYCL markers (`<sycl/sycl.hpp>` include, `sycl::queue`, or `sycl::buffer`) routes to SYCL_PROFILE. With Kokkos's probe returning False for these inputs, SYCL is the first candidate whose probe returns True and wins the content-probe pass."""
    assert detect_language("k.cpp", "#include <sycl/sycl.hpp>\nint main() {}\n") is SYCL_PROFILE
    assert detect_language("k.cpp", "void f() { sycl::queue q; }\n") is SYCL_PROFILE
    assert detect_language("k.cpp", "void f() { sycl::buffer<double> b(n); }\n") is SYCL_PROFILE


def test_detect_language_cpp_with_both_kokkos_and_sycl_markers_returns_kokkos():
    """A pathological `.cpp` source that triggers BOTH the Kokkos and SYCL probes (e.g. a Kokkos kernel that also references `sycl::queue` in a comment-stripped string) routes to KOKKOS_PROFILE. Kokkos precedes SYCL in PROFILES, so its probe is consulted first and the first True wins — this is the documented insertion-order tie-break."""
    mixed = (
        "#include <Kokkos_Core.hpp>\n"
        "#include <sycl/sycl.hpp>\n"
        "void f() { Kokkos::parallel_for(...); sycl::queue q; }\n"
    )
    assert detect_language("k.cpp", mixed) is KOKKOS_PROFILE


def test_detect_language_cpp_suffix_is_case_insensitive():
    """`.CPP` (uppercase) routes through the same suffix-normalization path as `.cpp`; SYCL markers in the body still win when Kokkos's probe returns False. Mirrors the `.CU` / `.HIP` case-insensitivity invariants."""
    assert detect_language("PATH/KERNEL.CPP", "sycl::queue q;\n") is SYCL_PROFILE
    assert detect_language("PATH/KERNEL.CPP", "") is KOKKOS_PROFILE


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


def test_sycl_profile_claims_dot_cpp():
    """SYCL_PROFILE.source_suffixes contains `.cpp`; unlike CUDA (`.cu`) and HIP (`.hip`), SYCL is NOT the exclusive owner of its suffix — Kokkos and OMP-offload also claim `.cpp`. This is the configuration that forces detect_language() to consult content probes for `.cpp` inputs."""
    assert ".cpp" in SYCL_PROFILE.source_suffixes
    cpp_claimants = [
        name for name, p in PROFILES.items() if ".cpp" in p.source_suffixes
    ]
    # Order matters: Kokkos must precede SYCL must precede OMP-offload so probes
    # are consulted in insertion order for ambiguous `.cpp` inputs.
    assert cpp_claimants == ["kokkos", "sycl", "omp_offload"], (
        f"Unexpected .cpp claimants or order: {cpp_claimants}"
    )


def test_sycl_profile_driver_filename_is_dot_cpp():
    """SYCL_PROFILE.driver_filename is `driver.cpp` so the compile and splice tools target the right extension; this is the same value Kokkos uses, which is correct — the splice and compile machinery is keyed off `kernel_stem` and per-profile `driver_filename`, not off uniqueness across profiles."""
    assert SYCL_PROFILE.driver_filename == "driver.cpp"


def test_sycl_profile_dynamic_verification_is_true():
    """SYCL_PROFILE enables the dynamic-verification chain — same finish-gate behavior as Kokkos / CUDA / HIP. The chain is language-agnostic; the profile only chooses the compile command and harness prompt."""
    assert SYCL_PROFILE.dynamic_verification is True


def test_sycl_profile_id_and_display_name():
    """SYCL_PROFILE.id is the lowercase `sycl` token used in PROFILES keys and the orchestrator's `baseline_harness_<id>` tool-name suffix; display_name is the human-readable `SYCL C++` shown in logs."""
    assert SYCL_PROFILE.id == "sycl"
    assert SYCL_PROFILE.display_name == "SYCL C++"


def test_sycl_profile_env_required_is_empty():
    """SYCL_PROFILE.env_required is empty — `AGENT_PRECISION_SYCL_CXX` is optional (defaults to `icpx`), and SYCL has no required-at-call-time env var equivalent to Kokkos's `AGENT_PRECISION_KOKKOS_ROOT`. Compiler presence is checked in preflight, not declared as required env."""
    assert SYCL_PROFILE.env_required == ()


# ---------- get_profile_by_id ----------


def test_get_profile_by_id_returns_registered_profile():
    """get_profile_by_id resolves each registered id to the corresponding LanguageProfile object."""
    assert get_profile_by_id("kokkos") is KOKKOS_PROFILE
    assert get_profile_by_id("cuda") is CUDA_PROFILE
    assert get_profile_by_id("hip") is HIP_PROFILE
    assert get_profile_by_id("sycl") is SYCL_PROFILE
    assert get_profile_by_id("omp_offload") is OMP_OFFLOAD_PROFILE


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
    assert "sycl" in msg
    assert "omp_offload" in msg


# ---------- OMP-offload profile ----------


def test_profiles_contains_omp_offload():
    """The PROFILES registry exposes the Phase C-3 OMP-offload profile under its canonical id `omp_offload`, keyed by its own `.id` field — same invariant the kokkos / cuda / hip / sycl checks apply."""
    assert "omp_offload" in PROFILES
    assert PROFILES["omp_offload"] is OMP_OFFLOAD_PROFILE


def test_profiles_insertion_order_puts_sycl_before_omp_offload():
    """In PROFILES, SYCL_PROFILE appears before OMP_OFFLOAD_PROFILE; this insertion order is the tie-break order in detect_language()'s content-probe pass for `.cpp` inputs that match multiple probes. Kokkos -> SYCL -> OMP-offload is the documented order; later additions must extend this chain rather than reorder it."""
    ids = list(PROFILES.keys())
    assert ids.index("sycl") < ids.index("omp_offload"), (
        f"PROFILES insertion order broken: sycl must precede omp_offload, got {ids}"
    )


def test_omp_offload_profile_claims_dot_cpp():
    """OMP_OFFLOAD_PROFILE.source_suffixes contains `.cpp`; it is the second non-exclusive `.cpp` claimant after SYCL. This forces detect_language() to consult content probes for `.cpp` inputs across THREE candidates."""
    assert ".cpp" in OMP_OFFLOAD_PROFILE.source_suffixes
    cpp_claimants = [
        name for name, p in PROFILES.items() if ".cpp" in p.source_suffixes
    ]
    assert cpp_claimants == ["kokkos", "sycl", "omp_offload"], (
        f"Unexpected .cpp claimants or order: {cpp_claimants}"
    )


def test_omp_offload_profile_driver_filename_is_dot_cpp():
    """OMP_OFFLOAD_PROFILE.driver_filename is `driver.cpp` — same as Kokkos and SYCL. The splice and compile machinery is keyed off `kernel_stem` and per-profile `driver_filename`, not off uniqueness across profiles."""
    assert OMP_OFFLOAD_PROFILE.driver_filename == "driver.cpp"


def test_omp_offload_profile_dynamic_verification_is_true():
    """OMP_OFFLOAD_PROFILE enables the dynamic-verification chain — same finish-gate behavior as Kokkos / CUDA / HIP / SYCL. The chain is language-agnostic; the profile only chooses the compile command and harness prompt."""
    assert OMP_OFFLOAD_PROFILE.dynamic_verification is True


def test_omp_offload_profile_id_and_display_name():
    """OMP_OFFLOAD_PROFILE.id is the lowercase `omp_offload` token used in PROFILES keys and the orchestrator's `baseline_harness_<id>` tool-name suffix; display_name is the human-readable `OpenMP target-offload C++` shown in logs."""
    assert OMP_OFFLOAD_PROFILE.id == "omp_offload"
    assert OMP_OFFLOAD_PROFILE.display_name == "OpenMP target-offload C++"


def test_omp_offload_profile_env_required_is_empty():
    """OMP_OFFLOAD_PROFILE.env_required is empty — `AGENT_PRECISION_OMP_CXX` and `AGENT_PRECISION_OMP_TARGET` are both optional (defaults `clang++` and `nvptx64-nvidia-cuda`), and OMP-offload has no required-at-call-time env equivalent to Kokkos's `AGENT_PRECISION_KOKKOS_ROOT`. Compiler presence is checked in preflight."""
    assert OMP_OFFLOAD_PROFILE.env_required == ()


def test_detect_language_cpp_with_omp_offload_marker_returns_omp_offload():
    """A `.cpp` source carrying the canonical OMP-offload marker (`#pragma omp target`) routes to OMP_OFFLOAD_PROFILE. The Kokkos and SYCL probes return False for these inputs (no `Kokkos::`, no `sycl::`), so OMP-offload's probe is the first to return True and wins the content-probe pass."""
    assert (
        detect_language("k.cpp", "void f() { #pragma omp target\n}\n")
        is OMP_OFFLOAD_PROFILE
    )
    # Variants that should all match: the probe is a prefix substring check.
    assert (
        detect_language(
            "k.cpp", "void f() { #pragma omp target teams num_teams(1) {} }\n"
        )
        is OMP_OFFLOAD_PROFILE
    )
    assert (
        detect_language("k.cpp", "void f() { #pragma omp target data map(to:a) }\n")
        is OMP_OFFLOAD_PROFILE
    )


def test_detect_language_cpp_with_host_only_omp_does_not_return_omp_offload():
    """A `.cpp` source using `#pragma omp parallel` / `#pragma omp for` but NOT `#pragma omp target` is host-only OpenMP — it does NOT need the offload toolchain. The probe is strict about requiring the `target` token, so host-only OMP falls through to the insertion-order tie-break (Kokkos default)."""
    host_only = (
        "void f(int N) {\n"
        "    #pragma omp parallel for\n"
        "    for (int i = 0; i < N; ++i) {}\n"
        "}\n"
    )
    # No Kokkos / SYCL / OMP-target markers — falls back to Kokkos via insertion-order.
    assert detect_language("k.cpp", host_only) is KOKKOS_PROFILE


def test_detect_language_cpp_with_kokkos_and_omp_offload_markers_returns_kokkos():
    """A pathological `.cpp` source that triggers BOTH the Kokkos and OMP-offload probes routes to KOKKOS_PROFILE — Kokkos precedes OMP-offload in PROFILES, so its probe is consulted first and the first True wins. Same insertion-order tie-break that resolves Kokkos vs SYCL collisions."""
    mixed = (
        "#include <Kokkos_Core.hpp>\n"
        "void f() {\n"
        "    #pragma omp target\n"
        "    Kokkos::parallel_for(...);\n"
        "}\n"
    )
    assert detect_language("k.cpp", mixed) is KOKKOS_PROFILE


def test_detect_language_cpp_with_sycl_and_omp_offload_markers_returns_sycl():
    """A pathological `.cpp` source that triggers BOTH the SYCL and OMP-offload probes (but NOT Kokkos) routes to SYCL_PROFILE — SYCL precedes OMP-offload in PROFILES. Documents the full 3-way tie-break chain Kokkos -> SYCL -> OMP-offload."""
    mixed = (
        "#include <sycl/sycl.hpp>\n"
        "void f() {\n"
        "    #pragma omp target\n"
        "    sycl::queue q;\n"
        "}\n"
    )
    assert detect_language("k.cpp", mixed) is SYCL_PROFILE
