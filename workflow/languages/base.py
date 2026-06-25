"""LanguageProfile dataclass shared by every kernel-language profile.

A profile is the single source of truth for everything that varies by
kernel language: the driver file's extension, the compile command, the
splice-sentinel byte strings, the env vars that must be set, and the
baseline-harness agent's system prompt. The deterministic tools in
workflow.tools (compile_baseline_driver, splice_rewritten_kernel, ...)
consult these fields via the profile passed to them; the orchestrator
loop never branches on the language itself.

Default sentinels are C-style `//` comments because every kernel
language in the v1 scope (Kokkos C++, CUDA, SYCL, HIP, OpenMP target
offload) uses `//` line comments. A future Fortran or Python profile
would override `sentinel_begin` / `sentinel_end` accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# C-style sentinel defaults. Every v1 profile inherits these unchanged.
# A future Fortran profile would override with `! ---- ...`; a Python
# profile with `# ---- ...`. Keeping them on the dataclass (not as
# module-level constants the tools layer reads directly) is what makes
# that override mechanical.
DEFAULT_SENTINEL_BEGIN = "// ---- KERNEL BEGIN ----"
DEFAULT_SENTINEL_END = "// ---- KERNEL END ----"


def make_error_result(stderr: str) -> dict:
    """Build the uniform `_error()`-shaped dict every preflight returns.

    Profiles' `preflight()` returns either None (toolchain present, env
    fine) or this dict (so workflow.tools._compile_driver can hand it
    straight back to the orchestrator without rewrapping). The shape
    matches the rest of workflow.tools' result dicts so the orchestrator
    sees one schema regardless of which tool failed and why.
    """
    return {
        "status": "error",
        "stdout": "",
        "stderr": stderr,
        "artifacts": [],
    }


@dataclass(frozen=True)
class LanguageProfile:
    """Per-language settings for the dynamic-verification chain.

    Field guide:

      id                  Short slug ("kokkos", "cuda", "sycl", ...).
                          Used as the `language_id` value the orchestrator
                          threads through every tool call, and as the
                          suffix on the per-language baseline-harness agent
                          name in registry.AGENTS
                          (`baseline_harness_<id>`).

      display_name        Human-readable label for error messages and
                          the BASELINE STEP block in the initial user
                          message ("Kokkos C++", "CUDA C++", ...).

      source_suffixes     File extensions this profile claims, lowercased,
                          including the leading dot ((".cpp",), (".cu",),
                          (".cpp", ".hip"), ...). Used by
                          `detect_language()` for the first dispatch
                          pass.

      driver_filename     Name of the driver source file written by
                          spawn_baseline_harness and read by
                          splice_rewritten_kernel
                          ("driver.cpp" for Kokkos/SYCL/HIP-cpp/OMP-off,
                          "driver.cu" for CUDA, "driver.hip" for HIP-hip,
                          ...). The compiled binary path is always
                          `<driver_dir>/driver` regardless.

      sentinel_begin
      sentinel_end        Byte-exact strings the harness must emit and
                          the splice tool string-matches against, each
                          on its own line with no surrounding whitespace.
                          Default to the C-style sentinels every v1
                          language can carry.

      env_required        Tuple of env var names this profile reads. Surfaced
                          to the orchestrator via the BASELINE STEP block so
                          a missing var is diagnosed up front rather than
                          mid-chain. Profiles that read env vars only for
                          optional overrides (e.g. CUDA's
                          AGENT_PRECISION_CUDA_ARCH) may leave this empty.

      dynamic_verification
                          When False, `_FinishGateState` does NOT require
                          compare_outputs to pass before finish, and
                          `_format_baseline_block` emits a "skipped"
                          message instead of the full chain instructions.
                          True for all v1 profiles; the field exists as
                          an escape hatch for future
                          static-verification-only languages.

      baseline_harness_system_prompt
      baseline_harness_output_schema
                          Per-language system prompt and JSON Schema for
                          the harness agent. registry.AGENTS gets one
                          entry per profile keyed `baseline_harness_<id>`
                          using these two fields.

      build_compile_command(driver_src, driver_bin) -> list[str]
                          Callable that returns the argv list for the
                          compile subprocess. Receives both paths so a
                          profile can position them however its compiler
                          expects. Implemented as a callable rather than
                          a templated string so a profile can read env
                          vars at compile time (CUDA reads
                          AGENT_PRECISION_CUDA_ARCH; HIP reads
                          AGENT_PRECISION_HIP_CXX; ...) instead of at
                          import time.

      preflight() -> dict | None
                          Returns None when the env/toolchain is ready
                          and the compile may proceed. Returns a
                          `make_error_result(...)`-shaped dict when not
                          (e.g. KOKKOS_ROOT unset, nvcc not on PATH).
                          workflow.tools._compile_driver short-circuits
                          on the dict.

      detect_from_source(kernel_source) -> bool
                          Used only when the source suffix is claimed by
                          more than one profile. Should look at
                          structural markers (include lines, namespace
                          usage) rather than bare substrings, to avoid
                          comment false positives. Return True iff the
                          file is unambiguously this language.

      probe_precisions    Tuple of precision tokens this profile's probe
                          pipeline (v1) will run before invoking the
                          analyst. Each token names a storage precision
                          for the kernel's parameters / intermediates
                          (`"quad"`, `"double"`, `"float"`, and the
                          special `"mixed_io"` which keeps outputs at
                          `baseline_precision` and downcasts
                          intermediates to float). An empty tuple (the
                          default for v1) means "no probe for this
                          profile" — the orchestrator skips the probe
                          step entirely. Only Kokkos populates this in
                          v1; CUDA/HIP/SYCL/OMP-offload remain
                          probe-less until the deferred Commit 6 lands.

      baseline_precision  Precision token for the baseline driver
                          itself, i.e. the one the
                          baseline_harness agent emits and that
                          compare_outputs measures everything else
                          against. `"double"` for v1 profiles other
                          than Kokkos (which uses `"quad"` to give the
                          probe a true ground truth, since float-vs-
                          double drift is invisible when the baseline
                          is itself double). Threaded into the
                          BASELINE PRECISION line of the harness's
                          user message so the harness writes a driver
                          of the requested precision; profiles that
                          want a precision other than `"double"` must
                          carry prompt text that honors the directive.
    """

    id: str
    display_name: str
    source_suffixes: tuple[str, ...]
    driver_filename: str
    env_required: tuple[str, ...]
    dynamic_verification: bool
    baseline_harness_system_prompt: str
    baseline_harness_output_schema: dict
    build_compile_command: Callable[[Path, Path], list[str]]
    preflight: Callable[[], dict | None]
    detect_from_source: Callable[[str], bool]
    sentinel_begin: str = DEFAULT_SENTINEL_BEGIN
    sentinel_end: str = DEFAULT_SENTINEL_END
    probe_precisions: tuple[str, ...] = ()
    baseline_precision: str = "double"


__all__ = [
    "LanguageProfile",
    "DEFAULT_SENTINEL_BEGIN",
    "DEFAULT_SENTINEL_END",
    "make_error_result",
]
