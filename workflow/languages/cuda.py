"""CUDA C++ language profile.

CUDA is the second language profile to land (Kokkos was the first; this
module mirrors workflow.languages.kokkos field-for-field). The profile
owns the nvcc compile command, the `nvcc on PATH` preflight, the
detect-by-source probe used to disambiguate a `.cpp` file (no — CUDA
owns `.cu` outright, so the probe is a defensive helper that just looks
for canonical CUDA tokens), and the per-language baseline-harness system
prompt that teaches the harness agent how to write a self-contained CUDA
driver around an arbitrary kernel.

Two environment variables influence this profile at compile time:

  AGENT_PRECISION_CUDA_ARCH  Optional. The `-arch=` value passed to
                             nvcc. Defaults to `sm_89` (Ada Lovelace),
                             which is what the development workstation
                             carries; override on hosts with older or
                             newer GPUs. The variable is read at compile
                             time, not at import time, so a test that
                             monkeypatches the env affects every
                             subsequent compile in the same process.

The preflight also reports cleanly when `nvcc` is not on PATH so the
operator gets a one-line diagnostic instead of a confusing
FileNotFoundError from subprocess.run.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .base import LanguageProfile, make_error_result


# Environment variable that, when set, overrides the default `-arch=`
# value passed to nvcc. Intentionally namespaced under the project's
# AGENT_PRECISION_* prefix to avoid colliding with nvcc's own
# environment knobs (NVCC_PREPEND_FLAGS etc.).
ARCH_ENV = "AGENT_PRECISION_CUDA_ARCH"

# Default compute capability the project targets when ARCH_ENV is
# unset. sm_89 is Ada Lovelace (RTX 40-series, L4, L40, ...); it covers
# the development workstation. Overriding via ARCH_ENV is cheap; the
# explicit default is preferred over `-arch=native` because a clear
# unsupported-arch error at compile time is easier to triage than a
# runtime "no kernel image is available for execution" on a different
# host.
DEFAULT_ARCH = "sm_89"

# The compiler binary name. nvcc's own driver is what the project
# requires; alternatives like `clang -x cuda` exist but are not in scope
# for v1 — the harness prompt assumes the nvcc-specific kernel-launch
# syntax `kernel<<<...>>>(...)` and the runtime-API headers
# (`<cuda_runtime.h>`).
NVCC = "nvcc"
CXX_STD = "-std=c++17"
OPT_FLAGS = ("-O2",)

# Phase 1b: detection token for the CUDA quad-precision probe path.
# nvcc has no `__float128` support on either the host or device side
# (verified empirically), so a quad oracle for CUDA cannot be a real
# GPU driver. The baseline_harness therefore emits the quad driver as
# plain C++ (no `<cuda_runtime.h>`, no `__global__`, no `<<<...>>>`
# launch syntax) that reproduces the kernel on the host in
# `__float128` via <quadmath.h>. We detect that shape by scanning the
# driver source for `__float128` — a GCC-specific extension keyword
# that is exceedingly unlikely to appear in a real CUDA driver
# (`nvcc` would reject it), so a false positive is essentially
# impossible in practice. When the token is found,
# `_build_compile_command` switches from `nvcc` to plain `g++` (host-
# only) and appends `-lquadmath` — the same libquadmath link the
# Kokkos profile uses for its own quad driver. The two profiles
# deliberately share the token / lib names so the mechanism reads
# uniformly across the codebase.
QUAD_PROBE_TOKEN = "__float128"
QUAD_LIB = "-lquadmath"

# g++ command used for the CUDA quad driver (host-only compile).
# Deliberately mirrors the Kokkos g++ line (same std, same -O2) so
# both profiles' quad drivers behave the same numerically — the
# quadmath transcendentals are what actually drive the accuracy, not
# any -O flag difference. `-fopenmp` is deliberately omitted here
# (the CUDA quad driver is a serial host loop; nothing needs OpenMP)
# where the Kokkos quad line includes it because the surrounding
# Kokkos-shaped driver expects it.
QUAD_HOST_CXX = "g++"


def _build_compile_command(driver_src: Path, driver_bin: Path) -> list[str]:
    """Assemble the compile argv list for a CUDA driver compile.

    Two paths, selected by sniffing the driver source:

      * If the source contains `__float128` (`QUAD_PROBE_TOKEN`), this
        is the Phase 1b host-side quad oracle: compile with plain g++
        and append `-lquadmath`. No nvcc, no `-arch`, no GPU required
        at build or run time. Symmetric with the Kokkos quad path,
        which also switches to a quadmath link when it sees the token.

      * Otherwise, this is a real CUDA driver: compile with nvcc,
        honoring AGENT_PRECISION_CUDA_ARCH (defaulting to sm_89).
        The arch env var is read at call time (not import time) so a
        test that monkeypatches the env affects every subsequent
        compile in the same process.

    The sniff is a substring match on the source file's text (same
    idiom as `workflow.languages.kokkos._build_compile_command`).
    `driver_src` is guaranteed to exist by this point (`_compile_driver`
    checks it above us); a read failure here would surface as a
    FileNotFoundError, same as it would from the subprocess attempt
    that follows.

    Assumes preflight has already verified nvcc is on PATH. The quad
    path does NOT require nvcc — a host without nvcc can still compile
    the quad driver — but the preflight fails uniformly for all
    precisions, so a nvcc-less host cannot reach this compile at all
    today. If we later want to permit "quad-only on a nvcc-less host",
    the preflight would need to become precision-aware; deferred until
    a real use case demands it.
    """
    if QUAD_PROBE_TOKEN in driver_src.read_text():
        # `-x c++` is required because the driver filename is `driver.cu`
        # (a profile-wide constant used by the splice/compare/measure
        # chain), and g++ refuses to compile a `.cu` suffix as C++ — it
        # treats the source as a linker script and fails at link time
        # (verified empirically: `g++ -O2 driver.cu -lquadmath -o driver`
        # reports "file format not recognized; treating as linker
        # script"). `-x c++` explicitly overrides the suffix-based
        # language detection and must appear before the source file for
        # g++'s left-to-right argument scan. Renaming to `driver.cpp`
        # for the quad cell was rejected as an alternative because it
        # would leak precision-awareness into `LanguageProfile
        # .driver_filename` (currently one filename per language), and
        # `-x c++` is a strict local fix.
        return [
            QUAD_HOST_CXX,
            CXX_STD,
            *OPT_FLAGS,
            "-x",
            "c++",
            str(driver_src),
            QUAD_LIB,
            "-o",
            str(driver_bin),
        ]
    arch = os.environ.get(ARCH_ENV, DEFAULT_ARCH)
    return [
        NVCC,
        CXX_STD,
        *OPT_FLAGS,
        f"-arch={arch}",
        str(driver_src),
        "-o",
        str(driver_bin),
    ]


def _build_syntax_check_command(driver_src: Path) -> list[str] | None:
    """Assemble the argv list for a pre-write CUDA driver check.

    Two paths, selected by the same `__float128` sniff as
    `_build_compile_command`:

      * If the source contains `__float128` (Phase 1b host-side quad
        oracle), return a `g++ -fsyntax-only` line. g++ has a true
        parse-only mode, so this is a clean strict subset of the real
        quad compile: same `-std=c++17`, no `-lquadmath` (link is not
        exercised in a syntax check; the `__float128` type is a GCC
        built-in with no header dependency, same rationale as the
        Kokkos quad path). Returns None if g++ is not on PATH.

      * Otherwise, return the nvcc surrogate. Returns None when nvcc
        is not on PATH — the harness-validation gate then skips
        silently rather than failing every run on a host without a
        CUDA toolchain (validation is a quality improvement, not a
        hard requirement, exactly as for the Kokkos gate).

        nvcc has no true parse-only mode: it rejects `-fsyntax-only`
        (verified empirically — `nvcc fatal: Unknown option
        '-fsyntax-only'`). The closest it offers is `-c -o /dev/null`,
        which compiles both the host and device sides to an object but
        never links (no CUDA runtime libraries, no GPU required beyond
        nvcc itself) and never writes an artifact. That is a strict
        subset of the real compile flags (`_build_compile_command`):
        same `-std`, `-O2`, and `-arch`, but the final `-o <bin>` is
        replaced with `-c -o /dev/null`. It catches the malformed-
        driver class the gate exists for (e.g. the inconsistent alias-
        naming that motivated the Kokkos gate) at a fraction of a full
        compile+link, before any file is written to disk.

    The `-arch` value is read at call time (same contract as
    `_build_compile_command`) so a monkeypatched env is honored.
    """
    if QUAD_PROBE_TOKEN in driver_src.read_text():
        if shutil.which(QUAD_HOST_CXX) is None:
            return None
        # `-x c++` is required for the same reason as in the compile
        # command: g++ would otherwise refuse the `.cu` suffix. See
        # `_build_compile_command` for the full explanation.
        return [
            QUAD_HOST_CXX,
            CXX_STD,
            "-fsyntax-only",
            "-x",
            "c++",
            str(driver_src),
        ]
    if shutil.which(NVCC) is None:
        return None
    arch = os.environ.get(ARCH_ENV, DEFAULT_ARCH)
    return [
        NVCC,
        CXX_STD,
        *OPT_FLAGS,
        f"-arch={arch}",
        "-c",
        str(driver_src),
        "-o",
        os.devnull,
    ]


def _preflight() -> dict | None:
    """Verify nvcc is reachable before invoking the compiler.

    Returns None when `shutil.which("nvcc")` finds the compiler.
    Otherwise returns a make_error_result()-shaped dict the caller hands
    straight back to the orchestrator — no subprocess is spawned.

    AGENT_PRECISION_CUDA_ARCH is intentionally NOT validated here; if
    the operator typoes the arch, nvcc itself will return a clear
    `Unknown option '-arch=sm_typo'` error during the compile, which is
    a cleaner signal than a Python-side allowlist of capabilities that
    would need to track every new GPU release.
    """
    if shutil.which(NVCC) is None:
        return make_error_result(
            f"{NVCC} not found on PATH. Install the CUDA toolkit and "
            f"ensure {NVCC} is reachable, or set AGENT_PRECISION_CUDA_ARCH "
            f"on a host that has a CUDA toolchain."
        )
    return None


def _detect_from_source(kernel_source: str) -> bool:
    """Probe a source file to decide whether it's CUDA.

    CUDA claims `.cu` exclusively (see source_suffixes below), so
    detect_language() never actually reaches this probe under v1 — the
    suffix alone disambiguates. The helper is implemented for symmetry
    with the other profiles and as a safety net in case a future profile
    decides to also claim `.cu` (which would force a content-based
    tiebreak). The probe looks for the canonical CUDA runtime header or
    a `__global__` / `__device__` qualifier, both of which are
    structural markers that won't be triggered by an arbitrary comment.
    """
    if "<cuda_runtime.h>" in kernel_source:
        return True
    if "__global__" in kernel_source or "__device__" in kernel_source:
        return True
    return False


# Per-language baseline-harness contract for CUDA. The output schema is
# structurally identical to the Kokkos schema (a self-contained driver
# source, the kernel function name it calls, an inputs summary string,
# and the list of output array names) — the comparator and splice tools
# downstream don't care which language produced the driver, so the
# JSON shape is shared. Only the descriptions are tweaked to say "CUDA"
# where Kokkos says "Kokkos".
BASELINE_HARNESS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "drivers": {
            "type": "object",
            "description": (
                "Four self-contained .cu driver translation units, one per "
                "probe precision. Each MUST inline the kernel source "
                "verbatim between the KERNEL BEGIN/END sentinels and on "
                "execution write reference outputs to ./reference.json. "
                "The `quad` driver is a plain-C++ host port that compiles "
                "with g++ + libquadmath (nvcc has no __float128); the "
                "other three compile with nvcc. All four share the same "
                "kernel body, the same per-parameter <ParamName>Type "
                "alias NAMES, the same RNG_SEED, and the same output-"
                "array names and lengths; they differ in the alias RHSes "
                "(which set per-parameter storage precision), any host-"
                "side scratch values that flow into the kernel, the JSON "
                "floating-point formatting, and (for `quad` only) the "
                "compile model itself. See the PER-PRECISION DRIVERS "
                "block in the system prompt for the full contract."
            ),
            "properties": {
                "quad": {
                    "type": "string",
                    "description": (
                        "Host-only quad-precision oracle. Does NOT use "
                        "nvcc or the CUDA runtime: emit plain C++ that "
                        "unrolls the kernel launch into a serial host "
                        "`for` loop, uses `__float128` throughout via "
                        "<quadmath.h>, and writes reference.json values "
                        "via `quadmath_snprintf(buf, sizeof(buf), "
                        "\"%.34Qg\", value)`. The compile step auto-"
                        "detects the driver as a quad driver by scanning "
                        "for the `__float128` token and switches from "
                        "nvcc to `g++ -lquadmath`. Feeds the probe-"
                        "evidence pipeline as the ground-truth oracle "
                        "against which `double` and `float` are measured, "
                        "and is promoted to `baselines/<stem>/reference"
                        ".json` after `probe_compare` succeeds so the "
                        "finish-gate comparator measures the rewritten "
                        "kernel against true quad ground truth. See the "
                        "`quad` bullet in PER-PRECISION DRIVERS below for "
                        "the intrinsic-refusal clause and the required "
                        "transcendental substitutions."
                    ),
                },
                "double": {
                    "type": "string",
                    "description": (
                        "Uniform-double driver: every FP alias resolves to "
                        "`double` / `double*`; JSON values `%.17g`. Feeds "
                        "the probe-evidence pipeline as a same-precision "
                        "point of comparison against `float` and "
                        "`original`. NOT the splice scaffold."
                    ),
                },
                "float": {
                    "type": "string",
                    "description": (
                        "Uniform-float driver: every FP alias resolves to "
                        "`float` / `float*`; JSON values `%.9g`. Feeds the "
                        "probe-evidence pipeline as the "
                        "aggressive-downcast point of comparison."
                    ),
                },
                "original": {
                    "type": "string",
                    "description": (
                        "Preserves the user's exact per-parameter kernel "
                        "types verbatim (no coercion to a uniform "
                        "precision). Serves two roles: (1) canonical "
                        "splice scaffold copied to baselines/<stem>/"
                        "driver.cu, and (2) sole source of the pre-"
                        "rewrite baseline `timing` block read by "
                        "measure_speedup. For all-double kernels this is "
                        "byte-identical to the `double` driver; emit it "
                        "anyway (both roles need the file at its own "
                        "known path)."
                    ),
                },
            },
            "required": ["quad", "double", "float", "original"],
        },
        "kernel_function_name": {
            "type": "string",
            "description": (
                "Name of the kernel function (`__global__`) the driver "
                "launches. Must match a function defined in the inlined "
                "kernel source."
            ),
        },
        "inputs_summary": {
            "type": "string",
            "description": (
                "One-line human-readable summary of the chosen inputs, "
                "e.g. 'N=16384, seed=42, x,y ~ U(-1,1)'. Mirrors the "
                "'inputs' block the driver writes into reference.json."
            ),
        },
        "output_arrays": {
            "type": "array",
            "description": (
                "Names of the arrays the driver writes under the 'outputs' "
                "key of reference.json. A future mechanical comparator "
                "uses this list to know which arrays to read back."
            ),
            "items": {"type": "string"},
        },
    },
    "required": [
        "drivers",
        "kernel_function_name",
        "inputs_summary",
        "output_arrays",
    ],
}


BASELINE_HARNESS_SYSTEM_PROMPT = """You are the baseline-harness agent
for a mixed-precision rewriting workflow.

You will be given a CUDA C++ kernel source. Your job is to write four
self-contained driver programs (three real CUDA drivers plus one host-
only quad-precision oracle) that, when compiled and run later, exercise
the kernel on a fixed set of inputs and write reproducible reference
outputs to ./reference.json. Those JSON files will eventually be the
baselines against which a rewritten (lower-precision) version of the
same kernel is compared.

You do NOT compile, run, or simulate any of the drivers. You do NOT
invent numerical output values. Your only output is the four driver
sources.

PER-PRECISION DRIVERS. You emit FOUR drivers in a single submit_result
call under `drivers.{quad,double,float,original}`. All four MUST share:

  - the same inlined kernel source (byte-identical between the
    `// ---- KERNEL BEGIN ----` and `// ---- KERNEL END ----` sentinels
    — EXCEPT the `quad` driver, which is a plain-C++ host port with
    its own per-element math substitutions; see the `quad` bullet
    below),
  - the same per-parameter `<ParamName>Type` alias NAMES (item 6),
  - the same `static constexpr int RNG_SEED = ...;` line (item 3),
  - the same input sizes, the same output array names, and the same
    output array lengths (the comparator requires shape-identical
    reference.json across precisions).

The four drivers differ ONLY in:

  - the RHS of each per-parameter `<ParamName>Type` alias,
  - any host-side scratch values that ultimately flow into the kernel
    (RNG fill loops, staging-buffer element types, etc.) — these must
    be the appropriate precision end-to-end, not down-converted
    through `double` mid-driver,
  - the reference.json floating-point formatting (item 8),
  - for `quad` only: the compile model itself (plain C++ + libquadmath,
    no nvcc, no CUDA runtime) and the per-element math substitutions
    (see the `quad` bullet below).

Per-precision rules:

  - `double`: aliases resolve to `double` / `double*` (pointer aliases
    replace their pointee with `double`, keeping any const-qualifiers);
    JSON values written with `"%.17g"`. This driver exists purely to
    feed the probe-evidence pipeline as a uniform-double point of
    comparison against `quad`, `float`, and `original`; it is NOT the
    splice scaffold (that role belongs to `original`).
  - `float`: aliases resolve to `float` / `float*` (same
    const-qualifier rule); JSON values written with `"%.9g"`. Host-
    side RNG fill and any staging buffers that flow into the kernel
    must be `float` end-to-end — do NOT fill a `std::vector<double>`
    and then `cudaMemcpy` a `float`-typed device buffer from it.
  - `quad`: SPECIAL CASE — this driver does NOT use CUDA at all. nvcc
    has no `__float128` support on either the host or device side
    (verified empirically), so a real CUDA quad driver is
    fundamentally uncompilable. Instead, emit the quad driver as
    plain C++ that reproduces the kernel on the host in quad
    precision — the compile step sniffs the driver source for
    `__float128` and automatically switches from nvcc to `g++
    -std=c++17 -O2 <src> -lquadmath -o <bin>`. Concretely, this
    driver:
      * does NOT `#include <cuda_runtime.h>`, does NOT call
        `cudaMalloc` / `cudaMemcpy` / `cudaDeviceSynchronize`, does
        NOT use `__global__` / `__device__` qualifiers, does NOT
        launch anything via `kernel<<<...>>>`. There is no CUDA
        runtime linked into this driver; using any CUDA API is a
        link error at best and undefined behavior at worst;
      * `#include <quadmath.h>` and uses `__float128` throughout;
        replace each CUDA math intrinsic with its `q`-suffixed
        quadmath equivalent — `sqrtf`/`sqrt` -> `sqrtq`,
        `sinf`/`sin` -> `sinq`, similarly `cosq`, `expq`, `logq`,
        `fabsq`, `powq`, `atan2q`, `tanhq`, etc. For fused-multiply-
        add patterns (`a*b + c`), use `fmaq(a, b, c)` — this preserves
        an extra bit of precision the naive `a*b + c` loses and
        matches what a real quad oracle should compute;
      * for reductions (sums of many elements), use Kahan or pairwise
        summation rather than a naive accumulator. Even in quad
        precision, a naive left-to-right sum of `N=1e6` terms can
        lose several ulp of accuracy relative to the true infinite-
        precision sum; the whole point of the quad oracle is to be
        accurate enough that its numbers are trustworthy ground
        truth for a `--sig-figs 6` comparison. A simple Kahan loop
        (`__float128 sum = 0, c = 0; for (...) { __float128 y =
        x[i] - c; __float128 t = sum + y; c = (t - sum) - y; sum =
        t; }`) is sufficient;
      * DOES NOT use the GNU `q` / `Q` numeric-literal suffix for
        `__float128` constants. C++23 disallows it as an extension
        and g++ rejects it under `-std=c++17` without
        `-fext-numeric-literals` (which the compile step does NOT
        pass). Write `__float128(0.0)`, `(__float128)1.5`, or
        `static_cast<__float128>(0.0)` instead of `0.0q` / `1.5q`.
        This includes ALL constants used inside the kernel body —
        accumulator initializers (`__float128 ax = __float128(0.0);`),
        scalar multipliers, comparison thresholds, anything that
        would otherwise be a bare floating literal;
      * replaces each `T*` / `const T*` device-pointer kernel
        argument with a plain contiguous host buffer
        (`std::vector<__float128>` or `std::unique_ptr<__float128[]>`).
        The aliases in this driver resolve to those host types (e.g.
        `using aType = std::vector<__float128>;`, `using bType =
        const std::vector<__float128>&;`, `using alphaType =
        __float128;`). The kernel body accesses them via `[]` at the
        same indices as the CUDA version's `blockIdx.x * blockDim.x
        + threadIdx.x` — see the loop rule below;
      * replaces the kernel's `<<<blocks, threads>>>(...)` launch and
        its per-thread `int i = blockIdx.x * blockDim.x +
        threadIdx.x; if (i < N) { ... }` guard with a plain serial
        `for (int i = 0; i < N; ++i) { ... }` containing the SAME
        kernel body text (with CUDA math intrinsics swapped for
        their quadmath equivalents). The kernel function itself
        stays inside the sentinels and keeps its `<ParamName>Type`
        alias-typed signature — only the alias RHSes and the per-
        element math change;
      * writes JSON values via `quadmath_snprintf(buf, sizeof(buf),
        "%.34Qg", value)` (NOT via `snprintf` / `<<`, which do not
        understand `__float128`). The compile step auto-links
        `-lquadmath` when the driver source contains the token
        `__float128`. Local `std::uniform_real_distribution<...>`
        returns `double` — convert explicitly with
        `static_cast<__float128>(...)` when storing into the alias-
        typed buffer.

    REFUSAL CLAUSE. If the kernel under test uses CUDA rounding-mode
    intrinsics (`__fadd_rd`, `__fadd_ru`, `__fmul_rd`, `__fmul_ru`,
    `__fma_rd`, `__fma_rn`, `__fma_ru`, `__fma_rz`, and their `_rn`
    / `_rz` variants, or the corresponding `__d*_r?` doubles) or
    other CUDA-specific rounding-mode / precision-control intrinsics
    that have no `__float128` analogue, DO NOT emit a quad driver.
    Instead, populate `drivers.quad` with a short C++ program that
    prints an explanatory error to stderr and `std::exit(2)` — the
    orchestrator's probe pipeline will treat the missing/failing
    quad cell as a hard error (there is no ground-truth oracle for
    this kernel) and stop before the analyst stage, which is the
    correct behavior. The other three drivers (double, float,
    original) MUST still be emitted normally. Rationale: silently
    stripping the rounding-mode intrinsic and computing a "close
    enough" quad value would produce an oracle that disagrees with
    the user's kernel by exactly the amount the intrinsic was
    controlling, contaminating every downstream verdict.

    The quad driver's job is purely to produce a ground-truth
    `reference.json` against which the (CUDA-based) rewritten driver
    is later compared by the comparator; it is NEVER a splice
    target, so it does not need to share a CUDA runtime with the
    other drivers.
  - `original`: aliases resolve to the EXACT floating-point types
    the user wrote in the kernel source verbatim, parameter by
    parameter. If the user's kernel declares
    `__global__ void kernel(double* a, const float* b, double alpha)`,
    the aliases here are
    `using aType = double*;`,
    `using bType = const float*;`,
    `using alphaType = double;` — no coercion to a uniform precision,
    even if the mix looks unusual. The JSON output format follows
    the DOMINANT precision of the user's kernel I/O parameters:
    `%.9g` if every floating-point I/O parameter is `float`, `%.17g`
    otherwise (i.e. as soon as any I/O parameter is `double`, use
    `%.17g` for every output value — mixed-precision JSON files
    remain readable by the same parser as the `double` driver). This
    driver serves two orthogonal roles: (1) it is the canonical
    splice scaffold — the orchestrator writes it to
    `baselines/<stem>/driver.cu` and the rewriter splices its
    kernel into a copy of it, so `main()` outside the sentinels must
    already match the user's original kernel signature; (2) it is
    the sole source of the baseline `timing` block that
    `measure_speedup` reports as the pre-rewrite wall-clock — the
    `double` driver's timing is numerically meaningless as "the
    user's baseline" because `double` is a uniform-precision rewrite
    of a potentially-mixed kernel. For all-double kernels this
    driver is byte-identical to the `double` driver; emit it anyway
    (both paths — probe evidence and speedup baseline — need it at
    its own known path).

SPLICE-TARGET ROLE. The orchestrator writes `drivers["original"]`
(NOT `drivers["double"]`, NOT `drivers["quad"]`) to
`baselines/<stem>/driver.cu` as the canonical splice scaffold. The
rewriter later splices its kernel between the sentinels of that
file to produce `baselines/<stem>/rewritten/driver.cu`. Because
`main()` in the `original` driver already constructs kernel
arguments through aliases that match the user's exact parameter
types, the rewriter can redefine those aliases (e.g. downcast an
`aType` from `double*` to `float*`) without touching `main()`; the
change propagates through the signature for free. The other three
drivers (quad, double, float) live only under
`baselines/<stem>/probe/<precision>/` and feed the probe-evidence
pipeline.

The quad driver additionally serves as the ground-truth oracle: its
`reference.json` from seed=42 is promoted to
`baselines/<stem>/reference.json` (overwriting whatever
`run_baseline_driver` wrote there from the original driver) so the
finish-gate comparator measures the rewritten kernel against true
quad ground truth, not against a same-or-lower-precision reference.

The `original` driver additionally serves as the pre-rewrite
wall-clock reference for `measure_speedup`: its `timing` block from
seed=42 is read directly out of
`baselines/<stem>/probe/original_seed42/reference.json` as the
"baseline" side of the speedup ratio. The `timing` block is present
in every driver (item 11) but the `double` and `quad` blocks are
NOT used for speedup — the `double` driver is a uniform-precision
rewrite of a potentially-mixed kernel, and the `quad` driver is
plain C++/quadmath (not CUDA), so neither represents what "the
user's kernel before rewriting" actually runs at.

None of these driver variants changes the kernel function body
(except the `quad` driver, which rewrites CUDA math intrinsics to
their quadmath equivalents and unrolls the `<<<...>>>` launch into
a serial host `for` loop — see the quad bullet above). They change
the alias RHSes (item 6 below), any host-side scratch values that
flow into the kernel, and the JSON output formatting (item 8).

Hard requirements on each driver:

1. Single translation unit. Inline the kernel source verbatim into the
   driver (above main()). Do not introduce a build system or external
   headers beyond <cuda_runtime.h>, the C and C++ standard library, and
   anything the kernel itself already includes. (Exception: the quad
   driver omits <cuda_runtime.h> entirely and adds <quadmath.h>; see
   the quad bullet in PER-PRECISION DRIVERS above.)

   The inlined kernel MUST be bracketed by these two sentinel comment
   lines, verbatim, each on its own line and with no surrounding
   indentation:

       // ---- KERNEL BEGIN ----
       <the kernel source, unmodified>
       // ---- KERNEL END ----

   These exact strings are a hard contract: a later mechanical-
   verification step will string-replace the text between them to splice
   a rewritten kernel into this same driver template (so the rewritten
   kernel runs against bit-identical inputs, RNG, and JSON output
   layout). Do not paraphrase the sentinels, do not add extra dashes,
   do not move them inside a function body, and do not place any other
   code between them other than the kernel itself.

2. Deterministic execution. The driver must produce bit-identical
   output across runs and across machines (modulo GPU floating-point
   determinism within a single compute capability). Concretely:

   a. Do NOT use `atomicAdd` (or any other atomic floating-point op)
      in the driver's host-side reference computation. Order of atomic
      updates is hardware-scheduling-dependent and would make the
      reference non-reproducible. If the kernel under test uses
      atomics, that is a kernel design choice and is judged by the
      rewriter/verifier separately; the DRIVER must not introduce its
      own.

   b. Launch the kernel with a single, fixed grid/block configuration
      derived deterministically from the chosen input size (e.g.
      `int threads = 256; int blocks = (N + threads - 1) / threads;`).
      Do not query the device for an "optimal" launch shape.

   c. Seed any host-side RNG from a top-of-file constant declared
      exactly as `static constexpr int RNG_SEED = 42;` (use 42 unless
      the kernel's apparent domain demands otherwise). All three
      drivers MUST use the identical `RNG_SEED = <N>;` line (same
      integer, same declaration) so a later probe step can flip the
      seed with a single-line regex swap. The driver must produce
      the same numbers on every run.

3. Use cudaMalloc / cudaMemcpy for device allocations and host<->device
   transfers. After every CUDA runtime call AND after the kernel
   launch, check the returned cudaError_t (or call cudaGetLastError() /
   cudaDeviceSynchronize() after the launch) and abort with a clear
   stderr message on failure. A typical pattern:

       #define CUDA_CHECK(expr) do {                                 \\
         cudaError_t _err = (expr);                                  \\
         if (_err != cudaSuccess) {                                  \\
           std::fprintf(stderr, "CUDA error at %s:%d: %s\\n",        \\
               __FILE__, __LINE__, cudaGetErrorString(_err));        \\
           std::exit(1);                                             \\
         }                                                           \\
       } while (0)

   This pattern is what lets a misconfigured runtime (missing driver
   library, wrong LD_LIBRARY_PATH, unsupported compute capability)
   surface as a clear diagnostic instead of a silent wrong-answer.

4. Choose modest input sizes and distributions appropriate to the
   kernel from its signature and apparent scientific domain. Aim for a
   driver that runs in a few seconds on a single GPU, not hours.
   Typical N is in the 1e4 to 1e7 range depending on per-element cost.
   Document the inputs you chose in inputs_summary.

5. If the task message names a TARGET KERNEL, launch exactly that
   `__global__` function. Otherwise, infer the kernel function from
   the source — there should be exactly one obvious candidate.

6. Do not modify the kernel function. Do not change any variable's
   precision. Do not invent or rename kernel arguments. The whole point
   is to capture the *original* kernel's output as the reference.

    EXCEPTION — precision-alias contract. Immediately inside the
   '// ---- KERNEL BEGIN ----' sentinel and above the kernel function
   definition, emit one `using` alias per kernel parameter whose type
   involves a floating-point scalar or a pointer-to-floating-point
   (CUDA kernels typically take `float*` / `double*` device pointers).
   Naming convention: `<ParamName>Type` (CamelCase of the parameter
   name + 'Type' suffix). The kernel function's parameter list MUST
   then refer to those aliases, not to the underlying types. The
   kernel body is otherwise unchanged. For example, a kernel
   originally declared as

       __global__ void kernel(
           double* a, const double* b,
           double alpha, double beta, int N) { ... }

   becomes, inside the sentinels:

       using aType = double*;
       using bType = const double*;
       using alphaType = double;
       using betaType = double;

       __global__ void kernel(aType a, bType b,
                              alphaType alpha, betaType beta,
                              int N) { ... }

   Integer parameters (sizes, counts, indices) do NOT get aliases —
   only floating-point scalars and floating-point pointers. The kernel
   body itself stays byte-for-byte identical to the original; only the
   parameter type tokens in the function header are replaced with the
   alias names.

   `main()`, outside the sentinels, MUST use those aliases anywhere it
   constructs values that flow into the kernel launch. Use the matching
   alias (e.g. `aType` for the `a` argument, `alphaType` for the
   `alpha` argument) to declare the device pointers and scalar values
   that flow into the kernel launch. The aliases are the single point
   of truth for kernel I/O precision: a later rewriter redefines the
   aliases inside the sentinels to change kernel precision end-to-end,
   and `main()` inherits the change for free.

   Staging-buffer rule for `const T*` kernel arguments. When a kernel
   parameter's alias is a const pointer (e.g.
   `using aType = const double*;`), allocate the device buffer through
   the non-const counterpart of the alias's pointee type so cudaMemcpy
   can write into it, then bind to the const-aliased pointer for the
   launch:

       // correct: derive the writable pointee type from the alias
       using aElem = std::remove_const_t<std::remove_pointer_t<aType>>;
       aElem* a_nc = nullptr;
       CUDA_CHECK(cudaMalloc(&a_nc, N * sizeof(aElem)));
       CUDA_CHECK(cudaMemcpy(a_nc, host_a.data(),
                             N * sizeof(aElem), cudaMemcpyHostToDevice));
       aType a = a_nc;  // const-binds; precision is whatever aType says
       kernel<<<blocks, threads>>>(a, ...);

   Do NOT hardcode the staging buffer's element type as `double`. A
   hardcoded `double* a_nc; cudaMalloc(&a_nc, N * sizeof(double));`
   breaks the contract: when the rewriter redefines `aType` to
   `const float*`, the `aType a = a_nc;` assignment no longer compiles
   because `double*` does not convert to `const float*`. The
   `std::remove_const_t<std::remove_pointer_t<...>>` form is the only
   one that survives a precision change to the alias.

   For local host scratch (std::vector buffers, RNG distributions) that
   do not directly become kernel arguments, plain `double` is fine —
   only the values that cross the kernel boundary need to flow through
   the aliases.

7. After the kernel launch, cudaMemcpy any device outputs you intend to
   record back to host buffers before iterating them for JSON emission.
   Call cudaDeviceSynchronize() before the copy-back so launch errors
   surface here, not later.

8. Write the reference output to './reference.json' (relative to the
   driver's working directory) using std::ofstream and the precision-
   specific format string mandated by the PER-PRECISION DRIVERS block
   above (`"%.17g"` for `double` and `original`-when-any-param-is-
   double, `"%.9g"` for `float` and `original`-when-all-params-are-
   float). Do NOT pull in a third-party JSON library — hand-roll the
   writer; output arrays are flat arrays of scalars, so a few loops
   with manual braces, commas, and newlines are sufficient.

9. The JSON document must have exactly this shape:

       {
         "kernel": "<kernel_function_name>",
         "seed": <integer seed>,
         "inputs": { "N": <int>, ... },
         "outputs": { "<name>": [ <double>, ... ], ... },
         "timing": {
           "trials_timed": 10,
           "mean_sec": <float>,
           "stddev_sec": <float>,
           "min_sec": <float>,
           "max_sec": <float>
         }
       }

   "inputs" carries enough metadata for a human reader to understand
   what the driver did (sizes, distributions if represented as
   strings). "outputs" carries one named flat array per output the
   comparator will check. The names under "outputs" must match
   output_arrays in your submit_result payload.

   "timing" carries wall-clock statistics for the kernel launch (see
   item 11 below). It is required in every reference.json so that the
   downstream `measure_speedup` tool can compute a mean/stddev
   speedup from the baseline vs rewritten references. The comparator
   ignores timing values numerically — only `outputs` is checked
   against the tolerance — but it does require both files to have
   identically-shaped top-level keys, so the `timing` block must
   always be present.

10. Begin each driver with a top-of-file comment that tells the
    operator to `cd` into the driver's own directory before running,
    so ./reference.json lands next to the driver source. The
    `original` driver lives at `baselines/<file_stem>/driver.cu` and
    the three probe drivers live at
    `baselines/<file_stem>/probe/{quad,double,float}/driver.cu`.
    Also mention the compile command in a comment (a typical nvcc
    build line is fine for the three CUDA drivers; the quad driver
    should mention `g++ -std=c++17 -O2 driver.cu -lquadmath -o
    driver` — the compile step auto-detects the switch from nvcc to
    g++ by scanning for the `__float128` token, so operators do not
    have to remember it, but the comment is a useful hint).

11. Kernel timing. Repeat the kernel launch N=11 times: 1 untimed
    warmup launch followed by 10 timed trials. Time only the kernel
    launch itself — NOT device allocation, host<->device transfers,
    or JSON emission. For the three CUDA drivers (double, float,
    original), use cudaEvent for GPU timing:
    `cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);`
    then per trial, `cudaEventRecord(start)`, launch the kernel,
    `cudaEventRecord(stop)`, `cudaEventSynchronize(stop)`,
    `float ms; cudaEventElapsedTime(&ms, start, stop);`. Convert to
    seconds (`ms / 1000.0`) and push into a `std::vector<double>`.
    For the quad driver (plain host C++), use `std::chrono` around
    the serial `for` loop instead:
    `auto t0 = std::chrono::steady_clock::now(); /* the for loop */;
    auto t1 = std::chrono::steady_clock::now(); double sec =
    std::chrono::duration<double>(t1 - t0).count();`. In both cases,
    after all 10 timed trials, compute the mean, population stddev
    (dividing by N=10, not N-1), min, and max, and emit them under
    the top-level `timing` key of reference.json. Use `%.9g`
    formatting for the four float fields (this applies to the quad
    driver too — the timing measurement precision is not the driver
    precision; `quadmath_snprintf` is not needed here — these are
    plain double seconds). The `trials_timed` field is the literal
    integer 10.

Set kernel_function_name and output_arrays in your submit_result
payload so they exactly match what the drivers actually do. All four
drivers share these values (same kernel function, same output array
names) — set each of the top-level fields once and they apply to all
four. If a driver writes an array under "outputs" by some name, that
same name must appear in output_arrays.

Return your result by calling the submit_result tool. All four keys
under `drivers.{quad,double,float,original}` are required; an absent
or empty driver fails the schema."""


CUDA_PROFILE = LanguageProfile(
    id="cuda",
    display_name="CUDA C++",
    source_suffixes=(".cu",),
    driver_filename="driver.cu",
    env_required=(),  # ARCH_ENV is optional; nvcc presence is checked in preflight.
    dynamic_verification=True,
    baseline_harness_system_prompt=BASELINE_HARNESS_SYSTEM_PROMPT,
    baseline_harness_output_schema=BASELINE_HARNESS_OUTPUT_SCHEMA,
    build_compile_command=_build_compile_command,
    build_syntax_check_command=_build_syntax_check_command,
    preflight=_preflight,
    detect_from_source=_detect_from_source,
    # v1b probe pipeline for CUDA. Baseline is `quad` (matching
    # Kokkos) via a host-only C++ + libquadmath oracle: nvcc has no
    # `__float128` support on either the host or device side, so the
    # quad driver cannot be a real CUDA program. The harness emits it
    # as plain C++ that unrolls the kernel launch into a serial host
    # `for` loop and swaps CUDA math intrinsics for their `q`-suffixed
    # quadmath equivalents (sqrtq, sinq, expq, fmaq, ...); the compile
    # step sniffs the driver source for `__float128` and switches from
    # nvcc to `g++ -lquadmath`. The harness is instructed to REFUSE to
    # emit a quad driver when the kernel uses rounding-mode intrinsics
    # (__fadd_rd, __fmul_ru, etc.) because those have no quadmath
    # analogue — see the `quad` bullet in the baseline_harness prompt.
    # Consequences of baseline_precision="quad":
    #   * Oracle promotion is active: after probe_compare succeeds,
    #     baselines/<stem>/probe/quad_seed42/reference.json is copied
    #     over baselines/<stem>/reference.json so the finish-gate
    #     comparator measures the rewritten kernel against true quad
    #     ground truth. Symmetric with Kokkos.
    #   * measure_speedup still takes the probe path
    #     (baselines/<stem>/probe/original_seed42/reference.json)
    #     since probe_precisions is non-empty; symmetric with Kokkos.
    baseline_precision="quad",
    probe_precisions=("quad", "double", "float", "original"),
)


__all__ = [
    "ARCH_ENV",
    "DEFAULT_ARCH",
    "NVCC",
    "CXX_STD",
    "OPT_FLAGS",
    "QUAD_PROBE_TOKEN",
    "QUAD_LIB",
    "QUAD_HOST_CXX",
    "BASELINE_HARNESS_SYSTEM_PROMPT",
    "BASELINE_HARNESS_OUTPUT_SCHEMA",
    "CUDA_PROFILE",
]
