"""OpenMP target-offload language profile.

OpenMP target offload (commonly "OMP-offload") uses `#pragma omp target`
directives to move execution from the host onto an accelerator (GPU or
discrete accelerator). Unlike CUDA / HIP (free `__global__` functions with
launch syntax) and SYCL (lambdas submitted to a queue), OMP-offload kernels
are *plain C/C++ functions* called from a `#pragma omp target` region —
the compiler does the heavy lifting of identifying the offload region,
generating device code, marshaling data via `map(...)` clauses, and
launching it. From a splice / alias perspective this is the closest of
the four to CUDA / HIP: the function signature is real and editable.

The compiler ecosystem is more fragmented than CUDA's (one vendor) or
HIP's (one vendor + ROCm). The two dominant OMP-offload paths are:

  - LLVM Clang (`clang++ -fopenmp -fopenmp-targets=<triple>`). Open,
    cross-vendor, supports both nvptx64 and amdgcn back-ends. This is
    the v0 default.
  - NVIDIA HPC SDK `nvc++` (`-mp=gpu`). Production-grade on NVIDIA
    hardware but uses different flag syntax and is single-vendor.

Intel's `icpx` also supports OMP-offload with clang-style flags, so the
default `clang++` driver name is overridable via AGENT_PRECISION_OMP_CXX
to cover the (icpx, nvc++, vendored clang) escape hatches without
hard-coding any of them.

Target triple selection IS a compile-time concern in OMP-offload (unlike
SYCL, where it's runtime), so this profile exposes a second env knob:
AGENT_PRECISION_OMP_TARGET. Default `nvptx64-nvidia-cuda` mirrors the
HIP profile's `gfx90a` default — pick the most common scientific-
computing target as a sensible out-of-the-box choice; the operator
overrides for `amdgcn-amd-amdhsa` (AMD) or other triples.

The main differences from CUDA / HIP / SYCL:

  - Kernels are plain functions, not __global__ / not lambdas. The
    splice operates on the function body, same as CUDA / HIP, but the
    launch is a `#pragma omp target` block around a call to the
    function (or around the function body itself).
  - Memory model is map-clause based: `map(to:a) map(from:c)
    map(tofrom:b)`. The harness mandates explicit `map(tofrom:)` (or
    `map(to:)` + `map(from:)`) clauses for every array passed to the
    target region — implicit mapping is fragile across compilers.
  - Reproducibility is harder to enforce than SYCL's in-order queue
    because OMP doesn't have a queue. The harness mandates
    `omp_set_num_threads(1)` to serialize team execution and forbids
    `reduction(+:...)` clauses (OMP reductions are unordered).
  - GPU-arch is a compile-time concern. Unlike SYCL (runtime device
    selector) but like CUDA / HIP, the target triple must be baked in.

The precision-alias contract is structurally identical to CUDA / HIP:
plain pointer/scalar function parameters get per-parameter aliases
(`using <ParamName>Type = ...;`) immediately inside the kernel
sentinels. Map clauses, function signatures, and the call site all
flow through the aliases automatically once defined.

UNIT-TESTED, NOT SMOKE-VALIDATED. This profile was landed without an
end-to-end run against a real OMP-offload toolchain because no host
with a working `clang++ -fopenmp -fopenmp-targets=...` install was
available at implementation time. The CUDA profile needed prompt
iteration during its smoke test; expect MORE iteration here because
the cross-compiler flag-syntax differences (clang vs nvc++) and the
map-clause / reduction-determinism rules are richer than CUDA's
free-function model.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .base import LanguageProfile, make_error_result


# Environment variable that, when set, overrides the default OMP-offload
# compiler driver. Intentionally namespaced under the project's
# AGENT_PRECISION_* prefix so it does not collide with vendor knobs like
# CXX (which build systems already overload heavily) or with OMP's own
# OMP_* runtime variables.
CXX_ENV = "AGENT_PRECISION_OMP_CXX"

# Environment variable that, when set, overrides the default offload
# target triple. Common values: `nvptx64-nvidia-cuda` (NVIDIA), and
# `amdgcn-amd-amdhsa` (AMD). The literal env value is passed verbatim
# to `-fopenmp-targets=<value>`; no allow-list validation because the
# universe of valid triples is compiler-version-dependent.
TARGET_ENV = "AGENT_PRECISION_OMP_TARGET"

# Default compiler binary. `clang++` is the LLVM driver and supports
# both nvptx and amdgcn back-ends via -fopenmp-targets=<triple>. The
# operator can override CXX_ENV to `icpx` (Intel oneAPI, clang-flavored
# flag syntax), `nvc++` (NVIDIA HPC SDK, but flag syntax differs and
# the operator would also need to override TARGET handling), or a
# fully-qualified path to a vendored compiler. The `-fopenmp` and
# `-fopenmp-targets=<triple>` flags are appended unconditionally by
# _build_compile_command — both icpx and clang++ accept them; nvc++
# does NOT and is not recommended for the v0 escape hatch.
DEFAULT_CXX = "clang++"

# Default offload target triple. nvptx64-nvidia-cuda covers NVIDIA
# GPUs (the most common scientific-computing target). The operator
# overrides TARGET_ENV for AMD (`amdgcn-amd-amdhsa`) or for less
# common triples.
DEFAULT_TARGET = "nvptx64-nvidia-cuda"

# Compile flags. C++17 matches SYCL and HIP defaults; -O2 matches the
# other profiles' default optimization level. `-fopenmp` enables
# OpenMP, and `-fopenmp-targets=<triple>` enables device-side code
# generation for the chosen offload target. These are appended inside
# _build_compile_command rather than baked into OPT_FLAGS to keep the
# "OMP-essential flags" and "performance flags" categories visually
# distinct.
CXX_STD = "-std=c++17"
OPT_FLAGS = ("-O2",)
OMP_FLAG = "-fopenmp"


def _resolve_compiler() -> str:
    """Return the OMP-offload compiler driver name, honoring CXX_ENV.

    Read at call time, not at import time, so a test that
    monkeypatches the env affects every subsequent compile in the
    same process. Returns the literal env value when set (no
    validation here — `_preflight` checks PATH separately so a typo
    surfaces as a "not found on PATH" diagnostic).
    """
    return os.environ.get(CXX_ENV, DEFAULT_CXX)


def _resolve_target() -> str:
    """Return the OMP-offload target triple, honoring TARGET_ENV.

    Same call-time-read contract as `_resolve_compiler` so tests that
    monkeypatch the env affect every subsequent compile. Returns the
    literal env value when set; no allow-list validation because the
    universe of valid triples is compiler-version-dependent and we
    prefer to let the compiler's own diagnostics surface a typo
    (rather than maintain a Python-side list that would silently
    reject a brand-new triple a future compiler adds).
    """
    return os.environ.get(TARGET_ENV, DEFAULT_TARGET)


def _build_compile_command(driver_src: Path, driver_bin: Path) -> list[str]:
    """Assemble the OMP-offload compile argv list.

    Reads AGENT_PRECISION_OMP_CXX and AGENT_PRECISION_OMP_TARGET at
    call time (not import time) so a test that monkeypatches either
    env affects every subsequent compile in the same process. Assumes
    preflight has already verified the chosen compiler is on PATH.
    The two compile-time OMP flags are appended individually so the
    argv list reads as a self-documenting record of what the profile
    promises (compiler, std, optimization, openmp, target).
    """
    cxx = _resolve_compiler()
    target = _resolve_target()
    return [
        cxx,
        CXX_STD,
        *OPT_FLAGS,
        OMP_FLAG,
        f"-fopenmp-targets={target}",
        str(driver_src),
        "-o",
        str(driver_bin),
    ]


def _preflight() -> dict | None:
    """Verify the OMP-offload compiler is reachable before invoking it.

    Returns None when `shutil.which(<resolved compiler>)` finds the
    driver. Otherwise returns a make_error_result()-shaped dict the
    caller hands straight back to the orchestrator — no subprocess
    is spawned.

    AGENT_PRECISION_OMP_CXX is intentionally NOT validated for "looks
    like a real OMP-offload compiler"; if the operator sets it to
    `g++` (which has limited offload support) by mistake, the compile
    itself will fail with a clear diagnostic about missing
    -fopenmp-targets handling, which is a cleaner signal than a
    Python-side allowlist that would need to track every future
    compiler version's offload capabilities.

    AGENT_PRECISION_OMP_TARGET is not validated at preflight either,
    because it is only meaningful in the context of the chosen
    compiler — `clang++` and `nvc++` accept different triple
    spellings, and the compiler's own "no offload runtime for
    <triple>" diagnostic is more useful than a Python-side check.
    """
    cxx = _resolve_compiler()
    if shutil.which(cxx) is None:
        return make_error_result(
            f"{cxx} not found on PATH. Install an OMP-offload "
            f"toolchain (LLVM clang++ with offload runtime / Intel "
            f"oneAPI icpx) and ensure {cxx} is reachable, or set "
            f"{CXX_ENV} to the driver binary name on a host that has "
            f"an OMP-offload-capable compiler."
        )
    return None


def _detect_from_source(kernel_source: str) -> bool:
    """Probe a `.cpp` source to decide whether it's OMP-offload.

    OMP-offload shares the `.cpp` suffix with Kokkos and SYCL, so
    this probe IS consulted at runtime by `detect_language()` for
    `.cpp` inputs. It looks for a single structural marker: the
    `#pragma omp target` directive. This is intentionally strict —
    host-only OpenMP code uses `#pragma omp parallel` / `#pragma omp
    for` / etc, which would false-positive a looser
    `#pragma omp`-anywhere check. Only `#pragma omp target`
    unambiguously identifies a kernel that needs the offload
    toolchain.

    Variants the probe accepts:
      - `#pragma omp target` (plain)
      - `#pragma omp target teams`
      - `#pragma omp target data`
      - `#pragma omp target enter data`
      - `#pragma omp target update`

    All of these begin with `#pragma omp target` as a prefix, so a
    substring search catches them uniformly. The Kokkos and SYCL
    probes are consulted before this one (insertion order in
    PROFILES); a `.cpp` source matching multiple probes (vanishingly
    unlikely in practice — `Kokkos::` and `sycl::queue` and `#pragma
    omp target` rarely co-occur) is treated as the earliest match.
    """
    return "#pragma omp target" in kernel_source


# Per-language baseline-harness contract for OMP-offload. The output
# schema is structurally identical to the Kokkos / CUDA / HIP / SYCL
# schemas (a self-contained driver source, the kernel function name it
# launches, an inputs summary string, and the list of output array
# names) — the comparator and splice tools downstream don't care which
# language produced the driver, so the JSON shape is shared. Only the
# descriptions are tweaked to say "OMP-offload" where the others say
# "Kokkos" / "CUDA" / "HIP" / "SYCL".
BASELINE_HARNESS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "driver_source": {
            "type": "string",
            "description": (
                "The full driver source as a single self-contained .cpp "
                "translation unit. Must inline the kernel source verbatim, "
                "compile with an OMP-offload-capable C++ compiler "
                "(clang++ -fopenmp -fopenmp-targets=<triple> by default), "
                "and on execution write reference outputs to "
                "./reference.json."
            ),
        },
        "kernel_function_name": {
            "type": "string",
            "description": (
                "Name of the kernel function the driver invokes from "
                "inside its `#pragma omp target` region. OMP-offload "
                "kernels are plain C/C++ functions (not __global__ / "
                "not lambdas), so this is the function originally given "
                "in the kernel source."
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
        "driver_source",
        "kernel_function_name",
        "inputs_summary",
        "output_arrays",
    ],
}


BASELINE_HARNESS_SYSTEM_PROMPT = """You are the baseline-harness agent
for a mixed-precision rewriting workflow.

You will be given an OpenMP target-offload C++ kernel source. Your job
is to write a self-contained driver program that, when compiled with an
OMP-offload-capable C++ compiler (clang++ -fopenmp
-fopenmp-targets=<triple> by default) and run later, exercises the
kernel on a fixed set of inputs and writes a reproducible reference
output to ./reference.json. That JSON file will eventually be the
baseline against which a rewritten (lower-precision) version of the
same kernel is compared.

You do NOT compile, run, or simulate the kernel. You do NOT invent
numerical output values. Your only output is the driver source.

Hard requirements on the driver:

1. Single translation unit. Inline the kernel source verbatim into the
   driver (above main()). Do not introduce a build system or external
   headers beyond <omp.h>, the C and C++ standard library, and anything
   the kernel itself already includes.

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
   determinism within a single device). Concretely:

   a. Call `omp_set_num_threads(1);` at the top of main(), before any
      OMP construct. OMP has no equivalent of SYCL's in-order queue or
      CUDA's stream 0; the only reliable way to make team execution
      deterministic on the host side is to limit it to one thread. The
      device-side thread count inside the target region is controlled
      separately by the `#pragma omp target teams num_teams(1)
      thread_limit(1)` clauses (see below).

   b. Do NOT use `reduction(+:...)` (or any other reduction clause) on
      floating-point variables. OMP reduction order is unspecified
      across teams / threads and would make the reference non-
      reproducible. If the kernel under test uses reductions, that is a
      kernel design choice and is judged by the rewriter / verifier
      separately; the DRIVER must compute its reference output without
      introducing reductions.

   c. Launch the kernel from a `#pragma omp target` region with
      single-team / single-thread bounds for the baseline:

          #pragma omp target teams num_teams(1) thread_limit(1) \\
              map(to: ...) map(from: ...)
          {
              // call into the kernel function here, or inline its body
          }

      This is deliberately conservative — single-team / single-thread
      is the slowest configuration, but it is the only one that
      guarantees a reproducible reference across compilers and
      devices. A future smoke-validation phase may relax this once a
      cross-compiler reproducibility story is established.

   d. Seed any host-side RNG with a fixed integer (use 42 unless the
      kernel's apparent domain demands otherwise). The driver must
      produce the same numbers on every run.

3. Map clauses are MANDATORY for every array that crosses the
   host/device boundary. Implicit data movement (relying on the
   compiler's default-mapping rules) is fragile across clang++ vs
   icpx vs nvc++ and would make the reference non-portable. Use:

     - `map(to: a[0:N])` for inputs the kernel only reads.
     - `map(from: c[0:N])` for outputs the kernel only writes (the
       device-side initial values are undefined).
     - `map(tofrom: b[0:N])` for inputs the kernel reads AND writes.

   Always specify the explicit array section (`a[0:N]`, not bare `a`)
   so the compiler does not silently size the transfer based on the
   pointer's static type. The map clauses go on the `#pragma omp
   target` directive itself, not on an enclosing `#pragma omp target
   data` region (the target-data form is permitted but more
   compiler-sensitive than the inline form for v0).

4. Choose modest input sizes and distributions appropriate to the
   kernel from its signature and apparent scientific domain. Aim for
   a driver that runs in a few seconds on a single device, not hours.
   Because the baseline runs single-team / single-thread, prefer
   sizes near the lower end of the "typical" range (1e3 to 1e5
   elements depending on per-element cost). Document the inputs you
   chose in inputs_summary.

5. If the task message names a TARGET KERNEL, wrap that kernel.
   Otherwise, infer the kernel function from the source — there
   should be exactly one obvious candidate.

6. Do not modify the kernel function. Do not change any variable's
   precision. Do not invent or rename kernel arguments. The whole
   point is to capture the *original* kernel's output as the
   reference.

   EXCEPTION — precision-alias contract. Immediately inside the
   '// ---- KERNEL BEGIN ----' sentinel and above the kernel function
   definition, emit one `using` alias per floating-point kernel
   parameter. Naming convention: `<ParamName>Type` (CamelCase of the
   parameter name + 'Type' suffix). The kernel function header,
   internal variable declarations, and the call from inside the
   `#pragma omp target` region MUST then refer to those aliases, not
   to the underlying types. For example, a kernel originally
   declared

       void axpy(double* c, const double* a, const double* b, int N);

   becomes, inside the sentinels:

       using cType = double;
       using aType = double;
       using bType = double;

       void axpy(cType* c, const aType* a, const bType* b, int N) {
           // body unchanged
       }

   The map clauses, the host buffer declarations, and the call site
   inside `#pragma omp target` all flow through the aliases
   automatically once defined.

   Integer parameters (sizes, counts, indices like `int N`) do NOT
   get aliases — only floating-point parameters. Scalar
   floating-point parameters (e.g. a `double alpha`) DO get aliases,
   same convention.

   Host-side buffer declarations that back a mapped array must use
   the alias as their element type, so that redefining the alias to
   `float` does not break the `map(to:)` / `map(from:)` clause:

       std::vector<aType> host_a(N);
       // ... fill host_a ...
       #pragma omp target map(to: host_a.data()[0:N]) ...

   Do NOT hardcode the host vector as `std::vector<double>` when its
   alias is `aType`. A hardcoded
   `std::vector<double> host_a(N);` paired with an `aType`-typed
   kernel parameter breaks the contract: when the rewriter redefines
   `aType` to `float`, the map transfer interprets `sizeof(double)`
   bytes per element on the host side and `sizeof(float)` on the
   device, silently corrupting the transfer.

   For local host scratch (RNG distributions, intermediate
   computations) that does NOT cross into a mapped buffer, plain
   `double` is fine — only the values that flow through map clauses
   need to go through the aliases.

7. After the `#pragma omp target` region exits, the `map(from:...)`
   and `map(tofrom:...)` clauses have already synchronized the
   device-side outputs back to the host pointers. You may iterate
   the host buffers directly for JSON emission — no explicit
   synchronization call is needed (the implicit barrier at the end
   of the target region handles it).

8. Write the reference output to './reference.json' (relative to the
   driver's working directory) using std::ofstream and "%.17g"
   formatting for floating-point values. Do NOT pull in a third-party
   JSON library — hand-roll the writer; output arrays are flat arrays
   of doubles, so a few loops with manual braces, commas, and newlines
   are sufficient.

9. The JSON document must have exactly this shape:

       {
         "kernel": "<kernel_function_name>",
         "seed": <integer seed>,
         "inputs": { "N": <int>, ... },
         "outputs": { "<name>": [ <double>, ... ], ... }
       }

   "inputs" carries enough metadata for a human reader to understand
   what the driver did (sizes, distributions if represented as
   strings). "outputs" carries one named flat array per output the
   comparator will check. The names under "outputs" must match
   output_arrays in your submit_result payload.

10. Begin the driver with a top-of-file comment that tells the
    operator to `cd` into the baseline directory (baselines/<file_stem>/)
    before running, so ./reference.json lands next to the driver
    source. Also mention the compile command in a comment (a typical
    `clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -std=c++17
    -O2 driver.cpp -o driver` build line is fine; the operator will
    adapt it).

Set kernel_function_name and output_arrays in your submit_result
payload so they exactly match what the driver actually does. If your
driver writes an array under "outputs" by some name, that same name
must appear in output_arrays.

Return your result by calling the submit_result tool."""


OMP_OFFLOAD_PROFILE = LanguageProfile(
    id="omp_offload",
    display_name="OpenMP target-offload C++",
    source_suffixes=(".cpp",),
    driver_filename="driver.cpp",
    env_required=(),  # CXX_ENV / TARGET_ENV are optional; compiler presence is checked in preflight.
    dynamic_verification=True,
    baseline_harness_system_prompt=BASELINE_HARNESS_SYSTEM_PROMPT,
    baseline_harness_output_schema=BASELINE_HARNESS_OUTPUT_SCHEMA,
    build_compile_command=_build_compile_command,
    preflight=_preflight,
    detect_from_source=_detect_from_source,
)


__all__ = [
    "CXX_ENV",
    "TARGET_ENV",
    "DEFAULT_CXX",
    "DEFAULT_TARGET",
    "CXX_STD",
    "OPT_FLAGS",
    "OMP_FLAG",
    "BASELINE_HARNESS_SYSTEM_PROMPT",
    "BASELINE_HARNESS_OUTPUT_SCHEMA",
    "OMP_OFFLOAD_PROFILE",
]
