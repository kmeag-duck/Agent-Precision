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


def _build_compile_command(driver_src: Path, driver_bin: Path) -> list[str]:
    """Assemble the nvcc argv list for a CUDA driver compile.

    Reads AGENT_PRECISION_CUDA_ARCH at call time (not import time) so a
    test that monkeypatches the env affects every subsequent compile in
    the same process. Assumes preflight has already verified nvcc is on
    PATH.
    """
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
    """Assemble the nvcc argv list for a pre-write CUDA driver check.

    Returns None when nvcc is not on PATH — the harness-validation gate
    then skips silently rather than failing every run on a host without
    a CUDA toolchain (validation is a quality improvement, not a hard
    requirement, exactly as for the Kokkos gate).

    nvcc has no true parse-only mode: it rejects `-fsyntax-only`
    (verified empirically — `nvcc fatal: Unknown option '-fsyntax-only'`).
    The closest it offers is `-c -o /dev/null`, which compiles both the
    host and device sides to an object but never links (no CUDA runtime
    libraries, no GPU required beyond nvcc itself) and never writes an
    artifact. That is a strict subset of the real compile flags
    (`_build_compile_command`): same `-std`, `-O2`, and `-arch`, but the
    final `-o <bin>` is replaced with `-c -o /dev/null`. It catches the
    malformed-driver class the gate exists for (e.g. the inconsistent
    alias-naming that motivated the Kokkos gate) at a fraction of a full
    compile+link, before any file is written to disk.

    The `-arch` value is read at call time (same contract as
    `_build_compile_command`) so a monkeypatched env is honored.
    """
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
                "Three self-contained .cu driver translation units, one per "
                "probe precision. Each MUST inline the kernel source "
                "verbatim between the KERNEL BEGIN/END sentinels, compile "
                "with nvcc, and on execution write reference outputs to "
                "./reference.json. All three share the same kernel body, "
                "the same per-parameter <ParamName>Type alias NAMES, the "
                "same RNG_SEED, and the same output-array names and "
                "lengths; they differ only in the alias RHSes (which set "
                "per-parameter storage precision), any host-side scratch "
                "values that flow into the kernel, and the JSON floating-"
                "point formatting. See the PER-PRECISION DRIVERS block in "
                "the system prompt for the full contract."
            ),
            "properties": {
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
            "required": ["double", "float", "original"],
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

You will be given a CUDA C++ kernel source. Your job is to write a
self-contained CUDA driver program that, when compiled with nvcc and run
later, exercises the kernel on a fixed set of inputs and writes a
reproducible reference output to ./reference.json. That JSON file will
eventually be the baseline against which a rewritten (lower-precision)
version of the same kernel is compared.

You do NOT compile, run, or simulate the kernel. You do NOT invent
numerical output values. Your only output is the three driver sources.

PER-PRECISION DRIVERS. You emit THREE drivers in a single
submit_result call under `drivers.{double,float,original}`. All three
MUST share:

  - the same inlined kernel source (byte-identical between the
    `// ---- KERNEL BEGIN ----` and `// ---- KERNEL END ----` sentinels),
  - the same per-parameter `<ParamName>Type` alias NAMES (item 6),
  - the same `static constexpr int RNG_SEED = ...;` line (item 3),
  - the same input sizes, the same output array names, and the same
    output array lengths (the comparator requires shape-identical
    reference.json across precisions).

The three drivers differ ONLY in:

  - the RHS of each per-parameter `<ParamName>Type` alias,
  - any host-side scratch values that ultimately flow into the kernel
    (RNG fill loops, staging-buffer element types, etc.) — these must
    be the appropriate precision end-to-end, not down-converted
    through `double` mid-driver,
  - the reference.json floating-point formatting (item 8).

Per-precision rules:

  - `double`: aliases resolve to `double` / `double*` (pointer aliases
    replace their pointee with `double`, keeping any const-qualifiers);
    JSON values written with `"%.17g"`. This driver exists purely to
    feed the probe-evidence pipeline as a uniform-double point of
    comparison against `float` and `original`; it is NOT the splice
    scaffold (that role belongs to `original`).
  - `float`: aliases resolve to `float` / `float*` (same
    const-qualifier rule); JSON values written with `"%.9g"`. Host-
    side RNG fill and any staging buffers that flow into the kernel
    must be `float` end-to-end — do NOT fill a `std::vector<double>`
    and then `cudaMemcpy` a `float`-typed device buffer from it.
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
(NOT `drivers["double"]`) to `baselines/<stem>/driver.cu` as the
canonical splice scaffold. The rewriter later splices its kernel
between the sentinels of that file to produce
`baselines/<stem>/rewritten/driver.cu`. Because `main()` in the
`original` driver already constructs kernel arguments through
aliases that match the user's exact parameter types, the rewriter
can redefine those aliases (e.g. downcast an `aType` from `double*`
to `float*`) without touching `main()`; the change propagates
through the signature for free. The other two drivers (double,
float) live only under `baselines/<stem>/probe/<precision>/` and
feed the probe-evidence pipeline.

The `original` driver additionally serves as the pre-rewrite
wall-clock reference for `measure_speedup`: its `timing` block from
seed=42 is read directly out of
`baselines/<stem>/probe/original_seed42/reference.json` as the
"baseline" side of the speedup ratio. The `timing` block is present
in every driver (item 11) but the `double` block is NOT used for
speedup — the `double` driver is a uniform-precision rewrite of a
potentially-mixed kernel, so it does not represent what "the user's
kernel before rewriting" actually runs at.

None of these driver variants changes the kernel function body.
They change the alias RHSes (item 6 below), any host-side scratch
values that flow into the kernel, and the JSON output formatting
(item 8).

Hard requirements on each driver:

1. Single translation unit. Inline the kernel source verbatim into the
   driver (above main()). Do not introduce a build system or external
   headers beyond <cuda_runtime.h>, the C and C++ standard library, and
   anything the kernel itself already includes.

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
    the two probe drivers live at
    `baselines/<file_stem>/probe/{double,float}/driver.cu`. Also
    mention the compile command in a comment (a typical nvcc build
    line is fine; the operator will adapt it).

11. Kernel timing. Repeat the kernel launch N=11 times: 1 untimed
    warmup launch followed by 10 timed trials. Time only the kernel
    launch itself — NOT device allocation, host<->device transfers,
    or JSON emission. Use cudaEvent for GPU timing:
    `cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);`
    then per trial, `cudaEventRecord(start)`, launch the kernel,
    `cudaEventRecord(stop)`, `cudaEventSynchronize(stop)`,
    `float ms; cudaEventElapsedTime(&ms, start, stop);`. Convert to
    seconds (`ms / 1000.0`) and push into a `std::vector<double>`.
    After all 10 timed trials, compute the mean, population stddev
    (dividing by N=10, not N-1), min, and max, and emit them under
    the top-level `timing` key of reference.json. Use `%.9g`
    formatting for the four float fields. The `trials_timed` field
    is the literal integer 10.

Set kernel_function_name and output_arrays in your submit_result
payload so they exactly match what the drivers actually do. The three
drivers share these values (same kernel function, same output array
names) — set each of the top-level fields once and they apply to all
three. If a driver writes an array under "outputs" by some name, that
same name must appear in output_arrays.

Return your result by calling the submit_result tool. All three keys
under `drivers.{double,float,original}` are required; an absent or
empty driver fails the schema."""


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
    # v1a probe pipeline for CUDA. Baseline is `double` (not `quad`
    # like Kokkos) because nvcc has no `__float128` support and there
    # is no comparably portable software-quad path on the device side.
    # A host-side quad oracle port is Phase 1b (see AGENTS.md
    # "Planned next steps"). Consequences of baseline_precision=
    # "double" (documented in AGENTS.md):
    #   * Oracle promotion is a no-op — the finish-gate comparator
    #     measures rewritten output against the `original`-precision
    #     `baselines/<stem>/reference.json`, not a promoted quad
    #     reference. For typical `--sig-figs 6` runs on double
    #     baselines this suffices (double has ~15-17 sig figs of
    #     headroom over the tolerance).
    #   * measure_speedup still takes the probe path
    #     (baselines/<stem>/probe/original_seed42/reference.json)
    #     since probe_precisions is non-empty; symmetric with Kokkos.
    baseline_precision="double",
    probe_precisions=("double", "float", "original"),
)


__all__ = [
    "ARCH_ENV",
    "DEFAULT_ARCH",
    "NVCC",
    "CXX_STD",
    "OPT_FLAGS",
    "BASELINE_HARNESS_SYSTEM_PROMPT",
    "BASELINE_HARNESS_OUTPUT_SCHEMA",
    "CUDA_PROFILE",
]
