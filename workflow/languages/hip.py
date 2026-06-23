"""HIP C++ language profile.

HIP is AMD's CUDA-shaped GPU runtime: same `kernel<<<...>>>(...)` launch
syntax, same `__global__` / `__device__` qualifiers, same per-thread
indexing intrinsics. The main differences from CUDA are:

  - The compiler is `hipcc` (a clang-based driver), not nvcc.
  - The runtime header is `<hip/hip_runtime.h>`, not `<cuda_runtime.h>`.
  - The runtime API is `hipMalloc` / `hipMemcpy` / `hipError_t` /
    `hipGetErrorString` / `hipDeviceSynchronize` / `hipGetLastError`,
    not the cuda* equivalents.
  - GPU target architectures use clang's `--offload-arch=<gfxXXX>` form
    (e.g. `gfx90a` for MI200/MI250X; `gfx942` for MI300; `gfx1100` for
    RX 7900-class consumer cards).

The precision-alias contract, splice sentinels, JSON output shape,
determinism rules, and staging-buffer rule are language-independent
invariants of the workflow and are reused from the CUDA prompt verbatim
(modulo the runtime API tokens).

v0 scope: `.hip` suffix only. Real ROCm codebases often use `.cpp`
files with `#include <hip/hip_runtime.h>` instead, which would put HIP
in the `.cpp` content-probe pool alongside Kokkos / SYCL / OpenMP
offload. Deferred until we have a HIP-capable host to smoke-test the
content-probe disambiguation against.

Two environment variables influence this profile at compile time:

  AGENT_PRECISION_HIP_ARCH  Optional. The `--offload-arch=` value passed
                            to hipcc. Defaults to `gfx90a` (MI200 /
                            MI250X, the most common HPC AMD target);
                            override on hosts with newer GPUs (`gfx942`
                            for MI300A/MI300X) or consumer cards
                            (`gfx1100` for RX 7900-class). Read at
                            compile time so monkeypatching the env in
                            tests affects every subsequent compile in
                            the same process.

The preflight checks that `hipcc` is on PATH and reports cleanly when
it is not so the operator gets a one-line diagnostic instead of a
confusing FileNotFoundError from subprocess.run.

UNIT-TESTED, NOT SMOKE-VALIDATED. This profile was landed without an
end-to-end run against a real HIP toolchain because no ROCm host was
available at implementation time. The CUDA profile (which this mirrors)
needed prompt iteration during its smoke test; expect similar iteration
on first real use.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .base import LanguageProfile, make_error_result


# Environment variable that, when set, overrides the default
# `--offload-arch=` value passed to hipcc. Intentionally namespaced
# under the project's AGENT_PRECISION_* prefix to avoid colliding with
# hipcc's own environment knobs (HIPCC_COMPILE_FLAGS_APPEND etc.).
ARCH_ENV = "AGENT_PRECISION_HIP_ARCH"

# Default architecture the project targets when ARCH_ENV is unset.
# gfx90a is MI200 / MI250X — the most common AMD HPC target as of the
# v0 timeline (Frontier, LUMI, Adastra). Override via ARCH_ENV is cheap;
# the explicit default is preferred over `--offload-arch=native` because
# a clear unsupported-arch error at compile time is easier to triage
# than a runtime kernel-image error on a different host.
DEFAULT_ARCH = "gfx90a"

# The compiler binary name. hipcc is ROCm's standard driver; alternative
# paths like `clang++ -x hip` exist but are not in scope for v1 — the
# harness prompt assumes hipcc's classic CUDA-style kernel-launch syntax
# `kernel<<<...>>>(...)` and the runtime headers (`<hip/hip_runtime.h>`).
HIPCC = "hipcc"
CXX_STD = "-std=c++17"
OPT_FLAGS = ("-O2",)


def _build_compile_command(driver_src: Path, driver_bin: Path) -> list[str]:
    """Assemble the hipcc argv list for a HIP driver compile.

    Reads AGENT_PRECISION_HIP_ARCH at call time (not import time) so a
    test that monkeypatches the env affects every subsequent compile in
    the same process. Assumes preflight has already verified hipcc is
    on PATH.
    """
    arch = os.environ.get(ARCH_ENV, DEFAULT_ARCH)
    return [
        HIPCC,
        CXX_STD,
        *OPT_FLAGS,
        f"--offload-arch={arch}",
        str(driver_src),
        "-o",
        str(driver_bin),
    ]


def _preflight() -> dict | None:
    """Verify hipcc is reachable before invoking the compiler.

    Returns None when `shutil.which("hipcc")` finds the compiler.
    Otherwise returns a make_error_result()-shaped dict the caller hands
    straight back to the orchestrator — no subprocess is spawned.

    AGENT_PRECISION_HIP_ARCH is intentionally NOT validated here; if
    the operator typoes the arch, hipcc itself will return a clear
    unsupported-target error during the compile, which is a cleaner
    signal than a Python-side allowlist of architectures that would
    need to track every new AMD GPU release.
    """
    if shutil.which(HIPCC) is None:
        return make_error_result(
            f"{HIPCC} not found on PATH. Install the ROCm toolchain and "
            f"ensure {HIPCC} is reachable, or set AGENT_PRECISION_HIP_ARCH "
            f"on a host that has a HIP toolchain."
        )
    return None


def _detect_from_source(kernel_source: str) -> bool:
    """Probe a source file to decide whether it's HIP.

    HIP claims `.hip` exclusively in v0 (see source_suffixes below), so
    detect_language() never actually reaches this probe under v1 — the
    suffix alone disambiguates. The helper is implemented for symmetry
    with the other profiles and as a safety net in case a future
    revision adds `.cpp` to HIP's suffix list (which would force a
    content-based tiebreak against Kokkos / SYCL / OpenMP offload). The
    probe looks for the canonical HIP runtime header, which is a
    structural marker that won't be triggered by an arbitrary comment.
    """
    if "<hip/hip_runtime.h>" in kernel_source:
        return True
    return False


# Per-language baseline-harness contract for HIP. The output schema is
# structurally identical to the Kokkos and CUDA schemas (a self-contained
# driver source, the kernel function name it calls, an inputs summary
# string, and the list of output array names) — the comparator and
# splice tools downstream don't care which language produced the driver,
# so the JSON shape is shared. Only the descriptions are tweaked to say
# "HIP" where the others say "Kokkos" or "CUDA".
BASELINE_HARNESS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "driver_source": {
            "type": "string",
            "description": (
                "The full driver source as a single self-contained .hip "
                "translation unit. Must inline the kernel source verbatim, "
                "compile with hipcc, and on execution write reference "
                "outputs to ./reference.json."
            ),
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
        "driver_source",
        "kernel_function_name",
        "inputs_summary",
        "output_arrays",
    ],
}


BASELINE_HARNESS_SYSTEM_PROMPT = """You are the baseline-harness agent
for a mixed-precision rewriting workflow.

You will be given a HIP C++ kernel source. Your job is to write a
self-contained HIP driver program that, when compiled with hipcc and run
later, exercises the kernel on a fixed set of inputs and writes a
reproducible reference output to ./reference.json. That JSON file will
eventually be the baseline against which a rewritten (lower-precision)
version of the same kernel is compared.

You do NOT compile, run, or simulate the kernel. You do NOT invent
numerical output values. Your only output is the driver source.

Hard requirements on the driver:

1. Single translation unit. Inline the kernel source verbatim into the
   driver (above main()). Do not introduce a build system or external
   headers beyond <hip/hip_runtime.h>, the C and C++ standard library,
   and anything the kernel itself already includes.

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
   determinism within a single architecture). Concretely:

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

   c. Seed any host-side RNG with a fixed integer (use 42 unless the
      kernel's apparent domain demands otherwise). The driver must
      produce the same numbers on every run.

3. Use hipMalloc / hipMemcpy for device allocations and host<->device
   transfers. After every HIP runtime call AND after the kernel launch,
   check the returned hipError_t (or call hipGetLastError() /
   hipDeviceSynchronize() after the launch) and abort with a clear
   stderr message on failure. A typical pattern:

       #define HIP_CHECK(expr) do {                                  \\
         hipError_t _err = (expr);                                   \\
         if (_err != hipSuccess) {                                   \\
           std::fprintf(stderr, "HIP error at %s:%d: %s\\n",         \\
               __FILE__, __LINE__, hipGetErrorString(_err));         \\
           std::exit(1);                                             \\
         }                                                           \\
       } while (0)

   This pattern is what lets a misconfigured runtime (missing driver
   library, wrong LD_LIBRARY_PATH, unsupported architecture) surface
   as a clear diagnostic instead of a silent wrong-answer.

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
   (HIP kernels typically take `float*` / `double*` device pointers).
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
   the non-const counterpart of the alias's pointee type so hipMemcpy
   can write into it, then bind to the const-aliased pointer for the
   launch:

       // correct: derive the writable pointee type from the alias
       using aElem = std::remove_const_t<std::remove_pointer_t<aType>>;
       aElem* a_nc = nullptr;
       HIP_CHECK(hipMalloc(&a_nc, N * sizeof(aElem)));
       HIP_CHECK(hipMemcpy(a_nc, host_a.data(),
                           N * sizeof(aElem), hipMemcpyHostToDevice));
       aType a = a_nc;  // const-binds; precision is whatever aType says
       kernel<<<blocks, threads>>>(a, ...);

   Do NOT hardcode the staging buffer's element type as `double`. A
   hardcoded `double* a_nc; hipMalloc(&a_nc, N * sizeof(double));`
   breaks the contract: when the rewriter redefines `aType` to
   `const float*`, the `aType a = a_nc;` assignment no longer compiles
   because `double*` does not convert to `const float*`. The
   `std::remove_const_t<std::remove_pointer_t<...>>` form is the only
   one that survives a precision change to the alias.

   For local host scratch (std::vector buffers, RNG distributions) that
   do not directly become kernel arguments, plain `double` is fine —
   only the values that cross the kernel boundary need to flow through
   the aliases.

7. After the kernel launch, hipMemcpy any device outputs you intend to
   record back to host buffers before iterating them for JSON emission.
   Call hipDeviceSynchronize() before the copy-back so launch errors
   surface here, not later.

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

10. Begin the driver with a top-of-file comment that tells the operator
    to `cd` into the baseline directory (baselines/<file_stem>/) before
    running, so ./reference.json lands next to the driver source. Also
    mention the compile command in a comment (a typical hipcc build
    line is fine; the operator will adapt it).

Set kernel_function_name and output_arrays in your submit_result
payload so they exactly match what the driver actually does. If your
driver writes an array under "outputs" by some name, that same name
must appear in output_arrays.

Return your result by calling the submit_result tool."""


HIP_PROFILE = LanguageProfile(
    id="hip",
    display_name="HIP C++",
    source_suffixes=(".hip",),
    driver_filename="driver.hip",
    env_required=(),  # ARCH_ENV is optional; hipcc presence is checked in preflight.
    dynamic_verification=True,
    baseline_harness_system_prompt=BASELINE_HARNESS_SYSTEM_PROMPT,
    baseline_harness_output_schema=BASELINE_HARNESS_OUTPUT_SCHEMA,
    build_compile_command=_build_compile_command,
    preflight=_preflight,
    detect_from_source=_detect_from_source,
)


__all__ = [
    "ARCH_ENV",
    "DEFAULT_ARCH",
    "HIPCC",
    "CXX_STD",
    "OPT_FLAGS",
    "BASELINE_HARNESS_SYSTEM_PROMPT",
    "BASELINE_HARNESS_OUTPUT_SCHEMA",
    "HIP_PROFILE",
]
