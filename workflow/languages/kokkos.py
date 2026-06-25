"""Kokkos C++ language profile.

This profile owns the constants and prompt that lived inline in
workflow/tools.py and workflow/registry.py before the per-language
refactor. Keeping them grouped here lets a reader see everything that's
Kokkos-specific in one place, and lets workflow.tools fall back to
re-exporting them as module-level names for backward compatibility with
tests that imported them directly.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .base import LanguageProfile, make_error_result


# Environment variable that names the Kokkos install prefix (the
# directory containing include/ and lib/). Intentionally namespaced so
# it does not collide with Kokkos's own CMake convention (Kokkos_ROOT)
# or with a system-wide install.
ROOT_ENV = "AGENT_PRECISION_KOKKOS_ROOT"

# Compile flags. C++20 is required by the baseline driver template; the
# OpenMP host backend is what the Kokkos install in this repo was built
# with, so -fopenmp is mandatory at link time. Kokkos here is shipped as
# static archives (libkokkoscore.a, libkokkoscontainers.a), so no rpath
# is needed and the resulting binary has no libkokkos* in its dynamic
# NEEDED list.
CXX = "g++"
CXX_STD = "-std=c++20"
OPT_FLAGS = ("-O2", "-fopenmp")
KOKKOS_LIBS = ("-lkokkoscore", "-lkokkoscontainers")
EXTRA_LIBS = ("-lpthread", "-ldl")


def _build_compile_command(driver_src: Path, driver_bin: Path) -> list[str]:
    """Assemble the g++ argv list for a Kokkos driver compile.

    Reads AGENT_PRECISION_KOKKOS_ROOT at call time (not import time) so
    a test that monkeypatches the env affects every subsequent compile
    in the same process. Assumes preflight has already verified the
    var is set and the directory layout looks Kokkos-shaped.
    """
    root = Path(os.environ[ROOT_ENV])
    include_dir = root / "include"
    lib_dir = root / "lib"
    return [
        CXX,
        CXX_STD,
        *OPT_FLAGS,
        f"-I{include_dir}",
        f"-L{lib_dir}",
        str(driver_src),
        *KOKKOS_LIBS,
        *EXTRA_LIBS,
        "-o",
        str(driver_bin),
    ]


def _preflight() -> dict | None:
    """Verify the Kokkos install before invoking the compiler.

    Returns None when AGENT_PRECISION_KOKKOS_ROOT is set and points at
    a directory that has both include/ and lib/ subdirectories.
    Otherwise returns a make_error_result()-shaped dict the caller hands
    straight back to the orchestrator.
    """
    kokkos_root = os.environ.get(ROOT_ENV)
    if not kokkos_root:
        return make_error_result(
            f"{ROOT_ENV} is not set. Point it at a Kokkos install "
            f"prefix (the directory containing include/ and lib/)."
        )
    root = Path(kokkos_root)
    include_dir = root / "include"
    lib_dir = root / "lib"
    if not include_dir.is_dir() or not lib_dir.is_dir():
        return make_error_result(
            f"{ROOT_ENV}={kokkos_root!r} does not look like a "
            f"Kokkos install prefix (missing include/ or lib/)."
        )
    return None


def _detect_from_source(kernel_source: str) -> bool:
    """Probe a `.cpp` source to decide whether it's a Kokkos kernel.

    Looks for any of three structural markers: the canonical include
    line, any use of the Kokkos:: namespace, or the KOKKOS_LAMBDA macro
    (the standard Kokkos parallel-construct lambda wrapper, which is
    unmistakably Kokkos-only). All three are structural — a stray
    "Kokkos" in a comment won't trigger the include check, a bare
    reference to "Kokkos" without "::" won't trigger the namespace
    check, and KOKKOS_LAMBDA is a vendor-prefixed macro that no other
    .cpp-claiming framework defines. The combined heuristic is what
    `detect_language()` uses to disambiguate a .cpp file between Kokkos
    and SYCL / HIP-cpp / OpenMP-offload.
    """
    if "<Kokkos_Core.hpp>" in kernel_source:
        return True
    if "Kokkos::" in kernel_source:
        return True
    if "KOKKOS_LAMBDA" in kernel_source:
        return True
    return False


# Per-language baseline-harness contract. Extracted verbatim from the
# pre-refactor registry.BASELINE_HARNESS_SYSTEM_PROMPT so the Kokkos
# behavior is byte-for-byte unchanged.
BASELINE_HARNESS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "driver_source": {
            "type": "string",
            "description": (
                "The full driver source as a single self-contained .cpp "
                "translation unit. Must inline the kernel source verbatim, "
                "compile against a standard Kokkos toolchain, and on "
                "execution write reference outputs to ./reference.json."
            ),
        },
        "kernel_function_name": {
            "type": "string",
            "description": (
                "Name of the kernel function the driver calls. Must match a "
                "function defined in the inlined kernel source."
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

You will be given a Kokkos C++ kernel source. Your job is to write a
self-contained C++ driver program that, when compiled and run later,
exercises the kernel on a fixed set of inputs and writes a reproducible
reference output to ./reference.json. That JSON file will eventually be
the baseline against which a rewritten (lower-precision) version of the
same kernel is compared.

You do NOT compile, run, or simulate the kernel. You do NOT invent
numerical output values. Your only output is the driver source.

BASELINE PRECISION directive. The task message may include a line of
the form `BASELINE PRECISION: <token>` where `<token>` is one of
`double`, `float`, or `quad`. If absent, default to `double`. The
directive selects the storage precision of every floating-point
quantity the driver constructs that flows into the kernel (RHS of
every per-parameter `using <ParamName>Type = ...;` alias defined
below) AND the precision of the values written to reference.json.

  - `double`: aliases resolve to `double` / `Kokkos::View<double*>`;
    JSON values written with `"%.17g"` (the historical default).
  - `float`: aliases resolve to `float` / `Kokkos::View<float*>`;
    JSON values written with `"%.9g"`.
  - `quad`: aliases resolve to `__float128` /
    `Kokkos::View<__float128*>`; the driver `#include <quadmath.h>`
    and writes JSON values via `quadmath_snprintf(buf, sizeof(buf),
    "%.34Qg", value)` (NOT via `snprintf` / `<<`, which do not
    understand `__float128`). The compile step that follows expects
    `-lquadmath` on the link line; you do not emit the compile
    command, but the host-scratch helpers you write (RNG fill,
    deep_copy targets) must therefore use `__float128` end-to-end
    rather than down-converting through `double` mid-driver. Local
    `std::uniform_real_distribution<...>` returns `double` — convert
    explicitly with `static_cast<__float128>(...)` when storing into
    the alias-typed view.

The directive does NOT change the kernel function body. It changes
the alias RHSes (item 6 below), the JSON output formatting (item 8),
and any host-side scratch values that ultimately become kernel
arguments.

Hard requirements on the driver:

1. Single translation unit. Inline the kernel source verbatim into the
   driver (above main()). Do not introduce a build system or external
   headers beyond <Kokkos_Core.hpp>, the C and C++ standard library, and
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

2. Use Kokkos::initialize / Kokkos::finalize. Run on the serial host
   execution space (Kokkos::Serial / Kokkos::HostSpace). This is a v0
   reproducibility constraint: parallel reductions are order-dependent
   and would make the baseline non-deterministic.

3. Seed any RNG with a fixed integer. The driver must produce the
   same numbers on every run.

   The seed MUST be exposed as a single named C++ constant, written
   verbatim on its own line near the top of the driver (above the
   '// ---- KERNEL BEGIN ----' sentinel, so the seed declaration is
   NOT inside the splice region):

       static constexpr int RNG_SEED = 42;

   Use `42` unless the kernel's apparent domain demands otherwise. The
   declaration line must match this shape exactly (the tokens
   `static`, `constexpr`, `int`, `RNG_SEED`, `=`, the integer literal,
   `;`, in that order, separated by single spaces, no trailing
   comment) so a later probe-pipeline tool can deterministically find
   and replace the integer literal to re-run the driver at a different
   seed without touching the kernel body. Every other reference to the
   seed in the driver (RNG construction, the JSON 'seed' field) MUST
   read `RNG_SEED` rather than re-typing the integer.

4. Choose modest input sizes and distributions appropriate to the
   kernel from its signature and apparent scientific domain. Aim for a
   driver that runs in a few seconds, not hours. Typical N is in the
   1e4 to 1e6 range depending on per-element cost. Document the inputs
   you chose in inputs_summary.

5. If the task message names a TARGET KERNEL, call exactly that
   function. Otherwise, infer the kernel function from the source —
   there should be exactly one obvious candidate.

6. Do not modify the kernel function. Do not change any variable's
   precision. Do not invent or rename kernel arguments. The whole point
   is to capture the *original* kernel's output as the reference.

    EXCEPTION — precision-alias contract. Immediately inside the
   '// ---- KERNEL BEGIN ----' sentinel and above the kernel function
   definition, emit one `using` alias per kernel parameter whose type
   involves a floating-point scalar or a Kokkos::View of a floating-
   point scalar. Naming convention: `<ParamName>Type` (CamelCase of the
   parameter name + 'Type' suffix). The kernel function's parameter
   list MUST then refer to those aliases, not to the underlying types.
   The kernel body is otherwise unchanged. For example, a kernel
   originally declared as

       void kernel(
           Kokkos::View<double*> a,
           Kokkos::View<const double*> b,
           double alpha, double beta) { ... }

   becomes, inside the sentinels:

       using aType = Kokkos::View<double*>;
       using bType = Kokkos::View<const double*>;
       using alphaType = double;
       using betaType = double;

       void kernel(aType a, bType b, alphaType alpha, betaType beta) { ... }

   Integer parameters (sizes, counts, indices) do NOT get aliases —
   only floating-point scalars and floating-point Views. The kernel
   body itself stays byte-for-byte identical to the original; only the
   parameter type tokens in the function header are replaced with the
   alias names.

    `main()`, outside the sentinels, MUST use those aliases anywhere it
   constructs values that flow into the kernel call. Use the matching
   alias (e.g. `aType` for the `a` argument, `alphaType` for the
   `alpha` argument) to declare the values that flow into the kernel
   call. The aliases are the single point of truth for kernel I/O
   precision: a later rewriter redefines the aliases inside the
   sentinels to change kernel precision end-to-end, and `main()`
   inherits the change for free.

   Staging-view rule for `View<const T*>` kernel arguments. When a
   kernel parameter's alias is a const View (e.g.
   `using aType = Kokkos::View<const double*>;`), you cannot write into
   a view of that type directly to populate it. Use a writable staging
   view whose element type is DERIVED FROM THE ALIAS, then assign it
   into the const-aliased view that you pass to the kernel:

       // correct: staging view's element type tracks the alias
       Kokkos::View<typename aType::non_const_value_type*> a_nc("a", N);
       // ... fill a_nc through its host mirror ...
       aType a = a_nc;  // const-binds; precision is whatever aType says
       kernel(a, ...);

   Do NOT hardcode the staging view's element type as `double`. A
   hardcoded `Kokkos::View<double*> a_nc(...)` breaks the contract: when
   the rewriter redefines `aType` to `Kokkos::View<const float*>`, the
   `aType a = a_nc;` assignment no longer compiles because
   `View<double*>` does not convert to `View<const float*>`. The
   `typename <ParamName>Type::non_const_value_type` form is the only
   one that survives a precision change to the alias.

   For local host scratch (std::vector buffers, RNG distributions) that
   do not directly become kernel arguments, plain `double` is fine —
   only the values that cross the kernel boundary need to flow through
   the aliases.

7. Kokkos::deep_copy any device Views you read from back to host Views
   before iterating them for JSON emission.

8. Write the reference output to './reference.json' (relative to the
   driver's working directory) using std::ofstream. Floating-point
   formatting depends on the BASELINE PRECISION directive: `%.17g`
   for `double`, `%.9g` for `float`, `quadmath_snprintf("%.34Qg", ...)`
   for `quad`. Do NOT pull in a third-party JSON library — hand-roll
   the writer; output arrays are flat arrays of one floating-point
   type, so a few loops with manual braces, commas, and newlines are
   sufficient.

9. The JSON document must have exactly this shape:

       {
         "kernel": "<kernel_function_name>",
         "seed": <integer seed, value of RNG_SEED>,
         "inputs": { "N": <int>, ... },
         "outputs": { "<name>": [ <floating-point>, ... ], ... }
       }

   "inputs" carries enough metadata for a human reader to understand
   what the driver did (sizes, distributions if represented as
   strings). "outputs" carries one named flat array per output the
   comparator will check. The names under "outputs" must match
   output_arrays in your submit_result payload.

10. Begin the driver with a top-of-file comment that tells the operator
    to `cd` into the baseline directory (baselines/<file_stem>/) before
    running, so ./reference.json lands next to the driver source. Also
    mention the compile command in a comment (a typical Kokkos build
    line is fine; the operator will adapt it).

Set kernel_function_name and output_arrays in your submit_result
payload so they exactly match what the driver actually does. If your
driver writes an array under "outputs" by some name, that same name
must appear in output_arrays.

Return your result by calling the submit_result tool."""


KOKKOS_PROFILE = LanguageProfile(
    id="kokkos",
    display_name="Kokkos C++",
    source_suffixes=(".cpp",),
    driver_filename="driver.cpp",
    env_required=(ROOT_ENV,),
    dynamic_verification=True,
    baseline_harness_system_prompt=BASELINE_HARNESS_SYSTEM_PROMPT,
    baseline_harness_output_schema=BASELINE_HARNESS_OUTPUT_SCHEMA,
    build_compile_command=_build_compile_command,
    preflight=_preflight,
    detect_from_source=_detect_from_source,
    # v1 probe pipeline. The baseline is quad (highest available
    # precision via libquadmath) so the probe measures float / double
    # drift against true ground truth rather than against a same-or-
    # lower-precision reference. probe_precisions enumerates the
    # additional configurations the probe runs before invoking the
    # analyst; `mixed_io` keeps outputs at baseline precision but
    # downcasts intermediates to float, giving the analyst a coarse
    # signal on output-vs-intermediate sensitivity without per-variable
    # instrumentation. Kokkos is the only v1 profile with a populated
    # probe set; CUDA/HIP/SYCL/OMP-offload remain `probe_precisions=()`
    # until the deferred Commit 6 lands.
    baseline_precision="quad",
    probe_precisions=("quad", "double", "float", "mixed_io"),
)


__all__ = [
    "ROOT_ENV",
    "CXX",
    "CXX_STD",
    "OPT_FLAGS",
    "KOKKOS_LIBS",
    "EXTRA_LIBS",
    "BASELINE_HARNESS_SYSTEM_PROMPT",
    "BASELINE_HARNESS_OUTPUT_SCHEMA",
    "KOKKOS_PROFILE",
]
