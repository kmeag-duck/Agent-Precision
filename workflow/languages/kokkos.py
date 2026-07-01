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

# v1: detection token for the quad-precision probe path. When the
# baseline_harness's BASELINE PRECISION directive resolves to `quad`,
# the emitted driver source uses `__float128` and `quadmath_snprintf`,
# both of which require linking against libquadmath. We detect that by
# looking for the type name in the source — it cannot appear in a
# non-quad build because it is a GCC-specific extension keyword with
# no business in float or double drivers. A plain substring match is
# safe: the keyword is not used in any Kokkos / STL header the driver
# transitively includes (it would only appear after preprocessing,
# which this read does not do), and a stray `__float128` in a comment
# still warrants linking the lib (no harm, and the comment is
# vanishingly unlikely outside an intentional quad driver). The flag
# is appended AFTER the source file on the link line so GNU ld's
# left-to-right symbol resolution sees `__float128`-referencing
# symbols in the .o before it scans -lquadmath.
QUAD_PROBE_TOKEN = "__float128"
QUAD_LIB = "-lquadmath"


def _build_compile_command(driver_src: Path, driver_bin: Path) -> list[str]:
    """Assemble the g++ argv list for a Kokkos driver compile.

    Reads AGENT_PRECISION_KOKKOS_ROOT at call time (not import time) so
    a test that monkeypatches the env affects every subsequent compile
    in the same process. Assumes preflight has already verified the
    var is set and the directory layout looks Kokkos-shaped.

    Also peeks at `driver_src` to decide whether to append
    `-lquadmath` (the GNU libquadmath link) — needed when (and only
    when) the harness emitted a `__float128`/`quadmath_snprintf`
    driver for the v1 probe pipeline's `quad` baseline. The file is
    guaranteed to exist by this point (`_compile_driver` checks it
    above us); a read failure here would surface as a
    FileNotFoundError, same as it would from the subprocess attempt
    that follows, so we let it propagate rather than swallowing it.
    """
    root = Path(os.environ[ROOT_ENV])
    include_dir = root / "include"
    lib_dir = root / "lib"
    extra_libs: tuple[str, ...] = EXTRA_LIBS
    if QUAD_PROBE_TOKEN in driver_src.read_text():
        extra_libs = extra_libs + (QUAD_LIB,)
    return [
        CXX,
        CXX_STD,
        *OPT_FLAGS,
        f"-I{include_dir}",
        f"-L{lib_dir}",
        str(driver_src),
        *KOKKOS_LIBS,
        *extra_libs,
        "-o",
        str(driver_bin),
    ]


def _build_syntax_check_command(driver_src: Path) -> list[str] | None:
    """Assemble the g++ argv list for a syntax-only Kokkos driver check.

    Returns None when AGENT_PRECISION_KOKKOS_ROOT is unset or when the
    install prefix doesn't look Kokkos-shaped — the harness-validation
    gate then skips silently rather than failing every run on a host
    without a Kokkos install (validation is a quality improvement,
    not a hard requirement).

    The flag set is a strict subset of the real compile flags: no
    -L, no -l<lib>, no -o. `-fsyntax-only` stops after parsing +
    typechecking, so the linker is never invoked and the .o is never
    written. The include path IS required — without it, every
    <Kokkos_Core.hpp> include fails and the check false-negatives on
    every driver. The quad-driver's `__float128` is a GCC built-in
    (not a header dependency), so `-lquadmath` is irrelevant for a
    syntax check and deliberately omitted.

    -fopenmp stays in because Kokkos's OpenMP host backend uses
    `#pragma omp` directives that g++ warns about (and, with -Werror
    somewhere upstream, could fail) when the flag is missing.
    """
    kokkos_root = os.environ.get(ROOT_ENV)
    if not kokkos_root:
        return None
    root = Path(kokkos_root)
    include_dir = root / "include"
    if not include_dir.is_dir():
        return None
    return [
        CXX,
        CXX_STD,
        "-fsyntax-only",
        "-fopenmp",
        f"-I{include_dir}",
        str(driver_src),
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
        "drivers": {
            "type": "object",
            "description": (
                "Four self-contained .cpp driver translation units, one per "
                "probe precision. Each value is the full driver source as "
                "a single string; each MUST inline the same kernel source "
                "verbatim between the same splice sentinels, MUST declare "
                "the same `static constexpr int RNG_SEED = ...;` constant "
                "above the kernel begin sentinel, MUST share input sizes "
                "and output array shapes (the comparator requires shape-"
                "identical reference.json across precisions), and MUST "
                "compile against a standard Kokkos toolchain. The four "
                "drivers differ only in (a) the per-parameter "
                "`<ParamName>Type` alias RHSes inside the kernel sentinels, "
                "(b) any host-side scratch values that flow into the "
                "kernel, and (c) the reference.json floating-point "
                "formatting. The 'quad' driver is also the canonical "
                "baseline (its reference.json is the finish-gate "
                "ground truth); the other three feed the analyst probe "
                "pipeline."
            ),
            "properties": {
                "quad": {
                    "type": "string",
                    "description": (
                        "Driver with floating-point aliases resolved to "
                        "`__float128` and reference.json values written "
                        "via `quadmath_snprintf(buf, sizeof(buf), "
                        "\"%.34Qg\", value)`. Includes <quadmath.h>; the "
                        "compile step auto-links -lquadmath when the "
                        "source contains the token __float128."
                    ),
                },
                "double": {
                    "type": "string",
                    "description": (
                        "Driver with floating-point aliases resolved to "
                        "`double` and reference.json values written with "
                        "`\"%.17g\"`."
                    ),
                },
                "float": {
                    "type": "string",
                    "description": (
                        "Driver with floating-point aliases resolved to "
                        "`float` and reference.json values written with "
                        "`\"%.9g\"`."
                    ),
                },
                "mixed_io": {
                    "type": "string",
                    "description": (
                        "Driver in which kernel I/O aliases (the values "
                        "main() constructs and reads back) resolve to the "
                        "baseline precision (`__float128`), but any "
                        "intermediate buffers the kernel writes and then "
                        "reads internally (if visible from the kernel "
                        "signature as a distinct output-then-input View) "
                        "resolve to `float`. For kernels with no such "
                        "intermediate, this driver may be byte-identical "
                        "to the `quad` driver — emit it anyway. "
                        "reference.json formatting matches the I/O "
                        "precision (quad / quadmath_snprintf %.34Qg)."
                    ),
                },
            },
            "required": ["quad", "double", "float", "mixed_io"],
        },
        "kernel_function_name": {
            "type": "string",
            "description": (
                "Name of the kernel function the drivers call. Must match a "
                "function defined in the inlined kernel source. Shared "
                "across all four drivers."
            ),
        },
        "inputs_summary": {
            "type": "string",
            "description": (
                "One-line human-readable summary of the chosen inputs, "
                "e.g. 'N=16384, seed=42, x,y ~ U(-1,1)'. Mirrors the "
                "'inputs' block every driver writes into reference.json "
                "(shape-identical across precisions)."
            ),
        },
        "output_arrays": {
            "type": "array",
            "description": (
                "Names of the arrays each driver writes under the 'outputs' "
                "key of reference.json. The comparator uses this list to "
                "know which arrays to read back. Shared across all four "
                "drivers (the probe pipeline depends on shape-identical "
                "output layouts)."
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

You will be given a Kokkos C++ kernel source. Your job is to write a
self-contained C++ driver program that, when compiled and run later,
exercises the kernel on a fixed set of inputs and writes a reproducible
reference output to ./reference.json. That JSON file will eventually be
the baseline against which a rewritten (lower-precision) version of the
same kernel is compared.

You do NOT compile, run, or simulate the kernel. You do NOT invent
numerical output values. Your only output is the four driver sources.

PER-PRECISION DRIVERS. You emit FOUR drivers in a single
submit_result call under `drivers.{quad,double,float,mixed_io}`. All
four MUST share:

  - the same inlined kernel source (byte-identical between the
    `// ---- KERNEL BEGIN ----` and `// ---- KERNEL END ----` sentinels),
  - the same per-parameter `<ParamName>Type` alias NAMES (item 6),
  - the same `static constexpr int RNG_SEED = ...;` line (item 3),
  - the same input sizes, the same output array names, and the same
    output array lengths (the comparator requires shape-identical
    reference.json across precisions).

The four drivers differ ONLY in:

  - the RHS of each per-parameter `<ParamName>Type` alias,
  - any host-side scratch values that ultimately flow into the kernel
    (RNG fill loops, deep_copy targets, etc.) — these must be the
    appropriate precision end-to-end, not down-converted through
    `double` mid-driver,
  - the reference.json floating-point formatting (item 8).

Per-precision rules (apply to all four alias RHSes uniformly except
in `mixed_io`):

  - `double`: aliases resolve to `double` / `Kokkos::View<double*>`;
    JSON values written with `"%.17g"`. This driver is also the
    canonical splice scaffold (see SPLICE-TARGET ROLE below) — the
    rewriter will splice lower-precision kernels into a copy of it.
  - `float`: aliases resolve to `float` / `Kokkos::View<float*>`;
    JSON values written with `"%.9g"`.
  - `quad`: SPECIAL CASE — this driver does NOT use Kokkos at all.
    Kokkos's math intrinsics (`Kokkos::sqrt`, `Kokkos::sin`, …) have
    no `__float128` overload, so any kernel that calls a Kokkos math
    function inside a `KOKKOS_LAMBDA` is uncompilable at quad
    precision. Emit the quad driver as plain C++ that:
      * does NOT `#include <Kokkos_Core.hpp>`, does NOT call
        `Kokkos::initialize` / `Kokkos::finalize`, does NOT use
        `Kokkos::parallel_for` / `KOKKOS_LAMBDA` / `Kokkos::View`;
      * `#include <quadmath.h>` and uses `__float128` throughout;
        replace each `Kokkos::sqrt(x)` with `sqrtq(x)`, each
        `Kokkos::sin(x)` with `sinq(x)`, similarly for `cosq`,
        `expq`, `logq`, `fabsq`, `powq`, `atan2q`, etc.;
      * DOES NOT use the GNU `q` / `Q` numeric-literal suffix for
        `__float128` constants. C++23 disallows it as an extension
        and g++ rejects it under `-std=c++20` without
        `-fext-numeric-literals` (which the compile step does NOT
        pass). Write `__float128(0.0)`, `(__float128)1.5`, or
        `static_cast<__float128>(0.0)` instead of `0.0q` / `1.5q`.
        This includes ALL constants used inside the kernel body —
        accumulator initializers (`__float128 ax = __float128(0.0);`),
        scalar multipliers, comparison thresholds, anything that
        would otherwise be a bare floating literal;
      * replaces each `Kokkos::View<T*>` argument with a plain
        contiguous host buffer (`std::vector<__float128>` or
        `std::unique_ptr<__float128[]>`), accessed with `[]` instead
        of `()`. The aliases in this driver resolve to those host
        types (e.g. `using aType = std::vector<__float128>;`,
        `using alphaType = __float128;`);
      * replaces the kernel's `parallel_for(N, KOKKOS_LAMBDA(int i){
        ... })` loop body with a plain serial `for (int i = 0; i < N;
        ++i) { ... }` containing the SAME kernel body text (with
        Kokkos math calls swapped for their `q`-suffixed quadmath
        equivalents). The kernel function itself stays inside the
        sentinels and keeps its `<ParamName>Type` alias-typed
        signature — only the alias RHSes and the per-element math
        change;
      * writes JSON values via `quadmath_snprintf(buf, sizeof(buf),
        "%.34Qg", value)` (NOT via `snprintf` / `<<`, which do not
        understand `__float128`). The compile step auto-links
        `-lquadmath` when the driver source contains the token
        `__float128`. Local `std::uniform_real_distribution<...>`
        returns `double` — convert explicitly with
        `static_cast<__float128>(...)` when storing into the alias-
        typed buffer.
    The quad driver's job is purely to produce a ground-truth
    `reference.json` against which the (Kokkos-based) rewritten
    driver is later compared by the comparator; it is NEVER a
    splice target, so it does not need to share a Kokkos runtime
    with the other drivers.
  - `mixed_io`: aliases for kernel arguments that main() constructs
    AND reads back (the kernel's external I/O) resolve to `double`
    (matching the canonical splice-scaffold precision); JSON
    formatting matches (`"%.17g"`). The exception is any kernel
    parameter that is clearly an intermediate buffer (a View the
    kernel writes early and reads back later within the same kernel
    invocation, exposed in the signature only because Kokkos requires
    it) — those aliases resolve to `float`. If no such intermediate
    is identifiable from the kernel signature, this driver is
    byte-identical to the `double` driver; emit it anyway (the
    probe pipeline still consumes it).

SPLICE-TARGET ROLE. The orchestrator writes `drivers["double"]` (NOT
`drivers["quad"]`) to `baselines/<stem>/driver.cpp` as the canonical
splice scaffold. The rewriter later splices its kernel between the
sentinels of that file to produce `baselines/<stem>/rewritten/
driver.cpp`. The other three drivers (quad, float, mixed_io) live
only under `baselines/<stem>/probe/<precision>/` and feed the probe-
evidence pipeline. The quad driver additionally serves as the
ground-truth oracle: its `reference.json` from seed=42 is promoted
to `baselines/<stem>/reference.json` (overwriting whatever
`run_baseline_driver` wrote there from the double driver) so the
finish-gate comparator measures the rewritten kernel against true
quad ground truth, not against a same-or-lower-precision reference.

None of these driver variants changes the kernel function body
(except the `quad` driver, which rewrites Kokkos math calls to
quadmath equivalents and unrolls the parallel_for into a serial
host loop — see the quad bullet above). They change the alias RHSes
(item 6 below), any host-side scratch values that flow into the
kernel, and the JSON output formatting (item 8).

Hard requirements on the driver:

1. Single translation unit. Inline the kernel source verbatim into the
   driver (above main()). Do not introduce a build system or external
   headers beyond <Kokkos_Core.hpp>, the C and C++ standard library, and
   anything the kernel itself already includes. (Exception: the quad
   driver omits <Kokkos_Core.hpp> entirely and adds <quadmath.h>; see
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

2. Use Kokkos::initialize / Kokkos::finalize. Run on the serial host
   execution space (Kokkos::Serial / Kokkos::HostSpace). This is a v0
   reproducibility constraint: parallel reductions are order-dependent
   and would make the baseline non-deterministic. (Exception: the
   quad driver runs as plain C++ with a serial host `for` loop and
   omits Kokkos entirely — see the quad bullet in PER-PRECISION
   DRIVERS above. Reproducibility is satisfied by the serial loop
   alone.)

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
   before iterating them for JSON emission. (Not applicable to the
   quad driver, which uses plain host buffers and reads them
   directly.)

8. Each driver writes its reference output to './reference.json'
   (relative to its own working directory; the orchestrator will
   place each driver in a separate directory before running it) using
   std::ofstream. Floating-point formatting follows the per-precision
   rules above: `%.17g` for `double`, `%.9g` for `float`,
   `quadmath_snprintf("%.34Qg", ...)` for `quad` and `mixed_io`. Do
   NOT pull in a third-party JSON library — hand-roll the writer;
   output arrays are flat arrays of one floating-point type, so a few
   loops with manual braces, commas, and newlines are sufficient.

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
payload so they exactly match what the four drivers actually do
(shape-identical across precisions). If your drivers write an array
under "outputs" by some name, that same name must appear in
output_arrays.

Populate `drivers.quad`, `drivers.double`, `drivers.float`, and
`drivers.mixed_io` with the four full driver sources as described in
PER-PRECISION DRIVERS above. All four are required; an absent or
empty driver fails the schema.

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
    build_syntax_check_command=_build_syntax_check_command,
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
