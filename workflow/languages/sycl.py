"""SYCL C++ language profile.

SYCL is the open Khronos standard for single-source heterogeneous C++.
Unlike CUDA / HIP, SYCL kernels are lambdas submitted to a queue rather
than free `__global__` functions with `<<<...>>>` launch syntax. The
canonical compiler is Intel's `icpx -fsycl` (DPC++); AdaptiveCpp /
hipSYCL is the other major implementation but uses a different driver
(`acpp`) and is out of scope for v0.

The main differences from CUDA / HIP:

  - Kernels are lambdas inside a `q.submit([&](sycl::handler& h) {...})`
    block; there is no free-standing kernel function to alias.
  - Memory model: this profile mandates the `sycl::buffer` +
    `sycl::accessor` model rather than USM (`sycl::malloc_device`).
    Buffers are portable across all SYCL implementations and force a
    well-defined host-side synchronization point (`host_accessor` after
    `q.wait()`) which makes the baseline reproducible by construction.
  - Determinism: SYCL queues are out-of-order by default. This profile
    mandates `sycl::queue q{sycl::property::queue::in_order{}}` so the
    baseline driver behaves analogously to CUDA streams (in-order by
    default) and to Kokkos::Serial. Without this, two `parallel_for`
    submissions to the same queue could complete in either order and
    a kernel that reads its own previous output would race.
  - No GPU-arch environment variable. SYCL targets are selected at
    runtime via the device selector (`sycl::default_selector_v` /
    `sycl::gpu_selector_v`) and JIT-compiled by the runtime; an
    ahead-of-time `-fsycl-targets=...` knob exists but is deferred to
    a future smoke-validation phase. This is a deliberate departure
    from CUDA (`-arch=sm_89`) and HIP (`--offload-arch=gfx90a`).

The compiler binary is configurable via AGENT_PRECISION_SYCL_CXX
because the SYCL ecosystem is less monolithic than CUDA / HIP — some
sites use `icpx`, others `clang++ -fsycl`, others a vendored `dpcpp`
build. Defaulting to `icpx` covers the most common case; the env
override is a single string that names the compiler binary (the
`-fsycl` flag is always appended by `_build_compile_command`).

The precision-alias contract is structurally different from CUDA / HIP
because SYCL kernels are lambdas, not free functions. Instead of
aliasing function parameter types, the harness aliases the buffer
element types: `using aType = double; sycl::buffer<aType> a_buf(...);`
and the accessor inside the lambda inherits the alias via the buffer
template. The rewriter then redefines `aType` to `float` (or another
narrower type) and the change propagates through the buffer / accessor
/ kernel-body chain for free, exactly as it does for CUDA / HIP raw
pointers.

UNIT-TESTED, NOT SMOKE-VALIDATED. This profile was landed without an
end-to-end run against a real SYCL toolchain (icpx / DPC++ / oneAPI)
because no SYCL-capable host was available at implementation time. The
CUDA profile (which is much closer in shape to a free-function kernel
model) needed prompt iteration during its smoke test; expect MORE
iteration here because the lambda + accessor + alias contract is
substantially more involved than the raw-pointer model.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .base import LanguageProfile, make_error_result


# Environment variable that, when set, overrides the default SYCL
# compiler driver. Intentionally namespaced under the project's
# AGENT_PRECISION_* prefix so it does not collide with vendor knobs
# like CXX (which build systems already overload heavily) or with
# oneAPI's own SYCL_* variables (which configure the runtime, not the
# compiler).
CXX_ENV = "AGENT_PRECISION_SYCL_CXX"

# Default compiler binary. `icpx` is Intel's oneAPI DPC++ driver, the
# most widely deployed SYCL 2020 implementation as of the v0 timeline.
# Alternatives the operator may set CXX_ENV to include `clang++`
# (upstream DPC++ build), `dpcpp` (older oneAPI driver alias), or a
# fully-qualified path to a vendored SYCL compiler. The `-fsycl` flag
# is appended unconditionally by _build_compile_command — every known
# SYCL driver accepts it (it is the standard way to enable SYCL
# compilation across icpx / clang++ / dpcpp).
DEFAULT_CXX = "icpx"

# Compile flags. C++17 is the SYCL 2020 minimum; -O2 matches the other
# profiles' default optimization level. `-fsycl` is the flag every
# known SYCL driver uses to enable SYCL compilation; it is appended
# inside _build_compile_command rather than baked into OPT_FLAGS to
# keep the "SYCL-essential flags" and "performance flags" categories
# visually distinct.
CXX_STD = "-std=c++17"
OPT_FLAGS = ("-O2",)
SYCL_FLAG = "-fsycl"


def _resolve_compiler() -> str:
    """Return the SYCL compiler driver name, honoring CXX_ENV.

    Read at call time, not at import time, so a test that
    monkeypatches the env affects every subsequent compile in the
    same process. Returns the literal env value when set (no
    validation here — `_preflight` checks PATH separately so a typo
    surfaces as a "not found on PATH" diagnostic).
    """
    return os.environ.get(CXX_ENV, DEFAULT_CXX)


def _build_compile_command(driver_src: Path, driver_bin: Path) -> list[str]:
    """Assemble the SYCL compile argv list.

    Reads AGENT_PRECISION_SYCL_CXX at call time (not import time) so
    a test that monkeypatches the env affects every subsequent
    compile in the same process. Assumes preflight has already
    verified the chosen compiler is on PATH. No `-fsycl-targets=...`
    flag — SYCL device selection happens at runtime via the device
    selector, not at compile time, in v0.
    """
    cxx = _resolve_compiler()
    return [
        cxx,
        CXX_STD,
        *OPT_FLAGS,
        SYCL_FLAG,
        str(driver_src),
        "-o",
        str(driver_bin),
    ]


def _preflight() -> dict | None:
    """Verify the SYCL compiler is reachable before invoking it.

    Returns None when `shutil.which(<resolved compiler>)` finds the
    driver. Otherwise returns a make_error_result()-shaped dict the
    caller hands straight back to the orchestrator — no subprocess
    is spawned.

    AGENT_PRECISION_SYCL_CXX is intentionally NOT validated for
    "looks like a real SYCL compiler"; if the operator sets it to
    `gcc` by mistake, the compile itself will fail with a clear
    "unrecognized option '-fsycl'" error, which is a cleaner signal
    than a Python-side allowlist of SYCL driver names that would
    need to track every future compiler that adopts SYCL support.
    """
    cxx = _resolve_compiler()
    if shutil.which(cxx) is None:
        return make_error_result(
            f"{cxx} not found on PATH. Install a SYCL toolchain "
            f"(Intel oneAPI / DPC++ / AdaptiveCpp) and ensure {cxx} "
            f"is reachable, or set {CXX_ENV} to the driver binary "
            f"name on a host that has a SYCL toolchain."
        )
    return None


def _detect_from_source(kernel_source: str) -> bool:
    """Probe a `.cpp` source to decide whether it's SYCL.

    SYCL shares the `.cpp` suffix with Kokkos (and eventually with
    OpenMP-offload), so this probe IS consulted at runtime by
    `detect_language()` for `.cpp` inputs. It looks for any of three
    structural markers — all SYCL-specific tokens that no other
    profile's source uses:

      - `<sycl/sycl.hpp>` — the canonical SYCL 2020 unified header.
      - `sycl::queue` — every SYCL program constructs a queue.
      - `sycl::buffer` — present in any buffer/accessor-style program
        (the model this profile mandates for the baseline driver,
        and the most common pattern in scientific SYCL code).

    A stray "sycl" in a comment won't trigger any of these because
    each marker requires the `::` namespace qualifier or the
    angle-bracketed include path. The Kokkos probe is consulted
    before this one (insertion order in PROFILES); a `.cpp` source
    matching both probes (vanishingly unlikely in practice) is
    treated as Kokkos.
    """
    if "<sycl/sycl.hpp>" in kernel_source:
        return True
    if "sycl::queue" in kernel_source:
        return True
    if "sycl::buffer" in kernel_source:
        return True
    return False


# Per-language baseline-harness contract for SYCL. The output schema
# is structurally identical to the Kokkos / CUDA / HIP schemas (a
# self-contained driver source, the kernel function name it
# launches, an inputs summary string, and the list of output array
# names) — the comparator and splice tools downstream don't care
# which language produced the driver, so the JSON shape is shared.
# Only the descriptions are tweaked to say "SYCL" where the others
# say "Kokkos" / "CUDA" / "HIP", and to note that the "kernel
# function name" is really the wrapping function that holds the
# `q.submit([&](sycl::handler& h) {...})` block (SYCL kernels are
# lambdas; the splice operates on the wrapping function's body).
BASELINE_HARNESS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "driver_source": {
            "type": "string",
            "description": (
                "The full driver source as a single self-contained .cpp "
                "translation unit. Must inline the kernel source verbatim, "
                "compile with a SYCL driver (icpx -fsycl by default), and "
                "on execution write reference outputs to ./reference.json."
            ),
        },
        "kernel_function_name": {
            "type": "string",
            "description": (
                "Name of the wrapping function that contains the SYCL "
                "kernel submission (`q.submit([&](sycl::handler& h){...})`). "
                "SYCL kernels are lambdas, not free functions, so this is "
                "the host-side function the driver calls — typically the "
                "function originally given in the kernel source."
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

You will be given a SYCL C++ kernel source. Your job is to write a
self-contained SYCL driver program that, when compiled with a SYCL
driver (icpx -fsycl by default) and run later, exercises the kernel on
a fixed set of inputs and writes a reproducible reference output to
./reference.json. That JSON file will eventually be the baseline against
which a rewritten (lower-precision) version of the same kernel is
compared.

You do NOT compile, run, or simulate the kernel. You do NOT invent
numerical output values. Your only output is the driver source.

Hard requirements on the driver:

1. Single translation unit. Inline the kernel source verbatim into the
   driver (above main()). Do not introduce a build system or external
   headers beyond <sycl/sycl.hpp>, the C and C++ standard library, and
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
   determinism within a single device). Concretely:

   a. Construct the queue with the in-order property:

          sycl::queue q{sycl::property::queue::in_order{}};

       SYCL queues are out-of-order by default; without the in-order
       property, two submissions to the same queue may complete in
       either order and a kernel that depends on a prior submission's
       output would race. The in-order queue is what gives this driver
       a single, well-defined execution order.

   b. Do NOT use `sycl::atomic_ref` floating-point operations in the
      driver's reference computation. Order of atomic updates is
      hardware-scheduling-dependent and would make the reference
      non-reproducible. If the kernel under test uses atomics, that
      is a kernel design choice and is judged by the rewriter /
      verifier separately; the DRIVER must not introduce its own.

   c. Launch the kernel with a single, fixed nd-range or range
      derived deterministically from the chosen input size (e.g.
      `sycl::range<1>{N}` or `sycl::nd_range<1>{N, 256}`). Do not
      query the device for an "optimal" work-group size.

   d. Seed any host-side RNG with a fixed integer (use 42 unless the
      kernel's apparent domain demands otherwise). The driver must
      produce the same numbers on every run.

3. Use the buffer + accessor memory model, NOT USM. Allocate host
   data in `std::vector`, wrap it in `sycl::buffer<T>` for the
   submission, and acquire `sycl::accessor` instances inside the
   `q.submit([&](sycl::handler& h){...})` block. After the kernel
   completes, read outputs back via `sycl::host_accessor` (which
   blocks until the kernel finishes) before iterating for JSON
   emission. A typical pattern:

       std::vector<double> host_a(N), host_c(N);
       // ... fill host_a ...
       {
         sycl::buffer<double> a_buf(host_a.data(), sycl::range<1>{N});
         sycl::buffer<double> c_buf(host_c.data(), sycl::range<1>{N});
         q.submit([&](sycl::handler& h) {
           sycl::accessor a{a_buf, h, sycl::read_only};
           sycl::accessor c{c_buf, h, sycl::write_only, sycl::no_init};
           h.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) {
             c[i] = /* kernel body */;
           });
         });
         q.wait();
         sycl::host_accessor c_host{c_buf, sycl::read_only};
         for (size_t i = 0; i < N; ++i) host_c[i] = c_host[i];
       }

   Wrap any throwing SYCL operations in try/catch for
   `sycl::exception` and abort with a clear stderr message on
   failure. SYCL surfaces device errors through exceptions, not
   error codes; an uncaught exception terminates with a confusing
   `std::terminate` rather than a useful diagnostic.

4. Choose modest input sizes and distributions appropriate to the
   kernel from its signature and apparent scientific domain. Aim for
   a driver that runs in a few seconds on a single device, not hours.
   Typical N is in the 1e4 to 1e7 range depending on per-element cost.
   Document the inputs you chose in inputs_summary.

5. If the task message names a TARGET KERNEL, wrap the SYCL
   submission for that kernel. Otherwise, infer the kernel function
   from the source — there should be exactly one obvious candidate.

6. Do not modify the kernel function. Do not change any variable's
   precision. Do not invent or rename kernel arguments. The whole
   point is to capture the *original* kernel's output as the
   reference.

    EXCEPTION — precision-alias contract. Immediately inside the
   '// ---- KERNEL BEGIN ----' sentinel and above the kernel
   function definition, emit one `using` alias per floating-point
   buffer element type the kernel touches. SYCL kernels are
   lambdas (not free functions), so the alias attaches to the
   buffer element type rather than to function parameters. Naming
   convention: `<BufferName>Type` (CamelCase of the buffer name +
   'Type' suffix). The buffer declarations, accessor declarations
   inside the submission, and any explicit casts in the kernel body
   MUST then refer to those aliases, not to the underlying types.
   For example, a kernel originally using

       sycl::buffer<double> a_buf(host_a.data(), sycl::range<1>{N});
       sycl::buffer<double> c_buf(host_c.data(), sycl::range<1>{N});

   becomes, inside the sentinels:

       using aType = double;
       using cType = double;

       sycl::buffer<aType> a_buf(host_a.data(), sycl::range<1>{N});
       sycl::buffer<cType> c_buf(host_c.data(), sycl::range<1>{N});

   The kernel body inside the parallel_for lambda stays
   byte-for-byte identical; the accessor types propagate from
   `sycl::buffer<aType>` through `sycl::accessor a{a_buf, ...}`
   automatically, so the body's `a[i]` already has the aliased
   element type.

   Integer buffers (index arrays, indirection tables) do NOT get
   aliases — only floating-point element types. Scalar kernel
   arguments captured by the lambda (e.g. a scalar `alpha`) also
   get an alias when floating-point:

       using alphaType = double;
       alphaType alpha = 1.5;
       q.submit([&](sycl::handler& h) {
         sycl::accessor a{a_buf, h, sycl::read_only};
         sycl::accessor c{c_buf, h, sycl::write_only};
         h.parallel_for(sycl::range<1>{N}, [=, alpha](sycl::id<1> i) {
           c[i] = alpha * a[i];
         });
       });

   Staging-buffer rule for read-only buffers backed by host data.
   When you wrap a host `std::vector` in a `sycl::buffer<aType>`,
   the vector's element type must match `aType` exactly so that
   redefining `aType` to `float` does not break the wrap. Declare
   the host vector through the alias too:

       std::vector<aType> host_a(N);
       // ... fill host_a ...
       sycl::buffer<aType> a_buf(host_a.data(), sycl::range<1>{N});

   Do NOT hardcode the host vector as `std::vector<double>` when
   its buffer's alias is `aType`. A hardcoded
   `std::vector<double> host_a(N); sycl::buffer<aType> a_buf(host_a.data(), ...)`
   breaks the contract: when the rewriter redefines `aType` to
   `float`, the buffer constructor sees a `double*` host pointer and
   a `float` element type and either silently reinterprets the bytes
   or fails to compile depending on SYCL implementation.

   For local host scratch (RNG distributions, intermediate
   computations) that do NOT back a SYCL buffer, plain `double` is
   fine — only the values that cross into a SYCL buffer need to
   flow through the aliases.

7. After `q.wait()`, acquire `sycl::host_accessor` instances for
   every output buffer you intend to record, and iterate them for
   JSON emission. `host_accessor` blocks until the device work
   completes and synchronizes the buffer's host backing — that is
   how this driver guarantees the JSON writer sees the kernel's
   completed output.

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
    `icpx -fsycl -std=c++17 -O2 driver.cpp -o driver` build line is
    fine; the operator will adapt it).

Set kernel_function_name and output_arrays in your submit_result
payload so they exactly match what the driver actually does. If your
driver writes an array under "outputs" by some name, that same name
must appear in output_arrays.

Return your result by calling the submit_result tool."""


SYCL_PROFILE = LanguageProfile(
    id="sycl",
    display_name="SYCL C++",
    source_suffixes=(".cpp",),
    driver_filename="driver.cpp",
    env_required=(),  # CXX_ENV is optional; compiler presence is checked in preflight.
    dynamic_verification=True,
    baseline_harness_system_prompt=BASELINE_HARNESS_SYSTEM_PROMPT,
    baseline_harness_output_schema=BASELINE_HARNESS_OUTPUT_SCHEMA,
    build_compile_command=_build_compile_command,
    preflight=_preflight,
    detect_from_source=_detect_from_source,
)


__all__ = [
    "CXX_ENV",
    "DEFAULT_CXX",
    "CXX_STD",
    "OPT_FLAGS",
    "SYCL_FLAG",
    "BASELINE_HARNESS_SYSTEM_PROMPT",
    "BASELINE_HARNESS_OUTPUT_SCHEMA",
    "SYCL_PROFILE",
]
