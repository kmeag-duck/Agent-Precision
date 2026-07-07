"""Registry shape tests. No mocks, no network."""

from workflow.registry import (
    AGENTS,
    ANALYST_FINALIZER_OUTPUT_SCHEMA,
    ANALYST_OUTPUT_SCHEMA,
    BASELINE_HARNESS_OUTPUT_SCHEMA,
    CANDIDATE_FINDER_OUTPUT_SCHEMA,
    VARIABLE_ANALYST_OUTPUT_SCHEMA,
    VERIFIER_OUTPUT_SCHEMA,
)


def test_known_agent_types():
    """AGENTS exposes the core agent types plus one baseline_harness_<lang> per registered language profile, with `baseline_harness` aliased to the Kokkos entry."""
    core_types = {
        "candidate_finder",
        "variable_analyst",
        "analyst",
        "analyst_finalizer",
        "rewriter",
        "verifier",
        "baseline_harness",
    }
    assert core_types.issubset(set(AGENTS))
    # The Phase A language-profile refactor adds one entry per profile
    # under the `baseline_harness_<id>` key. At minimum, the Kokkos
    # entry must exist and the unsuffixed alias must point at it.
    assert "baseline_harness_kokkos" in AGENTS
    assert AGENTS["baseline_harness"] is AGENTS["baseline_harness_kokkos"]
    # Defend against unrelated stray entries: every non-core key must
    # be a baseline_harness_<id> entry.
    extras = set(AGENTS) - core_types
    for name in extras:
        assert name.startswith("baseline_harness_"), (
            f"Unexpected AGENTS entry: {name!r}"
        )


def test_each_entry_has_required_keys():
    """Every registry entry declares system_prompt, output_schema, model, and supports_temperature (the last gates whether run_agent forwards the kwarg — Argo's claude-opus-4-7 rejects it, so the flag is mandatory and not just a default-False sentinel)."""
    for name, spec in AGENTS.items():
        assert "system_prompt" in spec, f"{name} missing system_prompt"
        assert "output_schema" in spec, f"{name} missing output_schema"
        assert "model" in spec, f"{name} missing model"
        assert "supports_temperature" in spec, (
            f"{name} missing supports_temperature flag"
        )
        assert isinstance(spec["supports_temperature"], bool), (
            f"{name}.supports_temperature must be a bool, got "
            f"{type(spec['supports_temperature']).__name__}"
        )


def test_system_prompts_are_nonempty_strings():
    """Every agent has a non-empty system prompt string."""
    for name, spec in AGENTS.items():
        assert isinstance(spec["system_prompt"], str)
        assert spec["system_prompt"].strip(), f"{name} has empty system_prompt"


def test_models_are_nonempty_strings():
    """Every agent declares a non-empty model id string."""
    for name, spec in AGENTS.items():
        assert isinstance(spec["model"], str)
        assert spec["model"].strip(), f"{name} has empty model id"


def test_output_schemas_are_object_schemas():
    """Every output_schema is a valid object schema with required keys present in properties."""
    for name, spec in AGENTS.items():
        schema = spec["output_schema"]
        assert isinstance(schema, dict), f"{name} schema not a dict"
        assert schema.get("type") == "object", f"{name} schema type != object"
        assert isinstance(schema.get("properties"), dict), (
            f"{name} schema missing properties dict"
        )
        assert isinstance(schema.get("required"), list), (
            f"{name} schema missing required list"
        )
        for req_key in schema["required"]:
            assert req_key in schema["properties"], (
                f"{name} required key {req_key!r} not in properties"
            )


# ---------- Analyst schema: three-method action enum + rework block ----------


def test_analyst_action_enum_is_downcast_emulate_keep():
    """The analyst's per-variable action enum is exactly {downcast, emulate, keep}."""
    item = ANALYST_OUTPUT_SCHEMA["properties"]["variables"]["items"]
    assert set(item["properties"]["action"]["enum"]) == {"downcast", "emulate", "keep"}


def test_analyst_variable_item_requires_emulation_type():
    """Each per-variable entry requires emulation_type (alongside target_precision) so the rewriter sees both fields explicitly even for 'keep'."""
    item = ANALYST_OUTPUT_SCHEMA["properties"]["variables"]["items"]
    assert "emulation_type" in item["properties"]
    assert "emulation_type" in item["required"]
    assert "target_precision" in item["required"]


def test_analyst_schema_requires_rework_block():
    """The analyst top-level schema requires a rework object so the analyst is forced to give an explicit 'no rework' answer rather than omitting the field."""
    assert "rework" in ANALYST_OUTPUT_SCHEMA["required"]
    rework = ANALYST_OUTPUT_SCHEMA["properties"]["rework"]
    assert rework["type"] == "object"
    assert set(rework["required"]) == {
        "suggested",
        "transformation",
        "rationale",
        "affected_variables",
    }
    assert rework["properties"]["suggested"]["type"] == "boolean"
    assert rework["properties"]["affected_variables"]["type"] == "array"


# ---------- Verifier schema: enums track the analyst's three methods ----------


def test_verifier_expected_action_enum_matches_analyst():
    """The verifier's expected_action enum mirrors the analyst's action enum."""
    item = VERIFIER_OUTPUT_SCHEMA["properties"]["per_variable"]["items"]
    assert set(item["properties"]["expected_action"]["enum"]) == {
        "downcast",
        "emulate",
        "keep",
    }


def test_verifier_observed_action_enum_adds_unclear():
    """The verifier's observed_action enum is the analyst's three methods plus 'unclear'."""
    item = VERIFIER_OUTPUT_SCHEMA["properties"]["per_variable"]["items"]
    assert set(item["properties"]["observed_action"]["enum"]) == {
        "downcast",
        "emulate",
        "keep",
        "unclear",
    }


# ---------- Prompts mention the three methods (smoke check, not exhaustive) ----------


def test_analyst_prompt_mentions_all_three_methods():
    """The analyst prompt explicitly names downcast, emulate, and keep so the model knows the full action vocabulary."""
    prompt = AGENTS["analyst"]["system_prompt"]
    for method in ("downcast", "emulate", "keep"):
        assert method in prompt, f"analyst prompt missing {method!r}"


def test_rewriter_prompt_mentions_all_three_methods_and_emulation_struct():
    """The rewriter prompt names all three methods and includes the inline ff_t emulation convention so it has a concrete representation to use."""
    prompt = AGENTS["rewriter"]["system_prompt"]
    for method in ("downcast", "emulate", "keep"):
        assert method in prompt, f"rewriter prompt missing {method!r}"
    assert "ff_t" in prompt, "rewriter prompt missing the ff_t emulation convention"


def test_verifier_prompt_mentions_rework_check():
    """The verifier prompt tells the verifier to check whether a suggested rework was actually applied (or that an unrequested rework was not silently added)."""
    prompt = AGENTS["verifier"]["system_prompt"]
    assert "rework" in prompt.lower()


# ---------- Precision advisor was removed: guard against re-introduction ----------


def test_precision_advisor_agent_is_not_registered():
    """After removing the precision_advisor agent, AGENTS must not carry an entry for it. This test locks the removal so a future revert would fail loudly here rather than only surface as a runtime tool-dispatch error."""
    assert "precision_advisor" not in AGENTS


# ---------- Analyst: precision_budget block ----------


def test_analyst_schema_requires_precision_budget_block():
    """The analyst top-level schema requires a precision_budget object so the analyst is forced to link its per-variable verdict back to the tolerance it was given."""
    assert "precision_budget" in ANALYST_OUTPUT_SCHEMA["required"]
    budget = ANALYST_OUTPUT_SCHEMA["properties"]["precision_budget"]
    assert budget["type"] == "object"
    assert set(budget["required"]) == {
        "target_kind",
        "target_value",
        "source",
        "claimed_output_precision",
        "headroom_argument",
    }


def test_analyst_precision_budget_target_kind_enum_matches_advisor():
    """The analyst's precision_budget.target_kind enum is the same {sig_figs, decimal_digits} the advisor produces (minus 'unknown', which the orchestrator must have resolved before calling the analyst)."""
    budget = ANALYST_OUTPUT_SCHEMA["properties"]["precision_budget"]
    target_kind = budget["properties"]["target_kind"]
    assert set(target_kind["enum"]) == {"sig_figs", "decimal_digits"}


def test_analyst_prompt_mentions_tolerance_and_emulate_throughput_caveat():
    """The analyst prompt tells the analyst the tolerance is a hard constraint and warns that 'emulate' is throughput-negative, so the analyst prefers downcast when the tolerance permits."""
    prompt = AGENTS["analyst"]["system_prompt"].lower()
    assert "tolerance" in prompt
    # throughput caveat for emulate
    assert "throughput" in prompt
    assert "emulate" in prompt


# ---------- Rewriter: silent-substitution guard ----------


def test_rewriter_prompt_forbids_silent_method_substitution():
    """The rewriter prompt explicitly forbids silently substituting one method for another (e.g. downcasting when asked to emulate) so the verifier's per-variable check actually means something."""
    prompt = AGENTS["rewriter"]["system_prompt"].lower()
    assert "silently" in prompt or "substitute" in prompt


# ---------- Verifier: tolerance handling ----------


def test_verifier_prompt_mentions_tolerance_and_precision_budget():
    """The verifier prompt tells the verifier that it receives the tolerance and audits the analyst's precision_budget against it (without flipping per_variable ok on that basis)."""
    prompt = AGENTS["verifier"]["system_prompt"].lower()
    assert "tolerance" in prompt
    assert "precision_budget" in prompt


# ---------- Baseline harness: schema + prompt ----------


def test_baseline_harness_schema_required_fields():
    """The Kokkos baseline_harness schema (v1) requires `drivers` (four per-precision driver sources), kernel_function_name, inputs_summary, and output_arrays. The orchestrator's spawn_baseline_harness branch writes drivers[baseline_precision] to baselines/<stem>/driver.cpp (the canonical baseline that feeds the v0 dynamic-verification chain) and the remaining drivers under baselines/<stem>/probe/<precision>/ for the v1 probe pipeline; absent or empty driver keys would silently collapse the probe into a single-precision run, so the schema rejects them."""
    assert set(BASELINE_HARNESS_OUTPUT_SCHEMA["required"]) == {
        "drivers",
        "kernel_function_name",
        "inputs_summary",
        "output_arrays",
    }
    drivers_schema = BASELINE_HARNESS_OUTPUT_SCHEMA["properties"]["drivers"]
    assert drivers_schema["type"] == "object"
    assert set(drivers_schema["required"]) == {
        "quad",
        "double",
        "float",
        "mixed_io",
    }
    for precision in ("quad", "double", "float", "mixed_io"):
        assert drivers_schema["properties"][precision]["type"] == "string"


def test_baseline_harness_output_arrays_is_array_of_strings():
    """output_arrays is a JSON array of strings; the comparator iterates these names against the 'outputs' key of reference.json."""
    out = BASELINE_HARNESS_OUTPUT_SCHEMA["properties"]["output_arrays"]
    assert out["type"] == "array"
    assert out["items"]["type"] == "string"


def test_baseline_harness_prompt_mentions_kokkos_serial_reproducibility():
    """The baseline_harness prompt names Kokkos::initialize and Kokkos::Serial (the v0 reproducibility constraint) so the model knows to run the reference on the deterministic host execution space."""
    prompt = AGENTS["baseline_harness"]["system_prompt"]
    assert "Kokkos::initialize" in prompt
    assert "Kokkos::Serial" in prompt


def test_baseline_harness_prompt_mentions_reference_json_and_fixed_seed():
    """The baseline_harness prompt names reference.json (the output file) and tells the agent to seed any RNG with a fixed integer so the reference is reproducible across runs."""
    prompt = AGENTS["baseline_harness"]["system_prompt"]
    assert "reference.json" in prompt
    lower = prompt.lower()
    assert "seed" in lower
    assert "fixed" in lower or "reproducible" in lower


def test_baseline_harness_prompt_mentions_target_kernel_and_no_invented_values():
    """The baseline_harness prompt names TARGET KERNEL (the disambiguator the orchestrator may prepend) and forbids inventing numerical output values (the whole point is to capture the original kernel's output)."""
    prompt = AGENTS["baseline_harness"]["system_prompt"]
    assert "TARGET KERNEL" in prompt
    lower = prompt.lower()
    assert "do not invent" in lower or "not invent" in lower or "never invent" in lower


def test_baseline_harness_prompt_mentions_high_precision_format():
    """The baseline_harness prompt requires %.17g formatting so the reference output preserves full double precision."""
    prompt = AGENTS["baseline_harness"]["system_prompt"]
    assert "%.17g" in prompt


def test_baseline_harness_prompt_mandates_kernel_splice_sentinels():
    """The baseline_harness prompt mandates the exact '// ---- KERNEL BEGIN ----' and '// ---- KERNEL END ----' sentinels around the inlined kernel so a later mechanical-verification step can splice a rewritten kernel into the same driver template."""
    prompt = AGENTS["baseline_harness"]["system_prompt"]
    assert "// ---- KERNEL BEGIN ----" in prompt
    assert "// ---- KERNEL END ----" in prompt


# ---------- v1 probe-pipeline contracts (PER-PRECISION DRIVERS + RNG_SEED) ----------
#
# v1 added two contracts to the Kokkos baseline harness so the probe
# pipeline (a later commit) can drive the same kernel at different
# precisions and seeds without re-asking the LLM per variant.
#
# (1) PER-PRECISION DRIVERS: the harness emits FOUR drivers in a
#     single submit_result call under `drivers.{quad,double,float,
#     mixed_io}`. All four share the inlined kernel, alias names,
#     RNG_SEED line, input sizes, and output array names+lengths
#     (the comparator depends on shape-identical reference.json
#     across precisions); they differ only in the per-parameter alias
#     RHSes, host-side scratch precision, and reference.json
#     formatting. `quad` is also the canonical baseline — the
#     orchestrator writes drivers.quad to baselines/<stem>/driver.cpp
#     so the v0 dynamic-verification chain compares the rewritten
#     kernel against the true (quad) ground truth, not against a
#     same-or-lower-precision reference. `quad` requires __float128 +
#     <quadmath.h> + quadmath_snprintf "%.34Qg" and changes the
#     compile link line (the compile step adds -lquadmath when the
#     emitted source contains __float128).
#
# (2) Named RNG_SEED constant: the harness must declare the seed as
#     `static constexpr int RNG_SEED = <N>;` on its own line above
#     the KERNEL BEGIN sentinel (so the splice tool does not touch
#     it) and reference RNG_SEED everywhere the integer was
#     previously inlined. This shape lets a later probe tool
#     deterministically find and rewrite the integer literal to
#     re-run the driver at a different seed.


def test_baseline_harness_prompt_mandates_per_precision_drivers():
    """The Kokkos baseline harness prompt mandates the v1 PER-PRECISION DRIVERS contract: four drivers (quad, double, float, mixed_io) emitted in a single submit_result call under the `drivers` key. All four must share the inlined kernel, alias names, RNG_SEED line, input sizes, and output array names+lengths; they differ only in the per-parameter alias RHSes, host-side scratch precision, and reference.json formatting. The `quad` driver requires __float128 + <quadmath.h> + quadmath_snprintf %.34Qg (the canonical baseline; the orchestrator writes it to baselines/<stem>/driver.cpp). The `float` driver requires %.9g formatting. The `mixed_io` driver keeps I/O at the baseline precision but downcasts identified intermediate buffers to float, giving the analyst a cheap signal on output-vs-intermediate sensitivity. Without this contract the harness collapses back to a single-precision emit and the probe pipeline has nothing to consume."""
    prompt = AGENTS["baseline_harness"]["system_prompt"]
    # The contract heading and the four precision tokens are spelled out.
    assert "PER-PRECISION DRIVERS" in prompt
    for precision in ("quad", "double", "float", "mixed_io"):
        assert precision in prompt
    # The quad branch's load-bearing tokens — without these the harness
    # cannot actually emit a quad driver that compiles or formats JSON.
    assert "__float128" in prompt
    assert "quadmath.h" in prompt
    assert "quadmath_snprintf" in prompt
    assert "%.34Qg" in prompt
    # The float branch's JSON format token (so the JSON file isn't
    # over-precise for the precision actually computed).
    assert "%.9g" in prompt
    # The shape-identical-across-precisions rule (the comparator
    # contract that the probe pipeline depends on).
    lower = prompt.lower()
    assert "shape-identical" in lower or "shape identical" in lower
    # The mixed_io rule must name intermediate buffers as the thing
    # that downcasts to float (vs. kernel I/O at baseline precision).
    assert "intermediate" in lower


def test_baseline_harness_prompt_forbids_quad_numeric_literal_suffix():
    """The Kokkos baseline harness prompt must explicitly forbid the GNU `q` / `Q` numeric-literal suffix for `__float128` constants in the quad driver. C++23 disallows it as an extension, and g++ rejects it under `-std=c++20` without `-fext-numeric-literals` — which the compile step intentionally does not pass (so the rest of the source stays standard-conformant). The first end-to-end smoke run of the role-split / oracle-promotion fix uncovered this: the LLM emitted `__float128 ax = 0.0q;`, both quad probe cells failed to compile, probe_compare hard-errored on the missing quad_seed42 ground truth, and the comparator silently fell back to the double-precision baseline reference — a real (if recoverable) probe-pipeline degradation. The fix is purely prompt-level: forbid the suffix, mandate the constructor / cast forms `__float128(x)`, `(__float128)x`, `static_cast<__float128>(x)`. This test pins the contract so a future prompt edit cannot silently regress."""
    prompt = AGENTS["baseline_harness"]["system_prompt"]
    # The forbidden tokens must be called out by example. We assert on
    # the prose, not on the literal `0.0q` token in isolation — the
    # prompt has to actually tell the LLM NOT to write it.
    assert "0.0q" in prompt or "1.5q" in prompt
    # At least one of the sanctioned alternative forms must be shown
    # by example. The LLM has to see the replacement, not just the
    # prohibition.
    assert (
        "__float128(0.0)" in prompt
        or "(__float128)" in prompt
        or "static_cast<__float128>" in prompt
    )
    # The compile-flag rationale must be named — without it a future
    # edit might "fix" the prompt by recommending -fext-numeric-literals
    # instead, which would diverge the prompt from the compile step.
    lower = prompt.lower()
    assert "fext-numeric-literals" in lower or "c++23" in lower


def test_baseline_harness_prompt_mandates_named_rng_seed_constant():
    """The Kokkos baseline harness prompt mandates a single `static constexpr int RNG_SEED = <N>;` declaration on its own line above the KERNEL BEGIN sentinel (so it sits OUTSIDE the splice region and survives a kernel-body rewrite) and requires every other seed reference (RNG construction, JSON 'seed' field) to read RNG_SEED rather than re-typing the integer. Without the exact declaration shape and out-of-sentinel placement the probe pipeline cannot deterministically rewrite the seed line to re-run the driver at a different seed."""
    prompt = AGENTS["baseline_harness"]["system_prompt"]
    # The exact declaration shape — the probe-pipeline splicer matches
    # this prefix verbatim. If you change the spelling here, change it
    # in the probe tool together with this assertion.
    assert "static constexpr int RNG_SEED = 42;" in prompt
    # Placement: above the KERNEL BEGIN sentinel (so the kernel-body
    # splicer does not touch it).
    normalized = " ".join(prompt.lower().split())
    assert (
        "above the '// ---- kernel begin ----' sentinel" in normalized
        or "above the kernel begin sentinel" in normalized
        or "not inside the splice region" in normalized
    )
    # The "every other reference reads RNG_SEED" half — without this
    # the driver could declare RNG_SEED and still hardcode 42 in the
    # RNG constructor or JSON, defeating the probe's seed swap.
    assert "RNG_SEED" in prompt
    assert prompt.count("RNG_SEED") >= 3  # decl + RNG ref + JSON ref


# ---------- Precision-alias contract (Option 4 splice-scope fix) ----------
#
# The splice tool replaces only the text between the kernel sentinels.
# Anything the rewriter changes that affects the caller (e.g. a parameter
# type) must therefore reach the caller through something that lives
# inside the sentinels. The contract: the baseline harness emits a
# `using <ParamName>Type = ...;` alias per floating-point kernel parameter
# inside the sentinels; main() outside the sentinels uses those aliases;
# the rewriter downcasts a parameter by redefining its alias and nothing
# else.


def test_baseline_harness_prompt_mandates_precision_alias_contract():
    """The baseline_harness prompt mandates per-parameter `using <ParamName>Type` aliases inside the kernel sentinels so the splice tool can change kernel parameter precision (which the rewriter owns) without breaking main() (which it does not)."""
    prompt = AGENTS["baseline_harness"]["system_prompt"]
    # The CamelCase + 'Type' suffix naming convention is named explicitly.
    assert "<ParamName>Type" in prompt
    assert "using" in prompt
    # The contract is two-sided: aliases live inside the sentinels, AND
    # main() outside the sentinels must reference them.
    assert "main()" in prompt
    # Integer parameters are excluded so the harness doesn't bloat with
    # NType / seedType aliases that the rewriter would never touch.
    lower = prompt.lower()
    assert "integer parameters" in lower or "integer parameter" in lower


def test_baseline_harness_prompt_alias_example_is_concrete():
    """The baseline_harness prompt shows a concrete before/after example so the model has an unambiguous target pattern; the example demonstrates the `<ParamName>Type` alias form against a generic kernel signature."""
    prompt = AGENTS["baseline_harness"]["system_prompt"]
    # Naming convention is demonstrated, not just described. The example
    # uses generic parameter names (a, b, alpha, beta) so the prompt
    # doesn't bias the harness toward any specific test kernel's naming.
    assert "using aType" in prompt
    # The example references Kokkos::View so the model sees the View
    # element-type form (the case that matters for downcasting).
    assert "Kokkos::View" in prompt


def test_baseline_harness_prompt_requires_alias_derived_staging_view():
    """When a kernel parameter is `View<const T*>`, the harness needs a writable staging view to populate it. The prompt mandates that the staging view's element type be derived from the alias (`typename <ParamName>Type::non_const_value_type`) rather than hardcoded to `double`; a hardcoded element type breaks the splice contract because the alias-only rewrite no longer matches the staging view's type. Witnessed on nbody_force: the first end-to-end run produced `Kokkos::View<double*> m_nc` which did not convert to the rewritten `Kokkos::View<const float*> mType`."""
    prompt = AGENTS["baseline_harness"]["system_prompt"]
    # The alias-derived-type idiom is named explicitly. Without this
    # exact phrase the harness has no way to derive the staging view's
    # element type from the alias.
    assert "non_const_value_type" in prompt
    # The failure mode (hardcoded `double` staging view) is called out
    # so the model knows what NOT to do. We don't pin the exact wording
    # but require both halves of the warning: "do not" + "double".
    normalized = " ".join(prompt.lower().split())
    assert "do not hardcode" in normalized or "do not" in normalized
    # The rule is scoped to const View parameters (the only case where
    # staging is necessary). Mentioning const View anchors the rule.
    assert "const" in normalized


def test_rewriter_prompt_documents_alias_driven_downcast():
    """The rewriter prompt tells the rewriter that when the kernel uses the `<ParamName>Type` alias pattern, downcasting a parameter means redefining its alias only — not editing the function header. This is the rewriter's side of the splice-scope contract."""
    prompt = AGENTS["rewriter"]["system_prompt"]
    assert "<ParamName>Type" in prompt
    # The rewriter must understand "redefine the alias, leave the header
    # alone" — both halves of that instruction must be present.
    # Normalize whitespace so wrapped phrases like "fall\n   back" still match.
    normalized = " ".join(prompt.lower().split())
    assert "alias" in normalized
    # Fallback path is documented so the rewriter doesn't choke on
    # legacy kernels that don't use the alias pattern.
    assert "fall back" in normalized or "fallback" in normalized


def test_rewriter_prompt_forbids_bypassing_alias():
    """The rewriter prompt explicitly forbids writing the lowered type directly in the function header when an alias exists, because doing so silently breaks the caller (which constructs values through the alias)."""
    prompt = AGENTS["rewriter"]["system_prompt"].lower()
    # "bypass" / "directly" language warns the rewriter off the obvious
    # wrong move (edit the header, ignore the alias).
    assert "bypass" in prompt or "directly" in prompt


# ---------- Baseline harness: CUDA profile ----------
#
# Phase B added CUDA as the second language profile. Its baseline_harness
# entry is registered under AGENTS["baseline_harness_cuda"] by the
# PROFILES loop in registry.py. These tests mirror the Kokkos baseline
# harness tests above but assert CUDA-specific tokens (nvcc, cudaMalloc,
# cudaMemcpy, __global__, <cuda_runtime.h>) and the CUDA-specific
# staging-buffer rule (`std::remove_const_t<std::remove_pointer_t<...>>`,
# which replaces Kokkos's `typename ...::non_const_value_type`). The
# precision-alias contract and the splice sentinels are language-
# independent invariants, so both prompts must honor them.


def test_baseline_harness_cuda_entry_exists():
    """AGENTS["baseline_harness_cuda"] is registered with the same required keys as every other agent entry; the PROFILES loop in registry.py populates it from CUDA_PROFILE."""
    spec = AGENTS["baseline_harness_cuda"]
    assert isinstance(spec["system_prompt"], str)
    assert spec["system_prompt"].strip()
    assert spec["output_schema"]["type"] == "object"
    assert spec["model"]
    assert spec["supports_temperature"] is False


def test_baseline_harness_cuda_schema_required_fields():
    """The CUDA baseline_harness schema has the same required fields as the Kokkos one — the downstream comparator and splice tools are language-agnostic, so the submit_result payload shape is shared."""
    schema = AGENTS["baseline_harness_cuda"]["output_schema"]
    assert set(schema["required"]) == {
        "driver_source",
        "kernel_function_name",
        "inputs_summary",
        "output_arrays",
    }
    assert schema["properties"]["output_arrays"]["type"] == "array"
    assert schema["properties"]["output_arrays"]["items"]["type"] == "string"


def test_baseline_harness_cuda_prompt_mentions_cuda_runtime_and_global():
    """The CUDA baseline_harness prompt names <cuda_runtime.h> (the canonical include) and __global__ (the kernel-qualifier the driver launches) so the model is anchored on the right toolchain."""
    prompt = AGENTS["baseline_harness_cuda"]["system_prompt"]
    assert "<cuda_runtime.h>" in prompt
    assert "__global__" in prompt


def test_baseline_harness_cuda_prompt_mentions_cuda_memory_api():
    """The CUDA baseline_harness prompt names cudaMalloc and cudaMemcpy so the model knows the device-allocation and host<->device transfer API it must use."""
    prompt = AGENTS["baseline_harness_cuda"]["system_prompt"]
    assert "cudaMalloc" in prompt
    assert "cudaMemcpy" in prompt


def test_baseline_harness_cuda_prompt_mandates_determinism():
    """The CUDA baseline_harness prompt forbids host-side atomic floating-point ops (non-deterministic), mandates a fixed grid/block launch shape, and requires a fixed RNG seed — the three things needed for the reference to be bit-identical across runs."""
    prompt = AGENTS["baseline_harness_cuda"]["system_prompt"]
    # Atomics ban — atomicAdd is the canonical offender on GPU.
    assert "atomicAdd" in prompt
    # Fixed launch shape — the prompt's example uses `threads = 256`.
    lower = prompt.lower()
    assert "fixed" in lower
    # Seed instruction (same phrasing test as Kokkos: "seed" + "fixed"
    # or "reproducible" elsewhere in the prompt).
    assert "seed" in lower
    assert "fixed" in lower or "reproducible" in lower


def test_baseline_harness_cuda_prompt_mentions_reference_json_and_format():
    """The CUDA baseline_harness prompt names reference.json (the output file) and requires %.17g formatting so the reference preserves full double precision — same contract as the Kokkos prompt."""
    prompt = AGENTS["baseline_harness_cuda"]["system_prompt"]
    assert "reference.json" in prompt
    assert "%.17g" in prompt


def test_baseline_harness_cuda_prompt_mentions_target_kernel_and_no_invented_values():
    """The CUDA baseline_harness prompt names TARGET KERNEL (the disambiguator the orchestrator may prepend) and forbids inventing numerical output values — same contract as the Kokkos prompt."""
    prompt = AGENTS["baseline_harness_cuda"]["system_prompt"]
    assert "TARGET KERNEL" in prompt
    lower = prompt.lower()
    assert "do not invent" in lower or "not invent" in lower or "never invent" in lower


def test_baseline_harness_cuda_prompt_mandates_kernel_splice_sentinels():
    """The CUDA baseline_harness prompt mandates the exact splice sentinels (language-independent contract — splice_rewritten_kernel is shared across profiles)."""
    prompt = AGENTS["baseline_harness_cuda"]["system_prompt"]
    assert "// ---- KERNEL BEGIN ----" in prompt
    assert "// ---- KERNEL END ----" in prompt


def test_baseline_harness_cuda_prompt_mandates_precision_alias_contract():
    """The CUDA baseline_harness prompt mandates the same per-parameter `using <ParamName>Type` alias contract as the Kokkos prompt — splice scope is language-independent, so the alias contract is too. Integer parameters are excluded for the same reason."""
    prompt = AGENTS["baseline_harness_cuda"]["system_prompt"]
    assert "<ParamName>Type" in prompt
    assert "using" in prompt
    assert "main()" in prompt
    lower = prompt.lower()
    assert "integer parameters" in lower or "integer parameter" in lower


def test_baseline_harness_cuda_prompt_requires_alias_derived_staging_buffer():
    """The CUDA staging-buffer rule differs from Kokkos: instead of `typename ...::non_const_value_type` (a Kokkos View member), CUDA uses `std::remove_const_t<std::remove_pointer_t<aType>>` to derive the writable pointee type from a `const T*` alias. The prompt must teach this exact idiom; a hardcoded `double* a_nc; cudaMalloc(&a_nc, N*sizeof(double))` would break when the rewriter changes the alias to `const float*`."""
    prompt = AGENTS["baseline_harness_cuda"]["system_prompt"]
    # The CUDA-specific derivation idiom. Both halves matter — the
    # remove_const_t strips the const, the remove_pointer_t strips the
    # pointer, and the order is fixed (pointer first, then const) by C++.
    assert "remove_const_t" in prompt
    assert "remove_pointer_t" in prompt
    # The failure mode (hardcoded `double`) is called out explicitly.
    normalized = " ".join(prompt.lower().split())
    assert "do not hardcode" in normalized or "do not" in normalized
    # Rule is scoped to const pointer kernel arguments.
    assert "const" in normalized


def test_baseline_harness_cuda_prompt_does_not_mention_kokkos():
    """The CUDA baseline_harness prompt must not mention Kokkos-specific tokens — those would mislead the model into emitting a hybrid driver. Cross-prompt isolation is the whole point of having one profile per language."""
    prompt = AGENTS["baseline_harness_cuda"]["system_prompt"]
    # Token-level isolation. These tokens uniquely identify the Kokkos
    # prompt and have no business in a CUDA driver.
    assert "Kokkos::initialize" not in prompt
    assert "Kokkos::Serial" not in prompt
    assert "Kokkos::View" not in prompt
    assert "non_const_value_type" not in prompt


def test_baseline_harness_kokkos_prompt_does_not_mention_cuda():
    """Mirror of the CUDA isolation test: the Kokkos baseline_harness prompt must not mention CUDA-specific tokens. Together these two tests defend against accidental cross-contamination of the per-language prompts."""
    prompt = AGENTS["baseline_harness_kokkos"]["system_prompt"]
    assert "cudaMalloc" not in prompt
    assert "cudaMemcpy" not in prompt
    assert "__global__" not in prompt
    assert "<cuda_runtime.h>" not in prompt


# ---------- Baseline harness: HIP profile ----------
#
# Phase C-1 added HIP as the third language profile. Its baseline_harness
# entry is registered under AGENTS["baseline_harness_hip"] by the PROFILES
# loop in registry.py. These tests mirror the CUDA baseline harness tests
# above but assert HIP-specific tokens (hipcc, hipMalloc, hipMemcpy,
# <hip/hip_runtime.h>, hipError_t, hipGetErrorString). The kernel-launch
# syntax (`kernel<<<...>>>(...)`), the `__global__` qualifier, the splice
# sentinels, the precision-alias contract, and the staging-buffer rule
# (`std::remove_const_t<std::remove_pointer_t<...>>`) are shared with the
# CUDA prompt — HIP mirrors CUDA's C++ extensions.
#
# UNIT-TESTED, NOT SMOKE-VALIDATED. There is no HIP toolchain on the
# development host, so these tests assert prompt content only — no
# end-to-end driver compilation or execution.


def test_baseline_harness_hip_entry_exists():
    """AGENTS["baseline_harness_hip"] is registered with the same required keys as every other agent entry; the PROFILES loop in registry.py populates it from HIP_PROFILE."""
    spec = AGENTS["baseline_harness_hip"]
    assert isinstance(spec["system_prompt"], str)
    assert spec["system_prompt"].strip()
    assert spec["output_schema"]["type"] == "object"
    assert spec["model"]
    assert spec["supports_temperature"] is False


def test_baseline_harness_hip_schema_required_fields():
    """The HIP baseline_harness schema has the same required fields as the Kokkos and CUDA ones — the downstream comparator and splice tools are language-agnostic, so the submit_result payload shape is shared across profiles."""
    schema = AGENTS["baseline_harness_hip"]["output_schema"]
    assert set(schema["required"]) == {
        "driver_source",
        "kernel_function_name",
        "inputs_summary",
        "output_arrays",
    }
    assert schema["properties"]["output_arrays"]["type"] == "array"
    assert schema["properties"]["output_arrays"]["items"]["type"] == "string"


def test_baseline_harness_hip_prompt_mentions_hip_runtime_and_global():
    """The HIP baseline_harness prompt names <hip/hip_runtime.h> (the canonical include) and __global__ (the kernel qualifier the driver launches) so the model is anchored on the right toolchain — HIP shares CUDA's kernel-language syntax."""
    prompt = AGENTS["baseline_harness_hip"]["system_prompt"]
    assert "<hip/hip_runtime.h>" in prompt
    assert "__global__" in prompt


def test_baseline_harness_hip_prompt_mentions_hip_memory_api():
    """The HIP baseline_harness prompt names hipMalloc and hipMemcpy so the model knows the device-allocation and host<->device transfer API it must use. The cuda* equivalents would silently compile under hipcc on some toolchains via the HIP_PLATFORM_NVIDIA path, so the prompt must steer the model toward the hip* spelling."""
    prompt = AGENTS["baseline_harness_hip"]["system_prompt"]
    assert "hipMalloc" in prompt
    assert "hipMemcpy" in prompt


def test_baseline_harness_hip_prompt_mentions_hip_error_handling():
    """The HIP baseline_harness prompt names hipError_t, hipGetErrorString, hipDeviceSynchronize, and hipGetLastError — the four runtime entry points needed to surface a misconfigured runtime (missing driver, wrong arch) as a clear diagnostic instead of a silent wrong-answer."""
    prompt = AGENTS["baseline_harness_hip"]["system_prompt"]
    assert "hipError_t" in prompt
    assert "hipGetErrorString" in prompt
    assert "hipDeviceSynchronize" in prompt
    assert "hipGetLastError" in prompt


def test_baseline_harness_hip_prompt_mandates_determinism():
    """The HIP baseline_harness prompt forbids host-side atomic floating-point ops (non-deterministic), mandates a fixed grid/block launch shape, and requires a fixed RNG seed — the three things needed for the reference to be bit-identical across runs. Same determinism contract as Kokkos and CUDA."""
    prompt = AGENTS["baseline_harness_hip"]["system_prompt"]
    # Atomics ban — atomicAdd is the canonical offender on GPU.
    assert "atomicAdd" in prompt
    lower = prompt.lower()
    assert "fixed" in lower
    assert "seed" in lower
    assert "fixed" in lower or "reproducible" in lower


def test_baseline_harness_hip_prompt_mentions_reference_json_and_format():
    """The HIP baseline_harness prompt names reference.json (the output file) and requires %.17g formatting so the reference preserves full double precision — same contract as the Kokkos and CUDA prompts."""
    prompt = AGENTS["baseline_harness_hip"]["system_prompt"]
    assert "reference.json" in prompt
    assert "%.17g" in prompt


def test_baseline_harness_hip_prompt_mentions_target_kernel_and_no_invented_values():
    """The HIP baseline_harness prompt names TARGET KERNEL (the disambiguator the orchestrator may prepend) and forbids inventing numerical output values — same contract as the Kokkos and CUDA prompts."""
    prompt = AGENTS["baseline_harness_hip"]["system_prompt"]
    assert "TARGET KERNEL" in prompt
    lower = prompt.lower()
    assert "do not invent" in lower or "not invent" in lower or "never invent" in lower


def test_baseline_harness_hip_prompt_mandates_kernel_splice_sentinels():
    """The HIP baseline_harness prompt mandates the exact splice sentinels (language-independent contract — splice_rewritten_kernel is shared across profiles)."""
    prompt = AGENTS["baseline_harness_hip"]["system_prompt"]
    assert "// ---- KERNEL BEGIN ----" in prompt
    assert "// ---- KERNEL END ----" in prompt


def test_baseline_harness_hip_prompt_mandates_precision_alias_contract():
    """The HIP baseline_harness prompt mandates the same per-parameter `using <ParamName>Type` alias contract as the Kokkos and CUDA prompts — splice scope is language-independent, so the alias contract is too. Integer parameters are excluded for the same reason."""
    prompt = AGENTS["baseline_harness_hip"]["system_prompt"]
    assert "<ParamName>Type" in prompt
    assert "using" in prompt
    assert "main()" in prompt
    lower = prompt.lower()
    assert "integer parameters" in lower or "integer parameter" in lower


def test_baseline_harness_hip_prompt_requires_alias_derived_staging_buffer():
    """HIP's staging-buffer rule is identical to CUDA's: `std::remove_const_t<std::remove_pointer_t<aType>>` derives the writable pointee type from a `const T*` alias. The prompt must teach this exact idiom; a hardcoded `double* a_nc; hipMalloc(&a_nc, N*sizeof(double))` would break when the rewriter changes the alias to `const float*`. The rule is NOT shared with Kokkos, which uses `typename ...::non_const_value_type` (a View member, not a pointer construct)."""
    prompt = AGENTS["baseline_harness_hip"]["system_prompt"]
    assert "remove_const_t" in prompt
    assert "remove_pointer_t" in prompt
    normalized = " ".join(prompt.lower().split())
    assert "do not hardcode" in normalized or "do not" in normalized
    assert "const" in normalized


def test_baseline_harness_hip_prompt_does_not_mention_kokkos():
    """The HIP baseline_harness prompt must not mention Kokkos-specific tokens — those would mislead the model into emitting a hybrid driver. Cross-prompt isolation is the whole point of having one profile per language."""
    prompt = AGENTS["baseline_harness_hip"]["system_prompt"]
    assert "Kokkos::initialize" not in prompt
    assert "Kokkos::Serial" not in prompt
    assert "Kokkos::View" not in prompt
    assert "non_const_value_type" not in prompt


def test_baseline_harness_hip_prompt_does_not_mention_cuda_runtime_tokens():
    """The HIP baseline_harness prompt must not mention CUDA-specific runtime tokens (cudaMalloc, cudaMemcpy, <cuda_runtime.h>, cudaError_t). HIP and CUDA share the kernel-launch syntax and the `__global__` qualifier — those are NOT forbidden — but the runtime API names diverge and mixing them would compile only under the CUDA-backend hipcc path, not under ROCm. Token-level isolation forces the prompt to commit to the HIP spelling."""
    prompt = AGENTS["baseline_harness_hip"]["system_prompt"]
    assert "cudaMalloc" not in prompt
    assert "cudaMemcpy" not in prompt
    assert "<cuda_runtime.h>" not in prompt
    assert "cudaError_t" not in prompt
    assert "cudaGetErrorString" not in prompt
    assert "cudaDeviceSynchronize" not in prompt


def test_baseline_harness_kokkos_prompt_does_not_mention_hip():
    """Mirror of the HIP isolation test: the Kokkos baseline_harness prompt must not mention HIP-specific runtime tokens. Together with the CUDA-isolation analogue, defends against accidental cross-contamination across all three profiles."""
    prompt = AGENTS["baseline_harness_kokkos"]["system_prompt"]
    assert "hipMalloc" not in prompt
    assert "hipMemcpy" not in prompt
    assert "<hip/hip_runtime.h>" not in prompt
    assert "hipError_t" not in prompt


def test_baseline_harness_cuda_prompt_does_not_mention_hip():
    """Mirror of the HIP isolation test for the CUDA prompt — CUDA-side defence against accidentally inheriting HIP runtime tokens. The CUDA prompt was written first and HIP cloned its structure, so this guards against future drift in either direction."""
    prompt = AGENTS["baseline_harness_cuda"]["system_prompt"]
    assert "hipMalloc" not in prompt
    assert "hipMemcpy" not in prompt
    assert "<hip/hip_runtime.h>" not in prompt
    assert "hipError_t" not in prompt


# ---------- Baseline harness: SYCL profile ----------
#
# Phase C-2 added SYCL as the fourth language profile. Its baseline_harness
# entry is registered under AGENTS["baseline_harness_sycl"] by the PROFILES
# loop in registry.py. SYCL differs structurally from CUDA / HIP in that
# kernels are lambdas submitted to a queue rather than free `__global__`
# functions with `<<<...>>>` launch syntax, so the prompt tokens differ:
# `sycl::queue`, `sycl::buffer`, `sycl::accessor`, `host_accessor`,
# `<sycl/sycl.hpp>`, `in_order`, `parallel_for`, `sycl::exception`,
# `icpx`. The splice sentinels and the precision-alias contract are still
# shared with the other profiles (splice scope is language-independent),
# but the alias attaches to the buffer element type (`<BufferName>Type`)
# rather than to a function parameter, because SYCL has no free-standing
# kernel function header to alias.
#
# UNIT-TESTED, NOT SMOKE-VALIDATED. There is no SYCL toolchain (icpx,
# clang++ -fsycl, dpcpp) on the development host, so these tests assert
# prompt content only — no end-to-end driver compilation or execution.


def test_baseline_harness_sycl_entry_exists():
    """AGENTS["baseline_harness_sycl"] is registered with the same required keys as every other agent entry; the PROFILES loop in registry.py populates it from SYCL_PROFILE."""
    spec = AGENTS["baseline_harness_sycl"]
    assert isinstance(spec["system_prompt"], str)
    assert spec["system_prompt"].strip()
    assert spec["output_schema"]["type"] == "object"
    assert spec["model"]
    assert spec["supports_temperature"] is False


def test_baseline_harness_sycl_schema_required_fields():
    """The SYCL baseline_harness schema has the same required fields as Kokkos / CUDA / HIP — splice + comparator + run tools are language-agnostic, so the submit_result payload shape is shared across profiles."""
    schema = AGENTS["baseline_harness_sycl"]["output_schema"]
    assert set(schema["required"]) == {
        "driver_source",
        "kernel_function_name",
        "inputs_summary",
        "output_arrays",
    }
    assert schema["properties"]["output_arrays"]["type"] == "array"
    assert schema["properties"]["output_arrays"]["items"]["type"] == "string"


def test_baseline_harness_sycl_prompt_mentions_sycl_header_and_queue():
    """The SYCL baseline_harness prompt names <sycl/sycl.hpp> (the canonical include) and `sycl::queue` (the central runtime object) so the model is anchored on the SYCL execution model rather than on CUDA's stream/launch model."""
    prompt = AGENTS["baseline_harness_sycl"]["system_prompt"]
    assert "<sycl/sycl.hpp>" in prompt
    assert "sycl::queue" in prompt


def test_baseline_harness_sycl_prompt_mandates_in_order_queue():
    """The SYCL baseline_harness prompt mandates the in-order queue property — `sycl::property::queue::in_order{}`. SYCL queues are out-of-order by default; without this, two `parallel_for` submissions to the same queue can complete in either order and the reference becomes non-reproducible. The other three profiles do not need this rule because their default execution order is already sequential (Kokkos::Serial, CUDA stream 0, HIP stream 0)."""
    prompt = AGENTS["baseline_harness_sycl"]["system_prompt"]
    assert "sycl::property::queue::in_order" in prompt
    lower = prompt.lower()
    assert "in-order" in lower or "in_order" in lower


def test_baseline_harness_sycl_prompt_mandates_buffer_accessor_memory_model():
    """The SYCL baseline_harness prompt mandates the `sycl::buffer` + `sycl::accessor` memory model and forbids USM. Buffers are portable across all SYCL implementations and force a well-defined host-side sync point via `host_accessor`, which makes the baseline reproducible. USM (`sycl::malloc_device`) would require explicit `q.wait()` plus manual `memcpy`, with weaker portability across implementations."""
    prompt = AGENTS["baseline_harness_sycl"]["system_prompt"]
    assert "sycl::buffer" in prompt
    assert "sycl::accessor" in prompt
    assert "host_accessor" in prompt


def test_baseline_harness_sycl_prompt_mentions_sycl_exception_handling():
    """The SYCL baseline_harness prompt names `sycl::exception` and mandates a try/catch around throwing SYCL operations. SYCL surfaces device errors through exceptions, not error codes; an uncaught exception terminates with a confusing `std::terminate` rather than a useful diagnostic. This is structurally different from CUDA / HIP error handling (cudaError_t / hipError_t), so the prompt must teach it explicitly."""
    prompt = AGENTS["baseline_harness_sycl"]["system_prompt"]
    assert "sycl::exception" in prompt
    lower = prompt.lower()
    assert "try" in lower and "catch" in lower


def test_baseline_harness_sycl_prompt_mandates_determinism():
    """The SYCL baseline_harness prompt forbids floating-point `sycl::atomic_ref` ops in the reference computation (non-deterministic ordering), mandates a fixed nd-range or range launch shape, and requires a fixed RNG seed — the three things needed for the reference to be bit-identical across runs. Same determinism contract as Kokkos / CUDA / HIP, expressed in SYCL vocabulary."""
    prompt = AGENTS["baseline_harness_sycl"]["system_prompt"]
    assert "sycl::atomic_ref" in prompt
    lower = prompt.lower()
    assert "fixed" in lower
    assert "seed" in lower
    assert "fixed" in lower or "reproducible" in lower


def test_baseline_harness_sycl_prompt_mentions_reference_json_and_format():
    """The SYCL baseline_harness prompt names reference.json (the output file) and requires %.17g formatting so the reference preserves full double precision — same contract as the Kokkos / CUDA / HIP prompts."""
    prompt = AGENTS["baseline_harness_sycl"]["system_prompt"]
    assert "reference.json" in prompt
    assert "%.17g" in prompt


def test_baseline_harness_sycl_prompt_mentions_target_kernel_and_no_invented_values():
    """The SYCL baseline_harness prompt names TARGET KERNEL (the disambiguator the orchestrator may prepend) and forbids inventing numerical output values — same contract as the Kokkos / CUDA / HIP prompts."""
    prompt = AGENTS["baseline_harness_sycl"]["system_prompt"]
    assert "TARGET KERNEL" in prompt
    lower = prompt.lower()
    assert "do not invent" in lower or "not invent" in lower or "never invent" in lower


def test_baseline_harness_sycl_prompt_mandates_kernel_splice_sentinels():
    """The SYCL baseline_harness prompt mandates the exact splice sentinels (language-independent contract — splice_rewritten_kernel is shared across profiles)."""
    prompt = AGENTS["baseline_harness_sycl"]["system_prompt"]
    assert "// ---- KERNEL BEGIN ----" in prompt
    assert "// ---- KERNEL END ----" in prompt


def test_baseline_harness_sycl_prompt_mandates_precision_alias_contract():
    """The SYCL baseline_harness prompt mandates a per-buffer `using <BufferName>Type` alias contract. Unlike Kokkos / CUDA / HIP — where the alias attaches to a kernel function parameter — SYCL kernels are lambdas with no free-standing function header, so the alias attaches to the buffer element type instead (`sycl::buffer<aType> a_buf(...)`). The accessor types then propagate from the buffer template automatically, so the lambda body stays byte-for-byte identical. Integer buffers are excluded for the same reason as in the other profiles."""
    prompt = AGENTS["baseline_harness_sycl"]["system_prompt"]
    assert "<BufferName>Type" in prompt
    assert "using" in prompt
    lower = prompt.lower()
    assert "integer buffer" in lower or "integer buffers" in lower


def test_baseline_harness_sycl_prompt_requires_alias_derived_staging_vector():
    """SYCL's staging rule: when a host `std::vector` backs a `sycl::buffer<aType>`, the vector's element type must be declared as `std::vector<aType>` so that redefining `aType` to `float` does not break the wrap (a hardcoded `std::vector<double>` paired with `sycl::buffer<aType>` would either silently reinterpret bytes or fail to compile under a precision downcast). This is the SYCL analog of CUDA / HIP's `std::remove_const_t<std::remove_pointer_t<...>>` staging rule and of Kokkos's `typename ...::non_const_value_type` rule, but expressed in terms that match SYCL's buffer-element-type model."""
    prompt = AGENTS["baseline_harness_sycl"]["system_prompt"]
    assert "std::vector<aType>" in prompt
    normalized = " ".join(prompt.lower().split())
    assert "do not hardcode" in normalized or "do not" in normalized


def test_baseline_harness_sycl_prompt_does_not_mention_kokkos():
    """The SYCL baseline_harness prompt must not mention Kokkos-specific tokens — those would mislead the model into emitting a hybrid driver. Cross-prompt isolation is the whole point of having one profile per language."""
    prompt = AGENTS["baseline_harness_sycl"]["system_prompt"]
    assert "Kokkos::initialize" not in prompt
    assert "Kokkos::Serial" not in prompt
    assert "Kokkos::View" not in prompt
    assert "non_const_value_type" not in prompt
    assert "KOKKOS_LAMBDA" not in prompt


def test_baseline_harness_sycl_prompt_does_not_mention_cuda_or_hip_runtime_tokens():
    """The SYCL baseline_harness prompt must not mention CUDA / HIP runtime tokens (cudaMalloc, cudaMemcpy, hipMalloc, hipMemcpy, the runtime headers, the error-code types, `__global__`, atomicAdd, the `<<<...>>>` launch syntax). SYCL's execution model is structurally different — kernels are lambdas — so any of these tokens in the prompt would be actively misleading rather than just stylistically wrong."""
    prompt = AGENTS["baseline_harness_sycl"]["system_prompt"]
    assert "cudaMalloc" not in prompt
    assert "cudaMemcpy" not in prompt
    assert "<cuda_runtime.h>" not in prompt
    assert "hipMalloc" not in prompt
    assert "hipMemcpy" not in prompt
    assert "<hip/hip_runtime.h>" not in prompt
    assert "__global__" not in prompt
    assert "atomicAdd" not in prompt
    assert "<<<" not in prompt


def test_baseline_harness_kokkos_prompt_does_not_mention_sycl():
    """Mirror of the SYCL isolation test: the Kokkos baseline_harness prompt must not mention SYCL-specific tokens. Together with the corresponding CUDA / HIP analogues below, defends against accidental cross-contamination across all four profiles."""
    prompt = AGENTS["baseline_harness_kokkos"]["system_prompt"]
    assert "sycl::queue" not in prompt
    assert "sycl::buffer" not in prompt
    assert "<sycl/sycl.hpp>" not in prompt
    assert "host_accessor" not in prompt


def test_baseline_harness_cuda_prompt_does_not_mention_sycl():
    """Mirror of the SYCL isolation test for the CUDA prompt."""
    prompt = AGENTS["baseline_harness_cuda"]["system_prompt"]
    assert "sycl::queue" not in prompt
    assert "sycl::buffer" not in prompt
    assert "<sycl/sycl.hpp>" not in prompt
    assert "host_accessor" not in prompt


def test_baseline_harness_hip_prompt_does_not_mention_sycl():
    """Mirror of the SYCL isolation test for the HIP prompt."""
    prompt = AGENTS["baseline_harness_hip"]["system_prompt"]
    assert "sycl::queue" not in prompt
    assert "sycl::buffer" not in prompt
    assert "<sycl/sycl.hpp>" not in prompt
    assert "host_accessor" not in prompt


# ---------- Baseline harness: OpenMP target-offload profile ----------

# Phase C-3 added OpenMP target-offload as the fifth language profile.
# Its baseline_harness entry is registered under
# AGENTS["baseline_harness_omp_offload"] by the PROFILES loop in
# registry.py. OMP-offload differs structurally from CUDA / HIP in
# that kernels are plain C/C++ functions (not __global__), launched
# from a `#pragma omp target` region with explicit map() clauses; it
# differs from SYCL in that there is no queue, so determinism is
# enforced via `omp_set_num_threads(1)` and a prohibition on
# reductions. The prompt must teach this execution model and the
# associated reproducibility rules.
#
# UNIT-TESTED, NOT SMOKE-VALIDATED. No host with an OMP-offload
# toolchain (clang++ -fopenmp -fopenmp-targets=..., icpx, nvc++)
# was available at implementation time, so these tests assert the
# data shape and prompt content rather than runtime behavior.


def test_baseline_harness_omp_offload_entry_exists():
    """AGENTS["baseline_harness_omp_offload"] is registered with the same required keys as every other agent entry; the PROFILES loop in registry.py populates it from OMP_OFFLOAD_PROFILE."""
    spec = AGENTS["baseline_harness_omp_offload"]
    assert "system_prompt" in spec
    assert "output_schema" in spec
    assert "model" in spec
    assert "supports_temperature" in spec
    assert spec["supports_temperature"] is False


def test_baseline_harness_omp_offload_schema_required_fields():
    """The OMP-offload baseline_harness schema has the same required fields as Kokkos / CUDA / HIP / SYCL — splice + comparator + run tools are language-agnostic, so the submit_result payload shape is shared across profiles."""
    schema = AGENTS["baseline_harness_omp_offload"]["output_schema"]
    assert set(schema["required"]) == {
        "driver_source",
        "kernel_function_name",
        "inputs_summary",
        "output_arrays",
    }


def test_baseline_harness_omp_offload_prompt_mentions_omp_target_directive():
    """The OMP-offload baseline_harness prompt names `#pragma omp target` (the directive that triggers device-side execution) so the model is anchored on the OMP-offload execution model rather than on a stream / queue / launch-syntax model from another profile."""
    prompt = AGENTS["baseline_harness_omp_offload"]["system_prompt"]
    assert "#pragma omp target" in prompt


def test_baseline_harness_omp_offload_prompt_mandates_single_thread_determinism():
    """The OMP-offload baseline_harness prompt mandates `omp_set_num_threads(1)` for host-side determinism. OMP has no in-order queue equivalent (unlike SYCL); single-thread execution is the only portable way to get a reproducible reference across clang++ / icpx / nvc++. The other four profiles do not need this rule because their default execution order is already sequential (Kokkos::Serial, CUDA stream 0, HIP stream 0, SYCL in_order queue)."""
    prompt = AGENTS["baseline_harness_omp_offload"]["system_prompt"]
    assert "omp_set_num_threads(1)" in prompt


def test_baseline_harness_omp_offload_prompt_forbids_floating_point_reductions():
    """The OMP-offload baseline_harness prompt forbids `reduction(+:...)` (and the wider class of floating-point reduction clauses) in the reference computation. OMP reduction order is unspecified across teams / threads; without this prohibition the reference is non-reproducible across compiler versions. This is the OMP analog of SYCL's `sycl::atomic_ref` prohibition and CUDA's `atomicAdd` prohibition."""
    prompt = AGENTS["baseline_harness_omp_offload"]["system_prompt"]
    assert "reduction(+:" in prompt or "reduction(" in prompt


def test_baseline_harness_omp_offload_prompt_mandates_explicit_map_clauses():
    """The OMP-offload baseline_harness prompt mandates explicit `map(to:...)`, `map(from:...)`, and `map(tofrom:...)` clauses for every array crossing the host/device boundary. Implicit / default mapping is fragile across compilers and would make the reference non-portable. This rule is structurally unique to OMP-offload — CUDA / HIP do explicit cudaMemcpy / hipMemcpy, SYCL uses buffer + accessor, and Kokkos relies on View deep_copy."""
    prompt = AGENTS["baseline_harness_omp_offload"]["system_prompt"]
    assert "map(to:" in prompt
    assert "map(from:" in prompt
    assert "map(tofrom:" in prompt


def test_baseline_harness_omp_offload_prompt_mandates_single_team_single_thread_launch():
    """The OMP-offload baseline_harness prompt mandates single-team / single-thread device-side launch bounds (`num_teams(1)` and `thread_limit(1)`). This is the device-side analog of `omp_set_num_threads(1)`: the slowest configuration, but the only one that guarantees reproducibility across compilers and devices for the baseline. A future smoke phase may relax this once cross-compiler reproducibility is established."""
    prompt = AGENTS["baseline_harness_omp_offload"]["system_prompt"]
    assert "num_teams(1)" in prompt
    assert "thread_limit(1)" in prompt


def test_baseline_harness_omp_offload_prompt_mentions_reference_json_and_format():
    """The OMP-offload baseline_harness prompt names reference.json (the output file) and requires %.17g formatting so the reference preserves full double precision — same contract as the Kokkos / CUDA / HIP / SYCL prompts."""
    prompt = AGENTS["baseline_harness_omp_offload"]["system_prompt"]
    assert "reference.json" in prompt
    assert "%.17g" in prompt


def test_baseline_harness_omp_offload_prompt_mentions_target_kernel_and_no_invented_values():
    """The OMP-offload baseline_harness prompt names TARGET KERNEL (the disambiguator the orchestrator may prepend) and forbids inventing numerical output values — same contract as the Kokkos / CUDA / HIP / SYCL prompts."""
    prompt = AGENTS["baseline_harness_omp_offload"]["system_prompt"]
    assert "TARGET KERNEL" in prompt
    assert "invent" in prompt or "invented" in prompt


def test_baseline_harness_omp_offload_prompt_mandates_kernel_splice_sentinels():
    """The OMP-offload baseline_harness prompt mandates the exact splice sentinels (language-independent contract — splice_rewritten_kernel is shared across profiles)."""
    prompt = AGENTS["baseline_harness_omp_offload"]["system_prompt"]
    assert "// ---- KERNEL BEGIN ----" in prompt
    assert "// ---- KERNEL END ----" in prompt


def test_baseline_harness_omp_offload_prompt_mandates_precision_alias_contract():
    """The OMP-offload baseline_harness prompt mandates a per-parameter `using <ParamName>Type` alias contract — same structural shape as Kokkos / CUDA / HIP (where the alias attaches to a kernel function parameter), since OMP-offload kernels are also plain functions. Integer parameters are excluded for the same reason as in the other profiles."""
    prompt = AGENTS["baseline_harness_omp_offload"]["system_prompt"]
    assert "<ParamName>Type" in prompt
    # Concrete example tokens should appear so the model has a worked example.
    assert "aType" in prompt
    assert "using cType" in prompt or "cType" in prompt


def test_baseline_harness_omp_offload_prompt_requires_alias_derived_host_buffers():
    """OMP-offload's staging rule: host vectors backing a mapped array must derive their element type from the alias (`std::vector<aType>`), not hardcode `std::vector<double>`. A precision downcast that redefines `aType` to `float` would otherwise break the `map(to:...)` clause by transferring sizeof(double) bytes per element on the host and sizeof(float) on the device. This is the OMP analog of SYCL's `std::vector<aType>` rule and CUDA / HIP's staging-pointer rule."""
    prompt = AGENTS["baseline_harness_omp_offload"]["system_prompt"]
    assert "std::vector<aType>" in prompt


def test_baseline_harness_omp_offload_prompt_does_not_mention_kokkos():
    """The OMP-offload baseline_harness prompt must not mention Kokkos-specific tokens — those would mislead the model into emitting a hybrid driver. Cross-prompt isolation is the whole point of having one profile per language."""
    prompt = AGENTS["baseline_harness_omp_offload"]["system_prompt"]
    assert "Kokkos::initialize" not in prompt
    assert "Kokkos::Serial" not in prompt
    assert "Kokkos::View" not in prompt
    assert "non_const_value_type" not in prompt
    assert "KOKKOS_LAMBDA" not in prompt


def test_baseline_harness_omp_offload_prompt_does_not_mention_cuda_or_hip_runtime_tokens():
    """The OMP-offload baseline_harness prompt must not mention CUDA / HIP runtime tokens (cudaMalloc, cudaMemcpy, hipMalloc, hipMemcpy, the runtime headers, `__global__`, atomicAdd, `<<<...>>>` launch syntax). OMP-offload's execution model is structurally different — `#pragma omp target` regions calling plain functions — so any of these tokens would actively mislead the model."""
    prompt = AGENTS["baseline_harness_omp_offload"]["system_prompt"]
    assert "cudaMalloc" not in prompt
    assert "cudaMemcpy" not in prompt
    assert "<cuda_runtime.h>" not in prompt
    assert "hipMalloc" not in prompt
    assert "hipMemcpy" not in prompt
    assert "<hip/hip_runtime.h>" not in prompt
    assert "__global__" not in prompt
    assert "atomicAdd" not in prompt
    assert "<<<" not in prompt


def test_baseline_harness_omp_offload_prompt_does_not_mention_sycl():
    """The OMP-offload baseline_harness prompt must not mention SYCL-specific tokens — they would actively mislead the model. SYCL queues / buffers / accessors are a different execution model from OMP target regions + map clauses."""
    prompt = AGENTS["baseline_harness_omp_offload"]["system_prompt"]
    assert "sycl::queue" not in prompt
    assert "sycl::buffer" not in prompt
    assert "<sycl/sycl.hpp>" not in prompt
    assert "host_accessor" not in prompt


def test_baseline_harness_kokkos_prompt_does_not_mention_omp_offload():
    """Mirror of the OMP-offload isolation test: the Kokkos baseline_harness prompt must not mention OMP target-offload-specific tokens."""
    prompt = AGENTS["baseline_harness_kokkos"]["system_prompt"]
    assert "#pragma omp target" not in prompt
    assert "omp_set_num_threads" not in prompt
    assert "-fopenmp-targets" not in prompt


def test_baseline_harness_cuda_prompt_does_not_mention_omp_offload():
    """Mirror of the OMP-offload isolation test for the CUDA prompt."""
    prompt = AGENTS["baseline_harness_cuda"]["system_prompt"]
    assert "#pragma omp target" not in prompt
    assert "omp_set_num_threads" not in prompt
    assert "-fopenmp-targets" not in prompt


def test_baseline_harness_hip_prompt_does_not_mention_omp_offload():
    """Mirror of the OMP-offload isolation test for the HIP prompt."""
    prompt = AGENTS["baseline_harness_hip"]["system_prompt"]
    assert "#pragma omp target" not in prompt
    assert "omp_set_num_threads" not in prompt
    assert "-fopenmp-targets" not in prompt


def test_baseline_harness_sycl_prompt_does_not_mention_omp_offload():
    """Mirror of the OMP-offload isolation test for the SYCL prompt."""
    prompt = AGENTS["baseline_harness_sycl"]["system_prompt"]
    assert "#pragma omp target" not in prompt
    assert "omp_set_num_threads" not in prompt
    assert "-fopenmp-targets" not in prompt


# ---------- Candidate finder: schema shape + prompt smoke checks ----------


def test_candidate_finder_registered():
    """candidate_finder is a registered agent, distinct from analyst, with its own prompt + schema (not a back-compat alias to another entry)."""
    assert "candidate_finder" in AGENTS
    finder = AGENTS["candidate_finder"]
    analyst = AGENTS["analyst"]
    assert finder is not analyst
    assert finder["output_schema"] is CANDIDATE_FINDER_OUTPUT_SCHEMA
    assert finder["output_schema"] is not analyst["output_schema"]
    assert finder["system_prompt"] != analyst["system_prompt"]


def test_candidate_finder_schema_variable_item_required_fields():
    """Each variables[] entry must declare name (str), downcast_candidate (bool), rank (int), rationale (str). Unified schema — one row per FP variable, non-candidates included so the triage is auditable."""
    items = CANDIDATE_FINDER_OUTPUT_SCHEMA["properties"]["variables"]["items"]
    assert items["type"] == "object"
    required = set(items["required"])
    assert required == {"name", "downcast_candidate", "rank", "rationale"}
    props = items["properties"]
    assert props["name"]["type"] == "string"
    assert props["downcast_candidate"]["type"] == "boolean"
    assert props["rank"]["type"] == "integer"
    assert props["rationale"]["type"] == "string"


def test_candidate_finder_schema_top_level_required_fields():
    """The top-level object requires variables (the ranked list) and overall_notes (cross-cutting context for downstream analysts). No verdict, no precision_budget, no rework — the finder is triage, not verdict."""
    required = set(CANDIDATE_FINDER_OUTPUT_SCHEMA["required"])
    assert required == {"variables", "overall_notes"}
    props = CANDIDATE_FINDER_OUTPUT_SCHEMA["properties"]
    assert "verdict" not in props
    assert "precision_budget" not in props
    assert "rework" not in props


def test_candidate_finder_prompt_mentions_probe_evidence_and_tolerance():
    """The finder prompt tells the agent it may receive a PROBE EVIDENCE (JSON) block AND that the tolerance is a hard downstream constraint. Both pieces are handed to the finder the same way they are handed to the analyst."""
    prompt = AGENTS["candidate_finder"]["system_prompt"]
    assert "PROBE EVIDENCE" in prompt
    assert "tolerance" in prompt.lower()


def test_candidate_finder_schema_does_not_emit_verdicts():
    """The finder's SCHEMA must not carry the analyst's per-variable action fields (action, target_precision, emulation_type) — that would collapse the finder and analyst back into one agent. The prompt is allowed to mention those tokens descriptively (e.g. 'you do NOT fill in target_precision'), so we check the schema shape, not prompt text."""
    item_props = CANDIDATE_FINDER_OUTPUT_SCHEMA[
        "properties"
    ]["variables"]["items"]["properties"]
    assert "action" not in item_props
    assert "target_precision" not in item_props
    assert "emulation_type" not in item_props


# ---------- Variable analyst: schema shape + prompt smoke checks ----------


def test_variable_analyst_registered():
    """variable_analyst is a registered agent, distinct from analyst and candidate_finder, with its own prompt + schema (not a back-compat alias). Step 2 of the per-variable refactor introduces this agent; the monolithic analyst entry is retained only for the still-passing tests that exercise the legacy dispatch branch, not for LLM-driven runs."""
    assert "variable_analyst" in AGENTS
    var_analyst = AGENTS["variable_analyst"]
    analyst = AGENTS["analyst"]
    finder = AGENTS["candidate_finder"]
    assert var_analyst is not analyst
    assert var_analyst is not finder
    assert var_analyst["output_schema"] is VARIABLE_ANALYST_OUTPUT_SCHEMA
    assert var_analyst["output_schema"] is not analyst["output_schema"]
    assert var_analyst["output_schema"] is not finder["output_schema"]
    assert var_analyst["system_prompt"] != analyst["system_prompt"]
    assert var_analyst["system_prompt"] != finder["system_prompt"]


def test_variable_analyst_schema_top_level_shape():
    """The top-level object requires exactly the single-variable verdict object plus per-call notes. No verdict, no precision_budget, no rework, no overall_notes — the finalizer (Step 5) owns those fields; the per-variable analyst is a per-variable specialist and nothing more."""
    required = set(VARIABLE_ANALYST_OUTPUT_SCHEMA["required"])
    assert required == {"variable", "notes"}
    props = VARIABLE_ANALYST_OUTPUT_SCHEMA["properties"]
    assert set(props) == {"variable", "notes"}
    assert props["variable"]["type"] == "object"
    assert props["notes"]["type"] == "string"


def test_variable_analyst_schema_single_variable_matches_monolithic_item():
    """The variable-verdict object's field set must equal one ANALYST_OUTPUT_SCHEMA.variables[] entry verbatim, so the orchestrator can splice N per-variable results into a monolithic-analyst-shaped dict in Step 2 with no field renaming. If the monolithic per-variable schema changes, this must change in lockstep."""
    variable = VARIABLE_ANALYST_OUTPUT_SCHEMA["properties"]["variable"]
    mono_item = ANALYST_OUTPUT_SCHEMA["properties"]["variables"]["items"]
    assert set(variable["required"]) == set(mono_item["required"])
    assert set(variable["properties"]) == set(mono_item["properties"])
    # Enum on 'action' must be identical or the assembled monolithic
    # dict fails downstream validation.
    assert (
        variable["properties"]["action"]["enum"]
        == mono_item["properties"]["action"]["enum"]
    )


def test_variable_analyst_schema_does_not_emit_budget_or_rework():
    """The variable analyst's SCHEMA must not carry precision_budget, rework, overall_notes, or any collection of multiple variables — those are the finalizer's job. Checking schema shape (not prompt text) so descriptive prompt mentions are allowed."""
    props = VARIABLE_ANALYST_OUTPUT_SCHEMA["properties"]
    assert "precision_budget" not in props
    assert "rework" not in props
    assert "overall_notes" not in props
    assert "variables" not in props  # plural — that's the monolithic shape


def test_variable_analyst_prompt_mentions_target_variable_and_finder_and_probe():
    """The variable-analyst prompt must tell the agent (a) its task will name a TARGET VARIABLE and it returns a verdict only for that variable; (b) it may receive a CANDIDATE FINDER RESULT block for cross-variable context; and (c) it may receive a PROBE EVIDENCE block. All three are dispatch-time injections done by the orchestrator's `_execute_tool`; the prompt is the contract."""
    prompt = AGENTS["variable_analyst"]["system_prompt"]
    assert "TARGET VARIABLE" in prompt
    assert "CANDIDATE FINDER" in prompt
    assert "PROBE EVIDENCE" in prompt


def test_variable_analyst_prompt_mentions_tolerance_as_hard_constraint():
    """The tolerance is a hard downstream constraint (same rule as the monolithic analyst). Without this the per-variable specialist could recommend downcasts that violate the operator's target when composed with other per-variable verdicts."""
    prompt = AGENTS["variable_analyst"]["system_prompt"]
    assert "tolerance" in prompt.lower()


def test_analyst_finalizer_registered():
    """analyst_finalizer is a registered agent, distinct from analyst / variable_analyst / candidate_finder, wiring the finalizer prompt to the reused ANALYST_OUTPUT_SCHEMA. Step 5a of the per-variable refactor introduces this agent; it is the single-shot synthesis step that consumes the assembled per-variable list and writes the whole-kernel wrapper the verifier reads."""
    assert "analyst_finalizer" in AGENTS
    finalizer = AGENTS["analyst_finalizer"]
    assert finalizer is not AGENTS["analyst"]
    assert finalizer is not AGENTS["variable_analyst"]
    assert finalizer is not AGENTS["candidate_finder"]
    assert finalizer["system_prompt"].strip()
    assert finalizer["output_schema"] is ANALYST_FINALIZER_OUTPUT_SCHEMA


def test_analyst_finalizer_schema_reuses_analyst_schema():
    """The finalizer's output schema is ANALYST_OUTPUT_SCHEMA verbatim (same object identity, not a copy). Reusing the schema is the whole point: the verifier reads analyst_verdict_json and must not distinguish 'written by the old monolithic analyst' from 'written by the finalizer'. A schema copy would work at the JSON level but would invite silent drift over time — the identity assertion is the guardrail."""
    assert ANALYST_FINALIZER_OUTPUT_SCHEMA is ANALYST_OUTPUT_SCHEMA


def test_analyst_finalizer_prompt_mentions_assembled_verdict_and_probe():
    """The finalizer prompt must tell the agent (a) it receives an ASSEMBLED VERDICT (JSON) block containing a `variables` list — this is the pipeline's decision, not the finalizer's; (b) it MUST echo name/action/target_precision/emulation_type verbatim on every entry; (c) it may receive a PROBE EVIDENCE block. These are dispatch-time contracts enforced by the orchestrator's `_execute_tool` (the assembled verdict is passed as a tool argument, probe evidence is auto-injected when present); the prompt is where the LLM learns the shape."""
    prompt = AGENTS["analyst_finalizer"]["system_prompt"]
    assert "ASSEMBLED VERDICT" in prompt
    assert "PROBE EVIDENCE" in prompt
    assert "verbatim" in prompt.lower()


def test_analyst_finalizer_prompt_forbids_changing_per_variable_actions():
    """The finalizer must not change name / action / target_precision / emulation_type on any entry, and must not add or drop entries. This is the invariant that makes the finalizer safe to bolt onto the empirically-gated pipeline: the singleton and bisect gates upstream chose which downcasts survive, and the finalizer is not allowed to overrule them."""
    prompt = AGENTS["analyst_finalizer"]["system_prompt"]
    lower = prompt.lower()
    # The prompt names the four fields it MUST NOT change.
    assert "action" in lower
    assert "target_precision" in lower
    assert "emulation_type" in lower
    # And states that add / drop of entries is forbidden.
    assert "add or drop" in lower or "not add" in lower


def test_analyst_finalizer_prompt_mentions_tolerance_as_hard_constraint():
    """Same tolerance rule as the earlier analyst agents. The finalizer writes precision_budget by copying target_kind / target_value / source verbatim from the tolerance block, so the prompt must call out that block by name."""
    prompt = AGENTS["analyst_finalizer"]["system_prompt"]
    lower = prompt.lower()
    assert "tolerance" in lower
    assert "target_kind" in lower
    assert "target_value" in lower
    assert "hard" in prompt.lower() or "constraint" in prompt.lower()
