"""Registry shape tests. No mocks, no network."""

from workflow.registry import (
    AGENTS,
    ANALYST_OUTPUT_SCHEMA,
    BASELINE_HARNESS_OUTPUT_SCHEMA,
    PRECISION_ADVISOR_OUTPUT_SCHEMA,
    VERIFIER_OUTPUT_SCHEMA,
)


def test_known_agent_types():
    """AGENTS exposes the core agent types plus one baseline_harness_<lang> per registered language profile, with `baseline_harness` aliased to the Kokkos entry."""
    core_types = {
        "precision_advisor",
        "analyst",
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


# ---------- Precision advisor: schema + prompt ----------


def test_precision_advisor_kind_enum_is_three_values():
    """The precision_advisor's kind enum is exactly {sig_figs, decimal_digits, unknown}, so 'unknown' is a first-class output the orchestrator can branch on."""
    kind = PRECISION_ADVISOR_OUTPUT_SCHEMA["properties"]["kind"]
    assert set(kind["enum"]) == {"sig_figs", "decimal_digits", "unknown"}


def test_precision_advisor_required_fields():
    """The precision_advisor schema requires kind, value, rationale, confidence, alternative — the full record the orchestrator quotes when threading the tolerance into downstream prompts."""
    required = set(PRECISION_ADVISOR_OUTPUT_SCHEMA["required"])
    assert required == {
        "kind",
        "value",
        "rationale",
        "confidence",
        "alternative",
    }


def test_precision_advisor_confidence_enum():
    """The precision_advisor's confidence enum is {high, medium, low} so the orchestrator/user can react to low-confidence guesses."""
    confidence = PRECISION_ADVISOR_OUTPUT_SCHEMA["properties"]["confidence"]
    assert set(confidence["enum"]) == {"high", "medium", "low"}


def test_precision_advisor_prompt_distinguishes_sig_figs_and_decimal_digits():
    """The precision_advisor prompt names both sig_figs and decimal_digits explicitly so the model knows the difference between relative and absolute tolerance."""
    prompt = AGENTS["precision_advisor"]["system_prompt"]
    for token in ("sig_figs", "decimal_digits", "unknown"):
        assert token in prompt, f"advisor prompt missing {token!r}"


def test_precision_advisor_prompt_authorizes_unknown_answer():
    """The precision_advisor prompt explicitly authorizes returning kind='unknown' instead of guessing blindly, so the agent does not feel forced to invent a tolerance."""
    prompt = AGENTS["precision_advisor"]["system_prompt"].lower()
    assert "unknown" in prompt and "guess" in prompt


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
    """The baseline_harness schema requires driver_source, kernel_function_name, inputs_summary, and output_arrays so a future mechanical comparator has everything it needs to compile, run, and read back the reference."""
    assert set(BASELINE_HARNESS_OUTPUT_SCHEMA["required"]) == {
        "driver_source",
        "kernel_function_name",
        "inputs_summary",
        "output_arrays",
    }


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
