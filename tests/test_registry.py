"""Registry shape tests. No mocks, no network."""

from workflow.registry import (
    AGENTS,
    ANALYST_OUTPUT_SCHEMA,
    BASELINE_HARNESS_OUTPUT_SCHEMA,
    PRECISION_ADVISOR_OUTPUT_SCHEMA,
    VERIFIER_OUTPUT_SCHEMA,
)


def test_known_agent_types():
    """AGENTS exposes exactly the precision_advisor, analyst, rewriter, verifier, and baseline_harness types."""
    assert set(AGENTS) == {
        "precision_advisor",
        "analyst",
        "rewriter",
        "verifier",
        "baseline_harness",
    }


def test_each_entry_has_required_keys():
    """Every registry entry declares system_prompt, output_schema, model."""
    for name, spec in AGENTS.items():
        assert "system_prompt" in spec, f"{name} missing system_prompt"
        assert "output_schema" in spec, f"{name} missing output_schema"
        assert "model" in spec, f"{name} missing model"


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
