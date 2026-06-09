"""Registry shape tests. No mocks, no network."""

from workflow.registry import AGENTS, ANALYST_OUTPUT_SCHEMA, VERIFIER_OUTPUT_SCHEMA


def test_known_agent_types():
    """AGENTS exposes exactly the analyst, rewriter, and verifier types."""
    assert set(AGENTS) == {"analyst", "rewriter", "verifier"}


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
