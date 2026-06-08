"""Registry shape tests. No mocks, no network."""

from workflow.registry import AGENTS


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
