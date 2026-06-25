"""Tests for the baseline_harness agent (registry + run_agent dispatch).

The baseline_harness agent is the first component of the planned dynamic
verifier: given a Kokkos C++ kernel, it produces a self-contained C++
driver source whose execution writes a reproducible reference output to
./reference.json. These tests exercise the agent through the generic
run_agent path (no network) and the registry shape, mirroring the
conventions of test_run_agent.py.
"""

from workflow.run_agent import run_agent

from .conftest import FakeResponse, ToolUseBlock


def test_run_agent_baseline_harness_returns_submit_result_input(fake_anthropic):
    """run_agent('baseline_harness', ...) returns the submit_result input and forces the baseline_harness output schema (v1: four per-precision drivers under `drivers`)."""
    payload = {
        "drivers": {
            "quad": (
                "// cd baselines/vector_add/ before running\n"
                "#include <Kokkos_Core.hpp>\n"
                "#include <quadmath.h>\n"
                "int main() { Kokkos::initialize(); Kokkos::finalize(); }\n"
            ),
            "double": (
                "// cd baselines/vector_add/ before running\n"
                "#include <Kokkos_Core.hpp>\n"
                "int main() { Kokkos::initialize(); Kokkos::finalize(); }\n"
            ),
            "float": (
                "// cd baselines/vector_add/ before running\n"
                "#include <Kokkos_Core.hpp>\n"
                "int main() { Kokkos::initialize(); Kokkos::finalize(); }\n"
            ),
            "mixed_io": (
                "// cd baselines/vector_add/ before running\n"
                "#include <Kokkos_Core.hpp>\n"
                "#include <quadmath.h>\n"
                "int main() { Kokkos::initialize(); Kokkos::finalize(); }\n"
            ),
        },
        "kernel_function_name": "vector_add",
        "inputs_summary": "N=16384, seed=42, x,y ~ U(-1,1)",
        "output_arrays": ["z"],
    }
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input=payload)],
            stop_reason="tool_use",
        ),
    ])

    result = run_agent("baseline_harness", "KERNEL SOURCE")

    assert result == payload
    assert len(fake.messages.calls) == 1
    call = fake.messages.calls[0]
    # The runner must force the agent to call submit_result.
    assert call["tool_choice"] == {"type": "tool", "name": "submit_result"}
    # The submit_result tool's input_schema must be the registry's
    # baseline_harness output_schema (verified by required-field set).
    tools = call["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "submit_result"
    assert set(tools[0]["input_schema"]["required"]) == {
        "drivers",
        "kernel_function_name",
        "inputs_summary",
        "output_arrays",
    }
    # The `drivers` object itself requires all four precision keys —
    # the schema rejects an absent or empty precision driver, which is
    # the only signal that prevents the harness from collapsing the
    # probe pipeline by silently omitting a variant.
    drivers_schema = tools[0]["input_schema"]["properties"]["drivers"]
    assert set(drivers_schema["required"]) == {
        "quad",
        "double",
        "float",
        "mixed_io",
    }
    # The user message passed to the model is exactly the task string.
    user_messages = call["messages"]
    assert user_messages == [{"role": "user", "content": "KERNEL SOURCE"}]
