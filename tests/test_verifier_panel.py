"""Tests for workflow.verifier_panel.

Two layers:

- aggregate_verifier_verdicts — pure-function, no API. Rule-by-rule
  coverage of strict-verdict aggregation, faithfulness-lens
  per_variable ownership, and concerns union + dedup + lens-prefix.

- run_verifier_panel — uses fake_anthropic to verify lens-per-call
  dispatch (each lens gets its own suffix; all calls share the same
  task and temperature) and lens-order preservation of returned
  verdicts.

- VERIFIER_LENSES integrity — a tiny invariant check on the canonical
  lens list so a future edit cannot silently break the ordering
  convention the aggregator depends on (faithfulness must be index 0).
"""

import pytest

from workflow.verifier_panel import (
    VERIFIER_LENSES,
    aggregate_verifier_verdicts,
    run_verifier_panel,
)

from .conftest import FakeResponse, ToolUseBlock


def _verdict(
    verdict: str = "accept",
    per_variable: list[dict] | None = None,
    concerns: list[str] | None = None,
) -> dict:
    """Build a minimal verifier-shaped result dict for tests."""
    return {
        "verdict": verdict,
        "per_variable": per_variable or [],
        "concerns": concerns or [],
    }


def _pv(name: str, ok: bool = True, observed: str = "downcast") -> dict:
    """Build a minimal per_variable entry."""
    return {
        "name": name,
        "expected_action": "downcast",
        "observed_action": observed,
        "ok": ok,
        "note": "",
    }


# ---------- input validation ----------


def test_aggregate_raises_on_empty_input():
    """Aggregating zero verdicts raises ValueError (caller bug — the panel size cannot be zero)."""
    with pytest.raises(ValueError, match="zero verifier verdicts"):
        aggregate_verifier_verdicts([], [])


def test_aggregate_raises_when_verdicts_and_names_disagree():
    """verdicts and lens_names must be parallel — a length mismatch is a caller bug because lens names key the disagreement report and prefix concerns."""
    with pytest.raises(ValueError, match="same length"):
        aggregate_verifier_verdicts([_verdict()], ["faithfulness", "budget"])


# ---------- verdict: strict (all-accept required) ----------


def test_strict_accept_only_when_every_lens_accepts():
    """The aggregated verdict is accept iff every lens returned accept. Unanimous accept passes through; a single reject anywhere flips the whole panel."""
    v_a = _verdict("accept")
    agg, report = aggregate_verifier_verdicts(
        [v_a, v_a, v_a], ["faithfulness", "budget", "edge_cases"]
    )
    assert agg["verdict"] == "accept"
    assert report["dissenting_lenses"] == []


def test_any_lens_reject_flips_aggregate_to_reject():
    """A single lens rejecting forces the aggregate to reject — the panel is conservative by design (false-conservative is recoverable, false-aggressive is a silent precision regression)."""
    accept = _verdict("accept")
    reject = _verdict("reject", concerns=["budget too tight"])
    agg, report = aggregate_verifier_verdicts(
        [accept, reject, accept], ["faithfulness", "budget", "edge_cases"]
    )
    assert agg["verdict"] == "reject"
    assert report["dissenting_lenses"] == ["budget"]
    assert report["lens_verdicts"] == {
        "faithfulness": "accept",
        "budget": "reject",
        "edge_cases": "accept",
    }


def test_unknown_verdict_string_is_treated_as_reject():
    """A verdict value other than 'accept' (including missing or malformed) is treated as a dissent — strict accept means strict accept."""
    weird = {"verdict": "maybe", "per_variable": [], "concerns": []}
    accept = _verdict("accept")
    agg, report = aggregate_verifier_verdicts(
        [accept, weird], ["faithfulness", "budget"]
    )
    assert agg["verdict"] == "reject"
    assert "budget" in report["dissenting_lenses"]


# ---------- per_variable: faithfulness lens owns it ----------


def test_per_variable_taken_from_lens_zero_verbatim():
    """per_variable is taken verbatim from the faithfulness lens (lens 0). The other lenses' per_variable contents are discarded because the lenses are specialized, not redundant."""
    lens0 = _verdict(per_variable=[_pv("x"), _pv("y")])
    lens1 = _verdict(per_variable=[_pv("z")])  # ignored
    lens2 = _verdict(per_variable=[])  # ignored
    agg, _ = aggregate_verifier_verdicts(
        [lens0, lens1, lens2], ["faithfulness", "budget", "edge_cases"]
    )
    assert agg["per_variable"] == [_pv("x"), _pv("y")]


def test_per_variable_is_a_copy_not_a_reference():
    """The aggregated per_variable list is a copy of the lens-0 list, so a downstream mutation cannot retroactively contaminate the source verdict (a defensive choice that future code paths may rely on)."""
    lens0_list = [_pv("x")]
    lens0 = _verdict(per_variable=lens0_list)
    agg, _ = aggregate_verifier_verdicts(
        [lens0], ["faithfulness"]
    )
    agg["per_variable"].append(_pv("y"))
    assert lens0_list == [_pv("x")]


# ---------- concerns: union, deduped, lens-prefixed ----------


def test_concerns_union_across_all_lenses_with_lens_prefix():
    """concerns are the union across all lenses. Each entry is prefixed with [<lens_name>] so a rewriter-retry prompt can see which lens raised which concern (the primary win of the panel)."""
    lens0 = _verdict(concerns=["downcast on accumulator looks wrong"])
    lens1 = _verdict(concerns=["headroom_argument is hand-wavy"])
    lens2 = _verdict(concerns=["fails for inputs near zero"])
    agg, _ = aggregate_verifier_verdicts(
        [lens0, lens1, lens2], ["faithfulness", "budget", "edge_cases"]
    )
    assert agg["concerns"] == [
        "[faithfulness] downcast on accumulator looks wrong",
        "[budget] headroom_argument is hand-wavy",
        "[edge_cases] fails for inputs near zero",
    ]


def test_concerns_dedupe_keeps_first_lens_to_raise_each_string():
    """When two lenses raise the same concern (exact string after .strip()), only the first lens's prefixed entry survives. Lens order drives first-seen, so a concern first raised by faithfulness keeps the [faithfulness] prefix even if budget raises the same string."""
    lens0 = _verdict(concerns=["budget is tight"])
    lens1 = _verdict(concerns=["budget is tight  "])  # whitespace difference
    lens2 = _verdict(concerns=["budget is tight"])
    agg, _ = aggregate_verifier_verdicts(
        [lens0, lens1, lens2], ["faithfulness", "budget", "edge_cases"]
    )
    assert agg["concerns"] == ["[faithfulness] budget is tight"]


def test_concerns_empty_or_whitespace_only_are_dropped():
    """Empty strings and whitespace-only entries are dropped silently — they would clutter the rewriter feedback without adding signal."""
    lens0 = _verdict(concerns=["", "  ", "real concern"])
    agg, _ = aggregate_verifier_verdicts(
        [lens0], ["faithfulness"]
    )
    assert agg["concerns"] == ["[faithfulness] real concern"]


def test_concerns_missing_field_treated_as_empty_list():
    """A lens that returns no 'concerns' key (malformed result) is treated as raising no concerns — no exception, no contamination."""
    no_concerns_key = {"verdict": "accept", "per_variable": []}
    agg, _ = aggregate_verifier_verdicts(
        [no_concerns_key], ["faithfulness"]
    )
    assert agg["concerns"] == []


# ---------- disagreement report ----------


def test_report_records_k_lens_verdicts_and_concerns_by_lens():
    """The disagreement report carries k, the per-lens verdicts, the dissenting-lens list, and the raw per-lens concern lists (without lens prefixes — those are only in the aggregated output)."""
    lens0 = _verdict("accept", concerns=["c0"])
    lens1 = _verdict("reject", concerns=["c1"])
    _, report = aggregate_verifier_verdicts(
        [lens0, lens1], ["faithfulness", "budget"]
    )
    assert report == {
        "k": 2,
        "lens_verdicts": {"faithfulness": "accept", "budget": "reject"},
        "dissenting_lenses": ["budget"],
        "concerns_by_lens": {"faithfulness": ["c0"], "budget": ["c1"]},
    }


# ---------- output schema integrity ----------


def test_aggregated_output_has_all_required_top_level_keys():
    """The aggregated verdict carries every key VERIFIER_OUTPUT_SCHEMA requires, so the orchestrator's finish-gate and rewriter-retry code can consume it without schema-mismatch surprises."""
    agg, _ = aggregate_verifier_verdicts(
        [_verdict()], ["faithfulness"]
    )
    assert set(agg) == {"verdict", "per_variable", "concerns"}


# ---------- VERIFIER_LENSES integrity ----------


def test_verifier_lenses_canonical_order_starts_with_faithfulness():
    """The aggregator's per_variable rule depends on lens 0 being the faithfulness lens. A future edit that reorders the list (e.g. alphabetizes it) must not be silent — this test fails if it happens."""
    assert VERIFIER_LENSES[0]["name"] == "faithfulness"


def test_verifier_lenses_all_have_name_and_suffix():
    """Every lens entry has a non-empty 'name' and 'suffix' — the panel runner depends on both for dispatch and the aggregator depends on names for the disagreement report."""
    for lens in VERIFIER_LENSES:
        assert lens["name"]
        assert lens["suffix"]


def test_verifier_lens_names_are_unique():
    """Lens names key the disagreement report's lens_verdicts dict and prefix concerns; duplicates would silently merge lens identities."""
    names = [lens["name"] for lens in VERIFIER_LENSES]
    assert len(names) == len(set(names))


# ---------- run_verifier_panel ----------


def test_run_verifier_panel_dispatches_one_call_per_lens(
    fake_anthropic, monkeypatch
):
    """run_verifier_panel sends K calls to messages.create — one per lens — each carrying its own system suffix and all sharing the same task + temperature (with supports_temperature flipped on for verifier so the kwarg actually reaches the wire; the default registry value is False because Argo's claude-opus-4-7 rejects temperature)."""
    from workflow.registry import AGENTS
    monkeypatch.setitem(AGENTS["verifier"], "supports_temperature", True)

    payloads = [
        _verdict("accept", per_variable=[_pv(f"v{i}")], concerns=[f"c{i}"])
        for i in range(3)
    ]
    fake = fake_anthropic([
        FakeResponse(
            content=[ToolUseBlock(name="submit_result", input=p)],
            stop_reason="tool_use",
        )
        for p in payloads
    ])

    lenses = [
        {"name": "faithfulness", "suffix": "SUF_F"},
        {"name": "budget", "suffix": "SUF_B"},
        {"name": "edge_cases", "suffix": "SUF_E"},
    ]
    results = run_verifier_panel("TASK", lenses, temperature=0.7)

    assert len(results) == 3
    assert len(fake.messages.calls) == 3

    # Every call shared the same task + temperature.
    for call in fake.messages.calls:
        assert call["temperature"] == 0.7
        assert call["messages"][0]["content"] == "TASK"

    # The set of system prompts seen across the K calls includes each
    # lens suffix exactly once. (Thread-pool ordering is non-deterministic,
    # so we assert set equality rather than positional equality.)
    suffixes_seen = {call["system"].rsplit("\n\n", 1)[-1] for call in fake.messages.calls}
    assert suffixes_seen == {"SUF_F", "SUF_B", "SUF_E"}


def test_run_verifier_panel_rejects_empty_lens_list():
    """run_verifier_panel with no lenses is a caller bug — the panel has nothing to dispatch."""
    with pytest.raises(ValueError, match="zero lenses"):
        run_verifier_panel("task", [], temperature=0.7)
