"""Tests for workflow.aggregator.aggregate_analyst_verdicts.

Pure-function tests: no API calls, no monkeypatching. Exercises the
voting rules (majority, conservative tiebreak, omission-as-keep),
rework aggregation (strict-majority for True), precision_budget
source-of-truth selection, and disagreement report contents.
"""

import pytest

from workflow.aggregator import aggregate_analyst_verdicts


def _verdict(
    variables: list[dict] | None = None,
    rework: dict | None = None,
    precision_budget: dict | None = None,
    overall_notes: str = "",
) -> dict:
    """Build a minimal analyst-shaped verdict dict for tests."""
    return {
        "variables": variables or [],
        "rework": rework
        or {
            "suggested": False,
            "transformation": "",
            "rationale": "",
            "affected_variables": [],
        },
        "precision_budget": precision_budget
        or {
            "target_kind": "sig_figs",
            "target_value": 6,
            "source": "user_cli",
            "claimed_output_precision": "~7 sig figs",
            "headroom_argument": "ok",
        },
        "overall_notes": overall_notes,
    }


def _var(
    name: str,
    action: str,
    target_precision: str = "",
    emulation_type: str = "",
    reason: str = "",
) -> dict:
    return {
        "name": name,
        "action": action,
        "target_precision": target_precision,
        "emulation_type": emulation_type,
        "reason": reason,
    }


# ---------- input validation ----------


def test_raises_on_empty_input():
    """Aggregating zero verdicts raises ValueError (caller bug)."""
    with pytest.raises(ValueError, match="zero verdicts"):
        aggregate_analyst_verdicts([])


# ---------- variables: unanimous ----------


def test_unanimous_per_variable_keeps_action_and_aux_fields():
    """When all K verdicts agree on a variable, action / target_precision / emulation_type / reason are taken verbatim and no disagreement is reported."""
    v = _verdict(
        variables=[_var("x", "downcast", target_precision="float", reason="bounded")]
    )
    agg, report = aggregate_analyst_verdicts([v, v, v])
    assert agg["variables"] == [
        _var("x", "downcast", target_precision="float", reason="bounded")
    ]
    assert report["variable_disagreements"] == {}


# ---------- variables: majority ----------


def test_majority_action_wins_over_minority():
    """A 2-vs-1 split on action picks the majority and records the split in the disagreement report."""
    v_down = _verdict(variables=[_var("x", "downcast", target_precision="float")])
    v_keep = _verdict(variables=[_var("x", "keep")])
    agg, report = aggregate_analyst_verdicts([v_down, v_down, v_keep])
    assert agg["variables"][0]["action"] == "downcast"
    assert agg["variables"][0]["target_precision"] == "float"
    assert report["variable_disagreements"]["x"] == {
        "action_votes": {"downcast": 2, "keep": 1},
        "winning_action": "downcast",
    }


# ---------- variables: conservative tiebreak ----------


def test_tiebreak_prefers_keep_over_emulate_over_downcast():
    """A K=3 three-way tie (1 each) resolves to 'keep' (the most conservative). K=4 split 2-2 between downcast and emulate resolves to 'emulate'."""
    v_d = _verdict(variables=[_var("x", "downcast", target_precision="float")])
    v_e = _verdict(variables=[_var("x", "emulate", emulation_type="float-float")])
    v_k = _verdict(variables=[_var("x", "keep")])
    agg, _ = aggregate_analyst_verdicts([v_d, v_e, v_k])
    assert agg["variables"][0]["action"] == "keep"

    agg2, _ = aggregate_analyst_verdicts([v_d, v_d, v_e, v_e])
    assert agg2["variables"][0]["action"] == "emulate"
    assert agg2["variables"][0]["emulation_type"] == "float-float"


# ---------- variables: omission counts as keep ----------


def test_variable_omitted_by_some_verdicts_counts_as_keep_vote():
    """A variable named by some verdicts but omitted by others is included in the output; omissions count as 'keep' votes (conservative). 1 downcast + 2 omissions resolves to 'keep'."""
    v_down = _verdict(variables=[_var("x", "downcast", target_precision="float")])
    v_empty = _verdict(variables=[])
    agg, report = aggregate_analyst_verdicts([v_down, v_empty, v_empty])
    assert agg["variables"][0]["action"] == "keep"
    # Disagreement is recorded because the votes were not unanimous.
    assert "x" in report["variable_disagreements"]
    assert report["variable_disagreements"]["x"]["action_votes"] == {
        "downcast": 1,
        "keep": 2,
    }


# ---------- variables: target_precision picked only from winning-action voters ----------


def test_target_precision_modal_among_winning_action_voters_only():
    """When 'downcast' wins, target_precision is the modal value among the downcast voters — voters who chose a different action (and thus have target_precision='') do not contaminate the choice."""
    v_d_float = _verdict(
        variables=[_var("x", "downcast", target_precision="float")]
    )
    v_d_half = _verdict(
        variables=[_var("x", "downcast", target_precision="half")]
    )
    v_k = _verdict(variables=[_var("x", "keep")])  # target_precision=""
    # 2 votes 'downcast' (float, half), 1 vote 'keep'. Downcast wins.
    # Among downcast voters: float and half tie 1-1; first input wins -> float.
    agg, _ = aggregate_analyst_verdicts([v_d_float, v_d_half, v_k])
    assert agg["variables"][0]["action"] == "downcast"
    assert agg["variables"][0]["target_precision"] == "float"


# ---------- variables: name normalization ----------


def test_variable_names_normalized_for_voting_but_raw_form_preserved():
    """Names differing only in case / whitespace vote together; the most-common raw spelling is used in the output."""
    v1 = _verdict(variables=[_var("Xx", "downcast", target_precision="float")])
    v2 = _verdict(variables=[_var(" xx ", "downcast", target_precision="float")])
    v3 = _verdict(variables=[_var("xx", "downcast", target_precision="float")])
    agg, report = aggregate_analyst_verdicts([v1, v2, v3])
    assert len(agg["variables"]) == 1
    # Each raw form appears once; first by Counter.most_common stability is 'Xx'.
    assert agg["variables"][0]["name"] in {"Xx", " xx ", "xx"}
    assert report["variable_disagreements"] == {}


# ---------- variables: deterministic output order ----------


def test_variable_output_order_is_sorted_for_determinism():
    """Aggregated variables are emitted in sorted (normalized) name order so identical inputs in different orderings still produce identical aggregated dicts."""
    v = _verdict(variables=[_var("z", "keep"), _var("a", "keep")])
    agg, _ = aggregate_analyst_verdicts([v])
    assert [e["name"] for e in agg["variables"]] == ["a", "z"]


# ---------- rework: strict majority for True ----------


def test_rework_requires_strict_majority_for_suggested_true():
    """rework.suggested wins only on a strict majority. K=3 with 2 true -> True (2 > 1.5). K=4 with 2 true -> False (2 > 2.0 is False)."""
    yes = _verdict(
        rework={
            "suggested": True,
            "transformation": "Kahan",
            "rationale": "summation drift",
            "affected_variables": ["s"],
        }
    )
    no = _verdict()  # suggested=False
    agg_3, _ = aggregate_analyst_verdicts([yes, yes, no])
    assert agg_3["rework"]["suggested"] is True
    assert agg_3["rework"]["transformation"] == "Kahan"

    agg_4, _ = aggregate_analyst_verdicts([yes, yes, no, no])
    assert agg_4["rework"]["suggested"] is False
    assert agg_4["rework"]["transformation"] == ""
    assert agg_4["rework"]["affected_variables"] == []


def test_rework_picks_modal_transformation_and_matching_prose():
    """When suggested wins, transformation is the modal string among the true voters, and rationale + affected_variables are taken from the first voter whose transformation matches that modal string."""
    kahan_v1 = _verdict(
        rework={
            "suggested": True,
            "transformation": "Kahan",
            "rationale": "first",
            "affected_variables": ["s"],
        }
    )
    kahan_v2 = _verdict(
        rework={
            "suggested": True,
            "transformation": "Kahan",
            "rationale": "second",
            "affected_variables": ["s", "t"],
        }
    )
    other = _verdict(
        rework={
            "suggested": True,
            "transformation": "pairwise",
            "rationale": "other",
            "affected_variables": ["u"],
        }
    )
    agg, _ = aggregate_analyst_verdicts([other, kahan_v1, kahan_v2])
    assert agg["rework"]["suggested"] is True
    assert agg["rework"]["transformation"] == "Kahan"
    # rationale comes from FIRST verdict whose transformation matches the modal one.
    assert agg["rework"]["rationale"] == "first"
    assert agg["rework"]["affected_variables"] == ["s"]


def test_rework_disagreement_recorded_only_when_split():
    """The rework_disagreement key appears only when the rework vote was not unanimous."""
    yes = _verdict(
        rework={
            "suggested": True,
            "transformation": "Kahan",
            "rationale": "r",
            "affected_variables": [],
        }
    )
    no = _verdict()
    _, report_split = aggregate_analyst_verdicts([yes, yes, no])
    assert report_split["rework_disagreement"] == {
        "true": 2,
        "false": 1,
        "winning": True,
    }
    _, report_unan = aggregate_analyst_verdicts([no, no, no])
    assert "rework_disagreement" not in report_unan


# ---------- precision_budget + overall_notes: most-aligned source ----------


def test_precision_budget_taken_from_most_aligned_verdict():
    """precision_budget + overall_notes are taken from the verdict whose per-variable answers most agree with the aggregated consensus. Index of that source verdict is reported."""
    common_budget = {
        "target_kind": "sig_figs",
        "target_value": 6,
        "source": "user_cli",
    }
    # Three verdicts on two variables x, y.
    # Consensus will be x=downcast (2-1), y=keep (3-0).
    # Verdict 0: x=downcast, y=keep  -> aligns with both: score 2.
    # Verdict 1: x=downcast, y=keep  -> aligns with both: score 2.
    # Verdict 2: x=keep, y=keep      -> aligns only on y: score 1.
    # Tie between 0 and 1 -> first wins.
    v0 = _verdict(
        variables=[_var("x", "downcast", "float"), _var("y", "keep")],
        precision_budget={
            **common_budget,
            "claimed_output_precision": "P0",
            "headroom_argument": "H0",
        },
        overall_notes="N0",
    )
    v1 = _verdict(
        variables=[_var("x", "downcast", "float"), _var("y", "keep")],
        precision_budget={
            **common_budget,
            "claimed_output_precision": "P1",
            "headroom_argument": "H1",
        },
        overall_notes="N1",
    )
    v2 = _verdict(
        variables=[_var("x", "keep"), _var("y", "keep")],
        precision_budget={
            **common_budget,
            "claimed_output_precision": "P2",
            "headroom_argument": "H2",
        },
        overall_notes="N2",
    )
    agg, report = aggregate_analyst_verdicts([v0, v1, v2])
    assert agg["precision_budget"]["claimed_output_precision"] == "P0"
    assert agg["precision_budget"]["headroom_argument"] == "H0"
    assert agg["overall_notes"] == "N0"
    assert report["consensus_source_verdict_index"] == 0


# ---------- report: shape and k ----------


def test_report_records_k_and_consensus_source_even_on_unanimous():
    """The report always carries k (the ensemble size) and the chosen consensus source index, even when nothing was disagreed on."""
    v = _verdict(variables=[_var("x", "keep")])
    _, report = aggregate_analyst_verdicts([v, v])
    assert report["k"] == 2
    assert report["consensus_source_verdict_index"] in (0, 1)
    assert report["variable_disagreements"] == {}
    assert "rework_disagreement" not in report


# ---------- output schema integrity ----------


def test_aggregated_output_has_all_required_top_level_keys():
    """The aggregated verdict carries every key the analyst schema requires, so the downstream verifier can consume it without schema-mismatch surprises."""
    v = _verdict(variables=[_var("x", "keep")])
    agg, _ = aggregate_analyst_verdicts([v])
    assert set(agg) == {"variables", "rework", "precision_budget", "overall_notes"}
    assert set(agg["rework"]) == {
        "suggested",
        "transformation",
        "rationale",
        "affected_variables",
    }
    assert {
        "target_kind",
        "target_value",
        "source",
        "claimed_output_precision",
        "headroom_argument",
    } <= set(agg["precision_budget"])
