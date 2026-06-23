"""Deterministic aggregator that folds K independent analyst verdicts
into a single verdict via per-variable majority vote.

Used by `run_agent_ensemble` to turn ensemble noise into a stable
decision. The aggregated output conforms to the analyst's output schema
(see registry.ANALYST_OUTPUT_SCHEMA), so the verifier and rewriter
downstream do not need to change.

Aggregation rules (all deterministic given input order):

  Variables
  - Names are normalized (lowercase + strip) before voting. The
    aggregated entry uses the most-commonly-spelled raw form.
  - A variable that appears in fewer than K verdicts is treated as if
    the missing verdicts voted 'keep'. This is the conservative default
    and matches what would happen in practice (the rewriter would leave
    an un-mentioned variable alone).
  - Action vote is plurality; ties resolve to the most conservative
    action under keep > emulate > downcast. Conservative tiebreaks bias
    the system toward safety; the dynamic-verification chain
    (compare_outputs) will catch any false-conservative that produced a
    pointlessly slow rewrite, but a false-aggressive can produce a
    silent precision regression.
  - target_precision / emulation_type are taken as the modal value
    among verdicts that voted for the WINNING action (so a 'downcast'
    winner doesn't inherit a target_precision from a 'keep' voter,
    which would have target_precision='').
  - reason is taken from the first verdict (by input order) that voted
    for the winning action.

  Rework
  - rework.suggested is a STRICT majority vote (`sum > K/2`); ties
    default to False. Rationale: rework is the more aggressive choice
    (Kahan rewriting, reformulation, etc.), so requiring a strict
    majority avoids forcing a kernel-shape change on the rewriter when
    the analyst panel was actually split.
  - If suggested wins, transformation is the modal string among the
    true-suggested verdicts, and rationale + affected_variables are
    taken from the first verdict whose transformation matches that
    modal string.
  - If suggested loses, all rework fields are the explicit "no rework"
    sentinel (suggested=False, transformation='', rationale='',
    affected_variables=[]) regardless of what the minority voters said.

  Precision budget + overall_notes
  - target_kind / target_value / source are echoed from the task and
    should agree across all K verdicts. Pick from the most-aligned
    verdict (see below); a disagreement here is logged in the report
    but does not block aggregation.
  - claimed_output_precision and headroom_argument are taken from the
    "most-aligned" verdict — the one whose per-variable answers match
    the aggregated consensus on the most variables. Ties broken by
    input order. This keeps the budget block internally consistent
    with the chosen verdict instead of stitching prose from
    incompatible sources.
  - overall_notes is taken from the same most-aligned verdict.

The disagreement report (returned alongside the aggregated verdict) is
intended for the orchestrator trace, not for downstream agents — it
lists variables whose action vote was split, whether the rework vote
was split, and which input verdict supplied the precision_budget +
overall_notes prose. Reading the report is how the operator detects
borderline cases that may warrant a re-run with a different prompt.
"""

from collections import Counter

# Lower = more conservative. Tiebreaks on action votes pick the LOWEST
# value: keep > emulate > downcast. See module docstring for why.
_ACTION_CONSERVATISM = {"keep": 0, "emulate": 1, "downcast": 2}

_VALID_ACTIONS = frozenset(_ACTION_CONSERVATISM)

_EMPTY_REWORK = {
    "suggested": False,
    "transformation": "",
    "rationale": "",
    "affected_variables": [],
}


def _normalize(name: str) -> str:
    return name.lower().strip()


def aggregate_analyst_verdicts(
    verdicts: list[dict],
) -> tuple[dict, dict]:
    """Fold K analyst verdicts into one via per-variable majority vote.

    `verdicts` is a list of analyst result dicts as returned by
    `run_agent("analyst", ...)`. Order matters only for deterministic
    tiebreaks (modal-tie picks the first occurrence).

    Returns (aggregated_verdict, disagreement_report). The aggregated
    verdict conforms to the analyst output schema. The disagreement
    report is metadata for the orchestrator trace:
      {
        "k": <int>,
        "variable_disagreements": {
          "<name>": {"action_votes": {<action>: <count>, ...},
                     "winning_action": "<action>"},
          ...
        },
        "rework_disagreement": {"true": <int>, "false": <int>,
                                "winning": <bool>}  # omitted if unanimous
        "consensus_source_verdict_index": <int>,
      }
    """
    if not verdicts:
        raise ValueError("Cannot aggregate zero verdicts")

    # ---- per-variable vote ---------------------------------------------
    # Map normalized name -> list of raw names (for choosing display form)
    name_raw_forms: dict[str, list[str]] = {}
    for v in verdicts:
        for entry in v.get("variables", []):
            norm = _normalize(entry["name"])
            name_raw_forms.setdefault(norm, []).append(entry["name"])

    agg_variables: list[dict] = []
    variable_disagreements: dict[str, dict] = {}

    for norm in sorted(name_raw_forms):
        display = Counter(name_raw_forms[norm]).most_common(1)[0][0]
        # Per-verdict entry for this variable (None if the verdict omitted it)
        per_verdict = [
            next(
                (
                    e
                    for e in v.get("variables", [])
                    if _normalize(e["name"]) == norm
                ),
                None,
            )
            for v in verdicts
        ]
        # Omissions count as 'keep' votes; unknown actions are coerced to
        # 'keep' too, since an action outside the enum cannot be acted on
        # by the rewriter and conservative is the safe direction.
        action_votes = [
            e["action"] if e and e.get("action") in _VALID_ACTIONS else "keep"
            for e in per_verdict
        ]
        counts = Counter(action_votes)
        top = max(counts.values())
        winners = [a for a, c in counts.items() if c == top]
        winning_action = min(winners, key=lambda a: _ACTION_CONSERVATISM[a])

        winning_entries = [
            e
            for e, a in zip(per_verdict, action_votes)
            if a == winning_action and e is not None
        ]
        if winning_entries:
            target_precision = Counter(
                e.get("target_precision", "") for e in winning_entries
            ).most_common(1)[0][0]
            emulation_type = Counter(
                e.get("emulation_type", "") for e in winning_entries
            ).most_common(1)[0][0]
            reason = winning_entries[0].get("reason", "")
        else:
            # Winning action was 'keep' driven entirely by omissions
            target_precision = ""
            emulation_type = ""
            reason = ""

        agg_variables.append(
            {
                "name": display,
                "action": winning_action,
                "target_precision": target_precision,
                "emulation_type": emulation_type,
                "reason": reason,
            }
        )

        if len(counts) > 1:
            variable_disagreements[display] = {
                "action_votes": dict(counts),
                "winning_action": winning_action,
            }

    # ---- rework vote ---------------------------------------------------
    rework_votes = [
        bool(v.get("rework", {}).get("suggested", False)) for v in verdicts
    ]
    true_count = sum(rework_votes)
    # Strict majority required for True (ties → False). For K=3, 2 trues
    # win (2 > 1.5). For K=4, 2 trues lose (2 > 2.0 is False).
    rework_suggested = true_count > len(verdicts) / 2
    if rework_suggested:
        true_reworks = [
            v["rework"]
            for v, vote in zip(verdicts, rework_votes)
            if vote
        ]
        modal_transformation = Counter(
            r.get("transformation", "") for r in true_reworks
        ).most_common(1)[0][0]
        chosen = next(
            r for r in true_reworks
            if r.get("transformation", "") == modal_transformation
        )
        agg_rework = {
            "suggested": True,
            "transformation": chosen.get("transformation", ""),
            "rationale": chosen.get("rationale", ""),
            "affected_variables": list(chosen.get("affected_variables", [])),
        }
    else:
        agg_rework = dict(_EMPTY_REWORK)
        agg_rework["affected_variables"] = []  # fresh list per call

    # ---- pick the most-aligned source for budget + notes ----------------
    def alignment_score(verdict: dict) -> int:
        score = 0
        verdict_actions = {
            _normalize(e["name"]): e.get("action", "keep")
            for e in verdict.get("variables", [])
        }
        for agg_entry in agg_variables:
            norm = _normalize(agg_entry["name"])
            voted = verdict_actions.get(norm, "keep")
            if voted == agg_entry["action"]:
                score += 1
        return score

    scored = [(alignment_score(v), -i, v, i) for i, v in enumerate(verdicts)]
    # max by (score, -i) → highest score, then earliest index on tie
    _, _, best_verdict, best_index = max(scored)

    src_budget = best_verdict.get("precision_budget", {}) or {}
    agg_precision_budget = {
        "target_kind": src_budget.get("target_kind", ""),
        "target_value": src_budget.get("target_value", 0),
        "source": src_budget.get("source", ""),
        "claimed_output_precision": src_budget.get(
            "claimed_output_precision", ""
        ),
        "headroom_argument": src_budget.get("headroom_argument", ""),
    }
    agg_overall_notes = best_verdict.get("overall_notes", "")

    # ---- assemble report ------------------------------------------------
    report: dict = {
        "k": len(verdicts),
        "variable_disagreements": variable_disagreements,
        "consensus_source_verdict_index": best_index,
    }
    if len(set(rework_votes)) > 1:
        report["rework_disagreement"] = {
            "true": true_count,
            "false": len(verdicts) - true_count,
            "winning": rework_suggested,
        }

    return (
        {
            "variables": agg_variables,
            "rework": agg_rework,
            "precision_budget": agg_precision_budget,
            "overall_notes": agg_overall_notes,
        },
        report,
    )
