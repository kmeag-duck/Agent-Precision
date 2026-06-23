"""Perspective-diverse verifier panel.

The single-shot verifier sees the rewrite once through a generalist
lens; three identical verifier runs would all miss the same blindspot.
This module runs the same verifier agent K times in parallel, each
under a different *lens* (a system-prompt suffix that focuses the
verifier on one failure mode), then folds the K verdicts deterministic-
ally into a single verdict + disagreement report.

Three things live here:

  VERIFIER_LENSES
    Ordered list of {name, suffix} dicts. Order is the convention for
    per_variable source-of-truth: lens 0 is the faithfulness lens, and
    its per_variable list becomes the aggregated per_variable list
    verbatim. The other lenses focus on different axes (precision
    budget headroom, numerical edge cases) and we ignore their
    per_variable contents even though the schema forces them to fill
    it in.

  run_verifier_panel(task, lenses, temperature)
    Spawns one run_agent('verifier', task, ..., system_prompt_suffix=
    lens['suffix']) per lens in parallel via a ThreadPoolExecutor.
    Returns the K verdicts in lens order so the aggregator's source-
    of-truth conventions are deterministic.

  aggregate_verifier_verdicts(verdicts, lens_names) -> (agg, report)
    Folds K verdicts into one. Aggregation rules are strict by design:

      verdict
        STRICT — accept iff every lens returned accept. Any dissent
        flips to reject. Same conservative bias as the analyst
        aggregator's keep > emulate > downcast tiebreak: a false-
        conservative is recoverable (one extra trip through the
        dynamic-verification chain); a false-aggressive is a silent
        precision regression.

      per_variable
        Owned by the faithfulness lens (lens 0). Faithfulness is the
        only lens whose job is per-variable boundary-type inspection;
        the budget and edge_cases lenses care about cross-cutting
        properties. This is *not* a vote — the lenses are specialized,
        not redundant.

      concerns
        Union of all lenses' concerns, deduped by exact string match
        (after .strip()), preserving first-seen order across lens
        order. Each concern is prefixed with '[<lens_name>] ' so a
        rewriter-retry prompt can see which lens raised which concern.
        This is the primary win of the panel: the rewriter-retry
        feedback gets richer than any single lens could produce.

The disagreement report shape mirrors aggregator.py's:
  {
    "k": <int>,
    "lens_verdicts": {<lens_name>: "accept" | "reject", ...},
    "dissenting_lenses": [<lens_name>, ...],   # lenses that voted reject
    "concerns_by_lens": {<lens_name>: [<raw concern str>, ...], ...},
  }

It is emitted alongside the aggregated verdict so the orchestrator
trace can record which lens flipped the verdict.
"""

from concurrent.futures import ThreadPoolExecutor

from .run_agent import run_agent

VERIFIER_LENSES: list[dict] = [
    {
        "name": "faithfulness",
        "suffix": (
            "LENS: faithfulness.\n"
            "Focus on whether the rewritten source faithfully implements "
            "the analyst's per-variable verdict. For each variable named "
            "in the verdict, decide observed_action from the declared "
            "boundary type (argument type, View element type, local "
            "declared type) — not from internal casts. Treat a variable "
            "declared at original precision but cast to a narrower "
            "precision inside expressions as 'keep' (with ok=false if "
            "the analyst asked for 'downcast'). This is the lens that "
            "owns per_variable; fill it in carefully. Other lenses run "
            "in parallel and cover budget headroom and edge cases."
        ),
    },
    {
        "name": "budget",
        "suffix": (
            "LENS: precision_budget soundness.\n"
            "Focus on whether the analyst's precision_budget actually "
            "justifies meeting the tolerance under the rewritten verdict. "
            "Read claimed_output_precision and headroom_argument and ask: "
            "given where the dominant rounding error in the rewritten "
            "kernel comes from (downcast accumulators, emulated wider "
            "types, surviving doubles), does the argument hold? Is the "
            "claim tight against the tolerance? Is the headroom_argument "
            "missing, hand-wavy, or contradicted by the per-variable "
            "verdict? Raise every budget worry as a concerns entry. You "
            "must still fill in per_variable to satisfy the schema, but "
            "the faithfulness lens owns it — your per_variable will be "
            "ignored by the aggregator. Set verdict='reject' if the "
            "budget argument fails to justify the claim under the "
            "tolerance, even if the rewrite is faithful."
        ),
    },
    {
        "name": "edge_cases",
        "suffix": (
            "LENS: numerical edge cases.\n"
            "Focus on inputs and regimes that would expose precision "
            "loss the analyst did not anticipate. Think about: "
            "cancellation between nearby-magnitude values that survive "
            "into a downcast subtraction, accumulator drift over many "
            "steps when the accumulator was downcast or marked 'keep' "
            "without rework, subnormals and overflow at the new "
            "precision boundary, bounded-vs-unbounded magnitudes that "
            "no longer fit the narrower range, asymmetric inputs that "
            "the headroom argument did not consider. Raise every "
            "credible edge-case worry as a concerns entry. You must "
            "still fill in per_variable to satisfy the schema, but the "
            "faithfulness lens owns it — your per_variable will be "
            "ignored by the aggregator. Set verdict='reject' if you "
            "find a plausible input class that would push the "
            "rewritten kernel outside the tolerance, even if the "
            "rewrite is faithful and the budget argument is sound."
        ),
    },
]


def run_verifier_panel(
    task: str,
    lenses: list[dict],
    temperature: float,
) -> list[dict]:
    """Run `run_agent('verifier', task, ...)` once per lens in parallel.

    Returns the K verdicts in the same order as `lenses` so the
    aggregator's source-of-truth conventions (faithfulness owns
    per_variable, lens order drives concerns dedup) are deterministic.

    Failures in any single lens propagate (one exception fails the whole
    panel); the caller decides whether to retry the panel or fall back
    to single-shot. Same shape as run_agent_ensemble for symmetry.
    """
    if not lenses:
        raise ValueError("Cannot run a verifier panel with zero lenses")

    with ThreadPoolExecutor(max_workers=len(lenses)) as pool:
        futures = [
            pool.submit(
                run_agent,
                "verifier",
                task,
                temperature=temperature,
                system_prompt_suffix=lens["suffix"],
            )
            for lens in lenses
        ]
        return [f.result() for f in futures]


def aggregate_verifier_verdicts(
    verdicts: list[dict],
    lens_names: list[str],
) -> tuple[dict, dict]:
    """Fold K lensed verifier verdicts into one verdict + report.

    `verdicts` is the list returned by `run_verifier_panel`, in lens
    order. `lens_names` is the parallel list of lens names (e.g.
    ['faithfulness', 'budget', 'edge_cases']); names are used to prefix
    concerns and to key the disagreement report. The two lists must
    have the same length.

    Returns (aggregated_verdict, disagreement_report). The aggregated
    verdict conforms to VERIFIER_OUTPUT_SCHEMA (verdict, per_variable,
    concerns), so the orchestrator's existing finish-gate code and
    rewriter-retry prompt construction need no changes.
    """
    if not verdicts:
        raise ValueError("Cannot aggregate zero verifier verdicts")
    if len(verdicts) != len(lens_names):
        raise ValueError(
            f"verdicts ({len(verdicts)}) and lens_names "
            f"({len(lens_names)}) must have the same length"
        )

    # ---- verdict: strict (all-accept required) -----------------------
    per_lens_verdicts: dict[str, str] = {}
    dissenting: list[str] = []
    for name, v in zip(lens_names, verdicts):
        lens_verdict = v.get("verdict", "reject")
        per_lens_verdicts[name] = lens_verdict
        if lens_verdict != "accept":
            dissenting.append(name)
    agg_verdict = "accept" if not dissenting else "reject"

    # ---- per_variable: from faithfulness (lens 0) verbatim -----------
    # Schema requires per_variable on every verdict, so lens 0 always
    # has one; budget / edge_cases lenses also fill it in but we drop
    # those because the lenses are specialized, not redundant.
    agg_per_variable = list(verdicts[0].get("per_variable", []))

    # ---- concerns: union, deduped, prefixed by lens name -------------
    # First-seen order across lens order is what determines dedup
    # winners: a concern first surfaced by faithfulness keeps its
    # [faithfulness] prefix even if budget later raises the same string.
    seen: set[str] = set()
    agg_concerns: list[str] = []
    concerns_by_lens: dict[str, list[str]] = {}
    for name, v in zip(lens_names, verdicts):
        raw = list(v.get("concerns", []) or [])
        concerns_by_lens[name] = raw
        for concern in raw:
            key = concern.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            agg_concerns.append(f"[{name}] {key}")

    report: dict = {
        "k": len(verdicts),
        "lens_verdicts": per_lens_verdicts,
        "dissenting_lenses": dissenting,
        "concerns_by_lens": concerns_by_lens,
    }

    return (
        {
            "verdict": agg_verdict,
            "per_variable": agg_per_variable,
            "concerns": agg_concerns,
        },
        report,
    )
