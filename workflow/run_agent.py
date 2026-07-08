"""Generic agent runner — the only place where agent calls hit the API.

run_agent(type, task) -> dict

Looks up `type` in the registry, forces the agent to call a `submit_result`
tool whose input schema matches the agent's declared output_schema, and
returns the parsed tool input.

Adding a new agent type = adding an entry to registry.AGENTS — no changes here.

run_agent_ensemble(type, task, k, temperature) -> list[dict]

Runs `run_agent` K times in parallel (I/O-bound, thread pool is fine)
and returns the K result dicts in submission order. Callers feed those
results into aggregator.aggregate_analyst_verdicts (or an equivalent
type-specific aggregator) to fold ensemble noise into a stable decision.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import anthropic

from .registry import AGENTS


# One-shot guard so we warn at most once per process when a caller
# requests a temperature for a model whose registry entry declares
# `supports_temperature: False`. The kwarg is silently dropped on the
# wire (it'd HTTP-400 otherwise), but the operator deserves to know
# their ensemble's diversity knob isn't actually doing anything.
_TEMPERATURE_DROP_WARNED: set[str] = set()


DEFAULT_SINGLE_SHOT_TEMPERATURE = 0.0
"""Temperature applied to every single-shot agent call.

The single-shot path used to omit `temperature` entirely, which made
the Anthropic SDK fall back to the API default (historically ~1.0).
That produced run-to-run divergence on the harder kernels (especially
the `mixed` ones, where two defensible per-variable verdicts compete).
0.0 is the right default for the K=1 single-shot path: we want
reproducibility, not diversity. Ensemble callers pass their own
temperature (typically 0.7) to deliberately recover diversity for
the aggregator to fold. Override by passing an explicit `temperature`
keyword to `run_agent`."""


def run_agent(
    type: str,
    task: str,
    temperature: float | None = None,
    system_prompt_suffix: str | None = None,
) -> dict:
    """Run one agent and return its structured result.

    `temperature` defaults to None. When None, this function uses
    DEFAULT_SINGLE_SHOT_TEMPERATURE (currently 0.0) so single-shot
    runs are reproducible across invocations. Pass an explicit float
    to override; ensemble callers pass 0.7 for diversity, callers
    that want the raw API default must pass it themselves.

    Historical note: the prior contract was "None means omit
    temperature, let the API default apply", which read as
    deterministic-by-default but actually produced high-variance
    output. The new contract trades that hidden default for an
    explicit one so the single-shot path is reproducible without the
    caller having to know.

    `system_prompt_suffix` defaults to None, which preserves the
    existing system prompt verbatim. Pass a non-empty string to
    append a per-call addendum after the registry's base prompt
    (separated by a blank line). This is how the verifier panel
    runs the same agent under different lenses — each lens is a
    suffix that focuses the verifier on one failure mode
    (faithfulness, precision_budget, edge cases) without forking
    the base agent definition. The base prompt stays the single
    source of truth.
    """
    if type not in AGENTS:
        raise ValueError(
            f"Unknown agent type: {type!r}. Known: {list(AGENTS)}"
        )
    spec = AGENTS[type]

    submit_result_tool = {
        "name": "submit_result",
        "description": (
            "Submit your final structured result for this task. "
            "Calling this tool ends your turn."
        ),
        "input_schema": spec["output_schema"],
    }

    system_prompt = spec["system_prompt"]
    if system_prompt_suffix is not None:
        system_prompt = f"{system_prompt}\n\n{system_prompt_suffix}"

    # max_tokens ceiling for any single agent call. Bumped from 8192
    # to 32768 because the Kokkos v1 baseline_harness emits FOUR full
    # per-precision drivers in one submit_result call (see
    # workflow/languages/kokkos.py:BASELINE_HARNESS_OUTPUT_SCHEMA),
    # and a single Kokkos driver is already 100-200 lines of source —
    # four of them easily blow past 8192 output tokens. When the model
    # truncates mid-tool-use it returns `stop_reason='max_tokens'`
    # with a partial (often empty) tool_use input, which then surfaces
    # downstream as a baffling "submit_result payload has no keys"
    # RuntimeError in orchestrator._execute_tool. 32768 matches Argo's
    # documented opus-4-7 output cap; smaller agents (analyst, verifier,
    # etc.) cost nothing extra because max_tokens is a ceiling, not a
    # target. If a future agent type genuinely needs more, lift this
    # to a per-entry registry field rather than bumping the global
    # again.
    # Explicit per-request timeout (seconds). The SDK refuses
    # non-streaming requests when its own conservative estimate
    # (a function of `max_tokens`) exceeds 10 minutes, raising
    # `ValueError("Streaming is required for operations that may
    # take longer than 10 minutes.")` at create() time. With
    # `max_tokens=32768` we trip that guard even though our actual
    # response times are seconds, not minutes. Passing an explicit
    # `timeout` is the SDK-sanctioned escape hatch — it signals
    # "operator accepts responsibility for this duration" and the
    # guard is skipped. 600s matches AGENT_PRECISION_ORCHESTRATOR_
    # TIMEOUT_SEC's default in the orchestrator and is the upper
    # bound on how long any single tool call should ever take in
    # this workflow.
    create_kwargs = {
        "model": spec["model"],
        "max_tokens": 32768,
        "system": system_prompt,
        "tools": [submit_result_tool],
        "tool_choice": {"type": "tool", "name": "submit_result"},
        "messages": [{"role": "user", "content": task}],
        "timeout": 600.0,
    }

    # Temperature handling. Some model snapshots reject the
    # `temperature` kwarg outright (HTTP 400
    # "temperature is deprecated for this model" — Argo's
    # `claude-opus-4-7` snapshot is the motivating case), so the
    # per-entry `supports_temperature` flag in the registry gates
    # whether we forward it at all. The current registry uses
    # `claude-sonnet-4-6`, which accepts temperature, so every entry
    # sets True; the drop path stays wired for the day a future
    # registry entry points at a model that doesn't. When the flag is
    # False, the kwarg is dropped — including the single-shot default
    # of 0.0 — and the model's internal sampling is used. If an
    # explicit temperature was requested but dropped, warn once per
    # process per agent type: this is the only signal the operator
    # gets that K-of-K ensemble calls aren't actually diversified by
    # temperature anymore.
    supports_temperature = spec.get("supports_temperature", False)
    if supports_temperature:
        effective_temperature = (
            DEFAULT_SINGLE_SHOT_TEMPERATURE
            if temperature is None
            else temperature
        )
        create_kwargs["temperature"] = effective_temperature
    elif temperature is not None and type not in _TEMPERATURE_DROP_WARNED:
        print(
            f"[run_agent] warning: temperature={temperature} requested "
            f"for agent {type!r} (model {spec['model']!r}), but the "
            f"registry entry has supports_temperature=False. The kwarg "
            f"is being dropped to avoid an HTTP 400. Ensemble diversity "
            f"is reduced to whatever internal stochasticity the model "
            f"provides. This warning is shown once per agent type per "
            f"process.",
            file=sys.stderr,
        )
        _TEMPERATURE_DROP_WARNED.add(type)

    client = anthropic.Anthropic()
    response = client.messages.create(**create_kwargs)

    # Defensive guard against backends (notably some Argo proxy
    # configurations) that return HTTP 200 with a body the SDK
    # accepts but cannot fully unmarshal into content blocks. The
    # SDK's Message.content is typed list[ContentBlock] and should
    # be []-on-empty, not None; if we see None we want a loud error
    # naming the response, not a downstream TypeError.
    if response.content is None:
        raise RuntimeError(
            f"Agent {type!r} received a response with content=None. "
            f"stop_reason={response.stop_reason}, "
            f"response_id={getattr(response, 'id', '<unknown>')}. "
            f"This usually indicates a backend/proxy returned a "
            f"malformed message body; retry, or inspect the proxy logs."
        )

    # With tool_choice forcing submit_result, the expected stop_reason
    # is "tool_use". Any other value is a smoking gun for downstream
    # weirdness: "max_tokens" means the model truncated mid-tool-use
    # and the submit_result input dict is almost certainly partial /
    # empty (this is the primary failure mode that prompted the
    # max_tokens=32768 bump above — kept as a runtime tripwire in
    # case a future schema regrows past the new ceiling, or in case a
    # proxy quietly caps output tokens below what we requested).
    # "end_turn" means the model decided it was done without calling
    # the forced tool, which contradicts tool_choice and points at a
    # proxy that's stripping or rewriting the tool_choice field.
    # Warn loudly with the diagnostic context — the actual
    # "submit_result missing" or "empty input dict" failure will
    # surface a few lines below this; this print just makes the
    # WHY visible without rerunning.
    if response.stop_reason != "tool_use":
        usage = getattr(response, "usage", None)
        print(
            f"[run_agent] warning: agent {type!r} returned "
            f"stop_reason={response.stop_reason!r} (expected 'tool_use'). "
            f"response_id={getattr(response, 'id', '<unknown>')}, "
            f"usage={usage}. "
            f"If stop_reason is 'max_tokens', the submit_result input "
            f"is likely truncated and the next failure will be a "
            f"missing or empty payload — raise max_tokens in "
            f"workflow/run_agent.py.",
            file=sys.stderr,
        )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_result":
            return _coerce_tool_input_to_dict(
                block.input, type, getattr(response, "id", "<unknown>")
            )

    raise RuntimeError(
        f"Agent {type!r} did not call submit_result. "
        f"stop_reason={response.stop_reason}, "
        f"blocks={[b.type for b in response.content]}"
    )


def _coerce_tool_input_to_dict(
    raw: object, agent_type: str, response_id: str
) -> dict:
    """Normalize `tool_use.input` payloads that come in as JSON strings.

    The Anthropic SDK types `ContentBlock.input` as `dict`, and on
    api.anthropic.com the field is always a decoded object. Some Argo
    proxy paths (notably the `:8083` `claude-argo-proxy.py` transparent
    shim) forward the upstream body without re-parsing, and the
    upstream occasionally emits `input` as a JSON-encoded string.
    Downstream code (aggregator, orchestrator tool-result plumbing) all
    assumes a dict, so a string reaches
    `aggregate_analyst_verdicts` as an `'str' object has no attribute
    'get'` mystery-traceback deep inside the K-fold — that was the
    concrete failure mode that motivated this guard on the K=3
    nbody_force retry run (2026-07-01).

    Silent JSON-decode is the deliberate policy: the upstream payload
    is well-formed 99% of the time, the proxy quirk is transient, and
    turning it into a hard crash would make every K>1 Argo run
    fragile. If the string is not valid JSON, or decodes to something
    other than a dict, we raise a RuntimeError naming the agent and
    response id (same idiom as the `content=None` guard above) so the
    failure is diagnosable at the source instead of six frames deep.
    """
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Agent {agent_type!r} returned tool_use.input as a "
                f"string that is not valid JSON: {exc}. "
                f"response_id={response_id}, "
                f"payload_preview={raw[:200]!r}. "
                f"This usually indicates a proxy is forwarding an "
                f"unparsed upstream body; retry, or inspect the proxy "
                f"logs."
            ) from exc
        if not isinstance(decoded, dict):
            raise RuntimeError(
                f"Agent {agent_type!r} returned tool_use.input as a "
                f"JSON-encoded {type(decoded).__name__}, expected object. "
                f"response_id={response_id}, "
                f"decoded_preview={str(decoded)[:200]!r}."
            )
        return decoded
    raise RuntimeError(
        f"Agent {agent_type!r} returned tool_use.input of type "
        f"{type(raw).__name__}, expected dict or JSON-string. "
        f"response_id={response_id}."
    )


def run_agent_ensemble(
    type: str,
    task: str,
    k: int,
    temperature: float,
) -> list[dict]:
    """Run `run_agent(type, task, temperature)` K times in parallel.

    Returns the K result dicts in submission (index) order, so callers
    that care about ordering (e.g. tiebreak-by-first-input rules in the
    aggregator) see a deterministic sequence.

    The calls are I/O-bound on the Anthropic API, so a ThreadPoolExecutor
    is the right concurrency primitive — no GIL contention worth
    measuring. Failures in any single run propagate (one exception
    fails the whole ensemble); the caller decides whether to retry the
    ensemble or fall back to a single-shot run.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k == 1:
        # Skip the thread pool for the trivial case; mostly so the
        # single-shot path stays trivially testable.
        return [run_agent(type, task, temperature=temperature)]

    with ThreadPoolExecutor(max_workers=k) as pool:
        futures = [
            pool.submit(run_agent, type, task, temperature=temperature)
            for _ in range(k)
        ]
        return [f.result() for f in futures]
