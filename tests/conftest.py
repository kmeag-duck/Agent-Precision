"""Shared test fixtures.

The workflow instantiates `anthropic.Anthropic()` inside `run_agent` and
`run_orchestrator`. To exercise those code paths without hitting the network
we replace the `anthropic.Anthropic` symbol with a factory that returns a
`FakeAnthropic`. Each `FakeAnthropic` is preloaded with a list of scripted
responses; every call to `client.messages.create(...)` pops the next one.

The scripted responses are plain duck-typed objects, not `anthropic.types.*`
instances, so the tests do not pin a particular SDK version.

This conftest also installs a `pytest_collection_modifyitems` hook that
appends each test's docstring first line to its displayed node id, so
`python -m pytest -v` reads as a self-describing checklist.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    name: str
    input: dict
    id: str = "tool_use_id"
    type: str = "tool_use"


@dataclass
class FakeResponse:
    content: list
    stop_reason: str = "tool_use"


@dataclass
class FakeMessages:
    responses: list
    calls: list = field(default_factory=list)

    def create(self, **kwargs) -> FakeResponse:
        # Snapshot the messages list so later mutations by the orchestrator
        # don't retroactively change what each recorded call observed.
        snapshot = dict(kwargs)
        if "messages" in snapshot:
            snapshot["messages"] = [dict(m) for m in snapshot["messages"]]
        self.calls.append(snapshot)
        if not self.responses:
            raise AssertionError(
                "FakeAnthropic ran out of scripted responses; "
                f"call kwargs were: {kwargs}"
            )
        return self.responses.pop(0)


@dataclass
class FakeAnthropic:
    messages: FakeMessages

    @classmethod
    def with_responses(cls, responses: list) -> "FakeAnthropic":
        return cls(messages=FakeMessages(responses=list(responses)))


def pytest_collection_modifyitems(items):
    """Append each test's docstring first line to its displayed node id.

    With `-v`, pytest prints node ids like
    `tests/test_x.py::test_foo[param]`. We append ` -- <docstring>` so the
    output reads as a self-describing checklist of what each test covers.
    Tests without docstrings are left untouched.
    """
    for item in items:
        doc = (getattr(item, "obj", None).__doc__ or "").strip()
        if not doc:
            continue
        first_line = doc.splitlines()[0].strip()
        if not first_line:
            continue
        item._nodeid = f"{item._nodeid} -- {first_line}"


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Returns a function: install_fake(responses) -> FakeAnthropic.

    Calling `install_fake([resp1, resp2, ...])` monkeypatches
    `anthropic.Anthropic` so the next instantiation in production code
    returns a FakeAnthropic with those scripted responses. The returned
    FakeAnthropic exposes `.messages.calls` for assertions.
    """
    holder: dict[str, Any] = {}

    def install_fake(responses: list) -> FakeAnthropic:
        fake = FakeAnthropic.with_responses(responses)
        holder["fake"] = fake
        monkeypatch.setattr("anthropic.Anthropic", lambda: fake)
        return fake

    return install_fake
