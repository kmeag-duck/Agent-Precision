"""Shared test fixtures.

The workflow instantiates `anthropic.Anthropic()` inside `run_agent` and
`run_orchestrator`. To exercise those code paths without hitting the network
we replace the `anthropic.Anthropic` symbol with a factory that returns a
`FakeAnthropic`. Each `FakeAnthropic` is preloaded with a list of scripted
responses; every call to `client.messages.create(...)` pops the next one.

The scripted responses are plain duck-typed objects, not `anthropic.types.*`
instances, so the tests do not pin a particular SDK version.

This conftest also installs two hooks that improve `python -m pytest -v`
output: `pytest_collection_modifyitems` appends each test's docstring
first line to its displayed node id, and `pytest_runtest_logstart` emits
a blank line plus an underlined header before the first test of each
file, so the verbose output reads as a self-describing checklist grouped
by file.
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


_last_logged_file: dict[str, str | None] = {"path": None}
_config: dict[str, object] = {"config": None}


def pytest_configure(config):
    """Stash the config so `pytest_runtest_logstart` can check verbosity and fetch the reporter lazily."""
    _config["config"] = config


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logstart(nodeid, location):
    """Print a header before the first test of each file under `-v`.

    `location` is `(filepath, lineno, domain)`. We track the last-seen
    filepath and, when it changes, write a blank line and an underlined
    header to the terminal so the verbose output is visually grouped by
    file. We skip this entirely when verbosity is <= 0 (the `-q` and
    default modes) so we do not break the one-line progress dot view.
    """
    config = _config["config"]
    if config is None or config.getoption("verbose") <= 0:
        return
    reporter = config.pluginmanager.getplugin("terminalreporter")
    if reporter is None:
        return
    filepath = location[0]
    if filepath == _last_logged_file["path"]:
        return
    _last_logged_file["path"] = filepath
    reporter.write_line("")
    reporter.write_line(filepath)
    reporter.write_line("-" * len(filepath))


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
