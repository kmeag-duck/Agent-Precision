"""Tests for kernel discovery (step 1): deterministic scan, CLI helpers,
and the kernel_extractor registry entry.

The deterministic scan (workflow.discovery) and the CLI's pure helpers
(workflow.discover) make zero network calls, so most tests here are plain
unit tests over a tmp fixture tree. The one path that would hit the
network — extract_kernels' per-file kernel_extractor call — is exercised
with an injected stub run_agent_fn, so no test needs an API key.
"""

import json

import pytest

from workflow import discovery
from workflow.discovery import CandidateFile, scan_codebase
from workflow import discover
from workflow.discover import (
    DiscoveredKernel,
    apply_filters,
    build_extractor_task,
    build_manifest,
    extract_kernels,
    parse_selection,
    rank_kernels,
    render_table,
)
from workflow.registry import AGENTS


# ---------------------------------------------------------------------------
# Deterministic scan (workflow.discovery)
# ---------------------------------------------------------------------------


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_scan_flags_kokkos_file(tmp_path):
    """A .cpp file with a Kokkos marker is shortlisted with a kokkos guess."""
    _write(
        tmp_path,
        "src/foo.cpp",
        "#include <Kokkos_Core.hpp>\nvoid k(){ Kokkos::parallel_for(...); }\n",
    )
    out = scan_codebase(tmp_path)
    assert len(out) == 1
    assert out[0].language_guess == "kokkos"
    assert "parallel_for" in out[0].matched_markers


def test_scan_flags_cuda_file(tmp_path):
    """A .cu file with __global__ is shortlisted with a cuda guess."""
    _write(tmp_path, "k.cu", "__global__ void add(float* a){}\n")
    out = scan_codebase(tmp_path)
    assert len(out) == 1
    assert out[0].language_guess == "cuda"
    assert "__global__" in out[0].matched_markers


def test_scan_skips_unclaimed_suffix(tmp_path):
    """Files whose suffix no profile claims (.txt, .py) are ignored."""
    _write(tmp_path, "notes.txt", "parallel_for KOKKOS_LAMBDA __global__\n")
    _write(tmp_path, "script.py", "def parallel_for(): pass\n")
    assert scan_codebase(tmp_path) == []


def test_scan_skips_claimed_suffix_without_marker(tmp_path):
    """A .cpp with no kernel marker is not shortlisted."""
    _write(tmp_path, "plain.cpp", "int main(){ return 0; }\n")
    assert scan_codebase(tmp_path) == []


def test_scan_prunes_ignore_dirs(tmp_path):
    """Files under build/ and .git/ are never walked."""
    _write(tmp_path, "build/gen.cpp", "Kokkos::parallel_for(...);\n")
    _write(tmp_path, ".git/x.cpp", "Kokkos::parallel_for(...);\n")
    _write(tmp_path, "real.cpp", "Kokkos::parallel_for(...);\n")
    out = scan_codebase(tmp_path)
    assert [p.name for p in (c.path for c in out)] == ["real.cpp"]


def test_scan_respects_max_files(tmp_path):
    """max_files truncates the (sorted) shortlist deterministically."""
    for i in range(5):
        _write(tmp_path, f"k{i}.cpp", "Kokkos::parallel_for(...);\n")
    out = scan_codebase(tmp_path, max_files=2)
    assert len(out) == 2
    # Sorted by path -> k0, k1 come first.
    assert [c.path.name for c in out] == ["k0.cpp", "k1.cpp"]


def test_scan_rejects_nonpositive_max_files(tmp_path):
    """max_files <= 0 raises ValueError (the cost guard cannot be disabled)."""
    with pytest.raises(ValueError):
        scan_codebase(tmp_path, max_files=0)


def test_scan_missing_root_raises(tmp_path):
    """A nonexistent root raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        scan_codebase(tmp_path / "nope")


def test_scan_single_file_input(tmp_path):
    """Passing a single file (not a dir) scans just that file."""
    f = _write(tmp_path, "one.cpp", "KOKKOS_LAMBDA\n")
    out = scan_codebase(f)
    assert len(out) == 1 and out[0].path == f


def test_scan_records_match_line(tmp_path):
    """The 1-based line number of a marker is recorded in match_lines."""
    _write(tmp_path, "k.cpp", "// header\n// header2\nKOKKOS_LAMBDA\n")
    out = scan_codebase(tmp_path)
    assert out[0].match_lines == (3,)


def test_kokkos_wins_shared_cpp_suffix_tiebreak(tmp_path):
    """For a .cpp claimed by kokkos/sycl/omp, kokkos wins the guess (insertion order)."""
    # parallel_for is a marker for both kokkos and sycl; kokkos is first
    # in candidate order, so it wins the guess.
    _write(tmp_path, "amb.cpp", "Kokkos::parallel_for(...);\n")
    out = scan_codebase(tmp_path)
    assert out[0].language_guess == "kokkos"


# ---------------------------------------------------------------------------
# Selection parsing (workflow.discover.parse_selection)
# ---------------------------------------------------------------------------


def test_parse_selection_list():
    """'1,3,5' -> sorted 0-based indices."""
    assert parse_selection("1,3,5", 5) == [0, 2, 4]


def test_parse_selection_all():
    """'all' -> every index."""
    assert parse_selection("all", 3) == [0, 1, 2]


def test_parse_selection_unsorted_input_is_sorted():
    """Out-of-order input is returned sorted."""
    assert parse_selection("3,1", 3) == [0, 2]


def test_parse_selection_rejects_out_of_range():
    """An index above n raises ValueError."""
    with pytest.raises(ValueError):
        parse_selection("4", 3)


def test_parse_selection_rejects_zero():
    """1-based selection rejects 0."""
    with pytest.raises(ValueError):
        parse_selection("0", 3)


def test_parse_selection_rejects_duplicate():
    """A repeated index raises ValueError."""
    with pytest.raises(ValueError):
        parse_selection("1,1", 3)


def test_parse_selection_rejects_nonnumber():
    """A non-integer item raises ValueError."""
    with pytest.raises(ValueError):
        parse_selection("1,x", 3)


def test_parse_selection_rejects_empty():
    """An empty spec raises ValueError."""
    with pytest.raises(ValueError):
        parse_selection("   ", 3)


# ---------------------------------------------------------------------------
# Filters, ranking, table, manifest (pure helpers)
# ---------------------------------------------------------------------------


def _kernel(**kw):
    base = dict(
        file="f.cpp",
        function_name="k",
        language="kokkos",
        start_line=1,
        end_line=2,
        floating_point=True,
        self_contained=True,
        rationale="r",
    )
    base.update(kw)
    return DiscoveredKernel(**base)


def test_apply_filters_only_fp():
    """--only-fp drops non-floating-point kernels."""
    ks = [_kernel(floating_point=True), _kernel(floating_point=False)]
    assert apply_filters(ks, only_fp=True, only_self_contained=False) == ks[:1]


def test_apply_filters_only_self_contained():
    """--only-self-contained drops non-self-contained kernels."""
    ks = [_kernel(self_contained=False), _kernel(self_contained=True)]
    out = apply_filters(ks, only_fp=False, only_self_contained=True)
    assert out == ks[1:]


def test_rank_puts_self_contained_fp_first():
    """Ranking floats self-contained + floating-point kernels to the top."""
    worst = _kernel(function_name="worst", self_contained=False, floating_point=False)
    best = _kernel(function_name="best", self_contained=True, floating_point=True)
    out = rank_kernels([worst, best])
    assert out[0].function_name == "best"


def test_render_table_has_header_and_rows():
    """The table renders a header plus one numbered row per kernel."""
    txt = render_table([_kernel(function_name="myfunc")])
    assert "FUNCTION" in txt
    assert "myfunc" in txt
    assert "\n  1  " in ("\n" + txt)  # index 1 present


def test_render_table_empty():
    """An empty kernel list renders a placeholder, not a crash."""
    assert "no candidate" in render_table([]).lower()


def test_render_table_shows_template_column():
    """The table has a TMPL column: '-' for none, names for templated."""
    plain = _kernel(function_name="plain", template_params=[])
    tmpl = _kernel(
        function_name="tmpl",
        template_params=[{"name": "T", "kind": "type", "suggested": "double"}],
    )
    txt = render_table([plain, tmpl])
    assert "TMPL" in txt
    # The templated kernel's param name appears; there is a '-' cell too.
    assert "T" in txt
    assert " - " in txt


def test_build_manifest_shape():
    """The manifest carries schema_version, root, and the selected kernels."""
    m = build_manifest("/repo", [_kernel(function_name="fn")])
    assert m["schema_version"] == discover.MANIFEST_SCHEMA_VERSION
    assert m["root"] == "/repo"
    assert m["selected"][0]["function_name"] == "fn"
    # Round-trips through JSON.
    assert json.loads(json.dumps(m))["selected"][0]["file"] == "f.cpp"


def test_build_manifest_carries_template_params():
    """The manifest serializes template_params per selected kernel."""
    tp = [{"name": "T", "kind": "type", "suggested": "float"}]
    m = build_manifest("/repo", [_kernel(template_params=tp)])
    assert m["selected"][0]["template_params"] == tp
    # Round-trips through JSON unchanged.
    assert json.loads(json.dumps(m))["selected"][0]["template_params"] == tp


# ---------------------------------------------------------------------------
# Task building + extract_kernels (with injected stub, no network)
# ---------------------------------------------------------------------------


def test_build_extractor_task_numbers_lines_and_surfaces_scan_context():
    """The task prefixes line numbers and includes the scan guess + markers."""
    cand = CandidateFile(
        path="f.cpp",
        language_guess="kokkos",
        matched_markers=("parallel_for",),
        match_lines=(2,),
    )
    task = build_extractor_task(cand, "line one\nline two\n")
    assert "SCAN LANGUAGE GUESS: kokkos" in task
    assert "parallel_for" in task
    assert "     1: line one" in task
    assert "     2: line two" in task


def test_extract_kernels_aggregates_stub_results(tmp_path):
    """extract_kernels flattens agent output into DiscoveredKernel records."""
    f = _write(tmp_path, "k.cpp", "Kokkos::parallel_for(...);\n")
    cand = CandidateFile(
        path=f, language_guess="kokkos", matched_markers=("parallel_for",)
    )

    def stub(agent_type, task):
        assert agent_type == "kernel_extractor"
        return {
            "kernels": [
                {
                    "function_name": "nbody_step",
                    "language": "kokkos",
                    "start_line": 1,
                    "end_line": 10,
                    "floating_point": True,
                    "self_contained": True,
                    "template_params": [],
                    "rationale": "direct-sum force",
                }
            ]
        }

    out = extract_kernels([cand], run_agent_fn=stub)
    assert len(out) == 1
    assert out[0].function_name == "nbody_step"
    assert out[0].file == str(f)
    assert out[0].template_params == []


def test_extract_kernels_captures_template_params(tmp_path):
    """extract_kernels copies the agent's template_params into the record."""
    f = _write(tmp_path, "k.cpp", "template<class T> void axpby(){}\n")
    cand = CandidateFile(
        path=f, language_guess="kokkos", matched_markers=("parallel_for",)
    )
    tp = [
        {"name": "T", "kind": "type", "suggested": "double"},
        {"name": "ExecSpace", "kind": "exec_space", "suggested": "Kokkos::Serial"},
    ]

    def stub(agent_type, task):
        return {
            "kernels": [
                {
                    "function_name": "axpby",
                    "language": "kokkos",
                    "start_line": 1,
                    "end_line": 1,
                    "floating_point": True,
                    "self_contained": True,
                    "template_params": tp,
                    "rationale": "templated axpby",
                }
            ]
        }

    out = extract_kernels([cand], run_agent_fn=stub)
    assert out[0].template_params == tp


def test_extract_kernels_defaults_missing_template_params(tmp_path):
    """A stub omitting template_params yields [] (back-compat)."""
    f = _write(tmp_path, "k.cpp", "void k(){}\n")
    cand = CandidateFile(
        path=f, language_guess="kokkos", matched_markers=("parallel_for",)
    )

    def stub(agent_type, task):
        return {
            "kernels": [
                {
                    "function_name": "k",
                    "language": "kokkos",
                    "start_line": 1,
                    "end_line": 1,
                    "floating_point": True,
                    "self_contained": True,
                    "rationale": "no template_params key",
                }
            ]
        }

    out = extract_kernels([cand], run_agent_fn=stub)
    assert out[0].template_params == []


def test_extract_kernels_survives_agent_failure(tmp_path):
    """A failing agent call on one file is skipped, not fatal."""
    f = _write(tmp_path, "k.cpp", "Kokkos::parallel_for(...);\n")
    cand = CandidateFile(
        path=f, language_guess="kokkos", matched_markers=("parallel_for",)
    )

    def boom(agent_type, task):
        raise RuntimeError("api down")

    assert extract_kernels([cand], run_agent_fn=boom) == []


# ---------------------------------------------------------------------------
# kernel_extractor registry entry
# ---------------------------------------------------------------------------


def test_kernel_extractor_registered():
    """kernel_extractor is a registered agent with the standard entry keys."""
    spec = AGENTS["kernel_extractor"]
    for key in ("system_prompt", "output_schema", "model", "supports_temperature"):
        assert key in spec


def test_kernel_extractor_schema_requires_kernels_array():
    """The output schema requires a `kernels` array of the expected items."""
    schema = AGENTS["kernel_extractor"]["output_schema"]
    assert schema["type"] == "object"
    assert "kernels" in schema["required"]
    item = schema["properties"]["kernels"]["items"]
    for f in (
        "function_name",
        "language",
        "start_line",
        "end_line",
        "floating_point",
        "self_contained",
        "template_params",
        "rationale",
    ):
        assert f in item["required"], f"item missing required field {f}"


def test_kernel_extractor_template_params_item_shape():
    """template_params is an array of {name, kind, suggested} with a kind enum."""
    schema = AGENTS["kernel_extractor"]["output_schema"]
    tp = schema["properties"]["kernels"]["items"]["properties"]["template_params"]
    assert tp["type"] == "array"
    item = tp["items"]
    for f in ("name", "kind", "suggested"):
        assert f in item["required"]
    assert set(item["properties"]["kind"]["enum"]) == {
        "type",
        "exec_space",
        "non_type",
        "unknown",
    }


def test_kernel_extractor_prompt_mentions_template_params_informational():
    """The prompt documents template_params as informational, and self_contained is not disqualified by templating alone."""
    prompt = AGENTS["kernel_extractor"]["system_prompt"].lower()
    assert "template_params" in prompt
    assert "informational" in prompt
    # Templating alone must NOT force self_contained=false.
    assert "not by itself" in prompt or "not by itself disqualifying" in prompt


def test_kernel_extractor_prompt_is_identification_only():
    """The prompt forbids rewriting/inventing and asks for line ranges."""
    prompt = AGENTS["kernel_extractor"]["system_prompt"].lower()
    assert "identification only" in prompt
    assert "do not" in prompt and "invent" in prompt
    assert "self_contained" in prompt
