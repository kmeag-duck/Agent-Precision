"""Deterministic codebase scan — step 1 of kernel discovery (no API calls).

This module is the cheap, read-only pre-filter of the hybrid
kernel-discovery pipeline. Given a directory tree (a real codebase such
as Kokkos Kernels) or a single file, it walks the tree, keeps only files
whose suffix a registered `LanguageProfile` claims (`.cpp`, `.cu`,
`.hip`), and pattern-matches for language-appropriate kernel markers
(`Kokkos::parallel_for`, `KOKKOS_LAMBDA`, `__global__`, ...). The output
is a bounded shortlist of `CandidateFile`s that the LLM `kernel_extractor`
agent then confirms and names (see `workflow/discover.py`).

Nothing here calls the network or writes to the codebase. The whole
module is pure/deterministic so it can be unit-tested without an API key.

Design notes:
  - Suffix filtering reuses the existing `PROFILES` registry rather than
    hardcoding extensions, so a new language profile automatically
    widens the scan.
  - Marker matching is intentionally coarse (substring/line scan). Its
    only job is to skip files with zero kernel-shaped content so the
    (paid) LLM pass sees a small shortlist. False positives are fine —
    the LLM filters them; false negatives are the real risk, so the
    marker set errs toward inclusion.
  - `max_files` caps how many shortlisted files flow downstream so a
    huge repo cannot blow up the LLM cost. Files are shortlisted in a
    stable sorted order so the cap is deterministic across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .languages import PROFILES

# Directory names we never descend into. Build trees and VCS metadata
# contain generated / vendored sources that are not the user's kernels
# and would balloon both the walk and the LLM cost.
DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "build",
        "_build",
        "cmake-build-debug",
        "cmake-build-release",
        "CMakeFiles",
        "node_modules",
        ".cache",
        "__pycache__",
        "install",
        "dist",
    }
)

# Per-language kernel markers. A file is shortlisted if it contains any
# marker whose owning language claims the file's suffix. Markers are
# grouped by language id so the reported `language_guess` reflects which
# group matched. Kept coarse on purpose (see module docstring).
KERNEL_MARKERS: dict[str, tuple[str, ...]] = {
    "kokkos": (
        "parallel_for",
        "parallel_reduce",
        "parallel_scan",
        "KOKKOS_LAMBDA",
        "KOKKOS_INLINE_FUNCTION",
        "KOKKOS_FUNCTION",
    ),
    "cuda": ("__global__", "__device__"),
    "hip": ("__global__", "__device__", "hipLaunchKernelGGL"),
    "sycl": ("parallel_for", "sycl::", "cgh."),
    "omp_offload": ("#pragma omp target", "#pragma omp teams"),
}


@dataclass(frozen=True)
class CandidateFile:
    """One source file the deterministic scan flagged as kernel-shaped.

    Fields:
      path            Path to the file (as walked, i.e. relative to the
                      scan root's parent or absolute, matching the input).
      language_guess  Language id (`kokkos`, `cuda`, ...) whose marker
                      group matched. This is a *guess* from the coarse
                      marker scan; the LLM `kernel_extractor` produces the
                      authoritative per-kernel language.
      matched_markers Sorted tuple of the distinct markers found in the
                      file. Useful for eyeballing why a file was flagged.
      match_lines     Sorted tuple of 1-based line numbers where a marker
                      first appeared (deduped, capped for compactness).
    """

    path: Path
    language_guess: str
    matched_markers: tuple[str, ...]
    match_lines: tuple[int, ...] = field(default_factory=tuple)


def _suffix_to_language_ids() -> dict[str, list[str]]:
    """Map each profile-claimed suffix to the language ids that claim it.

    Reuses the PROFILES registry so adding a language profile widens the
    scan with no change here. A suffix may map to several ids (`.cpp` is
    claimed by kokkos / sycl / omp_offload); marker matching then
    disambiguates which group's markers to look for.
    """
    out: dict[str, list[str]] = {}
    for profile in PROFILES.values():
        for suffix in profile.source_suffixes:
            out.setdefault(suffix.lower(), []).append(profile.id)
    return out


# Cap on how many match line numbers we record per file. Purely for
# compact output; the LLM re-reads the whole file anyway.
_MAX_MATCH_LINES = 20


def _scan_file_text(
    text: str, candidate_language_ids: list[str]
) -> tuple[str | None, tuple[str, ...], tuple[int, ...]]:
    """Scan one file's text for kernel markers.

    Returns (language_guess, matched_markers, match_lines). When no
    marker matches, language_guess is None and the tuples are empty (the
    caller drops the file). When several candidate language groups match
    (possible for shared-suffix .cpp), the first candidate id in
    `candidate_language_ids` order that produced a match wins the guess —
    which mirrors detect_language()'s insertion-order tie-break (Kokkos
    first for .cpp).
    """
    lines = text.splitlines()
    # Collect, per candidate language, which markers matched and where.
    per_lang_markers: dict[str, set[str]] = {}
    match_lines: set[int] = set()
    for lang_id in candidate_language_ids:
        markers = KERNEL_MARKERS.get(lang_id, ())
        for marker in markers:
            if marker not in text:
                continue
            per_lang_markers.setdefault(lang_id, set()).add(marker)
            for lineno, line in enumerate(lines, start=1):
                if marker in line:
                    match_lines.add(lineno)
                    break
    if not per_lang_markers:
        return None, (), ()
    # Language guess: first candidate (insertion order) that matched.
    language_guess = next(
        lang_id
        for lang_id in candidate_language_ids
        if lang_id in per_lang_markers
    )
    all_markers = sorted(
        {m for markers in per_lang_markers.values() for m in markers}
    )
    capped_lines = tuple(sorted(match_lines)[:_MAX_MATCH_LINES])
    return language_guess, tuple(all_markers), capped_lines


def scan_codebase(
    root: str | Path,
    max_files: int = 50,
    ignore_dirs: frozenset[str] = DEFAULT_IGNORE_DIRS,
) -> list[CandidateFile]:
    """Walk `root` and return a bounded shortlist of kernel-shaped files.

    `root` may be a directory (walked recursively, skipping
    `ignore_dirs`) or a single file (inspected directly). Only files
    whose suffix a registered LanguageProfile claims are considered; of
    those, only files containing at least one kernel marker for a
    claiming language are shortlisted.

    The shortlist is sorted by path (stable, deterministic) and then
    truncated to `max_files`. A non-positive `max_files` raises
    ValueError — an unbounded scan of a huge repo would defeat the cost
    guard. A `root` that does not exist raises FileNotFoundError.
    """
    if max_files <= 0:
        raise ValueError(f"max_files must be positive, got {max_files}")
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"scan root does not exist: {root_path}")

    suffix_map = _suffix_to_language_ids()

    if root_path.is_file():
        files = [root_path]
    else:
        files = _walk(root_path, ignore_dirs)

    candidates: list[CandidateFile] = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        candidate_language_ids = suffix_map.get(suffix)
        if not candidate_language_ids:
            continue
        try:
            text = file_path.read_text(errors="replace")
        except (OSError, UnicodeDecodeError):
            # Unreadable/binary file — silently skip. Discovery is
            # best-effort; a single bad file must not abort the scan.
            continue
        language_guess, markers, match_lines = _scan_file_text(
            text, candidate_language_ids
        )
        if language_guess is None:
            continue
        candidates.append(
            CandidateFile(
                path=file_path,
                language_guess=language_guess,
                matched_markers=markers,
                match_lines=match_lines,
            )
        )

    candidates.sort(key=lambda c: str(c.path))
    return candidates[:max_files]


def _walk(root: Path, ignore_dirs: frozenset[str]) -> list[Path]:
    """Recursively list files under `root`, pruning `ignore_dirs`.

    Implemented with an explicit stack over `Path.iterdir()` (rather
    than `os.walk`) so directory pruning by name is straightforward and
    the whole module stays in the pathlib idiom the rest of the codebase
    uses. Symlinked directories are not followed to avoid cycles.
    """
    files: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_symlink():
                # Follow symlinked files (read their target text) but
                # never descend into symlinked directories.
                if entry.is_file():
                    files.append(entry)
                continue
            if entry.is_dir():
                if entry.name in ignore_dirs:
                    continue
                stack.append(entry)
            elif entry.is_file():
                files.append(entry)
    return files


__all__ = [
    "CandidateFile",
    "DEFAULT_IGNORE_DIRS",
    "KERNEL_MARKERS",
    "scan_codebase",
]
