"""The golden corpus: committed reference cases every check runs against.

A case stores the *program*, not the clip, and the test recompiles it (plan 03
section 3.2).  That makes every corpus run a determinism regression test of the
compiler, and it makes a deliberate compiler change produce a reviewable diff via
``python -m evals.corpus bless``.

Loading and compiling the corpus needs no server, no browser, and no API key.
"""

from .hashing import (
    PORTABLE_PLATFORM_KEY,
    metrics_sha256,
    motion_sha256,
    observables_sha256,
    platform_key,
)
from .loader import (
    CORPUS_ROOT,
    CorpusCase,
    CorpusError,
    cases_root,
    compile_case,
    load_case,
    load_corpus,
    load_manifest,
    manifest_path,
    observe,
)
from .models import (
    CaseEntry,
    CorpusManifest,
    Coverage,
    CoverageAxis,
    DeterminismClass,
    ExpectedResult,
    Family,
    StoragePolicy,
)
from .verify import CaseComparison, Verdict, compare_case

__all__ = [
    "CORPUS_ROOT",
    "PORTABLE_PLATFORM_KEY",
    "CaseComparison",
    "CaseEntry",
    "CorpusCase",
    "CorpusError",
    "CorpusManifest",
    "Coverage",
    "CoverageAxis",
    "DeterminismClass",
    "ExpectedResult",
    "Family",
    "StoragePolicy",
    "Verdict",
    "cases_root",
    "compare_case",
    "compile_case",
    "load_case",
    "load_corpus",
    "load_manifest",
    "manifest_path",
    "metrics_sha256",
    "motion_sha256",
    "observables_sha256",
    "observe",
    "platform_key",
]
