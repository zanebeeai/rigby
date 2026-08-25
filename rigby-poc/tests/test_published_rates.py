"""L3 gate item 3 / plan 10 §9.5, as a failing test rather than a review note.

No published rate anywhere in the repository may be stated without its n and its
baseline. A rate without an n does not say whether it is 9 of 10 or 900 of 1000;
a rate without a baseline does not say whether it beats chance. The calibration
this rebuild is replacing was invalidated by exactly the second omission.
"""

from __future__ import annotations

import json

import pytest

from rigby_poc.published_rates import (
    WAIVERS,
    REPO_ROOT,
    Finding,
    scan_prose,
    scan_report,
    scan_repository,
    unwaived,
    waived_counts,
)

#: no compile, no corpus, no pipeline, no subprocess -- see docs/testing.md
pytestmark = pytest.mark.fast


@pytest.fixture(scope="module")
def findings() -> list[Finding]:
    return scan_repository()


def test_no_unwaived_published_rate_lacks_its_n_and_baseline(findings: list[Finding]) -> None:
    """The gate itself.

    If this fails, a rate was published without saying how many observations it
    rests on or what it is being compared against. Fix the statement -- add the n
    and the baseline -- rather than adding a waiver. Waivers are for artifacts
    another PR is already committed to regenerating.
    """

    offenders = unwaived(findings)
    assert not offenders, "published rates missing n or baseline:\n" + "\n".join(
        f"  {finding}" for finding in offenders
    )


def test_waived_files_do_not_acquire_new_bare_rates(findings: list[Finding]) -> None:
    """A waiver records a known deficiency; it does not license more of them."""

    counts = waived_counts(findings)
    for path, waiver in WAIVERS.items():
        assert counts[path] == waiver["findings"], (
            f"{path} was waived at {waiver['findings']} bare rate(s) and now has "
            f"{counts[path]}. Do not raise the waiver -- state the baseline."
        )


def test_finding_paths_are_posix_so_waivers_match_on_every_platform() -> None:
    """Regression: the Windows CI job caught this guard failing on Windows only.

    ``WAIVERS`` is keyed by POSIX-shaped repo-relative paths. A native ``str()``
    of a ``Path`` yields backslashes on Windows, so every waiver stopped matching
    and the guard went red there while passing on macOS -- a guard defeated by
    the exact class of platform defect the CI matrix exists to find.

    Asserting the separator directly is what makes this catchable on macOS, where
    a native ``str()`` and ``as_posix()`` are indistinguishable.
    """

    findings = scan_repository()
    assert findings, "expected the two waived files to still produce findings"
    for finding in findings:
        assert "\\" not in finding.path, (
            f"{finding.path!r} contains a backslash; keys must be POSIX so they "
            f"match WAIVERS on every platform"
        )
        assert finding.path in WAIVERS or "/" in finding.path


def test_every_waived_path_is_posix_and_exists() -> None:
    for path in WAIVERS:
        assert "\\" not in path, f"{path!r} must be a POSIX-shaped key"
        assert (REPO_ROOT / path).exists(), f"waiver for {path} outlived the file"


def test_the_guard_documents_what_it_cannot_catch() -> None:
    """A guard that looks like it covers a class and covers part of it is itself
    an instance of the defect -- present, authoritative-looking, and not derived
    from what it claims to describe. So the limitation lives in the module, not
    in a plan someone has to find.
    """

    import rigby_poc.published_rates as module

    documentation = module.__doc__ or ""
    assert "does NOT catch" in documentation
    for hole in (
        "_base_metrics",
        "STALE_AFTER_MUTATION",
        "reference_frame",
        "the wrong quantity",
        "prevalence",
    ):
        assert hole in documentation, (
            f"the known hole {hole!r} must stay documented on the guard itself"
        )


def test_every_waiver_names_an_owner_and_the_pr_that_retires_it() -> None:
    for path, waiver in WAIVERS.items():
        assert (REPO_ROOT / path).exists(), f"waiver for {path} outlived the file"
        for field in ("reason", "owner", "retires_in"):
            assert waiver.get(field), f"waiver for {path} must state `{field}`"
        assert len(waiver["reason"]) > 60, f"waiver for {path} needs a real reason"


# --- the classifier itself, which is where this guard is easy to get wrong ---


@pytest.mark.parametrize(
    "line",
    [
        "- At least 95% correct supported planning and at least 95% correct rejection.",
        "the automated corruption suite stays below 5% false acceptance",
        "no more than 5% may fail",
        "the visual gate requires 80% agreement",
    ],
)
def test_targets_are_not_published_rates(tmp_path, line: str) -> None:
    """A requirement makes no claim about observed data, so it needs no n."""

    path = tmp_path / "doc.md"
    path.write_text(line, encoding="utf-8")
    assert scan_prose(path) == []


@pytest.mark.parametrize(
    "line",
    [
        "gated on its 95% Clopper-Pearson lower bound",
        "the 95% LCB does not clear the threshold",
        "severity at 50% detection is not estimable",
        "p95 recompilation latency",
    ],
)
def test_statistical_parameters_are_not_published_rates(tmp_path, line: str) -> None:
    """`95% confidence` names a method, not a result."""

    path = tmp_path / "doc.md"
    path.write_text(line, encoding="utf-8")
    assert scan_prose(path) == []


def test_an_n_on_the_next_line_still_counts(tmp_path) -> None:
    """The unit of judgement is the paragraph, because prose wraps.

    A line-scoped check reported three false positives in this repository's own
    documentation, where the rate and its n sat on adjacent lines of one
    sentence. That is a defect in the guard, not in the writing.
    """

    path = tmp_path / "doc.md"
    path.write_text(
        "Semantic discrimination measured 82% across\n"
        "240 clips against a 50% chance baseline.\n",
        encoding="utf-8",
    )
    assert scan_prose(path) == []


def test_a_blank_line_ends_the_window(tmp_path) -> None:
    """A qualifier two paragraphs away does not qualify anything."""

    path = tmp_path / "doc.md"
    path.write_text(
        "The grader accepted 82%.\n\nSeparately, we ran 240 clips against a "
        "50% chance baseline.\n",
        encoding="utf-8",
    )
    found = scan_prose(path)
    assert [finding.rate for finding in found] == ["82%"]


def test_a_rate_inside_a_fenced_block_is_a_transcript_not_a_claim(tmp_path) -> None:
    """`97% cpu` in pasted `time` output is not a published rate."""

    path = tmp_path / "doc.md"
    path.write_text(
        "Measured:\n\n```\n344 passed\n77.01s user 0.67s system 97% cpu\n```\n",
        encoding="utf-8",
    )
    assert scan_prose(path) == []


def test_a_rate_outside_the_fence_is_still_caught(tmp_path) -> None:
    """Closing the fence must not swallow the prose after it."""

    path = tmp_path / "doc.md"
    path.write_text(
        "```\n97% cpu\n```\n\nThe grader accepted 82%.\n", encoding="utf-8"
    )
    assert [finding.rate for finding in scan_prose(path)] == ["82%"]


def test_a_rate_in_an_inline_code_span_is_a_quotation(tmp_path) -> None:
    """`[100%]` is pytest's progress output, not a published rate.

    The second false positive this scanner produced against this repository's own
    documentation. A code span quotes; it does not claim.
    """

    path = tmp_path / "doc.md"
    path.write_text(
        "The entire output of a green run is a row of dots and a `[100%]`.\n",
        encoding="utf-8",
    )
    assert scan_prose(path) == []


def test_a_rate_outside_a_code_span_on_the_same_line_is_caught(tmp_path) -> None:
    """Stripping spans must not swallow the prose around them."""

    path = tmp_path / "doc.md"
    path.write_text("Run `pytest -q` — the grader accepted 82%.\n", encoding="utf-8")
    assert [finding.rate for finding in scan_prose(path)] == ["82%"]


def test_a_bare_measured_rate_is_caught(tmp_path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("The judge agreed with humans 90% of the time.", encoding="utf-8")
    found = scan_prose(path)
    assert len(found) == 1
    assert found[0].missing == ("its n", "its baseline")


def test_a_measured_rate_with_n_but_no_baseline_is_caught(tmp_path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("Across 240 clips the grader accepted 82%.", encoding="utf-8")
    found = scan_prose(path)
    assert len(found) == 1
    assert found[0].missing == ("its baseline",)


def test_a_fully_stated_rate_passes(tmp_path) -> None:
    path = tmp_path / "doc.md"
    path.write_text(
        "Semantic discrimination was 82% across 240 clips against a 50% chance baseline.",
        encoding="utf-8",
    )
    assert scan_prose(path) == []


def test_report_rate_without_a_baseline_sibling_is_caught(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"block": {"pair_count": 10, "agreement_rate": 0.9}}))
    found = scan_report(path)
    assert len(found) == 1
    assert found[0].missing == ("its baseline",)
    assert found[0].locator == "block.agreement_rate"


def test_report_rate_with_both_siblings_passes(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps({"block": {"n": 240, "agreement_rate": 0.82, "baseline_rate": 0.5}})
    )
    assert scan_report(path) == []


def test_report_gates_are_not_treated_as_published_rates(tmp_path) -> None:
    """`minimum_agreement` is the gate the rate is checked against."""

    path = tmp_path / "report.json"
    path.write_text(json.dumps({"block": {"minimum_agreement_rate": 0.8}}))
    assert scan_report(path) == []


def test_report_scan_reaches_into_lists(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"runs": [{"n": 4, "accuracy": 0.75}]}))
    found = scan_report(path)
    assert len(found) == 1
    assert found[0].locator == "runs[0].accuracy"
