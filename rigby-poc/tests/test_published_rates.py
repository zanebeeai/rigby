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
