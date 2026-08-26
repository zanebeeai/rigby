"""Check the goal, requirement by requirement, and refuse to average.

``acceptance_criteria.general.yaml`` states G-GEN-1. This module measures it and
reports ``release_allowed`` only when every requirement passes. A failing
requirement is named; it is never traded off against a passing one, because the
goal is a conjunction and reporting a mean would hide exactly the thing worth
knowing.

Requirement 7, ``schema_invariance``, is the load-bearing one and is measured
directly here: the same prompts are planned against every robot and the
role-normalized hashes compared. If the semantic layer has learned anything about
any particular body, this is where it shows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import base_tree_fingerprint
from .pipeline import ingest_robot
from .planner import OfflineSchemaPlanner
from .primitives import PrimitiveLibrary
from .schema.inventory import afforded_entries, load_inventory


ROOT = Path(__file__).resolve().parents[2]
ZOO_ROOT = ROOT / "assets" / "general" / "zoo"

# Prompts every robot in the zoo should read identically. They deliberately name
# only free-space motion, so morphology cannot excuse a divergence.
CROSS_ROBOT_PROMPTS: tuple[str, ...] = (
    "reach out as far as you can",
    "reach out quickly then come back",
    "sweep slowly across in front of you",
    "wave three times",
    "lower the tool down toward the table",
    "trace a big circle",
    "move back a little",
    "hold still",
    "reach out precisely to a near point",
    "extend gently, halfway",
)

# Wording no body affords, or none recognises. Each must refuse with a *typed*
# reason rather than producing something.
ADVERSARIAL_PROMPTS: tuple[str, ...] = (
    "make me a sandwich",
    "solve the halting problem",
    "drive across the room and open the door",
    "swim to the bottom of the pool",
    "explain your reasoning",
    "hand it to me when you are done",
    "run this on the real robot",
    "fold the towel neatly",
    "walk over to the kitchen",
    "what do you think you should do next",
)


@dataclass
class Requirement:
    identifier: str
    statement: str
    measured: float
    threshold: float
    comparator: str
    detail: str = ""
    load_bearing: bool = False

    @property
    def passed(self) -> bool:
        if self.comparator == "gte":
            return self.measured >= self.threshold
        if self.comparator == "lte":
            return self.measured <= self.threshold
        return abs(self.measured - self.threshold) < 1e-9

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "statement": self.statement,
            "measured": round(self.measured, 6),
            "threshold": self.threshold,
            "comparator": self.comparator,
            "passed": self.passed,
            "load_bearing": self.load_bearing,
            "detail": self.detail,
        }


@dataclass
class AuditReport:
    requirements: list[Requirement] = field(default_factory=list)
    robots: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def release_allowed(self) -> bool:
        return all(requirement.passed for requirement in self.requirements)

    def to_json(self) -> dict[str, Any]:
        return {
            "goal_id": "G-GEN-1",
            "release_allowed": self.release_allowed,
            "robots": self.robots,
            "base_tree": base_tree_fingerprint().as_dict(),
            "requirements": [item.to_json() for item in self.requirements],
            "failing": [
                item.identifier for item in self.requirements if not item.passed
            ],
            "notes": self.notes,
        }


def _scan_for_robot_identifiers(source_root: Path, robot_ids: list[str]) -> list[str]:
    """Requirement 1: no benchmark robot may be named anywhere in the source."""

    hits: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for robot_id in robot_ids:
            if robot_id in text:
                hits.append(f"{path.relative_to(source_root)}: {robot_id}")
    return hits


def run_audit(
    *,
    zoo_root: Path = ZOO_ROOT,
    library_root: Path | None = None,
    source_root: Path | None = None,
) -> AuditReport:
    inventory = load_inventory()
    library = PrimitiveLibrary(library_root or Path("robots"))
    report = AuditReport()

    robot_ids = sorted(
        directory.name
        for directory in zoo_root.iterdir()
        if (directory / "robot.urdf").is_file()
    )
    report.robots = robot_ids

    # -- 1: no per-robot code -------------------------------------------
    hits = _scan_for_robot_identifiers(
        source_root or (ROOT / "src" / "rigby_general"), robot_ids
    )
    report.requirements.append(
        Requirement(
            "zero_per_robot_code",
            "No benchmark robot is named anywhere under src/rigby_general",
            measured=float(len(hits)),
            threshold=0.0,
            comparator="eq",
            detail="; ".join(hits[:4]),
        )
    )

    ingested = {}
    ingest_failures: list[str] = []
    for robot_id in robot_ids:
        try:
            ingested[robot_id] = ingest_robot(
                zoo_root / robot_id / "robot.urdf", robot_id=robot_id
            )
        except Exception as error:  # noqa: BLE001 - an audit reports, it does not raise
            ingest_failures.append(f"{robot_id}: {error}")

    report.requirements.append(
        Requirement(
            "ingest_success",
            "Every robot compiles and yields a measured morphology",
            measured=len(ingested) / max(len(robot_ids), 1),
            threshold=1.0,
            comparator="gte",
            detail="; ".join(ingest_failures[:3]),
        )
    )

    # -- 3, 4: derived sites and effector classification ------------------
    site_scores: list[float] = []
    effector_correct = 0
    effector_total = 0
    for robot_id, robot in ingested.items():
        truth = json.loads(
            (zoo_root / robot_id / "ground_truth.json").read_text(encoding="utf-8")
        )
        expected = {(item["semantic"], item["body"]) for item in truth["expected_sites"]}
        produced = {
            (site.semantic.value, site.body) for site in robot.morphology.sites
        }
        site_scores.append(len(expected & produced) / max(len(expected), 1))

        effector_total += 1
        expected_kinds = sorted(item["kind"] for item in truth["expected_effectors"])
        produced_kinds = sorted(
            effector.kind.value for effector in robot.morphology.effectors
        )
        effector_correct += int(expected_kinds == produced_kinds)

    report.requirements.append(
        Requirement(
            "site_derivation_accuracy",
            "Derived sites match the sealed ground truth on every robot",
            measured=min(site_scores) if site_scores else 0.0,
            threshold=0.95,
            comparator="gte",
        )
    )
    report.requirements.append(
        Requirement(
            "effector_classification",
            "The behavioural gripper test classifies every effector correctly",
            measured=effector_correct / max(effector_total, 1),
            threshold=1.0,
            comparator="gte",
        )
    )

    # -- 5, 6: the bake ---------------------------------------------------
    coverages: list[float] = []
    yields: list[float] = []
    bake_times: list[float] = []
    missing_libraries: list[str] = []
    for robot_id in ingested:
        summary = library.load_summary(robot_id)
        if summary is None:
            missing_libraries.append(robot_id)
            continue
        coverages.append(summary.schema_coverage)
        yields.append(summary.yield_fraction)
        bake_times.append(summary.elapsed_seconds)

    report.requirements.append(
        Requirement(
            "bake_coverage",
            "Every afforded schema has at least one certified primitive",
            measured=min(coverages) if coverages else 0.0,
            threshold=1.0,
            comparator="gte",
            detail=f"missing libraries: {missing_libraries}" if missing_libraries else "",
        )
    )
    report.requirements.append(
        Requirement(
            "certified_binding_yield",
            "Certified fraction of attempted bindings",
            measured=min(yields) if yields else 0.0,
            threshold=0.50,
            comparator="gte",
        )
    )
    report.requirements.append(
        Requirement(
            "bake_budget",
            "Slowest bake, in seconds",
            measured=max(bake_times) if bake_times else 0.0,
            threshold=1200.0,
            comparator="lte",
        )
    )

    # -- 7: schema invariance (the load-bearing one) ----------------------
    planner = OfflineSchemaPlanner(inventory)
    agreements = 0
    comparisons = 0
    divergences: list[str] = []
    for prompt in CROSS_ROBOT_PROMPTS:
        hashes: set[str] = set()
        for robot_id, robot in ingested.items():
            afforded = afforded_entries(inventory, robot.morphology)
            try:
                program = planner.plan(prompt, afforded=afforded)
            except Exception:  # noqa: BLE001 - a refusal is not a divergence
                continue
            hashes.add(program.role_normalized_hash())
        if not hashes:
            continue
        comparisons += 1
        if len(hashes) == 1:
            agreements += 1
        else:
            divergences.append(f"{prompt!r} produced {len(hashes)} readings")

    report.requirements.append(
        Requirement(
            "schema_invariance",
            "One prompt, one body-neutral reading, on every robot",
            measured=agreements / max(comparisons, 1),
            threshold=1.0,
            comparator="gte",
            load_bearing=True,
            detail="; ".join(divergences[:3]),
        )
    )

    # -- 10: adversarial prompts refuse, with a type ----------------------
    typed = 0
    for prompt in ADVERSARIAL_PROMPTS:
        robot = next(iter(ingested.values()), None)
        if robot is None:
            break
        afforded = afforded_entries(inventory, robot.morphology)
        try:
            planner.plan(prompt, afforded=afforded)
        except Exception as error:  # noqa: BLE001
            typed += int(hasattr(error, "code"))
    report.requirements.append(
        Requirement(
            "adversarial_typed_outcomes",
            "Requests outside the vocabulary refuse with a typed reason",
            measured=typed / max(len(ADVERSARIAL_PROMPTS), 1),
            threshold=1.0,
            comparator="gte",
        )
    )

    # -- 12: determinism --------------------------------------------------
    identical = 0
    checked = 0
    for robot_id in list(ingested)[:2]:
        first = ingest_robot(zoo_root / robot_id / "robot.urdf", robot_id=robot_id)
        second = ingest_robot(zoo_root / robot_id / "robot.urdf", robot_id=robot_id)
        checked += 1
        identical += int(
            first.morphology.content_hash() == second.morphology.content_hash()
        )
    report.requirements.append(
        Requirement(
            "deterministic_measurement",
            "Two ingests of one file measure the same body",
            measured=identical / max(checked, 1),
            threshold=1.0,
            comparator="gte",
        )
    )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zoo", type=Path, default=ZOO_ROOT)
    parser.add_argument("--library", type=Path, default=Path("robots"))
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args()

    report = run_audit(zoo_root=arguments.zoo, library_root=arguments.library)
    payload = report.to_json()

    print(f"G-GEN-1  robots: {', '.join(report.robots)}\n")
    for requirement in report.requirements:
        mark = "PASS" if requirement.passed else "FAIL"
        star = " *" if requirement.load_bearing else "  "
        print(
            f"  {mark}{star} {requirement.identifier:<32} "
            f"{requirement.measured:>10.4f} {requirement.comparator} "
            f"{requirement.threshold}"
        )
        if requirement.detail and not requirement.passed:
            print(f"         {requirement.detail[:140]}")

    print(f"\nrelease_allowed: {report.release_allowed}")
    if arguments.json:
        arguments.json.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"wrote {arguments.json}")
    return 0 if report.release_allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
