from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_v2.benchmark.live_runner import (
    ProductionSupportedStagePipeline,
    ResumableSupportedBenchmarkRunner,
    communicative_canary_cases,
    select_supported_cases,
)
from rigby_v2.benchmark.live_configuration import (
    discover_live_model_configuration,
)
from rigby_v2.benchmark.manifest import load_benchmark_manifest
from rigby_v2.benchmark.models import BenchmarkFamily
from rigby_v2.planner import OpenAISemanticPlanner
from rigby_v2.selection import (
    BestOfFiveOrchestrator,
    OpenAIAnonymousJudge,
)


ROOT = Path(__file__).resolve().parents[2]


def _configured_pipeline(
    run_root: Path,
    *,
    environ: Mapping[str, str] | None = None,
    dotenv_paths: Sequence[Path] | None = None,
) -> ProductionSupportedStagePipeline:
    configuration = discover_live_model_configuration(
        environ=environ,
        dotenv_paths=(
            tuple(dotenv_paths)
            if dotenv_paths is not None
            else (ROOT / ".env", ROOT.parent / ".env")
        ),
    )
    pipeline_id = (
        "rigby-v2-production-supported.v1.config-"
        + configuration.configuration_fingerprint[:20]
    )
    diagnostic = configuration.diagnostic()
    if not configuration.configured:
        return ProductionSupportedStagePipeline(
            external_model_diagnostic=diagnostic,
            pipeline_id=pipeline_id,
        )

    from openai import OpenAI

    artifacts = ContentAddressedArtifactStore(run_root / "artifacts")
    client_options = {"api_key": configuration.api_key}
    if configuration.base_url:
        client_options["base_url"] = configuration.base_url
    client = OpenAI(**client_options)
    opaque_id = configuration.configuration_fingerprint[:20]
    planner = OpenAISemanticPlanner(
        client=client,
        model=configuration.planner_model,
        planner_id=f"configured-openai-planner:{opaque_id}",
    )
    judge = OpenAIAnonymousJudge(
        client=client,
        model=configuration.judge_model,
        artifacts=artifacts,
        judge_id=f"configured-openai-judge:{opaque_id}",
    )
    return ProductionSupportedStagePipeline(
        planner=planner,
        selector=BestOfFiveOrchestrator(judge),
        artifacts=artifacts,
        external_model_diagnostic=diagnostic,
        pipeline_id=pipeline_id,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resume selected cases from the sealed 300-case supported benchmark. "
            "Incomplete external or physical evidence is recorded as a typed blocker."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "artifacts-v2" / "supported-live-benchmark",
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--family",
        action="append",
        choices=[item.value for item in BenchmarkFamily],
        default=[],
    )
    parser.add_argument(
        "--communicative-canary",
        type=int,
        choices=range(3, 6),
        metavar="COUNT",
    )
    parser.add_argument("--stage-attempt-limit", type=int, default=2)
    parser.add_argument("--max-model-calls-per-case", type=int, default=5)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Refuse to use an existing run directory.",
    )
    arguments = parser.parse_args()

    manifest = load_benchmark_manifest()
    if arguments.communicative_canary is not None:
        if arguments.case_id or arguments.family:
            parser.error("the canary option cannot be combined with case/family filters")
        cases = communicative_canary_cases(
            manifest, arguments.communicative_canary
        )
    else:
        cases = select_supported_cases(
            manifest,
            case_ids=arguments.case_id,
            families=arguments.family,
        )
    pipeline = _configured_pipeline(arguments.run_root.resolve())
    runner = ResumableSupportedBenchmarkRunner(
        run_root=arguments.run_root,
        manifest=manifest,
        pipeline=pipeline,
        selected_cases=cases,
        stage_attempt_limit=arguments.stage_attempt_limit,
        max_model_calls_per_case=arguments.max_model_calls_per_case,
        resume=not arguments.no_resume,
    )
    summary = runner.run()
    output = {
        "all_complete": summary.all_complete,
        "selected_case_count": len(summary.selected_case_ids),
        "status_counts": dict(summary.status_counts),
        "cases": [
            {
                "case_id": item["case_id"],
                "status": item["status"],
                "completed_stages": sorted(item["completed_stages"]),
                "blocker": item["blocker"],
            }
            for item in summary.cases
        ],
        "run_root": str(summary.run_root),
        "production_readiness": pipeline.readiness_diagnostic(),
    }
    print(json.dumps(output, indent=2))
    return 0 if summary.all_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
