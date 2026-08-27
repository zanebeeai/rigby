"""Concrete local MuJoCo executor for the five-candidate flywheel."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict

import mujoco

from rigby_core.artifacts import ContentAddressedArtifactStore
from rigby_v2.certification import (
    CandidateCertificationRequest,
    CertificationEngine,
    CertificationOutcome,
    certify_robustness,
    policy_from_manifests,
    predicates_from_manifest,
)
from rigby_core.errors import FailureCode, RigbyV2Error
from rigby_v2.evidence import MujocoEvidenceRenderer, anonymous_id_for
from rigby_v2.evidence.renderer import trace_archive_bytes
from rigby_v2.flywheel.schemas import (
    CandidateProposalV1,
    CandidateVariationV1,
    SemanticPlanV1,
)
from rigby_core.hashing import content_hash, sha256_bytes
from rigby_core.motion import TimingProfile, compile_motion_program
from rigby_v2.pipeline import CandidateExecution, SurvivorValidation
from rigby_v2.rigging import (
    StagedCanonicalRig,
    validate_simulated_glb_round_trip,
)
from rigby_v2.scenes import StagedScene
from rigby_v2.selection import CandidateCompiler, DeterministicCandidateCompiler
from rigby_v2.simulation import (
    LinearKeyframeTrajectory,
    SimulationConfig,
    canonical_standing_config,
    map_generalized_state,
    project_qpos_to_rig,
)


class MujocoExecutionCancelled(RigbyV2Error):
    def __init__(self) -> None:
        super().__init__(FailureCode.CANCELLED, "five-candidate MuJoCo execution was cancelled")


def _motion_sample_hz(program) -> int:  # type: ignore[no-untyped-def]
    """Return an explicitly authored coarse task-space knot rate.

    Runtime control remains fixed at 240 Hz and interpolates these knots.  A
    bounded lower rate is essential for global nonlinear task-space refinement
    and is opt-in so existing direct-joint candidates remain unchanged.
    """

    if "task_space_sample_hz" not in program.metadata:
        return 240
    value = program.metadata["task_space_sample_hz"]
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 60:
        raise ValueError("task_space_sample_hz must be an integer in [1, 60]")
    return value


class MujocoProgramCandidateCompiler(CandidateCompiler):
    """Make the proposal budget cover real rig compilation, not just text edits."""

    def __init__(
        self,
        rig_model: mujoco.MjModel,
        rig: StagedCanonicalRig,
        inner: CandidateCompiler | None = None,
    ) -> None:
        self.rig_model = rig_model
        self.rig = rig
        self.inner = inner or DeterministicCandidateCompiler()

    def _validate(self, program, variation):  # type: ignore[no-untyped-def]
        compile_motion_program(
            program,
            self.rig_model,
            self.rig.manifest,
            sample_hz=_motion_sample_hz(program),
            timing_profile=TimingProfile(variation.timing_style),
        )
        return program

    def compile(
        self,
        plan: SemanticPlanV1,
        variation: CandidateVariationV1,
        attempt: int,
    ):
        return self._validate(self.inner.compile(plan, variation, attempt), variation)

    def repair(
        self,
        plan: SemanticPlanV1,
        variation: CandidateVariationV1,
        attempt: int,
        failure: Exception,
    ):
        return self._validate(
            self.inner.repair(plan, variation, attempt, failure), variation
        )


class MujocoCertifiedExecutor:
    """Compile, simulate three times, certify, export-check, and render evidence."""

    def __init__(
        self,
        *,
        rig: StagedCanonicalRig,
        scene: StagedScene,
        artifacts: ContentAddressedArtifactStore,
        certification_engine: CertificationEngine | None = None,
        evidence_renderer: MujocoEvidenceRenderer | None = None,
        certifier_id: str = "rigby-v2-deterministic-mujoco-certifier",
        process_workers: int = 1,
    ) -> None:
        if scene.manifest.rig_asset_hash != rig.reference.sha256:
            raise ValueError("scene and canonical rig artifact hashes do not match")
        self.rig = rig
        self.scene = scene
        self.artifacts = artifacts
        self.certification_engine = certification_engine or CertificationEngine()
        self.evidence_renderer = evidence_renderer or MujocoEvidenceRenderer(artifacts)
        self.certifier_id = certifier_id
        if not 1 <= process_workers <= 5:
            raise ValueError("process_workers must lie in [1, 5]")
        self.process_workers = process_workers
        rig_xml = artifacts.read_bytes(rig.manifest.mjcf).decode("utf-8")
        self.rig_xml = rig_xml
        self.rig_model = mujoco.MjModel.from_xml_string(rig_xml)
        self.scene_model = scene.compiled.load_model()
        self.source_glb = artifacts.read_bytes(rig.manifest.visual_asset)
        self.candidate_compiler = MujocoProgramCandidateCompiler(
            self.rig_model, rig
        )
        self._certification_requests = {}
        self._certification_results = {}

    @staticmethod
    def _check_cancelled(cancel_check: Callable[[], bool] | None) -> None:
        if cancel_check is not None and cancel_check():
            raise MujocoExecutionCancelled()

    def _prepare_certification_request(
        self, proposal: CandidateProposalV1
    ) -> CandidateCertificationRequest:
        program = proposal.program
        if program.rig_id != self.rig.manifest.rig_id:
            raise ValueError("proposal targets a different canonical rig")
        if program.scene_id != self.scene.manifest.scene_id:
            raise ValueError("proposal targets a different compiled scene")
        compiled = compile_motion_program(
            program,
            self.rig_model,
            self.rig.manifest,
            sample_hz=_motion_sample_hz(program),
            timing_profile=TimingProfile(proposal.variation.timing_style),
            allowed_contact_pairs=self.scene.manifest.allowed_contact_pairs,
        )
        qpos, qvel = map_generalized_state(
            self.rig_model, self.scene_model, compiled.qpos, compiled.qvel
        )
        _, qacc = map_generalized_state(
            self.rig_model, self.scene_model, compiled.qpos, compiled.qacc
        )
        return CandidateCertificationRequest(
            simulation=self._simulation_request(
                proposal.candidate_id,
                program.duration_s,
                compiled.times_s,
                qpos,
                qvel,
                qacc,
            ),
            policy=policy_from_manifests(self.rig.manifest, self.scene.manifest),
            predicates=predicates_from_manifest(self.scene.manifest),
            export_reimport=self._export_reimport,
            planned_contacts=program.contacts,
            expected_model_hash=self.scene.compiled.mjz_sha256,
            rig_asset_hash=self.rig.reference.sha256,
            scene_rig_asset_hash=self.scene.manifest.rig_asset_hash,
            rig_coordinate_system=self.rig.manifest.coordinate_system,
            scene_coordinate_system=self.scene.manifest.coordinate_system,
        )

    def execute_many(
        self,
        proposals: Sequence[CandidateProposalV1],
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[CandidateExecution, ...]:
        """Compile and certify exactly five candidates through one batch path."""

        proposals = tuple(proposals)
        if len(proposals) != 5 or len({item.candidate_id for item in proposals}) != 5:
            raise ValueError("MuJoCo bulk execution requires exactly five unique candidates")
        requests: list[CandidateCertificationRequest] = []
        for proposal in proposals:
            self._check_cancelled(cancel_check)
            requests.append(self._prepare_certification_request(proposal))
        self._check_cancelled(cancel_check)
        certifications = self.certification_engine.certify_many(
            tuple(requests), max_workers=self.process_workers
        )
        prior_requests = dict(self._certification_requests)
        prior_results = dict(self._certification_results)
        executions: list[CandidateExecution] = []
        try:
            for proposal, request, certification in zip(
                proposals, requests, certifications, strict=True
            ):
                self._check_cancelled(cancel_check)
                executions.append(
                    self._materialize_execution(proposal, request, certification)
                )
        except BaseException:
            # Candidate result references commit as a unit.  Content-addressed
            # render blobs are immutable, so an interruption can only leave
            # harmless unreferenced blobs rather than a partial accepted batch.
            self._certification_requests.clear()
            self._certification_requests.update(prior_requests)
            self._certification_results.clear()
            self._certification_results.update(prior_results)
            raise
        return tuple(executions)

    def _materialize_execution(
        self,
        proposal: CandidateProposalV1,
        request: CandidateCertificationRequest,
        certification,
    ) -> CandidateExecution:
        self._certification_requests[proposal.candidate_id] = request
        self._certification_results[proposal.candidate_id] = certification
        if not certification.simulation_runs:
            raise RuntimeError("certification returned no authoritative simulation runs")
        run_hashes = tuple(
            sha256_bytes(trace_archive_bytes(run.trace))
            for run in certification.simulation_runs
        )
        resimulation_sha256 = content_hash(run_hashes)
        certification_payload = {
            "candidate_id": proposal.candidate_id,
            "outcome": certification.outcome.value,
            "violations": [asdict(item) for item in certification.violations],
            "predicates": [asdict(item) for item in certification.predicate_evaluations],
            "run_trace_sha256": run_hashes,
            "model_sha256": self.scene.compiled.mjz_sha256,
        }
        certification_sha256 = content_hash(certification_payload)
        first_run = certification.simulation_runs[0]
        if first_run.failure is not None:
            raise RigbyV2Error(
                FailureCode.SIMULATION_FAILED,
                "candidate simulation failed before evidence rendering: "
                f"{first_run.failure.code.value}: {first_run.failure.message}",
                details={
                    "candidate_id": proposal.candidate_id,
                    "simulation_failure": first_run.failure.code.value,
                },
            )
        maximum_penetration = max(
            (
                max(0.0, -contact.distance_m)
                for frame in first_run.trace.contacts
                for contact in frame.contacts
            ),
            default=0.0,
        )
        failing_predicate = next(
            (item.name for item in certification.predicate_evaluations if not item.passed),
            None,
        )
        violation = certification.violations[0] if certification.violations else None
        evidence = self.evidence_renderer.render(
            self.scene_model,
            first_run.trace,
            candidate_id=proposal.candidate_id,
            anonymous_id=anonymous_id_for(
                proposal.semantic_plan_hash, proposal.candidate_id
            ),
            metrics={
                "maximum_penetration_m": maximum_penetration,
                "certified": 1.0 if certification.certified else 0.0,
                "repeat_count": float(len(certification.simulation_runs)),
            },
            task_site=next(iter(self.scene.manifest.semantic_sites.values()), None),
        )
        return CandidateExecution(
            candidate_id=proposal.candidate_id,
            evidence=evidence,
            certified=certification.certified,
            certification_id=f"cert-{certification_sha256[:24]}",
            certification_sha256=certification_sha256,
            resimulation_result_sha256=resimulation_sha256,
            certifier_id=self.certifier_id,
            repeat_count=len(certification.simulation_runs),
            replay_exact=not any(
                item.code.value == "repeat_disagreement"
                for item in certification.violations
            ),
            failure_code=(
                violation.code.value
                if violation is not None
                else (
                    None
                    if certification.outcome is CertificationOutcome.CERTIFIED
                    else certification.outcome.value
                )
            ),
            failed_predicate=failing_predicate,
        )

    def execute(self, proposal: CandidateProposalV1) -> CandidateExecution:
        program = proposal.program
        if program.rig_id != self.rig.manifest.rig_id:
            raise ValueError("proposal targets a different canonical rig")
        if program.scene_id != self.scene.manifest.scene_id:
            raise ValueError("proposal targets a different compiled scene")
        compiled = compile_motion_program(
            program,
            self.rig_model,
            self.rig.manifest,
            sample_hz=_motion_sample_hz(program),
            timing_profile=TimingProfile(proposal.variation.timing_style),
            allowed_contact_pairs=self.scene.manifest.allowed_contact_pairs,
        )
        qpos, qvel = map_generalized_state(
            self.rig_model, self.scene_model, compiled.qpos, compiled.qvel
        )
        _, qacc = map_generalized_state(
            self.rig_model, self.scene_model, compiled.qpos, compiled.qacc
        )
        request = CandidateCertificationRequest(
            simulation=self._simulation_request(
                proposal.candidate_id, program.duration_s, compiled.times_s, qpos, qvel, qacc
            ),
            policy=policy_from_manifests(self.rig.manifest, self.scene.manifest),
            predicates=predicates_from_manifest(self.scene.manifest),
            export_reimport=self._export_reimport,
            planned_contacts=program.contacts,
            expected_model_hash=self.scene.compiled.mjz_sha256,
            rig_asset_hash=self.rig.reference.sha256,
            scene_rig_asset_hash=self.scene.manifest.rig_asset_hash,
            rig_coordinate_system=self.rig.manifest.coordinate_system,
            scene_coordinate_system=self.scene.manifest.coordinate_system,
        )
        certification = self.certification_engine.certify(request)
        self._certification_requests[proposal.candidate_id] = request
        self._certification_results[proposal.candidate_id] = certification
        if not certification.simulation_runs:
            raise RuntimeError("certification returned no authoritative simulation runs")
        run_hashes = tuple(
            sha256_bytes(trace_archive_bytes(run.trace))
            for run in certification.simulation_runs
        )
        resimulation_sha256 = content_hash(run_hashes)
        certification_payload = {
            "candidate_id": proposal.candidate_id,
            "outcome": certification.outcome.value,
            "violations": [asdict(item) for item in certification.violations],
            "predicates": [asdict(item) for item in certification.predicate_evaluations],
            "run_trace_sha256": run_hashes,
            "model_sha256": self.scene.compiled.mjz_sha256,
        }
        certification_sha256 = content_hash(certification_payload)
        first_run = certification.simulation_runs[0]
        maximum_penetration = max(
            (
                max(0.0, -contact.distance_m)
                for frame in first_run.trace.contacts
                for contact in frame.contacts
            ),
            default=0.0,
        )
        failing_predicate = next(
            (item.name for item in certification.predicate_evaluations if not item.passed),
            None,
        )
        violation = certification.violations[0] if certification.violations else None
        evidence = self.evidence_renderer.render(
            self.scene_model,
            first_run.trace,
            candidate_id=proposal.candidate_id,
            anonymous_id=anonymous_id_for(
                proposal.semantic_plan_hash, proposal.candidate_id
            ),
            metrics={
                "maximum_penetration_m": maximum_penetration,
                "certified": 1.0 if certification.certified else 0.0,
                "repeat_count": float(len(certification.simulation_runs)),
            },
            task_site=next(iter(self.scene.manifest.semantic_sites.values()), None),
        )
        return CandidateExecution(
            candidate_id=proposal.candidate_id,
            evidence=evidence,
            certified=certification.certified,
            certification_id=f"cert-{certification_sha256[:24]}",
            certification_sha256=certification_sha256,
            resimulation_result_sha256=resimulation_sha256,
            certifier_id=self.certifier_id,
            repeat_count=len(certification.simulation_runs),
            replay_exact=not any(
                item.code.value == "repeat_disagreement"
                for item in certification.violations
            ),
            failure_code=(
                violation.code.value
                if violation is not None
                else (
                    None
                    if certification.outcome is CertificationOutcome.CERTIFIED
                    else certification.outcome.value
                )
            ),
            failed_predicate=failing_predicate,
        )

    def validate_survivor(
        self,
        proposal: CandidateProposalV1,
        execution: CandidateExecution,
    ) -> SurvivorValidation:
        if proposal.candidate_id != execution.candidate_id or not execution.certified:
            raise ValueError("only the certified selected proposal may enter robustness")
        try:
            request = self._certification_requests[proposal.candidate_id]
            baseline = self._certification_results[proposal.candidate_id]
        except KeyError as error:
            raise ValueError("candidate was not executed by this MuJoCo executor") from error
        robustness = certify_robustness(
            request,
            engine=self.certification_engine,
            baseline=baseline,
        )
        variations = []
        first_failure = None
        for variation, result in robustness.variations:
            trace_hashes = tuple(
                sha256_bytes(trace_archive_bytes(run.trace))
                for run in result.simulation_runs
            )
            variations.append(
                {
                    "variation_id": variation.variation_id,
                    "outcome": result.outcome.value,
                    "violations": [asdict(item) for item in result.violations],
                    "predicates": [asdict(item) for item in result.predicate_evaluations],
                    "trace_sha256": trace_hashes,
                }
            )
            if not result.certified and first_failure is None:
                first_failure = (
                    result.violations[0].code.value
                    if result.violations
                    else result.outcome.value
                )
        validation_sha256 = content_hash(
            {
                "candidate_id": proposal.candidate_id,
                "baseline_certification_sha256": execution.certification_sha256,
                "variations": variations,
            }
        )
        return SurvivorValidation(
            passed=robustness.robust,
            variation_count=len(robustness.variations),
            validation_sha256=validation_sha256,
            failure_code=first_failure,
        )

    def _simulation_request(
        self,
        candidate_id,
        duration_s,
        times_s,
        qpos,
        qvel,
        qacc,
    ):
        from rigby_v2.simulation import SimulationRequest

        return SimulationRequest(
            model_xml=None,
            model_mjz=self.scene.compiled.mjz_bytes,
            trajectory=LinearKeyframeTrajectory(
                times_s=times_s,
                qpos=qpos,
                qvel=qvel,
                qacc=qacc,
            ),
            config=SimulationConfig(
                duration_s=duration_s,
                free_root_joint_name="pelvis_free",
                standing=canonical_standing_config(self.scene_model),
            ),
            initial_qpos=qpos[0],
            initial_qvel=qvel[0],
            request_id=candidate_id,
        )

    def _export_reimport(self, result):  # type: ignore[no-untyped-def]
        return validate_simulated_glb_round_trip(
            source_glb=self.source_glb,
            model_xml=self.rig_xml,
            rig=self.rig.manifest,
            times_s=result.trace.times_s,
            qpos=project_qpos_to_rig(
                self.scene_model, self.rig_model, result.trace.qpos
            ),
        )
