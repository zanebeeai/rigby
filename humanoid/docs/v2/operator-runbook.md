# Rigby v2 local operator runbook

This runbook operates the local-first MuJoCo service on Windows. Commands are run from `rigby-humanoid`. Never put production credentials in `.env`, console output, evidence files, or support bundles.

## 1. Reference machine and prerequisites

- Windows 11 x64, Python 3.12 x64, MuJoCo 3.11.0, ffmpeg, Git, and `uv`.
- Docker Desktop with Linux containers for the pinned local pgvector/PostgreSQL image, or an externally managed PostgreSQL instance with pgvector and the schema in `db/init`.
- At least 16 GB RAM and enough free disk for immutable artifacts, database backups, videos, traces, and temporary wheel builds.
- Research-only datasets are not downloaded by setup. Dataset manifests are policy records, not bundled data.

Run the offline preflight before installation or startup:

```powershell
.\.venv\Scripts\python.exe scripts\v2\preflight.py
```

The command is read-only: it does not install packages, download assets, start Docker, create directories, or modify configuration. Resolve every required failure. Runtime dependencies and the Hatchling build backend are exact-pinned; a relaxed or mismatched pin is a release blocker.

## 2. Local installation

1. Copy `.env.example` to `.env` outside any committed change. Replace the local database password and update `RIGBY_V2_DATABASE_URL` consistently.
2. Create a Python 3.12 environment and synchronize from the committed lock using the approved offline package cache. Do not use an unlocked `pip install -U`.
3. Run the wheel smoke check. It builds in a temporary directory with network access disabled and leaves the worktree unchanged:

```powershell
.\.venv\Scripts\python.exe scripts\v2\wheel_smoke.py
```

4. Re-run preflight and retain its JSON output as release evidence only after redacting machine usernames and secrets.

## 3. PostgreSQL

For local Docker PostgreSQL:

```powershell
docker compose -f compose.v2.yaml up -d postgres
docker compose -f compose.v2.yaml ps
```

Wait for `healthy`. The compose image is digest-pinned and binds only to `127.0.0.1:54329`. Do not expose that port publicly. For external PostgreSQL, provision TLS, least-privilege credentials, pgvector, backups, and the SQL under `db/init`; do not run the local container at the same time.

## 4. Startup order and readiness

1. Start PostgreSQL and confirm health.
2. Confirm the artifact directory is on the intended disk and is not a symlink to a broad or shared location.
3. Start the API bound to loopback:

```powershell
.\.venv\Scripts\rigby-v2.exe
```

4. Query `http://127.0.0.1:8010/api/v2/health`. `status` must be `ok`, database health must be present, and dependency-lock evidence must not be null.
5. Start one worker:

```powershell
.\.venv\Scripts\rigby-v2-worker.exe
```

6. Submit a small staging/simulation smoke job, wait for completion, verify artifact hashes, and ensure the queue does not accumulate expired leases.

Do not start multiple workers with the same `RIGBY_V2_WORKER_ID`. Do not make the API externally reachable without a separate authenticated reverse proxy and security review.

## 5. Performance gate

Run:

```powershell
.\.venv\Scripts\python.exe scripts\v2\benchmark.py
```

The fast harness processes exactly five ten-second candidates and reports scene compilation, trajectory preparation, native MuJoCo simulation, evidence generation, artifact finalization, candidate totals, and interpolated p95. VLM/network latency is excluded by definition.

For release evidence, run the production renderer path:

```powershell
.\.venv\Scripts\python.exe scripts\v2\production_renderer_benchmark.py --artifact-root .\tmp\production-renderer-benchmark --output .\tmp\production-renderer-report.json
```

This second command runs 240 Hz MuJoCo and the exact orbit, egocentric, and task-closeup raw-PNG/H.264 renderer for all five ten-second candidates. Release requires its p95 to be no greater than 90 seconds. The sealed reference measurement is under `assets/v2/benchmark`.

## 6. Legacy archive migration

Inventory the POC archive without copying its media payloads:

```powershell
.\.venv\Scripts\python.exe scripts\v2\import_legacy_archive.py .\results --mode metadata-only
```

The importer validates the complete index and folder topology before writing,
is content-hashed and idempotent, and creates only `legacy_candidate` records
with no release or embeddings. Use `--limit` for a canary. Hydrate selected
records later with `--mode full` only when their source bytes are needed for
re-simulation; hydration still does not make a record retrievable or certified.

## 7. Backup and restore

Back up PostgreSQL and the content-addressed artifact directory as one logical checkpoint. Quiesce new submissions first, allow leased jobs to finish, and record the database transaction time and artifact-root hash inventory.

```powershell
docker compose -f compose.v2.yaml exec -T postgres pg_dump -U rigby -d rigby -Fc -f /tmp/rigby-v2.dump
$rigbyPostgresContainer = docker compose -f compose.v2.yaml ps -q postgres
docker cp "${rigbyPostgresContainer}:/tmp/rigby-v2.dump" .\rigby-v2.dump
```

Copy `artifacts-v2` with a tool that preserves names and verifies hashes. Never recursively delete or overwrite the active artifact root. Restore into a new empty database and a new empty artifact directory, update a temporary `.env`, then verify database health, artifact hashes, one archived replay, and one new job before promotion. A backup is not accepted until this restore drill passes.

## 8. Upgrade and rollback

Before upgrade: stop new submissions, drain workers, capture database/artifact backups, save the old wheel and lock hash, and run the release checklist. Apply additive database migrations once. Start one canary worker, replay fixed jobs, and compare outcome/hashes/tolerances.

Rollback is forward-restorative: stop API/workers, restore the prior database checkpoint into a new database, point to the matching immutable artifact snapshot, reinstall the prior wheel from the offline cache, then rerun readiness and replay checks. Never use destructive Git or database reset commands as rollback.

## 9. Incident response

- **Database unavailable:** stop submission traffic, leave workers stopped, preserve logs, check Docker/external service and disk. Do not redirect to an empty database.
- **Worker crash/expired lease:** capture structured logs and job ID, allow lease recovery, retry only retryable failures, and preserve the failed trace/evidence.
- **Artifact hash mismatch:** stop the affected worker and API artifact download endpoint, preserve the store read-only, identify all database references to the hash, restore from verified backup, and treat unexplained mutation as an integrity incident.
- **Physics or replay regression:** quarantine the release, retain model/MJZ, seeds, solver, lock hash, and traces, and compare against the last certified MuJoCo version.
- **VLM/judge outage:** deterministic simulation may continue, but no candidate can be promoted or certified through the flywheel. Reject uncertainty; do not silently choose a candidate.
- **Dataset/license concern:** disable the affected namespace and retrieval records immediately. Do not delete provenance. Escalate to the dataset owner/reviewer.

## 10. Limitations and governance

- Supported: rigid objects, explicitly articulated mechanisms, tools, arbitrary contact sequences within certified packs, and bimanual manipulation.
- Typed unsupported outcomes: fluids, cloth/deformables, multi-character interaction, undeclared articulation, raw concave dynamic meshes, and production robot execution.
- SignAvatars and How2Sign remain in `research.language.sign`; they cannot supply generic gestures. Signing/ASL capability claims require documented Deaf expert review.
- HUMOTO and OmniRetarget-style structures remain in `research.object_contact` and may only support object/contact or interaction-retargeting research.
- Research-only/non-commercial licenses must be filtered at retrieval and reviewed before every external release. Do not ship source media in wheels, backups intended for distribution, or evidence bundles.
- Planner and judge models are explicit configuration, never silently upgraded. The production adapters use structured responses and stored anonymous evidence; a missing API credential or unavailable model blocks model-backed planning/judging rather than falling back to an unreviewed winner.

Embedding weights are a separate, offline inference asset. Prefetch only the
exact admitted SigLIP2 commit in a controlled network-enabled preparation step,
then verify it before runtime:

```powershell
python -m rigby_v2.library.model_cache prefetch --model siglip2-base-patch16-224
python -m rigby_v2.library.model_cache verify --model siglip2-base-patch16-224
```

The prefetch command seals every model file, total size, immutable revision,
license metadata, and the full snapshot-tree SHA-256. Inference calls
`local_files_only=True` with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`; a missing or changed snapshot is a typed failure.
The release canary command accepts only a hash-bound raw-frame archive and
writes a non-overwriting vector plus checksum:

```powershell
python scripts\v2\run_siglip2_canary.py `
  <raw-task-closeup-archive> `
  --archive-sha256 <full-content-sha256> `
  --frame frames/000150.png `
  --device cuda:0 `
  --output assets\v2\library\siglip2_keyframe_canary.v2.json
```

The sealed reference run emitted the same normalized 768-dimensional vector
twice. Cosmos-Embed1 remains disabled: do not prefetch or enable repository
custom code until the pinned revision passes security and license review and
the target platform has a measured resource budget. SigLIP2 evidence alone is
not a complete five-index release and must not make a record retrievable.

The supported benchmark runner is resumable by case and family. It never turns
an incomplete external, retrieval, physics, judge, human, or RAG stage into a
pass:

```powershell
python scripts\v2\run_supported_live_benchmark.py --communicative-canary 3
```

Configure `OPENAI_API_KEY`, `OPENAI_PLANNER_MODEL`, and `OPENAI_JUDGE_MODEL`
either in the process environment or in the workspace/project `.env`; the
process environment wins. The older `RIGBY_V2_PLANNER_MODEL` and
`RIGBY_V2_JUDGE_MODEL` names remain accepted as aliases. The runner never emits
these values: its console output and blocker artifacts contain only presence,
a configuration fingerprint, generic source type, and precise missing-authority
names. It creates no API client and makes no model call unless all three required
values are present.

Read `production_readiness` before interpreting a canary. A credential alone is
not sufficient. Missing planner/judge model selection, staged canonical rig or
scene, exact-five executor, active certified retrieval release/query/filters, or
final evidence provider is persisted at the stopping stage as a typed blocker
with `model_call_attempted=false` where applicable. Preserve the run directory:
its case checkpoint, attempt artifact, manifest preflight, and summary are the
authoritative record of how far the case honestly progressed.

Prepare genuine calibration media only from an explicit, hash-bound source
specification and three independently established rater hashes:

```powershell
.\.venv\Scripts\python.exe scripts\v2\prepare_human_review.py .\human-study\source-spec.json .\human-study\review-package
```

The generated rater packets omit action families, defect labels, source record
IDs, and ground truth. Keep `private/ground_truth.json` and
`private/source_map.json` away from raters. A generated package is not human
evidence: `human_calibration` remains blocked until all 600 assigned responses,
source-response files, and real rater-attestation files verify successfully.

The release-grade collection index uses schema `2.0` and is only an index: each
of its 600 assigned slots names a safe relative response path and exact SHA-256,
plus the safe relative path and SHA-256 of that rater's attestation. The retained
response JSON is authoritative for the verdict and rubric. Each of the three
attestation JSON files must identify its rater, affirm independent human review,
and bind the exact response hash for all 200 assigned pairs. Keep all referenced
files beneath one read-only evidence root; absolute paths, `..`, backslashes,
symlinks, missing files, reused response files, hash drift, incomplete slots, or
an attestation that does not bind its exact 200 responses fail closed.

Verify a genuinely collected bundle noninteractively:

```powershell
.\.venv\Scripts\python.exe scripts\v2\verify_human_study_collection.py `
  .\human-study\completed\collection-manifest.json `
  .\human-study\completed\retained-evidence `
  .\human-study\review-package\public\blueprint.json `
  .\human-study\review-package\private\ground_truth.json
```

The older schema `1.0` digest-only importer remains available for compatibility
and contract testing, but its result explicitly reports
`release_grade_files_verified=false` and must not satisfy release evidence.

### Goal 4 motion-quality exit study

The v2-versus-POC motion exit gate is separate from model calibration. Prepare
at least 30 uniquely matched held-out cases, with an authoritative trace and
summary for both variants and release-grade certification evidence for v2. Each
variant must include playable time-resolved evidence as a strict Rigby frame
archive: a ZIP containing `playback.json` and at least 30 contiguous PNG frames
at exactly 30 FPS. `playback.json` binds every timestamp and frame hash plus the
camera projection, resolution, and field of view. A contact sheet is optional
and supplemental; an image-only package fails this gate.

The source specification is schema `1.0`. Each pair has `pair_id`, a unique
`held_out_case_id`, `split="held_out"`, and `v2`/`poc` records. Each record names
safe relative motion-archive, raw-trace, and trace-summary paths with exact
SHA-256 values; the v2 record additionally names its certification JSON. Use
three independently established rater ID hashes:

```powershell
.\.venv\Scripts\python.exe scripts\v2\motion_exit_study.py prepare `
  .\motion-exit\source-spec.json `
  .\motion-exit\review-package
```

Preparation decodes and re-encodes every frame without identifying image
metadata, rebuilds a deterministic motion archive, binds the sanitized camera
and timestamp manifest, randomizes packet order, and presents every pair twice
per rater with left/right order reversed. Keep `private/variant_map.json` and
all `evidence/` files away from raters. Public packets must not contain variant,
v2, POC, case, or source labels.

For logic tests only, add `--synthetic-fixture`. That mode is permanently
ineligible for release even if every synthetic choice favors v2. Never relabel
synthetic responses as a release collection.

After genuine independent review, retain one response JSON for each of the 180
presentation slots (30 pairs x 3 raters x 2 reversed presentations) and one
distinct attestation JSON per rater. Each response records left/right/tie for
exactly `phase_timing`, `peak_speed`, `jerk_distribution`, and `hand_shape`.
Each attestation affirms independent real-human review and binds the exact hash
of every response assigned to that rater. The collection manifest is only an
index of safe relative paths and exact hashes; missing files, symlinks, reused
response bytes, hash-only placeholders, reversal disagreement, media/trace
tampering, or certification drift fail closed.

Verify the retained collection noninteractively:

```powershell
.\.venv\Scripts\python.exe scripts\v2\motion_exit_study.py verify `
  .\motion-exit\review-package `
  .\motion-exit\completed\collection-manifest.json `
  .\motion-exit\completed\retained-evidence
```

The gate takes each rater's reversal-consistent choice, computes a three-rater
majority per pair and dimension, then reports a one-sided exact binomial test
and exact 95% confidence interval. Release is allowed only when v2's lower
confidence bound exceeds 0.5 and `p < 0.05` independently on all four
dimensions, while every playable-media, trace, certification, response, and
attestation hash still verifies.

## 11. Ship decision

Collect the locally demonstrable evidence first:

```powershell
.\.venv\Scripts\python.exe scripts\v2\run_adversarial_acceptance.py
.\.venv\Scripts\python.exe scripts\v2\operator_dry_run.py
.\.venv\Scripts\python.exe scripts\v2\collect_local_release_evidence.py
```

This seals the deterministic adversarial firewall and live rollback drill in
addition to fresh-machine preflight, dependency-lock, wheel-smoke, and animated
five-candidate performance reports. It intentionally reports `can_ship=false`
while any requirement is absent. Populate one hash-addressed evidence artifact
for every remaining requirement in `docs/v2/release-checklist.json`, then run
`scripts/v2/release_check.py` against that manifest. Missing, failed,
duplicated, unknown, tampered, absolute, or path-escaping evidence blocks
release. Never manually override `can_ship=false`; correct the evidence or
create a new audited checklist version.
