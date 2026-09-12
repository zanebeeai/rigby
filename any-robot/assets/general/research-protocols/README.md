# Public research protocols

`transfer-bench-v1-r1` is the current G02 engineering registration. The registration
SHA-256 is `1dab70141652d41aae1e511937696f7c9bbf9cb891aecd19ed7565734f32fc49`.
The protocol SHA-256 is `422adfffee21102d55de6a8a77a9dca61bc9b90b7d36190b4a8940d4864d20e9`.

The initial `transfer-bench-v1` remains intact as a superseded, unscored draft.
`transfer-bench-amendment.json` records why revision r1 explicitly defines release
as absence of robot touching/penetration **and** absence of positive normal force
from robot contacts. MuJoCo collision margins can transmit force at positive
geometric separation; distance alone cannot prove release.

The task requires the complete cube geometry to remain in the destination region
for two continuous seconds, released and below the registered speed limits.
There is a supporting destination platform. This root predicate establishes
placement only: it does not itself require lifting, grasping or carrying. A prompt
requiring those events needs additional predicates before it can be scored.

Strict mode preserves absolute geometry, masses, placements, lighting, goal,
physics, start rules and sensor policy across bodies. Capability-normalized mode
explicitly scales object/fixture lengths and positions, camera/light positions,
floor extent and target bounds by body reach divided by the fixed 0.5 m reference;
object masses scale cubically. It keeps gravity, time, friction, speed limits,
the declared base start and reference length fixed. It is a disclosed engineering
probe and is not evidence for success in the identical world. Robot-mounted
camera extrinsics and their origin belong to the body manifest.

The three public engineering splits contain seeds 0–19, 20–39 and 40–139.
Seed separation does not constitute unseen topology, provenance, prompt or scene
generalization. The existing sealed benchmark identities and ground-truth label
files must remain with an independent custodian. These public files neither read
nor replace them.

The protocol reserves at most three attempts, 120 simulation seconds and 300 wall
seconds per trial. One runner invocation consumes one attempt and enforces the
time ceilings. It never retries internally. Success at the exact simulation cap
is eligible; no physics step may extend beyond the cap, allowing only 1e-12 s for
floating-point accumulation. Setup and artifact serialization are not hard
preemptible; no success is accepted after the wall deadline. A future aggregate
trial ledger must share budgets across invocations before research scoring.

`feasibility-rules.v1.json` defines the referenced independent geometry/mechanics
rule and the prerequisites for scoring. It is committed with the registration
before the final engineering probes and included in their hashed evidence
inventory. Feasible requires an independent witness; infeasible requires a
checkable necessary-condition violation. Failure to find a controller is unknown,
and all attempted trials remain in the reported denominator. All G02 probes
remain unassessed and `scored: false`.

The worker receives only declared RGB and joint encoder packets. Sensor adapters
are trusted and audited separately. This boundary prevents accidental API and
inherited Python-state access to simulator truth. It is not an OS sandbox against
hostile code running under the same account. The public protocol itself reveals
nominal scene parameters. Fully observed diagnostic packets have a distinct type
and are rejected by the sensor-only worker.

The new runner is independent of the existing fitted-world demonstration path.
G01 physical replay integration, a frozen feasibility/trial roster, a shared
retry-budget ledger and independent confirmatory split custody are prerequisites
for scored research trials. No current acceptance threshold has been changed.
