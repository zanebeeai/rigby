# Evidence-qualified body capabilities

`ingest_capability_body(path)` derives a `BodyCapabilityManifestV1` and compiled
model from a fixed-base rigid URDF. Uploaded joint/link labels become aliases;
structural identities are assigned before the existing measurement, grounding
and reference compiler run. Equivalent renamings and declaration permutations
therefore produce the same complete capability hash and runtime model.

The ordinary `ingest_robot` entry point remains available for reproducing prior
experiments. Its legacy name hints are not covered by the new invariance claim.
New transfer experiments must use this capability entry point. This change does
not alter the frozen `RobotAssetManifestV1` schema or bypass its existing gates.

Every capability cites measured evidence, a source declaration, a simulation
assumption or an explicit evidence gap. Geometric closure does not enable
physical grasp transport. Unknown grip, suction, payload, hardware sensing,
rolling and balance capabilities remain unavailable through `body.require`.
Continuous-joint sampling bounds are separate from source position bounds.

The profile refuses unsupported source extensions, coupled mimic mechanisms and
exactly indistinguishable sibling identities. Those refusals describe runtime
coverage; they do not imply an invalid robot or physically impossible task.
Generated direct torque motors, inferred effort limits, default acceleration,
convex mesh colliders and collision exclusions remain explicit assumptions.

From a workspace environment with the `any-robot` and `core` packages installed:

```sh
python -m rigby_general.capabilities capture path/to/robot.urdf output/body
python -m rigby_general.capabilities verify output/body
python -m rigby_general.capabilities render output/body output/media
python any-robot/scripts/audit_body_capabilities.py output/g03
```

Destinations must be new. Captures include the capability manifest, full
morphology, canonical source, runtime XML, binary MuJoCo model and sampled
kinematics. The binary model permits offline inspection replay without original
meshes; use the recorded MuJoCo version. Runtime XML alone is not a portable
mesh package. The bundled third-party test archive separately supplies the
verified original descriptions, meshes, attribution and license notices.

Inspection GIFs show a camera orbit of a static pose with chains, derived sites
and sampled kinematic workspace. They use zero physics steps. Workspace samples
are not collision-free reachability certificates; grasp points are geometric
proposals checked for point occupancy, not finite-object fit or force closure.

The G03 audit uses six public zoo bodies, three unchanged third-party URDFs,
three independently frozen analytic geometry fixtures and 120 adversarial
joint/link renamings and declaration permutations. It does not open sealed
benchmark labels or count inspection frames as successful physical trials.
