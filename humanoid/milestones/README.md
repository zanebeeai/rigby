# Milestones

Everything worth keeping from the gripper work, in one place. Each run in
`runs/` is a complete recording: every frame's joint poses, link geometry, block
position and orientation, contact forces, and penetration depth.

## Watch them

The dev server serves the viewer; any run in `runs/` can be loaded by name.

```bash
npm run dev --prefix frontend
```

Then open `http://localhost:5173/static/gripper.html?clip=<name>`, e.g.
`?clip=gripper-left-and-far`. Leave the parameter off for the latest run.

`RESULTS.txt` has the same table as below. `SENSING-BOUNDARY.txt` states, in one
place, what the machine is allowed to know.

## The milestone: it runs on what a real gripper could sense

The controller used to read the simulator in eighteen places — the object's true
position, its true half-extents, and a per-pad contact force. None of those exist
on a machine of this class. ALOHA (arXiv 2304.13705) does battery-slotting and
cup-opening at 80–90% with joint positions and RGB cameras and nothing else, so
that is the standard held to here.

What it has now:

| instrument | what it gives |
|---|---|
| joint encoders | exact, and the only thing that is |
| one wrist camera | 40° cone, 0.03–0.75 m, position error with depth, 8% size error, only while pointed at the object |
| contact | inferred from the encoders: driven shut, stopped, and not shut |
| grip effort | the motor command, not a load cell |
| bin position | known a priori — it is fixed furniture, not a percept |

No force sensor. No object pose. No object size. Nothing about the object while
it is out of view.

## Results

Twelve placements and sizes, the same controller and the same instruments for
all of them. `evals/gripper_scenarios.py` reproduces it.

| run | placed | peak lift | deepest penetration |
|---|---|---|---|
| centre, as tuned | yes | 28.86 cm | 0.036 mm |
| near, centred | yes | 28.58 cm | 0.057 mm |
| far, centred | yes | 29.73 cm | 0.215 mm |
| left | yes | 28.60 cm | 0.265 mm |
| right | yes | 28.62 cm | 0.290 mm |
| left and far | yes | 32.12 cm | 0.344 mm |
| right and near | yes | 28.94 cm | 0.315 mm |
| narrow block | yes | 29.28 cm | 0.056 mm |
| wide block | yes | 28.44 cm | 0.047 mm |
| tall thin block | yes | 25.51 cm | 0.034 mm |
| off the scan grid | yes | 28.69 cm | 0.316 mm |
| behind the reach | no | — | — |

**11 of 12.** The failure is a block placed 56 cm out, past the arm's reach,
where not placing it is the correct answer.

Penetration stays under 0.35 mm everywhere. That number is the one to watch: it
is how far the fingers are inside the block, and under the old position-welded
gripper it was 18.2 mm — a "grasp" that was really an overlap. These are real
grasps held by real contact forces.

## What losing the sensors broke, and what that revealed

Four things, and every one of them was a real bug the sensors had been hiding.

**It did not know where anything was.** Under ground truth the object's position
was readable from the first frame through the back of the robot's own head, so
the question never came up. With one camera on the wrist the machine starts
blind and refuses to reach, correctly. Looking is now a phase: it sweeps the
camera over the bench until something turns up, and acquires in under 1.5 s.

**Contact had the sign backwards.** "The finger did not reach its target"
detects contact exactly when there is none — under a commanded squeeze a blocked
finger is driven *past* its target and pinned there by the object.
Stopped-and-not-shut is the signature that survives being pushed.

**Obstruction was just lag.** An arm slewing at its rate limit sits a couple of
degrees behind its target for the whole move. Calling that an obstruction made
every fast reach refuse on alternating frames. Behind *and* stopped is
obstruction; behind and still moving is just moving. This one alone took the
sweep from 6/12 to 11/12.

**A refusal meant "re-aim at wherever I have sagged to".** Every refusal
returned the arm's current pose as its target. Under position welds that was a
true no-op, because the body was always exactly where it was put. Under torque
the arm droops a millimetre, the next frame adopts the droop as the goal, and
ten seconds of refusing walks the arm to the table one millimetre at a time —
which is what dragged the held block twenty centimetres across the bench while
every frame reported that nothing was happening.

## Caveat on the humanoid numbers

The humanoid hand's lifts (19.99 cm scripted, 31.64 cm model-planned) are **not
valid grasps**. That body is still position-welded, and under position welds the
holding force *is* the penetration — measured at −18.2 mm. See the correction
entry in `RIGBY_WORK_LOG.txt`. Only the torque-driven gripper runs in this
folder are physically valid.
