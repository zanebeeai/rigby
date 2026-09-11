# The gripper

Five systems. Each folder is one, each file inside it is a subsystem, and they
are listed here in the order data flows through them.

```
  body/      what the machine IS          declared, not compiled in
     |
  physics/   the simulated body           built FROM the declaration
     |
  sensing/   what it may know             the only way in
     |
  decision/  what to do about it          the layer this project is about
     |
  runs/      one task, end to end         and a recording of it
```

The claim being tested is that only `decision/` should have to be interesting.
A different robot ships a different `body/`, gets a different `physics/`, and
the same phases and refusals drive it. That is why the same sequence -- find,
look, approach, open, engulf, close, lift, carry, release -- drove a
nineteen-joint humanoid hand before it drove this two-finger gripper.

## body/

The manifest and the code that reads it. Joints and their ranges, segment
lengths, how fast each joint may move, and -- the part URDF has no word for --
the PRIMITIVES: what the body can be asked to do, what the amplitude of each one
means, and the conditions under which it refuses. Plus the METRICS: what it can
be asked about, and which instrument would answer.

The manifest sits next to its loader because that pairing is the system.

## physics/

The MuJoCo model, assembled from the manifest, and the computed-torque
controller that drives it. Torque rather than position, and that is not a
detail: under position control the holding force IS the penetration, and the
"grasps" this project reported for weeks were 18 mm of finger inside the block.
Under torque the object pushes back and the same task penetrates 0.2 mm.

Both scenes live here -- the bench with the bin, and the cabinet -- along with
the two cameras.

## sensing/

One file, and it is the whole boundary. Joint encoders, one wrist camera read as
actual pixels, and contact inferred from the encoders. No force sensor, no
object pose, no object size, nothing about the object while it is out of view.

`audit()` states the boundary in one place, so what the machine is allowed to
know is a thing you can read rather than a thing you have to trust.

## decision/

`solver.py` answers "where should the joints be". `pick_and_place.py` and
`cabinet.py` answer "what should the body be doing, and is that step finished".
The split is real: the solver knows nothing about tasks, and the tasks never
mention joints.

## runs/

Drives a task frame by frame, applies the rate limits once between deciding and
doing, and records what the body actually reached -- which is not the same as
what it was told, and the difference is the point of torque control.

## superseded/

The position-welded gripper. Kept because the comparison is the evidence for
everything in `physics/`, not because anything calls it.
