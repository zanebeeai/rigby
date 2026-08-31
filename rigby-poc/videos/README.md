# Videos

Two recordings of the same task -- pick the block off the bench and place it in
the elevated bin -- separated by what the machine was allowed to know.

## 01-lift-and-place-known-positions.webm

The controller could read the block's true position and size directly out of the
simulator. This is the reference: it establishes that the ARM can do the task,
so that any later failure is attributable to perception rather than to the body.

Placed in the bin, 28.9 cm peak lift, 0.036 mm deepest penetration.

## 02-lift-and-place-camera-only.webm

The same task with nothing but joint encoders, one wrist camera read as pixels,
and contact inferred from the encoders. No object pose, no object size, no force
sensor. The machine starts not knowing where anything is, sweeps the camera
across the bench until it finds something, goes and looks at it properly from a
square vantage, and only then reaches.

Placed in the bin. Across twelve placements and sizes, 9 of 12.

## What to watch for

The second run has a phase the first does not need at all: **looking**. Under
ground truth the object's position was readable from the first frame through the
back of the robot's own head, so the question never came up. One glance from
wherever the sweep happens to be pointing lands 38 mm out, which puts the hand
six centimetres from a six-centimetre block -- so the machine goes to a vantage
directly above what it believes it saw and looks again until the answer stops
moving. That second look is worth 38 mm down to 7 mm.

Penetration is the number to watch in both. It is how far the fingers are inside
the block, and it is how you tell a grasp from an overlap.

## 03-cabinet-attempt-not-working.webm

The unfinished task, kept because it is the honest state of it. The arm finds
the cabinet handle and closes on it, and the door does not open. Room camera
with the wrist camera inset, and a live readout of phase, door angle, jaw
opening and whether the grip has latched.
