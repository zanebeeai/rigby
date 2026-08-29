# Video

`gripper-pick-and-place.webm` — the three-segment arm finding the block on the
bench, gripping it, carrying it to the elevated bin and releasing it. 9.5 s,
recorded at 1280x860 from the viewer.

`gripper-pick-and-place.gif` — the same thing, half size, plays anywhere.

WebM opens in Edge, Chrome or VLC (double-clicking may hand it to a player that
does not know VP8; "Open with" one of those if so).

This is the **ground-truth-era** run: the controller could read the object's
true position and size directly. Kept deliberately as the reference the later,
harder versions are measured against — the current controller sees the block
only through a wrist camera and has to find it. Numbers for this run: placed in
the bin, 28.9 cm peak lift, 0.036 mm deepest penetration.

Re-record any exported run with:

    python evals/gripper_capture.py <clip-name> --base http://localhost:5173/static

with `npm run dev --prefix frontend` running. Clip names are the filenames in
`../runs/` without the `.json`.
