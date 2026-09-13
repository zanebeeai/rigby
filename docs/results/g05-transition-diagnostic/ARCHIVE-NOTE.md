# Diagnostic provenance and reproduction scope

These files preserve the completed independent diagnostic from
`C:/Users/hocke/GitHub/rigby-g03/any-robot/results/g05-transition-diagnostic`
at G03 commit `7e1da38328e65d865b07130d33b5ee4d56e748d5`.

`polynomial_repro.py` and `offending-keyframes.json` are portable together and
need only NumPy. They independently reproduce the two interval extrema.

The archived full `diagnose.py` is the original capture script, retaining its
local G01/G03 paths and output-directory assumptions. It requires those exact
worktrees/evidence and is not the portable scalar regression command. The
grounded program snapshots, source hashes and measured summaries are preserved
alongside it. No new physical task success is claimed by this diagnostic.
