"""Measurement probes behind plan 14.

Scripts, not analyzers, and deliberately so — for the same reason
:mod:`evals.contact_audit` is a script: a probe that emitted a metric key would
move ``metrics_sha256`` on all 47 cases and force a re-bless to answer a
question. Each probe is read-only against any tree, including ``main``, and each
prints the numbers a plan-14 claim rests on so a reviewer can re-derive them
instead of taking them from a document.

    python -m evals.probes.corpus_snapshot      state of the corpus on this machine
    python -m evals.probes.retarget_residual    per-bone visual -> physical feasibility
    python -m evals.probes.frame_alignment      is the retarget failure a frame convention?
    python -m evals.probes.standing             does the physical humanoid stand, and what does it cost
"""
