"""Stop a tree at a safe boundary, and start it again from what can be seen.

A safe boundary is the instant after a leaf has finished and before the next
node begins: the executor checks its interrupt there, so a request raised as
a leaf completes stops the tree with every enclosing node recording
``interrupted`` and nothing half-done. A checkpoint written at that instant
carries what the executor believed, which node completed, the clock, and
whatever world state the body's runtime chooses to hand over; it carries no
promise that any of the belief is still true.

A restart therefore does not resume. It opens a fresh executor with an empty
belief, lets the runtime look at the world -- hold still, sample the
declared sensors, decide the conditionals -- and runs the tree from its root
with the facts that observation established. A loop whose ``until`` those
facts satisfy ends before it acts; one they do not satisfy acts. The outcome
is the tree's own verdict: it completes, or it stops with a typed reason.
Nothing in the checkpoint's belief is copied into the new one, which is the
whole point: a restart that trusted its checkpoint would be trusting a world
that had time to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from ..contracts import Contract
from .contract import NodeKind, TaskTreeV1
from .executor import Interrupt, LeafContext, LeafOutcome, LeafRuntime


def safe_boundaries(tree: TaskTreeV1) -> tuple[str, ...]:
    """The leaves of the tree in execution order; boundary *k* is the instant
    the *k*-th completed leaf returns."""

    return tuple(node.node_id for node in tree.root.walk() if node.kind in (NodeKind.PRIMITIVE, NodeKind.OBSERVE))


class CheckpointV1(Contract):
    schema_version: str = "1.0"
    checkpoint_id: str = Field(min_length=1)
    tree_sha256: str = Field(min_length=64, max_length=64)
    library_sha256: str = Field(min_length=64, max_length=64)
    boundary_index: int = Field(ge=1)
    """Which completed leaf this checkpoint follows, counting from one."""
    boundary_node_id: str = Field(min_length=1)
    boundary_skill_id: str = Field(min_length=1)
    completed_leaves: tuple[str, ...] = ()
    belief_at_checkpoint: dict[str, Any] = {}
    """Kept for the record; a restart never reads it."""
    clock_s: float
    world_state: dict[str, Any] = {}
    """What the runtime handed over: for a physics body, the full integration
    state and the clock, so the world continues rather than resets."""
    reason: str = ""


@dataclass
class CheckpointingRuntime:
    """Wraps a body's runtime: after the ``stop_after``-th completed leaf it
    writes a checkpoint and requests the interrupt, so the executor stops at
    the next node it visits."""

    inner: LeafRuntime
    interrupt: Interrupt
    stop_after: int
    world_state: Any = None
    """A callable returning the world state to hand over, or ``None``."""
    completed: list[str] = field(default_factory=list)
    checkpoint: CheckpointV1 | None = None
    tree_sha256: str = ""
    library_sha256: str = ""

    def _after_leaf(self, context: LeafContext) -> None:
        self.completed.append(context.node.node_id)
        if len(self.completed) == self.stop_after and self.checkpoint is None:
            world = self.world_state() if callable(self.world_state) else (self.world_state or {})
            self.checkpoint = CheckpointV1(
                checkpoint_id=f"after-{self.stop_after:02d}-{context.node.skill_id}", tree_sha256=self.tree_sha256, library_sha256=self.library_sha256,
                boundary_index=self.stop_after, boundary_node_id=context.node.node_id, boundary_skill_id=context.node.skill_id, completed_leaves=tuple(self.completed),
                belief_at_checkpoint=dict(context.belief.facts), clock_s=float(context.clock.now()), world_state=dict(world), reason=f"checkpoint after leaf {self.stop_after} ({context.node.skill_id})",
            )
            self.interrupt.request(self.checkpoint.reason)

    def run_primitive(self, context: LeafContext) -> LeafOutcome:
        outcome = self.inner.run_primitive(context)
        self._after_leaf(context)
        return outcome

    def observe(self, context: LeafContext) -> dict[str, Any] | None:
        facts = self.inner.observe(context)
        self._after_leaf(context)
        return facts

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


__all__ = ["CheckpointV1", "CheckpointingRuntime", "safe_boundaries"]
