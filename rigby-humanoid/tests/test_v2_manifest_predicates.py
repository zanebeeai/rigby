import mujoco
import numpy as np

from rigby_v2.certification import ManifestStatePredicate, ObjectStateContext
from rigby_v2.certification.predicates import KinematicTrace
from rigby_v2.contracts import SceneStatePredicateV1
import pytest

pytestmark = pytest.mark.medium


def _context() -> ObjectStateContext:
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><joint name='hinge'/><geom type='sphere' size='.1' mass='1'/><site name='marker'/></body></worldbody></mujoco>"
    )
    trace = KinematicTrace(
        times_s=np.asarray([0.0, 1.0]),
        body_positions={},
        body_rotations={},
        joint_positions={"hinge": np.asarray([0.0, np.pi / 4.0])},
        site_positions={
            "marker": np.asarray([[0.0, 0.0, 0.5], [0.0, 0.0, 0.7]])
        },
        subtree_com={},
    )
    return ObjectStateContext(model=model, simulation=object(), kinematics=trace)  # type: ignore[arg-type]


def test_manifest_joint_predicate_converts_mujoco_radians_to_declared_degrees() -> None:
    predicate = ManifestStatePredicate(
        SceneStatePredicateV1(
            object_id="lever",
            name="activated",
            target="hinge",
            operator="ge",
            values=(35.0,),
            units="degrees",
        )
    )
    result = predicate.evaluate(_context())
    assert result.passed
    assert result.observed == 45.0


def test_manifest_site_lift_is_measured_relative_to_initial_world_pose() -> None:
    predicate = ManifestStatePredicate(
        SceneStatePredicateV1(
            object_id="block",
            name="lifted",
            target="marker",
            operator="ge",
            values=(0.15,),
            units="meters",
        )
    )
    result = predicate.evaluate(_context())
    assert result.passed
    assert np.isclose(result.observed, 0.2)
