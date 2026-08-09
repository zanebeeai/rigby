from __future__ import annotations

from scipy.spatial.transform import Rotation

from evals.capture import (
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    _png_size,
    finger_shape_diagnostics,
    motion_diagnostics,
    phase_sampling_points,
)


def test_phase_sampling_uses_compiled_phase_ranges() -> None:
    payload = {
        "clip": {
            "metrics": {
                "phase_ranges_s": [
                    {"kind": "present", "start_s": 0.0, "end_s": 0.4},
                    {"kind": "hold", "start_s": 0.4, "end_s": 1.2},
                    {"kind": "recover", "start_s": 1.2, "end_s": 1.6},
                ]
            }
        }
    }
    points = phase_sampling_points(payload)
    assert [point["label"] for point in points] == [
        "present_start",
        "present_early",
        "present_motion",
        "present_late",
        "presented_pose",
        "hold_start",
        "hold_midpoint",
        "hold_end",
        "recover_start",
        "recover_early",
        "recovery_motion",
        "recover_late",
        "recover_end",
    ]
    assert [round(float(point["time_s"]), 3) for point in points] == [
        0.04, 0.12, 0.2, 0.28, 0.36,
        0.48, 0.8, 1.12,
        1.24, 1.32, 1.4, 1.48, 1.56,
    ]


def test_shake_sampling_tracks_requested_cycle_extrema() -> None:
    payload = {
        "clip": {
            "metrics": {
                "wrist_shake_cycles": 2.0,
                "phase_ranges_s": [
                    {"kind": "shake", "start_s": 1.0, "end_s": 2.0},
                ],
            }
        }
    }
    points = phase_sampling_points(payload)
    assert [point["label"] for point in points] == [
        "shake_start",
        "shake_extreme_1",
        "shake_extreme_2",
        "shake_extreme_3",
        "shake_extreme_4",
        "shake_end",
    ]
    assert [round(float(point["time_s"]), 3) for point in points] == [
        1.03,
        1.125,
        1.375,
        1.625,
        1.875,
        1.97,
    ]


def test_pickup_sampling_exposes_approach_contact_lift_and_hold() -> None:
    kinds = ("reach", "preshape", "contact", "close", "lift", "hold", "recover")
    payload = {
        "clip": {
            "metrics": {
                "phase_ranges_s": [
                    {"kind": kind, "start_s": index * 0.5, "end_s": (index + 1) * 0.5}
                    for index, kind in enumerate(kinds)
                ]
            }
        }
    }

    points = phase_sampling_points(payload)
    labels = {str(point["label"]) for point in points}
    assert {
        "reach_midpoint",
        "preshape_ready",
        "contact_established",
        "grasp_closed",
        "lift_midpoint",
        "hold_midpoint",
        "recover_end",
    } <= labels


def test_run_sampling_shows_split_stride_during_flight() -> None:
    payload = {
        "program": {
            "primitives": [
                {
                    "kind": "body",
                    "body": {"action": "run", "cycles": 3.0},
                }
            ]
        },
        "clip": {
            "metrics": {
                "phase_ranges_s": [
                    {"kind": "body", "start_s": 0.0, "end_s": 3.0},
                ]
            }
        },
    }

    points = phase_sampling_points(payload)
    assert [round(float(point["time_s"]), 2) for point in points] == [
        0.15,
        0.70,
        1.70,
        2.70,
        2.85,
    ]


def test_compound_full_body_sampling_is_compact_but_keeps_jump_apices() -> None:
    payload = {
        "program": {
            "intent": "full_body",
            "primitives": [
                {"kind": "body", "body": {"action": "turn", "cycles": 1.0}},
                {"kind": "body", "body": {"action": "crouch", "cycles": 1.0}},
                {"kind": "body", "body": {"action": "jump", "cycles": 2.0}},
                {"kind": "recover"},
            ],
        },
        "clip": {
            "metrics": {
                "phase_ranges_s": [
                    {"kind": "body", "start_s": 0.0, "end_s": 1.0},
                    {"kind": "body", "start_s": 1.0, "end_s": 2.0},
                    {"kind": "body", "start_s": 2.0, "end_s": 4.0},
                    {"kind": "recover", "start_s": 4.0, "end_s": 5.0},
                ]
            }
        },
    }

    points = phase_sampling_points(payload)
    assert [point["label"] for point in points] == [
        "body_start",
        "body_midpoint",
        "body_end",
        "body_start_2",
        "body_midpoint_2",
        "body_end_2",
        "body_start_3",
        "body_early_3",
        "body_midpoint_3",
        "body_late_3",
        "body_end_3",
        "recover_start",
        "recovery_motion",
        "recover_end",
    ]


def test_concurrent_body_arm_cycle_sampling_keeps_every_reversal() -> None:
    payload = {
        "program": {
            "intent": "full_body",
            "primitives": [
                {
                    "kind": "body",
                    "trajectory": "oscillate",
                    "effectors": [{"hand": "right"}],
                    "body": {"action": "walk", "cycles": 3.0},
                    "parameters": {"trajectory_cycles": 3.0},
                },
                {"kind": "recover"},
            ],
        },
        "clip": {
            "metrics": {
                "phase_ranges_s": [
                    {"kind": "body", "start_s": 0.0, "end_s": 3.0},
                    {"kind": "recover", "start_s": 3.0, "end_s": 4.0},
                ]
            }
        },
    }

    points = phase_sampling_points(payload)
    body_labels = [point["label"] for point in points if point["phase"] == "body"]
    assert body_labels == [
        "body_arm_cycle_start",
        "body_arm_cycle_extreme_1",
        "body_arm_cycle_extreme_2",
        "body_arm_cycle_extreme_3",
        "body_arm_cycle_extreme_4",
        "body_arm_cycle_extreme_5",
        "body_arm_cycle_extreme_6",
        "body_arm_cycle_end",
    ]


def test_png_contract_reads_exact_raw_canvas_dimensions() -> None:
    header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR"
    header += CAPTURE_WIDTH.to_bytes(4, "big") + CAPTURE_HEIGHT.to_bytes(4, "big")
    assert _png_size(header) == (CAPTURE_WIDTH, CAPTURE_HEIGHT)


def test_motion_diagnostics_expose_measurements_without_acceptance_label() -> None:
    value = motion_diagnostics(
        {
            "clip": {
                "metrics": {
                    "max_angular_velocity_rad_s": 26.0,
                    "forearm_rotation_cycles": 3.0,
                    "forearm_rotation_amplitude_rad": 0.21,
                    "parallel_forearm_max_axis_error_deg": 13.0,
                    "parallel_forearm_max_frontal_axis_error_deg": 3.6,
                    "parallel_forearm_minimum_separation_m": 0.032,
                    "parallel_forearm_minimum_hand_separation_m": 0.291,
                    "travel_wheel_cross_body_fraction": 1.0,
                    "travel_wheel_maximum_opposite_elbow_distance_m": 0.176,
                    "travel_wheel_vertical_order_range_m": 0.357,
                    "travel_wheel_depth_order_range_m": 0.218,
                    "structural_valid": False,
                }
            }
        }
    )
    velocity = value["measurements"]["max_angular_velocity_rad_s"]
    assert velocity == {"value": 26.0, "maximum_reference": 9.5}
    assert value["measurements"]["forearm_rotation_cycles"] == {"value": 3.0}
    assert value["measurements"]["forearm_rotation_amplitude_rad"] == {"value": 0.21}
    assert value["measurements"]["parallel_forearm_max_axis_error_deg"] == {
        "value": 13.0,
        "maximum_reference": 20.0,
    }
    assert value["measurements"]["parallel_forearm_max_frontal_axis_error_deg"] == {
        "value": 3.6,
        "maximum_reference": 8.0,
    }
    assert value["measurements"]["parallel_forearm_minimum_separation_m"] == {
        "value": 0.032,
        "minimum_reference": 0.025,
    }
    assert value["measurements"]["parallel_forearm_minimum_hand_separation_m"] == {
        "value": 0.291,
        "minimum_reference": 0.28,
    }
    assert value["measurements"]["travel_wheel_cross_body_fraction"] == {
        "value": 1.0,
        "minimum_reference": 0.95,
    }
    assert value["measurements"][
        "travel_wheel_maximum_opposite_elbow_distance_m"
    ] == {"value": 0.176, "maximum_reference": 0.20}
    assert value["measurements"]["travel_wheel_vertical_order_range_m"] == {
        "value": 0.357,
        "minimum_reference": 0.30,
    }
    assert value["measurements"]["travel_wheel_depth_order_range_m"] == {
        "value": 0.218,
        "minimum_reference": 0.16,
    }
    assert value["measurements"]["object_vertical_drift_m"]["maximum_reference"] == 0.015
    assert value["measurements"]["palm_relative_object_slip_m"]["maximum_reference"] == 0.020
    assert "structural_valid" not in str(value)
    assert "structural_failures" not in str(value)


def test_sequence_drop_capture_labels_gravity_fall_instead_of_throw_apex() -> None:
    points = phase_sampling_points(
        {
            "program": {
                "intent": "sequence",
                "steps": [
                    {"intent": "full_body", "primitives": []},
                    {
                        "intent": "object_interaction",
                        "object_action": "drop",
                        "primitives": [],
                    },
                ],
            },
            "clip": {
                "metrics": {
                    "phase_ranges_s": [
                        {"kind": "flight", "start_s": 2.0, "end_s": 3.0}
                    ]
                }
            },
        }
    )

    labels = [str(point["label"]) for point in points]
    assert labels == ["fall_start", "fall_early", "fall_midpoint", "fall_late", "landed"]
    assert "flight_apex" not in labels


def test_full_body_diagnostics_do_not_apply_arm_only_angular_bounds() -> None:
    value = motion_diagnostics(
        {
            "program": {"intent": "full_body"},
            "clip": {
                "metrics": {
                    "max_angular_velocity_rad_s": 11.0,
                    "max_angular_acceleration_rad_s2": 120.0,
                    "max_angular_jerk_rad_s3": 3000.0,
                    "max_frame_rotation_delta_rad": 0.2,
                    "discontinuities": 0,
                    "final_support_center_offset_m": 0.02,
                    "final_foot_ground_error_m": 0.0,
                    "final_root_vertical_speed_m_s": 0.0,
                }
            },
        }
    )["measurements"]

    assert value["max_angular_velocity_rad_s"] == {"value": 11.0}
    assert value["max_angular_acceleration_rad_s2"] == {"value": 120.0}
    assert value["max_angular_jerk_rad_s3"] == {"value": 3000.0}
    assert value["max_frame_rotation_delta_rad"]["maximum_reference"] == 0.35
    assert value["rotational_discontinuities"]["maximum_reference"] == 0
    assert value["final_support_center_offset_m"]["maximum_reference"] == 0.14


def test_motion_diagnostics_recompute_the_joint_that_carries_the_shake() -> None:
    from evals.corruptions import CorruptionSpec, corrupt_clip
    from rigby_poc.compiler import compile_motion
    from rigby_poc.models import CompileRequest, PlanRequest, default_scene
    from rigby_poc.planner import OfflinePlanner

    scene = default_scene()
    program = OfflinePlanner().plan(
        PlanRequest(
            text="Throw up a hang-ten, shake it three times, then return to default.",
            scene=scene,
            provider="offline",
        )
    ).program
    base = compile_motion(CompileRequest(scene=scene, program=program, persist=False))
    corrupted = corrupt_clip(
        base,
        program.hand,
        CorruptionSpec(
            id="diagnostic-wrong-joint",
            kind="wrong_joint_shake",
            axis=(0.0, 0.0, 1.0),
            angle_rad=1.0,
            mode="move_forearm_oscillation_to_hand_joint",
        ),
    )
    payload = {
        "program": program.model_dump(mode="json"),
        "clip": corrupted.model_dump(mode="json"),
    }
    measurements = motion_diagnostics(payload)["measurements"]
    assert measurements["forearm_rotation_cycles"]["value"] == 0.0
    assert measurements["forearm_rotation_amplitude_rad"]["value"] < 1e-4
    assert measurements["wrist_deviation_cycles"]["value"] == 3.0
    assert measurements["wrist_deviation_amplitude_rad"]["value"] > 0.10


def test_finger_shape_diagnostics_are_recomputed_from_hold_frame() -> None:
    rotations = {}
    for digit, segment, angle in (
        ("Thumb", "Metacarpal", 0.02),
        ("Index", "Proximal", 0.0),
        ("Middle", "Proximal", 0.0),
        ("Ring", "Proximal", 0.0),
        ("Little", "Proximal", 0.03),
    ):
        quaternion = Rotation.from_euler("x", angle).as_quat()
        rotations[f"right{digit}{segment}"] = {
            "rotation": dict(zip(("x", "y", "z", "w"), quaternion, strict=True))
        }
    value = finger_shape_diagnostics(
        {
            "program": {
                "hand": "right",
                "primitives": [{"kind": "hold", "hand_shape": "hang_ten"}],
            },
            "clip": {
                "frames": [{"time_s": 0.5, "bones": rotations}],
                "metrics": {"phase_ranges_s": [{"kind": "hold", "start_s": 0.4, "end_s": 0.8}]},
            },
        }
    )
    assert value["requested_hand_shape"] == "hang_ten"
    assert value["normalized_curls"]["index"] == 0.0
    assert value["human_calibrated_reference"]["index"]["minimum_normalized_curl"] == 0.65
