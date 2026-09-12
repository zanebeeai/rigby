"""Audited simulator-to-sensor adapter with explicit per-joint address mapping.

The world adds free joints before the robot. Slicing qpos[:robot_nq] would leak
object poses into encoders; every reading here resolves a declared robot joint.
"""

from __future__ import annotations

import mujoco
from rigby_core.observations import (
    FullyObservedDiagnostic, ObservationProjector, SensorChannel, SensorDeclaration,
)

from .evaluator import inspection_copy
from .world import BenchmarkRefusal, CompiledBenchmarkWorld, ROBOT_PREFIX, WORLD_PREFIX


class SensorAdapter:
    def __init__(self, compiled: CompiledBenchmarkWorld):
        if compiled.body_manifest is None:
            raise BenchmarkRefusal("missing_body_manifest", "An acting sensor adapter requires a body")
        self.compiled = compiled
        self.channels, self.addresses = [], {}
        model = compiled.model
        policy = compiled.world.sensor_policy
        for index, joint_name in enumerate(compiled.body_manifest.robot.actuator_order):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ROBOT_PREFIX+joint_name)
            if joint < 0 or model.jnt_type[joint] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise BenchmarkRefusal("unsupported_encoder", "Declare a sensor mapping for each actuated joint")
            unit = "radian" if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE else "meter"
            for enabled, suffix, kind, field, address, measure in (
                (policy.joint_positions, "position", "joint_position", "qpos", int(model.jnt_qposadr[joint]), unit),
                (policy.joint_velocities, "velocity", "joint_velocity", "qvel", int(model.jnt_dofadr[joint]), unit+"_per_second"),
            ):
                if enabled:
                    name = f"encoder_{index}_{suffix}"
                    self.channels.append(SensorChannel(name=name, kind=kind, shape=(1,), unit=measure))
                    self.addresses[name] = (field, address)
        self.camera_name = WORLD_PREFIX+"camera"
        frame = "fixed_world_camera"
        if policy.camera_source == "declared_body_mount":
            mounts = compiled.body_manifest.camera_mounts
            if len(mounts) != 1:
                raise BenchmarkRefusal("ambiguous_sensor_mount", "This camera policy requires exactly one declared body mount")
            self.camera_name = ROBOT_PREFIX+mounts[0].channel
            frame = "declared_body_camera"
        if policy.rgb:
            self.channels.append(SensorChannel(name="rgb", kind="rgb", shape=(policy.height, policy.width, 3), unit="uint8", frame=frame))
        if policy.depth:
            self.channels.append(SensorChannel(name="depth", kind="depth", shape=(policy.height, policy.width), unit="meter", frame=frame))
        self.declaration = SensorDeclaration(channels=tuple(self.channels))
        self.projector = ObservationProjector(self.declaration)
        self.renderer = None
        if policy.rgb or policy.depth:
            self.renderer = mujoco.Renderer(model, height=policy.height, width=policy.width)

    def observe(self, data, *, sequence: int):
        readings = {name: (float(getattr(data, field)[address]),) for name, (field, address) in self.addresses.items()}
        if self.renderer is not None:
            sample = inspection_copy(self.compiled.model, data)
            self.renderer.update_scene(sample, camera=self.camera_name)
            if self.compiled.world.sensor_policy.rgb:
                self.renderer.disable_depth_rendering()
                readings["rgb"] = tuple(int(x) for x in self.renderer.render().ravel())
            if self.compiled.world.sensor_policy.depth:
                self.renderer.enable_depth_rendering()
                readings["depth"] = tuple(float(x) for x in self.renderer.render().ravel())
        packet = self.projector.project(sequence=sequence, time_s=float(data.time), readings=readings)
        if self.compiled.world.sensor_policy.mode != "declared_sensors":
            raise BenchmarkRefusal("diagnostic_requires_truth", "Use observe_diagnostic to label a fully observed baseline")
        return packet

    def observe_diagnostic(self, data, *, sequence: int, truth):
        # Build sensors through the same implementation, then label the privileged
        # union explicitly; a diagnostic never passes PolicyWorker.act().
        if self.compiled.world.sensor_policy.mode != "fully_observed_diagnostic":
            raise BenchmarkRefusal("privileged_observation_forbidden", "Sensor-only protocol cannot request evaluator truth")
        readings = {name: (float(getattr(data, field)[address]),) for name, (field, address) in self.addresses.items()}
        if self.renderer is not None:
            sample = inspection_copy(self.compiled.model, data)
            self.renderer.update_scene(sample, camera=self.camera_name)
            if self.compiled.world.sensor_policy.rgb:
                self.renderer.disable_depth_rendering()
                readings["rgb"] = tuple(int(x) for x in self.renderer.render().ravel())
            if self.compiled.world.sensor_policy.depth:
                self.renderer.enable_depth_rendering()
                readings["depth"] = tuple(float(x) for x in self.renderer.render().ravel())
        packet = self.projector.project(sequence=sequence, time_s=float(data.time), readings=readings)
        if truth.time_s != float(data.time):
            raise BenchmarkRefusal("diagnostic_clock_mismatch", "Diagnostic truth must match the sensor timestamp")
        return FullyObservedDiagnostic(observation=packet, evaluator_truth=truth)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
