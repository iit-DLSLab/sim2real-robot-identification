from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402


SYSID_DIR = Path(__file__).resolve().parent
GENERATED_DIR = SYSID_DIR / "generated"


@dataclass
class ProcessedTrajectory:
    # Arrays use rows for time samples and columns for joints or actuators.
    # Keep measured states separate from references and actual simulation inputs.
    source_path: Path
    sequence_name: str
    times: np.ndarray
    measured_qpos: np.ndarray
    measured_qvel: np.ndarray
    desired_qpos: np.ndarray
    ctrl: np.ndarray
    joint_names: list[str]
    actuator_names: list[str]
    kp: np.ndarray
    kd: np.ndarray


def load_torch_dataset(dataset_path: Path) -> dict[str, np.ndarray]:
    import torch

    # Load on CPU even when the recording was saved from a CUDA device.
    try:
        raw_data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch releases do not accept the weights_only keyword.
        raw_data = torch.load(dataset_path, map_location="cpu")

    # Downstream preprocessing and MuJoCo operate on NumPy arrays.
    dataset: dict[str, np.ndarray] = {}
    for key, value in raw_data.items():
        if torch.is_tensor(value):
            dataset[key] = value.detach().cpu().numpy()
        else:
            dataset[key] = np.asarray(value)
    return dataset


def load_dataset_actuator_gains(
    dataset_path: Path,
    num_joints: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = load_torch_dataset(dataset_path)
    if num_joints is None:
        dof_pos = np.asarray(dataset["dof_pos"], dtype=np.float64)
        num_joints = int(dof_pos.shape[1])
    if 'kp' not in dataset:
        print(f"{dataset_path} does not contain `kp` switching to config default")
    if 'kd' not in dataset:
        print(f"{dataset_path} does not contain `kd` switching to config default")

    # Recorded gains take precedence; older datasets use the configured defaults.
    kp = dataset['kp'] if 'kp' in dataset else np.full(num_joints, config.kp, dtype=np.float64)  
    kd = dataset['kd'] if 'kd' in dataset else np.full(num_joints, config.kd, dtype=np.float64)
    return kp, kd


def build_actuator_gain_map(
    joint_names: list[str],
    kp: np.ndarray,
    kd: np.ndarray,
) -> dict[str, tuple[float, float]]:
    if len(joint_names) != len(kp) or len(joint_names) != len(kd):
        raise ValueError(
            "Joint names and actuator gains must have the same length."
        )
    return {
        joint_name: (float(joint_kp), float(joint_kd))
        for joint_name, joint_kp, joint_kd in zip(joint_names, kp, kd, strict=True)
    }

def get_robot_scene_path(robot: str) -> Path:
    scene_path = REPO_ROOT / "robot_model" / robot / "scene_flat.xml"
    if not scene_path.exists():
        raise FileNotFoundError(f"Could not find scene file: {scene_path}")
    return scene_path


def get_robot_model_xml_path(robot: str) -> Path:
    # Follow the scene's include so fitting uses the model selected by the scene.
    scene_path = get_robot_scene_path(robot)
    scene_tree = ET.parse(scene_path)
    scene_root = scene_tree.getroot()
    include_element = scene_root.find("include")
    if include_element is None:
        raise RuntimeError(f"No <include> tag found in scene file {scene_path}.")
    include_path = include_element.get("file")
    if include_path is None:
        raise RuntimeError(f"The <include> tag in {scene_path} has no `file` attribute.")
    return (scene_path.parent / include_path).resolve()


def _remove_all_by_tag(root: ET.Element, tag: str) -> None:
    # Copy each child list so removing an element cannot skip its next sibling.
    for parent in root.iter():
        for child in list(parent):
            if child.tag == tag:
                parent.remove(child)

def _absolutize_file_attributes(root: ET.Element, base_dir: Path) -> None:
    # Generated XML lives elsewhere; preserve asset paths relative to the source.
    for element in root.iter():
        file_attribute = element.get("file")
        if file_attribute is None:
            continue
        file_path = Path(file_attribute)
        if file_path.is_absolute():
            continue
        element.set("file", str((base_dir / file_path).resolve()))


def _rewrite_actuators_as_general(
    root: ET.Element,
    actuator_gains: dict[str, tuple[float, float]] | None = None,
) -> None:
    actuator_element = root.find("actuator")
    # Preserve actuator order, transmission and limits when replacing motor tags.
    source_actuators: list[dict[str, str]] = []
    if actuator_element is not None:
        for actuator in actuator_element:
            joint_name = actuator.get("joint")
            if joint_name is None:
                continue
            actuator_spec = {
                "joint": joint_name,
                "gear": actuator.get("gear", "1"),
            }
            force_range = actuator.get("forcerange") or actuator.get("ctrlrange")
            if force_range is not None:
                actuator_spec["forcerange"] = force_range
            actuator_name = actuator.get("name")
            if actuator_name is not None:
                actuator_spec["name"] = actuator_name
            source_actuators.append(actuator_spec)

    _remove_all_by_tag(root, "motor")

    joint_ranges: dict[str, str] = {}
    for element in root.iter("joint"):
        joint_name = element.get("name")
        joint_range = element.get("range")
        if joint_name is not None and joint_range is not None:
            joint_ranges[joint_name] = joint_range

    if actuator_element is None:
        actuator_element = ET.SubElement(root, "actuator")
    else:
        for child in list(actuator_element):
            actuator_element.remove(child)

    for actuator_spec in source_actuators:
        joint_name = actuator_spec["joint"]
        joint_range = joint_ranges.get(joint_name)
        if joint_range is not None:
            actuator_spec["ctrlrange"] = joint_range
        if actuator_gains is not None and joint_name in actuator_gains:
            # Affine bias implements force = kp * ctrl - kp * q - kd * qvel.
            # The control input is therefore a desired position, not a torque.
            kp, kd = actuator_gains[joint_name]
            actuator_spec["biastype"] = "affine"
            actuator_spec["gainprm"] = f"{kp:.12g}"
            actuator_spec["biasprm"] = f"0 {-kp:.12g} {-kd:.12g}"
        ET.SubElement(actuator_element, "general", actuator_spec)


def _disable_all_collisions(root):
    # Remove contact forces from the joint-dynamics identification model.
    for geom in root.iter("geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")


def build_fixed_base_model_xml(
    robot: str,
    actuator_mode: str = "general",
    actuator_gains: dict[str, tuple[float, float]] | None = None,
) -> Path:
    source_xml = get_robot_model_xml_path(robot)
    output_dir = GENERATED_DIR / robot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_xml = output_dir / f"{robot}_fixed_base_sysid.xml"

    tree = ET.parse(source_xml)
    root = tree.getroot()
    # Fix the floating base and discard keyframes tied to its old state layout.
    # Base-motion sensors are not part of the joint trajectory fitting objective.
    _remove_all_by_tag(root, "freejoint")
    _remove_all_by_tag(root, "keyframe")
    _remove_all_by_tag(root, "accelerometer")
    _remove_all_by_tag(root, "gyro")
    _remove_all_by_tag(root, "framepos")
    _remove_all_by_tag(root, "framequat")
    if actuator_mode == "general":
        _rewrite_actuators_as_general(root, actuator_gains=actuator_gains)
    elif actuator_mode != "motor":
        raise ValueError(
            f"Unsupported actuator mode `{actuator_mode}`. "
            "Expected `motor` or `general`."
        )
    _absolutize_file_attributes(root, source_xml.parent)

    _disable_all_collisions(root)

    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    return output_xml


def get_actuated_joint_and_actuator_names(mujoco, model) -> tuple[list[str], list[str]]:
    # Preserve actuator order: dataset controls and gain arrays use this layout.
    joint_names: list[str] = []
    actuator_names: list[str] = []
    for actuator_id in range(model.nu):
        actuator_names.append(model.actuator(actuator_id).name)
        joint_id = int(model.actuator_trnid[actuator_id][0])
        joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        joint_names.append(joint_name)
    return joint_names, actuator_names


def compute_pd_torques(
    desired_qpos: np.ndarray,
    desired_qvel: np.ndarray,
    measured_qpos: np.ndarray,
    measured_qvel: np.ndarray,
    kp: float,
    kd: float,
    ctrlrange: np.ndarray | None = None,
) -> np.ndarray:
    # Reconstruct commands when a motor-mode recording has no torque channel.
    ctrl = kp * (desired_qpos - measured_qpos) - kd * (desired_qvel - measured_qvel)
    if ctrlrange is None:
        return ctrl
    lower = ctrlrange[:, 0]
    upper = ctrlrange[:, 1]
    return np.clip(ctrl, lower, upper)


def load_processed_dataset(
    dataset_path: Path,
    model,
    mujoco,
    actuator_mode: str = "general",
) -> ProcessedTrajectory:
    dataset = load_torch_dataset(dataset_path)
    if "time" not in dataset:
        raise KeyError(f"{dataset_path} must contain `time`.")
    elif "dof_pos" not in dataset:
        raise KeyError(f"{dataset_path} must contain `dof_pos`.")
    elif "des_dof_pos" not in dataset:
        raise KeyError(f"{dataset_path} must contain `des_dof_pos`.")
    elif "dof_vel" not in dataset:
        raise KeyError(f"{dataset_path} must contain `dof_vel`.")
    elif "des_dof_vel" not in dataset:
        raise KeyError(f"{dataset_path} must contain `des_dof_vel`.")

    # Use float64 consistently with MuJoCo and the numerical optimizer.
    times = np.asarray(dataset["time"], dtype=np.float64)
    measured_qpos = np.asarray(dataset["dof_pos"], dtype=np.float64)
    measured_qvel = np.asarray(dataset["dof_vel"], dtype=np.float64)
    #measured_qvel = np.zeros_like(measured_qpos) # Use placeholder if you don't have velocity measurements in the dataset
    desired_qpos = np.asarray(dataset["des_dof_pos"], dtype=np.float64)
    desired_qvel = np.asarray(dataset["des_dof_vel"], dtype=np.float64)
    #desired_qvel = np.zeros_like(desired_qpos) # Use placeholder if you don't have velocity measurements in the dataset

    joint_names, actuator_names = get_actuated_joint_and_actuator_names(mujoco, model)
    kp, kd = load_dataset_actuator_gains(dataset_path, num_joints=len(joint_names))
    print("=======================================================")
    print("|| joint_names:", joint_names, "||")
    print("=======================================================")
    print("|| actuator_names:", actuator_names, "||")
    print("=======================================================")
    if measured_qpos.shape[1] != len(joint_names):
        raise ValueError(
            f"{dataset_path} has {measured_qpos.shape[1]} joints, "
            f"but the MuJoCo model expects {len(joint_names)}."
        )
    if desired_qpos.shape[1] != len(joint_names):
        raise ValueError(
            f"{dataset_path} desired state has {desired_qpos.shape[1]} joints, "
            f"but the MuJoCo model expects {len(joint_names)}."
        )

    if actuator_mode == "general":
        ctrl = desired_qpos
    elif actuator_mode == "motor":
        # Prefer recorded torques; otherwise reconstruct them from the PD signals.
        if "des_dof_torque" in dataset:
            ctrl = np.asarray(dataset["des_dof_torque"], dtype=np.float64)
        else:
            ctrl = compute_pd_torques(
                desired_qpos=desired_qpos,
                desired_qvel=desired_qvel,
                measured_qpos=measured_qpos,
                measured_qvel=measured_qvel,
                kp=kp,
                kd=kd,
                ctrlrange=model.actuator_ctrlrange if model.nu else None,
            )
    else:
        raise ValueError(
            f"Unsupported actuator mode `{actuator_mode}`. "
            "Expected `motor` or `general`."
        )
    return ProcessedTrajectory(
        source_path=dataset_path,
        sequence_name=dataset_path.stem,
        times=times,
        measured_qpos=measured_qpos,
        measured_qvel=measured_qvel,
        desired_qpos=desired_qpos,
        ctrl=ctrl,
        joint_names=joint_names,
        actuator_names=actuator_names,
        kp=kp,
        kd=kd,
    )


def chunk_processed_trajectory(
    trajectory: ProcessedTrajectory, chunk_size: int
) -> list[ProcessedTrajectory]:
    if chunk_size <= 0:
        return [trajectory]
    if chunk_size < 2:
        raise ValueError("chunk_size must be >= 2.")

    chunks: list[ProcessedTrajectory] = []
    num_steps = trajectory.times.shape[0]
    start = 0
    chunk_index = 0
    print(trajectory.times)
    # Each complete chunk is simulated independently, starting at local time zero.
    # A trailing incomplete chunk is omitted; a shorter whole recording is kept.
    while start + chunk_size <= num_steps:
        end = start + chunk_size
        chunks.append(
            ProcessedTrajectory(
                source_path=trajectory.source_path,
                sequence_name=f"{trajectory.sequence_name}_chunk_{chunk_index:03d}",
                times=trajectory.times[start:end] - trajectory.times[start],
                measured_qpos=trajectory.measured_qpos[start:end],
                measured_qvel=trajectory.measured_qvel[start:end],
                desired_qpos=trajectory.desired_qpos[start:end],
                ctrl=trajectory.ctrl[start:end],
                joint_names=list(trajectory.joint_names),
                actuator_names=list(trajectory.actuator_names),
                kp=trajectory.kp,
                kd=trajectory.kd,
            )
        )
        start = end
        chunk_index += 1
    return chunks or [trajectory]


def processed_to_sysid_trajectory(sysid, model, trajectory: ProcessedTrajectory):
    import mujoco

    # Match the model's sensor layout explicitly, including interleaved sensors
    # and the Piper mimic joint, rather than assuming position-only columns.
    joint_indices = {name: i for i, name in enumerate(trajectory.joint_names)}
    measurement_columns = []
    for sensor_id in range(model.nsensor):
        sensor_type = model.sensor_type[sensor_id]
        if sensor_type == mujoco.mjtSensor.mjSENS_JOINTPOS:
            values = trajectory.measured_qpos
        elif sensor_type == mujoco.mjtSensor.mjSENS_JOINTVEL:
            values = trajectory.measured_qvel
        else:
            raise ValueError(
                f"Unsupported fitting sensor: {model.sensor(sensor_id).name}"
            )
        # Resolve the joint by name: sensor order need not match dataset columns.
        joint_name = model.joint(int(model.sensor_objid[sensor_id])).name
        measurement_columns.append(values[:, joint_indices[joint_name]])

    measurement_ts = sysid.TimeSeries.from_names(
        trajectory.times,
        np.column_stack(measurement_columns),
        model
    )
    control_ts = sysid.TimeSeries(
        trajectory.times,
        trajectory.ctrl
    )
    # Initialize both position and velocity from the first measured sample;
    # create_initial_state also allocates history entries for the compiled model.
    initial_state = sysid.create_initial_state(
        model,
        trajectory.measured_qpos[0],
        trajectory.measured_qvel[0],
    )
    return measurement_ts, control_ts, initial_state


def _as_scalar(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def get_identifiable_body_names(
    model,
    requested_body_names: list[str] | None = None,
) -> list[str]:
    """Return massive bodies whose inertia can affect an articulated joint.

    When names are supplied explicitly, only existence, uniqueness, and a
    positive nominal mass are checked.  Otherwise the fixed base and other
    bodies upstream of every joint are omitted because their inertial
    parameters cannot be observed in a fixed-base experiment.
    """
    if requested_body_names is not None:
        duplicate_names = sorted(
            {
                name
                for name in requested_body_names
                if requested_body_names.count(name) > 1
            }
        )
        if duplicate_names:
            raise ValueError(
                "Duplicate inertial body names: " + ", ".join(duplicate_names)
            )

        body_names: list[str] = []
        for body_name in requested_body_names:
            try:
                body = model.body(body_name)
            except KeyError as exc:
                raise ValueError(
                    f"Body `{body_name}` does not exist in the MuJoCo model."
                ) from exc
            if int(body.id) == 0:
                raise ValueError("The MuJoCo world body has no inertial parameters.")
            if _as_scalar(body.mass) <= 0.0:
                raise ValueError(
                    f"Body `{body_name}` must have a positive nominal mass."
                )
            body_names.append(body_name)
        return body_names

    body_names = []
    for body_id in range(1, model.nbody):
        body = model.body(body_id)
        if not body.name or _as_scalar(body.mass) <= 0.0:
            continue

        # Include fixed child links too when an ancestor has an articulated joint.
        # Exclude masses attached only to the fixed base because it cannot move.
        ancestor_id = body_id
        while ancestor_id > 0:
            if int(model.body_jntnum[ancestor_id]) > 0:
                body_names.append(body.name)
                break
            ancestor_id = int(model.body_parentid[ancestor_id])
    return body_names


def make_armature_modifier(joint_name):
    """Create a modifier that sets armature on a specific joint."""
    def modifier(spec, param):
        spec.joint(joint_name).armature = param.value[0]
    return modifier

def make_frictionloss_modifier(joint_name):
    """Create a modifier that sets frictionloss on a specific joint."""
    def modifier(spec, param):
        spec.joint(joint_name).frictionloss = param.value[0]
    return modifier

def make_damping_modifier(joint_name):
    """Create a modifier that sets damping on a specific joint."""
    def modifier(spec, param):
        spec.joint(joint_name).damping[0] = param.value[0]
    return modifier


def make_shared_joint_attribute_modifier(
    joint_names: tuple[str, ...],
    attribute: str,
):
    """Create one modifier that applies a dynamic parameter to many joints."""
    if attribute not in {"armature", "frictionloss", "damping"}:
        raise ValueError(f"Unsupported shared joint attribute `{attribute}`.")

    def modifier(spec, param):
        for joint_name in joint_names:
            joint = spec.joint(joint_name)
            # MjSpec stores damping in an array, unlike the scalar fields below.
            if attribute == "damping":
                joint.damping[0] = param.value[0]
            else:
                setattr(joint, attribute, param.value[0])

    return modifier


def group_equality_constrained_joints(
    model,
    joint_names: list[str],
) -> list[tuple[str, ...]]:
    """Group joints connected by MuJoCo joint equality constraints.

    Groups are transitive and preserve ``joint_names`` order. Equality
    constraints that do not connect two selected joints are ignored.
    """
    import mujoco

    ordered_names = list(dict.fromkeys(joint_names))
    selected_names = set(ordered_names)
    # Union-find combines direct and indirect joint constraints into shared groups.
    parent = {joint_name: joint_name for joint_name in ordered_names}

    def find(joint_name: str) -> str:
        while parent[joint_name] != joint_name:
            parent[joint_name] = parent[parent[joint_name]]
            joint_name = parent[joint_name]
        return joint_name

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    joint_equality_type = int(mujoco.mjtEq.mjEQ_JOINT)
    for equality_id in range(int(model.neq)):
        if int(model.eq_type[equality_id]) != joint_equality_type:
            continue
        if (
            hasattr(model, "eq_active0")
            and not bool(model.eq_active0[equality_id])
        ):
            continue
        first_id = int(model.eq_obj1id[equality_id])
        second_id = int(model.eq_obj2id[equality_id])
        if first_id < 0 or second_id < 0:
            continue
        first_name = model.joint(first_id).name
        second_name = model.joint(second_id).name
        if first_name in selected_names and second_name in selected_names:
            union(first_name, second_name)

    grouped_names: dict[str, list[str]] = {}
    for joint_name in ordered_names:
        grouped_names.setdefault(find(joint_name), []).append(joint_name)
    return [tuple(group) for group in grouped_names.values()]


def clip_parameter_values_inside_bounds(
    parameter_dict,
    margin_scale: float = 1e-6,
):
    """Clip free parameter values just inside their bounds.

    Unlike ``ParameterDict.move_off_bounds``, the margin is based on the
    current value scale rather than a fixed fraction of the full bound range.
    Widening an upper bound therefore does not move an already-feasible
    starting point by a large amount.
    """
    if not np.isfinite(margin_scale) or margin_scale <= 0.0:
        raise ValueError("`margin_scale` must be positive and finite.")

    values = parameter_dict.as_vector()
    lower, upper = parameter_dict.get_bounds()
    if values.size == 0:
        return parameter_dict
    if np.any(~np.isfinite(values)):
        raise ValueError("Initial parameter values must be finite.")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("Parameter bounds must be finite.")
    if np.any(lower >= upper):
        raise ValueError(
            "Every lower parameter bound must be below its upper bound."
        )

    absolute_margin = margin_scale * np.maximum(1.0, np.abs(values))
    margin = np.minimum(absolute_margin, 0.25 * (upper - lower))
    clipped_values = np.clip(values, lower + margin, upper - margin)
    parameter_dict.update_from_vector(clipped_values)
    return parameter_dict


def make_body_mass_modifier(sysid, body_name: str):
    """Adapt MuJoCo's one-element mass parameter to its scalar body setter."""
    def modifier(spec, param):
        return sysid.model_modifier.apply_body_mass_ipos(
            spec,
            body_name,
            mass=param.value[0],
        )
    return modifier


_LEG_PREFIXES = ("FL", "FR", "RL", "RR")
_LEG_BODY_PATTERN = re.compile(r"^(FL|FR|RL|RR)_(.+)$")


def group_quadruped_leg_bodies(body_names: list[str]) -> list[tuple[str, ...]]:
    """Group corresponding FL/FR/RL/RR bodies, preserving input order.

    A suffix is grouped only when all four legs are present. Partial groups and
    bodies that do not use the standard quadruped naming convention remain
    independent.
    """
    by_suffix: dict[str, dict[str, str]] = {}
    for body_name in body_names:
        match = _LEG_BODY_PATTERN.fullmatch(body_name)
        if match is None:
            continue
        prefix, suffix = match.groups()
        by_suffix.setdefault(suffix, {})[prefix] = body_name

    # Tie leg parameters only when all four corresponding bodies are present.
    complete_groups = {
        suffix: tuple(by_prefix[prefix] for prefix in _LEG_PREFIXES)
        for suffix, by_prefix in by_suffix.items()
        if all(prefix in by_prefix for prefix in _LEG_PREFIXES)
    }

    groups: list[tuple[str, ...]] = []
    consumed: set[str] = set()
    for body_name in body_names:
        if body_name in consumed:
            continue
        match = _LEG_BODY_PATTERN.fullmatch(body_name)
        group = complete_groups.get(match.group(2)) if match else None
        if group is None:
            group = (body_name,)
        groups.append(group)
        consumed.update(group)
    return groups


def make_shared_body_inertia_modifier(sysid, body_names: tuple[str, ...]):
    """Create one modifier that applies an inertial parameter to many bodies."""

    def modifier(spec, param):
        for body_name in body_names:
            if param.inertia_type == sysid.InertiaType.Mass:
                # Avoid MuJoCo 3.11's shape-(1,) array/scalar setter mismatch.
                sysid.model_modifier.apply_body_mass_ipos(
                    spec,
                    body_name,
                    mass=param.value[0],
                )
            else:
                sysid.model_modifier.apply_body_inertia(spec, body_name, param)

    return modifier


def _validate_relative_bounds(
    bounds: tuple[float, float],
    name: str,
    nominal: float,
    *,
    positive: bool = False,
) -> tuple[float, float]:
    lower, upper = (float(value) for value in bounds)
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower >= upper
        or not lower <= nominal <= upper
        or (positive and lower <= 0.0)
    ):
        qualifier = "positive, " if positive else ""
        raise ValueError(
            f"`{name}` must be {qualifier}ordered and contain its nominal value "
            f"{nominal}; got ({lower}, {upper})."
        )
    return lower, upper


def build_parameter_dict(
    sysid,
    model,
    joint_names: list[str],
    bounds: dict[str, tuple[float, float]],
    *,
    model_spec=None,
    body_names: list[str] | None = None,
    identify_link_mass: bool = False,
    identify_center_of_mass: bool = False,
    identify_inertia_tensor: bool = False,
    tie_quadruped_inertias: bool = False,
):
    parameter_dict = sysid.ParameterDict()
    # Equality-coupled joints share one value per dynamic attribute.
    joint_groups = group_equality_constrained_joints(model, joint_names)
    shared_joint_groups = [group for group in joint_groups if len(group) > 1]
    if shared_joint_groups:
        print("Shared equality-constrained joint groups:", shared_joint_groups)

    for joint_group in joint_groups:
        for attribute in ("armature", "frictionloss", "damping"):
            lower, upper = bounds[attribute]
            current_value = float(
                np.mean(
                    [
                        _as_scalar(getattr(model.joint(joint_name), attribute))
                        for joint_name in joint_group
                    ]
                )
            )
            parameter_prefix = (
                joint_group[0]
                if len(joint_group) == 1
                else "shared_" + "_".join(joint_group)
            )
            parameter_name = f"{parameter_prefix}_{attribute}"
            parameter = sysid.Parameter(
                parameter_name,
                nominal=current_value,
                min_value=lower,
                max_value=upper,
                modifier=make_shared_joint_attribute_modifier(
                    joint_group,
                    attribute,
                ),
            )
            # Initial guesses are adjusted independently of the nominal model.
            # main() subsequently moves these values inside the requested bounds.
            if attribute == "frictionloss":
                parameter.value[:] = current_value*0.1
            elif attribute == "armature":
                parameter.value[:] = current_value + np.ones_like(parameter.value) * 1e-2
            elif attribute == "damping":
                parameter.value[:] = current_value*2

            parameter_dict.add(parameter)

    if identify_link_mass or identify_center_of_mass or identify_inertia_tensor:
        body_names = get_identifiable_body_names(model, body_names)
        if not body_names:
            raise ValueError(
                "No massive articulated bodies are available for inertial "
                "identification. Use explicit body names to override the default."
            )
        print("Inertial identification bodies:", body_names)
    else:
        body_names = []

    if body_names:
        if model_spec is None:
            raise ValueError(
                "`model_spec` is required for inertial identification."
            )

        # Scale bounds contain 1; additive offsets contain 0, retaining the nominal
        # model as a feasible reference for the inertial parameterization.
        mass_scale_bounds = _validate_relative_bounds(
            bounds["link_mass_scale"],
            "link_mass_scale bounds",
            nominal=1.0,
            positive=True,
        )
        com_offset_bounds = _validate_relative_bounds(
            bounds["center_of_mass_offset"],
            "center_of_mass_offset bounds",
            nominal=0.0,
        )
        inertia_scale_bounds = _validate_relative_bounds(
            bounds["inertia_tensor_scale"],
            "inertia_tensor_scale bounds",
            nominal=1.0,
            positive=True,
        )
        shear_bounds = _validate_relative_bounds(
            bounds["inertia_tensor_shear"],
            "inertia_tensor_shear bounds",
            nominal=0.0,
        )

        if identify_inertia_tensor:
            # Full inertia also includes mass and CoM, so it takes precedence.
            # The pseudo-inertia representation maintains physical consistency.
            inertia_type = sysid.InertiaType.Pseudo
            parameter_suffix = "full_inertia"
        elif identify_center_of_mass:
            inertia_type = sysid.InertiaType.MassIpos
            parameter_suffix = "mass_center_of_mass"
        else:
            inertia_type = sysid.InertiaType.Mass
            parameter_suffix = "link_mass"

        body_groups = (
            group_quadruped_leg_bodies(body_names)
            if tie_quadruped_inertias
            else [(body_name,) for body_name in body_names]
        )
        if tie_quadruped_inertias:
            shared_groups = [group for group in body_groups if len(group) > 1]
            if shared_groups:
                print("Shared quadruped inertial groups:", shared_groups)

        for body_group in body_groups:
            # A shared group uses its first body's nominal inertia and applies
            # each fitted candidate to all corresponding bodies through a modifier.
            body_name = body_group[0]
            if len(body_group) > 1:
                param_prefix = f"shared_{body_name.split('_', 1)[1]}"
                modifier = make_shared_body_inertia_modifier(sysid, body_group)
            else:
                param_prefix = body_name
                modifier = None
                if inertia_type == sysid.InertiaType.Mass:
                    # MuJoCo 3.11's default Mass modifier passes a shape-(1,)
                    # ndarray to a scalar MjsBody.mass setter.
                    modifier = make_body_mass_modifier(sysid, body_name)
            parameter_dict.add(
                sysid.body_inertia_param(
                    # Inertia setup may modify the spec; isolate it from other groups.
                    spec=model_spec.copy(),
                    model=model,
                    body_name=body_name,
                    inertia_type=inertia_type,
                    mass_bound_mult=np.asarray(
                        mass_scale_bounds,
                        dtype=np.float64,
                    ),
                    ipos_bound_off=np.asarray(
                        com_offset_bounds,
                        dtype=np.float64,
                    ),
                    stretch_bound_mult=np.asarray(
                        inertia_scale_bounds,
                        dtype=np.float64,
                    ),
                    shear_bound_off=np.asarray(
                        shear_bounds,
                        dtype=np.float64,
                    ),
                    param_name=f"{param_prefix}_{parameter_suffix}",
                    modifier=modifier,
                )
            )
    print("Initial parameter vector:", parameter_dict.as_vector())
    return parameter_dict
