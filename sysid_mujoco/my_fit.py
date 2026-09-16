from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sysid_mujoco.common import build_fixed_base_model_xml
from sysid_mujoco.common import build_actuator_gain_map
from sysid_mujoco.common import build_parameter_dict
from sysid_mujoco.common import chunk_processed_trajectory
from sysid_mujoco.common import clip_parameter_values_inside_bounds
from sysid_mujoco.common import get_actuated_joint_and_actuator_names
from sysid_mujoco.common import load_dataset_actuator_gains
from sysid_mujoco.common import load_processed_dataset
from sysid_mujoco.common import ProcessedTrajectory
from sysid_mujoco.common import processed_to_sysid_trajectory

import mujoco
from mujoco import sysid
import numpy as np
import config


PIPER_TORQUE_SCALE_KP = 30.0
PIPER_TORQUE_SCALE_KD = 25.0
PIPER_TORQUE_SCALED_JOINTS = {}


def default_converted_paths(robot: str) -> list[Path]:
    return sorted((REPO_ROOT / "sysid_mujoco" / "converted" / robot).glob("*.npz"))


def default_dataset_paths(robot: str) -> list[Path]:
    """Return every dataset for the selected robot as an independent chunk."""
    return sorted((REPO_ROOT / "datasets" / robot).glob("*.pt"))


def default_output_dir(robot: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "sysid_mujoco" / "results" / robot / timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate per-joint dynamics and optional per-link inertial "
            "properties from robot datasets using MuJoCo sysid."
        )
    )
    parser.add_argument(
        "--robot",
        default=config.robot,
        help="Robot name. Defaults to config.robot.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Optional raw repository datasets (.pt). By default, every .pt file "
            "inside datasets/<robot> is loaded as an independent chunk."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save the fitted parameters and identified XML models.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("mujoco", "scipy", "scipy_parallel_fd"),
        default="mujoco",
        help="Optimizer backend exposed by mujoco.sysid.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=30,
        help="Maximum optimizer iterations.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="If > 0, split each source trajectory into non-overlapping chunks of this size.",
    )
    parser.add_argument(
        "--identify-delays",
        action="store_true",
        help="Estimate one shared control delay in seconds for all actuators.",
    )
    parser.add_argument(
        "--delay-num-samples",
        type=int,
        default=30,
        help="Actuator delay history buffer size (default: 30 samples).",
    )
    parser.add_argument(
        "--delay-interp",
        choices=("linear", "zoh", "cubic"),
        default="linear",
        help=(
            "Delay history interpolation (default: linear). zoh is zero-order "
            "hold and can give zero delay gradients during optimization."
        ),
    )
    parser.add_argument(
        "--delay-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(0.0, 0.01),
        help="Shared delay bounds in seconds (default: 0 to 0.01).",
    )
    parser.add_argument(
        "--damping-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(0.001, 0.5),
        help="Bounds for each joint damping parameter.",
    )
    parser.add_argument(
        "--armature-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(0.001, 0.5),
        help="Bounds for each joint armature parameter.",
    )
    parser.add_argument(
        "--frictionloss-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(0.001, 0.5),
        help="Bounds for each joint frictionloss parameter.",
    )
    parser.add_argument(
        "--identify-link-mass",
        action="store_true",
        help="Also identify one mass parameter for each selected link.",
    )
    parser.add_argument(
        "--identify-center-of-mass",
        action="store_true",
        help=(
            "Identify mass and 3-D local center of mass for each selected link "
            "using MuJoCo's MassIpos parameterization."
        ),
    )
    parser.add_argument(
        "--identify-inertia-tensor",
        action="store_true",
        help=(
            "Identify full inertia (mass, CoM, and tensor) for each selected "
            "link using MuJoCo's pseudo-inertia Cholesky parameterization."
        ),
    )
    parser.add_argument(
        "--inertial-bodies",
        nargs="+",
        default=None,
        metavar="BODY",
        help=(
            "Body names used by the optional inertial parameters. By default, "
            "all massive bodies downstream of a joint are used."
        ),
    )
    parser.add_argument(
        "--link-mass-scale-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(0.5, 5.0),
        help="Mass bounds as multipliers of each link's nominal mass.",
    )
    parser.add_argument(
        "--center-of-mass-offset-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(-0.03, 0.03),
        help="Additive bounds in metres around every nominal CoM component.",
    )
    parser.add_argument(
        "--inertia-tensor-scale-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(0.5, 1.5),
        help=(
            "Multiplicative stretch bounds for the physically consistent "
            "inertia-tensor parameterization."
        ),
    )
    parser.add_argument(
        "--inertia-tensor-shear-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(-0.25, 0.25),
        help="Dimensionless shear bounds for the inertia tensor.",
    )
    parser.add_argument(
        "--tie-quadruped-inertias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use one mass/CoM/inertia parameter for corresponding FL, FR, RL, "
            "and RR bodies (default: enabled)."
        ),
    )
    return parser.parse_args()


def configure_actuator_delays(spec, num_samples: int, interp: str, bounds=None):
    """Allocate history before compiling states; return safe delay bounds."""
    if num_samples < 2:
        raise ValueError("`--delay-num-samples` must be at least 2.")
    capacity = num_samples * spec.option.timestep
    lower, upper = (0.0, capacity) if bounds is None else bounds
    if not np.isfinite([lower, upper]).all() or not 0 <= lower < upper <= capacity:
        raise ValueError(
            "`--delay-bounds` must satisfy 0 <= LOWER < UPPER <= "
            f"{capacity:g} seconds (history buffer capacity)."
        )
    if upper <= spec.option.timestep:
        raise ValueError("The delay upper bound must exceed one simulation timestep.")
    interpolation_order = {"zoh": 0, "linear": 1, "cubic": 2}[interp]
    for actuator in spec.actuators:
        actuator.nsample = num_samples
        actuator.interp = interpolation_order
    return float(lower), float(upper)


def add_shared_actuator_delay_parameter(params, spec, bounds):
    """Fit a single native control delay shared by every actuator."""
    actuators = list(spec.actuators)
    if not actuators:
        raise ValueError("Delay identification requires at least one actuator.")
    lower, upper = bounds
    # Start above the one-timestep plateau, where interpolation has a gradient.
    initial_delay = (max(lower, spec.option.timestep) + upper) / 2
    def modifier(model_spec, param):
        for actuator in model_spec.actuators:
            actuator.delay = float(param.value[0])

    parameter = sysid.Parameter(
        "shared_actuator_delay",
        nominal=float(np.mean([actuator.delay for actuator in actuators])),
        min_value=lower,
        max_value=upper,
        modifier=modifier,
    )
    parameter.value[:] = initial_delay
    params.add(parameter)


def _is_piper_robot(robot: str) -> bool:
    return robot == "piper_l"


def _model_joint_names(model) -> list[str]:
    return [model.joint(joint_id).name for joint_id in range(model.njnt)]


def _scale_actuator_gains_for_fit(
    robot: str,
    joint_names: list[str],
    kp: np.ndarray,
    kd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaled_kp = np.asarray(kp, dtype=np.float64).copy()
    scaled_kd = np.asarray(kd, dtype=np.float64).copy()
    if not _is_piper_robot(robot):
        return scaled_kp, scaled_kd

    for index, joint_name in enumerate(joint_names):
        if joint_name in PIPER_TORQUE_SCALED_JOINTS:
            scaled_kp[index] *= PIPER_TORQUE_SCALE_KP
            scaled_kd[index] *= PIPER_TORQUE_SCALE_KD
    return scaled_kp, scaled_kd


def _prepare_fixed_base_xml_for_fit(robot: str, model_xml: Path) -> Path:
    if not _is_piper_robot(robot):
        return model_xml

    tree = ET.parse(model_xml)
    root = tree.getroot()
    joint_names = {
        joint.get("name")
        for joint in root.iter("joint")
        if joint.get("name") is not None
    }
    if "joint8" not in joint_names:
        raise ValueError("Piper model must contain `joint8`.")

    sensor_element = root.find("sensor")
    if sensor_element is None:
        sensor_element = ET.SubElement(root, "sensor")

    for sensor in sensor_element.findall("jointpos"):
        if sensor.get("joint") == "joint8":
            return model_xml

    ET.SubElement(
        sensor_element,
        "jointpos",
        {
            "name": "joint8_pos",
            "joint": "joint8",
        },
    )
    tree.write(model_xml, encoding="utf-8", xml_declaration=True)
    return model_xml


def _insert_piper_mimic_joint8(
    values: np.ndarray,
    model_joint_names: list[str],
    value_name: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"`{value_name}` must be a 2-D array.")
    if "joint7" not in model_joint_names or "joint8" not in model_joint_names:
        raise ValueError("Piper model must contain both `joint7` and `joint8`.")
    if values.shape[1] == len(model_joint_names):
        return values

    source_joint_names = [name for name in model_joint_names if name != "joint8"]
    if values.shape[1] != len(source_joint_names):
        raise ValueError(
            f"Piper `{value_name}` has {values.shape[1]} joints, expected "
            f"{len(source_joint_names)} without `joint8` or "
            f"{len(model_joint_names)} with `joint8`."
        )

    joint7_index = source_joint_names.index("joint7")
    joint8_index = model_joint_names.index("joint8")
    joint8_values = -values[:, joint7_index:joint7_index + 1]
    return np.concatenate(
        (
            values[:, :joint8_index],
            joint8_values,
            values[:, joint8_index:],
        ),
        axis=1,
    )


def _extend_piper_state_to_joint8(
    trajectory: ProcessedTrajectory,
    model,
) -> ProcessedTrajectory:
    model_joint_names = _model_joint_names(model)
    return replace(
        trajectory,
        measured_qpos=_insert_piper_mimic_joint8(
            trajectory.measured_qpos,
            model_joint_names,
            "measured_qpos",
        ),
        measured_qvel=_insert_piper_mimic_joint8(
            trajectory.measured_qvel,
            model_joint_names,
            "measured_qvel",
        ),
        desired_qpos=_insert_piper_mimic_joint8(
            trajectory.desired_qpos,
            model_joint_names,
            "desired_qpos",
        ),
        joint_names=model_joint_names,
    )


def build_model_sequences_from_source(
    sysid,
    mujoco,
    model,
    robot: str,
    dataset_paths: list[Path],
    chunk_size: int,
):
    measurement_ts = []
    control_ts = []
    initial_states = []

    for dataset_path in dataset_paths:

        processed = load_processed_dataset(
            dataset_path=dataset_path,
            model=model,
            mujoco=mujoco,
        )
        if _is_piper_robot(robot):
            processed = _extend_piper_state_to_joint8(processed, model)
        for chunk in chunk_processed_trajectory(processed, chunk_size):
            measurement_data, control_data, initial_state = processed_to_sysid_trajectory(sysid, model, chunk)
            measurement_ts.append(measurement_data)
            control_ts.append(control_data)
            initial_states.append(initial_state)

    return measurement_ts, control_ts, initial_states


@contextmanager
def _safe_parameter_distribution_spacing():
    """Keep MuJoCo's parameter-distribution grid valid for many rows.

    MuJoCo 3.11.0 hard-codes a vertical spacing of 0.05. Plotly rejects that
    value when the report contains more than 21 subplot rows.
    """
    from mujoco.sysid.report.sections import parameter_distribution

    plotly_subplots = parameter_distribution.plt_subplots
    original_make_subplots = plotly_subplots.make_subplots

    def make_subplots_with_safe_spacing(*args, **kwargs):
        rows = kwargs.get("rows", 1)
        vertical_spacing = kwargs.get("vertical_spacing")
        if rows > 1 and vertical_spacing is not None:
            maximum_spacing = 1.0 / (rows - 1)
            if vertical_spacing > maximum_spacing:
                kwargs["vertical_spacing"] = 0.5 / (rows - 1)
        return original_make_subplots(*args, **kwargs)

    plotly_subplots.make_subplots = make_subplots_with_safe_spacing
    try:
        yield
    finally:
        plotly_subplots.make_subplots = original_make_subplots


def display_report(report, report_path: Path) -> Path:
    html = report.build()
    report_path.write_text(html, encoding="utf-8")
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display

        if getattr(get_ipython(), "kernel", None) is not None:
            display(HTML(html))
    except Exception:
        pass
    print(f"Report written to {report_path}")
    return report_path


def main() -> None:
    args = parse_args()
    if args.dataset is None:
        args.dataset = default_dataset_paths(args.robot)
    if not args.dataset:
        dataset_dir = REPO_ROOT / "datasets" / args.robot
        raise FileNotFoundError(f"No .pt datasets found in {dataset_dir}")

    print(f"Loading {len(args.dataset)} dataset chunk(s) for {args.robot}:")
    for dataset_path in args.dataset:
        print(f"  - {dataset_path}")
    if args.inertial_bodies is not None and not any(
        (
            args.identify_link_mass,
            args.identify_center_of_mass,
            args.identify_inertia_tensor,
        )
    ):
        raise ValueError(
            "`--inertial-bodies` requires at least one optional inertial "
            "identification flag."
        )

    dataset_kp, dataset_kd = load_dataset_actuator_gains(args.dataset[0])


    fixed_base_xml = _prepare_fixed_base_xml_for_fit(
        args.robot,
        build_fixed_base_model_xml(args.robot),
    )
    fixed_base_spec = mujoco.MjSpec.from_file(str(fixed_base_xml))
    fixed_base_spec.option.timestep = 1.0 / config.frequency_collection
    fixed_base_model = fixed_base_spec.compile()
    actuated_joint_names, _ = get_actuated_joint_and_actuator_names(mujoco, fixed_base_model)
    actuator_kp, actuator_kd = _scale_actuator_gains_for_fit(
        args.robot,
        actuated_joint_names,
        dataset_kp,
        dataset_kd,
    )
    fixed_base_xml = _prepare_fixed_base_xml_for_fit(
        args.robot,
        build_fixed_base_model_xml(
            args.robot,
            actuator_gains=build_actuator_gain_map(
                actuated_joint_names,
                actuator_kp,
                actuator_kd,
            ),
        ),
    )
    fixed_base_spec = mujoco.MjSpec.from_file(str(fixed_base_xml))
    fixed_base_spec.option.timestep = 1.0 / config.frequency_collection
    delay_bounds = None
    if args.identify_delays:
        delay_bounds = configure_actuator_delays(
            fixed_base_spec, args.delay_num_samples, args.delay_interp, args.delay_bounds
        )
        print(
            f"Estimating one shared actuator delay in {delay_bounds} seconds; "
            f"history={args.delay_num_samples}, interpolation={args.delay_interp}."
        )
        if args.delay_interp == "zoh":
            print("Warning: zoh can give zero delay gradients; prefer linear or cubic for fitting.")
    fixed_base_model = fixed_base_spec.compile()

    measurement_ts, control_ts, initial_states = build_model_sequences_from_source(
        sysid=sysid,
        mujoco=mujoco,
        model=fixed_base_model,
        robot=args.robot,
        dataset_paths=args.dataset,
        chunk_size=args.chunk_size,
    )

    joint_names = [fixed_base_model.joint(i).name for i in range(fixed_base_model.njnt)]
    sequence_names = [f"sequence_{index:03d}" for index, _ in enumerate(measurement_ts)]
    model_sequences = [sysid.ModelSequences(
        args.robot,
        fixed_base_spec,
        sequence_names[i],
        initial_states[i],
        control_ts[i],
        measurement_ts[i],
    ) for i in range(len(measurement_ts))]

    bounds = {
        "damping": tuple(float(value) for value in args.damping_bounds),
        "armature": tuple(float(value) for value in args.armature_bounds),
        "frictionloss": tuple(float(value) for value in args.frictionloss_bounds),
        "link_mass_scale": tuple(
            float(value) for value in args.link_mass_scale_bounds
        ),
        "center_of_mass_offset": tuple(
            float(value) for value in args.center_of_mass_offset_bounds
        ),
        "inertia_tensor_scale": tuple(
            float(value) for value in args.inertia_tensor_scale_bounds
        ),
        "inertia_tensor_shear": tuple(
            float(value) for value in args.inertia_tensor_shear_bounds
        ),
    }

    params = build_parameter_dict(
        sysid=sysid,
        model=fixed_base_model,
        model_spec=fixed_base_spec,
        joint_names=joint_names,
        bounds=bounds,
        body_names=args.inertial_bodies,
        identify_link_mass=args.identify_link_mass,
        identify_center_of_mass=args.identify_center_of_mass,
        identify_inertia_tensor=args.identify_inertia_tensor,
        tie_quadruped_inertias=args.tie_quadruped_inertias,
    )
    if delay_bounds is not None:
        add_shared_actuator_delay_parameter(params, fixed_base_spec, delay_bounds)
    
    clip_parameter_values_inside_bounds(params)
    # The optimizer flags values within 0.1% of a bound, even after clipping.
    params.move_off_bounds()
   
    residual_fn = sysid.build_residual_fn(models_sequences=model_sequences)

    opt_params, opt_result = sysid.optimize(
        initial_params=params,
        residual_fn=residual_fn,
        optimizer=args.optimizer,
        max_iters=args.max_iters,
        x_scale="jac",
    )


    output_dir = args.output_dir or default_output_dir(args.robot)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with _safe_parameter_distribution_spacing():
        report = sysid.default_report(
            models_sequences=model_sequences,
            initial_params=params,
            opt_params=opt_params,
            residual_fn=residual_fn,
            opt_result=opt_result,
            title=f"System Identification Report for {args.robot}",
            generate_videos=False,
        )
        display_report(report, output_dir / "report.html")


if __name__ == "__main__":
    main()
