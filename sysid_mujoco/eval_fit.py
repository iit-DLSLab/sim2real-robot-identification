from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from sysid_mujoco.common import ProcessedTrajectory
from sysid_mujoco.common import _absolutize_file_attributes
from sysid_mujoco.common import _disable_all_collisions
from sysid_mujoco.common import _remove_all_by_tag
from sysid_mujoco.common import _rewrite_actuators_as_general
from sysid_mujoco.common import build_actuator_gain_map
from sysid_mujoco.common import build_fixed_base_model_xml
from sysid_mujoco.common import get_actuated_joint_and_actuator_names
from sysid_mujoco.common import load_dataset_actuator_gains
from sysid_mujoco.common import load_processed_dataset


def parse_args() -> argparse.Namespace:
    # Select the recording, model variant and plot output for an open-loop replay.
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate model fitting quality by replaying a dataset in MuJoCo and "
            "plotting desired/measured/simulated trajectories."
        )
    )
    parser.add_argument(
        "--robot",
        default=config.robot,
        help="Robot name. Defaults to config.robot.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "datasets" / config.robot / "trajectory_1.pt",
        help="Dataset .pt path. Defaults to datasets/<robot>/trajectory_1.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder for plots and metrics JSON.",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show plots interactively (default: true).",
    )
    parser.add_argument(
        "--original",
        action="store_true",
        help=(
            "Load robot_model/<robot>/<robot>_original.xml instead of the default "
            "model selected via scene_flat.xml."
        ),
    )
    return parser.parse_args()


def default_output_dir(robot: str) -> Path:
    # Keep evaluation outputs separate from fitting reports and previous replays.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "sysid_mujoco" / "results" / robot / f"eval_{timestamp}"


def _model_joint_names(model: mujoco.MjModel) -> list[str]:
    return [model.joint(joint_id).name for joint_id in range(model.njnt)]


def _insert_piper_mimic_joint8(values: np.ndarray, model_joint_names: list[str]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Expected a 2-D trajectory array.")

    if "joint7" not in model_joint_names or "joint8" not in model_joint_names:
        return values

    if values.shape[1] == len(model_joint_names):
        return values

    source_joint_names = [name for name in model_joint_names if name != "joint8"]
    if values.shape[1] != len(source_joint_names):
        raise ValueError(
            "Piper trajectory dimensionality mismatch: expected either "
            f"{len(source_joint_names)} joints (without joint8) or "
            f"{len(model_joint_names)} joints (with joint8), got {values.shape[1]}."
        )

    joint7_index = source_joint_names.index("joint7")
    joint8_index = model_joint_names.index("joint8")
    # Reconstruct the passive finger with opposite sign in the model's joint order.
    joint8_values = -values[:, joint7_index : joint7_index + 1]
    return np.concatenate(
        (
            values[:, :joint8_index],
            joint8_values,
            values[:, joint8_index:],
        ),
        axis=1,
    )


def _extend_piper_trajectory_if_needed(
    trajectory: ProcessedTrajectory,
    model: mujoco.MjModel,
) -> ProcessedTrajectory:
    model_joint_names = _model_joint_names(model)
    # The mimic finger adds a state column but no extra actuator command.
    return replace(
        trajectory,
        measured_qpos=_insert_piper_mimic_joint8(trajectory.measured_qpos, model_joint_names),
        measured_qvel=_insert_piper_mimic_joint8(trajectory.measured_qvel, model_joint_names),
        desired_qpos=_insert_piper_mimic_joint8(trajectory.desired_qpos, model_joint_names),
        joint_names=model_joint_names,
    )


def _build_eval_model(robot: str, dataset_path: Path) -> tuple[mujoco.MjModel, Path]:
    # Compile a provisional model to map the recorded gains to actuator order.
    provisional_xml = build_fixed_base_model_xml(robot)
    provisional_model = mujoco.MjModel.from_xml_path(str(provisional_xml))
    joint_names, _ = get_actuated_joint_and_actuator_names(mujoco, provisional_model)
    kp, kd = load_dataset_actuator_gains(dataset_path, num_joints=len(joint_names))

    # Use the recording's controller gains when rebuilding the replay model.
    model_xml = build_fixed_base_model_xml(
        robot,
        actuator_gains=build_actuator_gain_map(joint_names, kp, kd),
    )
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    return model, model_xml


def _build_fixed_base_model_xml_from_source(
    robot: str,
    source_xml: Path,
    actuator_gains: dict[str, tuple[float, float]] | None = None,
) -> Path:
    output_dir = REPO_ROOT / "sysid_mujoco" / "generated" / robot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_xml = output_dir / f"{source_xml.stem}_fixed_base_sysid.xml"

    tree = ET.parse(source_xml)
    root = tree.getroot()
    # Match the fitting setup: fixed base, joint control and no contact forces.
    _remove_all_by_tag(root, "freejoint")
    _remove_all_by_tag(root, "keyframe")
    _remove_all_by_tag(root, "accelerometer")
    _remove_all_by_tag(root, "gyro")
    _remove_all_by_tag(root, "framepos")
    _remove_all_by_tag(root, "framequat")
    _rewrite_actuators_as_general(root, actuator_gains=actuator_gains)
    _absolutize_file_attributes(root, source_xml.parent)
    _disable_all_collisions(root)

    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    return output_xml


def _get_original_model_xml(robot: str) -> Path:
    # The original variant is explicit; do not silently fall back to another model.
    original_xml = REPO_ROOT / "robot_model" / robot / f"{robot}_original.xml"
    if not original_xml.is_file():
        raise FileNotFoundError(
            f"Original model not found: {original_xml}. "
            "Expected robot_model/<robot>/<robot>_original.xml."
        )
    return original_xml


def _build_eval_model_from_original(
    robot: str,
    dataset_path: Path,
) -> tuple[mujoco.MjModel, Path]:
    # Apply the same preprocessing and gain mapping to the original model variant.
    original_xml = _get_original_model_xml(robot)
    provisional_xml = _build_fixed_base_model_xml_from_source(robot, original_xml)
    provisional_model = mujoco.MjModel.from_xml_path(str(provisional_xml))
    joint_names, _ = get_actuated_joint_and_actuator_names(mujoco, provisional_model)
    kp, kd = load_dataset_actuator_gains(dataset_path, num_joints=len(joint_names))

    model_xml = _build_fixed_base_model_xml_from_source(
        robot,
        original_xml,
        actuator_gains=build_actuator_gain_map(joint_names, kp, kd),
    )
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    return model, model_xml


def simulate_open_loop(
    model: mujoco.MjModel,
    trajectory: ProcessedTrajectory,
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    times = trajectory.times
    ctrl = trajectory.ctrl
    model.opt.timestep = 1.0 / config.frequency_collection

    if len(times) < 2:
        raise ValueError("The trajectory must contain at least two samples.")
    if ctrl.shape[1] != model.nu:
        raise ValueError(
            f"Control dimensionality mismatch: ctrl has {ctrl.shape[1]} columns, "
            f"but model has {model.nu} actuators."
        )

    # Initialize from measurements once; later states evolve freely in simulation.
    # This exposes accumulated model error instead of resetting it every sample.
    data.qpos[:] = trajectory.measured_qpos[0]
    data.qvel[:] = trajectory.measured_qvel[0]
    mujoco.mj_forward(model, data)

    simulated_qpos = np.zeros_like(trajectory.measured_qpos)
    simulated_qvel = np.zeros_like(trajectory.measured_qvel)
    simulated_qpos[0] = data.qpos.copy()
    simulated_qvel[0] = data.qvel.copy()

    for step in range(len(times) - 1):
        dt = float(times[step + 1] - times[step])
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                "Non-increasing or invalid time vector in dataset at "
                f"index {step}: dt={dt}."
            )

        # Approximate the sample interval with whole simulation steps and hold
        # the recorded control constant over that interval.
        substeps = max(1, int(round(dt / model.opt.timestep)))
        data.ctrl[:] = ctrl[step]
        for _ in range(substeps):
            mujoco.mj_step(model, data)

        simulated_qpos[step + 1] = data.qpos.copy()
        simulated_qvel[step + 1] = data.qvel.copy()

    return simulated_qpos, simulated_qvel


def compute_metrics(
    measured_qpos: np.ndarray,
    simulated_qpos: np.ndarray,
    joint_names: list[str],
) -> dict[str, object]:
    # Report position errors in joint units, without the fitting loss normalization.
    # Average over time first, then average the per-joint metrics for the summary.
    error = simulated_qpos - measured_qpos
    rmse_per_joint = np.sqrt(np.mean(error**2, axis=0))
    mae_per_joint = np.mean(np.abs(error), axis=0)

    return {
        "rmse_mean": float(np.mean(rmse_per_joint)),
        "mae_mean": float(np.mean(mae_per_joint)),
        "rmse_per_joint": {
            joint_name: float(value)
            for joint_name, value in zip(joint_names, rmse_per_joint, strict=True)
        },
        "mae_per_joint": {
            joint_name: float(value)
            for joint_name, value in zip(joint_names, mae_per_joint, strict=True)
        },
    }


def plot_joint_trajectories(
    times: np.ndarray,
    desired_qpos: np.ndarray,
    measured_qpos: np.ndarray,
    simulated_qpos: np.ndarray,
    joint_names: list[str],
    robot: str,
    dataset_path: Path,
    output_path: Path,
    show: bool,
) -> None:
    if not show:
        # Select a noninteractive backend before importing pyplot for headless runs.
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_joints = measured_qpos.shape[1]
    num_cols = min(3, num_joints)
    num_rows = math.ceil(num_joints / num_cols)
    figure, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(5.4 * num_cols, 3.2 * num_rows),
        sharex=True,
        squeeze=False,
    )

    # Use one panel per joint and hide unused cells in the final row.
    for joint_index, axis in enumerate(axes.flat):
        if joint_index >= num_joints:
            axis.set_visible(False)
            continue

        axis.plot(
            times,
            desired_qpos[:, joint_index],
            label="desired",
            color="tab:orange",
            linestyle="--",
            linewidth=1.6,
        )
        axis.plot(
            times,
            measured_qpos[:, joint_index],
            label="measured",
            color="tab:blue",
            linewidth=1.5,
        )
        axis.plot(
            times,
            simulated_qpos[:, joint_index],
            label="simulated",
            color="tab:red",
            linewidth=1.2,
            alpha=0.95,
        )
        axis.set_title(joint_names[joint_index])
        axis.set_xlabel("Time [s]")
        axis.set_ylabel("Position")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle(
        f"Fit evaluation trajectories - {robot} - {dataset_path.name}",
        fontsize=12,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save the figure regardless of whether an interactive window is requested.
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved trajectory plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(figure)


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = args.output_dir or default_output_dir(args.robot)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compare either the scene-selected model or the explicitly requested original.
    if args.original:
        model, model_xml = _build_eval_model_from_original(args.robot, dataset_path)
    else:
        model, model_xml = _build_eval_model(args.robot, dataset_path)
    processed = load_processed_dataset(
        dataset_path=dataset_path,
        model=model,
        mujoco=mujoco,
        actuator_mode="general",
    )
    processed = _extend_piper_trajectory_if_needed(processed, model)

    # Replay the complete recording from its measured initial state.
    simulated_qpos, simulated_qvel = simulate_open_loop(model, processed)
    metrics = compute_metrics(
        measured_qpos=processed.measured_qpos,
        simulated_qpos=simulated_qpos,
        joint_names=processed.joint_names,
    )

    # Store input provenance alongside the metrics so results can be traced back.
    metrics_payload = {
        "robot": args.robot,
        "dataset": str(dataset_path),
        "model_xml": str(model_xml),
        "num_samples": int(processed.times.shape[0]),
        "metrics": metrics,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")
    print(f"Mean RMSE: {metrics['rmse_mean']:.6f}")
    print(f"Mean MAE:  {metrics['mae_mean']:.6f}")

    plot_joint_trajectories(
        times=processed.times,
        desired_qpos=processed.desired_qpos,
        measured_qpos=processed.measured_qpos,
        simulated_qpos=simulated_qpos,
        joint_names=processed.joint_names,
        robot=args.robot,
        dataset_path=dataset_path,
        output_path=output_dir / "joint_trajectories.png",
        show=args.show,
    )


if __name__ == "__main__":
    main()
