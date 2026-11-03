"""Trajectories of the collection scripts for joint-space system identification.

Every collection mode is a JointTrajectory, computed once before execution, checked by the operator with
describe() and then sampled by elapsed time at every control step:
  - setpoint: a random posture, held until it is reached (or a timeout) and then for a short static phase;
  - falling: a random posture held for a while, then the gains go to zero and the legs fall (quadrupeds);
  - chirp: back and forth between the sys_id_1 and sys_id_2 keyframes, with shrinking periods;
  - excitation: per-joint chirps and a posture sweep inside the same joint box, described below.

The "chirp" of the collection scripts moves every joint back and forth between the two keyframes sys_id_1 and
sys_id_2, all joints in phase, with a period shrinking from 3 s to 0.2 s. Each limb therefore travels along a
single segment of its joint space: inertia, gravity and friction of different joints are excited together and are
hard to tell apart in the fit.

The excitation trajectory implemented here keeps the same joint box (each joint stays between its sys_id_1 and
sys_id_2 values, i.e. the range already executed on the robot by the chirp) and a similar duration, but:
  - every joint follows its own exponential chirp, with its own sweep, phase and amplitude modulation, so the
    joints of a limb move independently and every frequency is visited at several velocity levels;
  - a slow posture sweep is added on top, so the limb visits the whole box and gravity acts on the links in
    many different configurations;
  - the chirp amplitude is limited by a velocity bound, so the high frequencies come with small amplitudes;
  - the trajectory starts from, and returns to, the current posture with smooth ramps (zero velocity at both ends).

On the fixed-base Go2 in MuJoCo, with the same duration and at full speed (speed_scale = 1), this lowers the
Cramer-Rao bound of all the 21 parameters of a leg (armature, damping and friction of the 3 joints, mass and CoM
of hip, thigh and calf) by 1.1x to 8x w.r.t. the chirp; at speed_scale = 0.5 it is still better than the chirp
on almost all of them (2-3x on mass and CoM), at 0.3 it is about as informative as the chirp. Parameters that
are structurally non-identifiable with a fixed base (e.g. the hip mass, or the calf mass vs. its CoM) are not
improved by any trajectory.

On the piper arm (high gains, so it follows the reference closely) full speed keeps the end effector fast for
much longer than the chirp; at speed_scale = 0.5 peak joint velocity, peak acceleration and end-effector speed
are all below those of the chirp. Hence the default 0.5: increase it only once the motion looks safe.

The trajectory is fully determined by its seed and speed scale: record the fit and the validation trajectories
with different seeds. Usage from a collection node (see run_collection_quadruped_ros2.py and
run_collection_manipulator_ros2.py):

    trajectory = quadruped_excitation_trajectory(leg_setpoint_1, leg_setpoint_2, start, frequency=200.0,
                                                 seed=12, speed_scale=0.5)
    print(trajectory.describe())
    ...
    elapsed = time.time() - start_time                                # at every control step
    desired_joint_pos = trajectory.sample(elapsed)
    kp, kd = trajectory.gains(elapsed, kp, kd)
    if trajectory.done(elapsed, joint_pos):
        save(trajectory.name, trajectory.metadata)                    # dataset prefix and extra entries
"""

from __future__ import annotations

import numpy as np

DEFAULT_SEED = 12  # best of 30 candidates for the Go2 in the MuJoCo identifiability analysis
# Time scaling of the whole motion (posture sweep, chirp and modulation frequencies, velocity bound):
# 1.0 is the full-speed design, 0.5 the same kind of motion at half the speed. Start low on a new robot.
DEFAULT_SPEED_SCALE = 0.5
DEFAULT_DURATION = 24.0  # [s], about the duration of the chirp
RAMP_DURATION = 2.0  # [s], smooth transition from the start posture and back to it
# Default bound of the reference velocity: joint range / 0.15 s, about 2/3 of the fastest chirp cycle
# (a 0.2 s period covers the range in 0.1 s). With the low gains used for sys-id the robot follows less than that.
VELOCITY_TIME_CONSTANT = 0.15  # [s]
BOUND_MARGIN = 0.97  # use 97% of the half range around the center of the box

# Chirp: periods of the back and forth cycles at full speed [s], as executed by the original collection scripts
CHIRP_PERIODS = (3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3)
DEFAULT_CHIRP_SPEED_SCALE = 1.0

# Setpoint: hold the posture until every group of joints (e.g. a leg) is within the tolerance, or until the
# timeout, then record a static phase
SETPOINT_REACH_TOLERANCE = 0.1  # [rad], norm of the error of a group of joints
SETPOINT_REACH_TIMEOUT = 2.0  # [s]
SETPOINT_STATIC_DURATION = 0.5  # [s]

# Falling: hold the posture, then zero gains (free fall) until the end
FALLING_DURATION = 2.5  # [s]
FALLING_HOLD_TIMES = (2.0, 0.2)  # [s], the first with probability 1/4, the second otherwise
FALLING_TARGET = 20.0  # [rad], target sent with zero gains during the fall


def generate_excitation(lower, upper, duration=DEFAULT_DURATION, frequency=200.0, max_velocity=None,
                        seed=DEFAULT_SEED, speed_scale=1.0):
    """Excitation inside the joint box [lower, upper], one column per joint.

    Each joint j follows
        q_j(t) = center_j + 0.97 * (s_j * half_j * slow_j(t) + A_j(t) * sin(phi_j(t)))
    where slow_j is a slow posture sweep in [-1, 1], phi_j the phase of an exponential chirp and A_j(t) the
    chirp amplitude. A_j uses at most the (1 - s_j) share of the half range left by the posture sweep, so the
    trajectory never leaves the box.

    Args:
        lower, upper: joint bounds, shape (k,). Joints with lower == upper stay still at that value.
        duration: trajectory duration [s].
        frequency: sampling frequency [Hz], normally the control frequency of the collection.
        max_velocity: bound of the reference velocity of the chirp component at full speed [rad/s], scalar or
            shape (k,). Default: joint range / VELOCITY_TIME_CONSTANT.
        seed: random seed. It determines sweep, phases and modulation of every joint.
        speed_scale: time scaling in (0, 1]. Every frequency and the velocity bound are multiplied by it, so the
            velocities scale with it and the accelerations roughly with its square. The random draws do not
            depend on it: the same seed gives the same kind of motion at any speed.

    Returns:
        Joint positions, shape (round(duration * frequency), k).
    """
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lower.shape != upper.shape or np.any(upper < lower):
        raise ValueError(f"Invalid joint box: lower {lower}, upper {upper}.")
    if not 0.0 < speed_scale <= 1.0:
        raise ValueError(f"speed_scale must be in (0, 1], got {speed_scale}.")
    center, half = 0.5 * (lower + upper), 0.5 * (upper - lower)
    if max_velocity is None:
        max_velocity = (upper - lower) / VELOCITY_TIME_CONSTANT
    max_velocity = speed_scale * np.broadcast_to(np.asarray(max_velocity, dtype=float), lower.shape)

    rng = np.random.default_rng(seed)
    n = int(round(duration * frequency))
    t = np.arange(n) / frequency
    positions = np.empty((n, lower.size))
    for j in range(lower.size):
        # Share of the half range used by the slow posture sweep; the rest is left to the chirp
        slow_share = rng.uniform(0.25, 0.45)

        # Slow posture sweep: two sinusoids with random amplitude, period (7-20 s at full speed) and phase,
        # normalized to [-1, 1]. Over the whole duration the joints of a limb reach many combinations of the box,
        # not a single segment.
        slow = sum(
            rng.uniform(0.5, 1.0)
            * np.sin(2 * np.pi * speed_scale * rng.uniform(0.05, 0.15) * t + rng.uniform(0, 2 * np.pi))
            for _ in range(2)
        )
        slow /= np.abs(slow).max()

        # Exponential chirp: the frequency grows from f0 (0.2-0.4 Hz at full speed) to f1 (2.5-4 Hz) over the
        # whole duration. Different sweeps and phases per joint decorrelate the joints of the same limb.
        f0, f1 = rng.uniform(0.2, 0.4), rng.uniform(2.5, 4.0)
        f0, f1 = speed_scale * f0, speed_scale * f1
        chirp_frequency = f0 * (f1 / f0) ** (t / duration)
        phase = 2 * np.pi * np.cumsum(chirp_frequency) / frequency + rng.uniform(0, 2 * np.pi)

        # Amplitude modulation between 40% and 100%: every frequency is visited at several velocity levels,
        # which helps to separate Coulomb friction (independent of the velocity) from viscous friction.
        modulation = 0.4 + 0.6 * (
            0.5 + 0.5 * np.sin(2 * np.pi * speed_scale * rng.uniform(0.1, 0.3) * t + rng.uniform(0, 2 * np.pi))
        )

        # Largest amplitude within the share of the box and below the velocity bound
        # (the peak velocity of A sin(2 pi f t) is 2 pi f A)
        amplitude = np.minimum((1 - slow_share) * half[j], max_velocity[j] / (2 * np.pi * chirp_frequency))
        amplitude = amplitude * modulation

        positions[:, j] = center[j] + BOUND_MARGIN * (slow_share * half[j] * slow + amplitude * np.sin(phase))
    return positions


def generate_chirp(setpoint_1, setpoint_2, frequency=200.0, speed_scale=DEFAULT_CHIRP_SPEED_SCALE):
    """Chirp between two postures, one column per joint.

    Every cycle goes linearly from setpoint_1 to setpoint_2 and back, at constant velocity, with the periods of
    CHIRP_PERIODS: all joints in phase, each at its own velocity. The trajectory starts and ends at setpoint_1.

    Args:
        setpoint_1, setpoint_2: postures, shape (k,).
        frequency: sampling frequency [Hz], normally the control frequency of the collection.
        speed_scale: time scaling in (0, 1]. Every period is divided by it, so the velocities scale with it.

    Returns:
        Joint positions, shape (n, k).
    """
    if not 0.0 < speed_scale <= 1.0:
        raise ValueError(f"speed_scale must be in (0, 1], got {speed_scale}.")
    setpoint_1, setpoint_2 = np.asarray(setpoint_1, dtype=float), np.asarray(setpoint_2, dtype=float)
    cycles = []
    for period in CHIRP_PERIODS:
        period /= speed_scale
        t = np.arange(int(round(period * frequency))) / frequency
        triangle = 1.0 - np.abs(2.0 * t / period - 1.0)  # 0 -> 1 -> 0 over the period
        cycles.append(setpoint_1 + triangle[:, None] * (setpoint_2 - setpoint_1))
    cycles.append(setpoint_1[None])
    return np.concatenate(cycles)


def add_start_end_ramps(positions, start, frequency, ramp_duration=RAMP_DURATION):
    """Blend smoothly from the start posture into the trajectory, and back to it at the end.

    Every sample is a convex combination of the start posture and of the trajectory, with a smoothstep weight
    (0 -> 1 in ramp_duration): the motion starts and ends at rest, and every intermediate posture lies between
    the start posture and a posture inside the box.

    Args:
        positions: trajectory, shape (n, k).
        start: start (and end) posture, shape (k,), e.g. the current desired posture of the robot.
        frequency: sampling frequency of positions [Hz].
        ramp_duration: duration of each ramp [s].
    """
    positions = np.asarray(positions, dtype=float)
    start = np.asarray(start, dtype=float).reshape(1, -1)
    n = positions.shape[0]
    k = min(int(round(ramp_duration * frequency)), n // 2)
    x = np.linspace(0.0, 1.0, k)
    smoothstep = x * x * (3 - 2 * x)  # zero derivative at both ends
    weight = np.ones(n)
    weight[:k] = smoothstep
    weight[n - k:] = smoothstep[::-1]
    return start + weight[:, None] * (positions - start)


def mirror_leg_to_quadruped(leg_positions, mirror_right_hips=True):
    """Replicate a hip/thigh/calf trajectory, shape (n, 3), on the four legs in the FL, FR, RL, RR order.

    Same convention as the chirp of run_collection_quadruped_ros2.py: the hip of the right legs (FR, RR) is
    mirrored, except for robots whose hip axes are not mirrored (e.g. pegasus). With a fixed base the legs are
    dynamically independent, so the same signal on every leg does not reduce the identifiability.
    """
    positions = np.tile(np.asarray(leg_positions, dtype=float), (1, 4))
    if mirror_right_hips:
        positions[:, 3] *= -1.0  # FR hip
        positions[:, 9] *= -1.0  # RR hip
    return positions


class JointTrajectory:
    """Precomputed joint trajectory, sampled by elapsed time with linear interpolation between samples.

    Args:
        positions: joint positions, shape (n, k), sampled at frequency.
        frequency: sampling frequency [Hz].
        name: kind of trajectory, also the prefix of the saved dataset (e.g. "excitation").
        metadata: extra entries saved with the dataset (e.g. the seed).
        label: name shown to the operator, by default the name.
    """

    def __init__(self, positions, frequency, name, metadata=None, label=None):
        self.positions = np.atleast_2d(np.asarray(positions, dtype=float))
        self.frequency = float(frequency)
        self.name = name
        self.metadata = dict(metadata or {})
        self.label = label or name

    @property
    def duration(self):
        """Duration of the trajectory [s]."""
        return (self.positions.shape[0] - 1) / self.frequency

    def sample(self, elapsed_time):
        """Desired joint positions at elapsed_time [s]. Before the start and after the end it holds the end samples."""
        x = float(np.clip(elapsed_time * self.frequency, 0.0, self.positions.shape[0] - 1))
        i = int(x)
        i_next = min(i + 1, self.positions.shape[0] - 1)
        alpha = x - i
        return (1.0 - alpha) * self.positions[i] + alpha * self.positions[i_next]

    def gains(self, elapsed_time, kp, kd):
        """PD gains at elapsed_time [s]: the nominal ones, unless the trajectory changes them."""
        return kp, kd

    def done(self, elapsed_time, joint_pos):
        """Whether the collection is complete, given the measured joint positions (flattened)."""
        return elapsed_time > self.duration

    def describe(self, joint_names=None):
        """Summary to check before executing: duration, range and peak velocity of the reference of every joint."""
        names = joint_names or [f"joint_{j}" for j in range(self.positions.shape[1])]
        velocity = np.abs(np.diff(self.positions, axis=0)).max(axis=0, initial=0.0) * self.frequency
        details = "".join(f", {key} {value}" for key, value in self.metadata.items())
        lines = [
            f"{self.label.capitalize()} trajectory{details}, {self.duration:.1f} s at {self.frequency:.0f} Hz",
            f"{'joint':<20}{'min [rad]':>11}{'max [rad]':>11}{'max ref vel [rad/s]':>21}",
        ]
        for name, low, high, vel in zip(names, self.positions.min(axis=0), self.positions.max(axis=0), velocity):
            lines.append(f"{name:<20}{low:>11.3f}{high:>11.3f}{vel:>21.2f}")
        return "\n".join(lines)


class SetpointTrajectory(JointTrajectory):
    """Hold a posture until every group of joints reaches it (or a timeout), then for a static phase.

    Args:
        target: posture, shape (k,).
        frequency: control frequency [Hz].
        num_groups: the k joints are split in num_groups equal groups (e.g. 4 legs), each reached when the norm
            of its error is below SETPOINT_REACH_TOLERANCE.
    """

    def __init__(self, target, frequency, num_groups=1):
        super().__init__(np.asarray(target, dtype=float)[None], frequency, "setpoint")
        self.num_groups = num_groups
        self.reached_time = None

    @property
    def target(self):
        return self.positions[0]

    def done(self, elapsed_time, joint_pos):
        if self.reached_time is None:
            error = (self.target - np.asarray(joint_pos, dtype=float)).reshape(self.num_groups, -1)
            target_reached = bool(np.all(np.linalg.norm(error, axis=1) < SETPOINT_REACH_TOLERANCE))
            if target_reached or elapsed_time > SETPOINT_REACH_TIMEOUT:
                self.reached_time = elapsed_time
                reason = "target reached" if target_reached else "reach timeout"
                print(f"Setpoint {reason}; recording {SETPOINT_STATIC_DURATION:.1f} s of static data.")
        return self.reached_time is not None and elapsed_time - self.reached_time >= SETPOINT_STATIC_DURATION

    def describe(self, joint_names=None):
        names = joint_names or [f"joint_{j}" for j in range(self.target.size)]
        lines = ["Setpoint (joint positions [rad]):"]
        lines += [f"{name:<20}{value:>11.3f}" for name, value in zip(names, self.target)]
        return "\n".join(lines)


class FallingTrajectory(JointTrajectory):
    """Hold a posture for hold_time, then free fall with zero gains until FALLING_DURATION."""

    def __init__(self, target, frequency, hold_time):
        super().__init__(np.asarray(target, dtype=float)[None], frequency, "falling", {"falling_hold_time": hold_time})
        self.hold_time = hold_time

    @property
    def duration(self):
        return FALLING_DURATION

    def sample(self, elapsed_time):
        if elapsed_time > self.hold_time:
            return np.full_like(self.positions[0], FALLING_TARGET)
        return self.positions[0].copy()

    def gains(self, elapsed_time, kp, kd):
        if elapsed_time > self.hold_time:
            return kp * 0.0, kd * 0.0
        return kp, kd


def random_quadruped_setpoint(rng=None):
    """Random posture of the 12 joints (FL, FR, RL, RR), with the hips of the right legs mirrored."""
    rng = np.random.default_rng() if rng is None else rng
    hip = rng.uniform(-0.0, 0.5)
    thigh = rng.uniform(-1.0, 0.5)
    calf = rng.uniform(-0.0, 1.5)
    leg = np.array([0.0 + hip, 1.21 + thigh, -2.794 + calf])
    return mirror_leg_to_quadruped(leg[None], mirror_right_hips=True)[0]


def quadruped_setpoint_trajectory(frequency, rng=None):
    """Random setpoint of a quadruped, reached when every leg is within the tolerance."""
    return SetpointTrajectory(random_quadruped_setpoint(rng), frequency, num_groups=4)


def quadruped_falling_trajectory(frequency, rng=None):
    """Random setpoint of a quadruped, then free fall. The hold time is drawn once per collection."""
    rng = np.random.default_rng() if rng is None else rng
    hold_time = FALLING_HOLD_TIMES[0] if rng.integers(0, 4) == 0 else FALLING_HOLD_TIMES[1]
    return FallingTrajectory(random_quadruped_setpoint(rng), frequency, hold_time)


def manipulator_setpoint_trajectory(setpoint_1, setpoint_2, frequency, rng=None):
    """Random setpoint of a manipulator inside the joint box of the chirp."""
    rng = np.random.default_rng() if rng is None else rng
    target = rng.uniform(np.minimum(setpoint_1, setpoint_2), np.maximum(setpoint_1, setpoint_2))
    return SetpointTrajectory(target, frequency, num_groups=1)


def quadruped_chirp_trajectory(leg_setpoint_1, leg_setpoint_2, frequency, speed_scale=DEFAULT_CHIRP_SPEED_SCALE,
                               mirror_right_hips=True):
    """Chirp of a fixed-base quadruped between the sys_id_1 and sys_id_2 keyframes, on the four legs.

    Args:
        leg_setpoint_1, leg_setpoint_2: hip, thigh, calf of the sys_id_1 and sys_id_2 keyframes (front-left leg).
        frequency: control frequency [Hz].
        speed_scale: time scaling in (0, 1], see generate_chirp.
        mirror_right_hips: mirror the hip of the right legs (False for pegasus).
    """
    leg = generate_chirp(leg_setpoint_1, leg_setpoint_2, frequency, speed_scale)
    positions = mirror_leg_to_quadruped(leg, mirror_right_hips)
    return JointTrajectory(positions, frequency, "trajectory", {"chirp_speed_scale": speed_scale}, label="chirp")


def manipulator_chirp_trajectory(setpoint_1, setpoint_2, frequency, speed_scale=DEFAULT_CHIRP_SPEED_SCALE):
    """Chirp of a manipulator between the sys_id_1 and sys_id_2 keyframes (without the gripper)."""
    positions = generate_chirp(setpoint_1, setpoint_2, frequency, speed_scale)
    return JointTrajectory(positions, frequency, "trajectory", {"chirp_speed_scale": speed_scale}, label="chirp")


def _excitation_metadata(seed, speed_scale):
    return {"excitation_seed": seed, "excitation_speed_scale": speed_scale}


def quadruped_excitation_trajectory(leg_setpoint_1, leg_setpoint_2, start, frequency, seed=DEFAULT_SEED,
                                    speed_scale=DEFAULT_SPEED_SCALE, mirror_right_hips=True,
                                    duration=DEFAULT_DURATION):
    """Excitation for a fixed-base quadruped, inside the joint box of the chirp.

    Args:
        leg_setpoint_1, leg_setpoint_2: hip, thigh, calf of the sys_id_1 and sys_id_2 keyframes (front-left leg).
        start: current desired posture of the 12 joints (FL, FR, RL, RR), where the trajectory starts and ends.
        frequency: control frequency [Hz].
        seed: random seed of the trajectory.
        speed_scale: time scaling in (0, 1], see generate_excitation.
        mirror_right_hips: mirror the hip of the right legs, as the chirp does (False for pegasus).
        duration: duration [s], ramps included.
    """
    lower = np.minimum(leg_setpoint_1, leg_setpoint_2)
    upper = np.maximum(leg_setpoint_1, leg_setpoint_2)
    leg = generate_excitation(lower, upper, duration=duration, frequency=frequency, seed=seed, speed_scale=speed_scale)
    positions = mirror_leg_to_quadruped(leg, mirror_right_hips)
    return JointTrajectory(add_start_end_ramps(positions, start, frequency), frequency, "excitation",
                           _excitation_metadata(seed, speed_scale))


def manipulator_excitation_trajectory(setpoint_1, setpoint_2, start, frequency, seed=DEFAULT_SEED,
                                      speed_scale=DEFAULT_SPEED_SCALE, duration=DEFAULT_DURATION):
    """Excitation for a manipulator, inside the joint box of the chirp.

    Args:
        setpoint_1, setpoint_2: joint positions of the sys_id_1 and sys_id_2 keyframes (without the gripper).
        start: current desired posture, where the trajectory starts and ends.
        frequency: control frequency [Hz].
        seed: random seed of the trajectory.
        speed_scale: time scaling in (0, 1], see generate_excitation.
        duration: duration [s], ramps included.
    """
    lower = np.minimum(setpoint_1, setpoint_2)
    upper = np.maximum(setpoint_1, setpoint_2)
    positions = generate_excitation(lower, upper, duration=duration, frequency=frequency, seed=seed,
                                    speed_scale=speed_scale)
    return JointTrajectory(add_start_end_ramps(positions, start, frequency), frequency, "excitation",
                           _excitation_metadata(seed, speed_scale))
