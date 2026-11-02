"""Rigid-body inertial parameters (mass, center of mass, inertia) appended to the joint parameters of PACE.

``BodyParams`` identifies them as corrections of the nominal model that are physically consistent for any value
within the bounds. Corresponding FL/FR/RL/RR leg bodies can share their parameters, with mirrored CoM offsets.
"""

from __future__ import annotations

import re

import torch

_LEG_BODY_PATTERN = re.compile(r"^(FL|FR|RL|RR)_(.+)$")
_LEG_PREFIXES = ("FL", "FR", "RL", "RR")
_MIRROR_TOLERANCE = 1e-3  # [m] CoM components below it carry no mirroring information


def group_quadruped_leg_bodies(body_names):
    """Group corresponding FL/FR/RL/RR bodies (only when all four legs are present), preserving input order."""
    by_suffix = {}
    for name in body_names:
        match = _LEG_BODY_PATTERN.fullmatch(name)
        if match is not None:
            by_suffix.setdefault(match.group(2), {})[match.group(1)] = name
    complete = {
        suffix: tuple(by_prefix[prefix] for prefix in _LEG_PREFIXES)
        for suffix, by_prefix in by_suffix.items()
        if all(prefix in by_prefix for prefix in _LEG_PREFIXES)
    }
    groups, consumed = [], set()
    for name in body_names:
        if name in consumed:
            continue
        match = _LEG_BODY_PATTERN.fullmatch(name)
        group = complete.get(match.group(2)) if match else None
        group = group if group is not None else (name,)
        groups.append(group)
        consumed.update(group)
    return groups


def mirror_signs(com, reference_com, body_name, reference_name):
    """Signs mapping a CoM offset of the reference leg body to another leg body.

    An axis is mirrored when the nominal CoMs are mirrored along it. Without information (small components) the
    lateral axis is mirrored between left and right legs, the others are not.
    """
    signs = torch.ones(3, dtype=com.dtype)
    for axis in range(3):
        if abs(com[axis]) > _MIRROR_TOLERANCE and abs(reference_com[axis]) > _MIRROR_TOLERANCE:
            signs[axis] = 1.0 if com[axis] * reference_com[axis] > 0.0 else -1.0
        elif axis == 1 and body_name[1] != reference_name[1]:
            signs[axis] = -1.0
    return signs


def principal_second_moments(inertia):
    """Eigenvectors and principal second moments of the mass distribution of inertia tensors (..., 3, 3)."""
    moments, axes = torch.linalg.eigh(0.5 * (inertia + inertia.transpose(-1, -2)))
    # I_k = L_j + L_l  <=>  L_k = tr(I) / 2 - I_k, non-negative for physically consistent bodies
    second_moments = 0.5 * moments.sum(dim=-1, keepdim=True) - moments
    return axes, second_moments


def inertia_from_second_moments(axes, second_moments):
    moments = second_moments.sum(dim=-1, keepdim=True) - second_moments
    return axes @ torch.diag_embed(moments) @ axes.transpose(-1, -2)


class BodyParams:
    """Mass, CoM and inertia of selected bodies, appended to the PACE parameter vector.

    Parameters of each body group (a body, or its FL/FR/RL/RR copies when tie_quadruped is set):
      mass_scale             multiplier of the nominal mass; the inertia scales with it      [log scale]
      com_offset_{x,y,z}     offset of the CoM in the body frame [m], mirrored between legs
      inertia_scale_{1,2,3}  multipliers of the principal second moments of the mass distribution [log scale]
    Positive second moments are equivalent to the triangle inequality of the principal moments of inertia, so every
    value within the bounds is a physically consistent body. The principal axes are kept at their nominal value.
    """

    def __init__(
        self,
        articulation,
        offset,
        is_newton,
        body_names=None,
        identify_mass=True,
        identify_com=False,
        identify_inertia=False,
        tie_quadruped=True,
        mass_scale_bounds=(0.5, 2.0),
        com_offset_bounds=(-0.03, 0.03),
        inertia_scale_bounds=(0.5, 2.0),
    ):
        if not 0.0 < mass_scale_bounds[0] <= 1.0 <= mass_scale_bounds[1]:
            raise ValueError(f"mass scale bounds must be positive and contain 1, got {mass_scale_bounds}.")
        if not 0.0 < inertia_scale_bounds[0] <= 1.0 <= inertia_scale_bounds[1]:
            raise ValueError(f"inertia scale bounds must be positive and contain 1, got {inertia_scale_bounds}.")
        if not com_offset_bounds[0] <= 0.0 <= com_offset_bounds[1]:
            raise ValueError(f"CoM offset bounds must contain 0, got {com_offset_bounds}.")
        self.is_newton = is_newton
        self.device = articulation.device

        all_masses = articulation.data.body_mass.torch[0].detach().cpu()
        if body_names:
            body_ids, names = articulation.find_bodies(body_names, preserve_order=True)
        else:
            # the root is fixed in the experiment: its inertial parameters cannot be observed
            body_ids = [i for i in range(1, articulation.num_bodies) if all_masses[i] > 0.0]
            names = [articulation.body_names[i] for i in body_ids]
        if not body_ids:
            raise ValueError("No bodies selected for inertial identification.")
        if any(all_masses[i] <= 0.0 for i in body_ids):
            raise ValueError("Bodies selected for inertial identification must have a positive nominal mass.")
        self.body_names = list(names)
        self.body_ids = torch.tensor(body_ids, dtype=torch.int32, device=self.device)
        self.groups = group_quadruped_leg_bodies(self.body_names) if tie_quadruped else [(n,) for n in self.body_names]

        self.nominal_mass = all_masses[body_ids].to(self.device)
        self.nominal_com_pose = articulation.data.body_com_pose_b.torch[0, body_ids].detach().clone()
        nominal_inertia = articulation.data.body_inertia.torch[0, body_ids].detach().reshape(-1, 3, 3).double().cpu()
        axes, second_moments = principal_second_moments(nominal_inertia)
        self.principal_axes, self.nominal_second_moments = axes.to(self.device), second_moments.to(self.device)
        violating = (self.nominal_second_moments < 0.0).any(dim=-1)
        if bool(violating.any()):
            print("[WARN]: nominal inertia violates the triangle inequality, clamped for:",
                  [self.body_names[i] for i in torch.nonzero(violating).flatten().tolist()])
        scale = self.nominal_second_moments.sum(dim=-1, keepdim=True)
        self.nominal_second_moments = self.nominal_second_moments.clamp_min(1e-9 * scale)

        # parameter layout, one block of columns per group
        self.identify_mass = identify_mass or identify_com or identify_inertia
        self.identify_com = identify_com or identify_inertia
        self.identify_inertia = identify_inertia
        families = [("mass_scale", mass_scale_bounds, True)] if self.identify_mass else []
        if self.identify_com:
            families += [(f"com_offset_{axis} [m]", com_offset_bounds, False) for axis in "xyz"]
        if self.identify_inertia:
            families += [(f"inertia_scale_{axis}", inertia_scale_bounds, True) for axis in "123"]
        per_group = len(families)
        self.names = [f"{self._group_label(group)} {family}" for group in self.groups for family, _, _ in families]
        self.bounds = torch.tensor([bounds for _ in self.groups for _, bounds, _ in families], dtype=torch.float32)
        self.log_scale = torch.tensor([log for _ in self.groups for _, _, log in families], dtype=torch.bool)
        self.slice = slice(offset, offset + len(self.names))

        # for each body: columns of its group and mirroring of the CoM offset w.r.t. the first body of the group
        group_of = {name: g for g, group in enumerate(self.groups) for name in group}
        first = torch.tensor([group_of[name] * per_group for name in self.body_names], device=self.device)
        self._mass_cols = first if self.identify_mass else None
        com_start = int(self.identify_mass)
        self._com_cols = first[:, None] + com_start + torch.arange(3, device=self.device) if self.identify_com else None
        inertia_start = com_start + 3 * int(self.identify_com)
        self._inertia_cols = first[:, None] + inertia_start + torch.arange(3, device=self.device) if self.identify_inertia else None
        com = self.nominal_com_pose[:, :3].cpu()
        self._com_signs = torch.stack([
            mirror_signs(com[i], com[self.body_names.index(self.groups[group_of[name]][0])],
                         name, self.groups[group_of[name]][0])
            for i, name in enumerate(self.body_names)
        ]).to(self.device)

    @staticmethod
    def _group_label(group):
        if len(group) == 1:
            return group[0]
        return "/".join(name.split("_", 1)[0] for name in group) + "_" + group[0].split("_", 1)[1]

    def num_params(self):
        return len(self.names)

    def body_values(self, values):
        """Masses (K, B), CoM positions (K, B, 3) and inertias about the CoM (K, B, 3, 3) for params (K, P)."""
        values = values.to(self.device, torch.float32)
        num = values.shape[0]
        mass_scale = values[:, self._mass_cols] if self._mass_cols is not None else torch.ones((num, len(self.body_names)), device=self.device)
        masses = self.nominal_mass * mass_scale
        coms = self.nominal_com_pose[:, :3].expand(num, -1, -1).clone()
        if self._com_cols is not None:
            coms += values[:, self._com_cols] * self._com_signs
        second_moments = self.nominal_second_moments * mass_scale.double()[..., None]
        if self._inertia_cols is not None:
            second_moments = second_moments * values[:, self._inertia_cols].double()
        inertias = inertia_from_second_moments(self.principal_axes, second_moments).float()
        return masses, coms, inertias

    def write(self, articulation, values):
        """Write the inertial params (num_envs, P) into the simulation."""
        masses, coms, inertias = self.body_values(values)
        env_ids = torch.arange(values.shape[0], dtype=torch.int32, device=self.device)
        articulation.set_masses_index(masses=masses.contiguous(), body_ids=self.body_ids, env_ids=env_ids)
        # the inertia also changes with the mass, so it is always rewritten
        articulation.set_inertias_index(inertias=inertias.reshape(values.shape[0], -1, 9).contiguous(), body_ids=self.body_ids, env_ids=env_ids)
        if self._com_cols is not None:
            # Newton takes the CoM position only, PhysX the full pose with the (nominal) principal axes orientation
            if not self.is_newton:
                coms = torch.cat([coms, self.nominal_com_pose[:, 3:].expand(coms.shape[0], -1, -1)], dim=-1)
            articulation.set_coms_index(coms=coms.contiguous(), body_ids=self.body_ids, env_ids=env_ids)

    def to_dict(self, values):
        """Absolute inertial values for params (K, P), as stored by Isaac Lab (inertia about the CoM)."""
        masses, coms, inertias = self.body_values(values)
        return {"body_names": self.body_names,
                "param_names": self.names,
                "params": values.detach().cpu(),
                "mass": masses.cpu(),
                "com": coms.cpu(),
                "com_quat_xyzw": self.nominal_com_pose[:, 3:].cpu(),
                "inertia": inertias.cpu(),
                }

    def describe(self, values):
        """Lines with the params of the first row and [min, max] over the rows."""
        values = values.detach().cpu()
        lines = []
        for i, name in enumerate(self.names):
            column = values[:, i]
            row = f"{name:<40}{column[0].item():<10.4g}"
            if values.shape[0] > 1:
                row += f" [{column.min().item():<10.4g}, {column.max().item():<10.4g}]"
            lines.append(row)
        return lines
