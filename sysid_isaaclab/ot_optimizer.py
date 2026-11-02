"""Diversity-regularized optimal-transport optimizer for PACE.

Gradient-free particle optimizer based on the Sinkhorn coordinate descent of
"Sampling-Based Control via Entropy-Regularized Optimal Transport" (Pacelli et al., arXiv:2605.02147),
used here to return a *set* of plausible parameter combinations instead of a single one.

Given N particles Theta = {theta_1, ..., theta_N} it minimizes

    F(Theta) = 1/N sum_i L(theta_i) / L_best  -  lambda * D(Theta)    s.t.  L(theta_i) <= L_max

where L is the PACE tracking loss, D(Theta) = 1/N logdet(K(Theta) + delta I) is a log-det (DPP) diversity
with an RBF kernel K, and L_max = (1 + loss_tolerance) * L_best (or an absolute loss_limit).

Each iteration evaluates the particles together with proposals sampled around them (plus uniform global
samples), then:
  1. builds the target marginal p_j ~ exp(-beta * excess_j) on all evaluated points, where excess_j is the
     loss excess over the best loss normalized by the loss limit (steeper beyond the limit);
  2. couples particles and evaluated points with entropic OT (log-domain Sinkhorn), using the transport cost
     1/2 ||theta_i - y_j||^2 plus the coordinate-wise diversity term of F (log-det gain of replacing
     theta_i with y_j);
  3. moves each particle towards its barycentric projection with a relaxed step.
All evaluated points within the loss limit also feed a plausible set, chosen greedily to minimize F.

Two changes w.r.t. the paper, both reducing to it when disabled:
  - marginal_relaxation < 1 relaxes the target marginal (unbalanced OT). With balanced OT every particle is
    transported towards the best region found so far, so plausible basins discovered later are lost;
  - covariance_learning_rate > 0 learns the shape of the proposal covariance from the coupling (rank-mu
    update as in CMA-ES), needed on ill-conditioned identification problems.
"""

from __future__ import annotations

import math
import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from isaaclab.utils import configclass


@configclass
class OTSinkhornOptimizerCfg:
    """Diversity-regularized OT optimizer configuration."""
    max_iteration: int = 50
    num_particles: int = 32  # also the size of the returned set of plausible combinations
    diversity_weight: float = 0.02  # lambda, in units of relative loss (L / L_best)
    loss_tolerance: float = 0.1  # plausible if L <= (1 + loss_tolerance) * L_best
    loss_limit: float | None = None  # absolute loss limit, overrides loss_tolerance if set
    diversity_bandwidth: float = 0.05  # RBF bandwidth, as RMS fraction of the bounds range
    sigma: float = 0.2  # initial std of the local proposals in normalized units (bounds mapped to [-1, 1])
    sigma_min: float = 0.01
    sigma_decay: float = 0.98
    covariance_learning_rate: float = 0.2  # shape of the proposal covariance learned from the coupling, 0: isotropic
    global_ratio: float = 0.1  # fraction of proposals sampled uniformly within the bounds
    step_size: float = 0.5  # relaxed barycentric step
    ot_epsilon: float = 0.05  # entropic regularization, times the median transport cost
    marginal_relaxation: float = 0.75  # 1: balanced OT as in the paper (single basin); < 1: soft target marginal
    beta_max: float = 3.0  # max inverse temperature: points at the loss limit weigh exp(-beta_max) of the best
    ess_fraction: float = 0.005  # min effective sample size of the target marginal, fraction of the evaluated points
    sinkhorn_iterations: int = 200
    save_interval: int = 10
    save_optimization_process: bool = False  # consume more disk space if True, saves optimization process after finishing
    seed: int = 0


class OTSinkhornOptimizer:
    _DPP_JITTER = 1e-3  # delta in logdet(K + delta I), caps the penalty of a duplicate to -lambda * log(delta)
    _LIMIT_PENALTY = 10.0  # extra slope of the target energy beyond the loss limit
    _SINKHORN_TOL = 1e-4

    def __init__(self, bounds, batch_size, log_dir, joint_order, data, device, cfg: OTSinkhornOptimizerCfg, separate_dynamic_friction=False):

        self.cfg = cfg
        self.joint_order = joint_order
        self.max_iteration = cfg.max_iteration
        self.save_interval = cfg.save_interval
        self.device = device
        self.save_optimization_process = cfg.save_optimization_process
        self.separate_dynamic_friction = separate_dynamic_friction
        self._timer_start = datetime.now()  # timer for logging purposes

        # create log_dir in YY_MM_DD_hh-mm-ss format
        folder_time = datetime.now().strftime("%y_%m_%d_%H-%M-%S")
        log_dir = os.path.join(log_dir, folder_time)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = TensorboardSummaryWriter(log_dir=log_dir)
        torch.save({"bounds": bounds,
                    "joint_order": joint_order,
                    "dof_pos": data["dof_pos"],
                    "des_dof_pos": data["des_dof_pos"],
                    "time": data["time"],
                    "separate_dynamic_friction": separate_dynamic_friction,
                    "optimizer": "ot",
                    }, log_dir + "/config.pt")

        self.bounds = bounds
        num_params = bounds.shape[0]
        self.num_particles = cfg.num_particles
        if batch_size <= self.num_particles:
            raise ValueError(f"num_envs ({batch_size}) must be larger than num_particles ({self.num_particles}).")
        self.batch_size = batch_size
        self.generator = torch.Generator(device=device).manual_seed(cfg.seed)

        # params with equal bounds are fixed: kept at 0 so that they add no fake diversity
        self.free_params = bounds[:, 1] > bounds[:, 0]

        # particles live in the normalized box [-1, 1]^d, as the CMA-ES parameters
        self.particles = self._sample_uniform(self.num_particles)
        self.sigma = cfg.sigma
        self.covariance = torch.diag(self.free_params.float())  # proposal shape, normalized to trace = number of free params
        self.covariance_chol = self.covariance.clone()

        self.best_loss = math.inf
        self.best_params = torch.zeros(num_params, device=device)
        self.best_trajectory = None
        self.plausible_params = torch.zeros((0, num_params), device=device)
        self.plausible_losses = torch.zeros(0, device=device)

        self.scores_counter = 0
        self.iteration_counter = 0

        self.scores = torch.zeros(batch_size, device=device)
        self.scores_buffer = torch.zeros((self.max_iteration, batch_size), device=device)
        self.sim_dof_pos_buffer = torch.zeros((batch_size, data["dof_pos"].shape[0], len(joint_order)), device=device)

        self.params = torch.zeros((batch_size, num_params), device=device)
        self.sim_params = torch.zeros_like(self.params)
        if self.save_optimization_process:
            self.sim_params_buffer = torch.zeros((self.max_iteration, batch_size, num_params), device=device)

        num_joints = len(joint_order)
        self.armature_idx = slice(0, num_joints)
        self.damping_idx = slice(num_joints, 2 * num_joints)
        self.friction_idx = slice(2 * num_joints, 3 * num_joints)
        self.bias_idx = slice(3 * num_joints, 4 * num_joints)
        self.delay_idx = 4 * num_joints
        # dynamic to static friction ratios come after the delay, as in PACE (and before any appended param)
        self.friction_ratio_idx = slice(4 * num_joints + 1, 5 * num_joints + 1)
        # dynamic friction = ratio * static friction must stay within [0, static friction]
        if separate_dynamic_friction:
            ratio_bounds = bounds[self.friction_ratio_idx]
            if ratio_bounds.shape[0] != num_joints or torch.any(ratio_bounds[:, 0] < 0.0) or torch.any(ratio_bounds[:, 1] > 1.0) or torch.any(ratio_bounds[:, 0] > ratio_bounds[:, 1]):
                raise ValueError(f"Dynamic to static friction ratio bounds must satisfy 0 <= lower <= upper <= 1, got {ratio_bounds.tolist()}.")

        self._reset_population()
        print("OT optimizer initialized.")
        print("Current iteration: ", self.iteration_counter)

    def tell(self, sim_dof_pos, real_dof_pos):
        self.scores += torch.sum(torch.square(sim_dof_pos - real_dof_pos - self.sim_params[:, self.bias_idx]), dim=1)
        self.sim_dof_pos_buffer[:, self.scores_counter, :] = sim_dof_pos
        self.scores_counter += 1

    def evolve(self):
        self.scores /= self.scores_counter
        self.scores_buffer[self.iteration_counter, :] = self.scores
        if self.save_optimization_process:
            self.sim_params_buffer[self.iteration_counter, :, :] = self.sim_params

        # diverged simulations are dropped from the update
        losses = torch.nan_to_num(self.scores, nan=math.inf)
        min_loss, min_index = torch.min(losses, dim=0)
        if min_loss.item() < self.best_loss:
            self.best_loss = min_loss.item()
            self.best_params = self.params[min_index].clone()
            self.best_trajectory = self.sim_dof_pos_buffer[min_index].detach().cpu().clone()
        loss_limit = self._loss_limit()

        self._update_plausible_set(losses, loss_limit)
        stats = self._update_particles(losses, loss_limit)

        if self.save_interval > 0 and self.iteration_counter % self.save_interval == 0:
            self.save_checkpoint(self.iteration_counter)
        self._print_iteration(losses, loss_limit, stats)

        self.sigma = max(self.cfg.sigma_min, self.sigma * self.cfg.sigma_decay)
        self._reset_population()

        self.scores = torch.zeros_like(self.scores)
        self.scores_counter = 0
        self.iteration_counter += 1
        print("OT optimizer iteration: ", self.iteration_counter)

    def finished(self):
        finished = self.max_iteration <= self.iteration_counter
        if finished:
            print("OT optimization finished.")
            self.save_checkpoint(self.iteration_counter - 1, finished=True)
            self._print_plausible_set()
            print("Saved to: ", os.path.join(self.writer.log_dir, "plausible_params.pt"))
        return finished

    def _loss_limit(self):
        if self.cfg.loss_limit is not None:
            return self.cfg.loss_limit
        return (1.0 + self.cfg.loss_tolerance) * self.best_loss

    def _loss_scale(self, loss_limit):
        # loss distance between the best and the limit, so that the excess is 0 at the best and 1 at the limit
        if loss_limit > self.best_loss:
            return loss_limit - self.best_loss
        return max(self.cfg.loss_tolerance * self.best_loss, torch.finfo(torch.float32).tiny)

    def _reset_population(self):
        # batch layout: [particles | local proposals around each particle | global uniform proposals]
        num_particles, num_params = self.particles.shape
        num_proposals = self.batch_size - num_particles
        num_global = int(round(self.cfg.global_ratio * num_proposals))
        num_local = num_proposals - num_global
        owners = torch.arange(num_local, device=self.device) % num_particles
        noise = torch.randn((num_local, num_params), generator=self.generator, device=self.device) @ self.covariance_chol.T
        local = (self.particles[owners] + self.sigma * noise * self.free_params).clamp(-1.0, 1.0)
        self.params = torch.cat([self.particles, local, self._sample_uniform(num_global)], dim=0)
        self.sim_params = self._params_to_sim_params(self.params)

    def _sample_uniform(self, num):
        samples = 2.0 * torch.rand((num, self.bounds.shape[0]), generator=self.generator, device=self.device) - 1.0
        return samples * self.free_params

    def _update_particles(self, losses, loss_limit):
        valid = torch.isfinite(losses)
        pool = self.params[valid]
        loss_scale = self._loss_scale(loss_limit)

        # target marginal: Gibbs weights on the normalized loss excess, steeper beyond the loss limit
        excess = (losses[valid] - self.best_loss) / loss_scale
        energy = excess + self._LIMIT_PENALTY * torch.relu(excess - 1.0)
        beta = self._adapt_temperature(energy)
        log_p = torch.log_softmax(-beta * energy, dim=0)

        # coordinate-wise objective of F: diversity weight expressed in units of the normalized excess
        lambda_excess = self.cfg.diversity_weight * self.best_loss / loss_scale
        cost = 0.5 * torch.cdist(self.particles, pool).square()
        epsilon = max(self.cfg.ot_epsilon * cost.median().item(), 1e-12)
        log_kernel = -cost / epsilon + beta * lambda_excess * self._replacement_log_det_gain(pool)

        weights = torch.softmax(self._sinkhorn(log_kernel, log_p), dim=1)
        barycenters = weights @ pool
        self._update_covariance(weights, pool)
        eta = self.cfg.step_size
        self.particles = (1.0 - eta) * self.particles + eta * barycenters

        return {"beta": beta, "epsilon": epsilon}

    def _update_covariance(self, weights, pool):
        """Rank-mu style update: average coupling-weighted second moment of the displacements y_j - theta_i."""
        if self.cfg.covariance_learning_rate <= 0.0:
            return
        num_params = pool.shape[1]
        second_moment = torch.zeros((num_params, num_params), dtype=torch.float64, device=self.device)
        for particle, row in zip(self.particles, weights):
            displacement = (pool - particle).double()
            second_moment += (displacement * row.double()[:, None]).T @ displacement
        trace = torch.trace(second_moment)
        if trace <= 1e-12:  # all particles kept their position
            return
        lr = self.cfg.covariance_learning_rate
        num_free = int(self.free_params.sum())
        covariance = (1.0 - lr) * self.covariance.double() + lr * num_free * second_moment / trace
        covariance = num_free * covariance / torch.trace(covariance)
        jitter = 1e-8 * torch.eye(num_params, dtype=torch.float64, device=self.device)
        self.covariance = covariance.float()
        self.covariance_chol = torch.linalg.cholesky(covariance + jitter).float()

    def _adapt_temperature(self, energy):
        """Largest beta <= beta_max whose target marginal keeps an effective sample size of ess_fraction."""
        target_ess = self.cfg.ess_fraction * energy.numel()

        def ess(beta):
            weights = torch.softmax(-beta * energy, dim=0)
            return 1.0 / weights.square().sum().item()

        if ess(self.cfg.beta_max) >= target_ess:
            return self.cfg.beta_max
        low, high = 0.0, self.cfg.beta_max
        for _ in range(50):
            mid = 0.5 * (low + high)
            if ess(mid) >= target_ess:
                low = mid
            else:
                high = mid
        return low

    def _sinkhorn(self, log_kernel, log_p):
        """Log-domain Sinkhorn with uniform particle marginal. Returns the unnormalized log-coupling per row.

        The target marginal is KL-relaxed (unbalanced OT): its scaling is raised to marginal_relaxation = kappa / (kappa + eps).
        """
        num_particles = log_kernel.shape[0]
        log_q = torch.full((num_particles,), -math.log(num_particles), device=log_kernel.device)
        f = torch.zeros(num_particles, device=log_kernel.device)
        g = torch.zeros_like(log_p)
        for _ in range(self.cfg.sinkhorn_iterations):
            f = log_q - torch.logsumexp(log_kernel + g[None, :], dim=1)
            g = self.cfg.marginal_relaxation * (log_p - torch.logsumexp(log_kernel + f[:, None], dim=0))
            row_mass = torch.logsumexp(log_kernel + f[:, None] + g[None, :], dim=1).exp()
            if (row_mass - log_q.exp()).abs().sum().item() < self._SINKHORN_TOL:
                break
        return log_kernel + g[None, :]

    def _diversity_kernel(self, a, b):
        # RBF on the RMS distance over the free params, measured as a fraction of the bounds range ([-1, 1] -> [0, 1])
        a, b = a[:, self.free_params], b[:, self.free_params]
        sq_dist = torch.cdist(a, b).square() / (4.0 * a.shape[1])
        return torch.exp(-0.5 * sq_dist / self.cfg.diversity_bandwidth ** 2)

    def _replacement_log_det_gain(self, pool):
        """log s_i(y_j): log-det gain of putting y_j in place of particle i (posterior variance of y_j given the others)."""
        delta = self._DPP_JITTER
        particles = self.particles.double()
        gram = self._diversity_kernel(particles, particles) + delta * torch.eye(particles.shape[0], dtype=torch.float64, device=self.device)
        inv_gram = torch.linalg.inv(gram)
        inv_diag = torch.diagonal(inv_gram)
        k = self._diversity_kernel(pool.double(), particles)
        inv_k = k @ inv_gram
        # quadratic form with particle i removed, via the block inverse (A_-i)^-1 = A^-1_-i,-i - a a^T / A^-1_ii
        quad_without_i = (inv_k * k).sum(dim=1, keepdim=True) - 2.0 * k * inv_k + k.square() * inv_diag
        quad = quad_without_i - (inv_k - k * inv_diag).square() / inv_diag
        schur = (1.0 + delta - quad).clamp_min(delta)
        return schur.log().T.float()

    def _update_plausible_set(self, losses, loss_limit):
        valid = torch.isfinite(losses)
        candidates = torch.cat([self.plausible_params, self.params[valid]], dim=0)
        candidate_losses = torch.cat([self.plausible_losses, losses[valid]], dim=0)
        feasible = candidate_losses <= loss_limit
        candidates, candidate_losses = candidates[feasible], candidate_losses[feasible]
        selected = self._greedy_select(candidates, candidate_losses / max(self.best_loss, torch.finfo(torch.float32).tiny))
        order = torch.argsort(candidate_losses[selected])
        self.plausible_params = candidates[selected][order]
        self.plausible_losses = candidate_losses[selected][order]

    def _greedy_select(self, candidates, relative_losses):
        """Greedy minimization of F over subsets of the candidates (incremental Cholesky of the DPP kernel)."""
        num_candidates = candidates.shape[0]
        num_selected = min(self.num_particles, num_candidates)
        delta = self._DPP_JITTER
        x = candidates.double()
        chol = torch.zeros((num_candidates, num_selected), dtype=torch.float64, device=self.device)
        schur = torch.full((num_candidates,), 1.0 + delta, dtype=torch.float64, device=self.device)
        available = torch.ones(num_candidates, dtype=torch.bool, device=self.device)
        selected = []
        for t in range(num_selected):
            score = relative_losses.double() - self.cfg.diversity_weight * schur.clamp_min(delta).log()
            score[~available] = math.inf
            j = int(torch.argmin(score))
            selected.append(j)
            available[j] = False
            e = (self._diversity_kernel(x, x[j:j + 1]).squeeze(1) - chol[:, :t] @ chol[j, :t]) / schur[j].sqrt()
            chol[:, t] = e
            schur = schur - e.square()
        return torch.tensor(selected, dtype=torch.long, device=self.device)

    def _plausible_diversity(self):
        if self.plausible_params.shape[0] == 0:
            return 0.0
        x = self.plausible_params.double()
        gram = self._diversity_kernel(x, x) + self._DPP_JITTER * torch.eye(x.shape[0], dtype=torch.float64, device=self.device)
        return torch.linalg.slogdet(gram)[1].item() / x.shape[0]

    def _print_iteration(self, losses, loss_limit, stats):
        particle_losses = losses[:self.num_particles]
        print("Min score: ", torch.min(losses).item(), " best so far: ", self.best_loss, " loss limit: ", loss_limit)
        print(f"Plausible set: {self.plausible_params.shape[0]} / {self.num_particles} | "
              f"particles within the limit: {int((particle_losses <= loss_limit).sum())} / {self.num_particles} | "
              f"diversity: {self._plausible_diversity():.3f}")
        print(f"beta: {stats['beta']:.3g} | sigma: {self.sigma:.3g} | OT epsilon: {stats['epsilon']:.3g}")
        print(f"Elapsed time: {(datetime.now() - self._timer_start).total_seconds():.1f} seconds")
        self._timer_start = datetime.now()
        self._log(losses, loss_limit, stats)

    def _params_to_sim_params(self, params):
        sim_params = (params + 1.0) / 2.0  # change range from 0 to 1
        sim_params = self.bounds[:, 0] + sim_params * (self.bounds[:, 1] - self.bounds[:, 0])  # range from lower to upper bound
        return sim_params

    def dynamic_friction(self, sim_params=None):
        """Dynamic Coulomb friction per env and joint, equal to the static friction unless identified separately."""
        sim_params = self.sim_params if sim_params is None else sim_params
        if self.separate_dynamic_friction:
            return sim_params[:, self.friction_idx] * sim_params[:, self.friction_ratio_idx]
        return sim_params[:, self.friction_idx]

    def get_best_sim_params(self):
        return self._params_to_sim_params(self.best_params)

    def get_plausible_sim_params(self):
        return self._params_to_sim_params(self.plausible_params)

    def _log(self, losses, loss_limit, stats):
        particle_losses = losses[:self.num_particles]
        self.writer.add_scalar("0_Episode/score", torch.min(losses).item(), self.iteration_counter)
        self.writer.add_scalar("0_Episode/best_score", self.best_loss, self.iteration_counter)
        self.writer.add_scalar("0_Episode/particles_mean_score", particle_losses[torch.isfinite(particle_losses)].mean().item(), self.iteration_counter)
        self.writer.add_scalar("5_OT/loss_limit", loss_limit, self.iteration_counter)
        self.writer.add_scalar("5_OT/plausible_set_size", self.plausible_params.shape[0], self.iteration_counter)
        self.writer.add_scalar("5_OT/plausible_diversity", self._plausible_diversity(), self.iteration_counter)
        self.writer.add_scalar("5_OT/beta", stats["beta"], self.iteration_counter)
        self.writer.add_scalar("5_OT/epsilon", stats["epsilon"], self.iteration_counter)
        self.writer.add_scalar("5_OT/sigma", self.sigma, self.iteration_counter)
        if self.plausible_params.shape[0] == 0:
            return
        plausible = self.get_plausible_sim_params()
        for i in range(len(self.joint_order)):
            self.writer.add_histogram("4_Bias/plausible_" + self.joint_order[i], plausible[:, self.bias_idx][:, i], self.iteration_counter)
            self.writer.add_histogram("3_Static_Dynamic_Friction/plausible_" + self.joint_order[i], plausible[:, self.friction_idx][:, i], self.iteration_counter)
            self.writer.add_histogram("2_Viscous_Friction/plausible_" + self.joint_order[i], plausible[:, self.damping_idx][:, i], self.iteration_counter)
            self.writer.add_histogram("1_Armature/plausible_" + self.joint_order[i], plausible[:, self.armature_idx][:, i], self.iteration_counter)
            if self.separate_dynamic_friction:
                self.writer.add_histogram("6_Dynamic_Friction/plausible_" + self.joint_order[i], self.dynamic_friction(plausible)[:, i], self.iteration_counter)
        self.writer.add_histogram("0_Delay/plausible", plausible[:, self.delay_idx], self.iteration_counter)

    def _plausible_dict(self):
        sim_params = self.get_plausible_sim_params().cpu()
        return {"sim_params": sim_params,
                "loss": self.plausible_losses.cpu(),
                "best_loss": self.best_loss,
                "loss_limit": self._loss_limit(),
                "joint_order": self.joint_order,
                "armature": sim_params[:, self.armature_idx],
                "viscous_friction": sim_params[:, self.damping_idx],
                "friction": sim_params[:, self.friction_idx],
                "dynamic_friction": self.dynamic_friction(sim_params),
                "bias": sim_params[:, self.bias_idx],
                "delay": sim_params[:, self.delay_idx],
                }

    def _print_plausible_set(self):
        plausible = self._plausible_dict()
        num = plausible["sim_params"].shape[0]
        print(f"Plausible combinations: {num} with loss <= {plausible['loss_limit']:.6g} (best {self.best_loss:.6g})")
        if num == 0:
            return
        print("Values of the best combination, [min, max] over the plausible set:")
        names = ["armature", "viscous_friction", "friction", "bias"]
        if self.separate_dynamic_friction:
            names.insert(3, "dynamic_friction")
        print(f"{'joint':<20}" + "".join(f"{name:<36}" for name in names))
        for i, joint in enumerate(self.joint_order):
            row = f"{joint:<20}"
            for name in names:
                values = plausible[name][:, i]
                row += f"{values[0].item():<10.4g} [{values.min().item():<10.4g}, {values.max().item():<10.4g}]  "
            print(row)
        delay = plausible["delay"]
        print(f"{'delay':<20}{delay[0].item():<10.4g} [{delay.min().item():<10.4g}, {delay.max().item():<10.4g}]")

    def save_checkpoint(self, iteration, finished=False):
        if self.best_trajectory is not None:
            torch.save(self.best_trajectory, os.path.join(self.writer.log_dir, "best_trajectory.pt"))
        # best combination saved as mean_*.pt, so that plot_trajectory.py works as with CMA-ES
        torch.save(self.get_best_sim_params(), os.path.join(self.writer.log_dir, "mean_" + f"{iteration:03}" + ".pt"))
        torch.save(self._plausible_dict(), os.path.join(self.writer.log_dir, "plausible_params.pt"))
        if finished and self.save_optimization_process:
            torch.save({"params_buffer": self.sim_params_buffer,
                        "scores_buffer": self.scores_buffer, },
                       os.path.join(self.writer.log_dir, "progress.pt"))

    def close(self):
        self.writer.close()
