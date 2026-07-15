from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from ase import units
from ase.io import write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

from orchestr_ai.postprocessing.surfaces import MultiStateSurfaceEvaluator
from orchestr_ai.postprocessing.simulation import (
    _load_md_restart_frame,
    _set_md_initial_velocities_from_atoms,
)


HBAR_EV_FS = 0.6582119514


def _temperature_from_kinetic(atoms):
    if len(atoms) == 0:
        return 0.0
    return atoms.get_kinetic_energy() / (1.5 * units.kB * len(atoms))


def _write_namd_frame(path, atoms, step, time_fs, surface, active_state, nac, hop_info):
    if not path:
        return
    frame = atoms.copy()
    velocities = atoms.get_velocities()
    if velocities is not None:
        frame.set_array("velocities", np.asarray(velocities, dtype=float))
    active_forces = surface.forces_s1 if active_state == 1 else surface.forces_s0
    frame.set_array("forces", np.asarray(active_forces, dtype=float))
    frame.info.update(
        {
            "step": int(step),
            "time_fs": float(time_fs),
            "active_state": int(active_state),
            "E_s0_eV": float(surface.energy_s0),
            "E_s1_eV": float(surface.energy_s1),
            "gap_eV": float(surface.gap),
            "nac_norm_A_inv": float(np.linalg.norm(nac.reshape(-1))),
            "hop": str(hop_info.get("event", "none")),
        }
    )
    with open(path, "a") as handle:
        write(handle, frame, format="extxyz")


def _append_namd_log(path, row, *, print_to_screen=False):
    header = (
        "Step | Time(fs) | State | E_S0(eV) | E_S1(eV) | Gap(eV) | "
        "Ekin(eV) | Etot_active(eV) | T(K) | Pop_S0 | Pop_S1 | "
        "NAC_norm(1/A) | vdotd(1/fs) | HopProb | Random | Hop"
    )
    line = (
        f"{row['step']:6d} | {row['time_fs']:8.2f} | {row['state']:5d} | "
        f"{row['e0']:9.6f} | {row['e1']:9.6f} | {row['gap']:8.6f} | "
        f"{row['ekin']:9.6f} | {row['etot']:15.6f} | {row['temp']:7.2f} | "
        f"{row['pop0']:7.5f} | {row['pop1']:7.5f} | "
        f"{row['nac_norm']:13.6e} | {row['vdotd']:12.6e} | "
        f"{row['hop_prob']:8.5f} | {row['random']:6.4f} | {row['hop']}"
    )
    if path:
        fresh = not Path(path).exists() or Path(path).stat().st_size == 0
        with open(path, "a") as handle:
            if fresh:
                handle.write(header + "\n")
            handle.write(line + "\n")
    if print_to_screen or not path:
        print(header)
        print(line)


def _append_tunneling_log(path, row, *, print_to_screen=False):
    header = (
        "Step | Time(fs) | StateBefore | StateAfter | GapMin(eV) | "
        "dGapDt(eV/fs) | Praw | Pused | Random | Accepted | Event"
    )
    line = (
        f"{row['step']:6d} | {row['time_fs']:10.4f} | "
        f"{row['state_before']:11d} | {row['state_after']:10d} | "
        f"{row['gap_min']:10.6f} | {row['dgap_dt']:14.6e} | "
        f"{row['prob_raw']:9.6e} | {row['prob_used']:9.6e} | "
        f"{row['random']:8.6f} | {str(row['accepted']):>8s} | {row['event']}"
    )
    if path:
        fresh = not Path(path).exists() or Path(path).stat().st_size == 0
        with open(path, "a") as handle:
            if fresh:
                handle.write(header + "\n")
            handle.write(line + "\n")
    if print_to_screen or not path:
        print(header)
        print(line)


def _append_golden_rule_log(path, row, *, print_to_screen=False):
    header = (
        "Step | Time(fs) | StateBefore | StateAfter | Gap(eV) | "
        "vdotd(1/fs) | Phase(rad) | AmpAbs2 | Pinc | Pused | Random | Accepted | Event"
    )
    line = (
        f"{row['step']:6d} | {row['time_fs']:10.4f} | "
        f"{row['state_before']:11d} | {row['state_after']:10d} | "
        f"{row['gap']:9.6f} | {row['vdotd']:12.6e} | "
        f"{row['phase']:11.6f} | {row['amp_abs2']:9.6e} | "
        f"{row['prob_increment']:9.6e} | {row['prob_used']:9.6e} | "
        f"{row['random']:8.6f} | {str(row['accepted']):>8s} | {row['event']}"
    )
    if path:
        fresh = not Path(path).exists() or Path(path).stat().st_size == 0
        with open(path, "a") as handle:
            if fresh:
                handle.write(header + "\n")
            handle.write(line + "\n")
    if print_to_screen or not path:
        print(header)
        print(line)


def _lzs_gap_minimum_probability(gap_min, dgap_dt, tunneling_cfg):
    min_rate = float(tunneling_cfg.get("min_gap_rate_eV_per_fs", 1.0e-10))
    scale = float(tunneling_cfg.get("probability_scale", 1.0))
    max_probability = float(tunneling_cfg.get("max_probability", 1.0))
    rate = max(abs(float(dgap_dt)), min_rate)
    exponent = -np.pi * float(gap_min) ** 2 / (2.0 * HBAR_EV_FS * rate)
    prob_raw = float(np.exp(np.clip(exponent, -745.0, 0.0)))
    prob_used = min(max_probability, max(0.0, scale * prob_raw))
    return prob_raw, prob_used


def _is_gap_local_minimum(gap_history):
    if len(gap_history) < 3:
        return False
    before, middle, after = gap_history[-3:]
    return middle["gap"] < before["gap"] and middle["gap"] <= after["gap"]


class BeckAhnNACApproximator:
    """Gap-gradient NAC approximation using only state force derivatives."""

    def __init__(self, config):
        nac_cfg = config.get("namd", {}).get("nac", {})
        self.gap_floor = float(nac_cfg.get("gap_floor_eV", 0.02))
        self.max_norm = float(nac_cfg.get("max_norm", 100.0))
        self.scale = float(nac_cfg.get("scale", 1.0))

    def compute(self, surface):
        grad_gap = -(np.asarray(surface.forces_s1) - np.asarray(surface.forces_s0))
        denom = max(abs(float(surface.gap)), self.gap_floor)
        nac = self.scale * grad_gap / denom
        norm = float(np.linalg.norm(nac.reshape(-1)))
        if self.max_norm > 0.0 and norm > self.max_norm:
            nac = nac * (self.max_norm / norm)
        return np.asarray(nac, dtype=np.float64)


class FSSHState:
    def __init__(self, config):
        namd = config.get("namd", {})
        fssh = namd.get("fssh", {})
        self.active_state = int(namd.get("initial_state", 1))
        self.allow_reverse = bool(fssh.get("allow_reverse_hops", False))
        self.electronic_substeps = int(fssh.get("electronic_substeps", 20))
        self.decoherence = str(fssh.get("decoherence", "energy_based")).strip().lower()
        self.decoherence_c = float(fssh.get("decoherence_C_eV", 0.1))
        self.decoherence_gap_floor = float(fssh.get("decoherence_gap_floor_eV", 1.0e-6))
        self.frustrated_hop_decoherence = str(
            fssh.get("frustrated_hop_decoherence", "collapse_target")
        ).strip().lower()
        self.rng = np.random.default_rng(fssh.get("random_seed", None))
        self.coeff = np.zeros(2, dtype=np.complex128)
        self.coeff[self.active_state] = 1.0 + 0.0j

    def _rhs(self, coeff, energies, vdotd):
        e0, e1 = energies
        dc = np.empty(2, dtype=np.complex128)
        dc[0] = -1j * e0 / HBAR_EV_FS * coeff[0] - vdotd * coeff[1]
        dc[1] = -1j * e1 / HBAR_EV_FS * coeff[1] + vdotd * coeff[0]
        return dc

    def propagate_electrons(self, surface, velocities, nac, dt_fs):
        pops_old = np.abs(self.coeff) ** 2
        vdotd = float(np.sum(np.asarray(velocities, dtype=np.float64) * nac))
        energies = (float(surface.energy_s0), float(surface.energy_s1))
        nsub = max(1, self.electronic_substeps)
        h = float(dt_fs) / nsub
        for _ in range(nsub):
            c = self.coeff
            k1 = self._rhs(c, energies, vdotd)
            k2 = self._rhs(c + 0.5 * h * k1, energies, vdotd)
            k3 = self._rhs(c + 0.5 * h * k2, energies, vdotd)
            k4 = self._rhs(c + h * k3, energies, vdotd)
            self.coeff = c + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            norm = np.linalg.norm(self.coeff)
            if norm > 0.0:
                self.coeff /= norm
        pops_new = np.abs(self.coeff) ** 2
        return pops_old, pops_new, vdotd

    def hop_probability(self, dt_fs, vdotd):
        active = self.active_state
        target = 1 - active
        if active == 0 and not self.allow_reverse:
            return 0.0, target

        coherence = np.conjugate(self.coeff[0]) * self.coeff[1]
        transfer_rate = 2.0 * float(np.real(coherence * vdotd))
        if target == 0:
            transfer_rate *= -1.0

        denom = max(float(abs(self.coeff[active]) ** 2), 1.0e-12)
        prob = max(0.0, transfer_rate * float(dt_fs) / denom)
        return min(1.0, prob), target

    def populations(self):
        pops = np.abs(self.coeff) ** 2
        return float(pops[0]), float(pops[1])

    def _collapse_to_state(self, state):
        phase = np.angle(self.coeff[state]) if abs(self.coeff[state]) > 0.0 else 0.0
        self.coeff[:] = 0.0
        self.coeff[state] = np.exp(1j * phase)

    def apply_decoherence(self, surface, kinetic_energy, dt_fs):
        if self.decoherence in {"", "none", "off", "false"}:
            return
        if self.decoherence not in {"energy_based", "granucci_persico", "gp"}:
            raise ValueError(f"Unsupported FSSH decoherence mode '{self.decoherence}'.")

        active = self.active_state
        inactive = 1 - active
        inactive_amp = self.coeff[inactive]
        if abs(inactive_amp) <= 0.0:
            self._collapse_to_state(active)
            return

        gap = max(abs(float(surface.gap)), self.decoherence_gap_floor)
        ekin = max(float(kinetic_energy), 1.0e-12)
        tau_fs = (HBAR_EV_FS / gap) * (1.0 + self.decoherence_c / ekin)
        damping = float(np.exp(-float(dt_fs) / max(tau_fs, 1.0e-12)))

        inactive_phase = np.angle(inactive_amp)
        active_phase = np.angle(self.coeff[active]) if abs(self.coeff[active]) > 0.0 else 0.0
        inactive_mag = min(abs(inactive_amp) * damping, 1.0)
        active_mag = np.sqrt(max(0.0, 1.0 - inactive_mag * inactive_mag))
        self.coeff[inactive] = inactive_mag * np.exp(1j * inactive_phase)
        self.coeff[active] = active_mag * np.exp(1j * active_phase)

    def apply_frustrated_hop_decoherence(self, target):
        mode = self.frustrated_hop_decoherence
        if mode in {"", "none", "off", "false"}:
            return
        if mode in {"collapse_target", "zero_target"}:
            self.coeff[target] = 0.0
            self._collapse_to_state(self.active_state)
            return
        if mode == "collapse_active":
            self._collapse_to_state(self.active_state)
            return
        raise ValueError(f"Unsupported frustrated-hop decoherence mode '{mode}'.")

    def draw(self):
        return float(self.rng.random())


def _attempt_velocity_rescale(atoms, surface, old_state, new_state, nac):
    energies = [surface.energy_s0, surface.energy_s1]
    delta_e = float(energies[new_state] - energies[old_state])
    masses = atoms.get_masses().reshape(-1, 1)
    momenta = atoms.get_momenta()
    direction = np.asarray(nac, dtype=np.float64)
    norm = float(np.linalg.norm(direction.reshape(-1)))
    if norm <= 0.0:
        return False, "frustrated_zero_nac"
    direction = direction / norm

    a = float(np.sum(direction * direction / (2.0 * masses)))
    b = float(np.sum(momenta * direction / masses))
    disc = b * b - 4.0 * a * delta_e
    if a <= 0.0 or disc < 0.0:
        return False, "frustrated_insufficient_ke"

    root = np.sqrt(disc)
    candidates = [(-b + root) / (2.0 * a), (-b - root) / (2.0 * a)]
    alpha = min(candidates, key=abs)
    atoms.set_momenta(momenta + alpha * direction)
    return True, "accepted"


def _active_forces(surface, state):
    return surface.forces_s1 if state == 1 else surface.forces_s0


def _active_energy(surface, state):
    return surface.energy_s1 if state == 1 else surface.energy_s0


def run_namd(atoms, model_obj, device, config, neighbor_list=None):
    """Run two-state NAMD/FSSH starting on S1 and allowing collapse to S0."""

    namd = config.get("namd", {})
    dt_fs = float(namd.get("timestep_fs", 0.25))
    dt = dt_fs * units.fs
    nsteps = int(namd.get("steps", 1000))
    temperature = float(namd.get("temperature_K", config.get("md", {}).get("temperature_K", 300.0)))
    log_interval = int(namd.get("log_interval", 1))
    xyz_interval = int(namd.get("xyz_print_interval", 10))
    log_file = namd.get("log_file", "namd.log")
    traj_file = namd.get("trajectory_file_namd", namd.get("trajectory_file", "namd.xyz"))
    stop_after_first_hop = bool(namd.get("stop_after_first_hop", False))
    collapse_to_state = int(namd.get("collapse_to_state", 0))
    remove_translation = bool(namd.get("remove_translation", False))
    remove_rotation = bool(namd.get("remove_rotation", False))
    remove_drift_interval = int(namd.get("remove_drift_interval", 100) or 0)
    restart = bool(namd.get("restart", False))
    restart_file = namd.get("restart_file", traj_file)
    initial_velocity_mode = namd.get("initial_velocities", "auto")
    print_log = bool(namd.get("print_log", False))
    tunneling_cfg = namd.get("tunneling", {})
    tunneling_enabled = bool(tunneling_cfg.get("enabled", False))
    tunneling_model = str(tunneling_cfg.get("model", "lzs_gap_minimum")).strip().lower()
    tunneling_source_state = int(tunneling_cfg.get("source_state", 1))
    tunneling_target_state = int(tunneling_cfg.get("target_state", 0))
    tunneling_min_interval = float(tunneling_cfg.get("min_attempt_interval_fs", 0.0))
    tunneling_rescale = bool(tunneling_cfg.get("rescale_momentum", True))
    tunneling_collapse = bool(tunneling_cfg.get("collapse_on_accept", True))
    tunneling_stop_after_accept = bool(tunneling_cfg.get("stop_after_accept", False))
    tunneling_log_file = tunneling_cfg.get("log_file", "namd_tunneling.log")
    tunneling_print_log = bool(tunneling_cfg.get("print_log", False))
    last_tunneling_attempt_time = -np.inf
    gap_history = []

    gr_cfg = namd.get("golden_rule_leakage", {})
    gr_enabled = bool(gr_cfg.get("enabled", False))
    gr_source_state = int(gr_cfg.get("source_state", 1))
    gr_target_state = int(gr_cfg.get("target_state", 0))
    gr_probability_scale = float(gr_cfg.get("probability_scale", 1.0))
    gr_max_probability = float(gr_cfg.get("max_probability_per_step", 0.01))
    gr_rescale = bool(gr_cfg.get("rescale_momentum", True))
    gr_collapse = bool(gr_cfg.get("collapse_on_accept", True))
    gr_stop_after_accept = bool(gr_cfg.get("stop_after_accept", False))
    gr_reset_on_reject = bool(gr_cfg.get("reset_amplitude_on_reject", False))
    gr_reset_on_accept = bool(gr_cfg.get("reset_amplitude_on_accept", True))
    gr_log_interval = int(gr_cfg.get("log_interval", log_interval) or 0)
    gr_log_file = gr_cfg.get("log_file", "namd_golden_rule_leakage.log")
    gr_print_log = bool(gr_cfg.get("print_log", False))
    gr_phase = 0.0
    gr_amplitude = 0.0 + 0.0j
    gr_probability_reference = 0.0

    if tunneling_enabled and tunneling_model != "lzs_gap_minimum":
        raise ValueError(
            f"Unsupported NAMD tunneling model '{tunneling_model}'. "
            "Supported model: 'lzs_gap_minimum'."
        )

    step_offset = 0
    if restart:
        atoms, step_offset = _load_md_restart_frame(restart_file, atoms)
        print(f"Restarting NAMD from {restart_file} at saved step {step_offset}.")
    elif not _set_md_initial_velocities_from_atoms(atoms, mode=initial_velocity_mode):
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    if remove_translation:
        Stationary(atoms, preserve_temperature=True)
    if remove_rotation:
        ZeroRotation(atoms, preserve_temperature=True)

    evaluator = MultiStateSurfaceEvaluator(model_obj, device, config, neighbor_list)
    nac_model = BeckAhnNACApproximator(config)
    fssh = FSSHState(config)
    if fssh.active_state != 1:
        print("Warning: NAMD is configured for trajectories starting on S1; initial_state is not 1.")

    surface = evaluator.evaluate(atoms)
    forces = _active_forces(surface, fssh.active_state)
    gap_history.append(
        {
            "step": step_offset,
            "time_fs": step_offset * dt_fs,
            "gap": float(surface.gap),
        }
    )
    start = time.time()
    print(f"Running NAMD/FSSH: {nsteps} steps, dt={dt_fs} fs, initial_state={fssh.active_state}")
    if tunneling_enabled:
        print(
            "NAMD LZS gap-minimum tunneling correction enabled: "
            f"S{tunneling_source_state}->S{tunneling_target_state}, "
            f"min_attempt_interval={tunneling_min_interval} fs, "
            f"log='{tunneling_log_file}'."
        )
    if gr_enabled:
        print(
            "NAMD golden-rule leakage correction enabled: "
            f"S{gr_source_state}->S{gr_target_state}, "
            f"max_probability_per_step={gr_max_probability}, "
            f"log='{gr_log_file}'."
        )

    for local_step in range(nsteps + 1):
        step = step_offset + local_step
        time_fs = step * dt_fs
        nac = nac_model.compute(surface)
        velocities = atoms.get_velocities()
        vdotd = float(np.sum(np.asarray(velocities, dtype=np.float64) * nac))
        hop_prob = 0.0
        target = 1 - fssh.active_state
        random_value = np.nan
        hop_info = {"event": "none"}

        if tunneling_enabled and _is_gap_local_minimum(gap_history):
            before, middle, after = gap_history[-3:]
            if (
                fssh.active_state == tunneling_source_state
                and time_fs - last_tunneling_attempt_time >= tunneling_min_interval
            ):
                dgap_dt = abs(after["gap"] - before["gap"]) / max(
                    after["time_fs"] - before["time_fs"],
                    1.0e-12,
                )
                prob_raw, prob_used = _lzs_gap_minimum_probability(
                    middle["gap"],
                    dgap_dt,
                    tunneling_cfg,
                )
                tunnel_random = fssh.draw()
                state_before = fssh.active_state
                state_after = fssh.active_state
                accepted = False
                event = "lzs_rejected_random"
                last_tunneling_attempt_time = time_fs

                if tunnel_random <= prob_used:
                    if tunneling_rescale:
                        rescale_ok, rescale_event = _attempt_velocity_rescale(
                            atoms,
                            surface,
                            fssh.active_state,
                            tunneling_target_state,
                            nac,
                        )
                    else:
                        rescale_ok, rescale_event = True, "accepted_no_rescale"

                    if rescale_ok:
                        fssh.active_state = tunneling_target_state
                        if tunneling_collapse:
                            fssh._collapse_to_state(tunneling_target_state)
                        forces = _active_forces(surface, fssh.active_state)
                        velocities = atoms.get_velocities()
                        vdotd = float(np.sum(np.asarray(velocities, dtype=np.float64) * nac))
                        state_after = fssh.active_state
                        accepted = True
                        event = "lzs_accepted"
                        hop_info["event"] = event
                    else:
                        event = f"lzs_{rescale_event}"

                _append_tunneling_log(
                    tunneling_log_file,
                    {
                        "step": int(middle["step"]),
                        "time_fs": float(middle["time_fs"]),
                        "state_before": state_before,
                        "state_after": state_after,
                        "gap_min": float(middle["gap"]),
                        "dgap_dt": float(dgap_dt),
                        "prob_raw": prob_raw,
                        "prob_used": prob_used,
                        "random": tunnel_random,
                        "accepted": accepted,
                        "event": event,
                    },
                    print_to_screen=tunneling_print_log,
                )

        if hop_info["event"] == "none":
            _, _, vdotd = fssh.propagate_electrons(surface, velocities, nac, dt_fs)
            hop_prob, target = fssh.hop_probability(dt_fs, vdotd)

        if (
            gr_enabled
            and hop_info["event"] == "none"
            and fssh.active_state == gr_source_state
        ):
            state_before = fssh.active_state
            state_after = fssh.active_state
            accepted = False
            event = "gr_rejected_random"
            gap = float(surface.gap)
            gr_phase += gap * dt_fs / HBAR_EV_FS
            gr_amplitude += -float(vdotd) * np.exp(1j * gr_phase) * dt_fs
            amp_abs2 = float(abs(gr_amplitude) ** 2)
            prob_increment = max(0.0, amp_abs2 - gr_probability_reference)
            gr_probability_reference = max(gr_probability_reference, amp_abs2)
            prob_used = min(gr_max_probability, max(0.0, gr_probability_scale * prob_increment))
            gr_random = fssh.draw()

            if gr_random <= prob_used:
                if gr_rescale:
                    rescale_ok, rescale_event = _attempt_velocity_rescale(
                        atoms,
                        surface,
                        fssh.active_state,
                        gr_target_state,
                        nac,
                    )
                else:
                    rescale_ok, rescale_event = True, "accepted_no_rescale"

                if rescale_ok:
                    fssh.active_state = gr_target_state
                    if gr_collapse:
                        fssh._collapse_to_state(gr_target_state)
                    forces = _active_forces(surface, fssh.active_state)
                    velocities = atoms.get_velocities()
                    vdotd = float(np.sum(np.asarray(velocities, dtype=np.float64) * nac))
                    state_after = fssh.active_state
                    accepted = True
                    event = "gr_accepted"
                    hop_info["event"] = event
                    if gr_reset_on_accept:
                        gr_phase = 0.0
                        gr_amplitude = 0.0 + 0.0j
                        gr_probability_reference = 0.0
                else:
                    event = f"gr_{rescale_event}"

            if not accepted and gr_reset_on_reject:
                gr_phase = 0.0
                gr_amplitude = 0.0 + 0.0j
                gr_probability_reference = 0.0

            if gr_log_interval > 0 and local_step % gr_log_interval == 0:
                _append_golden_rule_log(
                    gr_log_file,
                    {
                        "step": step,
                        "time_fs": time_fs,
                        "state_before": state_before,
                        "state_after": state_after,
                        "gap": gap,
                        "vdotd": float(vdotd),
                        "phase": float(gr_phase),
                        "amp_abs2": amp_abs2,
                        "prob_increment": prob_increment,
                        "prob_used": prob_used,
                        "random": gr_random,
                        "accepted": accepted,
                        "event": event,
                    },
                    print_to_screen=gr_print_log,
                )

        if hop_info["event"] == "none" and hop_prob > 0.0:
            random_value = fssh.draw()
            if random_value < hop_prob:
                accepted, event = _attempt_velocity_rescale(
                    atoms,
                    surface,
                    fssh.active_state,
                    target,
                    nac,
                )
                hop_info["event"] = event
                if accepted:
                    fssh.active_state = target
                    fssh.coeff[:] = 0.0
                    fssh.coeff[target] = 1.0 + 0.0j
                    forces = _active_forces(surface, fssh.active_state)
                else:
                    fssh.apply_frustrated_hop_decoherence(target)
            else:
                hop_info["event"] = "rejected_random"

        ekin = atoms.get_kinetic_energy()
        fssh.apply_decoherence(surface, ekin, dt_fs)
        pop0, pop1 = fssh.populations()
        active_energy = _active_energy(surface, fssh.active_state)
        row = {
            "step": step,
            "time_fs": time_fs,
            "state": fssh.active_state,
            "e0": surface.energy_s0,
            "e1": surface.energy_s1,
            "gap": surface.gap,
            "ekin": ekin,
            "etot": active_energy + ekin,
            "temp": _temperature_from_kinetic(atoms),
            "pop0": pop0,
            "pop1": pop1,
            "nac_norm": float(np.linalg.norm(nac.reshape(-1))),
            "vdotd": vdotd,
            "hop_prob": hop_prob,
            "random": random_value,
            "hop": hop_info["event"],
        }
        if log_interval > 0 and local_step % log_interval == 0:
            _append_namd_log(log_file, row, print_to_screen=print_log)
        if xyz_interval > 0 and local_step % xyz_interval == 0:
            _write_namd_frame(traj_file, atoms, step, time_fs, surface, fssh.active_state, nac, hop_info)

        if hop_info["event"] == "accepted" and fssh.active_state == collapse_to_state:
            print(f"NAMD hop/collapse accepted at step {step}: now on S{fssh.active_state}.")
            if stop_after_first_hop:
                break
        if hop_info["event"] == "lzs_accepted" and fssh.active_state == collapse_to_state:
            print(f"NAMD LZS tunneling accepted near step {step}: now on S{fssh.active_state}.")
            if tunneling_stop_after_accept or stop_after_first_hop:
                break
        if hop_info["event"] == "gr_accepted" and fssh.active_state == collapse_to_state:
            print(f"NAMD golden-rule leakage accepted at step {step}: now on S{fssh.active_state}.")
            if gr_stop_after_accept or stop_after_first_hop:
                break

        if local_step == nsteps:
            break

        masses = atoms.get_masses().reshape(-1, 1)
        momenta = atoms.get_momenta()
        momenta = momenta + 0.5 * dt * forces
        atoms.set_positions(atoms.get_positions() + dt * momenta / masses)
        atoms.set_momenta(momenta)

        surface = evaluator.evaluate(atoms)
        forces = _active_forces(surface, fssh.active_state)
        atoms.set_momenta(atoms.get_momenta() + 0.5 * dt * forces)
        gap_history.append(
            {
                "step": step + 1,
                "time_fs": time_fs + dt_fs,
                "gap": float(surface.gap),
            }
        )
        if len(gap_history) > 3:
            gap_history = gap_history[-3:]

        if gr_enabled and fssh.active_state != gr_source_state:
            gr_phase = 0.0
            gr_amplitude = 0.0 + 0.0j
            gr_probability_reference = 0.0

        if remove_drift_interval > 0 and local_step > 0 and local_step % remove_drift_interval == 0:
            if remove_translation:
                Stationary(atoms, preserve_temperature=True)
            if remove_rotation:
                ZeroRotation(atoms, preserve_temperature=True)

    elapsed = time.time() - start
    print(f"NAMD finished in {elapsed:.2f} s. Final active state: S{fssh.active_state}.")
