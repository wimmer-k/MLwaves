import numpy as np
from MWD_utils import MWD, fit_energy_spectrum
import optuna

def objective_decay_flatness(trial, waveforms, params):
    decay_time = trial.suggest_float("decay_time", 5000.0, 40000.0)
    params = params.copy()
    params["decay_time"] = decay_time

    flatnesses = []
    np.random.seed(0)
    idx = np.random.choice(len(waveforms), size=min(500, len(waveforms)), replace=False)
    for i in idx:
        _, _, flat = MWD(waveforms[i], params)
        flatnesses.append(flat)
    mean_flatness = np.nanmean(flatnesses)
    return mean_flatness

def objective_window_resolution(trial, waveforms, params):
    # Extract ranges from params
    r_L = params.get("range_smoothing_L", [200.0, 1000.0])
    r_start = params.get("range_MWD_amp_start", [4000.0, 4500.0])
    r_length = params.get("range_MWD_amp_length", [100.0, 500.0])
    amp_limit = params.get("MWD_amp_limit", 5000.0)

    # Suggest parameters
    smoothing_L = trial.suggest_float("smoothing_L", r_L[0], r_L[1])
    amp_start = trial.suggest_float("MWD_amp_start", r_start[0], r_start[1])

    max_length = min(amp_limit - amp_start, r_length[1])
    if max_length < r_length[0]:
        raise optuna.exceptions.TrialPruned()

    amp_length = trial.suggest_float("MWD_amp_length", r_length[0], max_length)
    amp_stop = amp_start + amp_length

    # Copy and inject values
    params = params.copy()
    params["smoothing_L"] = smoothing_L
    params["MWD_amp_start"] = amp_start
    params["MWD_amp_stop"] = amp_stop

    # Sample and calculate energies
    np.random.seed(0)
    idx = np.random.choice(len(waveforms), size=min(500, len(waveforms)), replace=False)
    energies = []
    for i in idx:
        _, e, _ = MWD(waveforms[i], params)
        if not np.isnan(e):
            energies.append(e)

    if len(energies) < 50:
        if trial.number % 10 == 0:
            print(f"[Trial {trial.number}] Too few valid energies: {len(energies)}")
        return 1e6

    x, y, mean, sigma, res = fit_energy_spectrum(energies)
    if res is None or sigma <= 0 or res > 100:
        if trial.number % 10 == 0:
            print(f"[Trial {trial.number}] Bad fit - res: {res}, sigma: {sigma}")
        return 1e6

    # Store for dashboard
    trial.set_user_attr("res", res)
    trial.set_user_attr("smoothing_L", smoothing_L)
    trial.set_user_attr("amp_start", amp_start)
    trial.set_user_attr("amp_stop", amp_stop)
    trial.set_user_attr("amp_length", amp_length)

    return res

def objective_decay_resolution(trial, waveforms, params):
    decay_time = trial.suggest_float("decay_time", 1000.0, 5000.0)
    params = params.copy()
    params["decay_time"] = decay_time

    np.random.seed(0)
    idx = np.random.choice(len(waveforms), size=min(500, len(waveforms)), replace=False)
    energies = []
    for i in idx:
        _, e, _ = MWD(waveforms[i], params)
        if not np.isnan(e):
            energies.append(e)

    if len(energies) < 50:
        print(f"[Trial {trial.number}] Too few valid energies: {len(energies)}")
        return 1e6

    x, y, mean, sigma, res = fit_energy_spectrum(energies)
    if res is None or sigma <= 0 or res > 100:
        print(f"[Trial {trial.number}] Bad fit - res: {res}, sigma: {sigma}")
        return 1e6

    # Store for dashboard
    trial.set_user_attr("res", res)
    trial.set_user_attr("decay_time", decay_time)

    return res
