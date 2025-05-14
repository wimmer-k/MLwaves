import argparse
import yaml
import numpy as np
import pandas as pd
import optuna
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from MWD_utils import MWD, batch_estimate_taus, plot_energy_comparison_panel, fit_energy_spectrum
from optimization_objectives import (
    objective_decay_flatness,
    objective_window_resolution,
    objective_decay_resolution,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(input_file):
    df = pd.read_csv(input_file)
    waveforms = df[[c for c in df.columns if c.startswith("t")]].values
    return df, waveforms

def run_optimization(name, waveforms, params, objective_func, n_trials=50, verbose=0, enqueue=None, reuse_study=False):
    storage_dir = "optuna_studies"
    os.makedirs(storage_dir, exist_ok=True)
    storage_url = f"sqlite:///{storage_dir}/{name}_study.db"
    if not reuse_study:
        # remove old study db if it exists
        db_path = f"{name}_study.db"
        if os.path.exists(db_path):
            os.remove(db_path)
    study = optuna.create_study(
        study_name=name,
        direction="minimize",
        storage=storage_url,
        load_if_exists=reuse_study
    )
    
    if enqueue:
        study.enqueue_trial(enqueue)

    if verbose > 0:
        pbar = tqdm(total=n_trials, desc=f"{name} optimization", leave=False)
    def update_progress(study, trial):
        if verbose > 0 :
            pbar.update(1)
        if verbose > 1 and trial.value is not None and trial.value < 1e5:
            print(f"[Trial {trial.number}] value: {trial.value:.4f}, params: {trial.params}")

    # Run the optimization
    study.optimize(
        lambda trial: objective_func(trial, waveforms, params),
        n_trials=n_trials,
        callbacks=[update_progress]
    )

    return study.best_params, study

def plot_and_save(fig, filename, show):
    os.makedirs("output", exist_ok=True)
    fig.tight_layout()
    fig.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close(fig)
    #print(f"Saved plot to {filename}")

def main(config_path, show_plots=False, stop_after_tau=False, stop_after_window=False, reuse_study=False):
    np.random.seed(0)
    cfg = load_config(config_path)
    df, waveforms = load_data(cfg["input_file"])

    layer, x, y = cfg["layer"], cfg["x"], cfg["y"]
    settings = cfg.get("settings", {})
    ranges = cfg.get("optimization_ranges", {})
    verbose = settings.get("verbose", 1)

    params = cfg["initial_params"].copy()
    for key, val in ranges.items():
        if key == "amp_limit":
            params["MWD_amp_limit"] = val
        elif key == "amp_start":
            params["range_MWD_amp_start"] = val
        elif key == "amp_length":
            params["range_MWD_amp_length"] = val
        else:
            params[f"range_{key}"] = val

    tau_sample_size = settings.get("tau_sample_size", 500)
    n_trials_window = settings.get("n_trials_window", 50)
    n_trials_decay = settings.get("n_trials_decay", 50)

    if verbose > 0:
        print("\nStage 1: Estimate decay_time (tau) from waveform tails")
    #print(tau_sample_size,len(waveforms))
    taus = batch_estimate_taus(
        waveforms, sample_size=tau_sample_size, sampling=params["sampling"], plot=show_plots, layer=layer, x=x, y=y
    )

    taus = np.array(taus)
    taus = taus[(taus > 0) & (taus < 1e6)]

    if len(taus) == 0:
        print("Warning: No valid tau estimates found, using config value.")
        median_tau = params["decay_time"]
    else:
        median_tau = np.median(taus)
        if verbose > 0:
            print(f"Estimated median decay_time (tau) = {median_tau:.1f} ns")

    params_stage1 = params.copy()
    params_stage1["decay_time"] = median_tau

    np.random.seed(0)
    sample_indices = np.random.choice(len(waveforms), size=min(500, len(waveforms)), replace=False)
    orig_energy = df["energy"].values[sample_indices]
    init_energy = df["MWD"].values[sample_indices]
    best_energy_stage1 = []
    for i in sample_indices:
        _, energy, _ = MWD(waveforms[i], params_stage1)
        if not np.isnan(e_init):
            init_energy.append(e_init)
        if not np.isnan(energy):
            best_energy_stage1.append(energy)

    _, _, _, _, res_orig = fit_energy_spectrum(orig_energy)
    _, _, _, _, res_init = fit_energy_spectrum(init_energy)
    _, _, _, _, res_stage1 = fit_energy_spectrum(best_energy_stage1)
    if verbose > 0:
        print(f"[FEBEX] Original resolution: {res_orig:.4f}%")
        print(f"[Stage 0] Initial resolution: {res_init:.4f}%")
        print(f"[Stage 1] Current resolution: {res_stage1:.4f}%")

    # After estimating tau and before stopping (or continuing)
    if stop_after_tau:
        print("\nStopping after tau estimation.")
       
        plot_energy_comparison_panel(
            energies_list=[
                orig_energy,
                init_energy,
                best_energy_stage1,
                None,
                None
            ],
            labels=[
                "Original FEBEX",
                "Initial MWD",
                "Stage 1 (fixed tau)",
                "Stage 2 (opt shaping)",
                "Stage 3 (opt tau)"
            ],
            decay_times=[
                None,
                cfg["initial_params"]["decay_time"],
                median_tau,
                None,
                None
            ],
            colors=["gray", "blue", "green", "purple", "orange"],
            save_path=f"output/layer{layer}_x{x}_y{y}_energy_spectra.png",
            show=show_plots
        )

        return
    if verbose > 0:
        print("\nStage 2: Optimize Smoothing L, Amp Start/Stop for resolution")
    enqueue_window = {
        "smoothing_L": params_stage1["smoothing_L"],
        "MWD_amp_start": params_stage1["MWD_amp_start"],
        "MWD_amp_length": params_stage1["MWD_amp_stop"] - params_stage1["MWD_amp_start"]
    }
    best_window, window_study = run_optimization(
        f"layer{layer}_x{x}_y{y}_window", waveforms, params_stage1, objective_window_resolution,
        n_trials=n_trials_window, verbose=verbose, enqueue=enqueue_window, reuse_study=reuse_study
    )
    if verbose > 1: 
        print("lenght of trials: ", len(window_study.trials))
        #  Inspect trial 0 explicitly
        trial0 = window_study.trials[0]
        print("\n[DEBUG] Stage 2 - Trial 0:")
        print(f"  Value: {trial0.value}")
        print(f"  Params: {trial0.params}")
        print(f"  Expected (from enqueue): {enqueue_window}")
    # Extract true best trial from window_study
    valid_trials = [t for t in window_study.trials if t.value is not None and t.value < 1e5]
    if not valid_trials:
        print("[Stage 2] No valid trials - exiting.")
        return

    best_trial_window = min(valid_trials, key=lambda t: t.value)
        
    # Build params_stage2 from best_trial_window + fixed decay_time
    params_stage2 = params_stage1.copy()
    params_stage2["smoothing_L"] = best_trial_window.params["smoothing_L"]
    params_stage2["MWD_amp_start"] = best_trial_window.params["MWD_amp_start"]
    params_stage2["MWD_amp_length"] = best_trial_window.params["MWD_amp_length"]
    params_stage2["MWD_amp_stop"] = params_stage2["MWD_amp_start"] + params_stage2["MWD_amp_length"]
    params_stage2["decay_time"] = median_tau
    if verbose > 1:
        print("[Stage 2] Best parameters:")
        for k, v in params_stage2.items():
            print(f"  {k}: {v}")
 
    
    best_energy_stage2 = []
    for i in sample_indices:
        _, energy, _ = MWD(waveforms[i], params_stage2)
        if not np.isnan(energy):
            best_energy_stage2.append(energy)
    _, _, _, _, res_stage2 = fit_energy_spectrum(best_energy_stage2)
    if verbose > 0:
        print(f"[Stage 2] Best resolution: {res_stage2:.4f}%")
    if verbose > 1:
        print(f"[Stage 2] Trial 0 reported value: {trial0.value:.4f}")
    # After estimating tau and before stopping (or continuing)

    
    valid_trials = [t for t in window_study.trials if t.value is not None and t.value < 1e5]
    smoothing = [t.params["smoothing_L"] for t in valid_trials]
    amp_start = [t.params["MWD_amp_start"] for t in valid_trials]
    amp_length = [t.params["MWD_amp_length"] for t in valid_trials]
    res = [t.value for t in valid_trials]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].scatter(smoothing, res, alpha=0.7)
    axs[0].set_xlabel("Smoothing L (ns)")
    axs[0].set_ylabel("Resolution (%)")
    axs[0].set_title("Resolution vs. smoothing_L")

    axs[1].scatter(amp_start, res, alpha=0.7)
    axs[1].set_xlabel("Amp Start (ns)")
    axs[1].set_title("Resolution vs. amp_start")
    
    axs[2].scatter(amp_length, res, alpha=0.7)
    axs[2].set_xlabel("Amp Length (ns)")
    axs[2].set_title("Resolution vs. amp_length")
    
    fig.tight_layout()
    plot_path = f"output/layer{layer}_x{x}_y{y}_window_parameter_diagnostics.png"
    plt.savefig(plot_path)
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    if stop_after_window:
        print("\nStopping after tau estimation.")
       
        plot_energy_comparison_panel(
            energies_list=[
                orig_energy,
                init_energy,
                best_energy_stage1,
                best_energy_stage2,
                None
            ],
            labels=[
                "Original FEBEX",
                "Initial MWD",
                "Stage 1 (fixed tau)",
                "Stage 2 (opt shaping)",
                "Stage 3 (opt tau)"
            ],
            decay_times=[
                None,
                cfg["initial_params"]["decay_time"],
                median_tau,
                median_tau,
                None
            ],
            colors=["gray", "blue", "green", "purple", "orange"],
            save_path=f"output/layer{layer}_x{x}_y{y}_energy_spectra.png",
            show=show_plots
        )

        return

    # Stage 3: Optimize decay_time again for resolution
    if verbose > 0:
        print("\nStage 3: Optimize decay_time again for resolution")
    enqueue_decay = {"decay_time": params_stage2["decay_time"]}
    best_decay, decay_study = run_optimization(
        f"layer{layer}_x{x}_y{y}_decay",
        waveforms,
        params_stage2,
        objective_decay_resolution,
        n_trials=n_trials_decay,
        verbose=verbose,
        enqueue=enqueue_decay,
        reuse_study=reuse_study
    )
    
    # Extract true best trial from decay_study
    valid_trials = [t for t in decay_study.trials if t.value is not None and t.value < 1e5]
    if not valid_trials:
        print("[Stage 3] No valid trials - exiting.")
        return

    best_trial_decay = min(valid_trials, key=lambda t: t.value)

    # Push to final parameters for any further use
    params_stage3 = params_stage2.copy()
    params_stage3["decay_time"] = best_trial_decay.params["decay_time"]
    
    if verbose > 1:
        print("[Stage 3] Best parameters:")
        for k, v in params_stage3.items():
            print(f"  {k}: {v}")

    best_energy_stage3 = []
    for i in sample_indices:
        _, energy, _ = MWD(waveforms[i], params_stage3)
        if not np.isnan(energy):
            best_energy_stage3.append(energy)
    _, _, _, _, res_stage3 = fit_energy_spectrum(best_energy_stage3)
    if verbose > 0:
        print(f"[Stage 3] Best resolution: {res_stage3:.4f}%")

    # Save best parameters + resolution summary
    os.makedirs("results", exist_ok=True)
    result_data = {
        "layer": layer,
        "x": x,
        "y": y,
        "res_orig": float(res_orig),
        "res_init": float(res_init),
        "median_tau": float(median_tau),
        "res_stage1": float(res_stage1),
        "res_stage2": float(res_stage2),
        "res_stage3": float(res_stage3),
        "final_parameters": params_stage3  # includes smoothing_L, amp_start, decay_time, etc.
    }

    with open(f"results/best_params_layer{layer}_x{x}_y{y}.yaml", "w") as f:
        yaml.dump(result_data, f)
    
    # Plot resolution vs decay_time
    decay_records = [
        {"res": t.value, "decay_time": t.params.get("decay_time")}
        for t in decay_study.trials if t.value is not None and t.value < 1e5
    ]
    
    df_decay = pd.DataFrame(decay_records)
    if not df_decay.empty:
        plt.figure(figsize=(6, 4))
        plt.scatter(df_decay["decay_time"], df_decay["res"], alpha=0.7)
        plt.xlabel("Decay Time [ns]")
        plt.ylabel("Resolution [%]")
        plt.title("Resolution vs Decay Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"output/layer{layer}_x{x}_y{y}_decay_time_parameter_diagnostics.png")
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    plot_energy_comparison_panel(
        energies_list=[
            orig_energy,
            init_energy,
            best_energy_stage1,
            best_energy_stage2,
            best_energy_stage3
        ],
        labels=[
            "Original FEBEX",
            "Initial MWD",
            "Stage 1 (fixed tau)",
            "Stage 2 (opt shaping)",
            "Stage 3 (opt tau)"
        ],
        decay_times=[
            None,
            cfg["initial_params"]["decay_time"],
            median_tau,
            median_tau,
            params_stage3["decay_time"]
        ],
        colors=["gray", "blue", "green", "purple", "orange"],
        save_path=f"output/layer{layer}_x{x}_y{y}_energy_spectra.png",
        show=show_plots
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--stop_after_tau", action="store_true", help="Stop after tau estimation")
    parser.add_argument("--stop_after_window", action="store_true", help="Stop after window optimization")
    parser.add_argument("--show_plots", action="store_true", help="Show plots interactively")
    parser.add_argument("--reuse_study", action="store_true", help="Reuse existing Optuna study if it exists")
    args = parser.parse_args()

    main(
        args.config,
        show_plots=args.show_plots,
        stop_after_tau=args.stop_after_tau,
        stop_after_window=args.stop_after_window,
        reuse_study=args.reuse_study
    )
