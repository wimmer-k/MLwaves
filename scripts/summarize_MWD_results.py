import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import subprocess
import numpy as np

RESOLUTION_TYPES = [
    ("res_orig", "FEBEX", "red"),
    ("res_init", "Initial MWD", "blue"),
    ("res_stage3", "Optimized MWD", "green")
]

RESOLUTION_TYPES_FULL = [
    ("res_orig_full", "FEBEX", "red"),
    ("res_init_full", "Initial MWD", "blue"),
    ("res_opt_full", "Optimized MWD", "green"),
    ("res_mean_full", "Mean Params", "purple")
]

RESOLUTION_TYPES_WBEST = [
    ("res_orig_full", "FEBEX", "red"),
    ("res_init_full", "Initial MWD", "blue"),
    ("res_opt_full", "Optimized MWD", "green"),
    ("res_mean_full", "Mean Params", "purple"),
    ("res_best_full", "Best Params", "orange")
]

def collect_yaml_results(result_dir):
    result_files = Path(result_dir).glob("best_params_layer*_x*_y*.yaml")
    data = []
    for path in result_files:
        with open(path) as f:
            result = yaml.safe_load(f)
        if "final_parameters" not in result:
            print(f"Warning: {path.name} missing final_parameters")
            continue
        flat = result["final_parameters"]
        flat.update({
            "layer": result["layer"],
            "x": result["x"],
            "y": result["y"],
            "median_tau": result.get("median_tau", None),
            "res_orig": result["res_orig"],
            "res_init": result["res_init"],
            "res_stage1": result["res_stage1"],
            "res_stage2": result["res_stage2"],
            "res_stage3": result["res_stage3"],
        })
        data.append(flat)
    return pd.DataFrame(data)

def plot_resolution_comparison(df, output_prefix, resolution_types):
    df = df.sort_values(["layer", "x", "y"]).reset_index(drop=True)
    df["detector_id"] = df.index + 1

    # Line plot
    plt.figure(figsize=(12, 5))
    for col, label, color in resolution_types:
        if col in df:
            x = df["detector_id"].to_numpy()
            y = df[col].to_numpy()
            plt.plot(x, y, label=label, marker='o', color=color)
    plt.xlabel("Detector ID")
    plt.ylabel("Resolution (%)")
    plt.title("Resolution by Detector")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_resolution_by_detector.png")
    plt.close()

    # Histogram
    plt.figure(figsize=(8, 5))
    min_val = min(df[col].min() for col, _, _ in resolution_types if col in df)
    max_val = max(df[col].max() for col, _, _ in resolution_types if col in df)
    bins = np.linspace(min_val * 0.9, max_val * 1.1, 100)
    for col, label, color in resolution_types:
        if col in df:
            plt.hist(df[col], bins=bins, alpha=0.2, label=label, color=color, edgecolor=color)
    plt.xlabel("Resolution (%)")
    plt.ylabel("Count")
    plt.title("Resolution Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_resolution_distribution.png")
    plt.close()
    
    if "res_opt_full" in df.columns and "res_mean_full" in df.columns:
        plt.figure()
        plt.scatter(df["res_opt_full"], df["res_mean_full"], alpha=0.7, c="darkorange", edgecolors="k")
        plt.xlabel("Optimized MWD Resolution (%)")
        plt.ylabel("Mean Parameter Resolution (%)")
        plt.title("Mean vs Optimized Resolution")
        maxi = max(df["res_opt_full"].max() , df["res_mean_full"].max())
        mini = min(df["res_opt_full"].min() , df["res_mean_full"].min())
        #print(mini,maxi)
        plt.plot([mini, maxi], [mini, maxi], alpha=0.5, linestyle="--", color="gray")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_scatter_mean_vs_optimized.png")
        plt.close()

    # Fitted vs. optimized tau plot
    if "median_tau" in df.columns and "decay_time" in df.columns:
        plt.figure()
        plt.scatter(df["median_tau"], df["decay_time"], alpha=0.7, c="darkcyan", edgecolors="k")
        plt.xlabel("Fitted Median Tau (ns)")
        plt.ylabel("Optimized Decay Time (ns)")
        plt.title("Fitted vs Optimized Decay Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_fitted_vs_final_decay_time.png")
        plt.close()
        
def recalculate_resolutions_from_waveforms(df, csv_dir="data", result_dir="results", mean_params=None, best_params=None):
    from MWD_utils import MWD, fit_energy_spectrum

    recalc_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Re-evaluating CSVs"):
        layer, x, y = row["layer"], row["x"], row["y"]
        csv_file = Path("data/run0006") / f"layer{int(layer)}_x{int(x)}_y{int(y)}.csv"
        if not csv_file.exists():
            continue
        df_wave = pd.read_csv(csv_file)
        time_cols = [c for c in df_wave.columns if c.startswith("t")]
        waveforms = df_wave[time_cols].values
        orig_energies = df_wave["energy"].values
        e_init = df_wave["MWD"].values
        e_opt = [MWD(w, row, baseline_corr=True)[1] for w in waveforms]
        e_mean = [MWD(w, mean_params, baseline_corr=True)[1] for w in waveforms]

        _, _, _, _, res_orig = fit_energy_spectrum(orig_energies)
        _, _, _, _, res_init = fit_energy_spectrum(e_init)
        _, _, _, _, res_opt = fit_energy_spectrum(e_opt)
        _, _, _, _, res_mean = fit_energy_spectrum(e_mean)
        if best_params:
            e_best = [MWD(w, best_params, baseline_corr=True)[1] for w in waveforms]
            _, _, _, _, res_best = fit_energy_spectrum(e_best)
        else:
            res_best = np.nan

        recalc_data.append({
            "layer": layer, "x": x, "y": y,
            "res_orig_full": res_orig,
            "res_init_full": res_init,
            "res_opt_full": res_opt,
            "res_mean_full": res_mean,
            "res_best_full": res_best
        })

    df_full = pd.DataFrame(recalc_data)
    df_full.to_csv(Path(result_dir) / "recalculated_resolutions_full.csv", index=False)
    plot_resolution_comparison(df_full, Path(result_dir) / "full", RESOLUTION_TYPES_WBEST)

def main(result_dir, output_csv):
    df = collect_yaml_results(result_dir)
    output_path = Path(result_dir) / output_csv
    df.to_csv(output_path, index=False)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except Exception:
        git_hash = "N/A"
    with open(Path(result_dir) / "summary_info.txt", "w") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Git commit: {git_hash}\n")
        f.write(f"CSV output: {output_csv}\n")

    print(f"Saved results to {output_path}")
    plot_resolution_comparison(df, Path(result_dir) / "summary", RESOLUTION_TYPES)
    
    # 4-panel parameter distribution plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    param_list = ["smoothing_L", "MWD_amp_start", "MWD_amp_length", "decay_time"]
    for i, param in enumerate(param_list):
        ax = axs[i // 2, i % 2]
        if param in df.columns:
            ax.hist(df[param], bins=20, color="skyblue", edgecolor="k")
            ax.set_title(f"Distribution of {param}")
            ax.set_xlabel(param)
            ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(Path(result_dir) / "param_distributions_combined.png")
    plt.close()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="results", help="Directory containing best_params_*.yaml")
    parser.add_argument("--output_csv", default="mwd_batch_results.csv", help="Output CSV file")
    parser.add_argument("--recalculate_all", action="store_true", help="Recalculate energy resolutions from CSV using optimized and mean parameters")
    parser.add_argument("--plot_recalculated_only", action="store_true", help="Only plot previously recalculated resolution data")
    parser.add_argument("--best_params", help="YAML config file with best parameters to compare")
    args = parser.parse_args()

    if args.plot_recalculated_only:
        df = pd.read_csv(Path(args.result_dir) / "recalculated_resolutions_full.csv")
        plot_resolution_comparison(df, Path(args.result_dir) / "full", RESOLUTION_TYPES_FULL)
    elif args.recalculate_all:
        print("Recalculating resolutions from full waveform data...")
        df = pd.read_csv(Path(args.result_dir) / args.output_csv)
        mean_params = df[['smoothing_L', 'MWD_amp_start', 'MWD_amp_length', 'decay_time']].mean().to_dict()
        mean_params.update(df.iloc[0].to_dict())
        best_params = None
        if args.best_params:
            with open(args.best_params) as f:
                best_cfg = yaml.safe_load(f)
            best_params = best_cfg["initial_params"]
        recalculate_resolutions_from_waveforms(df, csv_dir='data', result_dir=args.result_dir, mean_params=mean_params, best_params=best_params)
    else:
        main(args.result_dir, args.output_csv)
