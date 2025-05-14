import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
            "res_orig": result["res_orig"],
            "res_init": result["res_init"],
            "res_stage1": result["res_stage1"],
            "res_stage2": result["res_stage2"],
            "median_tau": result.get("median_tau", None),
        "res_stage3": result["res_stage3"],
        })
        data.append(flat)
    return pd.DataFrame(data)


def plot_resolution_by_detector(df, output_dir="results"):
    df = df.sort_values(["layer", "x", "y"]).reset_index(drop=True)
    df["detector_id"] = df.index + 1

    plt.figure(figsize=(12, 5))
    plt.plot(df["detector_id"], df["res_orig"], label="FEBEX", marker='o')
    plt.plot(df["detector_id"], df["res_init"], label="Initial MWD", marker='o')
    plt.plot(df["detector_id"], df["res_stage3"], label="Optimized", marker='o')
    plt.xlabel("Detector ID (1-125)")
    plt.ylabel("Resolution (%)")
    plt.title("Energy Resolution by Detector")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/resolution_by_detector.png")
    plt.close()

def plot_combined_resolution_histogram(df, output_path="results/combined_resolution_distribution.png"):
    plt.figure(figsize=(8, 5))
    plt.hist(df["res_orig"], bins=30, alpha=0.5, label="FEBEX", color="gray", edgecolor="k")
    plt.hist(df["res_init"], bins=30, alpha=0.5, label="Initial MWD", color="blue", edgecolor="k")
    plt.hist(df["res_stage3"], bins=30, alpha=0.5, label="Optimized MWD", color="green", edgecolor="k")
    plt.xlabel("Resolution (%)")
    plt.ylabel("Count")
    plt.title("Resolution Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
def make_plots(df, output_dir="results"):
    Path(output_dir).mkdir(exist_ok=True)
    
    # Scatter plot of median_tau vs optimized decay_time
    plt.figure()
    plt.scatter(df["median_tau"], df["decay_time"], alpha=0.7)
    plt.xlabel("Median tau (ns)")
    plt.ylabel("Optimized decay_time (ns)")
    plt.title("Fitted vs Optimized Decay Time")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scatter_fitted_vs_final_decay_time.png")
    plt.close()

    for param in ["smoothing_L", "MWD_amp_start", "MWD_amp_length", "decay_time"]:
        if param in df.columns:
            df[param].hist(bins=30)
            plt.xlabel(param)
            plt.ylabel("Count")
            plt.title(f"Distribution of {param}")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/param_distribution_{param}.png")
            plt.close()

def main(result_dir, output_csv):
    df = collect_yaml_results(result_dir)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")
    make_plots(df)
    plot_combined_resolution_histogram(df)
    plot_resolution_by_detector(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="results", help="Directory containing best_params_*.yaml")
    parser.add_argument("--output_csv", default="results/MWD_batch_results.csv", help="Output CSV file")
    args = parser.parse_args()
    main(args.result_dir, args.output_csv)
