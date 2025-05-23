import sys
import os
import yaml
import subprocess
from joblib import Parallel, delayed
import multiprocessing
import argparse
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

# Paths and ranges
#runs = ["run0004", "run0005", "run0006", "run0017", "run0018", "run0019", "run0020"]
runs = ["run0153"]
layers = range(1, 6)
xs = range(5)
ys = range(5)

# Base paths
data_dir = "data"
param_dir = "results"
output_dir = "output"
config_dir = "config"
script_path = "scripts/batch_run_MWD.py"  # or your actual runner script

os.makedirs(config_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


def make_config(run_id, layer, x, y):
    input_file = f"{data_dir}/{run_id}/layer{layer}_x{x}_y{y}.csv"
    if not os.path.exists(input_file):
        return None  # skip if the file doesn't exist

    config = {
        "input_file": input_file,
        "MWD_params": f"{param_dir}/best_params_layer{layer}_x{x}_y{y}.yaml",
        "MWD_output": f"{output_dir}/MWD_{run_id}_layer{layer}_x{x}_y{y}.csv"
    }
    config_path = f"{config_dir}/mwd_{run_id}_layer{layer}_x{x}_y{y}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def run_mwd_job(config_path):
    #print(["python3", script_path, "--config", config_path])
    subprocess.run(["python3", "-u", script_path, "--config", config_path, "--quiet"],
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch run MWD across detector tiles.")
    parser.add_argument("--ncores", type=int, default=multiprocessing.cpu_count(),
                        help="Number of parallel cores to use (default: all available)")
    args = parser.parse_args()

    # Collect all valid configs inside the main block
    configs = []
    for run_id in runs:
        for layer in layers:
            for x in xs:
                for y in ys:
                    cfg = make_config(run_id, layer, x, y)
                    if cfg:
                        configs.append(cfg)

    print(f"Launching MWD on {len(configs)} files using {args.ncores} cores...")
    process_map(run_mwd_job, configs, max_workers=args.ncores)
    print("All MWD jobs completed.")
