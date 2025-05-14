import argparse
import os
import yaml
import subprocess
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

def generate_configs(coarse_config_path):
    with open(coarse_config_path) as f:
        grid = yaml.safe_load(f)

    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    for layer in grid["channels"]["layers"]:
        for x in grid["channels"]["x"]:
            for y in grid["channels"]["y"]:
                config = {
                    "layer": layer,
                    "x": x,
                    "y": y,
                    "input_file": grid["input_path_template"].format(layer=layer, x=x, y=y),
                    "initial_params": grid["initial_params"],
                    "optimization_ranges": grid["optimization_ranges"],
                    "settings": grid["settings"]
                }
                out_path = config_dir / f"layer{layer}_x{x}_y{y}.yaml"
                with open(out_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def run_one_config(cfg_path, show=False, reuse=False, max_retries=3):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{cfg_path.stem}.log"
    result_file = Path(f"results/best_params_{cfg_path.stem}.yaml")

    if result_file.exists():
        return f"Skipped {cfg_path.name}"

    cmd = ["python", "scripts/optimize_MWD.py", "--config", str(cfg_path)]
    if show:
        cmd.append("--show_plots")
    if reuse:
        cmd.append("--reuse_study")

    for attempt in range(1, max_retries + 1):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            with open(log_file, "w") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(proc.stdout)
                f.write("\n--- STDERR ---\n")
                f.write(proc.stderr)
            if proc.returncode == 0 and result_file.exists():
                return f"Success {cfg_path.name} (attempt {attempt})"
            else:
                print(f"[Retry {attempt}] {cfg_path.name} failed (code {proc.returncode})")
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(f"\n[ERROR on attempt {attempt}] {e}\n")

    return f"FAILED {cfg_path.name} after {max_retries} attempts"

def run_all_configs(show=False, reuse=False, n_jobs=None):
    config_dir = Path("config")
    config_files = sorted(config_dir.glob("layer*_x*_y*.yaml"))
    n_jobs = n_jobs or multiprocessing.cpu_count()
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_one_config)(cfg, show, reuse) for cfg in tqdm(config_files)
    )
    for r in results:
        print(r)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to base.yaml")
    parser.add_argument("--show_plots", action="store_true")
    parser.add_argument("--reuse_study", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of parallel workers")
    args = parser.parse_args()

    print("Generating configs...")
    generate_configs(args.config)

    print("Running optimization for all configs (parallel)...")
    run_all_configs(show=args.show_plots, reuse=args.reuse_study, n_jobs=args.n_jobs)

if __name__ == "__main__":
    main()
