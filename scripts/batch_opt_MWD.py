import argparse
import os
import yaml
import subprocess
from pathlib import Path
from tqdm import tqdm

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

def run_all_configs(show=False, reuse=False):
    config_dir = Path("config")
    config_files = sorted(config_dir.glob("layer*_x*_y*.yaml"))

    for cfg in tqdm(config_files, desc="Running MWD Optimization"):
        cmd = ["python3", "scripts/optimize_MWD.py", "--config", str(cfg)]
        if show:
            cmd.append("--show_plots")
        if reuse:
            cmd.append("--reuse_study")
        #print(cmd)
        subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to base.yaml")
    parser.add_argument("--show_plots", action="store_true")
    parser.add_argument("--reuse_study", action="store_true")
    args = parser.parse_args()

    print("Generating configs...")
    generate_configs(args.config)

    print("Running optimization for all configs...")
    run_all_configs(show=args.show_plots, reuse=args.reuse_study)

if __name__ == "__main__":
    main()
