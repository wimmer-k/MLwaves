import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
from MWD_utils import MWD, load_waveforms
import yaml


def run_mwd(waveforms, mwd_params_path, quiet=False):
    from MWD_utils import MWD
    with open(mwd_params_path, "r") as f:
        mwd_config = yaml.safe_load(f)
    mwd_params = mwd_config.get("final_parameters", mwd_config)
    mwd_energies = []
    for wf in tqdm(waveforms, desc="MWD energy calculation", disable=quiet):
        _, energy, _ = MWD(wf, mwd_params)
        mwd_energies.append(energy)
    return np.array(mwd_energies)

def main(config_path, quiet=False):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    data_path = cfg["input_file"]
    waveforms, orig_energies = load_waveforms(data_path)
    if not quiet:
        print("Loaded waveforms")
    
    if "MWD_params" in cfg:
        mwd_params_path = cfg["MWD_params"]
        mwd_energies = run_mwd(waveforms, mwd_params_path, quiet=quiet)
        output_path = cfg.get("MWD_output", "output/mwd_energies.csv")
        pd.DataFrame({"MWD_energy": mwd_energies}).to_csv(output_path, index=False)
        if not quiet:
            print(f"MWD energies saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run matched filter energy extraction or template generation.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--quiet", action="store_true", help="Suppress tqdm output (for parallel use)")
    args = parser.parse_args()

    main(args.config, quiet=args.quiet)

