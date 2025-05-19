# matched_filter_energy.py

import numpy as np
import pandas as pd
from scipy.signal import correlate
from tqdm import tqdm
import yaml
import os
import argparse
import matplotlib.pyplot as plt
from time import perf_counter

def align_by_slope(waveform, baseline_window, align_range, align_to, norm_window):
    baseline = np.mean(waveform[baseline_window[0]:baseline_window[1]])
    centered = waveform - baseline
    derivative = np.diff(centered)
    slope_idx = np.argmin(derivative[align_range[0]:align_range[1]]) + align_range[0]
    shift = align_to - slope_idx
    aligned = np.roll(centered, shift)
    min_val = np.min(aligned[align_to + norm_window[0]:align_to + norm_window[1]])
    normalized = aligned / abs(min_val)
    return normalized


def build_template(waveforms, config):
    baseline_window = config.get("baseline_window", [20, 80])
    align_range = config.get("align_range", [80, 150])
    align_to = config.get("align_to", 100)
    norm_window = config.get("normalization_window", [10, 30])
    template_count = config.get("template_count", 200)

    aligned_stack = np.array([
        align_by_slope(wf, baseline_window, align_range, align_to, norm_window)
        for wf in waveforms[:template_count]
    ])
    median_template = np.median(aligned_stack, axis=0)
    return median_template


def matched_filter_energy(waveform, template, baseline_window):
    wf = waveform - np.mean(waveform[baseline_window[0]:baseline_window[1]])
    template = template - np.mean(template)
    template /= np.linalg.norm(template)
    correlation = correlate(wf, template, mode='valid')
    return np.max(correlation)


def process_waveforms(waveforms, template, baseline_window):
    energies = []
    for wf in tqdm(waveforms, desc="Matched filter energy"):
        energy = matched_filter_energy(wf, template, baseline_window)
        energies.append(energy)
    return np.array(energies)


def load_waveforms(csv_path):
    df = pd.read_csv(csv_path)
    waveforms = df[[c for c in df.columns if c.startswith("t")]].values
    energies = df["energy"].values if "energy" in df.columns else None
    return waveforms, energies


def load_template(template_path):
    ext = os.path.splitext(template_path)[1].lower()
    if ext == ".npy":
        template = np.load(template_path)
    elif ext == ".csv":
        template = np.loadtxt(template_path, delimiter=",")
    else:
        raise ValueError("Unsupported template file format")
    return np.ravel(template)

def save_template(template, output_path):
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".npy":
        np.save(output_path, template)
    elif ext == ".csv":
        np.savetxt(output_path, template, delimiter=",")
    else:
        raise ValueError("Unsupported template file format for saving")

def run_mwd(waveforms, mwd_params_path):
    from MWD_utils import MWD
    with open(mwd_params_path, "r") as f:
        mwd_config= yaml.safe_load(f)
    # Extract only the actual parameter dict
    mwd_params = mwd_config.get("final_parameters", mwd_config)
    #print(mwd_params)
    mwd_energies = []
    for wf in tqdm(waveforms, desc="MWD energy"):
        _, energy, _ = MWD(wf, mwd_params)
        mwd_energies.append(energy)
    return np.array(mwd_energies)

        ## plt.figure(figsize=(10, 4))
        ## plt.plot(template)
        ## plt.title("Matched Filter Template")
        ## plt.xlabel("Sample")
        ## plt.ylabel("Amplitude")
        ## plt.grid(True)
        ## plt.tight_layout()
        ## plt.show()

def compare_spectra(energy_dict, nbins=500, normalize=True):
    """
    Compare multiple energy spectra given as a dictionary {label: energy_array}.
    Optionally normalize all to match the scale of the first entry.
    """
    labels = list(energy_dict.keys())
    energies = [np.array(energy_dict[k]) for k in labels]

    if len(energies) < 2:
        raise ValueError("Need at least two energy arrays to compare.")

    # Compute robust percentiles for clipping
    percentiles = [(np.percentile(e, 1), np.percentile(e, 99)) for e in energies]
    clipped = [e[(e >= p[0]) & (e <= p[1])] for e, p in zip(energies, percentiles)]

    # Plotting layout
    n = len(energies)
    fig, axs = plt.subplots(2, max(2, n), figsize=(14, 6))

    # Top row: individual histograms
    for i, (label, e, (pmin, pmax)) in enumerate(zip(labels, energies, percentiles)):
        bins = np.linspace(pmin, pmax, nbins)
        axs[0, i].hist(e, bins=bins, alpha=0.7)
        axs[0, i].set_title(label)
        axs[0, i].set_xlabel("Energy")
        axs[0, i].grid(True)

    # Bottom-center and bottom-right: 2D heatmaps to ref (0)
    for i in range(1, min(n, 3)):
        bins2d_x = np.linspace(*percentiles[0], nbins)
        bins2d_y = np.linspace(*percentiles[i], nbins)
        axs[1, i].hist2d(energies[0], energies[i], bins=[bins2d_x, bins2d_y], cmap="viridis")
        axs[1, i].set_title(f"2D: {labels[i]} vs {labels[0]}")
        axs[1, i].set_xlabel(labels[0])
        axs[1, i].set_ylabel(labels[i])
        axs[1, i].grid(True)

    # Step 3: Normalize (after top row!)
    if normalize:
        ref_mean = np.mean(clipped[0])
        ref_std = np.std(clipped[0])
        for i in range(1, len(energies)):
            mean_i = np.mean(clipped[i])
            std_i = np.std(clipped[i])
            energies[i] = (energies[i] - mean_i) / std_i * ref_std + ref_mean
            clipped[i] = (clipped[i] - mean_i) / std_i * ref_std + ref_mean

    # Bottom-left: overlay of all (scaled)
    combined = np.concatenate(clipped)
    bins = np.histogram_bin_edges(combined, bins=nbins)
    for i, label in enumerate(labels):
        axs[1, 0].hist(clipped[i], bins=bins, alpha=0.5, label=label)
    axs[1, 0].set_title("Overlay (scaled)")
    axs[1, 0].set_xlabel("Energy")
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    plt.tight_layout()
    plt.show()


def main(config_path, make_template=False, show_plots=False, compare=False):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_path = cfg["input_file"]
    waveforms, orig_energies = load_waveforms(data_path)
    print("Loaded waveforms")

    if make_template:
        print("Generating and saving template from waveforms...")
        template = build_template(waveforms, cfg)
        output_template = os.path.join("output", cfg.get("output_template", "template_generated.npy"))
        save_template(template, output_template)
        print(f"Template saved to {output_template}")

        if show_plots:
            # Plot the template
            plt.figure(figsize=(10, 4))
            plt.plot(template)
            plt.title("Generated Template")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        return

    # Use existing or build template for processing
    if "template_file" in cfg:
        template = load_template(cfg["template_file"])
        print("Loaded external template with length", len(template))
    else:
        print("Building template from waveform data...")
        template = build_template(waveforms, cfg)

    baseline_window = cfg.get("baseline_window", [20, 80])
    print("Processing waveforms...")
    start = perf_counter()
    energies = process_waveforms(waveforms, template, baseline_window)
    print(f"Matched filter done in {perf_counter() - start:.2f} s")
    
    output_path = cfg.get("output_file", "matched_energies.csv")
    pd.DataFrame({"matched_energy": energies}).to_csv(output_path, index=False)
    print(f"Saved matched filter energies to {output_path}")

    dictforplot = {"FEBEX": orig_energies, "Matched": energies}
    if compare and "MWD_params" in cfg:
        mwd_params_path = cfg["MWD_params"]
        print("MWD energy calculation..")
        start = perf_counter()
        mwd_energies = run_mwd(waveforms, mwd_params_path)
        print(f"MWD done in {perf_counter() - start:.2f} s")
        dictforplot["MWD"] = mwd_energies
        
    if show_plots:
        compare_spectra(dictforplot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run matched filter energy extraction or template generation.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--make-template", action="store_true", help="Generate and save a template instead of filtering")
    parser.add_argument("--show-plots", action="store_true", help="Show diagnostic plots")
    parser.add_argument("--compare", action="store_true", help="Run MWD and Matched filter, compare performance")
    args = parser.parse_args()

    main(args.config, make_template=args.make_template, show_plots=args.show_plots, compare=args.compare)

