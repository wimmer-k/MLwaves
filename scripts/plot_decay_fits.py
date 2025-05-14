import argparse
import numpy as np
import matplotlib.pyplot as plt
from MWD_utils import estimate_decay_time
import pandas as pd

def plot_decay_fits(input_file, sample_size=12, fallback_tau=None, sampling=10.0):
    df = pd.read_csv(input_file)
    waveforms = df[[c for c in df.columns if c.startswith("t")]].values

    np.random.seed(0)
    idx = np.random.choice(len(waveforms), sample_size, replace=False)

    taus = []
    plt.figure(figsize=(12, 8))
    for i, trace_idx in enumerate(idx):
        trace = waveforms[trace_idx]
        tau, tail_t, fitted = estimate_decay_time(trace, sampling=sampling, fallback_tau=fallback_tau)
        taus.append(tau)

        baseline = np.mean(trace[:100])
        trace_corr = baseline - trace
        t = np.arange(len(trace)) * sampling

        plt.subplot(3, 4, i + 1)
        plt.plot(t, trace_corr, label='Baseline-corrected')
        if fitted is not None:
            plt.plot(tail_t, fitted, 'r--', label=f'Decay tau={tau:.0f} ns')
        plt.title(f'Event {trace_idx}')
        plt.xlabel('Time (ns)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.tight_layout()

    plt.show()

    taus = np.array(taus)
    taus = taus[~np.isnan(taus)]

    if len(taus) > 0:
        plt.figure(figsize=(6, 4))
        plt.hist(taus, bins=20, color="skyblue", edgecolor="k")
        plt.xlabel('Decay Time tau (ns)')
        plt.ylabel('Count')
        plt.title('Histogram of Decay Time Estimates')
        plt.tight_layout()
        plt.show()
        print(f"\nMedian tau estimate: {np.median(taus):.1f} ns")
    else:
        print("\nNo valid tau estimates found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to waveform CSV file")
    parser.add_argument("--sample_size", type=int, default=12, help="Number of traces to plot")
    parser.add_argument("--tau", type=float, default=None, help="Fallback decay time (ns) if fit fails")
    parser.add_argument("--config", type=str, help="Optional YAML config with initial params")
    args = parser.parse_args()

    fallback_tau = args.tau
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        fallback_tau = cfg["initial_params"]["decay_time"]

    plot_decay_fits(args.input, sample_size=args.sample_size, fallback_tau=fallback_tau)

