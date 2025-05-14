import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

INPUT_DIR = "./data/test_0010_csv"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def analyze_channel(file_name):
    file_path = os.path.join(INPUT_DIR, file_name)
    df = pd.read_csv(file_path)
    
    time_cols = [col for col in df.columns if col.startswith("t")]
    waveforms = df[time_cols].values
    n_events, n_samples = waveforms.shape

    ### Plot 1: Amplitude vs Time Heatmap ###
    time_indices = np.tile(np.arange(n_samples), n_events)
    amplitudes = waveforms.flatten()

    plt.figure(figsize=(10, 6))
    plt.hist2d(time_indices, amplitudes, bins=[100, 100], cmap="viridis")
    plt.colorbar(label="Counts")
    plt.xlabel("Time Sample Index")
    plt.ylabel("Amplitude")
    plt.title(f"Amplitude vs Time - {file_name}")
    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, file_name.replace(".csv", "_heatmap.png"))
    plt.savefig(heatmap_path)
    plt.close()

    ### Plot 2: Energy Spectrum + Gaussian Fit ###
    energies = df["energy"].values
    counts, bins = np.histogram(energies, bins=200)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Try to find peak region automatically
    peak_idx = np.argmax(counts)
    mu0 = bin_centers[peak_idx]
    fit_min = mu0 - 4000
    fit_max = mu0 + 4000

    fit_mask = (bin_centers > fit_min) & (bin_centers < fit_max)
    x_fit = bin_centers[fit_mask]
    y_fit = counts[fit_mask]

    try:
        A0 = max(y_fit)
        sigma0 = (fit_max - fit_min) / 6
        popt, pcov = curve_fit(gaussian, x_fit, y_fit, p0=[A0, mu0, sigma0], maxfev=5000)
        A_fit, mu_fit, sigma_fit = popt
        resolution = (sigma_fit / mu_fit) * 100
        fit_label = f"Fit: mean={mu_fit:.2f}, sigma={sigma_fit:.2f}, Res={resolution:.2f}%"
    except Exception as e:
        popt = None
        fit_label = "Fit failed"

    plt.figure(figsize=(8, 5))
    plt.hist(energies, bins=200, alpha=0.3, label="Energy spectrum")
    if popt is not None:
        x_plot = np.linspace(fit_min, fit_max, 500)
        plt.plot(x_plot, gaussian(x_plot, *popt), 'r-', label=fit_label)
    plt.xlabel("Energy")
    plt.ylabel("Counts")
    plt.title(f"Energy Spectrum - {file_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    spectrum_path = os.path.join(OUTPUT_DIR, file_name.replace(".csv", "_spectrum.png"))
    plt.savefig(spectrum_path)
    plt.close()

    print(f"Processed {file_name:35} -> resolution: {fit_label}")

def main():
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".csv"))
    for f in files:
        analyze_channel(f)

if __name__ == "__main__":
    main()
