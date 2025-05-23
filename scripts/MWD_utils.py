from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

def estimate_decay_time(trace, sampling=10.0, baseline_window=(0, 100), fallback_tau=None):
    baseline = np.mean(trace[baseline_window[0]:baseline_window[1]])
    trace_corr = baseline - trace

    t = np.arange(len(trace)) * sampling
    peak_idx = np.argmax(trace_corr)
    tail_t = t[peak_idx:]
    tail_y = trace_corr[peak_idx:]

    try:
        popt, _ = curve_fit(exp_decay, tail_t - tail_t[0], tail_y, p0=[tail_y[0], 10000])
        A, tau = popt
        fitted = exp_decay(tail_t - tail_t[0], A, tau)
        return tau, tail_t, fitted
    except:
        # fallback: use initial guess curve if fit fails
        if fallback_tau is not None:
            A0 = tail_y[0]
            fitted = exp_decay(tail_t - tail_t[0], A0, fallback_tau)
            return fallback_tau, tail_t, fitted
        else:
            return np.nan, tail_t, None


def batch_estimate_taus(waveforms, sample_size=100, sampling=10.0, plot=True, layer=None, x=None, y=None):
    np.random.seed(0)
    N = len(waveforms)
    if sample_size > N:
        print(f"[Warning] Requested {sample_size} waveforms but only {N} available - reducing sample size.")
        sample_size = N
    idx = np.random.choice(len(waveforms), sample_size, replace=False)
    taus = []
    for i in idx:
        tau, _, _ = estimate_decay_time(waveforms[i], sampling)
        if not np.isnan(tau) and tau > 0 and tau < 1e4:
            taus.append(tau)
    taus = np.array(taus)

    if plot and len(taus) > 0:
        plt.figure()
        plt.hist(taus, bins=int(sample_size/10), color="skyblue", edgecolor="k")
        plt.xlabel('Decay Time tau (ns)')
        plt.ylabel('Count')
        plt.title('Histogram of Decay Time Estimates')
        plt.tight_layout()

        # Save plot
        if layer is not None and x is not None and y is not None:
            tau_plot_file = f"results/layer{layer}_x{x}_y{y}_tau_histogram.png"
            plt.savefig(tau_plot_file)
            print(f"Saved plot to {tau_plot_file}")

        if plot:
            plt.show()
        else:
            plt.close()

    return taus

def load_waveforms(csv_path):
    df = pd.read_csv(csv_path)
    waveforms = df[[c for c in df.columns if c.startswith("t")]].values
    energies = df["energy"].values if "energy" in df.columns else None
    return waveforms, energies

def MWD(trace, params, baseline_corr=True):
    """
    Python translation of MWD algorithm from C4 LisaRaw2Ana.cxx
    Returns MWD trace, energy, and flatness.
    """

    sampling = params["sampling"]
    smoothing_L = params["smoothing_L"]
    MWD_length = params["MWD_length"]
    decay_time = params["decay_time"]

    # Index conversions
    LL = int(smoothing_L / sampling)
    MM = int(MWD_length / sampling)
    tau = decay_time / sampling

    k0 = int(params["MWD_trace_start"] / sampling)
    kend = int(params["MWD_trace_stop"] / sampling)

    amp_start_idx = int(params["MWD_amp_start"] / sampling) - k0
    amp_stop_idx = int(params["MWD_amp_stop"] / sampling) - k0

    baseline_start_idx = int(params["MWD_baseline_start"] / sampling) - k0
    baseline_stop_idx = int(params["MWD_baseline_stop"] / sampling) - k0
    
    # Baseline correction
    baseline = np.mean(trace[20:150]) if baseline_corr else 0.0
    trace_febex_0 = (np.array(trace) - baseline) / 8.0
        
    # MWD trace calculation
    trace_mwd = []
    for kk in range(k0, min(kend, len(trace))):
        DM = 0.0
        sum0 = None
        for j in range(kk - LL, kk):
            if j < 1 or (j - MM) < 0 or j >= len(trace_febex_0):
                continue
            if sum0 is None:
                sum0 = np.sum(trace_febex_0[max(0, j - MM):j])
            else:
                if (j - MM - 1) >= 0:
                    sum0 -= trace_febex_0[j - MM - 1]
                sum0 += trace_febex_0[j - 1]
            DM += trace_febex_0[j] - trace_febex_0[j - MM] + sum0 / tau
        trace_mwd.append(DM / LL)

    trace_mwd = np.array(trace_mwd)

    # Energy and flatness calculation
    valid = (
        0 <= amp_start_idx < amp_stop_idx <= len(trace_mwd)
        and 0 <= baseline_start_idx < baseline_stop_idx <= len(trace_mwd)
    )

    if valid:
        energy_flat = np.mean(trace_mwd[amp_start_idx:amp_stop_idx])
        energy_base = np.mean(trace_mwd[baseline_start_idx:baseline_stop_idx])
        energy = abs(energy_flat - energy_base)
        flat_top = trace_mwd[amp_start_idx:amp_stop_idx]
        flatness = np.std(flat_top) / np.mean(flat_top)
    else:
        energy = np.nan
        flatness = np.nan

    return trace_mwd, energy, flatness
    
    
def gaussian(x, A, mean, sigma):
    return A * np.exp(-(x - mean)**2 / (2 * sigma**2))

def fit_energy_spectrum(energies, bins=200, expand_window=False):
    counts, edges = np.histogram(energies, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    rms = np.std(energies)
    peak = centers[np.argmax(counts)]
    window = 2 * rms if expand_window else 1 * rms
    mask = (centers > peak - window) & (centers < peak + window)
    x_fit = centers[mask]
    y_fit = counts[mask]
    
    try:
        A0 = max(y_fit)
        sigma0 = rms
        popt, _ = curve_fit(
            gaussian,
            x_fit,
            y_fit,
            p0=[A0, peak, sigma0],
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf])  # force sigma > 0
        )
        A, mean, sigma = popt
        resolution = sigma / mean * 100
        return centers, counts, mean, sigma, resolution
    except Exception as e:
        print(f"Fit failed: {e}")
        return centers, counts, None, None, None
    
def plot_energy_comparison_panel(energies_list, labels, decay_times, colors, save_path, show=False):
    import matplotlib.pyplot as plt
    num_panels = len(energies_list)
    fig, axs = plt.subplots(1, num_panels, figsize=(5 * num_panels, 4), sharey=True)

    for i, (energies, label, tau, color) in enumerate(zip(energies_list, labels, decay_times, colors)):
        ax = axs[i]
        if energies is None:
            ax.set_title(label + "\n(TBD)")
            ax.axis("off")
            continue

        x_vals, y_vals, mean, sigma, res = fit_energy_spectrum(energies)
        
        ax.bar(x_vals, y_vals, width=x_vals[1] - x_vals[0], alpha=0.6, color=color, label=label)
        if mean is not None:
            extra = f", tau={tau:.0f} ns" if tau else ""
            ax.plot(x_vals, gaussian(x_vals, max(y_vals), mean, sigma), 'k--',
                    label=f"sigma={sigma:.2f}, res={res:.3f}%{extra}")
        ax.set_title(label)
        ax.set_xlabel("Energy")
        if i == 0:
            ax.set_ylabel("Counts")
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    #print(f"Saved energy panel plot to {save_path}")
