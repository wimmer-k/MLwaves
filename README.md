# MLwaves Optimization Pipeline

## Overview

This tool performs optimization of Moving Window Deconvolution (MWD) parameters using waveform traces. It uses a three-stage pipeline driven by Optuna:

1. Estimate decay time (`tau`) from waveform tails
2. Optimize shaping parameters: `smoothing_L`, `MWD_amp_start`, `MWD_amp_length`
3. Re-optimize `decay_time` for best resolution

Each step is traceable, reproducible, and includes diagnostic output and plotting.

---

## Workflow

### Stage 1 – Estimate Decay Time

- `tau` is estimated using exponential tail fitting across a sample of waveforms
- This value is passed to Stage 2 and used for initial MWD energy reconstruction

### Stage 2 – Optimize Shaping Parameters

- Fixed `tau`, optimize `smoothing_L`, `MWD_amp_start`, `MWD_amp_length`
- Resolution vs. shaping parameters is plotted for diagnostics
- Best parameters are passed to Stage 3

### Stage 3 – Optimize Tau (Decay Time)

- Fixes best shaping parameters from Stage 2
- Optimizes `decay_time` using best resolution as objective

---

## Usage

### Run full pipeline:

```bash
python optimize_mwd.py --config config/layer1_x3_y1.yaml
```

### Show plots during execution:

```bash
python optimize_mwd.py --config config/layer1_x3_y1.yaml --show_plots
```

### Stop after tau estimation only:

```bash
python optimize_mwd.py --config config/layer1_x3_y1.yaml --stop_after_tau
```

### Stop after shaping optimization:

```bash
python optimize_mwd.py --config config/layer1_x3_y1.yaml --stop_after_window
```

### Reuse existing Optuna study files (default is fresh):

```bash
python optimize_mwd.py --config config/layer1_x3_y1.yaml --reuse_study
```

---

## Output

### Plots

- `output/layer{L}_x{X}_y{Y}_tau_histogram.png` - histogram of fitted tau
- `output/layer{L}_x{X}_y{Y}_energy_spectra.png` - 5-panel energy comparison plot
- `output/layer{L}_x{X}_y{Y}_window_parameter_diagnostics.png` - resolution vs shaping params
- `output/layer{L}_x{X}_y{Y}_decay_time_parameter_diagnostics.png` - resolution vs decay time

### Optimization Databases

- `layer{L}_x{X}_y{Y}_window_study.db`
- `layer{L}_x{X}_y{Y}_decay_study.db`

Use with Optuna dashboard:

```bash
optuna-dashboard sqlite:///layer1_x3_y1_window_study.db --port 8080
optuna-dashboard sqlite:///layer1_x3_y1_decay_study.db --port 8081
```

---

## Configuration

Edit your YAML config:

```yaml
layer: 1
x: 3
y: 1

input_file: data/runXXXX/layer1_x3_y1.csv

initial_params:
  sampling: 10.0
  smoothing_L: 200.0
  MWD_length: 400.0
  decay_time: 3300.0
  MWD_trace_start: 500.0
  MWD_trace_stop: 2000.0
  MWD_amp_start: 1350.0
  MWD_amp_stop: 1400.0
  MWD_baseline_start: 500.0
  MWD_baseline_stop: 650.0

optimization_ranges:
  smoothing_L: [100.0, 300.0]
  amp_start: [1300.0, 1400.0]
  amp_length: [10.0, 100.0]
  decay_time: [2000.0, 2500.0]
  amp_limit: 1500.0

settings:
  tau_sample_size: 3000
  n_trials_window: 50
  n_trials_decay: 30
```

---

## Dependencies

- numpy
- pandas
- scipy
- matplotlib
- tqdm
- optuna
- pyyaml
