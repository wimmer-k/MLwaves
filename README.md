# MLwaves Optimization Pipeline

Package contains several scripts and notebooks to analyze the LISA waveform data. 

---

## Moving Window Deconvolution optimization

This tool performs optimization of Moving Window Deconvolution (MWD) parameters using waveform traces. It uses a three-stage pipeline driven by Optuna:

1. Estimate decay time (`tau`) from waveform tails
2. Optimize shaping parameters: `smoothing_L`, `MWD_amp_start`, `MWD_amp_length`
3. Re-optimize `decay_time` for best resolution

Each step is traceable, reproducible, and includes diagnostic output and plotting.

### Workflow

#### Stage 1 – Estimate Decay Time

- `tau` is estimated using exponential tail fitting across a sample of waveforms
- This value is passed to Stage 2 and used for initial MWD energy reconstruction

#### Stage 2 – Optimize Shaping Parameters

- Fixed `tau`, optimize `smoothing_L`, `MWD_amp_start`, `MWD_amp_length`
- Resolution vs. shaping parameters is plotted for diagnostics
- Best parameters are passed to Stage 3

#### Stage 3 – Optimize Tau (Decay Time)

- Fixes best shaping parameters from Stage 2
- Optimizes `decay_time` using best resolution as objective


### Usage

#### Run full pipeline:

```bash
python3 scripts/optimize_mwd.py --config config/layer1_x3_y1.yaml
```

#### Show plots during execution:

```bash
python3 scripts/optimize_mwd.py --config config/layer1_x3_y1.yaml --show_plots
```

#### Stop after tau estimation only:

```bash
python3 scripts/optimize_mwd.py --config config/layer1_x3_y1.yaml --stop_after_tau
```

#### Stop after shaping optimization:

```bash
python3 scripts/optimize_mwd.py --config config/layer1_x3_y1.yaml --stop_after_window
```

#### Reuse existing Optuna study files (default is fresh):

```bash
python3 scripts/optimize_mwd.py --config config/layer1_x3_y1.yaml --reuse_study
```


#### Batch Execution with  `batch_opt_MWD.py`

Executes `optimize_mwd.py` over a grid of detectors (typically all x/y for a given layer).

```bash
python3 scripts/batch_opt_MWD.py --config config/coarse_grid.yaml
```

or using Python multiprocessing to speed up execution on multicore machines.

```bash
python3 scripts/batch_opt_MWD_parallel.py --config config/coarse_grid.yaml
```


### Output

#### Plots

- `output/layer{L}_x{X}_y{Y}_tau_histogram.png` - histogram of fitted tau
- `output/layer{L}_x{X}_y{Y}_energy_spectra.png` - 5-panel energy comparison plot
- `output/layer{L}_x{X}_y{Y}_window_parameter_diagnostics.png` - resolution vs shaping params
- `output/layer{L}_x{X}_y{Y}_decay_time_parameter_diagnostics.png` - resolution vs decay time

#### Optimization Databases

- `layer{L}_x{X}_y{Y}_window_study.db`
- `layer{L}_x{X}_y{Y}_decay_study.db`

#### Use with Optuna dashboard:

```bash
optuna-dashboard sqlite:///layer1_x3_y1_window_study.db --port 8080
optuna-dashboard sqlite:///layer1_x3_y1_decay_study.db --port 8081
```

#### Explore and compare results 'summarize_MWD_results.py'

This script summarizes all optimization results and optionally recalculates energy resolutions using:

- original FEBEX energy
- initial MWD parameters
- optimized parameters
- mean parameters
- externally defined "best" parameters

**Usage Examples:**

1. Full summary and plots:

```bash
python3 scripts/summarize_MWD_results.py
```

2. With full waveform recomputation:

```bash
python3 scripts/summarize_MWD_results.py --recalculate_all
```

3. Add a "best param" reference set:

```bash
python3 scripts/summarize_MWD_results.py --recalculate_all --best_config config/best_params.yaml
```

4. Just re-plot from existing data:

```bash
python3 scripts/summarize_MWD_results.py --plot_recalculated_only
```

Plots are saved in `results/` with consistent naming.


### Configuration

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

for the coarse search to optimize the parameters, use a master config:

```yaml
channels:
  layers: [1, 2, 3, 4, 5]
  x: [0, 1, 2, 3]
  y: [0, 1, 2, 3]

input_path_template: data/run0006/layer{layer}_x{x}_y{y}.csv

# General run settings
settings:
  tau_sample_size: 2000
  n_trials_window: 300
  n_trials_decay: 300
  verbose: 0

# Initial waveform parameters
initial_params:
  sampling: 10.0
  smoothing_L: 200.0
  MWD_length: 400.0
  decay_time: 2200.0
  MWD_trace_start: 500.0
  MWD_trace_stop: 2000.0
  MWD_amp_start: 1350.0
  MWD_amp_stop: 1400.0
  MWD_baseline_start: 500.0
  MWD_baseline_stop: 650.0

# Optimization ranges
optimization_ranges:
  smoothing_L: [50.0, 500.0]
  amp_start: [1200.0, 1500.0]
  amp_length: [10.0, 150.0]
  decay_time: [2000.0, 2500.0]
  amp_limit: 1500.0
```

---

## Matched Filter Energy Extraction

Matched filter energy extraction using a clean reference pulse or one built from waveform data.

### Usage

#### Basic command:

```bash
python3 scripts/matched_filter.py --config config/matched_test.yaml
```

with example config file:

```yaml
input_file: data/run0153/layer4_x2_y0.csv
output_file: output/matched_energies_layer4_x2_y0.csv

# Optional: use external template
template_file: output/template_generated.npy

MWD_params: results/best_params_layer4_x2_y0.yaml
```

#### Optional flags:
- `--make-template` generate and save a matched filter template from waveform data
- `--show-plots` show diagnostic plots comparing energy estimators
- `--compare` run MWD energy extraction using parameters from a YAML file, compare performance

To generate a template from data, primary beam or pulser, use the `--make-template` flag with a configuration file specifying the data, output file as well as the parameters for the matching.

```yaml
input_file: data/run0003/layer4_x2_y0.csv
output_file: output/template_layer4_x2_y0.npy

# If building a template:
template_count: 200
baseline_window: [20, 80]
align_range: [80, 150]
align_to: 100
normalization_window: [10, 30]
```

#### Plots:

When `--show-plots` is enabled, a four(or six)-panel summary is generated:

- Individual energy spectra
- Overlay (scaled) of two or more methods
- 2D heatmaps comparing matched vs other energies

Scaling is applied using robust percentiles (1st–99th), with outliers excluded from scaling but included in visual inspection.

---

## Dependencies

- numpy
- pandas
- scipy
- matplotlib
- tqdm
- optuna
- pyyaml
