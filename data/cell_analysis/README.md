# Cell analysis: ACGs, cell metrics and cell-type labels

Autocorrelogram (ACG) metrics follow CellExplorer (Petersen et al., 2021, *Neuron* 109:3594–3608; https://cellexplorer.org). Please cite CellExplorer if you use these functions.

## Files

| File | Content |
| --- | --- |
| `spikes_struct.mat` | Spike times (seconds) for all 902 units (`spikes.times`, `spikes.total`, `spikes.UID`) |
| `acg.mat` | ACGs used in the paper: `acg.narrow` (±50 ms, 0.5 ms bins; 201 × 902) and `acg.wide` (±500 ms, 1 ms bins; 1001 × 902), in spikes/s; one column per unit, in the order of `spikes_struct.mat` |
| `calc_ACG_metrics.m` | Adapted from CellExplorer: computes the ACGs with `compute_acg.m` instead of CellExplorer's compiled `CCG` function, so no sampling rate or MEX file is needed |
| `compute_acg.m` | ACG of one spike train (spike-pair intervals binned directly in seconds) |
| `fit_ACG.m` | CellExplorer's triple-exponential ACG fit (unmodified; requires the Curve Fitting Toolbox) |
| `normalize_acg_to_unit.m` | Divides each ACG by its maximum (zero-lag bin set to 0) |
| `Mean_ACG.m` | Raw and normalized mean ACG per unit from `acg.mat` |
| `Cell_metrics.xlsx` | All metrics per unit (see below) |

## Reproducing the metrics

Run from this folder, so that this `calc_ACG_metrics.m` is used rather than CellExplorer's original (which takes different inputs).

1. **ACGs.** `load('spikes_struct.mat'); acg_metrics = calc_ACG_metrics(spikes);` This reproduces `acg.mat` (`acg_metrics.acg_narrow`, `acg_metrics.acg_wide`).
2. **Mean ACG.** Run `Mean_ACG.m`: the raw mean of the wide ACG (`mean_acg`, used for clustering) and the normalized mean (`norm_acg`).
3. **ACG fits.** `fit = fit_ACG(acg.narrow, false);` on the raw (not normalized) narrow ACGs gives τ decay, τ rise and R². The fit is sensitive to initialisation, so refitting (or running on another machine) can give slightly different time constants. Use the values in `Cell_metrics.xlsx` to reproduce the paper's classification exactly.
4. **Firing rate and CV2** are computed in `02_trial_info.py`: firing rate = spike count / (last − first spike); CV2 = mean of 2|ISIᵢ₊₁ − ISIᵢ| / (ISIᵢ₊₁ + ISIᵢ) (Holt et al., 1996).
5. **Spike width** (trough to first post-trough peak of the mean waveform) is computed in `REVISIONS/waveform_spike_widths.py`.

`Cell_metrics.xlsx` contains, per unit: firing rate, CV2, raw and normalized mean ACG, τ decay, τ rise, R², burst indices (Royer 2012, doublets), theta modulation index and spike width.

## Cell-type classification

`07_spectral_clustering.py` keeps units with R² > 0.3 and runs spectral clustering on firing rate, raw mean ACG and τ rise; the cluster with the higher firing rate is labelled putative interneurons (IN), the other putative pyramidal cells (PY). Units with R² ≤ 0.3 are labelled by spike width (> 0.425 ms: PY; otherwise IN). The resulting labels are in `REVISIONS/revision_clustering_no_waveform_labels.xlsx` (ACG-based only: 127 PY, 167 IN) and `REVISIONS/revision_clustering_waveform_labels.xlsx` (ACG + spike width: 621 PY, 281 IN).
