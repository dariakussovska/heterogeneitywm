Autocorrelograms and ACG metrics were computed with CellExplorer (Petersen et al., 2021, Neuron 109:3594–3608; https://cellexplorer.org). 

To recompute them, install CellExplorer and add it to the MATLAB path; calc_ACG_metrics.m and fit_ACG.m are part of CellExplorer. 
1. Load spikes_struct.mat and run
'''acg_metrics = calc_ACG_metrics(spikes, sr)''',
with sr the sampling rate of the recordings, so that spike times in seconds are binned correctly.

CellExplorer's CCG function is a compiled MEX file; on macOS it may need to be recompiled or removed from quarantine (xattr -c). The narrow (±50 ms, 0.5 ms bins) and wide (±500 ms, 1 ms bins) ACGs used in the paper are provided in acg.mat (acg.narrow, acg.wide; one column per neuron, in the order of spikes_struct.mat). 

2. The triple-exponential fits (fit_ACG.m) were run on the raw narrow ACGs; the resulting τ decay, τ rise and R², together with firing rate, CV2 (Holt et al., 1996), raw and normalized mean ACG and spike width, are in Cell_metrics.xlsx.

Because this fit is sensitive to initialisation, refitting can give slightly different time constants; use the values in Cell_metrics.xlsx to reproduce the paper's classification exactly.
