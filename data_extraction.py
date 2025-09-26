import re, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pynwb import NWBHDF5IO

class NWBProcessor:
    def __init__(self, output_dir="/home/daria/PROJECT"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def extract_subject_id(self, file_path):
        m = re.search(r'sub-(\d+)', str(file_path))
        return int(m.group(1)) if m else None

    def get_image_order(self, nwbfile):
        stim_templates = nwbfile.stimulus_template['StimulusTemplates']
        sorted_keys = sorted(stim_templates.images.keys())
        # (optional) plotting omitted for brevity
        return {key: rank for rank, key in enumerate(sorted_keys)}

    def process_encoding_periods(self, nwbfile, image_order, subject_id):
        stim_pres = nwbfile.stimulus['StimulusPresentation']
        trial_data = nwbfile.intervals['trials']
        results = {1: [], 2: [], 3: []}

        for i in range(len(trial_data)):
            idx_base = i * 4
            for enc_num in [1, 2, 3]:
                stim_idx = stim_pres.data[idx_base + enc_num - 1]
                image_id = list(image_order.keys())[stim_idx]
                start_time = trial_data[f'timestamps_Encoding{enc_num}'][i]
                stop_time  = trial_data[f'timestamps_Encoding{enc_num}_end'][i]
                results[enc_num].append({
                    'subject_id': subject_id,
                    'trial_id': i + 1,
                    'start_time': start_time,
                    'stop_time':  stop_time,
                    'image_id': image_id,
                    'stimulus_index': stim_idx,
                    'image_rank': image_order[image_id],
                })

        return (pd.DataFrame(results[1]), pd.DataFrame(results[2]), pd.DataFrame(results[3]))

    def get_fixation_periods(self, nwbfile):
        events = nwbfile.acquisition['events']
        t, d = events.timestamps[:], events.data[:]
        out, trial_id = [], 0
        for i in range(len(d) - 1):
            if d[i] == 11 and d[i+1] == 1:
                trial_id += 1
                out.append({'start_time': t[i], 'end_time': t[i+1], 'trial_id': trial_id})
        return out

    def get_delay_periods(self, nwbfile):
        events = nwbfile.acquisition['events']
        t, d = events.timestamps[:], events.data[:]
        out, trial_id = [], 0
        for i in range(len(d) - 1):
            if d[i] == 6 and d[i+1] == 7:
                trial_id += 1
                out.append({'start_time': t[i], 'end_time': t[i+1], 'trial_id': trial_id})
        return out

    def get_probe_periods_with_trials(self, nwbfile):
        """Add Probe_Image_ID (and keep image_id for consistency)."""
        events = nwbfile.acquisition['events']
        t, d = events.timestamps[:], events.data[:]

        trials = nwbfile.intervals['trials']
        probe_in_out = trials['probe_in_out'].data[:]
        resp_acc     = trials['response_accuracy'].data[:]
        probe_ids    = trials['loadsProbe_PicIDs'].data[:]
        t_start      = trials['start_time'].data[:]
        t_stop       = trials['stop_time'].data[:]

        def to_scalar(x):
            if isinstance(x, (list, tuple, np.ndarray)): return x[0] if len(x) else np.nan
            return x

        out = []
        for trial_id, (s, e, pid, pin, acc) in enumerate(zip(t_start, t_stop, probe_ids, probe_in_out, resp_acc)):
            pid = to_scalar(pid); pin = to_scalar(pin); acc = to_scalar(acc)
            within = (t > s) & (t < e)
            idxs = np.where((d[:-1] == 7) & (d[1:] == 8) & within[:-1])[0]
            for idx in idxs:
                out.append({
                    'start_time': t[idx],
                    'end_time':   t[idx+1],
                    'trial_id':   trial_id + 1,
                    'image_id':   pid,
                    'Probe_Image_ID': pid,
                    'probe_in_out': pin,
                    'response_accuracy': acc
                })
        return out

    def calculate_spike_rates(self, nwbfile, period_data, subject_id, period_name):
        if not period_data:
            print(f"No {period_name} periods found.")
            return pd.DataFrame()

        all_rows = []
        n_units = len(nwbfile.units['spike_times'])
        for neuron_id in range(n_units):
            spikes_all = nwbfile.units['spike_times'][neuron_id]
            for p in period_data:
                s, e = p['start_time'], p['end_time']
                dur = e - s
                if dur > 0:
                    spikes = [sp for sp in spikes_all if s < sp < e]
                    rate = len(spikes) / dur
                else:
                    spikes, rate = [], 0.0
                row = {
                    'subject_id': subject_id,
                    'Neuron_ID': neuron_id,
                    'trial_id': p['trial_id'],
                    f'{period_name}_Start': s,
                    f'{period_name}_End':   e,
                    f'Spikes_in_{period_name}': [spikes],
                    f'Spikes_rate_{period_name}': rate,
                }
                for key in ['image_id', 'Probe_Image_ID', 'probe_in_out', 'response_accuracy']:
                    if key in p: row[key] = p[key]
                all_rows.append(row)
        return pd.DataFrame(all_rows)

    def add_spike_rates_to_encoding(self, nwbfile, df_encoding, period_name):
        n_units = len(nwbfile.units['spike_times'])
        rows = []
        for neuron_id in range(n_units):
            spikes_all = nwbfile.units['spike_times'][neuron_id]
            for _, r in df_encoding.iterrows():
                s, e = r['start_time'], r['stop_time']
                if pd.isna(s) or pd.isna(e) or (e - s) <= 0:
                    spikes, rate = [], 0.0
                else:
                    spikes = [sp for sp in spikes_all if s <= sp < e]
                    rate = len(spikes) / (e - s)
                row = r.to_dict()
                row.update({'Neuron_ID': neuron_id, 'Spikes': [spikes], f'Spikes_rate_{period_name}': rate})
                rows.append(row)
        return pd.DataFrame(rows)

    def process_single_file(self, filepath):
        print(f"Processing: {filepath}")
        with NWBHDF5IO(filepath, 'r', load_namespaces=True) as io:
            nwb = io.read()
            subj = self.extract_subject_id(filepath)
            image_order = self.get_image_order(nwb)

            df1, df2, df3 = self.process_encoding_periods(nwb, image_order, subj)
            df1 = self.add_spike_rates_to_encoding(nwb, df1, 'Encoding1')
            df2 = self.add_spike_rates_to_encoding(nwb, df2, 'Encoding2')
            df3 = self.add_spike_rates_to_encoding(nwb, df3, 'Encoding3')

            fix = self.calculate_spike_rates(nwb, self.get_fixation_periods(nwb), subj, 'Fixation')
            dly = self.calculate_spike_rates(nwb, self.get_delay_periods(nwb), subj, 'Delay')
            prb = self.calculate_spike_rates(nwb, self.get_probe_periods_with_trials(nwb), subj, 'Probe')

            return {'encoding1': df1, 'encoding2': df2, 'encoding3': df3,
                    'fixation': fix, 'delay': dly, 'probe': prb}

    def process_all_files(self, filepaths):
        buckets = {k: [] for k in ['encoding1','encoding2','encoding3','fixation','delay','probe']}
        for fp in filepaths:
            fp = Path(fp)
            if not fp.exists():
                print(f"✗ File not found: {fp}")
                continue
            try:
                res = self.process_single_file(fp)
                for k, df in res.items():
                    if not df.empty:
                        buckets[k].append(df)
                print(f"✓ Successfully processed {fp.name}")
            except Exception as e:
                print(f"✗ Error processing {fp.name}: {e}")

        for k, lst in buckets.items():
            if lst:
                out = pd.concat(lst, ignore_index=True)
                out_path = self.output_dir / f"all_spike_rate_data_{k}.xlsx"
                out.to_excel(out_path, index=False)
                print(f"✓ Saved {k} data: {out_path}")

