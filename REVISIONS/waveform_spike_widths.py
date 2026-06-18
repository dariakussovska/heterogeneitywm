import h5py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

WIDTH_THRESHOLD_MS = 0.43


def calculate_spike_width(avg_waveform, time_per_sample_ms):
    """
    Spike width = time from waveform trough to first post-trough peak.
    """

    trough_idx = np.argmin(avg_waveform)

    if trough_idx >= len(avg_waveform) - 5:
        return np.nan

    post_trough = avg_waveform[trough_idx:]

    peaks, _ = find_peaks(
        post_trough,
        prominence=0.05 * np.ptp(avg_waveform),
        distance=2,
        width=1
    )

    if len(peaks) == 0:
        return np.nan

    first_peak_idx = trough_idx + peaks[0]
    width_samples = first_peak_idx - trough_idx
    width_ms = width_samples * time_per_sample_ms

    return width_ms


def analyze_subject(filepath, subject_id):
    results = []

    with h5py.File(filepath, "r") as f:
        units_group = f["units"]

        waveforms_data = units_group["waveforms"][:]
        waveforms_index = units_group["waveforms_index"][:]

        if "sampling_rate" in units_group:
            sampling_rate = units_group["sampling_rate"][()]
            time_per_sample_ms = 1000.0 / sampling_rate
        else:
            time_per_sample_ms = 0.01

        start_idx = 0

        for neuron_id, end_idx in enumerate(waveforms_index, start=1):
            neuron_spikes = waveforms_data[start_idx:end_idx, :]
            avg_waveform = np.mean(neuron_spikes, axis=0)

            if np.max(avg_waveform) > abs(np.min(avg_waveform)):
                avg_waveform = -avg_waveform

            width_ms = calculate_spike_width(avg_waveform, time_per_sample_ms)

            results.append({
                "subject_id": subject_id,
                "neuron_id": neuron_id,
                "full_id": f"{subject_id}0{neuron_id}",
                "n_spikes": neuron_spikes.shape[0],
                "spike_width_ms": width_ms,
                "width_below_0.43ms": (
                    not np.isnan(width_ms) and width_ms < WIDTH_THRESHOLD_MS
                )
            })

            start_idx = end_idx

    return pd.DataFrame(results)


def main():
    filepaths = [
        f"/Users/darikussovska/Desktop/PROJECT/000469/sub-{i}/sub-{i}_ses-2_ecephys+image.nwb"
        for i in range(1, 22)
    ]

    all_subjects = []

    for subject_id, filepath in enumerate(filepaths, start=1):
        try:
            df = analyze_subject(filepath, subject_id)
            all_subjects.append(df)

        except FileNotFoundError:
            print(f"File not found for subject {subject_id}")

        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")

    combined_df = pd.concat(all_subjects, ignore_index=True)

    return combined_df

def main():
    filepaths = [
        f"/Users/darikussovska/Desktop/PROJECT/000469/sub-{i}/sub-{i}_ses-2_ecephys+image.nwb"
        for i in range(1, 22)
    ]

    all_subjects = []

    for subject_id, filepath in enumerate(filepaths, start=1):
        try:
            print(f"\nProcessing Subject {subject_id}")

            df = analyze_subject(filepath, subject_id)
            all_subjects.append(df)

            plot_all_waveforms_grid(df, subject_id)
            plot_spike_width_histogram(df, subject_id)

        except FileNotFoundError:
            print(f"File not found for subject {subject_id}")

        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")

    combined_df = pd.concat(all_subjects, ignore_index=True)

    output_df = combined_df[
        ["subject_id", "neuron_id", "spike_width_ms"]
    ].rename(columns={"neuron_id": "Neuron_ID"})

    output_path = "./spike_widths_all_subjects.xlsx"
    output_df.to_excel(output_path, index=False)

    print(f"\nSaved: {output_path}")

    return output_df

import matplotlib.pyplot as plt
import numpy as np


WIDTH_THRESHOLD_MS = 0.43


def plot_all_waveforms_grid(df, subject_id):
    n_neurons = len(df)
    n_cols = min(10, int(np.ceil(np.sqrt(n_neurons))))
    n_rows = int(np.ceil(n_neurons / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2, n_rows * 1.5)
    )

    fig.suptitle(
        f"Subject {subject_id}: Individual Waveforms\n"
        f"Threshold = {WIDTH_THRESHOLD_MS} ms",
        fontsize=14,
        y=1.02
    )

    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < n_neurons:
            row = df.iloc[i]

            waveform = row["avg_waveform"]
            time_axis = row["time_axis_ms"]
            width = row["spike_width_ms"]

            is_below_threshold = (
                not np.isnan(width) and width < WIDTH_THRESHOLD_MS
            )

            color = "red" if is_below_threshold else "black"

            ax.plot(time_axis, waveform, color=color, linewidth=1.2)

            title = f"N{int(row['neuron_id'])}"
            if not np.isnan(width):
                title += f"\nW={width:.3f} ms"
            else:
                title += "\nW=NaN"

            ax.set_title(title, fontsize=8, color=color)
            ax.set_xlabel("Time (ms)", fontsize=7)
            ax.set_ylabel("Amplitude", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)

        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_spike_width_histogram(df, subject_id):
    widths = df["spike_width_ms"].dropna().values

    plt.figure(figsize=(7, 5))
    plt.hist(widths, bins=30, edgecolor="black", alpha=0.75)
    plt.axvline(
        WIDTH_THRESHOLD_MS,
        linestyle="--",
        linewidth=2,
        label=f"{WIDTH_THRESHOLD_MS} ms threshold"
    )

    plt.xlabel("Spike Width (ms)")
    plt.ylabel("Number of Neurons")
    plt.title(
        f"Subject {subject_id}: Spike Width Distribution\n"
        f"n={len(widths)}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    spike_width_table = main()
