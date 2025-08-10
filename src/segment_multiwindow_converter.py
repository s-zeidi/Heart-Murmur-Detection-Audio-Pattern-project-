import os
import numpy as np
import pandas as pd
import librosa


def convert_to_multi_resolution_segments(
        df,
        split_name="train",  # <<< NEW: "train", "val", or "test"
        base_output_dir="../data/generated_data/multi_resolution_segments_3s/",
        audio_column="file_path",
        label_column="murmur_label",
        subject_column="subject_id",
        sr=4000,
        segment_seconds=3,
        overlap_seconds=1,
        n_mels=128,
        fft_size=512):

    # Build folder and file names based on split
    output_dir = os.path.join(base_output_dir, f"{split_name}_spectrograms({n_mels})")
    output_csv = os.path.join(base_output_dir, f"{split_name}_segment_data({n_mels}).csv")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Define spectrogram resolutions (in seconds)
    multi_window_settings = [
        {"frame_len": 0.025, "frame_hop": 0.010},
        {"frame_len": 0.050, "frame_hop": 0.025},
        {"frame_len": 0.100, "frame_hop": 0.050},
    ]

    segment_len = int(segment_seconds * sr)
    hop_len = int((segment_seconds - overlap_seconds) * sr)

    segment_entries = []

    for _, row in df.iterrows():
        file_path = row[audio_column]
        subject_id = row[subject_column]
        label = row[label_column]
        base = os.path.splitext(os.path.basename(file_path))[0]

        try:
            y, _ = librosa.load(file_path, sr=sr)

            for i, start in enumerate(range(0, len(y) - segment_len + 1, hop_len)):
                end = start + segment_len
                segment = y[start:end]

                multi_specs = []

                for win_cfg in multi_window_settings:
                    mel_win = int(win_cfg["frame_len"] * sr)
                    mel_hop = int(win_cfg["frame_hop"] * sr)

                    mel_spec = librosa.feature.melspectrogram(
                        y=segment,
                        sr=sr,
                        n_fft=fft_size,
                        hop_length=mel_hop,
                        win_length=mel_win,
                        n_mels=n_mels,
                        fmax=sr // 2
                    )
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    multi_specs.append(mel_spec_db)

                max_time = max(spec.shape[1] for spec in multi_specs)
                padded_specs = [
                    np.pad(spec, ((0, 0), (0, max_time - spec.shape[1])), mode='constant')
                    for spec in multi_specs
                ]
                stacked = np.stack(padded_specs)  # Shape: (3, n_mels, time)

                seg_name = f"{split_name}_{base}_seg{i}_multi_res.npy"
                seg_path = os.path.join(output_dir, seg_name)
                np.save(seg_path, stacked)

                segment_entries.append({
                    "subject_id": subject_id,
                    "murmur_label": label,
                    "segment_path": seg_path,
                    "segment_start": start / sr
                })

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue

    segment_df = pd.DataFrame(segment_entries)
    segment_df.to_csv(output_csv, index=False)

    print(f"✅ [{split_name.upper()}] Multi-resolution 3s segments saved to: {output_csv}")
    return segment_df
