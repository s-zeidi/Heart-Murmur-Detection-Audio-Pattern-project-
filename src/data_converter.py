# data_converter.py

import os
import numpy as np
import pandas as pd
import librosa

def convert_to_full_file_spectrograms(input_csv="../data/generated_data/processed_data.csv",
                                       output_csv="../data/generated_data/spectrogram_data.csv",
                                       output_dir="../data/generated_data/processed_spectrograms",
                                       sr=4000, n_mels=128, fmax=2000):
    """
    Converts each full .wav file to a Mel spectrogram and saves as .npy.
    Updates the dataset with spectrogram paths and returns the DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    def process_audio(file_path):
        try:
            y, _ = librosa.load(file_path, sr=sr)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            filename = os.path.basename(file_path).replace(".wav", ".npy")
            save_path = os.path.join(output_dir, filename)
            np.save(save_path, mel_spec_db)

            return save_path
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    df["spectrogram_path"] = df["file_path"].apply(process_audio)
    df = df.dropna(subset=["spectrogram_path"])
    df.to_csv(output_csv, index=False)

    print("✅ All audio files have been converted to full-length spectrograms and saved.")
    return df

def convert_to_segment_level(df,
                              audio_column="file_path",
                              label_column="murmur_label",
                              subject_column="subject_id",
                              output_dir="../data/generated_data/segmented_spectrograms",
                              output_csv="../data/generated_data/segment_level_data.csv",
                              sr=4000,
                              segment_seconds=3,
                              overlap_seconds=1,
                              n_mels=128,
                              fft_size=512,
                              frame_len=0.025,
                              frame_hop=0.0125):
    """
    Segments each .wav file into overlapping windows, extracts Mel spectrograms,
    saves each segment as a .npy file, and saves metadata to a CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    segment_len = int(segment_seconds * sr)
    hop_len = int((segment_seconds - overlap_seconds) * sr)
    mel_win = int(frame_len * sr)
    mel_hop = int(frame_hop * sr)

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

                seg_name = f"{base}_seg{i}.npy"
                seg_path = os.path.join(output_dir, seg_name)
                np.save(seg_path, mel_spec_db)

                segment_entries.append({
                    "subject_id": subject_id,
                    "murmur_label": label,
                    "segment_path": seg_path
                })

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue

    segment_df = pd.DataFrame(segment_entries)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    segment_df.to_csv(output_csv, index=False)

    print(f"✅ Segment-level spectrograms created and saved to: {output_csv}")
    return segment_df
