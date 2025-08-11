import os
import pandas as pd
import shutil

def update_test_csv_with_new_paths(
    test_csv_path,
    source_folder,
    output_folder,
    new_csv_path
):
    # Load CSV
    df = pd.read_csv(test_csv_path)

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all filenames from the source folder
    source_files = os.listdir(source_folder)

    # New segment paths to store
    new_paths = []

    print(f"\n🔍 Processing test CSV: {test_csv_path}")
    print(f"📁 Looking in source folder: {source_folder}")
    print(f"📤 Will save matched files to: {output_folder}\n")

    for i, row in df.iterrows():
        original_segment_path = row['segment_path']
        base_filename = os.path.basename(original_segment_path).replace(".npy", "")  # e.g., 84755_MV_seg3
        match_filename = f"{base_filename}_multi_res.npy"

        print(f"📌 Row {i+1}")
        print(f"  - Input path from CSV      : {original_segment_path}")
        print(f"  - Target filename to find  : {match_filename}")

        if match_filename in source_files:
            src_path = os.path.join(source_folder, match_filename)
            dst_path = os.path.join(output_folder, match_filename)

            # Copy file to new location
            shutil.copy2(src_path, dst_path)

            # Save the new full path in DataFrame
            new_paths.append(dst_path)

            print(f"  ✅ Found: {src_path}")
            print(f"  📥 Copied to: {dst_path}\n")
        else:
            new_paths.append("")
            print(f"  ❌ Not found: {match_filename} in {source_folder}\n")

    # Update CSV column
    df['segment_path'] = new_paths

    # Save the updated CSV
    df.to_csv(new_csv_path, index=False)
    print(f"✅ Updated CSV saved to: {new_csv_path}\n")

# ===== Example Usage =====
test_csv_path = "../data/generated_data/patient-dependent/val_random.csv"
source_folder = "../data/generated_data/patient-dependent/multi_resolution_segments_3s/total_spectrograms(128)"
output_folder = "../data/generated_data/patient-dependent/multi_resolution_segments_3s/val_spectrograms(128)"
new_csv_path = "../data/generated_data/patient-dependent/multi_resolution_segments_3s/val_segmented_multi.csv"

update_test_csv_with_new_paths(test_csv_path, source_folder, output_folder, new_csv_path)
