import os
import shutil
def rename_npy_files(input_dir, output_dir, prefix_to_remove="train_"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.npy') and filename.startswith(prefix_to_remove):
            # Remove the prefix
            new_filename = filename[len(prefix_to_remove):]

            # Full paths
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, new_filename)

            # Copy file with new name
            shutil.copy2(input_path, output_path)

            # Print what’s happening
            print(f"Original: {filename} -> Renamed: {new_filename}")

# Example usage
input_directory = "../data/generated_data/multi_resolution_segments_3s/train_spectrograms(128)"
output_directory = "../data/generated_data/patient-dependent/multi_resolution_segments_3s/total_spectrograms(128)"
rename_npy_files(input_directory, output_directory)