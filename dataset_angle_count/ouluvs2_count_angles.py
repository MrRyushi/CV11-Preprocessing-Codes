import os
import zipfile
from collections import Counter

# Define the folder containing the zip files
zip_folder = "./OuluVS2/OuluVS2/OuluVS2-zip"

# List of indicated speakers
indicated_speakers = {1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 45, 46, 47, 48, 50, 53}

# Initialize a counter for angle occurrences
angle_count = Counter()

# Loop through all zip files in the folder
for zip_filename in os.listdir(zip_folder):
    if zip_filename.startswith("orig_s") and zip_filename.endswith(".zip"):
        speaker_id = zip_filename.split("_s")[-1].split(".zip")[0]

        # Check if speaker ID is in the indicated list
        if speaker_id.isdigit() and int(speaker_id) in indicated_speakers:
            zip_path = os.path.join(zip_folder, zip_filename)

            # Open the ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                for file_name in zip_file.namelist():
                    # Extract the actual filename, ignoring the outer folder
                    base_name = os.path.basename(file_name)  # Get the last part of the path
                    parts = base_name.split("_")

                    # Ensure the filename matches "s<id>_v<angle>_u<utterance>.mp4"
                    if len(parts) >= 3 and parts[1].startswith("v") and base_name.endswith(".mp4"):
                        angle_id = parts[1][1:]  # Extract number after 'v'
                        if angle_id in {"1", "2", "3", "4", "5"}:  # Only count v1 to v5
                            angle_count[angle_id] += 1

# Print the results
print("Total count per angle:")
for angle in sorted(angle_count.keys(), key=int):
    print(f"v{angle}: {angle_count[angle]}")
