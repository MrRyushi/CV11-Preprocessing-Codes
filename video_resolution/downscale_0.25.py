import cv2
import os

# Input and output directories (using f-strings for dynamic modification)
input_directory = './ouluvs2/Non Frontal'  # Change this to your input folder path
base_output_directory = '25'  # Base name for the output directory

# Check if the folder name starts with 'orig_'
if input_directory.startswith('orig_'):
    # Extract the folder name after 'orig_'
    folder_name = input_directory[5:]
    output_directory = os.path.join(base_output_directory, folder_name)
else:
    # If it doesn't start with 'orig_', use the base name directly
    output_directory = base_output_directory

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# List all video files in the input directory
video_files = [f for f in os.listdir(input_directory) if f.endswith(('.mp4', '.avi', '.mov'))]
if not video_files:
    print("No video files found in the input directory.")
else:
    print(f"Found {len(video_files)} video files: {video_files}")

# Loop through all video files
for video_file in video_files:
    input_video_path = os.path.join(input_directory, video_file)
    output_video_path = os.path.join(output_directory, f'downscaled_{video_file}')

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        continue  # Skip this file and move to the next one

    # Get original video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the video
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the video

    # Calculate new dimensions for downscaling by 25%
    new_width = int(width * 0.25)
    new_height = int(height * 0.25)

    # Create a VideoWriter object to save the downscaled video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if there are no frames left

        # Downscale the frame by 25%
        downscaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # Write the downscaled frame to the output video
        out.write(downscaled_frame)

    # Release resources
    cap.release()
    out.release()
    print(f'Downscaled video saved as {output_video_path}.')