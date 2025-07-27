import os
import cv2

def extract_frames(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    
    for subfolder in os.listdir(input_dir):
        subfolder_path = os.path.join(input_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.endswith(".mp4"):  # Process only video files
                    video_path = os.path.join(subfolder_path, file)
                    text_file = video_path.replace(".mp4", ".txt")
                    
                    vid_name = os.path.splitext(file)[0]  # Get video name without extension
                    output_folder = os.path.join(output_dir, f"{subfolder}_{vid_name}")
                    source_folder = os.path.join(output_folder, "source")  # Create 'source' folder inside each video folder
                    os.makedirs(source_folder, exist_ok=True)
                    
                    # Copy the corresponding text file
                    if os.path.exists(text_file):
                        new_text_file = os.path.join(output_folder, f"{subfolder}_{vid_name}.txt")
                        os.system(f'cp "{text_file}" "{new_text_file}"')
                    
                    # Extract frames
                    cap = cv2.VideoCapture(video_path)
                    frame_count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_filename = os.path.join(source_folder, f"{frame_count:05d}.png")
                        cv2.imwrite(frame_filename, frame)
                        frame_count += 1
                    cap.release()
                    print(f"Extracted {frame_count} frames from {video_path}")

if __name__ == "__main__":
    input_directory = "../datasets/lrs3_test_set"
    output_directory = "../models/MonoNPHM/tracking_input_lrs3"
    extract_frames(input_directory, output_directory)