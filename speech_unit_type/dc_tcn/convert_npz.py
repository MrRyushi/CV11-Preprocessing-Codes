import cv2
import os
import numpy as np


#lrs2
'''
input_folder = '../words/lrs2_may/preprocessed_videos_all'
output_folder = '../../models/DC-TCN/datasets/lrs2_words_may'
'''
# lrs3
# input_folder = '../words/lrs3_may/preprocessed_videos_all'
# output_folder = '../../models/DC-TCN/datasets/lrs3_words_may'

# lrs3 face pose transformed
# input_folder = '../../datasets/lrs3_task4/preprocessed_videos_test'
# output_folder = '../../datasets/lrs3_task4/words_lrs3_transformed'
input_folder = '../../datasets/lrs3_task4/lrs3_raw_phoamb/test_dctcn'
output_folder = '../../datasets/lrs3_task4/lrs3_raw_phoamb/test_dctcn_npz'

# def get_video_duration(cap):
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     return frame_count / fps

def get_video_duration(fps, frame_count):
    if fps > 0 and frame_count > 0:
        return frame_count / fps
    return None

for root, dirs, files in os.walk(input_folder):
    for dir_name in dirs:
        word_folder = os.path.join(root, dir_name)
        
        # check for all subfolders
        for subfolder in ['test']: # val test
            subfolder_path = os.path.join(word_folder, subfolder)
            
            if not os.path.exists(subfolder_path):
                continue
            
            word_output_folder = os.path.join(output_folder, dir_name, subfolder)
            os.makedirs(word_output_folder, exist_ok=True)
            
            for file in os.listdir(subfolder_path):
                if file.endswith('.mp4'):
                    video_filename = os.path.join(subfolder_path, file)
                    video_output_filename = os.path.join(word_output_folder, file[:-4] + '.npz')
                    txt_file_path = video_output_filename.replace('.npz', '.txt')
                    
                    # Skip if already processed
                    if os.path.exists(video_output_filename):
                        print(f"Skipping already processed: {video_output_filename}")
                        continue
                        
                    cap = cv2.VideoCapture(video_filename)
                    
                    if not cap.isOpened():
                        print(f"Error: Unable to open video file {video_filename}.")
                        continue
                        
                    # list to store frames
                    frames = []
                    
                    # read frames from the video
                    while cap.isOpened(): # while ret:
                        ret, frame = cap.read()
                        if not ret:
                            break  # end of video
                        
                        # convert the frame to grayscale
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        
                        # append the frame to the list
                        frames.append(frame_gray)

                    # NEW: Read FPS and frame count before releasing cap
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # release the video capture object
                    cap.release()
                    
                    # convert the list of frames to a NumPy array
                    frames_array = np.array(frames)
                    
                    # save the frames as a .npz file
                    np.savez_compressed(video_output_filename, frames=frames_array)

                    # Skip if already processed
                    if os.path.exists(txt_file_path):
                        print(f"Skipping already processed: {txt_file_path}")
                        continue

                    # duration = get_video_duration(cap)
                    duration = get_video_duration(fps, frame_count)
                    
                    if duration is not None:
                        with open(txt_file_path, 'w') as f:
                            f.write(f"Duration: {duration:.2f} seconds\n")
                        print(f"Saved duration to {txt_file_path}")
                    
                    print(f"Video frames saved to {video_output_filename}.")

print("All videos processed.")
