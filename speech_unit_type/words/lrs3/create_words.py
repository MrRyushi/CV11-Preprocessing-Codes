import textgrid as tg
import os
import subprocess

os.environ["PATH"] += os.pathsep + os.path.abspath("../ffmpeg-7.0.2-i686-static")
def read_processed_videos(processed_file):
    if not os.path.exists(processed_file):
        with open(processed_file, 'w'): pass

    with open(processed_file, 'r') as f:
        return set(f.read().splitlines())

def add_to_processed_videos(processed_file, video_filename):
    with open(processed_file, 'a') as f:
        f.write(video_filename + '\n')
        
def get_unique_filename(file_path):
    """
    Checks if the file exists and increments the filename with _i if it does.
    """
    base, ext = os.path.splitext(file_path)
    i = 2
    new_file_path = file_path

    while os.path.exists(new_file_path):
        new_file_path = f"{base}_{i}{ext}"
        i += 1
    
    return new_file_path
    
def extract_intervals_and_split_video(timestamp_file, video_file, audio_file, output_video_folder, output_audio_folder, output_ts_folder, subfolder_name):
    base_name = os.path.basename(video_file).split('.')[0]
    grid = tg.TextGrid()
    grid.read(timestamp_file)
    
    for i, interval in enumerate(grid[0]):
        label = interval.mark.strip().upper()
        start_time, end_time = interval.minTime, interval.maxTime
        duration = end_time - start_time

        # word is too short add buffer
        if duration < 0.3:
            start_time -= 0.1
            end_time += 0.3 - duration
        else:
            end_time += 0.25

        if label:  # skip empty intervals
            # extend the end time if the next interval is empty
            if i + 1 < len(grid[0]) and grid[0][i + 1].mark.strip() == "":
                end_time = grid[0][i + 1].maxTime

            # extend the start time if the previous interval is empty
            if i - 1 >= 0 and grid[0][i - 1].mark.strip() == "":
                start_time = grid[0][i - 1].minTime
            
            # make video
            output_video_path = os.path.join(output_video_folder, f"{label}_{subfolder_name}_{base_name}.mp4")
            output_video_path = get_unique_filename(output_video_path)
            subprocess.run([
                "ffmpeg", "-i", video_file, "-ss", str(start_time), "-to", str(end_time),
                "-c:v", "libx264", "-c:a", "aac", '-loglevel', 'quiet', output_video_path
            ], check=True)
            print(f"Saved video for '{label}' as {output_video_path}")

            # make audio
            # output_audio_path = os.path.join(output_audio_folder, f"{label}_{subfolder_name}_{base_name}.wav")
            # output_audio_path = get_unique_filename(output_audio_path)
            # subprocess.run([
            #     "ffmpeg", "-i", audio_file, "-ss", str(start_time), "-to", str(end_time),
            #     "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", '-loglevel', 'quiet', output_audio_path
            # ], check=True)
            #print(f"Saved audio for '{label}' as {output_audio_path}")
            
            # create Text file with the label
            #output_ts_path = os.path.join(output_ts_folder, f"{label}_{subfolder_name}_{base_name}.txt")
            #output_ts_path = get_unique_filename(output_ts_path)
            #with open(output_ts_path, 'w') as text_file:
                #text_file.write(label)  # Write the label into the text file
            #print(f"Saved text file for '{label}' as {output_ts_path}")
            #print(f"Saved video, audio, txt for '{label}' as {output_ts_path}")

# NEW
def adjust_folder_name(subfolder_name):
    return subfolder_name.split('_')[0]

def process_all_videos_in_folder(base_video_folder, base_audio_folder, base_textgrid_folder, output_video_folder, output_audio_folder, output_ts_folder, processed_file):
    processed = 0
    processed_videos = read_processed_videos(processed_file)  # Load already processed videos

    for folder_name in os.listdir(base_video_folder):
        folder_name_wo_utterance_id = adjust_folder_name(folder_name) # NEW
        folder_path = os.path.join(base_video_folder, folder_name)
        
        if os.path.isdir(folder_path):
            for video_filename in os.listdir(folder_path):
                if video_filename.endswith(".mp4"):
                    video_file = os.path.join(folder_path, video_filename)
                    if video_file in processed_videos:
                        print(f"Skipping already processed video: {video_file}")
                        continue
                    
                    # find the TextGrid file by matching the subfolder and video filename
                    #textgrid_file_name = f"{folder_name}_{video_filename.split('.')[0]}.TextGrid"
                    #textgrid_file_name = f"{video_filename.split('.')[0]}.TextGrid" # NEW
                    textgrid_file_name = f"{video_filename.split('.')[0].replace('lrs3', '', 1)}.TextGrid" # NEW 
                    textgrid_file = os.path.join(base_textgrid_folder, textgrid_file_name)

                    audio_file = os.path.join(base_audio_folder, folder_name, f"{video_filename.split('.')[0]}.wav")
                    
                    # check if the transcription file exists
                    if not os.path.exists(textgrid_file):
                        print(f"TextGrid/Timestamp file for '{video_file}' not found. Skipping.\n")
                        continue

                    print(f"Processing {textgrid_file_name} with video {video_filename}...")
                    extract_intervals_and_split_video(textgrid_file, video_file, audio_file, output_video_folder, output_audio_folder, output_ts_folder, folder_name)

                    # keep track of processed videos
                    add_to_processed_videos(processed_file, video_file)
                    processed += 1

    print(f"Processed Videos: {processed}")

#base_video_folder = "../../models/auto_avsr/preparation/preprocessed_lrs3_test_set/lrs3/lrs3_video_seg16s/test"

# LRS3
# base_video_folder = "../../datasets/preprocessed_lrs3_trainval/lrs3/lrs3_video_seg16s/trainval"
# base_audio_folder = "../../datasets/preprocessed_lrs3_trainval/lrs3/lrs3_video_seg16s/trainval"
# base_textgrid_folder = "../mfa_words_timestamps/lrs3_trainval"
# output_video_folder = "../words/lrs3_may/preprocessed_videos_trainval"
# output_audio_folder = "../words/lrs3_may/audio_trainval"
# output_ts_folder = "../words/lrs3_may/transcripts_trainval"
# processed_file = "../words/lrs3_may/processed_videos_trainval.txt"

base_video_folder = "../../datasets/preprocessed_synthetic_front/lrs3/lrs3_video_seg16s/test"
base_audio_folder = "../../datasets/preprocessed_synthetic_front/lrs3/lrs3_video_seg16s/test"
base_textgrid_folder = "../mfa_words_timestamps/lrs3_test"
output_video_folder = "../../datasets/lrs3_task4/lrs3_preprocessed_synthetic_front_dctcn"
output_audio_folder = "../../datasets/lrs3_task4/lrs3_preprocessed_synthetic_front_dctcn"
output_ts_folder = "../../datasets/lrs3_task4/lrs3_task4/lrs3_preprocessed_synthetic_front_dctcn"
processed_file = "../../datasets/lrs3_task4/lrs3_preprocessed_synthetic_front_dctcn/processed_vids_test.txt"


# Create output directories if they don't exist
os.makedirs(output_video_folder, exist_ok=True)
os.makedirs(output_audio_folder, exist_ok=True)
os.makedirs(output_ts_folder, exist_ok=True)

process_all_videos_in_folder(base_video_folder, base_audio_folder, base_textgrid_folder, output_video_folder, output_audio_folder, output_ts_folder, processed_file)

print("Done!")
