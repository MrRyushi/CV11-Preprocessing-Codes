# pip install textgrid
# pip install ffmpeg
# pip install ffmpeg-python
import textgrid as tg
import os
import subprocess

os.environ["PATH"] += os.pathsep + os.path.abspath("../ffmpeg-7.0.2-i686-static")

def read_processed_videos(processed_file):
    # create the file if it doesn't exist
    if not os.path.exists(processed_file):
        with open(processed_file, 'w'): pass

    # read the processed videos
    with open(processed_file, 'r') as f:
        return set(f.read().splitlines())

def add_to_processed_videos(processed_file, video_filename):
    with open(processed_file, 'a') as f:
        f.write(video_filename + '\n')

def add_to_failed_videos(processed_file, video_filename):
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
    
def extract_intervals_and_split_video(timestamp_file, video_file, audio_file, output_video_folder, output_audio_folder, output_ts_folder):
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
            output_video_path = os.path.join(output_video_folder, f"{label}_{base_name}.mp4")
            output_video_path = get_unique_filename(output_video_path)
            subprocess.run([
                "ffmpeg", "-i", video_file, "-ss", str(start_time), "-to", str(end_time),
                "-c:v", "libx264", "-c:a", "aac", '-loglevel', 'quiet', output_video_path
            ], check=True)
            #print(f"Saved video for '{label}' as {output_video_path}")

            # make audio
            output_audio_path = os.path.join(output_audio_folder, f"{label}_{base_name}.wav")
            output_audio_path = get_unique_filename(output_audio_path)
            subprocess.run([
                "ffmpeg", "-i", audio_file, "-ss", str(start_time), "-to", str(end_time),
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", '-loglevel', 'quiet', output_audio_path
            ], check=True)
            #print(f"Saved audio for '{label}' as {output_audio_path}")

            # create Text file with the label
            '''
            output_ts_path = os.path.join(output_ts_folder, f"{label}_{base_name}.txt")
            output_ts_path = get_unique_filename(output_ts_path)
            with open(output_ts_path, 'w') as text_file:
                text_file.write(label)  # Write the label into the text file
            #print(f"Saved text file for '{label}' as {output_ts_path}")
            '''
            print(f"Saved video, audio, txt for '{label}' as {output_video_path}")

def process_all_videos_in_folder(video_folder, audio_folder, textgrid_folder, output_video_folder, output_audio_folder, output_ts_folder, processed_file, failed_file):
    processed = 0
    #valid_prefixes = ["s6", "s8", "s9", "s15", "s26", "s30", "s34", "s43", "s44", "s49", "s51", "s52"]
    processed_videos = read_processed_videos(processed_file)  # Load already processed videos
    
    for video_filename in os.listdir(video_folder):
        # the _v3, _v4 is added for non frontal videos only
        if video_filename.endswith(".mp4") and any(v in video_filename for v in ["_v3", "_v4", "_v5"]): #and any(video_filename.startswith(prefix) for prefix in valid_prefixes):
            if video_filename in processed_videos:
                print(f"Skipping already processed video: {video_filename}")
                continue
                
            video_file = os.path.join(video_folder, video_filename)
            
            # for test set
            textgrid_file_name = video_filename.replace('_v1', '').replace('_v2', '').replace('_v3', '').replace('_v4', '').replace('_v5', '').replace('.mp4', '.TextGrid')
            
            # for trainval set
            #textgrid_file_name = video_filename.replace('.mp4', '.TextGrid')
            
            textgrid_file = os.path.join(textgrid_folder, textgrid_file_name)
            
            audio_file = os.path.join(audio_folder, video_filename.replace('.mp4', '.wav'))

            # check if the transcription file exists
            if not os.path.exists(textgrid_file):
                print(f"TextGrid/Timestamp file for '{video_file}' not found. Skipping.")
                add_to_failed_videos(failed_file, video_filename)
                continue

            print(f"Processing {textgrid_file} with video {video_filename}...")
            extract_intervals_and_split_video(textgrid_file, video_file, audio_file, output_video_folder, output_audio_folder, output_ts_folder)

            # keep track of processed videos
            add_to_processed_videos(processed_file, video_filename)
            processed += 1
            print(f"Current Processed Videos: {processed}")

    print(f"Processed Videos: {processed}")

'''
# trainval
video_folder = "../../datasets/ouluvs2/trainval"
audio_folder = "../../datasets/ouluvs2/trainval"
textgrid_folder = "../mfa_words_timestamps/ouluvs2_trainval"
output_video_folder = "../../datasets/ouluvs2_task4/speech_unit_type/words/trainval_v2"
output_audio_folder = "../../datasets/ouluvs2_task4/speech_unit_type/words/trainval_v2"
output_ts_folder = "../../datasets/ouluvs2_task4/speech_unit_type/words/trainval_v2"
processed_file = "../../datasets/ouluvs2_task4/speech_unit_type/words/processed_videos_trainval_v2.txt"
failed_file = "../../datasets/ouluvs2_task4/speech_unit_type/words/failed_videos_trainval_v2.txt"
'''
# test
# video_folder = "../../datasets/preprocessed_ouluvs2_test2/ouluvs2/ouluvs2_video_seg16s"
# audio_folder = "../../datasets/preprocessed_ouluvs2_test2/ouluvs2/ouluvs2_video_seg16s"
# textgrid_folder = "../mfa_words_timestamps/ouluvs2_test"
# output_video_folder = "../../datasets/ouluvs2_task4/speech_unit_type/preprocessed_words_v2/ouluvs2/ouluvs2_video_seg16s"
# output_audio_folder = "../../datasets/ouluvs2_task4/speech_unit_type/preprocessed_words_v2/ouluvs2/ouluvs2_video_seg16s"
# output_ts_folder = "../../datasets/ouluvs2_task4/speech_unit_type/preprocessed_words_v2/ouluvs2/ouluvs2_text_seg16"
# processed_file = "../../datasets/ouluvs2_task4/speech_unit_type/preprocessed_words_v2/processed_videos_test_v2.txt"
# failed_file = "../../datasets/ouluvs2_task4/speech_unit_type/preprocessed_words_v2/failed_videos_test_v2.txt"

os.makedirs(output_video_folder, exist_ok=True)
os.makedirs(output_audio_folder, exist_ok=True)
os.makedirs(output_ts_folder, exist_ok=True)

process_all_videos_in_folder(video_folder, audio_folder, textgrid_folder, output_video_folder, output_audio_folder, output_ts_folder, processed_file, failed_file)

print("Done!")