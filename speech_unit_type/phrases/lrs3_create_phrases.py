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

def extract_phrases_and_split_video(output_subfolder, subfolder_path, timestamp_file, video_file, audio_file, output_audio_subfolder):
    # base_name = os.path.basename(video_file).split('.')[0]
    wont_be_used, base_name = split_speaker_and_utterance_id(os.path.basename(video_file).split('.')[0]) # NEW
    
    try:
        grid = tg.TextGrid()
        grid.read(timestamp_file)
    except Exception as e:
        print(f"Error reading TextGrid file {timestamp_file}: {e}")
        return
    
    # extract all words in order with their timestamps
    word_sequence = [(interval.mark.strip().upper(), interval.minTime, interval.maxTime) 
                     for interval in grid[0] if interval.mark.strip()]
    
    for phrase_file in os.listdir(subfolder_path):
        if phrase_file.endswith(".txt") and phrase_file.split("_")[-1].split(".")[0] == base_name:
            phrase_path = os.path.join(subfolder_path, phrase_file)
            phrase_num = phrase_file.split("_")[0]
            
            with open(phrase_path, 'r') as f:
                phrase = f.read().strip().upper()

            words = phrase.split()
            
            # Find exact phrase occurrence
            for i in range(len(word_sequence) - len(words) + 1):
                if all(word_sequence[i + j][0] == words[j] for j in range(len(words))):
                    start_time = word_sequence[i][1]
                    end_time = word_sequence[i + len(words) - 1][2] + 0.15
                    break
            else:
                print(f"Skipping {phrase_num}_{base_name}: Phrase not found in correct sequence.")
                continue

            # os.makedirs(output_subfolder, exist_ok=True)

            # output_video_path = os.path.join(output_subfolder, f"{phrase_num}_{base_name}.mp4")
            # subprocess.run([
            #     "ffmpeg", "-i", video_file, "-ss", str(start_time), "-to", str(end_time),
            #     "-c:v", "libx264", "-c:a", "aac", '-loglevel', 'quiet', output_video_path
            # ], check=True)
            
            # output_audio_path = os.path.join(output_audio_subfolder, f"{phrase_num}_{base_name}.wav")
            # subprocess.run([
            #     "ffmpeg", "-i", audio_file, "-ss", str(start_time), "-to", str(end_time),
            #     "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", '-loglevel', 'quiet', output_audio_path
            # ], check=True)
            
            # print(f"Saved video and audio for phrase {phrase_num} in {subfolder_path}")

            try:
                os.makedirs(output_subfolder, exist_ok=True)
                os.makedirs(output_audio_subfolder, exist_ok=True)

                output_video_path = os.path.join(output_subfolder, f"{phrase_num}_{base_name}.mp4")
                subprocess.run([
                    "ffmpeg", "-i", video_file, "-ss", str(start_time), "-to", str(end_time),
                    "-c:v", "libx264", "-c:a", "aac", "-loglevel", "quiet", output_video_path
                ], check=True)
                '''
                output_audio_path = os.path.join(output_audio_subfolder, f"{phrase_num}_{base_name}.wav")
                subprocess.run([
                    "ffmpeg", "-i", audio_file, "-ss", str(start_time), "-to", str(end_time),
                    "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", "-loglevel", "quiet", output_audio_path
                ], check=True)
                '''
                print(f"Saved video and audio for phrase {phrase_num} in {subfolder_path}")

            except subprocess.CalledProcessError as e:
                print(f"ffmpeg failed for {phrase_num}_{base_name}: {e}")

# NEW
def split_speaker_and_utterance_id(subfolder_name):
    return subfolder_name.split('_')[0], subfolder_name.split('_')[1]

def process_all_videos_in_folder(base_video_folder, base_audio_folder, base_textgrid_folder, phrase_transcripts_folder, output_base_folder, output_audio_folder, processed_file):
    processed = 0
    processed_videos = read_processed_videos(processed_file)
    
    for folder_name in os.listdir(base_video_folder):
        if '_' not in folder_name:
            print(f"Skipping folder without underscore: {folder_name}")
            continue
        try:
            folder_name_wo_utterance_id, video_name_wo_speaker_id = folder_name.split('_', 1)
        except ValueError:
            print(f"[ERROR] Bad folder name format: {folder_name}")
            continue

        folder_path = os.path.join(base_video_folder, folder_name)
        
        if os.path.isdir(folder_path):
            for video_filename in os.listdir(folder_path):
                if video_filename.endswith(".mp4"):
                    video_file = os.path.join(folder_path, video_filename)
                    if video_file in processed_videos:
                        print(f"Skipping already processed video: {video_file}")
                        continue
                    
                    textgrid_file_name = f"{folder_name}_{video_filename.split('.')[0]}.TextGrid"
                    #textgrid_file_name = f"{folder_name_wo_utterance_id}_{video_name_wo_speaker_id.split('.')[0]}.TextGrid" # NEW
                    textgrid_file = os.path.join(base_textgrid_folder, textgrid_file_name)
                    
                    phrase_subfolder = os.path.join(phrase_transcripts_folder, folder_name_wo_utterance_id) # CHANGED SECOND PARAM
                    full_base_name = f"{folder_name_wo_utterance_id}_{video_name_wo_speaker_id.split('.')[0]}"
                    audio_file = os.path.join(base_audio_folder, full_base_name, f"{full_base_name}.wav")
                    output_subfolder = os.path.join(output_base_folder, folder_name_wo_utterance_id) # CHANGED SECOND PARAM
                    output_audio_subfolder = os.path.join(output_audio_folder, folder_name_wo_utterance_id) # CHANGED SECOND PARAM
                    
                    if not os.path.exists(textgrid_file):
                        print(f"Missing textgrid for '{video_filename}'. Skipping.")
                        continue

                    if not os.path.exists(phrase_subfolder):
                        print(f"Missing phrase folder for '{video_filename}'. Skipping.")
                        continue
                        
                    print(f"Processing {textgrid_file} with video {video_filename}...")
                    extract_phrases_and_split_video(output_subfolder, phrase_subfolder, textgrid_file, video_file, audio_file, output_audio_subfolder)
                    
                    add_to_processed_videos(processed_file, video_file)
                    processed += 1
    
    print(f"Processed Videos: {processed}")

# lrs3 test
# base_video_folder = "../../models/auto_avsr_orig/preparation/preprocessed_lrs3_test_set/lrs3/lrs3_video_seg16s/test"
# base_audio_folder = "../audio/lrs3_test"
# base_textgrid_folder = "../mfa_words_timestamps/lrs3_test"
# phrase_transcripts_folder = "../phrases/lrs3/test/transcripts"
# output_base_folder = "../phrases/lrs3/test/preprocessed"
# output_audio_folder = "../phrases/lrs3/test/audio"
# processed_file = "../phrases/lrs3/processed_videos_test.txt"

base_video_folder = "../../datasets/face_pose_transformed/lrs3/test"
base_audio_folder = "../../datasets/face_pose_transformed/lrs3/test"
base_textgrid_folder = "../mfa_words_timestamps/lrs3_test"
phrase_transcripts_folder = "../phrases/lrs3/test/transcripts"
output_base_folder = "../../datasets/lrs3_task4/speech_unit_type/phrases/uncropped_phrases_video"
output_audio_folder = "../../datasets/lrs3_task4/speech_unit_type/phrases/uncropped_phrases_audio"
processed_file = "../../datasets/lrs3_task4/speech_unit_type/phrases/uncropped_phrases_lrs3_transformed.txt"

os.makedirs(output_base_folder, exist_ok=True)

process_all_videos_in_folder(base_video_folder, base_audio_folder, base_textgrid_folder, phrase_transcripts_folder, output_base_folder, output_audio_folder, processed_file)

print("Done!")
