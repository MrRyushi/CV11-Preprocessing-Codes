import textgrid as tg
import os
import subprocess

# Add ffmpeg to system path
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
    base_filename = os.path.basename(video_file).split('.')[0]
    base_name = base_filename.split("_")[-1]

    try:
        grid = tg.TextGrid()
        grid.read(timestamp_file)
    except Exception as e:
        print(f"Error reading TextGrid file {timestamp_file}: {e}")
        return
    
    word_sequence = [(interval.mark.strip().upper(), interval.minTime, interval.maxTime)
                     for interval in grid[0] if interval.mark.strip()]

    for phrase_file in os.listdir(subfolder_path):
        if phrase_file.endswith(".txt") and phrase_file.split(".")[0] == base_name:
            phrase_path = os.path.join(subfolder_path, phrase_file)
            phrase_num = base_name  # e.g. u10
            
            with open(phrase_path, 'r') as f:
                phrase = f.read().strip().upper()

            words = phrase.split()

            # Match phrase in word sequence
            for i in range(len(word_sequence) - len(words) + 1):
                if all(word_sequence[i + j][0] == words[j] for j in range(len(words))):
                    start_time = word_sequence[i][1]
                    end_time = word_sequence[i + len(words) - 1][2] + 0.15
                    break
            else:
                print(f"Skipping {phrase_num}_{base_filename}: Phrase not found in correct sequence.")
                continue

            os.makedirs(output_subfolder, exist_ok=True)
            os.makedirs(output_audio_subfolder, exist_ok=True)

            output_video_path = os.path.join(output_subfolder, f"{phrase_num}_{base_filename}.mp4")
            output_audio_path = os.path.join(output_audio_subfolder, f"{phrase_num}_{base_filename}.wav")

            # Save video segment
            subprocess.run([
                "ffmpeg", "-i", video_file, "-ss", str(start_time), "-to", str(end_time),
                "-c:v", "libx264", "-c:a", "aac", "-loglevel", "quiet", output_video_path
            ], check=True)

            # Save audio segment
            subprocess.run([
                "ffmpeg", "-i", audio_file, "-ss", str(start_time), "-to", str(end_time),
                "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", "-loglevel", "quiet", output_audio_path
            ], check=True)

            print(f"Saved video and audio for phrase {phrase_num} in {subfolder_path}")

def process_all_videos_in_folder(base_video_folder, base_audio_folder, base_textgrid_folder, phrase_transcripts_folder, output_base_folder, output_audio_folder, processed_file):
    processed = 0
    processed_videos = read_processed_videos(processed_file)

    for folder_name in os.listdir(base_video_folder):
        folder_path = os.path.join(base_video_folder, folder_name)

        if os.path.isdir(folder_path):
            for video_filename in os.listdir(folder_path):
                if video_filename.endswith(".mp4"):
                    video_file = os.path.join(folder_path, video_filename)
                    video_key = video_filename

                    if video_key in processed_videos:
                        print(f"Skipping already processed video: {video_key}")
                        continue

                    video_base = video_filename.split('.')[0]         # s10_v10_u10
                    speaker = video_base.split('_')[0]                # s10
                    utterance = video_base.split('_')[-1]             # u10
                    textgrid_file_name = f"{speaker}_{utterance}.TextGrid"
                    textgrid_file = os.path.join(base_textgrid_folder, textgrid_file_name)

                    phrase_subfolder = phrase_transcripts_folder
                    audio_file = os.path.join(base_audio_folder, folder_name, f"{video_base}.wav")
                    output_subfolder = os.path.join(output_base_folder, folder_name)
                    output_audio_subfolder = os.path.join(output_audio_folder, folder_name)

                    if not os.path.exists(textgrid_file):
                        print(f"Missing textgrid for '{video_filename}'. Skipping.")
                        continue

                    if not os.path.exists(phrase_subfolder):
                        print(f"Missing phrase folder for '{video_filename}'. Skipping.")
                        continue

                    print(f"Processing {textgrid_file} with video {video_filename}...")
                    extract_phrases_and_split_video(output_subfolder, phrase_subfolder, textgrid_file, video_file, audio_file, output_audio_subfolder)

                    add_to_processed_videos(processed_file, video_key)
                    processed += 1

    print(f"Processed Videos: {processed}")

# === Paths for OuluVS2 ===
base_video_folder = "../../datasets/ouluvs2/test"
base_audio_folder = "../../datasets/ouluvs2/test"
base_textgrid_folder = "../mfa_words_timestamps/ouluvs2_test"
phrase_transcripts_folder = "../phrases/ouluvs2/transcripts"
output_base_folder = "../../datasets/ouluvs2_task4/speech_unit_type/phrases/test_derived"
output_audio_folder = "../../datasets/ouluvs2_task4/speech_unit_type/phrases/test_derived"
processed_file = "../../datasets/ouluvs2_task4/speech_unit_type/phrases/test_derived/test_derived.txt"

# === Run script ===
os.makedirs(output_base_folder, exist_ok=True)
os.makedirs(output_audio_folder, exist_ok=True)

process_all_videos_in_folder(
    base_video_folder,
    base_audio_folder,
    base_textgrid_folder,
    phrase_transcripts_folder,
    output_base_folder,
    output_audio_folder,
    processed_file
)

print("Done!")
