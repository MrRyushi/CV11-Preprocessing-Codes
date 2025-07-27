import subprocess
import os

lrs2_source_directory = "../../datasets/lrs2/test"
lrs2_destination_directory = "../audio/lrs2/test"
lrs3_source_directory = "../../models/auto_avsr_orig/preparation/lrs3_trainval/lrs3/lrs3_video_seg16s/trainval"
lrs3_destination_directory = "../audio/lrs3_train_val"

os.environ["PATH"] += os.pathsep + os.path.abspath("../ffmpeg-7.0.2-i686-static")

# extract audio from video files
def extract_audio_from_videos(source_directory, destination_directory):
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    for dirpath, dirnames, filenames in os.walk(source_directory):
        # skip the root level folder and start with subfolders
        if dirpath == source_directory:
            continue
        
        # create the corresponding subdirectory in the destination folder
        relative_path = os.path.relpath(dirpath, source_directory)
        destination_path = os.path.join(destination_directory, relative_path)
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        
        # process video files in the folder
        for filename in filenames:
            if filename.endswith(('.mp4')):
                video_file = os.path.join(dirpath, filename)
                
                output_audio = os.path.join(destination_path, f"{os.path.splitext(filename)[0]}.wav")
                
                # extract audio
                subprocess.run(['ffmpeg', '-i', video_file, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', '-loglevel', 'quiet', output_audio])
                
                print(f"Audio extracted from {video_file} and saved as {output_audio}")
    
    print(f"All audio extractions are complete for {source_directory}.")    

extract_audio_from_videos(lrs2_source_directory, lrs2_destination_directory)
#extract_audio_from_videos(lrs3_source_directory, lrs3_destination_directory)

