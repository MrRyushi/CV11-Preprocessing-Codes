# only have the transcripts contain the text
import os

lrs2_source_directory = "../../datasets/lrs2/val"
lrs2_destination_directory = "../adjusted_transcripts/lrs2/val"
lrs3_source_directory = "../../models/auto_avsr_orig/preparation/lrs3_trainval/trainval"
lrs3_destination_directory = "../adjusted_transcripts/lrs3_trainval"

def adjust_transcripts(source_directory, destination_directory):
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    for dirpath, dirnames, filenames in os.walk(source_directory):
        # only process folders inside LRS2 (skip the root level)
        if dirpath == source_directory:
            continue
    
        # create the corresponding subdirectory in the destination folder
        relative_path = os.path.relpath(dirpath, source_directory)
        destination_path = os.path.join(destination_directory, relative_path)
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        
        for filename in filenames:
            if filename.endswith(".txt"):
                source_file_path = os.path.join(dirpath, filename)
                destination_file_path = os.path.join(destination_path, filename)
                
                # read the transcript file and extract the text
                with open(source_file_path, 'r') as file:
                    lines = file.readlines()
                
                # extract the text from the "Text:" line
                text = ""
                for line in lines:
                    if line.startswith("Text:"):
                        text = line.replace("Text: ", "").strip()
                
                # save the extracted text into the new file in the destination folder
                with open(destination_file_path, 'w') as file:
                    file.write(text)
    
                print(f"Processed file: {source_file_path} and saved to {destination_file_path}")
    
    print(f"All transcript files have been processed and saved to {destination_directory}")

adjust_transcripts(lrs2_source_directory, lrs2_destination_directory)
#adjust_transcripts(lrs3_source_directory, lrs3_destination_directory)