# restructure to word folder formats
import os
import shutil

def move_files_to_new_structure(base_folder, output_folder, subfolder_name):
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.mp4'):
                print(f"Processing file: {file}")
                
                # extract the WORD (label) and subfolder from the file name
                parts = file.split('_')
                if len(parts) >= 3:
                    word = parts[0]
                    print(f"Word: {word}, Subfolder: {subfolder_name}")
                    
                    target_folder = os.path.join(output_folder, word, subfolder_name)
                    os.makedirs(target_folder, exist_ok=True)  # create the folder if it doesn't exist
                    print(f"Target folder created: {target_folder}")
                    
                    source_file_path = os.path.join(root, file)
                    target_file_path = os.path.join(target_folder, file)

                    if os.path.exists(target_file_path):
                        print(f"Skipping {file}: Already exists at target location")
                        continue
                    
                    # move the file to the new folder
                    shutil.copy(source_file_path, target_file_path)
                    print(f"Copied {file} to {target_file_path}")

                else:
                    print(f"Skipping file {file}: Invalid filename format")

#####
lrs3_test = "../../datasets/lrs3_task4/lrs3_raw_phoamb/test/folder1"
lrs3_merged = "../../datasets/lrs3_task4/lrs3_raw_phoamb/test_dctcn"

move_files_to_new_structure(lrs3_test, lrs3_merged, 'test')
#####

# lrs3_trainval = "../words/lrs3_may/preprocessed_videos_trainval"
# lrs3_test = "../words/lrs3_may/preprocessed_videos_test"
# lrs3_merged = "../words/lrs3_may/preprocessed_videos_all"

# lrs2_train = "../words/lrs2_may/preprocessed_videos_train"
# lrs2_val = "../words/lrs2_may/preprocessed_videos_val"
# lrs2_test = "../words/lrs2_may/preprocessed_videos_test"
# lrs2_merged = "../words/lrs2_may/preprocessed_videos_all"


# #move_files_to_new_structure(lrs2_train, lrs2_merged, 'train')
# #move_files_to_new_structure(lrs2_val, lrs2_merged, 'val')
# #move_files_to_new_structure(lrs2_test, lrs2_merged, 'test')

# move_files_to_new_structure(lrs3_trainval, lrs3_merged, 'trainval')
# move_files_to_new_structure(lrs3_test, lrs3_merged, 'test')

print("File organization complete!")
