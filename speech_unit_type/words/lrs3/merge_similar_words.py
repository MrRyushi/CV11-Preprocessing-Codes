# combine the files that have the same word just with an '
# ex. aim's and aims
import os
import shutil

def normalize_word(word):
    if word.endswith("'S"):  # only process words ending in 'S
        return word.replace("'", "")
    return word

def merge_folders(parent_folder):
    word_map = {}
    
    
    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)
        
        if not os.path.isdir(folder_path):
            continue
        
        normalized = normalize_word(folder)
        
        if normalized in word_map:
            target_folder = word_map[normalized]
            if target_folder.endswith("'S") or target_folder.endswith("S"):
                target_folder = folder_path
                folder_path = word_map[normalized]
            print(f"Merging {folder_path} -> {target_folder}")

            if not os.path.exists(target_folder) or not os.path.exists(folder_path):
                continue
            
            # for subfolder in ["test", "train", "val"]: # train test
            for subfolder in ["test"]: # train test
                target_subfolder = os.path.join(target_folder, subfolder)
                source_subfolder = os.path.join(folder_path, subfolder)
                
                if os.path.exists(source_subfolder):
                    os.makedirs(target_subfolder, exist_ok=True)
                    # move files from source subfolder to target subfolder
                    for item in os.listdir(source_subfolder):
                        src_item = os.path.join(source_subfolder, item)
                        dest_item = os.path.join(target_subfolder, item)
                        
                        if os.path.exists(dest_item):
                            print(f"Conflict detected, skipping: {dest_item}")
                        else:
                            shutil.move(src_item, target_subfolder)
                    
                    # check if source subfolder is empty before deleting
                    if not os.listdir(source_subfolder):
                        os.rmdir(source_subfolder)
                        print(f'Deleted {source_subfolder}')
                    else:
                        print(f'Failed Delete {source_subfolder}')
            
            # check if the entire folder is empty before deleting
            if not os.listdir(folder_path):
                os.rmdir(folder_path)
                print(f'Deleted {folder_path}')
            else:
                print(f'Failed Delete {folder_path}')
        else:
            word_map[normalized] = folder_path

folder = "../../datasets/lrs3_task4/lrs3_dctcn_synfront/lrs3_synfront_npz"
# folder = "../../datasets/lrs3_task4/preprocessed_videos_test"
# folder = "../words/lrs3_may/preprocessed_videos_all"
# folder = "../../models/DC-TCN/datasets/lrs2_words_may"
merge_folders(folder)
