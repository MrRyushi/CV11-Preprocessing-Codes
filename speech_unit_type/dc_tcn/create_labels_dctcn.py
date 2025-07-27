import os

def create_labels_txt(base_folder, output_file):
    words = 0
    with open(output_file, 'w') as f:
        folder_names = [dir_name for dir_name in os.listdir(base_folder)
                        if os.path.isdir(os.path.join(base_folder, dir_name))]
        folder_names.sort()  # sort the folder names alphabetically
        
        for dir_name in folder_names:
            # added
            test_folder = os.path.join(base_folder, dir_name, 'test')
            if os.path.isdir(test_folder) and any(os.listdir(test_folder)):
                f.write(dir_name + '\n')
                print(f"Label added: {dir_name}")
                words += 1

    print(f"{words} Labels have been saved to {output_file}")

# base_folder = "../../models/DC-TCN/datasets/lrs2_words_may"
# output_file = "../../models/DC-TCN/labels/5942Lrs2List.txt"

base_folder = "../../datasets/lrs3_task4/lrs3_dctcn_synfront/lrs3_synfront_npz"
output_file = "../synfront_classes.txt"
'''
base_folder = "../../datasets/lrs3_task4/words_lrs3_transformed"
output_file = "../../datasets/lrs3_task4/words_lrs3_transformed_labels.txt"
'''

create_labels_txt(base_folder, output_file)
