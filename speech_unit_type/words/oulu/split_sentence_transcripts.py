import os
import string

# split each transcription file by lines and save each line with a custom name
def split_and_save_lines_from_multiple_files(input_folder, output_folder):
    """
    Processes multiple transcription files, splits them by lines,
    and saves each line in a separate file.
    The naming format will be 's<video_section>_v3_u<line_number>'.
    """
    total_files_processed = 0
    total_files_created = 0

    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output folder '{output_folder}' is ready.")
    except Exception as e:
        print(f"Error creating output folder: {e}")
        return

    try:
        files = os.listdir(input_folder)
        if not files:
            print(f"No files found in the input folder: {input_folder}")
            return
    except FileNotFoundError as e:
        print(f"Error: The input folder '{input_folder}' was not found. Please check the path.")
        return
    except Exception as e:
        print(f"Unexpected error while listing files in input folder: {e}")
        return

    for filename in files:
        if filename.split('.')[0] in ['s1', 's2', 's3', 's4', 's5', 's7', 's10', 's11', 's12', 's13', 's14', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's27', 's28', 's29', 's31', 's32', 's33', 's35', 's36', 's37', 's38', 's39', 's40', 's41', 's42', 's45', 's47', 's48', 's59', 's53'    ]:
            try:
                file_path = os.path.join(input_folder, filename)
                section_name = filename.split('.')[0]  # extract section name (e.g., 's6', 's7')

                with open(file_path, 'r') as file:
                    sentences = file.readlines()
                print(f"Processing file: {filename}...")

                total_files_processed += 1
                line_count = 60

                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        line_count += 1
                        # create a custom file name: e.g., s15_v3_u1
                        output_file = os.path.join(output_folder, f"{section_name}_u{line_count}.txt")
                        
                        try:
                            with open(output_file, 'w') as output:
                                output.write(sentence)
                            total_files_created += 1
                            print(f"Saved: {output_file}")
                        except Exception as e:
                            print(f"Error writing to file {output_file}: {e}")
            
            except FileNotFoundError as e:
                print(f"Error: The file '{filename}' was not found.")
            except Exception as e:
                print(f"Unexpected error while processing file '{filename}': {e}")
        else:
            print(f"Skipping file from excluded section: {filename}")

    print(f"Total files processed: {total_files_processed}")
    print(f"Total files created: {total_files_created}")

input_folder = 'train_val/train_val/transcript_sentence'  # Folder containing the transcription files (e.g., s6.txt, s7.txt, etc.)
output_folder = 'train_val/train_val/split_transcripts'  # Folder to save the processed files

# Process all transcription files in the input folder and save the lines with custom names
split_and_save_lines_from_multiple_files(input_folder, output_folder)
print(f"Done!")
