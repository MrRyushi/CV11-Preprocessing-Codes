import os

number_sequences = [
    "1 7 3 5 1 6 2 6 6 7",
    "4 0 2 9 1 8 5 9 0 4",
    "1 9 0 7 8 8 0 3 2 8",
    "4 9 1 2 1 1 8 5 5 1",
    "8 6 3 5 4 0 2 1 1 2",
    "2 3 9 0 0 1 6 7 6 4",
    "5 2 7 1 6 1 3 6 7 0",
    "9 7 4 4 4 3 5 5 8 7",
    "6 3 8 5 3 9 8 5 6 5",
    "7 3 2 4 0 1 9 9 5 0"
]

words = [
    "Excuse me",
    "Goodbye",
    "Hello",
    "How are you",
    "Nice to meet you",
    "See you",
    "I am sorry",
    "Thank you",
    "Have a good time",
    "You are welcome"
]

def write_to_files(folder_path):
    user_index = 31
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
        
    for i in range(len(words)):
        # Create and write to the corresponding u1.txt, u2.txt, u3.txt...
        for j in range(3):
            file_path = os.path.join(folder_path, f'u{user_index}.txt')
            try:
                # Write the number sequence to each corresponding file
                with open(file_path, 'a') as file:
                    #file.write(f"{number_sequences[i]}\n")
                     file.write(f"{words[i]}\n")
                print(f"Written to {file_path}")
            except Exception as e:
                print(f"Failed to write to {file_path}: {e}")
        
            user_index += 1

folder_path = 'train_val/train_val/split_transcripts'
write_to_files(folder_path)