import os
from collections import defaultdict

def count_npz_files(root_dir, output_file, output_total_file, output_total_all_file):
    word_counts = defaultdict(lambda: {"train": 0, "test": 0, "val": 0, "total": 0})
    overall_counts = {"train": 0, "test": 0, "val": 0, "total": 0}

    for word in sorted(os.listdir(root_dir)):
        word_path = os.path.join(root_dir, word)
        if os.path.isdir(word_path):
            total_count = 0
            for split in ["train", "test", "val"]:
                split_path = os.path.join(word_path, split)
                if os.path.isdir(split_path):
                    npz_files = [f for f in os.listdir(split_path) if f.endswith(".npz")]
                    count = len(npz_files)
                    word_counts[word][split] = count
                    overall_counts[split] += count  # add to overall counts for each split
                    total_count += count
            word_counts[word]["total"] = total_count
            overall_counts["total"] += total_count

    with open(output_file, "w") as f:
        for word, count in word_counts.items():
            f.write(f"Word: {word} -> Train: {count['train']}, Val: {count['val']}, Test: {count['test']}\n")

    with open(output_total_file, "w") as f:
        for word, count in word_counts.items():
            f.write(f"{word} -> {count['total']}\n")

    with open(output_total_all_file, "w") as f:
        f.write(f"Train count: {overall_counts['train']}\n")
        f.write(f"Val count: {overall_counts['val']}\n")
        f.write(f"Test count: {overall_counts['test']}\n")
        f.write(f"Total count: {overall_counts['total']}\n")

    print('Done!')

root_directory = "../../models/DC-TCN/datasets/lrs3_words_may_100filtered"
output_file = "../words/word_counts/86lrs3_word_counts_sets.txt"
output_total_file = "../words/word_counts/86lrs3_word_counts_total.txt"
output_total_all_file = "../words/word_counts/86lrs3_set_count_total.txt"

count_npz_files(root_directory, output_file, output_total_file, output_total_all_file)
