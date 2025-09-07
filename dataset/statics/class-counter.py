import os
from collections import Counter

DATASET_DIR = r"."

SPLITS = ["train", "valid", "test"]

# Class list (index = class id)
CLASS_NAMES = [
    'Bike', 'Building', 'Car', 'Person', 'Stairs', 'Traffic sign', 'Electrical Pole',
    'Road', 'Motorcycle', 'Dustbin', 'Dog', 'Manhole', 'Tree', 'Guard rail',
    'Pedestrian crosswalk', 'Truck', 'Bus', 'Bench'
]

def count_split(split_dir):
    """Count YOLO label instances in split_dir/labels/*.txt."""
    labels_dir = os.path.join(split_dir, "labels")
    counter = Counter()
    if not os.path.isdir(labels_dir):
        return counter

    for fname in os.listdir(labels_dir):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(labels_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    # YOLO format: <class_id> <cx> <cy> <w> <h>, therefore, we will check the first index for counting operation
                    parts = s.split()
                    try:
                        cls_id = int(parts[0])
                    except ValueError:
                        continue  # skip malformed line for avoiding undesirable exceptions

                    #count new class id in our counter for specific instance
                    counter[cls_id] += 1
        except OSError:
            # skip unreadable file for avoiding undesirable exceptions
            continue
    return counter

def as_list(counter, n_classes):
    """Convert Counter of class_id->count to list aligned with CLASS_NAMES order."""
    return [counter.get(i, 0) for i in range(n_classes)]

def format_section(title, counts):
    """Return formatted and ordered report for each section

        Args ->

            title: the title of report which will indicate which section is this report related to, eg. Train 
            counts: related counts for reported section(a list of numbers which are related to dataset's instances)
    """
    lines = []
    lines.append("-" * 30)
    lines.append(title)
    lines.append(" ----- each class info -----")
    width = max(len(n) for n in CLASS_NAMES) + 2
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"{name:<{width}}: {counts[i]}")
    lines.append(f" ----- total instances: {sum(counts)}")
    lines.append("-" * 60)
    return "\n".join(lines)    

def save_section_to_txt(title, counts, out_dir="results"):
    """Save one section into its own .txt file."""
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, title.replace(" ", "_") + ".txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(format_section(title, counts))
    print(f"[+] Saved {filename}")

def main():
    n_classes = len(CLASS_NAMES)

    # Per-split counts
    split_counts = {}
    for split in SPLITS:
        split_dir = os.path.join(DATASET_DIR, split)
        c = count_split(split_dir)
        split_counts[split] = as_list(c, n_classes)

    test_counts = split_counts.get("test", [0] * n_classes)
    train_counts = split_counts.get("train", [0] * n_classes)
    valid_counts = split_counts.get("valid", [0] * n_classes)

    # Accumulated (dataset) counts
    dataset_counts = [0] * n_classes
    for counts in split_counts.values():
        for i in range(n_classes):
            dataset_counts[i] += counts[i]

    # Print on console
    print(format_section("Test instances info", test_counts))
    print(format_section("Train instances info", train_counts))
    print(format_section("Valid instances info", valid_counts))
    print(format_section("Dataset instances info", dataset_counts))

    # Save to text files
    save_section_to_txt("Test instances info", test_counts)
    save_section_to_txt("Train instances info", train_counts)
    save_section_to_txt("Valid instances info", valid_counts)
    save_section_to_txt("Dataset instances info", dataset_counts)

if __name__ == "__main__":
    main()