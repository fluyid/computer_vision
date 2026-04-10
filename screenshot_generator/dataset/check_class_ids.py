from pathlib import Path
from collections import Counter

DATASET_DIR = Path("/Users/kailashnah/PycharmProjects/computer_vision/screenshot_generator/dataset")
SPLITS = ["train", "val", "test"]

def main() -> None:
    overall_counter = Counter()

    for split in SPLITS:
        labels_dir = DATASET_DIR / "labels" / split
        split_counter = Counter()
        bad_files = []

        print(f"\n=== {split.upper()} ===")

        for txt_file in sorted(labels_dir.glob("*.txt")):
            try:
                lines = txt_file.read_text(encoding="utf-8").splitlines()
            except Exception as exc:
                print(f"Could not read {txt_file.name}: {exc}")
                continue

            for line_num, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 1:
                    continue

                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    bad_files.append((txt_file.name, line_num, parts[0]))
                    continue

                split_counter[class_id] += 1
                overall_counter[class_id] += 1

                if class_id == 6:
                    bad_files.append((txt_file.name, line_num, class_id))

        print("Class counts:")
        for class_id, count in sorted(split_counter.items()):
            print(f"  class {class_id}: {count}")

        if bad_files:
            print("\nFiles containing class 6 or invalid class ids:")
            for item in bad_files[:50]:
                print(f"  {item}")
            if len(bad_files) > 50:
                print(f"  ... and {len(bad_files) - 50} more")

    print("\n=== OVERALL ===")
    for class_id, count in sorted(overall_counter.items()):
        print(f"class {class_id}: {count}")

if __name__ == "__main__":
    main()