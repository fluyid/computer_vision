from pathlib import Path
from shutil import copytree
from datetime import datetime

DATASET_DIR = Path("/Users/kailashnah/PycharmProjects/computer_vision/screenshot_generator/dataset")
LABELS_DIR = DATASET_DIR / "labels"

SPLITS = ["train", "val", "test"]

def backup_labels() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = DATASET_DIR / f"labels_backup_before_id_fix_{timestamp}"
    copytree(LABELS_DIR, backup_dir)
    return backup_dir

def fix_label_file(txt_file: Path) -> tuple[int, int]:
    """
    Returns:
        lines_changed, lines_skipped
    """
    changed = 0
    skipped = 0
    new_lines = []

    lines = txt_file.read_text(encoding="utf-8").splitlines()

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        if not line:
            new_lines.append(raw_line)
            continue

        parts = line.split()

        if len(parts) != 5:
            print(f"SKIP malformed line in {txt_file.name} line {line_number}: {raw_line!r}")
            new_lines.append(raw_line)
            skipped += 1
            continue

        class_str, x, y, w, h = parts

        try:
            class_id = int(class_str)
        except ValueError:
            print(f"SKIP non-integer class id in {txt_file.name} line {line_number}: {class_str!r}")
            new_lines.append(raw_line)
            skipped += 1
            continue

        if not (1 <= class_id <= 6):
            print(
                f"SKIP unexpected class id in {txt_file.name} line {line_number}: "
                f"{class_id} (expected 1-6 before fix)"
            )
            new_lines.append(raw_line)
            skipped += 1
            continue

        fixed_class_id = class_id - 1
        new_lines.append(f"{fixed_class_id} {x} {y} {w} {h}")
        changed += 1

    txt_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return changed, skipped

def main() -> None:
    if not LABELS_DIR.exists():
        raise FileNotFoundError(f"Labels folder not found: {LABELS_DIR}")

    print(f"Dataset dir: {DATASET_DIR}")
    print(f"Labels dir: {LABELS_DIR}")

    print("\nCreating backup...")
    backup_dir = backup_labels()
    print(f"Backup created at: {backup_dir}")

    total_files = 0
    total_changed_lines = 0
    total_skipped_lines = 0

    for split in SPLITS:
        split_dir = LABELS_DIR / split
        if not split_dir.exists():
            print(f"\nMissing split folder, skipping: {split_dir}")
            continue

        print(f"\n=== FIXING {split.upper()} ===")
        split_files = 0
        split_changed = 0
        split_skipped = 0

        for txt_file in sorted(split_dir.glob("*.txt")):
            changed, skipped = fix_label_file(txt_file)
            split_files += 1
            split_changed += changed
            split_skipped += skipped

        total_files += split_files
        total_changed_lines += split_changed
        total_skipped_lines += split_skipped

        print(f"Files processed: {split_files}")
        print(f"Lines changed: {split_changed}")
        print(f"Lines skipped: {split_skipped}")

    print("\n=== DONE ===")
    print(f"Total files processed: {total_files}")
    print(f"Total lines changed: {total_changed_lines}")
    print(f"Total lines skipped: {total_skipped_lines}")
    print(f"Backup location: {backup_dir}")
    print("\nNext step: rerun check_class_ids.py and confirm classes are now 0-5 only.")

if __name__ == "__main__":
    main()