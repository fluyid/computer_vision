from pathlib import Path
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DATA_YAML = PROJECT_DIR / "dataset.yaml"

MODEL_NAME = "yolo11n.pt"
EPOCHS = 30
IMAGE_SIZE = 640

def main() -> None:
    print(f"Project folder: {PROJECT_DIR}")
    print(f"Looking for dataset file: {DATA_YAML}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"dataset.yaml was not found at:\n{DATA_YAML}\n"
            "Put dataset.yaml in the same folder as this script."
        )

    print("Loading model...")
    model = YOLO(MODEL_NAME)

    print("Starting training...")
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        project=str(PROJECT_DIR / "runs"),
        name="lol_detect_baseline",
    )

    print("Training finished.")

if __name__ == "__main__":
    main()