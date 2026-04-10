from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_DIR / "runs"
BEST_WEIGHTS = RUNS_DIR / "lol_detect_baseline3" / "weights" / "best.pt"

MODE = input("images or video: " ).strip() # choose: "images" or "video"

# Image inference source:
IMAGE_SOURCE = PROJECT_DIR / "screenshot_generator" / "dataset" / "images" / "val"

# Video inference source:
VIDEO_SOURCE = PROJECT_DIR / "video_clips" / "input_video-teamfight.mkv"

IMAGE_SIZE = 640
CONFIDENCE = 0.25

IMAGE_RUN_NAME = "predict_images"
VIDEO_RUN_NAME = "predict_video"


def load_model() -> YOLO:
    print(f"Looking for trained weights: {BEST_WEIGHTS}")
    if not BEST_WEIGHTS.exists():
        raise FileNotFoundError(
            f"best.pt was not found at:\n{BEST_WEIGHTS}\n"
            "Update BEST_WEIGHTS so it points to your trained model."
        )
    print("Loading trained model...")
    return YOLO(str(BEST_WEIGHTS))


def run_images() -> None:
    source = Path(IMAGE_SOURCE)
    print(f"Image source: {source}")

    if not source.exists():
        raise FileNotFoundError(
            f"Image source was not found:\n{source}\n"
            "Set IMAGE_SOURCE to a real image file or folder."
        )

    model = load_model()

    print("Running image inference...")
    model.predict(
        source=str(source),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        save=True,
        project=str(RUNS_DIR),
        name=IMAGE_RUN_NAME,
    )

    print(f"Done. Check results in: {RUNS_DIR / IMAGE_RUN_NAME}")


def run_video() -> None:
    source = Path(VIDEO_SOURCE)
    print(f"Video source: {source}")

    if not source.exists():
        raise FileNotFoundError(
            f"Video source was not found:\n{source}\n"
            "Put your video in the project folder or update VIDEO_SOURCE."
        )

    model = load_model()

    print("Running video inference...")
    model.predict(
        source=str(source),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        save=True,
        project=str(RUNS_DIR),
        name=VIDEO_RUN_NAME,
    )

    print(f"Done. Check results in: {RUNS_DIR / VIDEO_RUN_NAME}")


def main() -> None:
    if MODE == "images":
        run_images()
    elif MODE == "video":
        run_video()
    else:
        raise ValueError("MODE must be 'images' or 'video'.")


if __name__ == "__main__":
    main()