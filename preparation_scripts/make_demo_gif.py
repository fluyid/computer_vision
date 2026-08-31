#!/usr/bin/env python3
"""Export a README-sized demo GIF from a video clip.

GIF has no compression between frames, so file size is driven by
duration x frame rate x resolution. This script encodes at the quality
you ask for, and if the result is over the size budget it steps down
through progressively cheaper settings until it fits.

Run with no arguments for interactive prompts:
    python make_demo_gif.py

Or pass everything up front:
    python make_demo_gif.py runs/predict_video/clip.mp4 docs/demo.gif --start 15 --duration 5
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class Settings:
    """One combination of export settings to try."""

    duration: float
    fps: int
    width: int

    def __str__(self) -> str:
        return f"{self.duration:g}s @ {self.fps}fps, {self.width}px wide"


def require_ffmpeg() -> None:
    """Fail early with a useful message rather than a confusing traceback."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"error: {tool} not found. Install it first (macOS: brew install ffmpeg)")


def probe_duration(video: Path) -> float:
    """Ask ffprobe how many seconds long the video is."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def encode(video: Path, out: Path, start: float, s: Settings, workdir: Path) -> int:
    """Render one GIF and return its size in bytes.

    Two passes, because GIF can only hold 256 colours per frame. The first
    pass looks at this specific clip and picks the best 256; the second uses
    them. Letting ffmpeg use its generic default palette instead produces
    visible colour banding.
    """
    palette = workdir / "palette.png"
    scale = f"scale={s.width}:-1:flags=lanczos"

    # stats_mode=diff weights the palette toward pixels that CHANGE between
    # frames, rather than treating a static background as equally important.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(s.duration), "-i", str(video),
         "-vf", f"fps={s.fps},{scale},palettegen=stats_mode=diff", "-y", str(palette)],
        check=True,
    )

    # dither=bayer uses a FIXED dot pattern for every frame. The default
    # picks a new pattern each frame, so unchanged areas of the image still
    # differ byte-for-byte and cannot be compressed away. This is usually the
    # single biggest size saving available.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(s.duration), "-i", str(video),
         "-i", str(palette),
         "-filter_complex",
         f"fps={s.fps},{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
         "-y", str(out)],
        check=True,
    )
    return out.stat().st_size


def ladder(start: Settings) -> list[Settings]:
    """Fallback settings, cheapest quality loss first.

    Order matters. Dropping the frame rate costs the least perceptually;
    shrinking the image costs the most, because the class labels burned into
    the video stop being readable. Trimming duration sits in between.
    """
    return [
        start,
        Settings(start.duration, max(8, start.fps - 2), start.width),
        Settings(start.duration, max(8, start.fps - 2), int(start.width * 0.9)),
        Settings(start.duration * 0.8, max(8, start.fps - 2), int(start.width * 0.9)),
        Settings(start.duration * 0.7, 8, int(start.width * 0.85)),
    ]


VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


def ask(prompt: str, default: str) -> str:
    """Prompt with a default. Enter accepts the default.

    Kept inside a function on purpose. Calling input() at module level means
    the prompt fires the moment anything imports the file, which makes the
    module impossible to reuse.
    """
    try:
        reply = input(f"{prompt} [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit("\ncancelled")
    return reply or default


def ask_number(prompt: str, default: float, *, cast=float, minimum: float = 0.0):
    """Prompt until the answer is actually a number in range.

    Re-asking beats crashing: a typo costs one line, not a whole run.
    """
    while True:
        raw = ask(prompt, str(default))
        try:
            value = cast(raw)
        except ValueError:
            print(f"  '{raw}' is not a number - try again")
            continue
        if value <= minimum:
            print(f"  needs to be greater than {minimum:g}")
            continue
        return value


def find_videos(root: Path) -> list[Path]:
    """Look for video files so the user can pick from a list instead of typing a path."""
    found = [p for p in root.rglob("*")
             if p.suffix.lower() in VIDEO_SUFFIXES
             and ".venv" not in p.parts
             and "video_clips" not in p.parts]
    return sorted(found, key=lambda p: p.stat().st_size, reverse=True)[:9]


def choose_video() -> Path:
    """Offer discovered videos as a numbered menu, with typing a path as the fallback."""
    candidates = find_videos(Path.cwd())
    if candidates:
        print("\nVideos found:")
        for i, p in enumerate(candidates, start=1):
            print(f"  {i}. {p}  ({p.stat().st_size / BYTES_PER_MB:.0f} MB)")
        print("  0. enter a path myself")
        while True:
            choice = ask("Which video?", "1")
            if choice == "0":
                break
            if choice.isdigit() and 1 <= int(choice) <= len(candidates):
                return candidates[int(choice) - 1]
            print("  pick a number from the list")

    while True:
        path = Path(ask("Path to video", "")).expanduser()
        if path.is_file():
            return path
        print(f"  no such file: {path}")


def interactive() -> argparse.Namespace:
    """Collect every setting by prompt."""
    print("Demo GIF exporter - press Enter to accept each default")
    video = choose_video()
    length = probe_duration(video)
    print(f"\n{video.name} is {length:.1f}s long")

    start = ask_number("Start time (seconds)", 0.0, minimum=-1.0)
    duration = ask_number("Clip length (seconds)", 5.0)
    output = Path(ask("Output path", "docs/demo.gif"))
    fps = int(ask_number("Frames per second", 10, cast=int))
    width = int(ask_number("Width in pixels", 640, cast=int))
    max_mb = ask_number("Size budget (MB)", 5.0)

    return argparse.Namespace(video=video, output=output, start=start,
                              duration=duration, fps=fps, width=width, max_mb=max_mb)


def main() -> None:
    require_ffmpeg()

    # No arguments at all means the user just ran the script - ask instead of
    # printing a usage error at them.
    if len(sys.argv) == 1:
        args = interactive()
    else:
        args = parse_args()

    run(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="source video")
    parser.add_argument("output", type=Path, help="destination .gif")
    parser.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    parser.add_argument("--duration", type=float, default=5.0, help="clip length in seconds")
    parser.add_argument("--fps", type=int, default=10, help="frames per second")
    parser.add_argument("--width", type=int, default=640, help="output width in pixels")
    parser.add_argument("--max-mb", type=float, default=5.0, help="size budget in MB")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if not args.video.is_file():
        sys.exit(f"error: no such file: {args.video}")

    length = probe_duration(args.video)
    if args.start >= length:
        sys.exit(f"error: --start {args.start}s is past the end of a {length:.1f}s video")
    if args.start + args.duration > length:
        args.duration = length - args.start
        print(f"note: clip trimmed to {args.duration:.1f}s to fit the source video")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    budget = args.max_mb * BYTES_PER_MB
    attempts = ladder(Settings(args.duration, args.fps, args.width))

    # TemporaryDirectory cleans up the palette even if something raises.
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for attempt, settings in enumerate(attempts, start=1):
            size = encode(args.video, args.output, args.start, settings, workdir)
            mb = size / BYTES_PER_MB
            status = "ok" if size <= budget else "over budget"
            print(f"  [{attempt}/{len(attempts)}] {settings} -> {mb:.1f} MB ({status})")
            if size <= budget:
                print(f"\nWrote {args.output} ({mb:.1f} MB)")
                return

    print(f"\nCould not reach {args.max_mb} MB. Smallest attempt is at {args.output}.")
    print("Try a shorter --duration, or pick a segment with less camera movement.")


if __name__ == "__main__":
    main()