#!/usr/bin/env python3
"""
process_videos.py — Index and convert demo videos for DINO-R.

Scans the videos/ directory, converts any non-H.264 MP4s to H.264 with
browser-optimized settings (-movflags +faststart for progressive streaming),
and writes videos.json for the demo page to load dynamically.

Requirements: ffmpeg and ffprobe must be installed and on PATH.
  Windows: https://www.gyan.dev/ffmpeg/builds/  (add bin/ to PATH)
  macOS:   brew install ffmpeg
  Linux:   apt install ffmpeg

Usage:
    python process_videos.py                    # index only → writes videos.json
    python process_videos.py --convert          # index + convert all non-H.264 files
    python process_videos.py --convert --preset fast --crf 20

After adding new videos, just re-run the script. The JSON is rewritten every run.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


# ── Dataset display metadata ──────────────────────────────────────────────────
# Keys must match the subdirectory names inside videos/.
# Add a new entry here when you add a new dataset folder.
DATASET_META: dict[str, dict] = {
    "cityscapes": {
        "label": "Cityscapes",
        "desc":  "Urban driving scenes from European cities — used as in-distribution training data.",
    },
    "nuscenes": {
        "label": "nuScenes",
        "desc":  "Multi-camera autonomous driving sequences from diverse driving environments.",
    },
    "bdd100k": {
        "label": "bdd100k",
        "desc":  "Large-scale driving video dataset with diverse weather and lighting conditions.",
    },
}

# Preferred tab order in the UI (unknown folders are appended alphabetically)
DATASET_ORDER = ["cityscapes", "nuscenes", "bdd100k"]


# ── ffmpeg helpers ────────────────────────────────────────────────────────────

def check_ffmpeg() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        sys.exit(
            f"Error: {', '.join(missing)} not found on PATH.\n"
            "Install ffmpeg (https://ffmpeg.org/download.html) and try again."
        )


def get_codec(path: Path) -> str:
    """Return the video codec name reported by ffprobe, e.g. 'h264', 'hevc'."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def convert_to_h264(src: Path, crf: int, preset: str) -> None:
    """
    Re-encode *src* to H.264/MP4 in-place.

    Key flags:
      -pix_fmt yuv420p    — broadest browser/device compatibility (required for iOS Safari)
      -movflags +faststart — moves moov atom to the start of the file so the browser
                             can begin playback before the full file is downloaded
                             (progressive streaming over plain HTTP)
      -an                 — strip audio track (demo videos are muted; reduces file size)
    """
    tmp = src.with_suffix(".converting.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(src),
                "-vcodec", "libx264",
                "-crf", str(crf),
                "-preset", preset,
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                str(tmp),
            ],
            check=True,
        )
        tmp.replace(src)  # atomic rename; replaces the original
        print(f"    ✓  {src.name}")
    except subprocess.CalledProcessError:
        tmp.unlink(missing_ok=True)
        raise


# ── Directory scan ────────────────────────────────────────────────────────────

def sort_key(d: Path) -> tuple:
    try:
        return (0, DATASET_ORDER.index(d.name))
    except ValueError:
        return (1, d.name)


def scan_and_convert(
    videos_dir: Path,
    convert: bool,
    crf: int,
    preset: str,
) -> list[dict]:
    dirs = sorted(
        (d for d in videos_dir.iterdir() if d.is_dir()),
        key=sort_key,
    )

    datasets = []
    for ds_dir in dirs:
        ds_name = ds_dir.name
        meta = DATASET_META.get(
            ds_name,
            {"label": ds_name, "desc": ""},
        )

        mp4_files = sorted(ds_dir.glob("*.mp4"))
        if not mp4_files:
            print(f"  [{ds_name}] no .mp4 files found — skipped")
            continue

        if convert:
            print(f"  [{ds_name}]")
            for vf in mp4_files:
                codec = get_codec(vf)
                if codec == "h264":
                    print(f"    –  {vf.name}  (already H.264, skipped)")
                else:
                    print(f"    ↻  {vf.name}  ({codec} → h264) …")
                    convert_to_h264(vf, crf=crf, preset=preset)

        videos = []
        for vf in mp4_files:
            web_path = "/".join(vf.relative_to(videos_dir.parent).parts)
            videos.append({"src": web_path})

        datasets.append(
            {
                "id":     ds_name,
                "label":  meta["label"],
                "desc":   meta["desc"],
                "videos": videos,
            }
        )
        print(f"  [{ds_name}] indexed {len(videos)} video(s)")

    return datasets


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index and convert DINO-R demo videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--videos-dir", default="videos", metavar="DIR",
        help="Root folder containing dataset subfolders (default: videos)",
    )
    parser.add_argument(
        "--output", default="videos.json", metavar="FILE",
        help="Output JSON path (default: videos.json)",
    )
    parser.add_argument(
        "--convert", action="store_true", default=True,
        help="Convert non-H.264 videos to H.264 in-place (requires ffmpeg)",
    )
    parser.add_argument(
        "--crf", type=int, default=23, metavar="N",
        help="H.264 quality: 0=lossless, 51=worst, 23=default (lower = better quality, larger file)",
    )
    parser.add_argument(
        "--preset", default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                 "medium", "slow", "slower", "veryslow"],
        help="ffmpeg encoding speed preset (default: medium). "
             "Slower preset = smaller file at same quality.",
    )
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    if not videos_dir.is_dir():
        sys.exit(f"Error: '{videos_dir}' is not a directory.")

    if args.convert:
        check_ffmpeg()

    print(f"Scanning '{videos_dir}/' …")
    datasets = scan_and_convert(videos_dir, args.convert, args.crf, args.preset)

    if not datasets:
        print("No datasets found. Nothing written.")
        return

    out = Path(args.output)
    out.write_text(
        json.dumps({"datasets": datasets}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {out}  ({len(datasets)} dataset(s), "
          f"{sum(len(d['videos']) for d in datasets)} video(s) total)")
    print("\nNote: 'rate' values are set to '—'. Edit videos.json to fill in the")
    print("      actual OOD rates (e.g. \"rate\": \"8.35%\") for each video.")
    if not args.convert:
        print("\nTip: run with --convert to also transcode non-H.264 videos.")


if __name__ == "__main__":
    main()
