#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["yt-dlp", "huggingface-hub"]
# ///
"""Download videos, a GGUF text model, and Kokoro TTS models.

Run with ``uv run download_materials.py`` to download all configured assets.
Existing files are reused by default; use --force-refresh to refresh them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import urlretrieve

import yt_dlp


VIDEO_URLS = (
    "https://www.youtube.com/watch?v=0c4KWfPhgWA", # Minecraft Parkour
    "https://www.youtube.com/watch?v=bnNwJ5H6Ek8", # Kinetic Sand
)
MODEL_ID = "unsloth/Qwen3-4B-GGUF"
MODEL_FILE = "Qwen3-4B-Q4_K_M.gguf"
KOKORO_FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin",
}
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "background_videos"
DEFAULT_MODEL_DIR = PROJECT_DIR / "text_model"
DEFAULT_TTS_MODEL_DIR = PROJECT_DIR / "tts_model"


def youtube_url(url: str) -> str:
    """Resolve a supplied YouTube URL or Google redirect to a direct watch URL."""
    parsed = urlparse(url)
    if parsed.hostname and parsed.hostname.lower().endswith("google.com"):
        target = parse_qs(parsed.query).get("url", [""])[0]
        if target:
            url = target
            parsed = urlparse(url)

    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host == "youtube.com" or host.endswith(".youtube.com"):
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    else:
        raise ValueError(f"Not a supported YouTube URL: {url}")

    if not video_id or not video_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"Could not find a valid YouTube video ID in: {url}")
    return f"https://www.youtube.com/watch?v={video_id}"


def check_model() -> int:
    """Make a lightweight metadata request to verify the HF repo is accessible."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(MODEL_ID)
    except Exception as exc:
        print(f"Hugging Face model check failed: {exc}", file=sys.stderr)
        return 1

    print(f"Model is reachable: {info.id} (revision: {info.sha})")
    return 0


def download_hf_model(repo_id: str, destination: Path, force_refresh: bool) -> int:
    """Download a Hugging Face repository snapshot into its destination."""
    try:
        from huggingface_hub import snapshot_download

        destination.mkdir(parents=True, exist_ok=True)
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(destination),
            force_download=force_refresh,
        )
    except Exception as exc:
        print(f"Hugging Face model download failed ({repo_id}): {exc}", file=sys.stderr)
        return 1

    print(f"Model files downloaded to: {path}")
    return 0


def download_model(destination: Path, force_refresh: bool) -> int:
    """Download the selected GGUF quantization from its Hugging Face model repo."""
    try:
        from huggingface_hub import hf_hub_download

        destination.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=MODEL_FILE,
            local_dir=str(destination),
            force_download=force_refresh,
        )
    except Exception as exc:
        print(f"GGUF model download failed ({MODEL_ID}/{MODEL_FILE}): {exc}", file=sys.stderr)
        return 1

    print(f"GGUF model downloaded to: {path}")
    return 0


def download_tts_model(destination: Path, force_refresh: bool) -> int:
    """Download the Kokoro ONNX model and voice pack."""
    try:
        destination.mkdir(parents=True, exist_ok=True)
        for filename, url in KOKORO_FILES.items():
            path = destination / filename
            if path.is_file() and not force_refresh:
                print(f"Already downloaded; skipping: {path}")
                continue
            partial_path = path.with_suffix(path.suffix + ".part")
            try:
                urlretrieve(url, partial_path)
                partial_path.replace(path)
            finally:
                partial_path.unlink(missing_ok=True)
            print(f"Kokoro model file downloaded to: {path}")
    except Exception as exc:
        print(f"Kokoro model download failed: {exc}", file=sys.stderr)
        return 1
    return 0


def download_videos(output_dir: Path, force_refresh: bool) -> int:
    """Download video-only streams, skipping completed files already on disk."""
    try:
        urls = [youtube_url(url) for url in VIDEO_URLS]
        output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    pending_urls = []
    for url in urls:
        video_id = parse_qs(urlparse(url).query)["v"][0]
        existing = list(output_dir.glob(f"* [{video_id}].*"))
        # Ignore incomplete yt-dlp temporary/partial files: these must resume or retry.
        complete = [p for p in existing if p.is_file() and not p.name.endswith((".part", ".ytdl"))]
        if complete and not force_refresh:
            print(f"Already downloaded; skipping {video_id}: {complete[0]}")
        else:
            pending_urls.append(url)

    if not pending_urls:
        print("All configured videos are already downloaded.")
        return 0

    options = {
        # bestvideo prefers a video-only format; best is the fallback if unavailable.
        "format": "bestvideo/best",
        "outtmpl": str(output_dir / "%(title).200B [%(id)s].%(ext)s"),
        "noplaylist": True,
        "ignoreerrors": True,
        "quiet": False,
        "overwrites": force_refresh,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        result = downloader.download(pending_urls)
    return 1 if result else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--check-model",
        action="store_true",
        help="send a lightweight metadata request to Hugging Face and exit",
    )
    actions.add_argument(
        "--download-model",
        action="store_true",
        help="download the selected GGUF file and Kokoro TTS models, then exit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"video output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"text model download directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--tts-model-dir",
        type=Path,
        default=DEFAULT_TTS_MODEL_DIR,
        help=f"TTS model download directory (default: {DEFAULT_TTS_MODEL_DIR})",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="redownload existing video and model files",
    )
    args = parser.parse_args()

    if args.check_model:
        return check_model()
    if args.download_model:
        result = download_model(args.model_dir, args.force_refresh)
        tts_result = download_tts_model(args.tts_model_dir, args.force_refresh)
        return 1 if result or tts_result else 0

    # Default action downloads videos and models.
    video_result = download_videos(args.output_dir, args.force_refresh)
    model_result = download_model(args.model_dir, args.force_refresh)
    tts_result = download_tts_model(args.tts_model_dir, args.force_refresh)
    return 1 if video_result or model_result or tts_result else 0


if __name__ == "__main__":
    raise SystemExit(main())
