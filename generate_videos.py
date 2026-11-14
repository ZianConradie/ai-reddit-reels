#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["llama-cpp-python", "kokoro-onnx", "soundfile", "python-dotenv"]
# ///
"""Generate narrated AITA-style vertical videos from local CPU models.

Text generation uses a GGUF through llama.cpp. Speech uses Kokoro ONNX and its
model files downloaded into ./tts_model. Requires ffmpeg/ffprobe.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_DIR = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_DIR / ".env")
except ImportError:
    # CLI and shell environment configuration remain usable if dotenv is absent.
    pass

DEFAULT_TEXT_MODEL_DIR = PROJECT_DIR / "text_model"
DEFAULT_TEXT_MODEL_FILE = DEFAULT_TEXT_MODEL_DIR / "Qwen3-4B-Q4_K_M.gguf"
DEFAULT_TTS_MODEL_DIR = PROJECT_DIR / "tts_model"
DEFAULT_TTS_MODEL_FILE = DEFAULT_TTS_MODEL_DIR / "kokoro-v1.0.onnx"
DEFAULT_TTS_VOICES_FILE = DEFAULT_TTS_MODEL_DIR / "voices-v1.0.bin"
DEFAULT_VIDEO_DIR = PROJECT_DIR / "background_videos"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output_videos"
DEFAULT_N_CTX = 4096
DEFAULT_TEXT_PROVIDER = os.getenv("TEXT_MODEL_PROVIDER", "local")
DEFAULT_TTS_PROVIDER = os.getenv("TTS_MODEL_PROVIDER", "local")
SUPPORTED_PROVIDERS = {"local", "openai-compatible"}

SYSTEM_PROMPT = """Write an original, fictional AITA-style story for social-media narration. Aim for about 240 to 320 spoken words (roughly twice the previous length). Use natural first-person Reddit-post voice, concrete details, a clear dilemma, and an ending that asks whether the narrator was wrong.

Before writing the final story, silently complete these three passes:
1. ROUGH DRAFT: establish the characters, relationships, setting, timeline, initial situation, important information, and the narrator's decision.
2. REREAD AND MARK: check the draft for missing setup, unclear references, unexplained actions, contradictions, timeline gaps, and facts introduced too late. Mark what needs clarification, then revise so every detail needed to understand a later event is introduced before it matters. In particular, make the cause and context of each consequential action clear before the reaction to it, so listeners never have a "huh, when did that happen?" moment.
3. FINAL DRAFT: rewrite as one coherent, chronological story with the fixes integrated naturally. Do not reveal the rough draft, checklist, markings, analysis, or these pass labels.

Output only the final story as one paragraph of plain spoken text. No heading, markdown, labels, stage directions, sound effects, parenthetical actions, or quotation marks around the whole story."""


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required command not found: {name}")
    return path


def openai_compatible_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0.9,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint without logging secrets."""
    if not base_url:
        raise RuntimeError("Provide --text-base-url or TEXT_MODEL_BASE_URL for the text API provider.")
    if not model:
        raise RuntimeError("Provide --text-api-model or TEXT_MODEL_API_MODEL for the text API provider.")
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=payload, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OpenAI-compatible API request failed: {exc}") from exc
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenAI-compatible API returned an unexpected chat completion response.") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenAI-compatible API returned empty story text.")
    return content.strip()


def load_text_model(model_path: Path, n_ctx: int):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("llama-cpp-python is missing; run with `uv run generate_videos.py`.") from exc
    if not model_path.is_file():
        raise RuntimeError(
            f"GGUF text model not found: {model_path}. Run `uv run download_materials.py --download-model` first."
        )
    return Llama(model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=0, verbose=False)


def generate_story(
    model_path: Path,
    rng: random.Random,
    n_ctx: int,
    provider: str,
    api_base_url: str,
    api_key: str,
    api_model: str,
    local_model: Any = None,
) -> str:
    # These broad fictionalized conflicts are inspiration only, not source stories.
    topic = rng.choice(
        [
            "a parent sets a new privacy boundary for a teenager, who feels unfairly judged",
            "a couple hosts a family gathering but declines overnight guests, upsetting relatives",
            "a small clothing mishap among friends is interpreted as disrespectful",
            "a temporary houseguest criticizes household rules agreed on beforehand",
            "partners disagree about lending their only reliable vehicle to an uninsured acquaintance",
            "someone declines a visit request from a former pet owner who previously ignored boundaries",
            "roommates disagree about intervening during a health-related emergency",
            "partners disagree about advance notice before relatives visit",
            "expectant parents disagree with relatives about how much personal news to share",
            "roommates disagree about how often partners can stay overnight",
            "family members with dietary needs arrange different food after a shared meal does not work",
            "someone chooses a personal commitment over attending a distant relative's event",
            "a person plans a lower-cost outing instead of paying for an expanded family gathering",
            "a homeowner mistakes a worker arriving early for an intruder",
            "a friend gives blunt feedback about a coworker's hygiene",
            "relatives argue over whether a meaningful family name can be reused",
            "an artist feels a compliment overlooks the effort behind their work",
            "a visitor tracks dirt into a recently cleaned home",
            "one friend grows resentful after becoming the default driver for a group",
            "friends debate whether to include a chronically disruptive member in a weekend plan",
            "a family organiser decides whether to invite a relative with a history of derailing events",
            "housemates disagree about laundry and shared chores after an unpleasant discovery",
        ]
    )
    prompt = (
        f"Use this conflict only as loose inspiration: {topic}. Create a new fictional scenario with "
        "different characters, relationships, setting, and specific events; do not retell or closely copy "
        "any example. Avoid weddings or travel as a default. Make the narrator's choice and the other "
        "person's reaction clear, with concrete details and a plausible dilemma. Aim for 240 to 320 spoken "
        "words. Before each consequential action or reaction, establish the relevant information, relationship, "
        "timeline, or prior event the listener needs to understand it. Review for missing setup, unclear "
        "references, contradictions, and late-introduced facts; quietly revise those issues, then output only "
        "the final story. Do not include draft notes or analysis. /no_think"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    if provider == "openai-compatible":
        story = openai_compatible_chat(
            base_url=api_base_url,
            api_key=api_key,
            model=api_model,
            messages=messages,
            max_tokens=640,
            temperature=0.9,
        )
    elif provider == "local":
        if local_model is None:
            local_model = load_text_model(model_path, n_ctx)
        model_seed = rng.randrange(0, 2**32)
        response = local_model.create_chat_completion(
            messages=messages,
            max_tokens=640,
            temperature=0.9,
            top_p=0.95,
            repeat_penalty=1.12,
            seed=model_seed,
        )
        story = (response["choices"][0]["message"]["content"] or "").strip()
    else:
        raise RuntimeError(f"Unsupported text provider: {provider}")

    # Never send model reasoning to TTS.
    story = re.sub(r"<think>.*?</think>", " ", story, flags=re.IGNORECASE | re.DOTALL)
    story = re.sub(r"<think>.*$", " ", story, flags=re.IGNORECASE | re.DOTALL)
    story = re.sub(r"</?think>", " ", story, flags=re.IGNORECASE)
    story = re.sub(r"^\s*(?:story|aita|title)\s*:\s*", "", story, flags=re.IGNORECASE)
    story = re.sub(r"\s*```(?:\w+)?\s*|\s*```\s*", " ", story)
    story = " ".join(story.split()).strip(" \"'`*_#")
    story = re.sub(r"\([^)]{0,80}\)|\*[^*]{1,80}\*|\[[^]]{1,80}\]", "", story)
    story = re.sub(r"\s+", " ", story).strip()
    if len(story.split()) < 35:
        raise RuntimeError(f"Text model returned too little usable story text: {story!r}")
    return story


def run_tts(
    story: str,
    provider: str,
    model_path: Path,
    voices_path: Path,
    output_wav: Path,
    api_base_url: str,
    api_key: str,
    api_model: str,
    api_voice: str,
) -> None:
    if provider == "openai-compatible":
        if not api_base_url:
            raise RuntimeError("Provide --tts-base-url or TTS_MODEL_BASE_URL for the TTS API provider.")
        if not api_model:
            raise RuntimeError("Provide --tts-api-model or TTS_MODEL_API_MODEL for the TTS API provider.")
        url = api_base_url.rstrip("/")
        nano_gpt_endpoint = url.endswith("/api/tts")
        if not nano_gpt_endpoint and not url.endswith("/audio/speech"):
            url += "/audio/speech"
        request_body = {"model": api_model, "input": story}
        # Nano-GPT's /api/tts endpoint accepts only model and input; voice and
        # response_format are not part of its documented request schema.
        if not nano_gpt_endpoint:
            request_body.update({"voice": api_voice, "response_format": "wav"})
        payload = json.dumps(request_body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = Request(url, data=payload, headers=headers, method="POST")
        audio = b""
        for attempt in range(3):
            try:
                with urlopen(request, timeout=300) as response:
                    audio = response.read()
                break
            except HTTPError as exc:
                detail = exc.read(2000).decode("utf-8", errors="replace").strip()
                if exc.code in {429, 500, 502, 503, 504} and attempt < 2:
                    delay = 2 ** attempt
                    print(
                        f"TTS API returned HTTP {exc.code}; retrying in {delay}s "
                        f"({attempt + 1}/2). {detail[:300]}",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
                message = f"OpenAI-compatible TTS API request failed: HTTP {exc.code} {exc.reason}"
                if detail:
                    message += f"; response: {detail[:1000]}"
                raise RuntimeError(message) from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt < 2:
                    delay = 2 ** attempt
                    print(
                        f"TTS API connection failed; retrying in {delay}s ({attempt + 1}/2): {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"OpenAI-compatible TTS API request failed after 3 attempts: {exc}") from exc
        if not audio:
            raise RuntimeError("OpenAI-compatible TTS API returned empty audio.")
        # TTS APIs may return audio directly or JSON containing a URL/base64.
        content_type = response.headers.get("Content-Type", "").lower()
        if "json" in content_type or audio.lstrip().startswith(b"{"):
            try:
                result = json.loads(audio.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("TTS API returned malformed JSON instead of audio.") from exc
            encoded = result.get("audio") or result.get("audio_base64") or result.get("data")
            audio_url = result.get("url") or result.get("audio_url")
            if isinstance(encoded, str):
                import base64

                audio = base64.b64decode(encoded)
            elif isinstance(audio_url, str):
                try:
                    with urlopen(audio_url, timeout=300) as audio_response:
                        audio = audio_response.read()
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    raise RuntimeError(f"Could not download generated audio URL: {exc}") from exc
            else:
                raise RuntimeError(f"TTS API returned JSON without audio bytes or URL: {str(result)[:500]}")
        # Some compatible services return MP3 or another format despite the
        # requested WAV. Convert based on file signature so downstream ffprobe
        # and video muxing always receive a WAV file.
        if audio.startswith(b"RIFF") and audio[8:12] == b"WAVE":
            output_wav.write_bytes(audio)
        else:
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("TTS API response is not WAV and ffmpeg is unavailable to convert it.")
            converted = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", "pipe:0", "-f", "wav", "pipe:1"],
                input=audio,
                capture_output=True,
                check=False,
            )
            if converted.returncode:
                detail = converted.stderr.decode("utf-8", errors="replace")[-1000:]
                raise RuntimeError(f"TTS API response was not a valid WAV or convertible audio: {detail}")
            output_wav.write_bytes(converted.stdout)
    elif provider == "local":
        try:
            import soundfile as sf
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise RuntimeError(
                "Kokoro TTS dependencies are missing. Run `uv run generate_videos.py --count 1` to install them."
            ) from exc

        if not model_path.is_file() or not voices_path.is_file():
            raise RuntimeError(
                f"Kokoro files not found. Expected {model_path} and {voices_path}; "
                "run `uv run download_materials.py --download-model` first."
            )
        tts = Kokoro(str(model_path), str(voices_path))
        samples, sample_rate = tts.create(story, voice="af_sarah", speed=1.0, lang="en-us")
        sf.write(str(output_wav), samples, sample_rate)
    else:
        raise RuntimeError(f"Unsupported TTS provider: {provider}")

    if not output_wav.is_file() or output_wav.stat().st_size == 0:
        raise RuntimeError("TTS provider did not produce an audio file.")


def probe_duration(ffprobe: str, path: Path) -> float:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Could not read media duration for {path}") from exc
    if duration <= 0:
        raise RuntimeError(f"Invalid media duration for {path}: {duration}")
    return duration


def ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def escape_ass(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def make_word_subtitles(story: str, duration: float, destination: Path) -> None:
    words = re.findall(r"\S+", story)
    # Approximate word-level timing from spoken-text length; no forced-alignment model is assumed.
    weights = [max(1, len(re.sub(r"[^\w']", "", word))) for word in words]
    total_weight = sum(weights)
    starts: list[float] = []
    ends: list[float] = []
    cursor = 0.0
    for weight in weights:
        starts.append(cursor)
        cursor += duration * weight / total_weight
        ends.append(cursor)

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,64,&H00FFFFFF,&H0000FFFF,&H00101010,&H80000000,-1,0,1,5,1,5,70,70,960,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    # Each visible caption has one active word at a time, split into short readable chunks.
    chunk_size = 6
    for first in range(0, len(words), chunk_size):
        last = min(first + chunk_size, len(words))
        karaoke = []
        for index in range(first, last):
            cs = max(1, round((ends[index] - starts[index]) * 100))
            karaoke.append(f"{{\\k{cs}}}{escape_ass(words[index])}")
        lines.append(
            f"Dialogue: 0,{ass_timestamp(starts[first])},{ass_timestamp(ends[last - 1])},Default,,0,0,0,,"
            + " ".join(karaoke)
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ffmpeg_filter_path(path: Path) -> str:
    # Escape characters that have special meaning inside an ffmpeg filter argument.
    value = path.resolve().as_posix()
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace(",", "\\,").replace("[", "\\[").replace("]", "\\]")


def create_video(
    ffmpeg: str,
    ffprobe: str,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    story: str,
    rng: random.Random,
) -> None:
    audio_duration = probe_duration(ffprobe, audio_path)
    video_duration = probe_duration(ffprobe, video_path)
    if video_duration < audio_duration:
        raise RuntimeError(
            f"Background clip is too short ({video_duration:.1f}s) for narration ({audio_duration:.1f}s): {video_path}"
        )
    start = rng.uniform(0, max(0.0, video_duration - audio_duration))
    with tempfile.TemporaryDirectory(prefix="reels-video-") as temp_name:
        subtitle_path = Path(temp_name) / "captions.ass"
        make_word_subtitles(story, audio_duration, subtitle_path)
        video_filter = (
            f"crop='min(iw,ih*9/16)':'min(ih,iw*16/9)':(iw-ow)/2:(ih-oh)/2,"
            f"subtitles='{ffmpeg_filter_path(subtitle_path)}'"
        )
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-t",
                f"{audio_duration:.3f}",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-vf",
                video_filter,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "21",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1, help="number of independent videos to generate (default: 1)")
    parser.add_argument(
        "--text-model",
        type=Path,
        default=DEFAULT_TEXT_MODEL_FILE,
        help=f"local GGUF text model file (default: {DEFAULT_TEXT_MODEL_FILE})",
    )
    parser.add_argument("--context-size", type=int, default=DEFAULT_N_CTX, help="llama.cpp context size (default: 4096)")
    parser.add_argument(
        "--text-provider",
        choices=("local", "openai-compatible"),
        default=DEFAULT_TEXT_PROVIDER,
        help="story generator backend (default: local; may also use TEXT_MODEL_PROVIDER)",
    )
    parser.add_argument("--text-base-url", default=os.getenv("TEXT_MODEL_BASE_URL", ""), help="OpenAI-compatible text API base URL, e.g. http://localhost:1234/v1")
    parser.add_argument("--text-api-model", default=os.getenv("TEXT_MODEL_API_MODEL", ""), help="model name required by the text API")
    parser.add_argument("--text-api-key", default=os.getenv("TEXT_MODEL_API_KEY", ""), help="text API key; may also be set via TEXT_MODEL_API_KEY")
    parser.add_argument(
        "--tts-model",
        type=Path,
        default=DEFAULT_TTS_MODEL_FILE,
        help=f"Kokoro ONNX model path (default: {DEFAULT_TTS_MODEL_FILE})",
    )
    parser.add_argument(
        "--tts-voices",
        type=Path,
        default=DEFAULT_TTS_VOICES_FILE,
        help=f"Kokoro voice pack path (default: {DEFAULT_TTS_VOICES_FILE})",
    )
    parser.add_argument(
        "--tts-provider",
        choices=("local", "openai-compatible"),
        default=DEFAULT_TTS_PROVIDER,
        help="speech synthesis backend (default: local; may also use TTS_MODEL_PROVIDER)",
    )
    parser.add_argument("--tts-base-url", default=os.getenv("TTS_MODEL_BASE_URL", ""), help="OpenAI-compatible TTS API base URL, e.g. http://localhost:1234/v1")
    parser.add_argument("--tts-api-model", default=os.getenv("TTS_MODEL_API_MODEL", ""), help="model name required by the TTS API")
    parser.add_argument("--tts-api-key", default=os.getenv("TTS_MODEL_API_KEY", ""), help="TTS API key; may also be set via TTS_MODEL_API_KEY")
    parser.add_argument("--tts-api-voice", default=os.getenv("TTS_MODEL_API_VOICE", "alloy"), help="voice name for OpenAI-compatible TTS API")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, help="optional seed for repeatable topic/video choices")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 1:
        print("--count must be at least 1", file=sys.stderr)
        return 2
    if args.context_size < 512:
        print("--context-size must be at least 512", file=sys.stderr)
        return 2
    for option, provider in (("--text-provider", args.text_provider), ("--tts-provider", args.tts_provider)):
        if provider not in {"local", "openai-compatible"}:
            print(f"Invalid {option}: {provider}", file=sys.stderr)
            return 2
    try:
        ffmpeg = require_tool("ffmpeg")
        ffprobe = require_tool("ffprobe")
        for directory, label in (
            (args.video_dir, "background video"),
        ):
            if not directory.is_dir():
                raise RuntimeError(f"{label} directory does not exist: {directory}")
        videos = sorted(
            path for path in args.video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm", ".m4v"}
        )
        if not videos:
            raise RuntimeError(f"No background video files found in {args.video_dir}")
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        run_seed = args.seed if args.seed is not None else time.time_ns() % (2**32)
        print(f"Using generation seed: {run_seed}", flush=True)
        rng = random.Random(run_seed)
        stage_dir = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.staging-", dir=args.output_dir.parent))
        stage_dir.chmod(0o755)
        output_dir = args.output_dir.resolve()
        backup_dir = output_dir.with_name(f".{output_dir.name}.backup-{uuid.uuid4().hex}")
        published = False
        try:
            local_model = load_text_model(args.text_model, args.context_size) if args.text_provider == "local" else None
            seen_stories: set[str] = set()
            max_attempts = args.count * 8
            created = 0
            attempts = 0
            while created < args.count and attempts < max_attempts:
                attempts += 1
                number = created + 1
                print(f"[{number}/{args.count}] Generating a unique story (attempt {attempts}/{max_attempts})...", flush=True)
                story = generate_story(
                    args.text_model,
                    rng,
                    args.context_size,
                    args.text_provider,
                    args.text_base_url,
                    args.text_api_key,
                    args.text_api_model,
                    local_model,
                )
                canonical_story = " ".join(re.findall(r"[a-z0-9]+", story.casefold()))
                if canonical_story in seen_stories:
                    print(f"[{number}/{args.count}] Duplicate story rejected; retrying.", file=sys.stderr, flush=True)
                    continue
                seen_stories.add(canonical_story)
                print(f"[{number}/{args.count}] Generating narration ({len(story.split())} words)...", flush=True)
                with tempfile.TemporaryDirectory(prefix="reels-audio-") as temp_name:
                    audio_path = Path(temp_name) / "narration.wav"
                    run_tts(
                        story,
                        args.tts_provider,
                        args.tts_model,
                        args.tts_voices,
                        audio_path,
                        args.tts_base_url,
                        args.tts_api_key,
                        args.tts_api_model,
                        args.tts_api_voice,
                    )
                    video_path = rng.choice(videos)
                    output_path = stage_dir / f"video{number}.mp4"
                    print(f"[{number}/{args.count}] Using background: {video_path.name}", flush=True)
                    create_video(ffmpeg, ffprobe, video_path, audio_path, output_path, story, rng)
                    if not output_path.is_file() or output_path.stat().st_size == 0:
                        raise RuntimeError(f"Video generation produced an empty file: {output_path}")
                    print(f"[{number}/{args.count}] Staged: {output_path.name}", flush=True)
                created += 1
            staged_videos = list(stage_dir.glob("*.mp4"))
            if created != args.count or len(staged_videos) != args.count:
                raise RuntimeError(f"Could not produce {args.count} unique videos; produced {len(staged_videos)}. Existing outputs were preserved.")
            if output_dir.exists():
                output_dir.rename(backup_dir)
            try:
                stage_dir.rename(output_dir)
                published = True
            except Exception:
                if backup_dir.exists():
                    backup_dir.rename(args.output_dir)
                raise
            if backup_dir.exists():
                try:
                    shutil.rmtree(backup_dir)
                except OSError as exc:
                    print(f"Warning: new batch is ready; could not remove old backup {backup_dir}: {exc}", file=sys.stderr, flush=True)
            print(f"Published exactly {args.count} unique videos to {args.output_dir}", flush=True)
            return 0
        finally:
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
            if not published and backup_dir.exists() and not output_dir.exists():
                backup_dir.rename(output_dir)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
