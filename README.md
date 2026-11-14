# ReelsRedditVideos

Generate vertical, narrated story videos with local models or OpenAI-compatible text and speech APIs. The project uses `yt-dlp` for background footage, `llama.cpp` for local GGUF text generation, Kokoro ONNX for local speech, and FFmpeg to assemble the final videos.

> **Content note:** The generator creates fictional AITA-style stories. It does not fetch or reproduce Reddit posts.

## Requirements

- Linux or macOS (the included systemd units are Linux-specific)
- [uv](https://docs.astral.sh/uv/)
- FFmpeg, including `ffmpeg` and `ffprobe`, available on `PATH`
- For local text generation: enough RAM for the selected GGUF model
- For API providers: a reachable compatible text endpoint and/or speech endpoint

Python dependencies for each script are declared inline and are installed/managed by `uv run --script`.

## Setup

1. Clone the repository and enter its directory.
2. Install `uv` and FFmpeg using your operating system's package manager or the upstream installation instructions.
3. (Optional) Copy `.env.example` to `.env` and set provider/API configuration. Keep `.env` private; it is ignored by Git.
4. Download background video and local model assets as needed:

   ```sh
   uv run download_materials.py
   ```

   This downloads the configured background videos, the Qwen3 4B GGUF, and Kokoro ONNX files. Model and video assets are several gigabytes and are intentionally not stored in Git. To download only the GGUF and TTS assets:

   ```sh
   uv run download_materials.py --download-model
   ```

   To refresh existing downloads, add `--force-refresh`.

## Generate videos

Generate one video (the default):

```sh
uv run --script generate_videos.py
```

Generate multiple videos or use a repeatable seed:

```sh
uv run --script generate_videos.py --count 3 --seed 1234
```

By default, the script reads local models from `text_model/` and `tts_model/`, background footage from `background_videos/`, and writes finished videos to `output_videos/`. Output files are excluded from Git. See all options with:

```sh
uv run --script generate_videos.py --help
uv run download_materials.py --help
```

## API providers

Text generation and speech can be configured independently. Set the provider and the corresponding API endpoint/model in `.env` (or export the same variables in your shell):

```dotenv
TEXT_MODEL_PROVIDER=openai-compatible
TEXT_MODEL_BASE_URL=https://your-text-service.example/v1
TEXT_MODEL_API_KEY=your-text-api-key
TEXT_MODEL_API_MODEL=your-text-model

TTS_MODEL_PROVIDER=openai-compatible
TTS_MODEL_BASE_URL=https://your-speech-service.example/v1
TTS_MODEL_API_KEY=your-speech-api-key
TTS_MODEL_API_MODEL=your-speech-model
TTS_MODEL_API_VOICE=alloy
```

Use a service's documented base URL and model identifier. Do not commit real API keys. Local providers remain the defaults; API mode does not require downloading the corresponding local model, but background videos and FFmpeg are still needed.

## Optional systemd timer

The repository includes `reelsredditvideos-generate.service` and `.timer` as examples for hourly generation on Linux. The service currently assumes the project is at `~/Desktop/ReelsRedditVideos` and `uv` at `~/.local/bin/uv`; edit those paths to match your installation before enabling the units. It generates ten videos per run and expects `.env`, downloaded assets, and FFmpeg to be available to the service user.

## Asset and content notes

- Background footage and model files are downloaded separately and are not included in this repository. Review and comply with each asset's license and the relevant service terms before use or redistribution.
- Generated stories are fictional. Review generated audio/video and ensure your publishing workflow follows applicable platform rules and disclosure requirements.
- Local GGUF and ONNX model files can require multiple gigabytes of disk space.

## License

No project license has been selected yet. Until a `LICENSE` file is added, the repository is not granting permission to redistribute or reuse the project. Add a license before publishing if you intend to make it open source.
