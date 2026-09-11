import yt_dlp
from pydub import AudioSegment
import os
import re
import glob
import shutil

AudioSegment.converter = shutil.which("ffmpeg")
AudioSegment.ffprobe = shutil.which("ffprobe")

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Clients to try, in order. android_vr is deliberately excluded — it's the
# client that was widely 403'ing in 2026 (YouTube anti-bot changes).
CLIENT_FALLBACK_ORDER = ["android", "web", "tv", "mweb", "ios"]

COMMON_OPTS = {
    "noplaylist": True,
    "quiet": True,
    "http_headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    },
}


def _vtt_to_text(vtt_path: str) -> str:
    """Strip VTT timestamps/formatting down to plain transcript text."""
    with open(vtt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    text_lines = []
    for line in lines:
        line = line.strip()
        if not line or line == "WEBVTT":
            continue
        if "-->" in line:  # timestamp line
            continue
        if line.isdigit():  # cue number
            continue
        line = re.sub(r"<[^>]+>", "", line)  # strip inline tags like <00:00:01.000>
        text_lines.append(line)

    # de-duplicate consecutive repeated lines (common in auto-captions)
    deduped = []
    for line in text_lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)

    return " ".join(deduped)


def try_get_captions(url: str, langs=("hi", "en")) -> str | None:
    """
    Try to fetch existing/auto-generated captions instead of downloading audio.
    This avoids the audio-stream 403 issue entirely for most videos.
    Returns plain transcript text, or None if no captions are available.
    """
    output_template = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    ydl_opts = {
        **COMMON_OPTS,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": list(langs),
        "subtitlesformat": "vtt",
        "outtmpl": output_template,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            vid_id = info.get("id")

        vtt_files = glob.glob(os.path.join(DOWNLOAD_DIR, f"{vid_id}*.vtt"))
        if not vtt_files:
            return None

        text = _vtt_to_text(vtt_files[0])

        for f in vtt_files:
            os.remove(f)

        return text if text.strip() else None
    except Exception as e:
        print(f"Caption fetch failed, will fall back to audio download: {e}")
        return None


def download_yt_audio(url: str) -> str:
    output_template = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    last_error = None

    for client in CLIENT_FALLBACK_ORDER:
        ydl_opts = {
            **COMMON_OPTS,
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "extractor_args": {"youtube": {"player_client": [client]}},
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
        }
        try:
            print(f"Trying player_client='{client}'...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                filename = os.path.splitext(filename)[0] + ".wav"
            return filename
        except yt_dlp.utils.DownloadError as e:
            print(f"player_client='{client}' failed: {e}")
            last_error = e
            continue

    raise RuntimeError(
        f"Could not download audio with any client ({CLIENT_FALLBACK_ORDER}). "
        f"Last error: {last_error}"
    )


def convert_to_wav(input_path):

    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if not ffmpeg_path or not ffprobe_path:
        raise RuntimeError(
            "FFmpeg is not installed. Please install ffmpeg and ffprobe."
        )

    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path

    audio = AudioSegment.from_file(input_path)

    output_path = "output.wav"
    audio.export(output_path, format="wav")

    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000  # convert minutes to milliseconds
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str):
    """
    Returns either:
      - a string (transcript text, when fetched directly from captions), or
      - a list of audio chunk paths (when Whisper transcription is needed)

    Callers (e.g. run_pipeline) should check the return type: if it's a str,
    skip Whisper and feed it straight into translation/chunking-for-RAG. If
    it's a list, run the existing Whisper transcription step on the chunks.
    """
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected yt URL. Trying captions first...")
        transcript = try_get_captions(source)
        if transcript:
            print("Got captions directly — skipping audio download & Whisper.")
            return transcript

        print("No captions available. Falling back to audio download...")
        wav_path = download_yt_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks