import yt_dlp
from pydub import AudioSegment
import os
import shutil

AudioSegment.converter = shutil.which("ffmpeg")
AudioSegment.ffprobe = shutil.which("ffprobe")

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_yt_audio(url: str)-> str:
    output_template= os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts ={
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info= ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        filename = os.path.splitext(filename)[0] + ".wav"
    return filename


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


def chunk_audio(wav_path: str, chunk_minutes: int= 10) ->list:
    audio= AudioSegment.from_wav(wav_path)
    chunk_ms= chunk_minutes * 60 * 1000  # convert minutes to milliseconds
    chunks= []
     
    for i, start in enumerate(range(0,len(audio), chunk_ms)):
        chunk= audio[start: start+ chunk_ms]
        chunk_path= f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
        
    return chunks

def process_input(source: str) ->list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected yt URL , downloading audio...")
        wav_path= download_yt_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path= convert_to_wav(source)
    print("Chunking audio...")
    chunks= chunk_audio(wav_path)
    print(f"Audio ready- {len(chunks)} chunk(s) created.")
    return chunks
