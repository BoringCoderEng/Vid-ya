import whisper
import os
import requests
from pydub import AudioSegment
import streamlit as st

SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL= os.getenv("WHISPER_MODEL", "base")


SARVAM_API_KEY= os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    try: SARVAM_API_KEY= st.secrets["SARVAM_API_KEY"]
    except Exception:
        raise RuntimeError(
            "SARVAM_API_KEY is not sent in environment/ .env/ streamlit secrets"
        )
SARVAM_STT_TRANSLATE_URL="https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL= os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")


_model=None

def load_model():
    global _model
    if _model is None:
        _model= whisper.load_model(WHISPER_MODEL)
        
    return _model

def transcribe_chunk_whisper(chunk_path: str, translate: bool= False)-> str:
    model= load_model()
    task= "translate" if translate else "transcribe"
    
    result= model.transcribe(chunk_path, task=task)
    
    return result["text"]


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        raise RuntimeError(
            f"Sarvam API error {response.status_code}: {response.text}"
        )
    return response.json().get("transcript", "")






def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()



def transcribe_chunk(chunk_path: str, language: str="english"):
    if language.lower() =="hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    
    return transcribe_chunk_whisper(chunk_path)
    
def transcribe_all(chunks: list, language: str="english")-> str:
    full_transcription= ""
    
    engine= "SARVAM AI"if language.lower()== "hinglish" else "Whisper"
    print(f"Using {engine} for transcription")
    
    for chunk_path in chunks:
        transcription= transcribe_chunk(chunk_path, language)
        full_transcription += transcription + " "
        
    return full_transcription