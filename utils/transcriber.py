import os
from groq import Groq

def transcribe_audio(file_path: str) -> str:
    """
    Sends the local audio file to the Groq API for transcription using Whisper-large-v3.
    """
    client = Groq()
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at: {file_path}")
        
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(file_path), audio_file.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
        )
    return transcription.text