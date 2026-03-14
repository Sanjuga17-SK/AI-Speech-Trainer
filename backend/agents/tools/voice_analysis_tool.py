import os
import json
import tempfile
import numpy as np
import librosa
from moviepy import VideoFileClip
from faster_whisper import WhisperModel
from agno.tools import tool
from dotenv import load_dotenv

load_dotenv()

def extract_audio_from_video(video_path: str, output_audio_path: str) -> str:
    """
    Extracts audio from a video file and saves it as an audio file.

    Args:
        video_path: Path to the input video file.
        output_audio_path: Path to save the extracted audio file.

    Returns:
        Path to the extracted audio file.
    """
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_path)
    audio_clip.close()
    video_clip.close()
    return output_audio_path

def load_whisper_model():
    try:
        model = WhisperModel("small", device="cpu", compute_type="int8")
        return model
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return None
    
def transcribe_audio(audio_file):
    """
    Transcribe the audio file using faster-whisper.
    
    Returns:
        str: Transcribed text or error/fallback message.
    """
    if not audio_file or not os.path.exists(audio_file):
        return "No audio file exists at the specified path."

    model = load_whisper_model()
    if not model:
        return "Error: Could not load Whisper model."

    segments, info = model.transcribe(audio_file, beam_size=5)
    transcription = " ".join([segment.text for segment in segments])
    return transcription

@tool(
    name="analyze_voice_attributes",
    description="Analyzes voice attributes such as speech rate, pitch, and volume, and provides a transcript.",
    show_result=True,
    stop_after_tool_call=True
)
def analyze_voice_attributes(video_path: str) -> str:
    """
    Analyzes voice attributes in an audio/video file to detect speech rate, pitch, and volume.

    Args:
        video_path: Path to the input video or audio file.

    Returns:
        A JSON string containing the transcribed text and vocal metrics.
    """
    ext = os.path.splitext(video_path)[1].lower()
    audio_path = video_path

    if ext in ['.mp4', '.mov', '.avi', '.mkv']:
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio_path = extract_audio_from_video(video_path, temp_audio.name)
        temp_audio.close()

    # Transcribe
    transcription = transcribe_audio(audio_path)

    # Vocal Analysis
    y, sr = librosa.load(audio_path)
    
    # Extract words for speech rate
    words = transcription.split()
    
    # Calculate speech rate
    duration = librosa.get_duration(y=y, sr=sr)
    speech_rate = len(words) / (duration / 60.0) if duration > 0 else 0 # words per minute

    # Pitch variation
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_variation = np.std(pitch_values) if pitch_values.size > 0 else 0

    # Volume consistency
    rms = librosa.feature.rms(y=y)[0]
    volume_consistency = np.std(rms)

    # Clean up temporary audio file if created
    if ext in ['.mp4', '.mov', '.avi', '.mkv'] and os.path.exists(audio_path):
        os.remove(audio_path)

    return json.dumps({
        "transcription": transcription,
        "speech_rate_wpm": str(round(speech_rate, 2)),
        "pitch_variation": str(round(pitch_variation, 2)),
        "volume_consistency": str(round(volume_consistency, 4))
    })