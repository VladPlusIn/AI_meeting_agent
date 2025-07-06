# coding: utf-8
"""Real-time speech recognition and meeting analysis."""
import threading
from datetime import datetime
import json
import os

import numpy as np
import pyaudio
import whisper
import noisereduce as nr
from openai import OpenAI

CONFIG_FILE = "config.json"


def load_or_create_config():
    """Load configuration, prompting for an API key if necessary."""
    config = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error reading config file: {e}")

    if not config.get("api_key"):
        api_key = input("Please enter your OpenAI API key: ").strip()
        config["api_key"] = api_key
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
            print(f"Config file '{CONFIG_FILE}' updated with the API key.")
        except Exception as e:
            print(f"Error saving config file: {e}")
    return config


config = load_or_create_config()
API_KEY = config.get("api_key")

# Global variables for session data
session_transcriptions = []
session_lock = threading.Lock()
session_timestamp = datetime.now().isoformat()


def analyze_conversation(transcriptions):
    """Analyze the conversation using GPT-4 via OpenAI."""
    conversation_text = "\n".join(transcriptions)
    prompt = f'''
You are an expert meeting analyst. Analyze the following meeting transcription and provide:
1. A meeting title (subject) summarizing the conversation.
2. A summary of the conversation in bullet points.
3. Any tasks discussed during the conversation (if any).

Format your answer as a JSON object with the following keys: "meeting_title", "summary", "discussed_tasks".

Meeting transcription:
"""{conversation_text}"""
'''
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEY,
        )
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert meeting analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=500,
        )
        analyzed_data = response.choices[0].message.content
        return json.loads(analyzed_data)
    except Exception as e:
        print(f"Error analyzing conversation: {e}")
        return {"meeting_title": "Unknown", "summary": [], "discussed_tasks": []}


def log_session(session_timestamp, transcriptions, analysis, log_file="transcription_log.json"):
    """Save the session transcriptions and analysis to a JSON file."""
    record = {
        "timestamp": session_timestamp,
        "meeting_title": analysis.get("meeting_title", "Unknown"),
        "raw_records": transcriptions,
        "summary": analysis.get("summary", []),
        "discussed_tasks": analysis.get("discussed_tasks", []),
    }

    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading log file, starting new log: {e}")
            data = []
    else:
        data = []

    data.append(record)
    try:
        with open(log_file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Session log saved to {log_file}")
    except Exception as e:
        print(f"Error saving session log: {e}")


def process_audio_in_real_time(model, audio_frames, rate):
    """Process a batch of audio frames and store the recognized text."""
    try:
        print("Processing audio...")
        audio_array = np.frombuffer(b"".join(audio_frames), dtype=np.int16).astype(np.float32) / 32768.0
        audio_denoised = nr.reduce_noise(y=audio_array, sr=rate)
        result = model.transcribe(audio_denoised, fp16=False, language="en")
        recognized_text = result.get("text", "").strip()
        print(f"Recognized Text: {recognized_text}")
        with session_lock:
            session_transcriptions.append(recognized_text)
    except Exception as e:
        print(f"Error processing audio: {e}")


def real_time_speech_recognition(chunk_size=1024, fmt=pyaudio.paInt16, channels=1, rate=16000):
    """Continuously read audio from the microphone and transcribe in real time."""
    print("Starting real-time speech recognition. Press Ctrl+C to stop.")
    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(format=fmt, channels=channels, rate=rate, input=True, frames_per_buffer=chunk_size)
    audio_frames = []
    print("Loading Whisper model...")
    model = whisper.load_model("small.en")
    try:
        while True:
            audio_data = stream.read(chunk_size, exception_on_overflow=False)
            audio_frames.append(audio_data)
            if len(audio_frames) >= int(rate / chunk_size * 5):
                thread = threading.Thread(
                    target=process_audio_in_real_time,
                    args=(model, audio_frames.copy(), rate),
                )
                thread.start()
                audio_frames = []
    except KeyboardInterrupt:
        print("Stopping real-time speech recognition.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio_interface.terminate()


if __name__ == "__main__":
    try:
        real_time_speech_recognition()
    except Exception as e:
        print(f"Failed to start speech recognition: {e}")
    finally:
        with session_lock:
            if session_transcriptions:
                analysis = analyze_conversation(session_transcriptions)
                log_session(session_timestamp, session_transcriptions, analysis)
            else:
                print("No transcriptions to log for this session.")
