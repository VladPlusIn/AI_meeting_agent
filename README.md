# AI Meeting Agent

This project provides a simple real‑time speech recorder that transcribes audio using [Whisper](https://github.com/openai/whisper) and summarizes the conversation with GPT‑4. The script saves each session to `transcription_log.json`.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the program:

```bash
python speech_recorder.py
```

The first run asks for your OpenAI API key and stores it in `config.json` (ignored by Git). After recording, a summary is generated and appended to `transcription_log.json`.

## Files

- `speech_recorder.py` – command‑line script extracted from the original notebook.
- `Speach_record_AI_agenr.ipynb` – original notebook version.
- `requirements.txt` – Python dependencies.

## Notes

Audio capture requires a working microphone and the `pyaudio` library, which may need additional system packages depending on your platform.
