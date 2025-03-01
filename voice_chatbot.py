import sounddevice as sd
import numpy as np
import requests
import json
import subprocess
import wave
import os
from vosk import Model, KaldiRecognizer

# ------------------------------------------------------------------------
# 1) Vosk Setup for Speech Recognition
# ------------------------------------------------------------------------
print("Loading Vosk model...")
vosk_model_path = "model/vosk-model-en-us-0.22"  # Adjust path if needed
model_vosk = Model(vosk_model_path)
SAMPLERATE = 16000  # For Vosk

# ------------------------------------------------------------------------
# 2) Ollama API
# ------------------------------------------------------------------------
OLLAMA_API = "http://localhost:11434/api/generate"

# ------------------------------------------------------------------------
# 3) Piper Command-Line TTS Setup
# ------------------------------------------------------------------------
piper_exe = "piper"  # Ensure Piper is in your PATH or provide full path.
piper_model = "model/en_GB-jenny_dioco-medium.onnx"  # Path to your Piper ONNX model.

def record_audio(duration=5, samplerate=SAMPLERATE):
    print("🎤 Speak now...")
    audio = sd.rec(int(duration * samplerate),
                   samplerate=samplerate,
                   channels=1,
                   dtype=np.int16)
    sd.wait()
    return audio.flatten()

def transcribe(audio):
    recognizer = KaldiRecognizer(model_vosk, SAMPLERATE)
    recognizer.AcceptWaveform(audio.tobytes())
    result_json = recognizer.Result()
    result = json.loads(result_json)
    return result.get("text", "").strip()

def chat_ollama(prompt):
    data = {"model": "llama3.2", "prompt": prompt}
    try:
        response = requests.post(OLLAMA_API, json=data)
        raw_text = response.text
        responses = []
        for line in raw_text.splitlines():
            try:
                js = json.loads(line)
                responses.append(js.get("response", ""))
                if js.get("done", False):
                    break
            except Exception as e:
                print("JSON parse error:", e)
        full_response = " ".join(responses).strip()
        return full_response if full_response else "No response."
    except Exception as e:
        print("Error connecting to Ollama:", e)
        return "Error connecting to Ollama."

def speak(text):
    print(f"🤖 Bot: {text}")
    wav_out = "tts_output.wav"
    cmd = [
        piper_exe,
        "--model", piper_model,
        "-f", wav_out,
        "--cuda"
    ]
    try:
        subprocess.run(cmd, input=text.encode("utf-8"), check=True)
    except FileNotFoundError:
        print(f"Error: Piper not found at '{piper_exe}'. Please install Piper or adjust the path.")
        return
    except subprocess.CalledProcessError as e:
        print(f"Error running Piper: {e}")
        return

    if os.path.exists(wav_out):
        with wave.open(wav_out, 'rb') as f:
            nchannels = f.getnchannels()
            sampwidth = f.getsampwidth()
            samplerate = f.getframerate()
            frames = f.readframes(f.getnframes())

            if sampwidth == 2:
                dtype = np.int16
            elif sampwidth == 1:
                dtype = np.uint8
            else:
                print(f"Unsupported sample width: {sampwidth}")
                os.remove(wav_out)
                return

            audio_array = np.frombuffer(frames, dtype=dtype)

            if nchannels > 1:
                numsamples = len(audio_array) // nchannels
                audio_array = audio_array.reshape(numsamples, nchannels)

            sd.play(audio_array, samplerate)
            sd.wait()

        os.remove(wav_out)

if __name__ == "__main__":
    print("🎙️ Voice Chatbot is running... Say something!")
    while True:
        audio_data = record_audio()
        text_in = transcribe(audio_data)
        if text_in:
            print(f"You: {text_in}")
            reply = chat_ollama(text_in)
            speak(reply)
