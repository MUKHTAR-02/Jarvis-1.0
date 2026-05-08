"""
============================================================
  🤖 Speech-to-Speech AI Chatbot — Powered by Groq (Free)
============================================================
  Stack:
    STT  → Groq Whisper (whisper-large-v3)
    LLM  → Groq LLaMA 3.3 70B
    TTS  → Microsoft Edge TTS (edge-tts, 100% free)

  How to use:
    1. Set your GROQ_API_KEY in the .env file or below
    2. Run:  python jarvis.py
    3. Press Enter to START recording
    4. Press Enter again to STOP and send
    5. Listen to the AI reply!
    6. Type 'quit' + Enter to exit
============================================================
"""

import os
import asyncio
import tempfile
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from groq import Groq
import edge_tts
import pygame

# ──────────────────────────────────────────────
#  CONFIG — edit these values
# ──────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

STT_MODEL   = "whisper-large-v3"           # Free Groq speech-to-text
CHAT_MODEL  = "llama-3.3-70b-versatile"    # Free Groq LLM
TTS_VOICE   = "en-US-JennyNeural"          # Edge TTS voice (free)
SAMPLE_RATE = 16000                        # 16 kHz — best for Whisper
CHANNELS    = 1                            # Mono microphone
MAX_TOKENS  = 300                          # Keep AI replies concise

SYSTEM_PROMPT = (
    "Your name is JARVIS"
    "You are a helpful and friendly AI assistant. "
    "Keep your responses concise, clear, and conversational. "
    "Avoid using bullet points or markdown — speak naturally."
)

# ──────────────────────────────────────────────
#  SETUP
# ──────────────────────────────────────────────

client = Groq(api_key=GROQ_API_KEY)
conversation_history = []  # Stores the full chat history for context

pygame.mixer.init()        # Initialize audio playback


# ──────────────────────────────────────────────
#  STEP 1: RECORD AUDIO
# ──────────────────────────────────────────────

def record_audio() -> np.ndarray | None:
    """
    Records from the microphone until the user presses Enter.
    Returns a NumPy array of audio samples, or None if too short.
    """
    recorded_chunks = []
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        """Called by sounddevice for every chunk of audio captured."""
        if not stop_event.is_set():
            recorded_chunks.append(indata.copy())

    # Open the microphone stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
    )

    stream.start()
    print("🎤  Recording... Press [Enter] to stop.")
    input()  # Wait for the user to press Enter
    stop_event.set()
    stream.stop()
    stream.close()

    if not recorded_chunks:
        return None

    audio = np.concatenate(recorded_chunks, axis=0)

    # Reject clips shorter than 0.5 seconds
    if len(audio) < SAMPLE_RATE * 0.5:
        return None

    return audio


# ──────────────────────────────────────────────
#  STEP 2: TRANSCRIBE (Speech → Text)
# ──────────────────────────────────────────────

def transcribe(audio: np.ndarray) -> str:
    """
    Saves audio to a temp WAV file and sends it to Groq Whisper.
    Returns the transcribed text string.
    """
    # Save to a temporary .wav file (Whisper needs a file, not raw bytes)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    sf.write(tmp_path, audio, SAMPLE_RATE)

    with open(tmp_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=audio_file,
            response_format="text",
        )

    os.unlink(tmp_path)  # Clean up temp file
    return result.strip() if result else ""


# ──────────────────────────────────────────────
#  STEP 3: GET AI REPLY (Text → Text)
# ──────────────────────────────────────────────

def get_ai_reply(user_text: str) -> str:
    """
    Sends the user's message to Groq LLaMA with full conversation history.
    Returns the AI's reply as a string.
    """
    # Add user message to history
    conversation_history.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *conversation_history,  # Send full history for context
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.7,
    )

    reply = response.choices[0].message.content.strip()

    # Add AI reply to history so it remembers the conversation
    conversation_history.append({"role": "assistant", "content": reply})

    return reply


# ──────────────────────────────────────────────
#  STEP 4: SPEAK (Text → Speech)
# ──────────────────────────────────────────────

async def _tts_generate(text: str, output_path: str):
    """Async helper: generates speech using Edge TTS and saves to file."""
    communicate = edge_tts.Communicate(text, voice=TTS_VOICE)
    await communicate.save(output_path)


def speak(text: str):
    """
    Converts text to speech using Edge TTS (Microsoft, free).
    Plays the audio using pygame.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    # Generate the audio file
    asyncio.run(_tts_generate(text, tmp_path))

    # Play it
    pygame.mixer.music.load(tmp_path)
    pygame.mixer.music.play()

    # Wait until playback finishes
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)

    pygame.mixer.music.unload()  # Release the file first
    os.unlink(tmp_path)          # Now safe to delete


# ──────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────

def main():
    print("\n" + "=" * 54)
    print("   🤖  AI Speech Chatbot  |  Powered by Groq (Free)")
    print("=" * 54)
    print("  → Press [Enter] to start speaking")
    print("  → Press [Enter] again to stop recording")
    print("  → Type  'quit' + [Enter] to exit")
    print("  → Type  'clear' + [Enter] to reset conversation")
    print("=" * 54 + "\n")

    speak("Hello! I am JARVIS How can I assist you?")

    while True:
        # ── Prompt the user ──────────────────────────────
        try:
            cmd = input("Press [Enter] to speak › ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if cmd == "quit":
            print("👋 Goodbye!")
            break

        if cmd == "clear":
            conversation_history.clear()
            print("🗑️  Conversation history cleared.\n")
            continue

        # ── Record ───────────────────────────────────────
        audio = record_audio()
        if audio is None:
            print("⚠️  Recording too short. Try again.\n")
            continue

        # ── Transcribe ───────────────────────────────────
        print("🔍 Transcribing...")
        user_text = transcribe(audio)
        if not user_text:
            print("⚠️  Couldn't understand audio. Try again.\n")
            continue
        print(f"   You › {user_text}")

        # ── Get AI reply ─────────────────────────────────
        print("🤔 Thinking...")
        reply = get_ai_reply(user_text)
        print(f"   AI  › {reply}\n")

        # ── Speak ────────────────────────────────────────
        print("🔊 Speaking...")
        speak(reply)
        print()


if __name__ == "__main__":
    main()