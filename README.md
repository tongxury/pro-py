# AURA Voice Agent

AI-powered mental health counselor using LiveKit Voice Pipeline.

## Features

- 🎙️ Real-time voice conversation
- 🧠 OpenAI GPT-4o for natural dialogue
- 🔊 Cartesia TTS for high-quality speech synthesis
- 🎯 Silero VAD for voice activity detection
- 📝 Deepgram STT for speech recognition

## Quick Start

```bash
# Install dependencies
pip install -e .

# Set environment variables
export LIVEKIT_URL="wss://your-livekit-url"
export LIVEKIT_API_KEY="your-api-key"
export LIVEKIT_API_SECRET="your-api-secret"
export OPENAI_API_KEY="your-openai-key"
export CARTESIA_API_KEY="your-cartesia-key"
export DEEPGRAM_API_KEY="your-deepgram-key"  # Optional, uses OpenAI STT by default

# Run in development mode
python -m voiceagent dev

# Or use the CLI
voiceagent dev
```

## Run Modes

```bash
# Development mode with hot reload
python -m voiceagent dev

# Production mode
python -m voiceagent start

# Connect to specific room
python -m voiceagent connect --room my_room
```

## Project Structure

```
pro-py/
├── pyproject.toml
├── README.md
├── .env.example
└── src/
    └── voiceagent/
        ├── __init__.py
        ├── main.py          # Entry point
        └── agent.py         # Voice agent implementation
```
