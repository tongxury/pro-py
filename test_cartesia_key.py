import os
import asyncio
import aiohttp
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CARTESIA_API_KEY")
VOICE_ID = "a53c3509-ec3f-425c-a223-977f5f7424dd" # Mei
MODEL_ID = "sonic-multilingual"

async def test_tts():
    print(f"Testing Cartesia API (Raw HTTP)...")
    print(f"Key: {API_KEY[:6]}...{API_KEY[-4:]}")
    print(f"Voice: {VOICE_ID}")
    print(f"Model: {MODEL_ID}")

    url = "https://api.cartesia.ai/tts/bytes"
    
    headers = {
        "X-API-Key": API_KEY,
        "Cartesia-Version": "2024-06-10",
        "Content-Type": "application/json"
    }

    payload = {
        "model_id": MODEL_ID,
        "voice": {
            "mode": "id",
            "id": VOICE_ID
        },
        "output_format": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": 44100
        },
        "language": "zh",
        "transcript": "你好，这是一个中文语音测试。"
    }

    try:
        async with aiohttp.ClientSession() as session:
            print("\nSending POST request...")
            async with session.post(url, headers=headers, json=payload) as resp:
                print(f"Status Code: {resp.status}")
                if resp.status == 200:
                    data = await resp.read()
                    print(f"Success! Received {len(data)} bytes.")
                    with open("test_output.wav", "wb") as f:
                        f.write(data)
                    print("Saved to test_output.wav")
                else:
                    print("Error Response:")
                    print(await resp.text())

    except Exception as e:
        print(f"\n\nERROR FAILED:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_tts())
