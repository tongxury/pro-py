import os
import aiohttp
from livekit.agents.log import logger

class TranscriptRecorder:
    def __init__(self, user_id: str, conversation_id: str, room_name: str):
        self._user_id = user_id
        self._conversation_id = conversation_id
        self._room_name = room_name
        self._api_base_url = os.getenv("API_BASE_URL", "https://api.larksings.com")
        self._api_url = f"{self._api_base_url}/api/va/transcripts"

    async def record(self, role: str, content: str):
        """
        Send transcript entry to the backend.
        """
        if not self._user_id or not self._conversation_id:  
            return

        payload = {
            "userId": self._user_id,
            "conversationId": self._conversation_id,
            "role": role,
            "content": content,
            "roomName": self._room_name
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self._api_url, json=payload) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        logger.warning(f"Failed to record transcript: {resp.status} - {text}")
                    else:
                        logger.debug(f"Transcript saved: [{role}] {content[:20]}...")
        except Exception as e:
            logger.error(f"Error recording transcript: {e}")
