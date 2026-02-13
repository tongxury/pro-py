"""
AURA Voice Agent - Main Entry Point (livekit-agents 1.3.12 API)
"""

import asyncio
import json
import os
import time
from typing import Optional, Any

from dotenv import load_dotenv
from livekit.agents import JobContext, JobProcess, WorkerOptions, cli, voice
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.log import logger
from livekit.plugins import cartesia, openai, silero, deepgram

from dataclasses import dataclass, field

@dataclass
class AgentConfig:
    name: str
    system_prompt: str
    greeting: str
    voice: str
    language: str
    llm_model: str
    tts_model: str

from voiceagent.services.recorder import TranscriptRecorder

load_dotenv()

class AuraAgent:
    def __init__(self, ctx: JobContext, config: AgentConfig, user_id: str = None, conversation_id: str = None):
        self.ctx = ctx
        self.config = config
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Initialize Recorder Plugin
        if user_id and conversation_id:
            self.recorder = TranscriptRecorder(user_id, conversation_id, ctx.room.name)
        else:
            self.recorder = None
        
        # Initialize providers
        logger.info("Initializing AuraAgent providers...")
        self.vad = ctx.proc.userdata["vad"]
        self.stt = self._init_stt()
        self.llm = openai.LLM(model=config.llm_model)
        self.tts = self._init_tts()
        
        self.chat_ctx = ChatContext()
        self.chat_ctx.add_message(role="system", content=config.system_prompt)
        
        # In 1.3.x, voice.Agent is the high-level orchestration class
        self.agent = voice.Agent(
            instructions=config.system_prompt,
            vad=self.vad,
            stt=self.stt,
            llm=self.llm,
            tts=self.tts,
            chat_ctx=self.chat_ctx,
        )
        
        # Hook into the agent's definition of user turn completion to log messages
        self.agent.on_user_turn_completed = self._on_user_turn_completed
        
        self._last_agent_content = ""

    def _init_stt(self):
        dg_key = os.getenv("DEEPGRAM_API_KEY")
        if dg_key and dg_key != "your-deepgram-key" and not dg_key.startswith("your-"):
            logger.info("Using Deepgram STT")
            return deepgram.STT(model="nova-2")
        logger.info("Using OpenAI STT (Deepgram key missing or placeholder)")
        return openai.STT()

    def _init_tts(self):
        # Fallback to OpenAI TTS if Cartesia is failing or key is problematic
        use_openai = os.getenv("USE_OPENAI_TTS", "true").lower() == "true"
        
        if use_openai:
            logger.info("Using OpenAI TTS")
            return openai.TTS(model="tts-1", voice="alloy")

        logger.info(f"Using Cartesia TTS with voice: {self.config.voice}, model: {self.config.tts_model}")
            
        return cartesia.TTS(
            model=self.config.tts_model,
            voice=self.config.voice,
            language=self.config.language,
        )

    def _get_text_content(self, item):
        # Try attribute first if it exists
        if hasattr(item, "text_content"):
            return item.text_content
        # Fallback to content field
        c = getattr(item, "content", "")
        if isinstance(c, list):
            # Join string parts
            return "\n".join([str(x) for x in c if isinstance(x, str)])
        return str(c)

    async def _on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage):
        """Called when the user has finished speaking, and the LLM is about to respond"""
        if new_message.role == "user":
            content = self._get_text_content(new_message)
            logger.info(f"📝 [History] User (via on_user_turn_completed): {content}")
            # Use Recorder Plugin
            if self.recorder:
                asyncio.create_task(self.recorder.record("user", content))

    async def start(self):
        # Create and start the session
        session = voice.AgentSession()
        
        @session.on("agent_state_changed")
        def on_agent_state(event: voice.AgentStateChangedEvent):
            logger.info(f"Agent state changed: {event.old_state} -> {event.new_state}")
            
            async def _check_and_log_agent():
                # Wait briefly for context to sync
                await asyncio.sleep(0.5)
                # CRITICAL: Ensure we read from the agent's internal context which gets updated
                items = self.agent.chat_ctx.items
                if not items:
                    return
                # Find the last assistant message
                target_item = None
                for item in reversed(items):
                    if item.type == "message" and item.role == "assistant":
                        target_item = item
                        break
                
                if target_item:
                    content = self._get_text_content(target_item) or ""
                    # Avoid duplicates
                    if hasattr(self, "_last_agent_content") and self._last_agent_content == content:
                        return
                    
                    self._last_agent_content = content
                    logger.info(f"📝 [History] Agent: {content}")
                    if self.recorder:
                        await self.recorder.record("agent", content)

            # Log when agent finishes speaking (speaking -> listening/thinking)
            if event.old_state == "speaking":
                asyncio.create_task(_check_and_log_agent())

        @session.on("error")
        def on_error(event: voice.ErrorEvent):
            logger.error(f"Session Error: {event.error}")

        # Start the session
        logger.info("Starting Voice Agent Session...")
        await session.start(self.agent, room=self.ctx.room)
        
        # Initial greeting
        if self.config.greeting:
            await asyncio.sleep(1)
            await session.say(self.config.greeting)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarm complete: VAD loaded")


async def entrypoint(ctx: JobContext):
    logger.info(f"Entrypoint triggered (Explicit Dispatch) for room: {ctx.room.name}")

    # --- 0. Robust Connection with Timeout ---
    max_connect_retries = 3
    connected = False
    for i in range(max_connect_retries):
        try:
            logger.info(f"Connecting to room (attempt {i+1}/{max_connect_retries})...")
            await asyncio.wait_for(ctx.connect(), timeout=10.0)
            logger.info(f"Successfully connected to room: {ctx.room.name}")
            connected = True
            break
        except Exception as e:
            logger.error(f"Connection attempt {i+1} failed: {e}")
        if i < max_connect_retries - 1:
            await asyncio.sleep(1.5)

    if not connected:
        return

    # --- 1. Priority Data Strategy: Job Metadata vs Room Metadata ---
    dispatch_info = ctx.job.metadata or ctx.room.metadata
    
    if not dispatch_info or len(dispatch_info) < 10:
        logger.info("Metadata not in job, waiting for room metadata sync...")
        for i in range(5):
            dispatch_info = ctx.room.metadata
            if dispatch_info and len(dispatch_info) > 10:
                break
            await asyncio.sleep(0.5)

    # --- 2. Process Configuration from Metadata (Go Backend) ---
    data = {}
    if dispatch_info:
        try:
            data = json.loads(dispatch_info)
        except Exception as e:
            logger.error(f"Failed to parse metadata: {e}")

    # Extract with no defaults (or minimal structural defaults)
    user_id = data.get("userId")
    conversation_id = data.get("conversationId")
    agent_name = data.get("agentName", "aura_zh")
    
    system_prompt = data.get("systemPrompt", "")
    greeting = data.get("greeting", "")
    # voice_id = data.get("voiceId", "")
    voice_id = "a53c3509-ec3f-425c-a223-977f5f7424dd"

    
    logger.info(f"Received Configuration - SystemPrompt Length: {dispatch_info}")
    print(dispatch_info)

    # --- 3. Initialize Agent Config ---
    config = AgentConfig(
        name=agent_name,
        system_prompt=system_prompt,
        greeting=greeting,
        voice=voice_id,
        language="zh" if "zh" in agent_name else "en",
        llm_model="gpt-4o-mini",
        tts_model="sonic-multilingual",
    )

    # --- 4. Start the Agent ---
    agent = AuraAgent(ctx, config, user_id=user_id, conversation_id=conversation_id)
    
    try:
        logger.info(f"Starting AURA ({agent_name}) for session {conversation_id or 'unknown'}...")
        await agent.start()
    except Exception as e:
        logger.exception(f"Critical error during agent startup: {e}")


def main():
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint, 
        prewarm_fnc=prewarm,
        agent_name="aura_zh", # Enable Explicit Dispatch
    ))


if __name__ == "__main__":
    main()
