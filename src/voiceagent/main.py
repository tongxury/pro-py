"""
AURA Voice Agent - Main Entry Point (livekit-agents 1.3.12 API)
"""

import asyncio
import json
import os
import aiohttp
from typing import Optional

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, JobProcess, WorkerOptions, cli, voice
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.log import logger
from livekit.plugins import cartesia, openai, silero, deepgram
from livekit.plugins import cartesia, openai, silero, deepgram

from voiceagent.agents import AGENTS, AgentConfig
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
        # Ideally AuraAgent would inherit from voice.Agent, but composition is fine with this hook.
        self.agent.on_user_turn_completed = self._on_user_turn_completed

    def _init_stt(self):
        dg_key = os.getenv("DEEPGRAM_API_KEY")
        if dg_key and dg_key != "your-deepgram-key" and not dg_key.startswith("your-"):
            logger.info("Using Deepgram STT")
            return deepgram.STT(model="nova-2")
        logger.info("Using OpenAI STT (Deepgram key missing or placeholder)")
        return openai.STT()

    def _init_tts(self):
        # Fallback to OpenAI TTS if Cartesia is failing or key is problematic
        # For debugging, we'll try OpenAI first if CARTESIA_API_KEY is missing or if we want to test
        use_openai = os.getenv("USE_OPENAI_TTS", "true").lower() == "true"
        
        if use_openai:
            logger.info("Using OpenAI TTS")
            return openai.TTS(model="tts-1", voice="alloy")

        logger.info(f"Using Cartesia TTS with voice: {self.config.voice}, model: {self.config.tts_model}")
        kwargs = {}
        if self.config.speed:
            kwargs["speed"] = self.config.speed
        if self.config.emotion:
            kwargs["emotion"] = self.config.emotion
            
        return cartesia.TTS(
            model=self.config.tts_model,
            voice=self.config.voice,
            language=self.config.language,
            **kwargs
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
        # Check connection state before connecting
        if self.ctx.room.connection_state == rtc.ConnectionState.CONN_DISCONNECTED:
             logger.info(f"Connecting to room: {self.ctx.room.name}...")
             await self.ctx.connect()
             logger.info(f"Connected to room: {self.ctx.room.name}")
        
        # Share available personas via attributes
        persona_list = [{"id": k, "name": v.name} for k, v in AGENTS.items()]
        await self.ctx.room.local_participant.set_attributes({
            "personas": json.dumps(persona_list)
        })
        
        # Create and start the session
        session = voice.AgentSession()
        
        @session.on("agent_state_changed")
        def on_agent_state(event: voice.AgentStateChangedEvent):
            logger.info(f"Agent state changed: {event.old_state} -> {event.new_state}")
            
            async def _check_and_log_agent():
                # Wait briefly for context to sync
                await asyncio.sleep(0.5)
                # CRITICAL FIX: Ensure we read from the agent's internal context which gets updated
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

        @session.on("user_state_changed")
        def on_user_state(event: voice.UserStateChangedEvent):
            logger.debug(f"User state changed: {event.old_state} -> {event.new_state}")
            # User messages are handled by _before_llm_cb, so we don't need logic here
            pass

        @session.on("error")
        def on_error(event: voice.ErrorEvent):
            logger.error(f"Agent session error: {event.error} (from {event.source})")

        # Start the agent session on the room
        logger.info("Starting Voice Agent Session...")
        await session.start(self.agent, room=self.ctx.room)
        logger.info("AURA session active")
        
        # Initial greeting
        if self.config.greeting:
            await asyncio.sleep(1)
            logger.info(f"Sending greeting: {self.config.greeting}")
            await session.say(self.config.greeting)


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarm complete: VAD loaded")


async def entrypoint(ctx: JobContext):
    logger.info(f"Entrypoint triggered for room: {ctx.room.name}")
    
    # --- 0. Connect FIRST to ensure Metadata is synced ---
    logger.info("Connecting to room to fetch metadata...")
    await ctx.connect()
    
    # --- 1. Read configuration from Room Metadata ---
    # Since Go puts config in Room Metadata (Auto-Join strategy), we read it here.
    dispatch_info = ctx.room.metadata
    logger.info(f"Checking Room Metadata: {dispatch_info}")
    
    print("Checking Room Metadata: ", dispatch_info)

    target_agent_id = "aura_zh"
    user_nickname = "User"
    memory_context = ""
    user_id = None

    if dispatch_info:
        try:
            data = json.loads(dispatch_info)
            user_id = data.get("userId")
            conversation_id = data.get("conversationId")
            
            # 1. Config override
            if "agentName" in data and data["agentName"] in AGENTS:
                target_agent_id = data["agentName"]
                logger.info(f"Switching to requested agent: {target_agent_id}")
            
            # 2. Profile info
            user_profile = data.get("userProfile", {})
            user_nickname = user_profile.get("nickname", "User")
            
            # 3. Memories
            memories = data.get("memories", [])
            if memories:
                memory_list_text = "\n".join([f"- {m}" for m in memories])
                memory_context = f"""
\n---
User Context:
Name: {user_nickname}

Things you remember about this user (IMPORTANT):
{memory_list_text}
---
"""
        except Exception as e:
            logger.error(f"Failed to parse metadata: {e}")

    # --- 2. Initialize Agent ---
    config = AGENTS.get(target_agent_id)
    if not config:
        config = AGENTS.get("aura_zh")

    # Inject memory into system prompt
    # We create a shallow copy of config to avoid modifying the global singleton
    from dataclasses import replace
    if memory_context:
        new_prompt = config.system_prompt + memory_context
        config = replace(config, system_prompt=new_prompt)
        logger.info("Memory context injected into System Prompt.")

    agent = AuraAgent(ctx, config, user_id=user_id, conversation_id=conversation_id)

    try:
        await agent.start()
    except Exception as e:
        logger.exception(f"Failed to start agent: {e}")


def main():
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint, 
        prewarm_fnc=prewarm,
        # agent_name="aura_zh"
    ))


if __name__ == "__main__":
    main()
