"""
AURA Voice Agent - Main Entry Point (livekit-agents 1.3.12 API)
"""

import asyncio
import json
import os
from typing import Optional

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, JobProcess, WorkerOptions, cli, voice
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.log import logger
from livekit.plugins import cartesia, openai, silero, deepgram

from voiceagent.agents import AGENTS, AgentConfig

load_dotenv()

class AuraAgent:
    def __init__(self, ctx: JobContext, config: AgentConfig):
        self.ctx = ctx
        self.config = config
        
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

    async def start(self):
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

        @session.on("user_state_changed")
        def on_user_state(event: voice.UserStateChangedEvent):
            logger.debug(f"User state changed: {event.old_state} -> {event.new_state}")
            
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
    
    # --- 1. Read configuration from Room Metadata ---
    # Since Go puts config in Room Metadata (Auto-Join strategy), we read it here.
    dispatch_info = ctx.room.metadata
    logger.info(f"Checking Room Metadata: {dispatch_info}")

    target_agent_id = "aura_zh"
    user_nickname = "User"
    memory_context = ""

    if dispatch_info:
        try:
            data = json.loads(dispatch_info)
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

    agent = AuraAgent(ctx, config)
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
