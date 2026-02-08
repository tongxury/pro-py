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
    # In Explicit Dispatch mode, Go sends metadata DIRECTLY with the job.
    # This is much faster and more reliable than waiting for room metadata.
    dispatch_info = ctx.job.metadata or ctx.room.metadata
    
    if not dispatch_info or len(dispatch_info) < 10:
        logger.info("Metadata not in job, waiting for room metadata sync...")
        for i in range(5):
            dispatch_info = ctx.room.metadata
            if dispatch_info and len(dispatch_info) > 10:
                break
            await asyncio.sleep(0.5)

    # --- 2. Process Configuration ---
    target_agent_id = "aura_zh"
    user_nickname = "User"
    memory_context = ""
    user_id = None
    conversation_id = None
    topic = None

    if dispatch_info:
        try:
            data = json.loads(dispatch_info)
            user_id = data.get("userId")
            conversation_id = data.get("conversationId")
            target_agent_id = data.get("agentName", target_agent_id)
            topic = data.get("topic")
            topic_greeting = data.get("topicGreeting")
            topic_instruction = data.get("topicInstruction")
            
            user_nickname = data.get("nickname", "User")
            
            memories = data.get("memories", [])
            if memories:
                memory_list_text = "\n".join([f"- {m}" for m in memories])
                memory_context = f"\n---\n[User Context & Memories]\nUser Name: {user_nickname}\nRelevant Memories:\n{memory_list_text}\n\nIMPORTANT: You MUST use the user's name ({user_nickname}) and their past memories to make the conversation feel warm, personal, and continuous. Do not be generic.\n---\n"
        except Exception as e:
            logger.error(f"Failed to parse metadata: {e}")

    # --- 3. Resolve and Prepare Agent ---
    config = AGENTS.get(target_agent_id, AGENTS.get("aura_zh"))

    from dataclasses import replace
    if memory_context:
        config = replace(config, system_prompt=config.system_prompt + memory_context)

    # --- 3.1 Topic Customization ---
    if topic:
        # 1. Use greeting from metadata if available, and personalize it
        if topic_greeting:
             final_greeting = topic_greeting
             # If nickname is available and valid, personalize the greeting
             if user_nickname and user_nickname != "User":
                 final_greeting = f"Hi {user_nickname}, {topic_greeting}"
             
             config = replace(config, greeting=final_greeting)
        
        # 2. Use instruction from metadata if available
        if topic_instruction:
             topic_instruction_text = f"\n\n[Current Topic Context]\nThe user has selected to talk about topic: '{topic}'.\nSpecial Instruction: {topic_instruction}\n\nTone Requirement: Be warm, empathetic, and intimate (make the user feel '亲切'). Start the conversation by acknowledging their situation or the topic gently."
             config = replace(config, system_prompt=config.system_prompt + topic_instruction_text)
        
        logger.info(f"Applying topic customization for: {topic}. Greeting: {config.greeting}")

    # --- 4. Start the Agent ---
    agent = AuraAgent(ctx, config, user_id=user_id, conversation_id=conversation_id)
    
    try:
        logger.info(f"Starting AURA ({target_agent_id}) for session {conversation_id or 'unknown'}...")
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
