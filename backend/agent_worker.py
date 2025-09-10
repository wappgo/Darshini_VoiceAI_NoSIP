# backend/agent_worker.py
import asyncio
import os
from dotenv import load_dotenv
from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    Agent,
    AgentSession,
    AutoSubscribe,
)
from livekit.plugins import elevenlabs, openai as openai_plugin, silero
import logging
import sys # Added for sys.argv modification

# Load environment variables at the very beginning of this script
load_dotenv()

# Configure logging for this agent worker
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')
logger = logging.getLogger("agent_worker") # Specific logger name

# Fetch necessary environment variables for the agent AND THE CLI
LIVEKIT_URL = os.getenv("LIVEKIT_URL") # <<< ADDED THIS
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY") # <<< ADDED THIS
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET") # <<< ADDED THIS
OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY_ENV = os.getenv("ELEVENLABS_API_KEY")


async def agent_entrypoint(ctx: JobContext):
    # ... (rest of your agent_entrypoint function remains the same) ...
    logger.info(f"[Agent] Entrypoint called for job {ctx.job.id} in room {ctx.room.name}")
    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info("[Agent] Connected to LiveKit room.")
        logger.info("[Agent] Waiting for 'frontend_user' to join...")
        await ctx.wait_for_participant(identity_prefix="frontend_user", timeout=60) 
        logger.info("[Agent] 'frontend_user' has joined the room.")
        agent = Agent(
            instructions="You are Echo, a friendly and concise voice assistant. Respond naturally."
        )
        vad_plugin = silero.VAD.load()
        stt_plugin = openai_plugin.STT(api_key=OPENAI_API_KEY_ENV, model="whisper-1")
        llm_plugin = openai_plugin.LLM(api_key=OPENAI_API_KEY_ENV, model="gpt-4o")
        tts_plugin = elevenlabs.TTS(
            api_key=ELEVENLABS_API_KEY_ENV,
            model_id="eleven_multilingual_v2", 
            voice_id="p9aflnsbBe1o0aDeQa97" 
        )
        logger.info("[Agent] VAD, STT, LLM, TTS plugins initialized.")
        session = AgentSession(
            vad=vad_plugin,
            stt=stt_plugin,
            llm=llm_plugin,
            tts=tts_plugin,
        )
        logger.info("[Agent] AgentSession created.")
        @session.on("transcript_updated")
        async def on_user_transcript(transcript: str, final: bool):
            logger.info(f"[Agent] Transcript: '{transcript}' (Final: {final})")
            if final and transcript: 
                await ctx.room.local_participant.publish_data(
                    payload=f"transcript:{transcript}".encode("utf-8"), topic="transcription"
                )
        @session.on("message_updated")
        async def on_agent_response_text(text: str): 
            logger.info(f"[Agent] LLM Response Text: '{text}'")
            if text: 
                await ctx.room.local_participant.publish_data(
                    payload=f"response:{text}".encode("utf-8"), topic="response"
                )
        await session.start(agent=agent, room=ctx.room)
        logger.info("[Agent] AgentSession started processing.")
        initial_greeting = "Hello! I'm Echo. How can I assist you today?"
        logger.info(f"[Agent] Attempting to say initial greeting: '{initial_greeting}'")
        await session.say(text=initial_greeting, allow_interruptions=True)
        logger.info("[Agent] Initial greeting processing initiated by session.say().")
        logger.info("[Agent] Entrypoint is now active and listening for events.")
        await asyncio.Future() 
    except asyncio.TimeoutError:
        logger.warning("[Agent] Timed out waiting for 'frontend_user'. Agent will disconnect if not already.")
    except Exception as e:
        logger.error(f"[Agent] CRITICAL ERROR in entrypoint: {e}", exc_info=True)
        raise
    finally:
        logger.info("[Agent] Entrypoint finished or terminated.")


if __name__ == "__main__":
    logger.info("--- Starting LiveKit Agent Worker (agent_worker.py) ---")

    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        logger.critical("CRITICAL: Core LiveKit credentials (URL, API_KEY, API_SECRET) missing in .env or not loaded. Agent worker cannot start. Exiting.")
        exit(1)
    if not OPENAI_API_KEY_ENV:
        logger.warning("WARNING: OPENAI_API_KEY missing. STT/LLM functionalities may fail.")
    if not ELEVENLABS_API_KEY_ENV:
        logger.warning("WARNING: ELEVENLABS_API_KEY missing. TTS functionality may fail.")

    worker_options = WorkerOptions(
        entrypoint_fnc=agent_entrypoint,
    )

    original_argv = list(sys.argv)
    # Ensure 'start' is the command if running programmatically like this.
    # If sys.argv already contains a command like 'dev' from user input, this will override it for this programmatic call.
    # If you want to allow user to pass 'dev' to 'python agent_worker.py dev', then you might need more complex argv handling here.
    # For now, we force 'start' to ensure it runs without issues related to 'dev' mode's signal handling in a non-main thread context if this script were ever threaded.
    # Since this script IS the main thread for the agent, 'dev' *should* work if passed on command line.
    # But to ensure programmatic default to 'start':
    if len(sys.argv) == 1: # No command provided, default to 'start'
        sys.argv.append('start')
    # If a command is already provided (e.g., python agent_worker.py dev), sys.argv will be like ['agent_worker.py', 'dev']
    # and cli.run_app will use that.

    logger.info(f"Calling cli.run_app (sys.argv before call: {sys.argv})")
    try:
        # cli.run_app() does not take standalone_mode.
        # It will use sys.argv to determine the command (e.g., 'start', 'dev').
        cli.run_app(worker_options)
    except SystemExit as se:
        logger.info(f"cli.run_app exited with SystemExit (code: {se.code}). This is often normal for CLI apps.")
    except Exception as e:
        logger.error(f"An unexpected error occurred running cli.run_app: {e}", exc_info=True)
    finally:
        # No need to restore sys.argv if cli.run_app handles it or if the process exits.
        # If we did modify it forcefully, then restoring would be: sys.argv = original_argv
        logger.info("--- LiveKit Agent Worker (agent_worker.py) finished or was interrupted. ---")