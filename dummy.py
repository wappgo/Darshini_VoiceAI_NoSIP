


import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from datetime import timedelta
import jwt 
import threading
import time 
import sys 
from typing import Any  # Added for type hint

# LiveKit specific imports
from livekit.api import AccessToken, VideoGrants
from livekit import agents, rtc
from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    Agent, 
    AgentSession,
    AutoSubscribe,
)
from livekit.agents.llm import LLMStream

# LiveKit Plugins
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel  # Corrected import

# Load environment variables at the very beginning
load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:3000", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')
logger = logging.getLogger(__name__)
agent_logger = logging.getLogger("livekit.agents")
agent_logger.setLevel(logging.INFO)

# Fetch environment variables
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY_ENV = os.getenv("ELEVENLABS_API_KEY")
DEEPGRAM_API_KEY_ENV = os.getenv("DEEPGRAM_API_KEY")

logger.info(f"MODULE LEVEL: LIVEKIT_URL: {LIVEKIT_URL}")
logger.info(f"MODULE LEVEL: LIVEKIT_API_KEY IS SET: {bool(LIVEKIT_API_KEY)}")
logger.info(f"MODULE LEVEL: DEEPGRAM_API_KEY IS SET: {bool(DEEPGRAM_API_KEY_ENV)}")
logger.info(f"MODULE LEVEL: OPENAI_API_KEY IS SET: {bool(OPENAI_API_KEY_ENV)}")

@app.get("/token")
async def get_token():
    logger.info("[TokenSvc] /token endpoint hit.")
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        logger.error("[TokenSvc] LiveKit credentials not configured properly.")
        raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

    video_grant_obj = VideoGrants(
        room_join=True,
        room="voice-assistant-room", 
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )

    token_builder = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token_builder.identity = "frontend_user" 
    token_builder.name = "Frontend User"
    token_builder.ttl = timedelta(hours=1)
    token_builder.with_grants(video_grant_obj)
    token_jwt = token_builder.to_jwt()

    logger.info(f"[TokenSvc] Generated token for frontend_user to join 'voice-assistant-room'")
    return {"token": token_jwt, "url": LIVEKIT_URL}


# --- Define the Assistant Agent ---
class GUIFocusedAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly voice assistant created by LiveKit, integrated into a web GUI. "
                "Your purpose is to engage in general conversation, answer questions, "
                "and provide helpful responses. Keep your answers concise, natural, "
                "and suitable for voice interaction. You are talking to a user through a web interface."
            )
        )

    async def publish_llm_stream_to_user(self, llm_stream: LLMStream) -> None:
        collected_text_for_frontend = ""
        sentences_for_tts = []

        async for sentence in self.llm_stream_sentences(llm_stream):
            collected_text_for_frontend += sentence
            sentences_for_tts.append(sentence)

        if collected_text_for_frontend and self.room:
            logger.info(f"[AssistantGUI] Publishing LLM response to frontend: '{collected_text_for_frontend}'")
            try:
                await self.room.local_participant.publish_data(
                    f"response:{collected_text_for_frontend}", "response"
                )
                logger.info(f"[AssistantGUI] Successfully published LLM response to frontend: '{collected_text_for_frontend}'")
            except Exception as e_publish:
                logger.error(f"[AssistantGUI] Error publishing LLM response data: {e_publish}", exc_info=True)

        async def tts_sentence_generator():
            for s in sentences_for_tts:
                yield s

        if sentences_for_tts:
            logger.info(f"[AssistantGUI] Synthesizing audio for: '{collected_text_for_frontend}'")
            try:
                await self.tts.synthesize(tts_sentence_generator())
                logger.info(f"[AssistantGUI] Audio synthesis complete for: '{collected_text_for_frontend}'")
            except Exception as e_tts:
                logger.error(f"[AssistantGUI] Error during TTS synthesis: {e_tts}", exc_info=True)
        else:
            logger.info("[AssistantGUI] No sentences from LLM to synthesize.")

# --- Agent Entrypoint ---
async def entrypoint(ctx: JobContext):
    room_name_from_ctx = ctx.room.name if ctx.room else "N/A_ROOM_IN_CTX"
    job_id_from_ctx = ctx.job.id if ctx.job else "N/A_JOB_IN_CTX"
    logger.info(f"--- [AgentEntry] AGENT LOGIC ENTRYPOINT CALLED --- Room: {room_name_from_ctx}, Job ID: {job_id_from_ctx}")
    logger.info(f"[AgentEntry] Job received for room: {room_name_from_ctx}, job_id: {job_id_from_ctx}")

    try:
        logger.info(f"[AgentEntry] Ensuring connection and setting subscriptions for job {job_id_from_ctx}.")
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY) 
        logger.info(f"[AgentEntry] Connected/Verified. Participant SID: {ctx.room.local_participant.sid}")
    except Exception as e_connect:
        logger.error(f"[AgentEntry] Error during ctx.connect(): {e_connect}", exc_info=True)
        raise 

    try:
        logger.info("[AgentEntry] Initializing plugins...")
        stt = deepgram.STT(
            model="nova-2",
            language="multi",
        )
        llm = openai.LLM(
            model="gpt-4o-mini",
        )
        tts = openai.TTS( 
            voice="ash",
        )
        vad = silero.VAD.load() 
        turn_detection = MultilingualModel()
        logger.info("[AgentEntry] Plugins initialized.")

        # Function to publish STT results to the frontend
        async def publish_stt_data(stt_event: Any, topic_prefix="transcription"):
            if stt_event.alternatives and stt_event.alternatives[0].text:
                text = stt_event.alternatives[0].text
                logger.info(f"[AgentEntry:{topic_prefix}] Publishing STT: '{text}'")
                
                await ctx.room.local_participant.publish_data(f"transcription:{text}", "transcription")
                logger.info(f"[AgentEntry:{topic_prefix}] Published STT to frontend: '{text}'")

                try:
                    await ctx.room.local_participant.publish_data(f"transcription:{text}", "transcription") 
                except Exception as e_publish_stt:
                    logger.error(f"[AgentEntry] Error publishing STT data: {e_publish_stt}", exc_info=True)

        # Define synchronous handlers that will schedule the async task
        def stt_interim_handler(stt_event: Any):
            logger.debug(f"[AgentEntry:STT_Handler] Interim event received: {stt_event.alternatives[0].text if stt_event.alternatives else 'No alternatives'}")
            asyncio.create_task(publish_stt_data(stt_event, "STT_Interim"))

        def stt_final_handler(stt_event: Any):
            logger.debug(f"[AgentEntry:STT_Handler] Final event received: {stt_event.alternatives[0].text if stt_event.alternatives else 'No alternatives'}")
            asyncio.create_task(publish_stt_data(stt_event, "STT_Final"))

        # Attach event listeners to the STT plugin instance
        stt.on("interim_transcript", stt_interim_handler) 
        stt.on("final_transcript", stt_final_handler)
        logger.info("[AgentEntry] STT event listeners attached to STT plugin instance.")

        session = AgentSession(
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
            turn_detection=turn_detection,
        )
        logger.info("[AgentEntry] AgentSession created.")

        assistant_agent_instance = GUIFocusedAssistant()
        logger.info("[AgentEntry] GUIFocusedAssistant instance created.")

        logger.info("[AgentEntry] Starting AgentSession...")
        main_agent_task = asyncio.create_task(session.start(
            room=ctx.room, 
            agent=assistant_agent_instance,
        ))
        logger.info("[AgentEntry] AgentSession.start() called and task created.")
        await asyncio.sleep(1)
        initial_greeting = "Hello! I'm your voice assistant, ready to chat. How can I help you today?"
        logger.info(f"[AgentEntry] Generating initial reply: '{initial_greeting}'")
        await session.generate_reply(instructions=initial_greeting)
        logger.info("[AgentEntry] Initial reply generation complete.")

        # Keep the session alive by waiting for the room to disconnect
        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(1)
            logger.debug("[AgentEntry] Room still connected, keeping session alive...")

        await main_agent_task
        logger.info("[AgentEntry] Main agent task completed.")

    except Exception as e_session:
        logger.error(f"[AgentEntry] Error during AgentSession setup or execution: {e_session}", exc_info=True)
        raise
    finally:
        logger.info(f"[AgentEntry] Agent logic finished or terminated for job {job_id_from_ctx}.")

def run_worker():
    try:
        logger.info("--- run_worker: Attempting to start LiveKit Agent Worker ---")
        if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
            logger.critical("!!! run_worker: CRITICAL - LiveKit connection credentials (URL, KEY, SECRET) NOT FOUND. Worker will NOT start.")
            return
        
        if not OPENAI_API_KEY_ENV: logger.warning("!!! run_worker: OpenAI API key not found. LLM/TTS plugins might fail.")
        if not DEEPGRAM_API_KEY_ENV: logger.warning("!!! run_worker: Deepgram API key not found. STT plugin might fail.")

        worker_options = WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
        
        original_argv = list(sys.argv)
        sys.argv = ['livekit_agent_script_embedded.py', 'start'] 
        logger.info(f"run_worker: Modified sys.argv for cli.run_app: {sys.argv}")
        
        logger.info("run_worker: Applying nest_asyncio and calling cli.run_app(worker_options)...")
        try:
            import nest_asyncio
            nest_asyncio.apply()
            logger.info("run_worker: Applied nest_asyncio.")
            cli.run_app(worker_options)
        finally:
            sys.argv = original_argv
            logger.info(f"run_worker: Restored original sys.argv: {sys.argv}")

        logger.info("--- run_worker: LiveKit Agent Worker `cli.run_app` finished. ---")

    except SystemExit as se:
        logger.info(f"run_worker: cli.run_app exited with SystemExit (code: {se.code}). Normal for shutdown/CLI error.")
    except Exception as e:
        logger.error(f"!!! run_worker: CRITICAL ERROR during run_worker or cli.run_app: {e}", exc_info=True)
    finally:
        logger.info("--- run_worker: Function execution complete or terminated. ---")


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    logger.info("--- __main__: main.py script execution started ---")

    if len(sys.argv) > 1 and sys.argv[1] == "download-files":
        logger.info("--- __main__: 'download-files' argument detected. ---")
        logger.info("--- __main__: This script will attempt to initialize plugins which may trigger downloads if models are not cached.")
        logger.info("--- __main__: For a dedicated download, use a simpler agent script: `python your_simple_agent.py download-files`")
        # The worker will initialize plugins. If you only want to download,
        # you'd ideally call a function that just initializes plugins and exits.
        # For now, this will proceed to start the full worker and Uvicorn.
        # The actual downloads happen when SileroVAD.load() or MultilingualModel() are called.
        pass 

    logger.info("--- __main__: Creating worker thread (LiveKitAgentWorker) ---")
    worker_thread = threading.Thread(target=run_worker, name="LiveKitAgentWorker", daemon=True) 
    worker_thread.start()
    logger.info("--- __main__: Worker thread started.")

    logger.info("--- __main__: Waiting a few seconds for agent worker to initialize before starting Uvicorn...")
    time.sleep(5) 
    
    if worker_thread.is_alive():
        logger.info("--- __main__: Worker thread IS ALIVE. Proceeding to start Uvicorn.")
    else:
        logger.error("--- __main__: Worker thread IS NOT ALIVE. It likely exited. Check 'run_worker' logs.")

    logger.info("--- __main__: Starting Uvicorn server for FastAPI app ---")
    try:
        uvicorn.run(
            "main:app", 
            host="0.0.0.0",
            port=8000, 
            reload=True, 
            log_level="info"
        )
    except Exception as e_uvicorn:
        logger.error(f"--- __main__: Uvicorn server failed: {e_uvicorn}", exc_info=True)
    
    logger.info("--- __main__: Uvicorn server has stopped. ---")
    logger.info("--- __main__: Script execution finished. ---")


































































# 14-05-2-25

# working suer query logging main.py





import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from datetime import timedelta
import threading
import time 
import sys 
import psutil
from typing import Any, AsyncIterator

# LiveKit specific imports
from livekit.api import AccessToken, VideoGrants
from livekit import agents, rtc
from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    Agent, 
    AgentSession,
    AutoSubscribe,
)
from livekit.agents.llm import LLMStream
from livekit.agents import voice

# LiveKit Plugins
from livekit.plugins import deepgram, openai, silero

# Load environment variables
load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:3000", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging with console and file output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_logs.log')
    ]
)
logger = logging.getLogger(__name__)
agent_logger = logging.getLogger("livekit.agents")
agent_logger.setLevel(logging.DEBUG)

# Fetch environment variables
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY_ENV = os.getenv("ELEVENLABS_API_KEY")
DEEPGRAM_API_KEY_ENV = os.getenv("DEEPGRAM_API_KEY")

logger.info(f"MODULE LEVEL: LIVEKIT_URL: {LIVEKIT_URL}")
logger.info(f"MODULE LEVEL: LIVEKIT_API_KEY IS SET: {bool(LIVEKIT_API_KEY)}")
logger.info(f"MODULE LEVEL: DEEPGRAM_API_KEY IS SET: {bool(DEEPGRAM_API_KEY_ENV)}")
logger.info(f"MODULE LEVEL: OPENAI_API_KEY IS SET: {bool(OPENAI_API_KEY_ENV)}")

@app.get("/token")
async def get_token():
    logger.info("[TokenSvc] /token endpoint hit.")
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        logger.error("[TokenSvc] LiveKit credentials not configured properly.")
        raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

    video_grant_obj = VideoGrants(
        room_join=True,
        room="voice-assistant-room", 
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )

    token_builder = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token_builder.identity = "frontend_user" 
    token_builder.name = "Frontend User"
    token_builder.ttl = timedelta(hours=1)
    token_builder.with_grants(video_grant_obj)
    token_jwt = token_builder.to_jwt()

    logger.info(f"[TokenSvc] Generated token for frontend_user to join 'voice-assistant-room'")
    return {"token": token_jwt, "url": LIVEKIT_URL}

# --- Define the Assistant Agent ---
class GUIFocusedAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly voice assistant created by LiveKit, integrated into a web GUI. "
                "Your purpose is to engage in general conversation, answer questions, "
                "and provide helpful responses. Keep your answers concise, natural, "
                "and suitable for voice interaction. You are talking to a user through a web interface."
            )
        )
        self.last_user_query = None
        logger.debug("[AssistantGUI] Initialized GUIFocusedAssistant.")

    async def on_transcript(self, transcript: str, is_final: bool) -> None:
        logger.debug(f"[AssistantGUI] Received transcript: '{transcript}', is_final={is_final}")
        if is_final and transcript:
            self.last_user_query = transcript
            logger.info(f"[AssistantGUI] USER_QUERY: '{self.last_user_query}'")

    async def publish_llm_stream_to_user(self, llm_stream: LLMStream) -> None:
        logger.debug("[AssistantGUI] Starting publish_llm_stream_to_user.")
        collected_text_for_frontend = ""
        sentences_for_tts = []

        async for sentence in self.llm_stream_sentences(llm_stream):
            logger.debug(f"[AssistantGUI] LLM sentence: '{sentence}'")
            collected_text_for_frontend += sentence
            sentences_for_tts.append(sentence)

        if self.last_user_query and collected_text_for_frontend:
            logger.info(
                f"[AssistantGUI] Conversation Exchange:\n"
                f"  USER_QUERY: '{self.last_user_query}'\n"
                f"  MODEL_RESPONSE: '{collected_text_for_frontend}'"
            )
        elif collected_text_for_frontend:
            logger.info(f"[AssistantGUI] MODEL_RESPONSE (No User Query): '{collected_text_for_frontend}'")
        else:
            logger.info("[AssistantGUI] MODEL_RESPONSE: (empty text)")

        if collected_text_for_frontend and self.room:
            logger.info(f"[AssistantGUI] Publishing LLM response to frontend: '{collected_text_for_frontend}'")
            for attempt in range(3):
                try:
                    await self.room.local_participant.publish_data(
                        f"response:{collected_text_for_frontend}", "response"
                    )
                    logger.info(f"[AssistantGUI] Successfully published LLM response to frontend: '{collected_text_for_frontend}'")
                    break
                except Exception as e_publish:
                    logger.error(f"[AssistantGUI] Error publishing LLM response data (attempt {attempt+1}): {e_publish}", exc_info=True)
                    if attempt < 2:
                        await asyncio.sleep(1)
                    else:
                        logger.error("[AssistantGUI] Failed to publish LLM response after 3 attempts.")

        async def tts_sentence_generator():
            for s in sentences_for_tts:
                yield s

        if sentences_for_tts:
            logger.info(f"[AssistantGUI] Synthesizing audio for: '{collected_text_for_frontend}'")
            try:
                await self.tts.synthesize(tts_sentence_generator())
                logger.info(f"[AssistantGUI] Audio synthesis complete for: '{collected_text_for_frontend}'")
            except Exception as e_tts:
                logger.error(f"[AssistantGUI] Error during TTS synthesis: {e_tts}", exc_info=True)
        else:
            logger.info("[AssistantGUI] No sentences from LLM to synthesize (response might be empty).")

        self.last_user_query = None
        logger.debug("[AssistantGUI] Finished publish_llm_stream_to_user.")

# --- Agent Entrypoint ---
async def entrypoint(ctx: JobContext):
    room_name_from_ctx = ctx.room.name if ctx.room else "N/A_ROOM_IN_CTX"
    job_id_from_ctx = ctx.job.id if ctx.job else "N/A_JOB_IN_CTX"
    logger.info(f"--- [AgentEntry] AGENT LOGIC ENTRYPOINT CALLED --- Room: {room_name_from_ctx}, Job ID: {job_id_from_ctx}")
    logger.debug("TEST: Entrypoint function is running.")
    logger.info("[AgentEntry] Running main.py with turn_detection=None workaround.")

    try:
        logger.info(f"[AgentEntry] Ensuring connection and setting subscriptions for job {job_id_from_ctx}.")
        for attempt in range(3):
            try:
                await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
                logger.info(f"[AgentEntry] Connected/Verified. Participant SID: {ctx.room.local_participant.sid}")
                break
            except Exception as e_connect:
                logger.error(f"[AgentEntry] Error during ctx.connect() (attempt {attempt+1}): {e_connect}", exc_info=True)
                if attempt < 2:
                    await asyncio.sleep(2)
                else:
                    logger.error("[AgentEntry] Failed to connect after 3 attempts.")
                    raise
    except Exception as e_connect:
        logger.error(f"[AgentEntry] Fatal error during ctx.connect(): {e_connect}", exc_info=True)
        raise

    try:
        logger.info("[AgentEntry] Initializing plugins...")
        stt = deepgram.STT(
            model="nova-2",
            language="multi",
        )
        logger.debug("[AgentEntry] Deepgram STT initialized with model=nova-2, language=multi")
        llm = openai.LLM(
            model="gpt-4o-mini",
        )
        logger.debug("[AgentEntry] OpenAI LLM initialized with model=gpt-4o-mini")
        tts = openai.TTS(
            voice="ash",
        )
        logger.debug("[AgentEntry] OpenAI TTS initialized with voice=ash")
        vad = silero.VAD.load()
        logger.debug("[AgentEntry] Silero VAD initialized with default settings")
        # Explicitly disable turn detection
        turn_detection = None
        logger.info("[AgentEntry] Plugins initialized (turn detection explicitly disabled).")

        async def publish_stt_data(stt_event: Any, topic_prefix="transcription"):
            if stt_event.alternatives and stt_event.alternatives[0].text:
                text = stt_event.alternatives[0].text
                logger.info(f"[AgentEntry:{topic_prefix}] Attempting to publish STT to frontend: '{text}'")
                for attempt in range(3):
                    try:
                        await ctx.room.local_participant.publish_data(f"transcription:{text}", "transcription")
                        logger.info(f"[AgentEntry:{topic_prefix}] Successfully published STT to frontend: '{text}'")
                        break
                    except Exception as e_publish_stt:
                        logger.error(f"[AgentEntry] Error publishing STT data (attempt {attempt+1}): {e_publish_stt}", exc_info=True)
                        if attempt < 2:
                            await asyncio.sleep(1)
                        else:
                            logger.error("[AgentEntry] Failed to publish STT data after 3 attempts.")
            else:
                logger.debug(f"[AgentEntry:{topic_prefix}] No text in STT event: {stt_event}")

        def stt_interim_handler(stt_event: Any):
            text = stt_event.alternatives[0].text if stt_event.alternatives else "No alternatives"
            logger.debug(f"[AgentEntry:STT_Handler] Interim event received: '{text}'")
            asyncio.create_task(publish_stt_data(stt_event, "STT_Interim"))

        def stt_final_handler(stt_event: Any):
            if stt_event.alternatives and stt_event.alternatives[0].text:
                text = stt_event.alternatives[0].text
                logger.info(f"[AgentEntry:STT_Handler] USER_QUERY: '{text}'")
                asyncio.create_task(publish_stt_data(stt_event, "STT_Final"))
            else:
                logger.debug(f"[AgentEntry:STT_Handler] Final STT event received but no text or alternatives: {stt_event}")

        def vad_handler(is_speech: bool):
            logger.debug(f"[AgentEntry:VAD] Speech detected: {is_speech}")

        stt.on("interim_transcript", stt_interim_handler)
        stt.on("final_transcript", stt_final_handler)
        vad.on("speech_detected", vad_handler)
        logger.info("[AgentEntry] STT and VAD event listeners attached.")

        # Workaround: Force-disable turn detection in AgentSession
        session = AgentSession(
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
            turn_detection=None,
        )
        logger.info("[AgentEntry] AgentSession created with turn_detection=None.")

        # Additional debug: Verify turn detection is disabled
        if hasattr(session, 'turn_detector') and session.turn_detector is not None:
            logger.warning("[AgentEntry] Turn detection is still enabled in AgentSession! Forcing disable.")
            session.turn_detector = None

        assistant_agent_instance = GUIFocusedAssistant()
        logger.info("[AgentEntry] GUIFocusedAssistant instance created.")

        logger.info("[AgentEntry] Starting AgentSession...")
        main_agent_task = asyncio.create_task(session.start(
            room=ctx.room,
            agent=assistant_agent_instance,
        ))
        logger.info("[AgentEntry] AgentSession.start() called and task created.")

        await asyncio.sleep(1)
        initial_greeting = "Hello! I'm your voice assistant, ready to chat. How can I help you today?"
        logger.info(f"[AgentEntry] Generating initial reply (agent self-initiated): '{initial_greeting}'")
        try:
            await session.generate_reply(instructions=initial_greeting)
            logger.info("[AgentEntry] Initial reply generation requested.")
        except Exception as e_greeting:
            logger.error(f"[AgentEntry] Error generating initial greeting: {e_greeting}", exc_info=True)

        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(1)
            logger.debug("[AgentEntry] Room still connected, keeping session alive...")

        await main_agent_task
        logger.info("[AgentEntry] Main agent task completed.")

    except Exception as e_session:
        logger.error(f"[AgentEntry] Error during AgentSession setup or execution: {e_session}", exc_info=True)
        raise
    finally:
        logger.info(f"[AgentEntry] Agent logic finished or terminated for job {job_id_from_ctx}.")

def check_and_free_port(port):
    """Check if a port is in use and attempt to free it."""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.net_connections(kind='inet'):
                if conn.laddr.port == port:
                    logger.warning(f"Port {port} is in use by PID {proc.pid} ({proc.name()}). Attempting to terminate.")
                    proc.terminate()
                    proc.wait(timeout=5)
                    logger.info(f"Successfully terminated process on port {port}.")
                    return
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    logger.debug(f"Port {port} is free or no process could be terminated.")

def run_worker():
    try:
        logger.info("--- run_worker: Attempting to start LiveKit Agent Worker ---")
        if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
            logger.critical("!!! run_worker: CRITICAL - LiveKit connection credentials (URL, KEY, SECRET) NOT FOUND. Worker will NOT start.")
            return
        
        if not OPENAI_API_KEY_ENV: logger.warning("!!! run_worker: OpenAI API key not found. LLM/TTS plugins might fail.")
        if not DEEPGRAM_API_KEY_ENV: logger.warning("!!! run_worker: Deepgram API key not found. STT plugin might fail.")

        # Check and free port 8081 for LiveKit worker
        check_and_free_port(8081)

        worker_options = WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
        
        original_argv = list(sys.argv)
        sys.argv = ['livekit_agent_script_embedded.py', 'start']
        logger.info(f"run_worker: Modified sys.argv for cli.run_app: {sys.argv}")
        
        logger.info("run_worker: Applying nest_asyncio and calling cli.run_app(worker_options)...")
        try:
            import nest_asyncio
            nest_asyncio.apply()
            logger.info("run_worker: Applied nest_asyncio.")
            cli.run_app(worker_options)
        finally:
            sys.argv = original_argv
            logger.info(f"run_worker: Restored original sys.argv: {sys.argv}")

        logger.info("--- run_worker: LiveKit Agent Worker `cli.run_app` finished. ---")

    except SystemExit as se:
        logger.info(f"run_worker: cli.run_app exited with SystemExit (code: {se.code}). This is often normal.")
    except Exception as e:
        logger.error(f"!!! run_worker: CRITICAL ERROR during run_worker or cli.run_app: {e}", exc_info=True)
    finally:
        logger.info("--- run_worker: Function execution complete or terminated. ---")

if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    logger.info("--- __main__: main.py script execution started ---")

    if len(sys.argv) > 1 and sys.argv[1] == "download-files":
        logger.info("--- __main__: 'download-files' argument detected. ---")
        logger.info("--- __main__: Plugins will initialize when the worker starts, triggering downloads if needed.")
        pass 

    # Check and free port 8000 for Uvicorn
    check_and_free_port(8000)

    logger.info("--- __main__: Creating worker thread (LiveKitAgentWorker) ---")
    worker_thread = threading.Thread(target=run_worker, name="LiveKitAgentWorker", daemon=True)
    worker_thread.start()
    logger.info("--- __main__: Worker thread started.")

    logger.info("--- __main__: Waiting a few seconds for agent worker to initialize before starting Uvicorn...")
    time.sleep(5)
    
    if worker_thread.is_alive():
        logger.info("--- __main__: Worker thread IS ALIVE. Proceeding to start Uvicorn.")
    else:
        logger.error("--- __main__: Worker thread IS NOT ALIVE. It likely exited. Check 'run_worker' logs for errors (e.g., missing credentials).")

    logger.info("--- __main__: Starting Uvicorn server for FastAPI app ---")
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except Exception as e_uvicorn:
        logger.error(f"--- __main__: Uvicorn server failed: {e_uvicorn}", exc_info=True)
    
    logger.info("--- __main__: Uvicorn server has stopped. ---")
    logger.info("--- __main__: Script execution finished. ---")









# 15-05-2025
# 11:56
# main.py



import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from datetime import timedelta
import threading
import time 
import sys 
import psutil
from typing import Any, AsyncIterator

# LiveKit specific imports
from livekit.api import AccessToken, VideoGrants
from livekit import agents, rtc
from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    Agent, 
    AgentSession,
    AutoSubscribe,
)
from livekit.agents.llm import LLMStream
from livekit.agents import voice

# LiveKit Plugins
from livekit.plugins import deepgram, openai, silero

# Load environment variables
load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:3000", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging with console and file output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_logs.log')
    ]
)
logger = logging.getLogger(__name__)
agent_logger = logging.getLogger("livekit.agents")
agent_logger.setLevel(logging.DEBUG)

# Fetch environment variables
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY_ENV = os.getenv("ELEVENLABS_API_KEY")
DEEPGRAM_API_KEY_ENV = os.getenv("DEEPGRAM_API_KEY")

logger.info(f"MODULE LEVEL: LIVEKIT_URL: {LIVEKIT_URL}")
logger.info(f"MODULE LEVEL: LIVEKIT_API_KEY IS SET: {bool(LIVEKIT_API_KEY)}")
logger.info(f"MODULE LEVEL: DEEPGRAM_API_KEY IS SET: {bool(DEEPGRAM_API_KEY_ENV)}")
logger.info(f"MODULE LEVEL: OPENAI_API_KEY IS SET: {bool(OPENAI_API_KEY_ENV)}")

@app.get("/token")
async def get_token():
    logger.info("[TokenSvc] /token endpoint hit.")
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        logger.error("[TokenSvc] LiveKit credentials not configured properly.")
        raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

    video_grant_obj = VideoGrants(
        room_join=True,
        room="voice-assistant-room", 
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )

    token_builder = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token_builder.identity = "frontend_user" 
    token_builder.name = "Frontend User"
    token_builder.ttl = timedelta(hours=1)
    token_builder.with_grants(video_grant_obj)
    token_jwt = token_builder.to_jwt()

    logger.info(f"[TokenSvc] Generated token for frontend_user to join 'voice-assistant-room'")
    return {"token": token_jwt, "url": LIVEKIT_URL}

# --- Define the Assistant Agent ---
class GUIFocusedAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly voice assistant created by LiveKit, integrated into a web GUI. "
                "Your purpose is to engage in general conversation, answer questions, "
                "and provide helpful responses. Keep your answers concise, natural, "
                "and suitable for voice interaction. You are talking to a user through a web interface."
            )
        )
        self.last_user_query = None
        logger.debug("[AssistantGUI] Initialized GUIFocusedAssistant.")

    async def on_transcript(self, transcript: str, is_final: bool) -> None:
        logger.debug(f"[AssistantGUI] Received transcript: '{transcript}', is_final={is_final}")
        if is_final and transcript:
            self.last_user_query = transcript
            logger.info(f"[AssistantGUI] USER_QUERY: '{self.last_user_query}'")

    async def publish_llm_stream_to_user(self, llm_stream: LLMStream) -> None:
        logger.debug("[AssistantGUI] Starting publish_llm_stream_to_user.")
        collected_text_for_frontend = ""
        sentences_for_tts = []

        try:
            async for sentence in self.llm_stream_sentences(llm_stream):
                logger.debug(f"[AssistantGUI] LLM sentence chunk: '{sentence.text}'")
                logger.info("1111")
                # logger.debug(f"[AssistantGUI] LLM sentence received: '{sentence}'")
                collected_text_for_frontend += sentence + " "
                sentences_for_tts.append(sentence.text)
        except Exception as e_llm:
            logger.error(f"[AssistantGUI] Error processing LLM stream: {e_llm}", exc_info=True)
            logger.error("[AssistantGUI] LLM stream failed or returned no data. Skipping normal response logging.")
            logger.info("2222")
            collected_text_for_frontend = "Error generating response."

        # Log the response regardless of content
        if self.last_user_query and collected_text_for_frontend:
            logger.info(
                f"[AssistantGUI] Conversation Exchange:\n"
                f"  USER_QUERY: '{self.last_user_query}'\n"
                f"  MODEL_RESPONSE: '{collected_text_for_frontend.strip()}'"
            )
            logger.info("333333")
        elif collected_text_for_frontend:
            logger.info(f"[AssistantGUI] MODEL_RESPONSE (No User Query): '{collected_text_for_frontend.strip()}'")
            logger.info("44444")
        else:
            logger.warning("[AssistantGUI] MODEL_RESPONSE: Empty or failed response.")
            logger.info("555")

        if collected_text_for_frontend and self.room:
            logger.info(f"[AssistantGUI] Publishing LLM response to frontend: '{collected_text_for_frontend.strip()}'")
            for attempt in range(3):
                try:
                    await self.room.local_participant.publish_data(
                        f"response:{collected_text_for_frontend.strip()}", "response"
                    )
                    logger.info(f"[AssistantGUI] Successfully published LLM response to frontend: '{collected_text_for_frontend.strip()}'")
                    break
                except Exception as e_publish:
                    logger.error(f"[AssistantGUI] Error publishing LLM response (attempt {attempt+1}): {e_publish}", exc_info=True)
                    if attempt < 2:
                        await asyncio.sleep(1)
                    else:
                        logger.error("[AssistantGUI] Failed to publish LLM response after 3 attempts.")

        async def tts_sentence_generator():
            for s in sentences_for_tts:
                yield s

        if sentences_for_tts:
            logger.info(f"[AssistantGUI] Synthesizing audio for: '{collected_text_for_frontend.strip()}'")
            try:
                await self.tts.synthesize(tts_sentence_generator())
                logger.info(f"[AssistantGUI] MODEL_RESPONSE: '{collected_text_for_frontend.strip()}'")
            except Exception as e_tts:
                logger.error(f"[AssistantGUI] Error during TTS synthesis: {e_tts}", exc_info=True)
        else:
            logger.warning("[AssistantGUI] No sentences for TTS synthesis.")

        self.last_user_query = None
        logger.debug("[AssistantGUI] Finished publish_llm_stream_to_user.")


# --- Agent Entrypoint ---
async def entrypoint(ctx: JobContext):
    room_name_from_ctx = ctx.room.name if ctx.room else "N/A_ROOM_IN_CTX"
    job_id_from_ctx = ctx.job.id if ctx.job else "N/A_JOB_IN_CTX"
    logger.info(f"--- [AgentEntry] AGENT LOGIC ENTRYPOINT CALLED --- Room: {room_name_from_ctx}, Job ID: {job_id_from_ctx}")
    logger.debug("TEST: Entrypoint function is running.")
    logger.info("[AgentEntry] Running main.py with turn_detection=None workaround.")

    try:
        logger.info(f"[AgentEntry] Ensuring connection and setting subscriptions for job {job_id_from_ctx}.")
        for attempt in range(3):
            try:
                await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
                logger.info(f"[AgentEntry] Connected/Verified. Participant SID: {ctx.room.local_participant.sid}")
                break
            except Exception as e_connect:
                logger.error(f"[AgentEntry] Error during ctx.connect() (attempt {attempt+1}): {e_connect}", exc_info=True)
                if attempt < 2:
                    await asyncio.sleep(2)
                else:
                    logger.error("[AgentEntry] Failed to connect after 3 attempts.")
                    raise
    except Exception as e_connect:
        logger.error(f"[AgentEntry] Fatal error during ctx.connect(): {e_connect}", exc_info=True)
        raise

    try:
        logger.info("[AgentEntry] Initializing plugins...")
        stt = deepgram.STT(
            model="nova-2",
            language="multi",
        )
        logger.debug("[AgentEntry] Deepgram STT initialized with model=nova-2, language=multi")
        llm = openai.LLM(
            model="gpt-4o-mini",
        )
        logger.debug("[AgentEntry] OpenAI LLM initialized with model=gpt-4o-mini")
        tts = openai.TTS(
            voice="ash",
        )
        logger.debug("[AgentEntry] OpenAI TTS initialized with voice=ash")
        vad = silero.VAD.load()
        logger.debug("[AgentEntry] Silero VAD initialized with default settings")
        # Explicitly disable turn detection
        turn_detection = None
        logger.info("[AgentEntry] Plugins initialized (turn detection explicitly disabled).")

        async def publish_stt_data(stt_event: Any, topic_prefix="transcription"):
            if stt_event.alternatives and stt_event.alternatives[0].text:
                text = stt_event.alternatives[0].text
                logger.info(f"[AgentEntry:{topic_prefix}] Attempting to publish STT to frontend: '{text}'")
                for attempt in range(3):
                    try:
                        await ctx.room.local_participant.publish_data(f"transcription:{text}", "transcription")
                        logger.info(f"[AgentEntry:{topic_prefix}] Successfully published STT to frontend: '{text}'")
                        break
                    except Exception as e_publish_stt:
                        logger.error(f"[AgentEntry] Error publishing STT data (attempt {attempt+1}): {e_publish_stt}", exc_info=True)
                        if attempt < 2:
                            await asyncio.sleep(1)
                        else:
                            logger.error("[AgentEntry] Failed to publish STT data after 3 attempts.")
            else:
                logger.debug(f"[AgentEntry:{topic_prefix}] No text in STT event: {stt_event}")

        def stt_interim_handler(stt_event: Any):
            text = stt_event.alternatives[0].text if stt_event.alternatives else "No alternatives"
            logger.debug(f"[AgentEntry:STT_Handler] Interim event received: '{text}'")
            asyncio.create_task(publish_stt_data(stt_event, "STT_Interim"))

        def stt_final_handler(stt_event: Any):
            if stt_event.alternatives and stt_event.alternatives[0].text:
                text = stt_event.alternatives[0].text
                logger.info(f"[AgentEntry:STT_Handler] USER_QUERY: '{text}'")
                asyncio.create_task(publish_stt_data(stt_event, "STT_Final"))
            else:
                logger.debug(f"[AgentEntry:STT_Handler] Final STT event received but no text or alternatives: {stt_event}")

        def vad_handler(is_speech: bool):
            logger.debug(f"[AgentEntry:VAD] Speech detected: {is_speech}")

        stt.on("interim_transcript", stt_interim_handler)
        stt.on("final_transcript", stt_final_handler)
        vad.on("speech_detected", vad_handler)
        logger.info("[AgentEntry] STT and VAD event listeners attached.")

        # Workaround: Force-disable turn detection in AgentSession
        session = AgentSession(
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
            turn_detection=None,
        )
        logger.info("[AgentEntry] AgentSession created with turn_detection=None.")

        # Additional debug: Verify turn detection is disabled
        if hasattr(session, 'turn_detector') and session.turn_detector is not None:
            logger.warning("[AgentEntry] Turn detection is still enabled in AgentSession! Forcing disable.")
            session.turn_detector = None

        assistant_agent_instance = GUIFocusedAssistant()
        logger.info("[AgentEntry] GUIFocusedAssistant instance created.")

        logger.info("[AgentEntry] Starting AgentSession...")
        main_agent_task = asyncio.create_task(session.start(
            room=ctx.room,
            agent=assistant_agent_instance,
        ))
        logger.info("[AgentEntry] AgentSession.start() called and task created.")

        await asyncio.sleep(1)
        initial_greeting = "Hello! I'm your voice assistant, ready to chat. How can I help you today?"
        logger.info(f"[AgentEntry] Generating initial reply (agent self-initiated): '{initial_greeting}'")
        try:
            await session.generate_reply(instructions=initial_greeting)
            logger.info("[AgentEntry] Initial reply generation requested.")
        except Exception as e_greeting:
            logger.error(f"[AgentEntry] Error generating initial greeting: {e_greeting}", exc_info=True)

        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(1)
            logger.debug("[AgentEntry] Room still connected, keeping session alive...")

        await main_agent_task
        logger.info("[AgentEntry] Main agent task completed.")

    except Exception as e_session:
        logger.error(f"[AgentEntry] Error during AgentSession setup or execution: {e_session}", exc_info=True)
        raise
    finally:
        logger.info(f"[AgentEntry] Agent logic finished or terminated for job {job_id_from_ctx}.")

def check_and_free_port(port):
    """Check if a port is in use and attempt to free it."""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.net_connections(kind='inet'):
                if conn.laddr.port == port:
                    logger.warning(f"Port {port} is in use by PID {proc.pid} ({proc.name()}). Attempting to terminate.")
                    proc.terminate()
                    proc.wait(timeout=5)
                    logger.info(f"Successfully terminated process on port {port}.")
                    return
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    logger.debug(f"Port {port} is free or no process could be terminated.")

def run_worker():
    try:
        logger.info("--- run_worker: Attempting to start LiveKit Agent Worker ---")
        if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
            logger.critical("!!! run_worker: CRITICAL - LiveKit connection credentials (URL, KEY, SECRET) NOT FOUND. Worker will NOT start.")
            return
        
        if not OPENAI_API_KEY_ENV: logger.warning("!!! run_worker: OpenAI API key not found. LLM/TTS plugins might fail.")
        if not DEEPGRAM_API_KEY_ENV: logger.warning("!!! run_worker: Deepgram API key not found. STT plugin might fail.")

        # Check and free port 8081 for LiveKit worker
        check_and_free_port(8081)

        worker_options = WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
        
        original_argv = list(sys.argv)
        sys.argv = ['livekit_agent_script_embedded.py', 'start']
        logger.info(f"run_worker: Modified sys.argv for cli.run_app: {sys.argv}")
        
        logger.info("run_worker: Applying nest_asyncio and calling cli.run_app(worker_options)...")
        try:
            import nest_asyncio
            nest_asyncio.apply()
            logger.info("run_worker: Applied nest_asyncio.")
            cli.run_app(worker_options)
        finally:
            sys.argv = original_argv
            logger.info(f"run_worker: Restored original sys.argv: {sys.argv}")

        logger.info("--- run_worker: LiveKit Agent Worker `cli.run_app` finished. ---")

    except SystemExit as se:
        logger.info(f"run_worker: cli.run_app exited with SystemExit (code: {se.code}). This is often normal.")
    except Exception as e:
        logger.error(f"!!! run_worker: CRITICAL ERROR during run_worker or cli.run_app: {e}", exc_info=True)
    finally:
        logger.info("--- run_worker: Function execution complete or terminated. ---")

if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    logger.info("--- __main__: main.py script execution started ---")

    if len(sys.argv) > 1 and sys.argv[1] == "download-files":
        logger.info("--- __main__: 'download-files' argument detected. ---")
        logger.info("--- __main__: Plugins will initialize when the worker starts, triggering downloads if needed.")
        pass 

    # Check and free port 8000 for Uvicorn
    check_and_free_port(8000)

    logger.info("--- __main__: Creating worker thread (LiveKitAgentWorker) ---")
    worker_thread = threading.Thread(target=run_worker, name="LiveKitAgentWorker", daemon=True)
    worker_thread.start()
    logger.info("--- __main__: Worker thread started.")

    logger.info("--- __main__: Waiting a few seconds for agent worker to initialize before starting Uvicorn...")
    time.sleep(5)
    
    if worker_thread.is_alive():
        logger.info("--- __main__: Worker thread IS ALIVE. Proceeding to start Uvicorn.")
    else:
        logger.error("--- __main__: Worker thread IS NOT ALIVE. It likely exited. Check 'run_worker' logs for errors (e.g., missing credentials).")

    logger.info("--- __main__: Starting Uvicorn server for FastAPI app ---")
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except Exception as e_uvicorn:
        logger.error(f"--- __main__: Uvicorn server failed: {e_uvicorn}", exc_info=True)
    
    logger.info("--- __main__: Uvicorn server has stopped. ---")
    logger.info("--- __main__: Script execution finished. ---")












# 15-05-2025
# 12:13pm
# so far so best







import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from datetime import timedelta
import threading
import time 
import sys 
import psutil
from typing import Any, AsyncIterator

# LiveKit specific imports
from livekit.api import AccessToken, VideoGrants
from livekit import agents, rtc
from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    Agent, 
    AgentSession,
    AutoSubscribe,
)
from livekit.agents.llm import LLMStream
from livekit.agents import voice

# LiveKit Plugins
from livekit.plugins import deepgram, openai, silero

# Load environment variables
load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:3000", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging with console and file output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_logs.log')
    ]
)
logger = logging.getLogger(__name__)
agent_logger = logging.getLogger("livekit.agents")
agent_logger.setLevel(logging.DEBUG)

# Fetch environment variables
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY_ENV = os.getenv("ELEVENLABS_API_KEY")
DEEPGRAM_API_KEY_ENV = os.getenv("DEEPGRAM_API_KEY")

logger.info(f"MODULE LEVEL: LIVEKIT_URL: {LIVEKIT_URL}")
logger.info(f"MODULE LEVEL: LIVEKIT_API_KEY IS SET: {bool(LIVEKIT_API_KEY)}")
logger.info(f"MODULE LEVEL: DEEPGRAM_API_KEY IS SET: {bool(DEEPGRAM_API_KEY_ENV)}")
logger.info(f"MODULE LEVEL: OPENAI_API_KEY IS SET: {bool(OPENAI_API_KEY_ENV)}")

@app.get("/token")
async def get_token():
    logger.info("[TokenSvc] /token endpoint hit.")
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        logger.error("[TokenSvc] LiveKit credentials not configured properly.")
        raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

    video_grant_obj = VideoGrants(
        room_join=True,
        room="voice-assistant-room", 
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )

    token_builder = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token_builder.identity = "frontend_user" 
    token_builder.name = "Frontend User"
    token_builder.ttl = timedelta(hours=1)
    token_builder.with_grants(video_grant_obj)
    token_jwt = token_builder.to_jwt()

    logger.info(f"[TokenSvc] Generated token for frontend_user to join 'voice-assistant-room'")
    return {"token": token_jwt, "url": LIVEKIT_URL}

# --- Define the Assistant Agent ---
class GUIFocusedAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly voice assistant created by LiveKit, integrated into a web GUI. "
                "Your purpose is to engage in general conversation, answer questions, "
                "and provide helpful responses. Keep your answers concise, natural, "
                "and suitable for voice interaction. You are talking to a user through a web interface."
            )
        )
        self.last_user_query = None # To store the last user query for context in logging
        # Ensure that the LLM and TTS plugins are attached to the agent instance
        # This is typically done by the AgentSession, but we need to access self.tts
        # If they are not directly on self, AgentSession will pass them during its methods.
        # For self.llm_stream_sentences and self.tts.synthesize to work, AgentSession must have initialized them.
        logger.debug("[AssistantGUI] Initialized GUIFocusedAssistant.")

    async def on_transcript(self, transcript: str, is_final: bool) -> None:
        # This method is part of the Agent base class and will be called by the AgentSession
        # when STT produces transcripts.
        logger.debug(f"[AssistantGUI:on_transcript] Received transcript: '{transcript}', is_final={is_final}")
        if is_final and transcript:
            self.last_user_query = transcript
            logger.info(f"[AssistantGUI:on_transcript] Stored last_user_query for logging context: '{self.last_user_query}'")
            # Note: Actual publishing of user transcript to frontend is handled by stt_final_handler in your entrypoint

    async def publish_llm_stream_to_user(self, llm_stream: LLMStream) -> None:
        # This method is called by AgentSession after the LLM generates a response stream.
        logger.debug("[AssistantGUI:publish_llm_stream_to_user] Starting to process and publish LLM stream.")
        collected_text_for_frontend = ""
        sentences_for_tts = []
        stream_processed_successfully = False # Flag to track if the stream loop ran

        try:
            # self.llm_stream_sentences is a helper from the base Agent class
            async for sentence_obj in self.llm_stream_sentences(llm_stream): 
                logger.debug(f"[AssistantGUI:publish_llm_stream_to_user] LLM sentence object received. Text: '{sentence_obj.text}'")
                logger.info("1111 - Processing LLM sentence from stream") 

                if sentence_obj.text: # Ensure the sentence text is not empty
                    collected_text_for_frontend += sentence_obj.text + " " # Append the TEXT content
                    sentences_for_tts.append(sentence_obj.text)
                
                stream_processed_successfully = True # Mark that we've processed at least one item

        except Exception as e_llm:
            logger.error(f"[AssistantGUI:publish_llm_stream_to_user] Error processing LLM stream: {e_llm}", exc_info=True)
            logger.info("2222 - Error occurred in LLM stream processing") 
            if not collected_text_for_frontend: 
                 collected_text_for_frontend = "Error: Could not generate a complete response due to a stream error."
            else: 
                 collected_text_for_frontend += " (Error occurred during response generation)"


        final_collected_text = collected_text_for_frontend.strip()

        # THIS IS THE TARGET LOG FOR BACKEND MODEL RESPONSE
        if self.last_user_query and final_collected_text:
            logger.info(
                f"[AssistantGUI:publish_llm_stream_to_user] Conversation Exchange:\n"
                f"  USER_QUERY: '{self.last_user_query}'\n"
                f"  MODEL_RESPONSE: '{final_collected_text}'"
            )
            logger.info("333333 - Logged Conversation Exchange (User Query + Model Response)")
        elif final_collected_text: 
            logger.info(f"[AssistantGUI:publish_llm_stream_to_user] MODEL_RESPONSE (No User Query context for this log entry): '{final_collected_text}'")
            logger.info("44444 - Logged Model Response (No User Query Context)")
        elif stream_processed_successfully: 
            logger.warning("[AssistantGUI:publish_llm_stream_to_user] MODEL_RESPONSE: LLM stream processed but yielded no text content.")
            logger.info("555A - Logged Empty Response (Stream Processed, No Text)")
        else: 
            logger.warning("[AssistantGUI:publish_llm_stream_to_user] MODEL_RESPONSE: Empty or failed response (LLM stream likely did not yield data or failed before starting).")
            logger.info("555B - Logged Empty/Failed Response (Stream Not Processed or Early Error)")


        # --- Publishing collected LLM response to frontend GUI ---
        if final_collected_text and self.room and self.room.local_participant:
            logger.info(f"[AssistantGUI:publish_llm_stream_to_user] Attempting to publish LLM response to frontend GUI: '{final_collected_text}'")
            try:
                await self.room.local_participant.publish_data(
                    payload=f"response:{final_collected_text}", # Ensure payload argument is named
                    topic="response"
                )
                logger.info(f"[AssistantGUI:publish_llm_stream_to_user] Successfully published LLM response to frontend GUI.")
            except Exception as e_publish:
                logger.error(f"[AssistantGUI:publish_llm_stream_to_user] Error publishing LLM response to frontend GUI: {e_publish}", exc_info=True)
        elif not final_collected_text:
            logger.warning("[AssistantGUI:publish_llm_stream_to_user] Not publishing to frontend: final_collected_text is empty.")
        elif not self.room:
            logger.error("[AssistantGUI:publish_llm_stream_to_user] Cannot publish to frontend: self.room is None.")
        elif not self.room.local_participant:
             logger.error("[AssistantGUI:publish_llm_stream_to_user] Cannot publish to frontend: self.room.local_participant is None.")


        # --- TTS Synthesis ---
        async def tts_sentence_generator_local(): 
            for s_text in sentences_for_tts:
                if s_text.strip():
                    yield s_text.strip()

        if sentences_for_tts:
            non_empty_sentences_for_tts = [s for s in sentences_for_tts if s.strip()]
            if non_empty_sentences_for_tts and hasattr(self, 'tts') and self.tts:
                logger.info(f"[AssistantGUI:publish_llm_stream_to_user] Starting TTS synthesis for: '{final_collected_text}'")
                try:
                    await self.tts.synthesize(tts_sentence_generator_local())
                    logger.info(f"[AssistantGUI:publish_llm_stream_to_user] TTS synthesis initiated for the response.")
                except Exception as e_tts:
                    logger.error(f"[AssistantGUI:publish_llm_stream_to_user] Error during TTS synthesis: {e_tts}", exc_info=True)
            elif not hasattr(self, 'tts') or not self.tts:
                 logger.error("[AssistantGUI:publish_llm_stream_to_user] TTS object (self.tts) not available for synthesis. AgentSession might not have attached it.")
            else: # non_empty_sentences_for_tts is empty
                logger.warning("[AssistantGUI:publish_llm_stream_to_user] No non-empty sentences for TTS synthesis, although stream processed and sentences were collected.")
        elif stream_processed_successfully: # Stream ran but no sentences were collected for TTS
            logger.warning("[AssistantGUI:publish_llm_stream_to_user] No sentences collected for TTS synthesis (stream yielded no text or only empty strings).")
        
        self.last_user_query = None # Clear after processing for the next turn
        logger.debug("[AssistantGUI:publish_llm_stream_to_user] Method finished.")


# --- Agent Entrypoint ---
async def entrypoint(ctx: JobContext):
    room_name_from_ctx = ctx.room.name if ctx.room else "N/A_ROOM_IN_CTX"
    job_id_from_ctx = ctx.job.id if ctx.job else "N/A_JOB_IN_CTX"
    logger.info(f"--- [AgentEntry] AGENT LOGIC ENTRYPOINT CALLED --- Room: {room_name_from_ctx}, Job ID: {job_id_from_ctx}")
    logger.debug("TEST: Entrypoint function is running.")
    logger.info("[AgentEntry] Running main.py with turn_detection=None workaround.")

    try:
        logger.info(f"[AgentEntry] Ensuring connection and setting subscriptions for job {job_id_from_ctx}.")
        for attempt in range(3):
            try:
                await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
                logger.info(f"[AgentEntry] Connected/Verified. Participant SID: {ctx.room.local_participant.sid}")
                break
            except Exception as e_connect:
                logger.error(f"[AgentEntry] Error during ctx.connect() (attempt {attempt+1}): {e_connect}", exc_info=True)
                if attempt < 2:
                    await asyncio.sleep(2)
                else:
                    logger.error("[AgentEntry] Failed to connect after 3 attempts.")
                    raise
    except Exception as e_connect:
        logger.error(f"[AgentEntry] Fatal error during ctx.connect(): {e_connect}", exc_info=True)
        raise

    try:
        logger.info("[AgentEntry] Initializing plugins...")
        stt = deepgram.STT(
            model="nova-2",
            language="multi",
        )
        logger.debug("[AgentEntry] Deepgram STT initialized with model=nova-2, language=multi")
        llm = openai.LLM(
            model="gpt-4o-mini",
        )
        logger.debug("[AgentEntry] OpenAI LLM initialized with model=gpt-4o-mini")
        tts = openai.TTS(
            voice="ash",
        )
        logger.debug("[AgentEntry] OpenAI TTS initialized with voice=ash")
        vad = silero.VAD.load()
        logger.debug("[AgentEntry] Silero VAD initialized with default settings")
        # Explicitly disable turn detection
        turn_detection = None
        logger.info("[AgentEntry] Plugins initialized (turn detection explicitly disabled).")

        async def publish_stt_data(stt_event: Any, topic_prefix="transcription"):
            if stt_event.alternatives and stt_event.alternatives[0].text:
                text = stt_event.alternatives[0].text
                logger.info(f"[AgentEntry:{topic_prefix}] Attempting to publish STT to frontend: '{text}'")
                for attempt in range(3):
                    try:
                        await ctx.room.local_participant.publish_data(f"transcription:{text}", "transcription")
                        logger.info(f"[AgentEntry:{topic_prefix}] Successfully published STT to frontend: '{text}'")
                        break
                    except Exception as e_publish_stt:
                        logger.error(f"[AgentEntry] Error publishing STT data (attempt {attempt+1}): {e_publish_stt}", exc_info=True)
                        if attempt < 2:
                            await asyncio.sleep(1)
                        else:
                            logger.error("[AgentEntry] Failed to publish STT data after 3 attempts.")
            else:
                logger.debug(f"[AgentEntry:{topic_prefix}] No text in STT event: {stt_event}")

        def stt_interim_handler(stt_event: Any):
            text = stt_event.alternatives[0].text if stt_event.alternatives else "No alternatives"
            logger.debug(f"[AgentEntry:STT_Handler] Interim event received: '{text}'")
            asyncio.create_task(publish_stt_data(stt_event, "STT_Interim"))

        def stt_final_handler(stt_event: Any):
            if stt_event.alternatives and stt_event.alternatives[0].text:
                text = stt_event.alternatives[0].text
                logger.info(f"[AgentEntry:STT_Handler] USER_QUERY: '{text}'")
                asyncio.create_task(publish_stt_data(stt_event, "STT_Final"))
            else:
                logger.debug(f"[AgentEntry:STT_Handler] Final STT event received but no text or alternatives: {stt_event}")

        def vad_handler(is_speech: bool):
            logger.debug(f"[AgentEntry:VAD] Speech detected: {is_speech}")

        stt.on("interim_transcript", stt_interim_handler)
        stt.on("final_transcript", stt_final_handler)
        vad.on("speech_detected", vad_handler)
        logger.info("[AgentEntry] STT and VAD event listeners attached.")

        # Workaround: Force-disable turn detection in AgentSession
        session = AgentSession(
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
            turn_detection=None,
        )
        logger.info("[AgentEntry] AgentSession created with turn_detection=None.")

        # Additional debug: Verify turn detection is disabled
        if hasattr(session, 'turn_detector') and session.turn_detector is not None:
            logger.warning("[AgentEntry] Turn detection is still enabled in AgentSession! Forcing disable.")
            session.turn_detector = None

        assistant_agent_instance = GUIFocusedAssistant()
        logger.info("[AgentEntry] GUIFocusedAssistant instance created.")

        logger.info("[AgentEntry] Starting AgentSession...")
        main_agent_task = asyncio.create_task(session.start(
            room=ctx.room,
            agent=assistant_agent_instance,
        ))
        logger.info("[AgentEntry] AgentSession.start() called and task created.")

        await asyncio.sleep(1)
        initial_greeting = "Hello! I'm your voice assistant, ready to chat. How can I help you today?"
        logger.info(f"[AgentEntry] Generating initial reply (agent self-initiated): '{initial_greeting}'")
        try:
            await session.generate_reply(instructions=initial_greeting)
            logger.info("[AgentEntry] Initial reply generation requested.")
        except Exception as e_greeting:
            logger.error(f"[AgentEntry] Error generating initial greeting: {e_greeting}", exc_info=True)

        while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            await asyncio.sleep(1)
            logger.debug("[AgentEntry] Room still connected, keeping session alive...")

        await main_agent_task
        logger.info("[AgentEntry] Main agent task completed.")

    except Exception as e_session:
        logger.error(f"[AgentEntry] Error during AgentSession setup or execution: {e_session}", exc_info=True)
        raise
    finally:
        logger.info(f"[AgentEntry] Agent logic finished or terminated for job {job_id_from_ctx}.")

def check_and_free_port(port):
    """Check if a port is in use and attempt to free it."""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.net_connections(kind='inet'):
                if conn.laddr.port == port:
                    logger.warning(f"Port {port} is in use by PID {proc.pid} ({proc.name()}). Attempting to terminate.")
                    proc.terminate()
                    proc.wait(timeout=5)
                    logger.info(f"Successfully terminated process on port {port}.")
                    return
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    logger.debug(f"Port {port} is free or no process could be terminated.")

def run_worker():
    try:
        logger.info("--- run_worker: Attempting to start LiveKit Agent Worker ---")
        if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
            logger.critical("!!! run_worker: CRITICAL - LiveKit connection credentials (URL, KEY, SECRET) NOT FOUND. Worker will NOT start.")
            return
        
        if not OPENAI_API_KEY_ENV: logger.warning("!!! run_worker: OpenAI API key not found. LLM/TTS plugins might fail.")
        if not DEEPGRAM_API_KEY_ENV: logger.warning("!!! run_worker: Deepgram API key not found. STT plugin might fail.")

        # Check and free port 8081 for LiveKit worker
        check_and_free_port(8081)

        worker_options = WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
        
        original_argv = list(sys.argv)
        sys.argv = ['livekit_agent_script_embedded.py', 'start']
        logger.info(f"run_worker: Modified sys.argv for cli.run_app: {sys.argv}")
        
        logger.info("run_worker: Applying nest_asyncio and calling cli.run_app(worker_options)...")
        try:
            import nest_asyncio
            nest_asyncio.apply()
            logger.info("run_worker: Applied nest_asyncio.")
            cli.run_app(worker_options)
        finally:
            sys.argv = original_argv
            logger.info(f"run_worker: Restored original sys.argv: {sys.argv}")

        logger.info("--- run_worker: LiveKit Agent Worker `cli.run_app` finished. ---")

    except SystemExit as se:
        logger.info(f"run_worker: cli.run_app exited with SystemExit (code: {se.code}). This is often normal.")
    except Exception as e:
        logger.error(f"!!! run_worker: CRITICAL ERROR during run_worker or cli.run_app: {e}", exc_info=True)
    finally:
        logger.info("--- run_worker: Function execution complete or terminated. ---")

if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    logger.info("--- __main__: main.py script execution started ---")

    if len(sys.argv) > 1 and sys.argv[1] == "download-files":
        logger.info("--- __main__: 'download-files' argument detected. ---")
        logger.info("--- __main__: Plugins will initialize when the worker starts, triggering downloads if needed.")
        pass 

    # Check and free port 8000 for Uvicorn
    check_and_free_port(8000)

    logger.info("--- __main__: Creating worker thread (LiveKitAgentWorker) ---")
    worker_thread = threading.Thread(target=run_worker, name="LiveKitAgentWorker", daemon=True)
    worker_thread.start()
    logger.info("--- __main__: Worker thread started.")

    logger.info("--- __main__: Waiting a few seconds for agent worker to initialize before starting Uvicorn...")
    time.sleep(5)
    
    if worker_thread.is_alive():
        logger.info("--- __main__: Worker thread IS ALIVE. Proceeding to start Uvicorn.")
    else:
        logger.error("--- __main__: Worker thread IS NOT ALIVE. It likely exited. Check 'run_worker' logs for errors (e.g., missing credentials).")

    logger.info("--- __main__: Starting Uvicorn server for FastAPI app ---")
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except Exception as e_uvicorn:
        logger.error(f"--- __main__: Uvicorn server failed: {e_uvicorn}", exc_info=True)
    
    logger.info("--- __main__: Uvicorn server has stopped. ---")
    logger.info("--- __main__: Script execution finished. ---")









# ///////////////////////////////////////////
















# opensource englog.py file which is working properly
# 24-05-25




# trial code for frontend logging - it is working perfectly







import asyncio
import os
import sys
import logging
import io
from datetime import timedelta
import psutil
from typing import Any, AsyncIterator

# Standard UTF-8 reconfiguration for console
try:
    if sys.stdout.encoding is None or 'utf-8' not in sys.stdout.encoding.lower():
         sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if sys.stderr.encoding is None or 'utf-8' not in sys.stderr.encoding.lower():
         sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, io.UnsupportedOperation) as e:
    logging.warning(f"Failed to reconfigure stdout/stderr encoding to UTF-8: {e}.")
except Exception as e:
     logging.error(f"Unexpected error during stdout/stderr reconfigure: {e}", exc_info=True)

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from livekit.api import AccessToken, VideoGrants
from livekit import agents, rtc # Ensure rtc is imported
from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    Agent,
    AgentSession,
    AutoSubscribe,
    ConversationItemAddedEvent, # Ensure this is imported
)
from livekit.agents.llm import LLMStream
from livekit.agents import voice
from livekit.agents import llm as livekit_llm
from livekit.agents.llm import ChatMessage, ChatContext
import pydantic_core
from livekit.plugins import deepgram, groq, elevenlabs, silero

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_logs_english_only.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
# (Formatter setup for logger and agent_logger as before) ...
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
         handler.setStream(sys.stdout)
    if hasattr(handler, 'encoding'): # Should be utf-8 from basicConfig
        handler.encoding = 'utf-8'
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

agent_logger = logging.getLogger("livekit.agents")
agent_logger.setLevel(logging.DEBUG) # Or INFO if DEBUG is too verbose
for handler_ref in logging.getLogger().handlers: # Get root handlers configured by basicConfig
    # Create a new handler for agent_logger if you want separate filtering or formatting
    # For simplicity, let agent_logger use the same handlers and formatters
    agent_logger.addHandler(handler_ref)
    # If you specifically want agent_logger to have its own formatting on existing handlers:
    # (though basicConfig applies to root, and child loggers usually propagate to root handlers)
    # if isinstance(handler_ref, logging.StreamHandler):
    #      handler_ref.setStream(sys.stdout) # Already done
    # if hasattr(handler_ref, 'encoding'):
    #     handler_ref.encoding = 'utf-8'   # Already done
    # handler_ref.setFormatter(logging.Formatter(
    #    '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    #    datefmt='%Y-%m-%d %H:%M:%S'
    # ))
agent_logger.propagate = False # Prevent double logging if handlers are added directly and also propagate


logger.info(f"MODULE LEVEL: LIVEKIT_URL: {LIVEKIT_URL}")
# ... (other env var logs)

app = FastAPI()
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/token")
async def get_token():
    # ... (get_token implementation as before, ensure room_name is consistent or managed)
    logger.info("[TokenSvc] /token endpoint hit.")
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        logger.error("[TokenSvc] LiveKit credentials not configured properly.")
        raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

    room_name = "voice-assistant-room-english"
    participant_identity = "frontend_user" # This is the identity the frontend user will use
    # The agent will have its own identity, typically auto-generated or set in WorkerOptions
    participant_name = "Frontend User"
    video_grant_obj = VideoGrants(
        room_join=True, room=room_name, can_publish=True, can_subscribe=True, can_publish_data=True,
    )
    token_builder = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token_builder.identity = participant_identity
    token_builder.name = participant_name
    token_builder.ttl = timedelta(hours=1)
    token_builder.with_grants(video_grant_obj)
    token_jwt = token_builder.to_jwt()
    logger.info(f"[TokenSvc] Generated token for identity '{participant_identity}' for room '{room_name}'")
    return {"token": token_jwt, "url": LIVEKIT_URL, "roomName": room_name}


class EnglishAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly and helpful voice assistant named Khushi. "
                "You speak and understand only English. "
                "Respond concisely and directly answer the user's query in English. "
                "Maintain context from the conversation history."
            )
        )
        self.last_user_query_for_log = None
        self.chat_history: list[ChatMessage] = [
             ChatMessage(role="system", content=[self.instructions])
        ]
        logger.info("--- [EnglishAssistant] __init__ CALLED --- Agent initialized for English only.")

    async def on_transcript(self, transcript: str, is_final: bool) -> None:
        log_level = logging.INFO if is_final else logging.DEBUG
        # logger.log(log_level, f"--- [EnglishAssistant] on_transcript --- Final: {is_final}, Text: '{transcript}'")

        # Publishing partial transcriptions for live feedback (optional)
        if self.room and self.room.local_participant and not is_final and transcript:
            try:
                payload_str_partial = f"transcription_update:{transcript}"
                # await self.room.local_participant.publish_data(payload=payload_str_partial.encode('utf-8'), topic="transcription_partial")
            except Exception as e_pub_tx:
                logger.error(f"--- [EnglishAssistant] on_transcript: Error publishing partial user transcript: {e_pub_tx}", exc_info=True)


        if is_final and transcript and transcript.strip():
            self.last_user_query_for_log = transcript.strip()
            # logger.info(f"--- [EnglishAssistant] on_transcript: Final user query (English): '{self.last_user_query_for_log}'. Generating LLM response.")

            # NOTE: We are moving the primary publishing of the *final* user query
            # to the `_on_conversation_item_added` event handler to ensure it aligns with session history.
            # If you still want to publish it from here for some reason, you can, but be mindful of duplicates.
            # Example of publishing the *final* user transcript from here (potentially redundant now):
            # if self.room and self.room.local_participant:
            #     try:
            #         payload_str_final = f"transcription:{self.last_user_query_for_log}"
            #         await self.room.local_participant.publish_data(payload=payload_str_final.encode('utf-8'), topic="transcription")
            #     except Exception as e_pub_tx:
            #         logger.error(f"--- [EnglishAssistant] on_transcript: Error publishing final user transcript: {e_pub_tx}", exc_info=True)


            if not hasattr(self, 'llm') or not self.llm:
                logger.error("--- [EnglishAssistant] on_transcript: self.llm is not available.")
                await self._send_error_response("Sorry, my brain is not available right now.")
                return

            try:
                user_chat_message = ChatMessage(role="user", content=[self.last_user_query_for_log])
                current_chat_turn_history = list(self.chat_history)
                current_chat_turn_history.append(user_chat_message)

                max_history_length = 21
                if len(current_chat_turn_history) > max_history_length:
                     current_chat_turn_history = [current_chat_turn_history[0]] + current_chat_turn_history[-(max_history_length - 1):]

                chat_ctx_for_llm = ChatContext(current_chat_turn_history)
                # logger.debug(f"--- [EnglishAssistant] on_transcript: Calling self.llm.chat(...) with history (count: {len(current_chat_turn_history)}).")
                llm_stream = self.llm.chat(chat_ctx=chat_ctx_for_llm)

                if not isinstance(llm_stream, AsyncIterator):
                    logger.error(f"--- [EnglishAssistant] on_transcript: llm.chat did NOT return an async iterator. Type: {type(llm_stream)}.")
                    await self._send_error_response("Sorry, I received an invalid response from the LLM.")
                    return

                # logger.info(f"--- [EnglishAssistant] on_transcript: self.llm.chat() returned stream. Calling self.handle_llm_response.")
                self.chat_history = current_chat_turn_history # Update history *before* processing response
                await self.handle_llm_response(llm_stream)

            except Exception as e_llm_interaction:
                logger.error(f"--- [EnglishAssistant] on_transcript: Uncaught error in LLM interaction: {e_llm_interaction}", exc_info=True)
                await self._send_error_response("Sorry, I encountered an error generating a response.")

    async def _send_error_response(self, error_msg: str):
        # (This function remains the same)
        logger.warning(f"--- [EnglishAssistant] Sending error response: {error_msg}")
        if self.room and self.room.local_participant:
             try:
                 # Publishing error messages to the frontend using the "response" topic
                 await self.room.local_participant.publish_data(payload=f"response:Error: {error_msg}".encode('utf-8'), topic="response")
                 if hasattr(self, 'tts') and self.tts:
                     async def error_gen(): yield error_msg
                     await self.tts.synthesize(error_gen())
             except Exception as e_pub_err:
                 logger.error(f"--- [EnglishAssistant] Failed to publish error message: {e_pub_err}", exc_info=True)

    async def handle_llm_response(self, llm_stream: LLMStream) -> None:
        # logger.info(f"--- [EnglishAssistant] handle_llm_response CALLED --- Processing English stream.")
        collected_text_for_frontend = ""
        temp_sentences_for_tts = []
        llm_stream_finished_successfully = False

        try:
            sentence_count = 0
            async for sentence_obj in self.llm_stream_sentences(llm_stream):
                sentence_count += 1
                if sentence_obj.text:
                    collected_text_for_frontend += sentence_obj.text + " "
                    temp_sentences_for_tts.append(sentence_obj.text)

                    # Publishing partial agent responses (optional)
                    if self.room and self.room.local_participant and sentence_obj.text.strip():
                         try:
                             partial_payload = f"response_partial:{sentence_obj.text.strip()}"
                             # await self.room.local_participant.publish_data(payload=partial_payload.encode('utf-8'), topic="response_partial")
                         except Exception as e_pub_partial:
                             logger.error(f"--- [EnglishAssistant] handle_llm_response: Error publishing partial English response: {e_pub_partial}", exc_info=True)

            llm_stream_finished_successfully = True
            # logger.info(f"--- [EnglishAssistant] handle_llm_response: English LLM stream processed. Sentences: {sentence_count}.")

        except Exception as e_llm_stream_processing:
            logger.error(f"--- [EnglishAssistant] handle_llm_response: Error processing English LLM stream: {e_llm_stream_processing}", exc_info=True)
            collected_text_for_frontend = "Error generating English response." if not collected_text_for_frontend.strip() else collected_text_for_frontend + " (Stream error)"
        finally:
            final_collected_text = collected_text_for_frontend.strip()
            log_context_info = f"for user query '{self.last_user_query_for_log}'" if self.last_user_query_for_log else "(agent-initiated)"

            if final_collected_text:
                # logger.info(f"--- [EnglishAssistant] handle_llm_response: FINAL ENGLISH MODEL_RESPONSE {log_context_info}: '{final_collected_text}'")

                # NOTE: We are moving the primary publishing of the *final* agent response
                # to the `_on_conversation_item_added` event handler.
                # If you still want to publish it from here:
                # if self.room and self.room.local_participant:
                #     try:
                #         await self.room.local_participant.publish_data(payload=f"response:{final_collected_text}".encode('utf-8'), topic="response")
                #     except Exception as e_pub_resp:
                #         logger.error(f"--- [EnglishAssistant] handle_llm_response: Error publishing final English response: {e_pub_resp}", exc_info=True)

                # Add to internal agent history (this agent instance's history)
                # This will eventually trigger conversation_item_added if the session's agent is this instance
                if self.chat_history and self.chat_history[-1].role == "assistant":
                     last_msg = self.chat_history[-1]
                     if not isinstance(last_msg.content, list): last_msg.content = [str(last_msg.content)] if last_msg.content is not None else [""]
                     last_msg.content[0] = (last_msg.content[0].strip() + " " + final_collected_text).strip()
                else: # This is usually how it works if session.say or generate_reply is used implicitly
                    self.chat_history.append(ChatMessage(role="assistant", content=[final_collected_text]))

                max_history_length = 21
                if len(self.chat_history) > max_history_length:
                     self.chat_history = [self.chat_history[0]] + self.chat_history[-(max_history_length - 1):]

            elif llm_stream_finished_successfully:
                logger.warning(f"--- [EnglishAssistant] handle_llm_response: LLM stream finished, no English text collected {log_context_info}.")
                await self._send_error_response("Sorry, I couldn't generate an English response.")
            else: # Stream processing error and no text collected
                logger.error(f"--- [EnglishAssistant] handle_llm_response: LLM stream failed for English text {log_context_info}.")
                # Error already sent if collected_text_for_frontend was empty from the except block

        if temp_sentences_for_tts and any(s.strip() for s in temp_sentences_for_tts) and hasattr(self, 'tts') and self.tts:
            # logger.info(f"--- [EnglishAssistant] handle_llm_response: Attempting English TTS with {len(temp_sentences_for_tts)} chunks.")
            async def gen_tts_stream():
                for s in temp_sentences_for_tts:
                    if s.strip(): yield s.strip()
            try:
                 await self.tts.synthesize(gen_tts_stream())
                 # logger.info("--- [EnglishAssistant] English TTS synthesis call completed.")
            except Exception as e_tts:
                logger.error(f"--- [EnglishAssistant] handle_llm_response: Error during English TTS synthesis: {e_tts}", exc_info=True)

        self.last_user_query_for_log = None # Reset after processing
        # logger.info(f"--- [EnglishAssistant] handle_llm_response FINISHED (English) ---")


async def entrypoint(ctx: JobContext):
    logger.info(f"--- [AgentEntry] AGENT ENTRYPOINT (English Only) --- Room: {ctx.room.name}")

    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"[AgentEntry] Connected to room '{ctx.room.name}'. Local SID: {ctx.room.local_participant.sid}")
    except Exception as e_connect:
        logger.critical(f"!!! [AgentEntry] CRITICAL - Agent failed to connect: {e_connect}", exc_info=True)
        sys.exit(f"Fatal error during ctx.connect(): {e_connect}")

    # ... (Plugin initialization and LLM test as before - ENSURE KEYS ARE SET) ...
    stt_plugin, llm_plugin, tts_plugin, vad_plugin = None, None, None, None
    # ... (rest of plugin init) ...
    try:
        stt_plugin = deepgram.STT(api_key=DEEPGRAM_API_KEY, language="en", model="nova-2-general")
        llm_plugin = groq.LLM(model="llama3-70b-8192", api_key=GROQ_API_KEY)
        tts_plugin = elevenlabs.TTS(api_key=ELEVENLABS_API_KEY)
        vad_plugin = silero.VAD.load()
        logger.info("[AgentEntry] All plugins initialized successfully for English operation.")
    except Exception as e_plugins:
        logger.critical(f"!!! [AgentEntry] CRITICAL - Error initializing plugins: {e_plugins}", exc_info=True)
        sys.exit(f"Error initializing plugins: {e_plugins}")


    session = AgentSession(stt=stt_plugin, llm=llm_plugin, tts=tts_plugin, vad=vad_plugin)
    logger.info("[AgentEntry] AgentSession created for English operation.")

    # -------- CORRECTED EVENT HANDLER REGISTRATION --------
    async def _publish_conversation_item_async(event_item: ChatMessage):
        """Helper async function to do the actual publishing"""
        item_role_str = "unknown"
        item_text_content = ""

        if event_item: # Check if event_item (which is event.item) is not None
            if event_item.role:
                item_role_str = str(event_item.role.value) if hasattr(event_item.role, 'value') else str(event_item.role)
            
            if hasattr(event_item, 'text_content') and event_item.text_content:
                item_text_content = event_item.text_content
            elif hasattr(event_item, 'content'):
                if isinstance(event_item.content, list):
                    str_parts = [str(part) for part in event_item.content if isinstance(part, str)]
                    item_text_content = " ".join(str_parts).strip()
                elif isinstance(event_item.content, str):
                    item_text_content = event_item.content
        
        # Publish to frontend using a new topic
        if ctx.room and ctx.room.local_participant and item_text_content:
            payload_to_send = ""
            topic_to_send = "lk_chat_history"

            if item_role_str.lower() == "user":
                payload_to_send = f"user_msg:{item_text_content}"
            elif item_role_str.lower() == "assistant":
                payload_to_send = f"agent_msg:{item_text_content}"
                logger.info(f"--- AGENT'S RESPONSE (via event, to be published) --- '{item_text_content}'")

            if payload_to_send:
                logger.info(f"[AgentEntry] Publishing to frontend via '{topic_to_send}': '{payload_to_send}'")
                try:
                    await ctx.room.local_participant.publish_data(
                        payload=payload_to_send.encode('utf-8'),
                        topic=topic_to_send
                    )
                except Exception as e_pub:
                    logger.error(f"[AgentEntry] Error publishing from _publish_conversation_item_async: {e_pub}", exc_info=True)

    # This is the synchronous callback registered with .on()
    def _on_conversation_item_added_sync(event: ConversationItemAddedEvent):
        item_role_str = "unknown"
        item_text_content = "" # Default to empty string

        if event.item: # Ensure event.item exists
            if event.item.role:
                # Handle if role is an enum (like ChatMessageRole.USER) or just a string
                item_role_str = str(event.item.role.value) if hasattr(event.item.role, 'value') else str(event.item.role)
            
            # Get text content robustly
            if hasattr(event.item, 'text_content') and event.item.text_content:
                item_text_content = event.item.text_content
            elif hasattr(event.item, 'content'): # Fallback for content list
                if isinstance(event.item.content, list):
                    str_parts = [str(part) for part in event.item.content if isinstance(part, str)] # Ensure parts are strings
                    item_text_content = " ".join(str_parts).strip()
                elif isinstance(event.item.content, str): # If content itself is a string
                    item_text_content = event.item.content
            
        # Original backend logging (this part is synchronous)
        logger.info(f"[AgentSession Event] Item Added (Sync Handler): Role='{item_role_str}', Text='{item_text_content}'")

        # Create a task to run the async publishing logic
        if event.item: # Only try to publish if there's an item
            asyncio.create_task(_publish_conversation_item_async(event.item))

    session.on("conversation_item_added", _on_conversation_item_added_sync)
    # -------- END CORRECTED EVENT HANDLER REGISTRATION --------

    assistant_agent_instance = EnglishAssistant() # Or your relevant assistant class
    logger.info("[AgentEntry] EnglishAssistant instance created.")

    logger.info("[AgentEntry] Starting AgentSession with EnglishAssistant...")
    try:
        await session.start(room=ctx.room, agent=assistant_agent_instance)
        logger.info("[AgentEntry] AgentSession.start() completed.")
    except Exception as e_session_start:
        logger.error(f"[AgentEntry] Error in session.start(): {e_session_start}", exc_info=True)
        raise
    finally:
        logger.info("[AgentEntry] AgentSession.start() block exited.")
    logger.info(f"[AgentEntry] Agent logic finished for job {ctx.job.id if ctx.job else 'N/A'}.")


# (check_and_free_port function as before)
def check_and_free_port(port):
    logger.debug(f"Checking if port {port} is in use...")
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.net_connections(kind='inet'):
                    if conn.laddr.port == port:
                        logger.warning(f"Port {port} in use by PID {proc.pid} ({proc.name()}). Terminating.")
                        try:
                            proc.terminate()
                            proc.wait(timeout=3)
                            if proc.is_running():
                                logger.warning(f"Process {proc.pid} did not terminate gracefully. Killing.")
                                proc.kill()
                                proc.wait(timeout=3)
                            logger.info(f"Handled process on port {port} (PID {proc.pid}).")
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as term_err:
                             logger.error(f"Failed to terminate PID {proc.pid} on port {port}: {term_err}")
                        except Exception as e:
                            logger.error(f"Unexpected error handling PID {proc.pid} on port {port}: {e}", exc_info=True)
                        return # Addressed the port, exit check
            except (psutil.NoSuchProcess, psutil.AccessDenied): continue
            except Exception as e: logger.error(f"Error iterating conns for PID {proc.pid}: {e}", exc_info=True)
    except Exception as e: logger.error(f"Error iterating procs for port {port}: {e}", exc_info=True)
    logger.debug(f"Port {port} appears free.")


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    logger.info("--- __main__: Script execution started (English Only Mode) ---")
    SCRIPT_NAME_FOR_UVICORN = "main" # CHANGE IF YOUR FILENAME IS DIFFERENT (e.g., "english_assistant")

    if len(sys.argv) > 1 and sys.argv[1] in ["start", "download-files", "package"]:
        logger.info(f"--- __main__: Running Agent Worker mode (English Only). ---")
        try:
            worker_options = WorkerOptions(entrypoint_fnc=entrypoint) # Ensure this points to your entrypoint
            cli.run_app(worker_options)
        except SystemExit as se:
             logger.info(f"--- __main__: cli.run_app exited with SystemExit (code: {se.code}).")
        except Exception as e:
            logger.critical(f"!!! __main__: CRITICAL ERROR in Agent Worker: {e}", exc_info=True)
            sys.exit(f"Agent Worker failed: {e}")
        logger.info("--- __main__: Script finished (Agent Worker mode). ---")
    else:
        logger.info("--- __main__: Starting FastAPI server (Token Server mode). ---")
        check_and_free_port(8000)
        try:
            uvicorn.run(
                f"{SCRIPT_NAME_FOR_UVICORN}:app",
                host="0.0.0.0", port=8000, reload=False, log_level="info",
            )
        except Exception as e_uvicorn:
            logger.critical(f"!!! __main__: CRITICAL - Uvicorn server failed: {e_uvicorn}", exc_info=True)
            sys.exit(f"FastAPI server failed: {e_uvicorn}")
        logger.info("--- __main__: Uvicorn server stopped. Script finished (Token Server mode). ---")