
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
from livekit.agents import llm as livekit_llm 
from livekit.agents.llm import ChatMessage, ChatContext
import pydantic_core

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
                "You are Echo, a friendly and helpful voice assistant. Respond concisely. "
                "When the user speaks, understand their query and provide a relevant answer."
            )
        )
        self.last_user_query_for_log = None 
        self.chat_history: list[ChatMessage] = [] 
        logger.info("--- [GUIFocusedAssistant] __init__ CALLED ---")

    async def on_transcript(self, transcript: str, is_final: bool) -> None:
        logger.info(f"--- [GUIFocusedAssistant] on_transcript CALLED --- Final: {is_final}, Text: '{transcript}'")
        if is_final and transcript:
            self.last_user_query_for_log = transcript
            
            logger.info(f"--- [GUIFocusedAssistant] on_transcript: User query received: '{transcript}'. Attempting to generate LLM response.")
            
            if not hasattr(self, 'llm') or not self.llm:
                logger.error("--- [GUIFocusedAssistant] on_transcript: self.llm is not available. Cannot generate LLM response.")
                return

            try:
                # Create ChatMessage for current user transcript
                # Content should be a list of strings for this SDK version based on previous Pydantic errors
                user_chat_message = ChatMessage(role="user", content=[transcript]) 
                
                current_chat_turn_history = list(self.chat_history)
                current_chat_turn_history.append(user_chat_message)

                if len(current_chat_turn_history) > 20: 
                    current_chat_turn_history = current_chat_turn_history[-20:]

                # Create ChatContext
                chat_ctx_for_llm = ChatContext(messages=current_chat_turn_history)

                logger.info(f"--- [GUIFocusedAssistant] on_transcript: Calling self.llm.chat(chat_ctx=...) with history (count: {len(current_chat_turn_history)})")
                
                llm_stream = await self.llm.chat(chat_ctx=chat_ctx_for_llm) # Use chat_ctx=

                logger.info(f"--- [GUIFocusedAssistant] on_transcript: self.llm.chat() returned stream. Now calling self.handle_llm_response.")
                self.chat_history = current_chat_turn_history 
                await self.handle_llm_response(llm_stream)

            except TypeError as te_chat: 
                 logger.error(f"--- [GUIFocusedAssistant] on_transcript: TypeError calling self.llm.chat(chat_ctx=...): {te_chat}. 'chat_ctx' or ChatMessage format still incorrect.", exc_info=True)
            except Exception as e_llm_call_or_handle:
                 logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error in LLM call sequence or in handle_llm_response: {e_llm_call_or_handle}", exc_info=True)
        
        if is_final and transcript: # Publish transcript to frontend
            if self.room and self.room.local_participant:
                logger.info(f"--- [GUIFocusedAssistant] on_transcript: Publishing FINAL user transcript to frontend: '{transcript}'")
                try:
                    await self.room.local_participant.publish_data(f"transcription:{transcript}", "transcription")
                except Exception as e_pub_tx:
                    logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error publishing user transcript: {e_pub_tx}", exc_info=True)

    async def handle_llm_response(self, llm_stream: LLMStream) -> None:
        logger.info(f"--- [GUIFocusedAssistant] handle_llm_response CALLED ---")
        collected_text_for_frontend = ""
        temp_sentences_for_tts = [] 
        stream_processed_successfully = False

        try:
            async for sentence_obj in self.llm_stream_sentences(llm_stream):
                logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Processing sentence: '{sentence_obj.text}'")
                if sentence_obj.text:
                    collected_text_for_frontend += sentence_obj.text + " "
                    temp_sentences_for_tts.append(sentence_obj.text)
                stream_processed_successfully = True
        except Exception as e_llm_stream_processing:
            logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error processing LLM stream: {e_llm_stream_processing}", exc_info=True)
            if not collected_text_for_frontend: collected_text_for_frontend = "Error processing response."
            else: collected_text_for_frontend += " (Error in stream)"
        
        final_collected_text = collected_text_for_frontend.strip()

        if final_collected_text: 
            # Content should be a list of strings
            assistant_chat_message = ChatMessage(role="assistant", content=[final_collected_text])
            self.chat_history.append(assistant_chat_message)
            if len(self.chat_history) > 20: self.chat_history = self.chat_history[-20:]

        log_context_info = f"for user query '{self.last_user_query_for_log}'" if self.last_user_query_for_log else "(initial/agent-initiated)"
        if final_collected_text: logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: FINAL MODEL_RESPONSE {log_context_info}: '{final_collected_text}'")
        else: logger.warning(f"--- [GUIFocusedAssistant] handle_llm_response: No final text from LLM {log_context_info}.")

        if final_collected_text and self.room and self.room.local_participant:
             logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Publishing agent response to GUI: '{final_collected_text}'")
             await self.room.local_participant.publish_data(payload=f"response:{final_collected_text}", topic="response")

        if temp_sentences_for_tts and any(s.strip() for s in temp_sentences_for_tts) and hasattr(self, 'tts') and self.tts:
            logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Attempting TTS.")
            async def gen_tts():
                for s in temp_sentences_for_tts: yield s.strip()
            await self.tts.synthesize(gen_tts())
        
        self.last_user_query_for_log = None 
        logger.info(f"--- [GUIFocusedAssistant] handle_llm_response FINISHED ---")


# --- Agent Entrypoint ---
async def entrypoint(ctx: JobContext):
    logger.info(f"--- [AgentEntry] AGENT LOGIC ENTRYPOINT CALLED (ChatContext Deep Dive) --- Room: {ctx.room.name}, Job: {ctx.job.id}")

    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"[AgentEntry] Connected to room. Local Participant SID: {ctx.room.local_participant.sid}")
    except Exception as e_connect:
        logger.error(f"[AgentEntry] Fatal error during ctx.connect(): {e_connect}", exc_info=True); raise

    logger.info("[AgentEntry] Initializing plugins...")
    llm_plugin = None
    try:
        stt = deepgram.STT(model="nova-2", language="multi", interim_results=False)
        if not OPENAI_API_KEY_ENV: raise ValueError("OpenAI API key not found for plugins.")
        llm_plugin = openai.LLM(model="gpt-4o-mini", api_key=OPENAI_API_KEY_ENV) 
        tts_plugin = openai.TTS(voice="ash", api_key=OPENAI_API_KEY_ENV)
        vad = silero.VAD.load() 
        logger.info("[AgentEntry] All plugins initialized successfully.")
    except Exception as e_plugins:
        logger.error(f"[AgentEntry] Error initializing plugins: {e_plugins}", exc_info=True); raise 

    # --- LLM PLUGIN DIRECT TEST ---
    logger.info("[AgentEntry] --- Testing LLM plugin directly START ---")
    llm_test_passed = False
    if not llm_plugin:
        logger.error("[AgentEntry] LLM Plugin is None, cannot run direct test.")
    else:
        test_prompt_text = "Hello, LLM. Confirm you are working by saying OK."
        
        # Step 1: Create a list of ChatMessage objects
        # Based on previous logs, ChatMessage(role="user", content=[string]) is valid for ChatMessage creation.
        chat_message_list: list[ChatMessage] = []
        try:
            chat_message_list = [ChatMessage(role="user", content=[test_prompt_text])]
            logger.info(f"[AgentEntry] LLM Direct Test: Successfully created chat_message_list: {chat_message_list}")
        except Exception as e_cm_create:
            logger.error(f"[AgentEntry] LLM Direct Test: FAILED to create ChatMessage list: {e_cm_create}", exc_info=True)
            chat_message_list = [] # Ensure it's an empty list if creation failed

        if chat_message_list:
            # Step 2: Try to create ChatContext
            chat_ctx_for_test: ChatContext | None = None
            try:
                logger.info(f"[AgentEntry] LLM Direct Test: Attempting ChatContext(chat_message_list) [POSITIONAL for messages]")
                chat_ctx_for_test = ChatContext(chat_message_list) # TRY POSITIONAL for messages list
                logger.info(f"[AgentEntry] LLM Direct Test: ChatContext created positionally: {chat_ctx_for_test}")
            except TypeError as te_ctx_pos:
                logger.warning(f"[AgentEntry] LLM Direct Test: ChatContext(positional_list) FAILED (TypeError: {te_ctx_pos}). Trying ChatContext().")
                try:
                    chat_ctx_for_test = ChatContext() # Try empty constructor
                    logger.info(f"[AgentEntry] LLM Direct Test: ChatContext() created empty. Appending messages.")
                    # If ChatContext has an 'messages' attribute that's a list, or an 'add_message' method
                    if hasattr(chat_ctx_for_test, 'messages') and isinstance(chat_ctx_for_test.messages, list):
                        for msg in chat_message_list:
                            chat_ctx_for_test.messages.append(msg)
                        logger.info(f"[AgentEntry] LLM Direct Test: Messages appended to chat_ctx_for_test.messages")
                    elif hasattr(chat_ctx_for_test, 'add_message'):
                         for msg in chat_message_list:
                            chat_ctx_for_test.add_message(msg) # Assuming an add_message method
                         logger.info(f"[AgentEntry] LLM Direct Test: Messages added via chat_ctx_for_test.add_message()")
                    else:
                        logger.warning("[AgentEntry] LLM Direct Test: Empty ChatContext created, but no obvious way to add messages to it.")
                        chat_ctx_for_test = None # Mark as unusable
                except Exception as e_ctx_empty:
                    logger.error(f"[AgentEntry] LLM Direct Test: Creating empty ChatContext or adding messages FAILED: {e_ctx_empty}", exc_info=True)
                    chat_ctx_for_test = None
            except Exception as e_ctx_other:
                logger.error(f"[AgentEntry] LLM Direct Test: Other error creating ChatContext: {e_ctx_other}", exc_info=True)
                chat_ctx_for_test = None

            # Step 3: If ChatContext was created, try calling llm_plugin.chat()
            if chat_ctx_for_test:
                try:
                    logger.info(f"[AgentEntry] LLM Direct Test: Attempting llm_plugin.chat(chat_ctx=chat_ctx_for_test)")
                    async for chunk in await llm_plugin.chat(chat_ctx=chat_ctx_for_test): 
                        logger.info(f"[AgentEntry] LLM Direct Test Chunk (with chat_ctx=): '{getattr(chunk, 'text', str(chunk))}'")
                        llm_test_passed = True; break 
                    if llm_test_passed: logger.info("[AgentEntry] LLM plugin direct test with chat_ctx= SUCCEEDED.")
                    else: logger.warning("[AgentEntry] LLM plugin direct test with chat_ctx=: no chunks.")
                except Exception as e_llm_call:
                    logger.error(f"[AgentEntry] LLM plugin direct test with chat_ctx= FAILED: {e_llm_call}", exc_info=True)
            else:
                logger.warning("[AgentEntry] LLM Direct Test: chat_ctx_for_test is None, skipping llm.chat() call.")
        else:
            logger.warning("[AgentEntry] LLM Direct Test: chat_message_list is empty, cannot proceed to create ChatContext.")
            
    if not llm_test_passed: logger.critical("[AgentEntry] !!! LLM direct test FAILED. Agent may not function. !!!")
    logger.info("[AgentEntry] --- Testing LLM plugin directly END ---")
    # --- END OF LLM PLUGIN DIRECT TEST ---

    session = AgentSession(stt=stt, llm=llm_plugin, tts=tts_plugin, vad=vad)
    logger.info("[AgentEntry] AgentSession created.")
    assistant_agent_instance = GUIFocusedAssistant() # Your class
    logger.info("[AgentEntry] GUIFocusedAssistant instance created.")
    logger.info("[AgentEntry] Starting AgentSession with GUIFocusedAssistant...")
    main_agent_task = asyncio.create_task(session.start(room=ctx.room, agent=assistant_agent_instance))
    logger.info("[AgentEntry] AgentSession.start() called. Session will manage greeting and turns.")

    logger.info("[AgentEntry] Main agent logic running. Waiting for AgentSession task to complete.")
    try:
        await main_agent_task 
    except asyncio.CancelledError: logger.info("[AgentEntry] Main agent_task was cancelled.")
    except Exception as e_main_task: logger.error(f"[AgentEntry] Main agent_task exited with error: {e_main_task}", exc_info=True)
    logger.info(f"[AgentEntry] Agent logic finished/terminated for job {ctx.job.id if ctx.job else 'N/A'}.")


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



