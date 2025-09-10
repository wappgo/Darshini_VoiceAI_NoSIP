



import asyncio
import os
# Moved load_dotenv to the very top
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from datetime import timedelta
import threading # Kept for check_and_free_port, but the main worker won't use it
import time
import sys
import psutil
from typing import Any, AsyncIterator # Import AsyncIterator

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
from livekit.plugins import openai
from livekit.plugins import silero

# --- Load environment variables and set up module-level variables immediately ---
load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY_ENV = os.getenv("ELEVENLABS_API_KEY")
DEEPGRAM_API_KEY_ENV = os.getenv("DEEPGRAM_API_KEY")

# --- Configure logging with console and file output ---
logging.basicConfig(
    level=logging.DEBUG, # Keep DEBUG to see detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_logs.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
agent_logger = logging.getLogger("livekit.agents")
agent_logger.setLevel(logging.DEBUG)

# Now log the module-level variables AFTER they have been defined
logger.info(f"MODULE LEVEL: LIVEKIT_URL: {LIVEKIT_URL}")
logger.info(f"MODULE LEVEL: LIVEKIT_API_KEY IS SET: {bool(LIVEKIT_API_KEY)}")
logger.info(f"MODULE LEVEL: DEEPGRAM_API_KEY IS SET: {bool(DEEPGRAM_API_KEY_ENV)}")
logger.info(f"MODULE LEVEL: OPENAI_API_KEY IS SET: {bool(OPENAI_API_KEY_ENV)}")


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
    token_builder.identity = "frontend_user" # This identity joins the room from the frontend
    token_builder.name = "Frontend User"
    token_builder.ttl = timedelta(hours=1)
    token_builder.with_grants(video_grant_obj)
    token_jwt = token_builder.to_jwt()

    logger.info(f"[TokenSvc] Generated token for frontend_user to join 'voice-assistant-room'")
    # Return both the token and the LiveKit URL for the frontend to connect
    return {"token": token_jwt, "url": LIVEKIT_URL}

# --- Define the Assistant Agent ---
class GUIFocusedAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Echo, a friendly and helpful voice assistant. Respond concisely. "
                "When the user speaks, understand their query and provide a relevant answer. "
                "Your primary language is Hindi and English, but you can understand multiple languages. "
                "If the user speaks in Hindi or a mix of Hindi, please respond in Hindi using Devanagari script. "
                "Do not use Urdu script or characters. Use only Devanagari for Hindi."
            )
        )
        self.last_user_query_for_log = None
        self.chat_history: list[ChatMessage] = []
        logger.info("--- [GUIFocusedAssistant] __init__ CALLED ---")

    async def on_transcript(self, transcript: str, is_final: bool) -> None:
        # This print statement is crucial for debugging raw STT output
        # With separate processes, this will appear in the terminal running the agent
        print(f"--- DEBUG RAW USER TRANSCRIPT --- Final: {is_final}, Text: '{transcript}'")

        # This logger.info logs the transcript using your configured logger (file + console)
        # This is where you should see readable Hindi if your terminal/file encoding is correct
        logger.info(f"--- [GUIFocusedAssistant] on_transcript CALLED --- Final: {is_final}, Text: '{transcript}'")
        logger.info(f"--- [DEBUG STT OUTPUT] Received: '{transcript}'")

        if is_final and transcript and transcript.strip():
            self.last_user_query_for_log = transcript.strip()

            logger.info(f"--- [GUIFocusedAssistant] on_transcript: User query received: '{self.last_user_query_for_log}'. Attempting to generate LLM response.")

            # LLM should have been initialized and passed via AgentSession
            if not hasattr(self, 'llm') or not self.llm:
                logger.error("--- [GUIFocusedAssistant] on_transcript: self.llm is not available. Cannot generate LLM response.")
                return

            try:
                user_chat_message = ChatMessage(role="user", content=[self.last_user_query_for_log])

                current_chat_turn_history = list(self.chat_history)
                current_chat_turn_history.append(user_chat_message)

                # Keep history length manageable
                if len(current_chat_turn_history) > 20:
                    current_chat_turn_history = current_chat_turn_history[-20:]

                # Pass messages list positionally as determined necessary by previous logs
                try:
                     chat_ctx_for_llm = ChatContext(current_chat_turn_history)
                     logger.debug("--- [GUIFocusedAssistant] on_transcript: Created ChatContext successfully (positional).")
                except TypeError as e_ctx_init:
                     logger.error(f"--- [GUIFocusedAssistant] on_transcript: Failed to create ChatContext positionally: {e_ctx_init}", exc_info=True)
                     return # Cannot proceed without valid context

                logger.info(f"--- [GUIFocusedAssistant] on_transcript: Calling self.llm.chat(...) with history (count: {len(current_chat_turn_history)})")

                # Call chat - Based on latest logs, trying without await here
                # If this fails with 'coroutine was never awaited', the problem is deeper.
                llm_stream = self.llm.chat(chat_ctx=chat_ctx_for_llm)

                # Validate if the result is actually an async iterator as expected
                logger.debug(f"--- [on_transcript] Type of llm_stream: {type(llm_stream)}")
                if not isinstance(llm_stream, AsyncIterator):
                    logger.error(f"--- [GUIFocusedAssistant] on_transcript: llm.chat did NOT return an async iterator. Type: {type(llm_stream)}. Cannot process response.")
                    # Log the object itself for inspection if possible, might be a coroutine
                    try: logger.error(f"--- [GUIFocusedAssistant] on_transcript: Received object from llm.chat: {llm_stream}")
                    except Exception: pass # Avoid errors logging the error object
                    return # Cannot process if it's not an async iterator

                logger.info(f"--- [GUIFocusedAssistant] on_transcript: self.llm.chat() returned stream (AsyncIterator). Now calling self.handle_llm_response.")
                # Update chat history ONLY if LLM call successfully returns a stream
                self.chat_history = current_chat_turn_history
                await self.handle_llm_response(llm_stream)

            except Exception as e_llm_call_or_handle:
                 # Catch generic exception for the entire LLM interaction flow
                 logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error in LLM interaction sequence: {e_llm_call_or_handle}", exc_info=True)

        if is_final:
            # Always publish final transcript to frontend, even if it's just whitespace
            # The frontend can decide how to display it.
            if self.room and self.room.local_participant:
                logger.info(f"--- [GUIFocusedAssistant] on_transcript: Publishing FINAL user transcript to frontend: '{transcript}'")
                try:
                    payload_str = f"transcription:{transcript}"
                    await self.room.local_participant.publish_data(payload=payload_str.encode('utf-8'), topic="transcription")
                except Exception as e_pub_tx:
                    logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error publishing user transcript: {e_pub_tx}", exc_info=True)

    async def handle_llm_response(self, llm_stream: LLMStream) -> None:
        logger.info(f"--- [GUIFocusedAssistant] handle_llm_response CALLED ---")
        collected_text_for_frontend = ""
        temp_sentences_for_tts = []
        llm_stream_finished_successfully = False

        # Ensure the stream is awaitable before entering async for loop
        # Although self.llm_stream_sentences is supposed to handle this,
        # let's be explicit based on the unexpected behavior.
        # If llm_stream is not an AsyncIterator, the check in on_transcript should prevent reaching here.

        try:
            # Using the llm_stream_sentences helper from the base Agent class
            # This asynchronously iterates over the LLM stream chunks
            async for sentence_obj in self.llm_stream_sentences(llm_stream):
                # --- DEBUG: Log each chunk/sentence from the LLM stream ---
                logger.debug(f"--- DEBUG LLM STREAM CHUNK --- Text: '{sentence_obj.text}'")
                # --- END DEBUG ---

                if sentence_obj.text:
                    collected_text_for_frontend += sentence_obj.text + " "
                    temp_sentences_for_tts.append(sentence_obj.text)
            llm_stream_finished_successfully = True # Set flag if loop completes without error
        except Exception as e_llm_stream_processing:
            logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error processing LLM stream: {e_llm_stream_processing}", exc_info=True)
            # Add a placeholder if stream processing fails and no text was collected
            if not collected_text_for_frontend.strip():
                 collected_text_for_frontend = "Error generating response."
            else:
                 collected_text_for_frontend += " (Stream processing error)"
        finally:
             # Log collected text regardless of processing success/error
             final_collected_text = collected_text_for_frontend.strip()
             log_context_info = f"for user query '{self.last_user_query_for_log}'" if self.last_user_query_for_log else "(initial/agent-initiated)"

             if final_collected_text:
                 # --- INFO: Log the final combined response ---
                 # This log should appear in the agent's terminal/log file
                 logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: FINAL MODEL_RESPONSE {log_context_info}: '{final_collected_text}'")
                 # --- END INFO ---

                 # Add to chat history only if collected text is non-empty
                 assistant_chat_message = ChatMessage(role="assistant", content=[final_collected_text])
                 self.chat_history.append(assistant_chat_message)
                 # Keep history length manageable
                 if len(self.chat_history) > 20: self.chat_history = self.chat_history[-20:]

                 # Publish response to frontend
                 if self.room and self.room.local_participant:
                      logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Publishing agent response to GUI: '{final_collected_text}'")
                      try:
                         payload_str = f"response:{final_collected_text}"
                         await self.room.local_participant.publish_data(payload=payload_str.encode('utf-8'), topic="response")
                      except Exception as e_pub_resp:
                         logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error publishing agent response: {e_pub_resp}", exc_info=True)
             elif llm_stream_finished_successfully:
                  # Log a warning if the stream completed but yielded no text
                  logger.warning(f"--- [GUIFocusedAssistant] handle_llm_response: LLM stream finished, but no text was collected {log_context_info}.")
             # If stream processing had an error and no text collected, the error is already logged above


        if temp_sentences_for_tts and any(s.strip() for s in temp_sentences_for_tts) and hasattr(self, 'tts') and self.tts:
            logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Attempting TTS with {len(temp_sentences_for_tts)} sentences.")
            async def gen_tts():
                for s in temp_sentences_for_tts:
                    if s.strip(): # Ensure we don't synthesize empty strings
                       yield s.strip()
            try:
                # Check if self.tts.synthesize is an async function and await it
                if asyncio.iscoroutinefunction(self.tts.synthesize):
                     await self.tts.synthesize(gen_tts())
                else:
                     # If it's not a coroutine function, maybe it's an async iterator method itself?
                     # This depends heavily on the plugin implementation. Default assumption is async def.
                     logger.warning("self.tts.synthesize is not a coroutine function. Attempting direct call.")
                     self.tts.synthesize(gen_tts()) # Try calling directly, might block or fail

            except Exception as e_tts:
                logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error during TTS synthesis: {e_tts}", exc_info=True)

        self.last_user_query_for_log = None
        logger.info(f"--- [GUIFocusedAssistant] handle_llm_response FINISHED ---")


# --- Agent Entrypoint ---
async def entrypoint(ctx: JobContext):
    logger.info(f"--- [AgentEntry] AGENT LOGIC ENTRYPOINT CALLED --- Room: {ctx.room.name}, Job: {ctx.job.id}")

    try:
        # Connect to the room using the context provided by cli.run_app
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"[AgentEntry] Connected to room. Local Participant SID: {ctx.room.local_participant.sid}")
    except Exception as e_connect:
        logger.error(f"[AgentEntry] Fatal error during ctx.connect(): {e_connect}", exc_info=True);
        # Exit if unable to connect to the room - the job cannot proceed
        # Use logger.critical before exiting
        logger.critical(f"!!! [AgentEntry] CRITICAL - Agent failed to connect to room: {e_connect}")
        sys.exit(f"Fatal error during ctx.connect(): {e_connect}")

    logger.info("[AgentEntry] Initializing plugins...")
    llm_plugin = None
    tts_plugin = None
    stt_plugin = None
    vad_plugin = None

    # Check for required OpenAI API key
    if not OPENAI_API_KEY_ENV:
        logger.critical("!!! [AgentEntry] CRITICAL - OPENAI_API_KEY environment variable is NOT set. OpenAI STT, LLM, and TTS plugins cannot initialize. !!!")
        sys.exit("OPENAI_API_KEY environment variable is not set.")

    try:
        logger.info("[AgentEntry] Initializing OpenAI STT plugin...")
        # OpenAI Whisper supports multiple languages automatically.
        stt_plugin = openai.STT(api_key=OPENAI_API_KEY_ENV, language="hi")
        logger.info("[AgentEntry] OpenAI STT plugin initialized.")

        logger.info("[AgentEntry] Initializing OpenAI LLM plugin...")
        llm_plugin = openai.LLM(model="gpt-4", api_key=OPENAI_API_KEY_ENV)
        logger.info("[AgentEntry] OpenAI LLM plugin initialized.")

        logger.info("[AgentEntry] Initializing OpenAI TTS plugin...")
        # Use a voice that might be good for multilingual speech like 'nova' or 'alloy'
        tts_plugin = openai.TTS(voice="nova", api_key=OPENAI_API_KEY_ENV)
        logger.info("[AgentEntry] OpenAI TTS plugin initialized.")

        logger.info("[AgentEntry] Initializing Silero VAD plugin...")
        # VAD helps detect speech vs silence
        vad_plugin = silero.VAD.load()
        logger.info("[AgentEntry] Silero VAD plugin initialized.")

        logger.info("[AgentEntry] All required plugins initialized successfully.")
    except Exception as e_plugins:
        logger.error(f"[AgentEntry] Error initializing plugins: {e_plugins}", exc_info=True);
        # Exit gracefully if plugins fail to initialize
        logger.critical(f"!!! [AgentEntry] CRITICAL - Error initializing plugins: {e_plugins}")
        sys.exit(f"Error initializing plugins: {e_plugins}")

    # --- LLM PLUGIN DIRECT TEST ---
    # This test is good for confirming the LLM plugin itself works.
    logger.info("[AgentEntry] --- Testing LLM plugin directly START ---")
    llm_test_passed = False
    llm_test_response_collected = ""

    if not llm_plugin:
        logger.error("[AgentEntry] LLM Plugin is None, cannot run direct test.")
    else:
        # Test prompts in English and Hindi
        test_prompt_text_en = "Hello, LLM. Confirm you are working by saying OK."
        test_prompt_text_hi = "नमस्ते। क्या आप हिंदी में बात कर सकते हैं? कृपया हिंदी में जवाब दीजिये।" # Added explicit Hindi response instruction

        chat_message_list: list[ChatMessage] = []
        try:
            # Create a short history ending with the Hindi prompt and instruction
            chat_message_list = [
                ChatMessage(role="user", content=["Test EN: " + test_prompt_text_en]),
                ChatMessage(role="user", content=["Test HI: " + test_prompt_text_hi])
            ]
            logger.info(f"[AgentEntry] LLM Direct Test: Created chat_message_list for English and Hindi tests.")
        except Exception as e_cm_create:
            logger.error(f"[AgentEntry] LLM Direct Test: FAILED to create ChatMessage list: {e_cm_create}", exc_info=True)

        if chat_message_list:
            chat_ctx_for_test: ChatContext | None = None
            try:
                # Pass messages list positionally as determined necessary by previous logs
                logger.info(f"[AgentEntry] LLM Direct Test: Attempting ChatContext(...) (positional)")
                chat_ctx_for_test = ChatContext(chat_message_list)
                logger.info(f"[AgentEntry] LLM Direct Test: ChatContext created.")
            except Exception as e_ctx_create:
                logger.error(f"[AgentEntry] LLM Direct Test: Error creating ChatContext: {e_ctx_create}", exc_info=True)
                chat_ctx_for_test = None

            if chat_ctx_for_test:
                try:
                    logger.info(f"[AgentEntry] LLM Direct Test: Attempting llm_plugin.chat(...)")

                    # --- HYPOTHESIS: plugin.chat returns async iterator directly ---
                    # If this causes 'coroutine was never awaited', revert to adding await back.
                    llm_test_stream = llm_plugin.chat(chat_ctx=chat_ctx_for_test)

                    # Immediately iterate over the result, assuming it's the async iterator
                    async for chunk in llm_test_stream:
                        logger.info(f"[LLM Test] Got chunk: {chunk}")
                        chunk_text = getattr(chunk.delta, 'content', None)  # <-- Fix is here
                        if chunk_text:
                            llm_test_response_collected += chunk_text
                            logger.debug(f"[AgentEntry] LLM Direct Test Chunk: '{chunk_text}'")

                    logger.info(f"[LLM Test] Final collected response: '{llm_test_response_collected.strip()}'")


                    # After the loop finishes, check if any text was collected
                    if llm_test_response_collected.strip():
                        llm_test_passed = True
                        # Log the final collected response at INFO level
                        logger.info(f"[AgentEntry] LLM plugin direct test SUCCEEDED. Collected response: '{llm_test_response_collected}'")
                    else:
                        logger.warning("[AgentEntry] LLM plugin direct test: Received stream, but no text chunks were collected.")

                # Removed asyncio.TimeoutError handling here as wait_for is removed

                except Exception as e_llm_call:
                    # This catches errors from llm_plugin.chat() call itself, or issues during streaming
                    logger.error(f"[AgentEntry] LLM plugin direct test FAILED during chat call or stream processing: {e_llm_call}", exc_info=True)
            else:
                logger.warning("[AgentEntry] LLM Direct Test: chat_ctx_for_test is None, skipping llm.chat() call.")
        else:
            logger.warning("[AgentEntry] LLM Direct Test: chat_message_list is empty.")

    if not llm_test_passed:
        logger.critical("[AgentEntry] !!! LLM direct test FAILED. LLM might not function correctly. !!!")
        # Decide if you want to exit if the LLM test fails.
        # For now, let's allow it to continue, but know the LLM is likely broken.
        # sys.exit("LLM plugin direct test failed.")
    else:
        logger.info("[AgentEntry] --- Testing LLM plugin directly END (Successful) ---")


    session = AgentSession(stt=stt_plugin, llm=llm_plugin, tts=tts_plugin, vad=vad_plugin)
    logger.info("[AgentEntry] AgentSession created with OpenAI STT, LLM, TTS and Silero VAD.")

    assistant_agent_instance = GUIFocusedAssistant()
    logger.info("[AgentEntry] GUIFocusedAssistant instance created.")

    logger.info("[AgentEntry] Starting AgentSession with GUIFocusedAssistant...")
    # --- CRITICAL FIX: Await session.start() ---
    # This *must* block the entrypoint function until the session terminates.
    # This is the standard way for an entrypoint in cli.run_app to keep running.
    # If this is still returning immediately, the issue is deeper in your environment setup
    # or the interaction with livekit-agents/rtc library in this threaded context.
    await session.start(room=ctx.room, agent=assistant_agent_instance)
    logger.info("[AgentEntry] AgentSession.start() finished/returned. This typically means the session has ended (e.g., room closed).")

    logger.info(f"[AgentEntry] Agent logic finished/terminated for job {ctx.job.id if ctx.job else 'N/A'}.")


def check_and_free_port(port):
    """Check if a port is in use and attempt to free it."""
    # This function might be less critical if you run processes separately
    # but can be useful during development. Removed threading use here.
    logger.debug(f"Checking if port {port} is in use...")
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.net_connections(kind='inet'):
                    if conn.laddr.port == port:
                        logger.warning(f"Port {port} is in use by PID {proc.pid} ({proc.name()}). Attempting to terminate.")
                        try:
                            # Attempt graceful termination first
                            proc.terminate()
                            proc.wait(timeout=3)
                            if proc.is_running():
                                 # If still running, attempt kill
                                 logger.warning(f"Process {proc.pid} did not terminate gracefully. Attempting kill.")
                                 proc.kill()
                                 proc.wait(timeout=3)
                            logger.info(f"Successfully handled potential process on port {port}.")
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as term_err:
                             logger.error(f"Failed to terminate process {proc.pid} on port {port}: {term_err}")
                        except Exception as e:
                             logger.error(f"Unexpected error while handling process {proc.pid} on port {port}: {e}", exc_info=True)
                        return # Port should be free now
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process might have exited or we don't have permission
                continue
            except Exception as e:
                 logger.error(f"Error iterating process connections for PID {proc.pid}: {e}", exc_info=True)

    except Exception as e:
         logger.error(f"Error iterating processes: {e}", exc_info=True)

    logger.debug(f"Port {port} appears free.")


# --- REMOVED run_worker function ---
# We are now running the worker directly in the main thread when needed.

if __name__ == "__main__":
    import faulthandler
    faulthandler.enable() # Helpful for debugging crashes
    logger.info("--- __main__: main.py script execution started ---")

    # Check command line arguments to determine if running agent or FastAPI
    # Example usage:
    # python main.py          # Runs FastAPI server for tokens
    # python main.py start    # Runs the LiveKit agent worker
    # python main.py download-files # Downloads agent plugin files (handled by cli.run_app)
    # python main.py package  # Packages the agent (handled by cli.run_app)

    if len(sys.argv) > 1 and sys.argv[1] in ["start", "download-files", "package"]:
        # --- Run the LiveKit Agent Worker ---
        logger.info(f"--- __main__: Command line argument '{sys.argv[1]}' detected. Running cli.run_app directly (Agent Worker mode). ---")

        # cli.run_app handles its own event loop and process management.
        # It requires being run in the main thread of the process in many environments.
        # We pass the original sys.argv to cli.run_app so it can parse commands like 'start'.
        try:
            # NOTE: nest_asyncio is usually NOT needed when cli.run_app runs in the main thread.
            # It's specifically for running asyncio loops where one is already running.
            # Keep it removed.
            # import nest_asyncio
            # nest_asyncio.apply() # REMOVED

            cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

        except SystemExit as se:
             # cli.run_app exits with SystemExit on normal shutdown
             logger.info(f"--- __main__: cli.run_app exited SystemExit (code: {se.code}).")
        except Exception as e:
             logger.error(f"--- __main__: CRITICAL ERROR during cli.run_app (Agent Worker): {e}", exc_info=True)
             sys.exit(f"Agent Worker process failed: {e}") # Exit process on failure

        logger.info("--- __main__: Script execution finished (Agent Worker mode). ---")

    else:
        # --- Run the FastAPI Server ---
        logger.info("--- __main__: No specific command line argument. Starting FastAPI server (Token Server mode). ---")

        # Check and free port 8000 for Uvicorn
        check_and_free_port(8000)

        logger.info("--- __main__: Starting Uvicorn server for FastAPI app ---")
        try:
            # Uvicorn manages its own event loop, typically in the main thread or worker threads it creates.
            uvicorn.run(
                "main:app",
                host="0.0.0.0",
                port=8000,
                reload=False, # reload=True is generally not compatible with Agent development
                log_level="info" # Or 'debug' for more Uvicorn details
            )
        except Exception as e_uvicorn:
            logger.error(f"--- __main__: Uvicorn server failed: {e_uvicorn}", exc_info=True)
            sys.exit(f"FastAPI server failed: {e_uvicorn}") # Exit process on failure

        logger.info("--- __main__: Uvicorn server has stopped. ---")

        logger.info("--- __main__: Script execution finished (Token Server mode). ---")