


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
from livekit.plugins.turn_detector.multilingual import MultilingualModel


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
    level=logging.DEBUG, # Keep at DEBUG for now to see all our debug logs
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_logs.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
agent_logger = logging.getLogger("livekit.agents")
agent_logger.setLevel(logging.DEBUG) # Ensure LiveKit's internal debug logs are also shown

import json

def log_structured(event: str, content: str, language: str = "unknown"):
    logger.info(json.dumps({
        "event": event,
        "language": language,
        "content": content
    }, ensure_ascii=False, indent=2))

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
                "**Always respond ONLY in the language the user used for their query.** "
                "आप एक सहायक AI ऑडियो असिस्टेंट हैं। केवल हिंदी में देवनागरी लिपि का उपयोग करके उत्तर दें। "
                "अपने उत्तर संक्षिप्त और मैत्रीपूर्ण रखें, और आपकी प्राथमिक भाषा हिंदी है। "
                "**If the user speaks in Hindi, respond ONLY in Hindi.** "
                "**If the user speaks in English, respond ONLY in English.** "
                "**Your primary language is Hindi, but strict language mirroring is required.**"
                
            )
        )
        self.last_user_query_for_log = None
        self.chat_history: list[ChatMessage] = []
        logger.info("--- [GUIFocusedAssistant] __init__ CALLED ---")

    async def on_transcript(self, transcript: str, is_final: bool) -> None:
        # This log should appear if the agent session is running and receiving transcripts
        logger.info("--- [GUIFocusedAssistant] on_transcript CALLED --- Final: %s, Text: %s", is_final, transcript)
        # The SDK itself might log the received transcript at DEBUG level before this (the structured JSON log).
        # Example: {"message": "received user transcript", "level": "DEBUG", "name": "livekit.agents", "user_transcript": "...", "language": "en/hi", "timestamp": "..."}

        if is_final and transcript:
            # logger.warning("--- [GUIFocusedAssistant] STT returned final result but NO transcript text. Possible STT failure or low-confidence speech.")
            self.last_user_query_for_log = transcript
            log_structured("user_query", transcript, language="hi")


            # logger.info(f"--- [GUIFocusedAssistant] on_transcript: User query received (final): '{transcript}'. Attempting to generate LLM response.")

            # Check if llm plugin is available
            if not hasattr(self, 'llm') or not self.llm:
                logger.error("--- [GUIFocusedAssistant] on_transcript: self.llm is not available. Cannot generate LLM response.")
                # Synthesize a fallback message if LLM is missing
                if hasattr(self, 'tts') and self.tts:
                    logger.info("--- [GUIFocusedAssistant] on_transcript: Attempting fallback TTS due to missing LLM.")
                    # Define the async generator function
                    async def gen_llm_error_fallback():
                         yield "I'm sorry, I cannot access the language model right now."
                    # Call the function to get the generator object and pass it to synthesize
                    await self.tts.synthesize(gen_llm_error_fallback())
                return

            # Check if tts plugin is available before proceeding, as handle_llm_response relies on it
            if not hasattr(self, 'tts') or not self.tts:
                 logger.error("--- [GUIFocusedAssistant] on_transcript: self.tts is not available. Cannot synthesize response.")
                 # We can still try to get the LLM response and log it, but can't speak.
                 # Continue processing but log the lack of TTS.
                 pass # Allow processing to continue to handle_llm_response for logging/frontend data

            try:
                # Create ChatMessage for current user transcript
                user_chat_message = ChatMessage(role="user", content=[transcript])

                # Append to chat history and prune if necessary
                current_chat_turn_history = list(self.chat_history) # Work on a copy for this turn
                current_chat_turn_history.append(user_chat_message)
                if len(current_chat_turn_history) > 20: # Keep history length reasonable
                    current_chat_turn_history = current_chat_turn_history[-20:]

                # Create ChatContext using positional argument
                logger.debug(f"--- [GUIFocusedAssistant] on_transcript: Creating ChatContext for LLM call...")
                chat_ctx_for_llm = ChatContext(current_chat_turn_history)
                logger.debug(f"--- [GUIFocusedAssistant] on_transcript: ChatContext created with {len(current_chat_turn_history)} messages.")

                logger.debug(f"--- [GUIFocusedAssistant] on_transcript: Calling self.llm.chat(...) with context...")

                # Call self.llm.chat - likely takes ChatContext as positional arg
                llm_stream = await self.llm.chat(chat_ctx_for_llm)
                logger.debug(f"--- [GUIFocusedAssistant] on_transcript: self.llm.chat() returned stream.")

                # Update chat history only AFTER a successful LLM chat call returns a stream object
                self.chat_history = current_chat_turn_history

                # Pass the stream to the handler
                await self.handle_llm_response(llm_stream)

            except Exception as e_llm_call_or_handle:
                 # Catching broader Exception for safety in the main turn processing
                 logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error during LLM turn processing or handle_llm_response: {e_llm_call_or_handle}", exc_info=True)
                 # Optionally, synthesize a fallback message on LLM processing error
                 if hasattr(self, 'tts') and self.tts:
                    logger.info("--- [GUIFocusedAssistant] on_transcript: Attempting fallback TTS due to LLM processing error.")
                    # Define the async generator function
                    async def gen_process_error_fallback():
                        yield "I encountered an error while processing your request. Please try again."
                    # Call the function to get the generator object and pass it to synthesize
                    await self.tts.synthesize(gen_process_error_fallback())


        # This publishes the user's transcript to the frontend, already logged above
        # Keep this if your frontend expects it
        if is_final and transcript:
            if self.room and self.room.local_participant:
                try:
                    await self.room.local_participant.publish_data(f"transcription:{transcript}", "transcription")
                except Exception as e_pub_tx:
                    logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error publishing user transcript: {e_pub_tx}", exc_info=True)


    async def handle_llm_response(self, llm_stream: LLMStream) -> None:
        logger.info(f"--- [GUIFocusedAssistant] handle_llm_response CALLED ---")
        collected_text_from_sentences = ""
        sentences_for_tts_final = []
        stream_processed_successfully = False # Flag to indicate if stream yielded content

        try:
            logger.debug("--- [GUIFocusedAssistant] handle_llm_response: Starting sentence processing for TTS.")
            # Iterate over the stream broken into sentences by AgentSession/voice pipeline helper
            # This is where the text content comes from.
            async for sentence_obj in self.llm_stream_sentences(llm_stream):
                 text_sentence = getattr(sentence_obj, 'text', str(sentence_obj)) # Extract text safely
                 # Log each sentence chunk as DEBUG - this helps see text arriving
                 logger.debug(f"--- [GUIFocusedAssistant] handle_llm_response: Processed sentence chunk for TTS: '{text_sentence}'")

                 if text_sentence:
                    # Collect text for the final log and frontend data
                    collected_text_from_sentences += text_sentence + " "
                    # Collect sentences for the final TTS call
                    sentences_for_tts_final.append(text_sentence)
                    stream_processed_successfully = True # At least one sentence was processed

        except Exception as e_llm_stream_processing:
            logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error processing LLM stream into sentences: {e_llm_stream_processing}", exc_info=True)
            # If stream processing failed, add a fallback message to TTS/log list
            error_text = "Error generating response."
            if collected_text_from_sentences:
                 error_text = collected_text_from_sentences + " (Error in stream processing)"

            collected_text_from_sentences = error_text # Update collected text to include error info
            # Add error text for TTS if nothing else was collected, or add it as a final sentence
            if not sentences_for_tts_final:
                 sentences_for_tts_final.append(error_text)
            # else: # Decide if you want to append error *after* partial response
            #      sentences_for_tts_final.append(error_text)


        finally:
             logger.debug("--- [GUIFocusedAssistant] handle_llm_response: Finished LLM stream processing.")


        # This is the final text that should be logged and sent to the frontend
        final_collected_text = collected_text_from_sentences.strip()

        # Log Intermediate/Final collected text (DEBUG level) - useful for debugging empty strings
        logger.debug(f"--- [GUIFocusedAssistant] handle_llm_response: Raw collected text from sentences (before strip): '{collected_text_from_sentences}'")
        logger.debug(f"--- [GUIFocusedAssistant] handle_llm_response: Trimmed final collected text: '{final_collected_text}'")
        logger.debug(f"--- [GUIFocusedAssistant] handle_llm_response: Sentences prepared for TTS (count {len(sentences_for_tts_final)}): {sentences_for_tts_final}")


        log_context_info = f"for user query '{self.last_user_query_for_log}'" if self.last_user_query_for_log else "(initial/agent-initiated)"

        # Check if we have *any* text to log (could be a successful response or an error message added above)
        if final_collected_text:
            # Add the agent's response to chat history
            # Content should be a list of strings for ChatMessage
            assistant_chat_message = ChatMessage(role="assistant", content=[final_collected_text])
            # Append to the *actual* chat history used for the next turn
            # self.chat_history was already updated with user message in on_transcript
            self.chat_history.append(assistant_chat_message)
            if len(self.chat_history) > 20: self.chat_history = self.chat_history[-20:] # Prune history

            # --- THIS IS THE MAIN LOG YOU WANT TO SEE FOR MODEL RESPONSES ---
            # Log the FINAL collected text at INFO level
            # This should log both English and Hindi text depending on the model's output
            log_structured("assistant_response", final_collected_text, language="hi")
        else:
            fallback_msg = "कोई उत्तर उत्पन्न नहीं हुआ।"
            log_structured("assistant_response", fallback_msg, language="hi")
            # Log if no text was collected/generated (even error text wasn't produced)
            logger.warning(f"--- [GUIFocusedAssistant] handle_llm_response: No final text collected from LLM stream {log_context_info}.")


        # Publish text to frontend (Assuming frontend handles empty strings gracefully)
        # Always publish the final collected text, even if it's empty or an error message,
        # so the frontend gets an update.
        if self.room and self.room.local_participant:
             # Only publish if there's collected text, or if the stream processing failed (implying an error message was added)
             if final_collected_text or not stream_processed_successfully:
                logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Publishing agent response text to GUI: '{final_collected_text}'")
                try:
                    # Ensure payload is not None or empty if you don't want to publish nothing
                    payload_to_publish = f"response:{final_collected_text}" if final_collected_text else "response:Error generating response."
                    await self.room.local_participant.publish_data(payload=payload_to_publish, topic="response")
                except Exception as e_pub_resp:
                    logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error publishing agent response text: {e_pub_resp}", exc_info=True)
             else:
                 logger.debug("--- [GUIFocusedAssistant] handle_llm_response: No text to publish to GUI.")


        # Attempt TTS synthesis ONLY if there's text prepared for it
        if sentences_for_tts_final and any(s.strip() for s in sentences_for_tts_final):
            logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Attempting TTS synthesis for {len(sentences_for_tts_final)} sentences.")
            # The generator yields each sentence to the TTS plugin
            async def gen_tts():
                for s in sentences_for_tts_final: yield s.strip() # Yield stripped sentences
            try:
                await self.tts.synthesize(gen_tts())
                logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: TTS synthesis initiated successfully.")
            except Exception as e_tts:
                logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error during TTS synthesis: {e_tts}", exc_info=True)
        else:
            logger.warning("--- [GUIFocusedAssistant] handle_llm_response: No text prepared for TTS synthesis.")

        self.last_user_query_for_log = None # Reset after turn is fully processed
        logger.info(f"--- [GUIFocusedAssistant] handle_llm_response FINISHED ---")

# --- Agent Entrypoint ---
async def entrypoint(ctx: JobContext):
    logger.info(f"--- [AgentEntry] AGENT LOGIC ENTRYPOINT CALLED --- Room: {ctx.room.name}, Job: {ctx.job.id}")

    try:
        # Connect to the room - this needs to succeed for anything else to work
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"[AgentEntry] Connected to room. Local Participant SID: {ctx.room.local_participant.sid}")
    except Exception as e_connect:
        logger.critical(f"[AgentEntry] Fatal error during ctx.connect(): {e_connect}", exc_info=True);
        # If connection fails, the job should terminate. Raise the exception.
        raise

    logger.info("[AgentEntry] Initializing plugins...")
    stt_plugin = None
    llm_plugin = None
    tts_plugin = None
    vad_plugin = None
    plugins_initialized_successfully = False

    turn_detector = MultilingualModel()
    logger.info("[AgentEntry] Multilingual turn detector initialized.")


    try:
        # Initialize STT - multi-language is key for Hindi detection
        stt_plugin = deepgram.STT(model="nova-2", language="hi", interim_results=True)
        logger.info("[AgentEntry] Deepgram STT plugin initialized.")

        # Initialize LLM
        if not OPENAI_API_KEY_ENV:
             logger.error("[AgentEntry] OpenAI API key not found for LLM/TTS plugins.")
             raise ValueError("OPENAI_API_KEY environment variable is not set.")
        llm_plugin = openai.LLM(model="gpt-4o-mini", api_key=OPENAI_API_KEY_ENV)
        logger.info("[AgentEntry] OpenAI LLM plugin initialized.")

        # Initialize TTS
        tts_plugin = openai.TTS(voice="ash", api_key=OPENAI_API_KEY_ENV) # Use a valid OpenAI voice
        logger.info("[AgentEntry] OpenAI TTS plugin initialized.")

        # Initialize VAD
        vad_plugin = silero.VAD.load()
        logger.info("[AgentEntry] Silero VAD plugin initialized.")

        logger.info("[AgentEntry] All essential plugins initialized successfully.")
        plugins_initialized_successfully = True

    except Exception as e_plugins:
        logger.critical(f"[AgentEntry] CRITICAL ERROR initializing plugins: {e_plugins}", exc_info=True);
        # If essential plugins like LLM or STT fail, the agent cannot function.
        # The entrypoint should terminate.
        # Consider sending a data message to the frontend about the failure if possible *after* connecting
        if ctx.room and ctx.room.local_participant:
             try:
                 # This might fail if the room connection is also unstable, but worth a try.
                 await ctx.room.local_participant.publish_data(payload=f"error:Agent failed to initialize plugins: {e_plugins}", topic="error")
             except Exception:
                 pass # Ignore errors publishing error messages
        raise # Re-raise the exception to terminate the job


    # --- Removed LLM PLUGIN DIRECT TEST ---
    logger.info("[AgentEntry] Skipping problematic LLM direct test.")
    # --- End Removed LLM PLUGIN DIRECT TEST ---


    # Create and start the AgentSession with the initialized plugins
    # This is where your custom agent logic (on_transcript, handle_llm_response) gets hooked up
    # Ensure all required plugins were initialized
    if stt_plugin and llm_plugin and tts_plugin and vad_plugin:
        session = AgentSession(stt=stt_plugin, llm=llm_plugin, tts=tts_plugin, vad=vad_plugin, turn_detection=turn_detector,)
        logger.info("[AgentEntry] AgentSession created with initialized plugins.")

        assistant_agent_instance = GUIFocusedAssistant() # Your custom agent logic instance
        logger.info("[AgentEntry] GUIFocusedAssistant instance created.")

        logger.info("[AgentEntry] Starting AgentSession with GUIFocusedAssistant...")
        # session.start awaits the completion of the agent's main loop
        main_agent_task = asyncio.create_task(session.start(room=ctx.room, agent=assistant_agent_instance))
        logger.info("[AgentEntry] AgentSession.start() called. Session will now manage audio/transcript/response turns.")

        logger.info("[AgentEntry] Main agent logic running. Waiting for AgentSession task to complete.")
        try:
            # Await the main agent task. This will block until the session ends (e.g., room closes)
            await main_agent_task
            logger.info("[AgentEntry] Main agent_task completed normally.")
        except asyncio.CancelledError:
            logger.info("[AgentEntry] Main agent_task was cancelled.")
        except Exception as e_main_task:
            # This catches exceptions that might occur *during* the session's main loop
            logger.error(f"[AgentEntry] Main agent_task exited with error during session runtime: {e_main_task}", exc_info=True)
    else:
         # This case should ideally be caught by the exception handling during plugin init,
         # but as a fallback log if something goes wrong.
         logger.critical("[AgentEntry] Cannot start AgentSession because essential plugins were not initialized.")


    logger.info(f"[AgentEntry] Agent logic finished/terminated for job {ctx.job.id if ctx.job else 'N/A'}.")


def check_and_free_port(port):
    """Check if a port is in use and attempt to free it."""
    # ... (rest of this function is unchanged) ...
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.net_connections(kind='inet'):
                if conn.laddr.port == port:
                    logger.warning(f"Port {port} is in use by PID {proc.pid} ({proc.name()}). Attempting to terminate.")
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                        logger.info(f"Successfully terminated process on port {port}.")
                    except psutil.TimeoutExpired:
                         logger.warning(f"Process PID {proc.pid} did not terminate within timeout, killing.")
                         proc.kill()
                         logger.info(f"Successfully killed process on port {port}.")
                    return # Exit after handling the first process found
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    logger.debug(f"Port {port} is free or no process could be terminated.")

def run_worker():
    try:
        logger.info("--- run_worker: Attempting to start LiveKit Agent Worker ---")
        # ... (rest of this function is unchanged) ...
        if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
            logger.critical("!!! run_worker: CRITICAL - LiveKit connection credentials (URL, KEY, SECRET) NOT FOUND. Worker will NOT start.")
            return

        # Plugin initialization will now happen in entrypoint, which includes env var checks there.
        # Removed redundant checks here, relying on entrypoint for detailed errors.
        # But keep warnings for visibility if env vars are missing.
        if not OPENAI_API_KEY_ENV: logger.warning("!!! run_worker: OpenAI API key not found. LLM/TTS plugins might fail during entrypoint initialization.")
        if not DEEPGRAM_API_KEY_ENV: logger.warning("!!! run_worker: Deepgram API key not found. STT plugin might fail during entrypoint initialization.")


        # Check and free port 8081 for LiveKit worker
        check_and_free_port(8081)

        worker_options = WorkerOptions(
            entrypoint_fnc=entrypoint,
        )

        original_argv = list(sys.argv)
        # Ensure we're only passing the command 'start' to the cli.run_app
        sys.argv = [original_argv[0], 'start'] if len(original_argv) > 0 else ['livekit_agent_script_embedded.py', 'start']
        logger.info(f"run_worker: Modified sys.argv for cli.run_app: {sys.argv}")

        logger.info("run_worker: Applying nest_asyncio and calling cli.run_app(worker_options)...")
        try:
            import nest_asyncio
            nest_asyncio.apply()
            logger.info("run_worker: Applied nest_asyncio.")
            # cli.run_app expects the 'start' command to be in sys.argv
            cli.run_app(worker_options)
        except SystemExit as se:
             # SystemExit 0 is normal when cli.run_app finishes successfully
             if se.code != 0:
                 logger.error(f"run_worker: cli.run_app exited with non-zero status code {se.code}.", exc_info=True)
             else:
                 logger.info(f"run_worker: cli.run_app exited successfully (code: {se.code}).")
        except Exception as e:
            # This will catch exceptions *within* cli.run_app itself, not the async entrypoint task
            logger.error(f"!!! run_worker: CRITICAL ERROR during cli.run_app: {e}", exc_info=True)
        finally:
            sys.argv = original_argv # Restore original argv
            logger.info(f"run_worker: Restored original sys.argv.")

        logger.info("--- run_worker: LiveKit Agent Worker `cli.run_app` finished. ---")

    except Exception as e_outer:
        # This catches exceptions that happen *outside* the cli.run_app call but inside run_worker
        logger.error(f"!!! run_worker: CRITICAL ERROR in outer run_worker block: {e_outer}", exc_info=True)
    finally:
        logger.info("--- run_worker: Function execution complete or terminated. ---")


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    logger.info("--- __main__: main.py script execution started ---")

    # Check and free port 8000 for Uvicorn
    check_and_free_port(8000)

    logger.info("--- __main__: Creating worker thread (LiveKitAgentWorker) ---")
    # Daemon thread will exit automatically when the main process (uvicorn) exits
    worker_thread = threading.Thread(target=run_worker, name="LiveKitAgentWorker", daemon=True)
    worker_thread.start()
    logger.info("--- __main__: Worker thread started.")

    # Give the worker thread a moment to initialize.
    # If entrypoint crashes instantly, the worker thread might die here.
    logger.info("--- __main__: Waiting a few seconds for agent worker to initialize before starting Uvicorn...")
    time.sleep(5)

    if worker_thread.is_alive():
        logger.info("--- __main__: Worker thread IS ALIVE. Proceeding to start Uvicorn.")
    else:
        # If the worker thread isn't alive, the agent won't work. Log this critically.
        logger.critical("--- __main__: Worker thread IS NOT ALIVE after initial sleep. Agent will not function. Check 'run_worker' logs for CRITICAL errors.")
        # Consider adding sys.exit(1) here if you want the script to stop if the worker fails to start
        # sys.exit(1)


    logger.info("--- __main__: Starting Uvicorn server for FastAPI app ---")
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False, # Keep reload=False for production/agent usage
            log_level="info" # Keep this info or higher for less noise from uvicorn itself
        )
    except Exception as e_uvicorn:
        logger.error(f"--- __main__: Uvicorn server failed: {e_uvicorn}", exc_info=True)

    logger.info("--- __main__: Uvicorn server has stopped. ---")
    logger.info("--- __main__: Script execution finished. ---")