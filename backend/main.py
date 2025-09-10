# # main.py
# import asyncio
# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# # from fastapi.staticfiles import StaticFiles # Not used in this version

# from livekit.api import AccessToken, VideoGrants
# from livekit import agents, rtc
# from livekit.agents import (
#     JobContext,
#     WorkerOptions,
#     cli,
#     Agent,
#     AgentSession,
#     AutoSubscribe,
#     # If you upgraded livekit-agents and these are available, uncomment them:
#     # JobType,
#     # AutoDisconnect,
# )
# from livekit.plugins import elevenlabs, openai, silero
# from fastapi.middleware.cors import CORSMiddleware
# # from starlette.responses import FileResponse # Not used in this version
# import uvicorn
# import logging
# from datetime import timedelta, timezone, datetime
# import jwt # For local decoding (pip install pyjwt)
# import threading
# import time # For diagnostic sleep
# import sys # REQUIRED for sys.argv modification

# # Load environment variables at the very beginning
# load_dotenv()

# app = FastAPI()

# origins = [
#     "http://localhost:3000", # Your frontend URL
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')
# logger = logging.getLogger(__name__) # Get logger for the current module (main)
# agent_logger = logging.getLogger("livekit.agents") # Get logger for livekit.agents specifically
# agent_logger.setLevel(logging.INFO) # Ensure agent logs are visible if not already by root config

# # Fetch environment variables after load_dotenv()
# LIVEKIT_URL = os.getenv("LIVEKIT_URL")
# LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
# LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
# OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY")
# ELEVENLABS_API_KEY_ENV = os.getenv("ELEVENLABS_API_KEY")

# # Log these crucial variables once at startup using the __main__ logger
# logger.info(f"MODULE LEVEL: LIVEKIT_URL: {LIVEKIT_URL}")
# logger.info(f"MODULE LEVEL: LIVEKIT_API_KEY IS SET: {bool(LIVEKIT_API_KEY)}")
# logger.info(f"MODULE LEVEL: LIVEKIT_API_SECRET IS SET: {bool(LIVEKIT_API_SECRET)}")
# logger.info(f"MODULE LEVEL: OPENAI_API_KEY IS SET: {bool(OPENAI_API_KEY_ENV)}")
# logger.info(f"MODULE LEVEL: ELEVENLABS_API_KEY IS SET: {bool(ELEVENLABS_API_KEY_ENV)}")


# @app.get("/token")
# async def get_token():
#     logger.info("[TokenSvc] /token endpoint hit.")
#     if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
#         logger.error("[TokenSvc] LiveKit credentials (URL, KEY, SECRET) not configured properly at module level.")
#         raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

#     if LIVEKIT_API_KEY and not LIVEKIT_API_KEY.startswith("API"):
#         logger.warning(f"[TokenSvc] LIVEKIT_API_KEY does not look like a standard API key: {LIVEKIT_API_KEY[:7]}...")

#     video_grant_obj = VideoGrants(
#         room_join=True,
#         room="voice-assistant-room",
#         can_publish=True,
#         can_subscribe=True,
#         can_publish_data=True, # Changed to True for consistency, agent needs it
#     )

#     token_builder = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
#     token_builder.identity = "frontend_user"
#     token_builder.name = "Frontend User"
#     token_builder.ttl = timedelta(hours=1)
#     token_builder.with_grants(video_grant_obj)

#     token_jwt = token_builder.to_jwt()

#     logger.info(f"--- [TokenSvc] GENERATED TOKEN FOR FRONTEND ---")
#     logger.info(f"Token JWT (first 30): {token_jwt[:30]}...")
#     logger.info(f"Server URL: {LIVEKIT_URL}")
#     logger.info(f"-------------------------------------------")

#     try:
#         decoded_payload = jwt.decode(token_jwt, options={"verify_signature": False, "verify_exp": False})
#         logger.info(f"[TokenSvc] Locally Decoded Payload for frontend token: {decoded_payload}")
#     except ImportError:
#         logger.warning("[TokenSvc] PyJWT not installed, cannot decode token locally. Run: pip install pyjwt")
#     except Exception as e:
#         logger.error(f"[TokenSvc] Error locally decoding frontend token: {e}")

#     return {"token": token_jwt, "url": LIVEKIT_URL}


# async def entrypoint(ctx: JobContext): # Keep this exact signature
#     entry_time = datetime.now().isoformat() # Make sure datetime is imported globally
#     room_name_from_ctx = ctx.room.name if ctx.room else "N/A_ROOM_IN_CTX"
#     job_id_from_ctx = ctx.job.id if ctx.job else "N/A_JOB_IN_CTX"
    
#     # Try writing to a simple text file immediately
#     # USE A *DIFFERENT* FILENAME to distinguish from minimal_agent_test.py's output
#     debug_file_name = "threaded_agent_entrypoint_debug.txt"
#     try:
#         with open(debug_file_name, "a") as f:
#             f.write(f"{entry_time} - THREADED AGENT MINIMAL ENTRYPOINT TOUCHED! Room: {room_name_from_ctx}, Job: {job_id_from_ctx}\n")
#     except Exception as e_file:
#         # Fallback to print if file write fails
#         print(f"DEBUG PRINT (Threaded Agent main.py): Failed to write to {debug_file_name}: {e_file}")

#     logger.info(f"--- [AgentEntry] !!! THREADED MINIMAL ENTRYPOINT CALLED !!! --- Room: {room_name_from_ctx}, Job ID: {job_id_from_ctx}")
    
#     try:
#         logger.info(f"[AgentEntry] Threaded Minimal: Attempting ctx.connect() for job {job_id_from_ctx}.")
#         await ctx.connect() # Simplest connection
#         logger.info(f"[AgentEntry] Threaded Minimal: Successfully called ctx.connect(). Participant SID: {ctx.room.local_participant.sid if ctx.room and ctx.room.local_participant else 'N/A'}")
#         logger.info("[AgentEntry] Threaded Minimal: Waiting indefinitely.")
#         await asyncio.Future() # Keep it alive
#     except Exception as e_connect:
#         logger.error(f"[AgentEntry] Threaded Minimal: Error during ctx.connect() or asyncio.Future(): {e_connect}", exc_info=True)
#         try: 
#             with open(debug_file_name, "a") as f:
#                 f.write(f"{datetime.now().isoformat()} - ERROR IN THREADED MINIMAL ENTRYPOINT: {e_connect}\n")
#         except: pass
#         raise 
#     finally:
#         logger.info(f"[AgentEntry] Threaded Minimal: Finished or terminated for job {job_id_from_ctx}.")
#         try: 
#             with open(debug_file_name, "a") as f:
#                 f.write(f"{datetime.now().isoformat()} - THREADED MINIMAL ENTRYPOINT FINALLY BLOCK for job {job_id_from_ctx}.\n")
#         except: pass

# def run_worker():
#     # This try-except is to catch any immediate error when this function is called
#     try:
#         logger.info("--- run_worker: TOP OF FUNCTION - Attempting to start LiveKit Agent Worker ---")
#         logger.info(f"run_worker: Using LIVEKIT_URL: {LIVEKIT_URL}")
#         logger.info(f"run_worker: LIVEKIT_API_KEY IS SET: {bool(LIVEKIT_API_KEY)}")
#         logger.info(f"run_worker: LIVEKIT_API_SECRET IS SET: {bool(LIVEKIT_API_SECRET)}")
#         logger.info(f"run_worker: OPENAI_API_KEY IS SET FOR PLUGINS: {bool(OPENAI_API_KEY_ENV)}")
#         logger.info(f"run_worker: ELEVENLABS_API_KEY IS SET FOR PLUGINS: {bool(ELEVENLABS_API_KEY_ENV)}")

#         if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
#             logger.critical("!!! run_worker: CRITICAL - LiveKit connection credentials (URL, KEY, SECRET) NOT FOUND.")
#             logger.critical("!!! run_worker: Worker will NOT start.")
#             return

#         if not OPENAI_API_KEY_ENV:
#             logger.warning("!!! run_worker: WARNING - OpenAI API key not found. Agent STT/LLM plugins might fail.")
#         if not ELEVENLABS_API_KEY_ENV:
#             logger.warning("!!! run_worker: WARNING - ElevenLabs API key not found. Agent TTS plugin might fail.")

#         logger.info(f"run_worker: Preparing WorkerOptions for URL: {LIVEKIT_URL}")
#         worker_options = WorkerOptions(entrypoint_fnc=entrypoint)
        
#         # ---- CRITICAL CHANGE FOR PROGRAMMATIC CLI EXECUTION ----
#         original_argv = list(sys.argv) # Store original argv
#         # The first argument can be anything (traditionally script name),
#         # the crucial part is the 'start' command for the agent CLI.
#         sys.argv = ['livekit_agent_script.py', 'start'] 
#         logger.info(f"run_worker: Modified sys.argv for cli.run_app: {sys.argv}")
#         # ---- END OF CRITICAL CHANGE ----
        
#         logger.info("run_worker: Calling cli.run_app(worker_options)... This will block until worker stops.")
#         try:
#             import nest_asyncio
#             nest_asyncio.apply()
#             logger.info("run_worker: Applied nest_asyncio.")
#             cli.run_app(worker_options) # This is a blocking call
#         finally:
#             sys.argv = original_argv # Restore original argv AFTER cli.run_app finishes or errors
#             logger.info(f"run_worker: Restored original sys.argv: {sys.argv}")

#         # This log will only be reached if cli.run_app returns normally (e.g., graceful shutdown)
#         logger.info("--- run_worker: LiveKit Agent Worker `cli.run_app` finished (returned normally). ---")

#     except SystemExit as se:
#         # SystemExit is often raised by cli.run_app on normal exit or if CLI parsing fails.
#         # Code 0 is usually a clean exit. Non-zero might indicate an issue.
#         # Code 2 from 'click' often means a usage/argument error.
#         if se.code == 0:
#             logger.info(f"run_worker: cli.run_app exited with SystemExit (code: {se.code}). This is usually a normal shutdown.")
#         else:
#             logger.warning(f"run_worker: cli.run_app exited with SystemExit (code: {se.code}). This might indicate an issue or CLI error.")
#     except Exception as e:
#         logger.error(f"!!! run_worker: CRITICAL ERROR during run_worker execution or within cli.run_app: {e}", exc_info=True)
#     finally:
#         # This finally block ensures this log is printed regardless of how run_worker exits.
#         logger.info("--- run_worker: Function execution complete or terminated. ---")


# if __name__ == "__main__":
#     # agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
#     import faulthandler
#     faulthandler.enable() # Enable before starting threads or complex imports
#     logger.info("--- __main__: faulthandler enabled ---")
#     logger.info("--- __main__: main.py script execution started ---")
#     logger.info("--- __main__: Creating worker thread (LiveKitAgentWorker) ---")
    
#     if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, OPENAI_API_KEY_ENV, ELEVENLABS_API_KEY_ENV]):
#         logger.critical("--- __main__: CRITICAL - One or more essential environment variables are missing. Application might not function correctly.")

#     worker_thread = threading.Thread(target=run_worker, name="LiveKitAgentWorker", daemon=False)
#     worker_thread.start()
#     logger.info("--- __main__: Worker thread started. It will attempt to connect the agent.")

#     logger.info("--- __main__: Waiting 5 seconds to observe worker thread startup logs before starting Uvicorn...")
#     time.sleep(5) 
    
#     if worker_thread.is_alive():
#         logger.info("--- __main__: Worker thread IS ALIVE after 5 seconds. Proceeding to start Uvicorn.")
#     else:
#         logger.error("--- __main__: Worker thread IS NOT ALIVE after 5 seconds. It likely exited. Check 'run_worker' logs for SystemExit or errors.")

#     logger.info("--- __main__: Starting Uvicorn server for FastAPI app ---")
#     try:
#         uvicorn.run(
#             "main:app",
#             host="0.0.0.0",
#             port=8000,
#             reload=False, 
#             log_level="info"
#         )
#     except Exception as e_uvicorn:
#         logger.error(f"--- __main__: Uvicorn server failed to start or crashed: {e_uvicorn}", exc_info=True)
    
#     logger.info("--- __main__: Uvicorn server has stopped. ---")

#     if worker_thread.is_alive():
#         logger.info("--- __main__: Uvicorn stopped. Worker thread is still alive. It might need to be manually stopped or will exit if agent job completes.")
#         # Note: For a truly clean shutdown, you might need to signal cli.run_app to stop,
#         # which typically happens via OS signals (Ctrl+C) that it handles.
#         # A simple .join() here might block indefinitely if cli.run_app isn't designed to exit
#         # programmatically without such a signal.
#         # worker_thread.join(timeout=10) 
#         # if worker_thread.is_alive():
#         #     logger.warning("--- __main__: Worker thread did not terminate after Uvicorn exit and join timeout.")
#         # else:
#         #     logger.info("--- __main__: Worker thread joined/finished successfully after Uvicorn exit.")
#     else:
#         logger.info("--- __main__: Worker thread was not alive after Uvicorn exit (it likely finished or errored earlier).")
    
#     logger.info("--- __main__: Script execution finished. ---")





























# import asyncio
# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import logging
# from datetime import timedelta
# # import jwt # Not used directly, AccessToken handles it
# import threading
# import time 
# import sys 
# from typing import Any, AsyncIterator # Added AsyncIterator for type hint

# # LiveKit specific imports
# from livekit.api import AccessToken, VideoGrants
# from livekit import agents, rtc # Added agents for llm.ChatMessage if needed later
# from livekit.agents import (
#     JobContext,
#     WorkerOptions,
#     cli,
#     Agent, 
#     AgentSession,
#     AutoSubscribe,
# )
# from livekit.agents.llm import LLMStream # LLMStream already imported

# # LiveKit Plugins
# from livekit.plugins import deepgram, openai, silero
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

# # Load environment variables at the very beginning
# load_dotenv()

# app = FastAPI()

# origins = [
#     "http://localhost:3000", 
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Configure logging (this should now take full effect for the agent too)
# logging.basicConfig(
#     level=logging.DEBUG, 
#     format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
#     handlers=[logging.StreamHandler(sys.stdout)] # Explicitly use stdout
# )
# logger = logging.getLogger(__name__) # For application-specific logs
# agent_logger = logging.getLogger("livekit.agents") # For logs from the livekit.agents library
# agent_logger.setLevel(logging.INFO) # Keep LiveKit agent library logs at INFO unless debugging it

# # Fetch environment variables
# LIVEKIT_URL = os.getenv("LIVEKIT_URL")
# LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
# LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
# OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY")
# ELEVENLABS_API_KEY_ENV = os.getenv("ELEVENLABS_API_KEY") # Not used in current agent, but good to have
# DEEPGRAM_API_KEY_ENV = os.getenv("DEEPGRAM_API_KEY")

# logger.info(f"MODULE LEVEL: LIVEKIT_URL: {LIVEKIT_URL}")
# logger.info(f"MODULE LEVEL: LIVEKIT_API_KEY IS SET: {bool(LIVEKIT_API_KEY)}")
# logger.info(f"MODULE LEVEL: DEEPGRAM_API_KEY IS SET: {bool(DEEPGRAM_API_KEY_ENV)}")
# logger.info(f"MODULE LEVEL: OPENAI_API_KEY IS SET: {bool(OPENAI_API_KEY_ENV)}")

# @app.get("/token")
# async def get_token():
#     logger.info("[TokenSvc] /token endpoint hit.")
#     if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
#         logger.error("[TokenSvc] LiveKit credentials not configured properly.")
#         raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

#     video_grant_obj = VideoGrants(
#         room_join=True,
#         room="voice-assistant-room", 
#         can_publish=True,
#         can_subscribe=True,
#         can_publish_data=True,
#     )

#     token_builder = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
#     token_builder.identity = "frontend_user" 
#     token_builder.name = "Frontend User"
#     token_builder.ttl = timedelta(hours=1)
#     token_builder.with_grants(video_grant_obj)
#     token_jwt = token_builder.to_jwt()

#     logger.info(f"[TokenSvc] Generated token for frontend_user to join 'voice-assistant-room'")
#     return {"token": token_jwt, "url": LIVEKIT_URL}


# # --- Define the Assistant Agent ---
# class GUIFocusedAssistant(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             instructions=(
#                 "You are a friendly voice assistant created by LiveKit, integrated into a web GUI. "
#                 "Your purpose is to engage in general conversation, answer questions, "
#                 "and provide helpful responses. Keep your answers concise, natural, "
#                 "and suitable for voice interaction. You are talking to a user through a web interface."
#             )
#         )

#     async def publish_llm_stream_to_user(self, llm_stream: LLMStream) -> None:
#         collected_text_for_frontend = ""
#         sentences_for_tts = []

#         async for sentence in self.llm_stream_sentences(llm_stream):
#             collected_text_for_frontend += sentence
#             sentences_for_tts.append(sentence)

#         # --- MODIFICATION: Explicit log for the model's complete response ---
#         if collected_text_for_frontend:
#             logger.info(f"MODEL_RESPONSE: '{collected_text_for_frontend}'")
#         else:
#             logger.info("MODEL_RESPONSE: (empty text)") # Log if response is empty

#         if collected_text_for_frontend and self.room:
#             # The log above is primary, this one is for data channel debugging if needed
#             # logger.info(f"[AssistantGUI] Publishing LLM response to frontend: '{collected_text_for_frontend}'") 
#             try:
#                 await self.room.local_participant.publish_data(
#                     f"response:{collected_text_for_frontend}", "response"
#                 )
#                 # logger.info(f"[AssistantGUI] Successfully published LLM response to frontend data channel.")
#             except Exception as e_publish:
#                 logger.error(f"[AssistantGUI] Error publishing LLM response data: {e_publish}", exc_info=True)

#         async def tts_sentence_generator():
#             for s in sentences_for_tts:
#                 yield s

#         if sentences_for_tts:
#             # The MODEL_RESPONSE log covers the content, this section logs TTS action
#             # logger.info(f"[AssistantGUI] Synthesizing audio for: '{collected_text_for_frontend}'")
#             try:
#                 await self.tts.synthesize(tts_sentence_generator())
#                 # logger.info(f"[AssistantGUI] Audio synthesis complete for: '{collected_text_for_frontend}'")
#             except Exception as e_tts:
#                 logger.error(f"[AssistantGUI] Error during TTS synthesis: {e_tts}", exc_info=True)
#         else:
#             logger.info("[AssistantGUI] No sentences from LLM to synthesize (response might have been empty).")

# # --- Agent Entrypoint ---
# async def entrypoint(ctx: JobContext):
#     room_name_from_ctx = ctx.room.name if ctx.room else "N/A_ROOM_IN_CTX"
#     job_id_from_ctx = ctx.job.id if ctx.job else "N/A_JOB_IN_CTX"
#     logger.info(f"--- [AgentEntry] AGENT LOGIC ENTRYPOINT CALLED --- Room: {room_name_from_ctx}, Job ID: {job_id_from_ctx}")
#     logger.info(f"[AgentEntry] Job received for room: {room_name_from_ctx}, job_id: {job_id_from_ctx}")

#     try:
#         logger.info(f"[AgentEntry] Ensuring connection and setting subscriptions for job {job_id_from_ctx}.")
#         await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY) 
#         logger.info(f"[AgentEntry] Connected/Verified. Participant SID: {ctx.room.local_participant.sid}")
#     except Exception as e_connect:
#         logger.error(f"[AgentEntry] Error during ctx.connect(): {e_connect}", exc_info=True)
#         raise 

#     try:
#         logger.info("[AgentEntry] Initializing plugins...")
#         stt = deepgram.STT(
#             model="nova-2",
#             language="multi",
#         )
#         llm = openai.LLM(
#             model="gpt-4o-mini",
#         )
#         tts = openai.TTS( 
#             voice="ash",
#         )
#         vad = silero.VAD.load() 
#         turn_detection = MultilingualModel() # This will now use the correct import
#         logger.info("[AgentEntry] Plugins initialized.")

#         # Function to publish STT results to the frontend
#         async def publish_stt_data(stt_event: Any, topic_prefix="transcription"):
#             if stt_event.alternatives and stt_event.alternatives[0].text:
#                 text = stt_event.alternatives[0].text
#                 # This log is about the action of publishing STT to frontend
#                 logger.info(f"[AgentEntry:{topic_prefix}] Attempting to publish STT to frontend: '{text}'")
#                 try:
#                     await ctx.room.local_participant.publish_data(f"transcription:{text}", "transcription")
#                     logger.info(f"[AgentEntry:{topic_prefix}] Successfully published STT to frontend: '{text}'")
#                 except Exception as e_publish_stt:
#                     logger.error(f"[AgentEntry] Error publishing STT data: {e_publish_stt}", exc_info=True)

#         # Define synchronous handlers that will schedule the async task
#         def stt_interim_handler(stt_event: Any):
#             # Keep debug for interim, as they can be noisy
#             logger.debug(f"[AgentEntry:STT_Handler] Interim event received: {stt_event.alternatives[0].text if stt_event.alternatives else 'No alternatives'}")
#             asyncio.create_task(publish_stt_data(stt_event, "STT_Interim"))

#         def stt_final_handler(stt_event: Any):
#             if stt_event.alternatives and stt_event.alternatives[0].text:
#                 text = stt_event.alternatives[0].text
#                 # --- MODIFICATION: Explicit log for the user's query ---
#                 logger.info(f"USER_QUERY: '{text}'")
#                 asyncio.create_task(publish_stt_data(stt_event, "STT_Final"))
#             else:
#                 # Changed from debug to info for visibility if this case occurs
#                 logger.info("[AgentEntry:STT_Handler] Final STT event received but no text or alternatives.")


#         # Attach event listeners to the STT plugin instance
#         stt.on("interim_transcript", stt_interim_handler) 
#         stt.on("final_transcript", stt_final_handler)
#         logger.info("[AgentEntry] STT event listeners attached to STT plugin instance.")

#         session = AgentSession(
#             stt=stt,
#             llm=llm,
#             tts=tts,
#             vad=vad,
#             turn_detection=turn_detection,
#         )
#         logger.info("[AgentEntry] AgentSession created.")

#         assistant_agent_instance = GUIFocusedAssistant()
#         logger.info("[AgentEntry] GUIFocusedAssistant instance created.")

#         logger.info("[AgentEntry] Starting AgentSession...")
#         main_agent_task = asyncio.create_task(session.start(
#             room=ctx.room, 
#             agent=assistant_agent_instance,
#         ))
#         logger.info("[AgentEntry] AgentSession.start() called and task created.")
        
#         # Wait a brief moment for the session to fully initialize before sending initial greeting
#         await asyncio.sleep(1) 
#         initial_greeting = "Hello! I'm your voice assistant, ready to chat. How can I help you today?"
#         logger.info(f"[AgentEntry] Generating initial reply (agent self-initiated): '{initial_greeting}'")
#         await session.generate_reply(instructions=initial_greeting) # This will also log MODEL_RESPONSE
#         logger.info("[AgentEntry] Initial reply generation requested.")

#         # Keep the session alive by waiting for the room to disconnect
#         while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
#             await asyncio.sleep(1)
#             logger.debug("[AgentEntry] Room still connected, keeping session alive...")

#         await main_agent_task # Ensure the agent task completes
#         logger.info("[AgentEntry] Main agent task completed.")

#     except Exception as e_session:
#         logger.error(f"[AgentEntry] Error during AgentSession setup or execution: {e_session}", exc_info=True)
#         raise
#     finally:
#         logger.info(f"[AgentEntry] Agent logic finished or terminated for job {job_id_from_ctx}.")

# def run_worker():
#     try:
#         logger.info("--- run_worker: Attempting to start LiveKit Agent Worker ---")
#         if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
#             logger.critical("!!! run_worker: CRITICAL - LiveKit connection credentials (URL, KEY, SECRET) NOT FOUND. Worker will NOT start.")
#             return
        
#         if not OPENAI_API_KEY_ENV: logger.warning("!!! run_worker: OpenAI API key not found. LLM/TTS plugins might fail.")
#         if not DEEPGRAM_API_KEY_ENV: logger.warning("!!! run_worker: Deepgram API key not found. STT plugin might fail.")

#         worker_options = WorkerOptions(
#             entrypoint_fnc=entrypoint,
#             # --- MODIFICATION: Prevent LiveKit from overriding our logging config ---
#         )
        
#         original_argv = list(sys.argv)
#         # This is a common way to invoke livekit-agent CLI programmatically
#         sys.argv = ['livekit_agent_script_embedded.py', 'start'] 
#         logger.info(f"run_worker: Modified sys.argv for cli.run_app: {sys.argv}")
        
#         logger.info("run_worker: Applying nest_asyncio and calling cli.run_app(worker_options)...")
#         try:
#             import nest_asyncio
#             nest_asyncio.apply()
#             logger.info("run_worker: Applied nest_asyncio.")
#             cli.run_app(worker_options) # This runs the agent worker
#         finally:
#             sys.argv = original_argv # Restore original sys.argv
#             logger.info(f"run_worker: Restored original sys.argv: {sys.argv}")

#         logger.info("--- run_worker: LiveKit Agent Worker `cli.run_app` finished. ---")

#     except SystemExit as se:
#         # cli.run_app can cause SystemExit on normal shutdown or CLI errors
#         logger.info(f"run_worker: cli.run_app exited with SystemExit (code: {se.code}). This is often normal.")
#     except Exception as e:
#         logger.error(f"!!! run_worker: CRITICAL ERROR during run_worker or cli.run_app: {e}", exc_info=True)
#     finally:
#         logger.info("--- run_worker: Function execution complete or terminated. ---")


# if __name__ == "__main__":
#     import faulthandler
#     faulthandler.enable() # Good for debugging low-level crashes
#     logger.info("--- __main__: main.py script execution started ---")

#     if len(sys.argv) > 1 and sys.argv[1] == "download-files":
#         logger.info("--- __main__: 'download-files' argument detected. ---")
#         logger.info("--- __main__: Plugins will initialize when the worker starts, triggering downloads if needed.")
#         # The script will proceed to start the worker and Uvicorn.
#         # For dedicated downloads, a simpler script just initializing plugins is better.
#         pass 

#     logger.info("--- __main__: Creating worker thread (LiveKitAgentWorker) ---")
#     # Running the agent worker in a separate thread
#     worker_thread = threading.Thread(target=run_worker, name="LiveKitAgentWorker", daemon=True) 
#     worker_thread.start()
#     logger.info("--- __main__: Worker thread started.")

#     logger.info("--- __main__: Waiting a few seconds for agent worker to initialize before starting Uvicorn...")
#     time.sleep(5) # Give the agent worker a moment to start up
    
#     if worker_thread.is_alive():
#         logger.info("--- __main__: Worker thread IS ALIVE. Proceeding to start Uvicorn.")
#     else:
#         logger.error("--- __main__: Worker thread IS NOT ALIVE. It likely exited. Check 'run_worker' logs for errors (e.g., missing credentials).")
#         # Optionally, you might want to exit here if the worker is critical and failed
#         # sys.exit(1) 

#     logger.info("--- __main__: Starting Uvicorn server for FastAPI app ---")
#     try:
#         uvicorn.run(
#             "main:app", 
#             host="0.0.0.0",
#             port=8000, 
#             reload=True, # Reload can sometimes have quirks with threads/subprocesses
#             log_level="info" # Uvicorn's own access/error logs
#         )
#     except Exception as e_uvicorn:
#         logger.error(f"--- __main__: Uvicorn server failed: {e_uvicorn}", exc_info=True)
    
#     logger.info("--- __main__: Uvicorn server has stopped. ---")
#     logger.info("--- __main__: Script execution finished. ---")





















# 15/05/25

# working model of english



# import asyncio
# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import logging
# from datetime import timedelta
# import threading
# import time 
# import sys 
# import psutil
# from typing import Any, AsyncIterator

# # LiveKit specific imports
# from livekit.api import AccessToken, VideoGrants
# from livekit import agents, rtc
# from livekit.agents import (
#     JobContext,
#     WorkerOptions,
#     cli,
#     Agent, 
#     AgentSession,
#     AutoSubscribe,
# )
# from livekit.agents.llm import LLMStream
# from livekit.agents import voice
# from livekit.agents import llm as livekit_llm 
# from livekit.agents.llm import ChatMessage, ChatContext
# import pydantic_core

# # LiveKit Plugins
# from livekit.plugins import deepgram, openai, silero

# # Load environment variables
# load_dotenv()

# app = FastAPI()

# origins = [
#     "http://localhost:3000", 
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Configure logging with console and file output
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler('agent_logs.log')
#     ]
# )
# logger = logging.getLogger(__name__)
# agent_logger = logging.getLogger("livekit.agents")
# agent_logger.setLevel(logging.DEBUG)

# # Fetch environment variables
# LIVEKIT_URL = os.getenv("LIVEKIT_URL")
# LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
# LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
# OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY")
# ELEVENLABS_API_KEY_ENV = os.getenv("ELEVENLABS_API_KEY")
# DEEPGRAM_API_KEY_ENV = os.getenv("DEEPGRAM_API_KEY")

# logger.info(f"MODULE LEVEL: LIVEKIT_URL: {LIVEKIT_URL}")
# logger.info(f"MODULE LEVEL: LIVEKIT_API_KEY IS SET: {bool(LIVEKIT_API_KEY)}")
# logger.info(f"MODULE LEVEL: DEEPGRAM_API_KEY IS SET: {bool(DEEPGRAM_API_KEY_ENV)}")
# logger.info(f"MODULE LEVEL: OPENAI_API_KEY IS SET: {bool(OPENAI_API_KEY_ENV)}")

# @app.get("/token")
# async def get_token():
#     logger.info("[TokenSvc] /token endpoint hit.")
#     if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
#         logger.error("[TokenSvc] LiveKit credentials not configured properly.")
#         raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

#     video_grant_obj = VideoGrants(
#         room_join=True,
#         room="voice-assistant-room", 
#         can_publish=True,
#         can_subscribe=True,
#         can_publish_data=True,
#     )

#     token_builder = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
#     token_builder.identity = "frontend_user" 
#     token_builder.name = "Frontend User"
#     token_builder.ttl = timedelta(hours=1)
#     token_builder.with_grants(video_grant_obj)
#     token_jwt = token_builder.to_jwt()

#     logger.info(f"[TokenSvc] Generated token for frontend_user to join 'voice-assistant-room'")
#     return {"token": token_jwt, "url": LIVEKIT_URL}

# # --- Define the Assistant Agent ---
# class GUIFocusedAssistant(Agent): 
#     def __init__(self) -> None:
#         super().__init__(
#             instructions=(
#                 "You are Echo, a friendly and helpful voice assistant. Respond concisely. "
#                 "When the user speaks, understand their query and provide a relevant answer."
#             )
#         )
#         self.last_user_query_for_log = None 
#         self.chat_history: list[ChatMessage] = [] 
#         logger.info("--- [GUIFocusedAssistant] __init__ CALLED ---")

#     async def on_transcript(self, transcript: str, is_final: bool) -> None:
#         logger.info(f"--- [GUIFocusedAssistant] on_transcript CALLED --- Final: {is_final}, Text: '{transcript}'")
#         if is_final and transcript:
#             self.last_user_query_for_log = transcript
            
#             logger.info(f"--- [GUIFocusedAssistant] on_transcript: User query received: '{transcript}'. Attempting to generate LLM response.")
            
#             if not hasattr(self, 'llm') or not self.llm:
#                 logger.error("--- [GUIFocusedAssistant] on_transcript: self.llm is not available. Cannot generate LLM response.")
#                 return

#             try:
#                 # Create ChatMessage for current user transcript
#                 # Content should be a list of strings for this SDK version based on previous Pydantic errors
#                 user_chat_message = ChatMessage(role="user", content=[transcript]) 
                
#                 current_chat_turn_history = list(self.chat_history)
#                 current_chat_turn_history.append(user_chat_message)

#                 if len(current_chat_turn_history) > 20: 
#                     current_chat_turn_history = current_chat_turn_history[-20:]

#                 # Create ChatContext
#                 chat_ctx_for_llm = ChatContext(messages=current_chat_turn_history)

#                 logger.info(f"--- [GUIFocusedAssistant] on_transcript: Calling self.llm.chat(chat_ctx=...) with history (count: {len(current_chat_turn_history)})")
                
#                 llm_stream = await self.llm.chat(chat_ctx=chat_ctx_for_llm) # Use chat_ctx=

#                 logger.info(f"--- [GUIFocusedAssistant] on_transcript: self.llm.chat() returned stream. Now calling self.handle_llm_response.")
#                 self.chat_history = current_chat_turn_history 
#                 await self.handle_llm_response(llm_stream)

#             except TypeError as te_chat: 
#                  logger.error(f"--- [GUIFocusedAssistant] on_transcript: TypeError calling self.llm.chat(chat_ctx=...): {te_chat}. 'chat_ctx' or ChatMessage format still incorrect.", exc_info=True)
#             except Exception as e_llm_call_or_handle:
#                  logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error in LLM call sequence or in handle_llm_response: {e_llm_call_or_handle}", exc_info=True)
        
#         if is_final and transcript: # Publish transcript to frontend
#             if self.room and self.room.local_participant:
#                 logger.info(f"--- [GUIFocusedAssistant] on_transcript: Publishing FINAL user transcript to frontend: '{transcript}'")
#                 try:
#                     await self.room.local_participant.publish_data(f"transcription:{transcript}", "transcription")
#                 except Exception as e_pub_tx:
#                     logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error publishing user transcript: {e_pub_tx}", exc_info=True)

#     async def handle_llm_response(self, llm_stream: LLMStream) -> None:
#         logger.info(f"--- [GUIFocusedAssistant] handle_llm_response CALLED ---")
#         collected_text_for_frontend = ""
#         temp_sentences_for_tts = [] 
#         stream_processed_successfully = False

#         try:
#             async for sentence_obj in self.llm_stream_sentences(llm_stream):
#                 logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Processing sentence: '{sentence_obj.text}'")
#                 if sentence_obj.text:
#                     collected_text_for_frontend += sentence_obj.text + " "
#                     temp_sentences_for_tts.append(sentence_obj.text)
#                 stream_processed_successfully = True
#         except Exception as e_llm_stream_processing:
#             logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error processing LLM stream: {e_llm_stream_processing}", exc_info=True)
#             if not collected_text_for_frontend: collected_text_for_frontend = "Error processing response."
#             else: collected_text_for_frontend += " (Error in stream)"
        
#         final_collected_text = collected_text_for_frontend.strip()

#         if final_collected_text: 
#             # Content should be a list of strings
#             assistant_chat_message = ChatMessage(role="assistant", content=[final_collected_text])
#             self.chat_history.append(assistant_chat_message)
#             if len(self.chat_history) > 20: self.chat_history = self.chat_history[-20:]

#         log_context_info = f"for user query '{self.last_user_query_for_log}'" if self.last_user_query_for_log else "(initial/agent-initiated)"
#         if final_collected_text: logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: FINAL MODEL_RESPONSE {log_context_info}: '{final_collected_text}'")
#         else: logger.warning(f"--- [GUIFocusedAssistant] handle_llm_response: No final text from LLM {log_context_info}.")

#         if final_collected_text and self.room and self.room.local_participant:
#              logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Publishing agent response to GUI: '{final_collected_text}'")
#              await self.room.local_participant.publish_data(payload=f"response:{final_collected_text}", topic="response")

#         if temp_sentences_for_tts and any(s.strip() for s in temp_sentences_for_tts) and hasattr(self, 'tts') and self.tts:
#             logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Attempting TTS.")
#             async def gen_tts():
#                 for s in temp_sentences_for_tts: yield s.strip()
#             await self.tts.synthesize(gen_tts())
        
#         self.last_user_query_for_log = None 
#         logger.info(f"--- [GUIFocusedAssistant] handle_llm_response FINISHED ---")


# # --- Agent Entrypoint ---
# async def entrypoint(ctx: JobContext):
#     logger.info(f"--- [AgentEntry] AGENT LOGIC ENTRYPOINT CALLED (ChatContext Deep Dive) --- Room: {ctx.room.name}, Job: {ctx.job.id}")

#     try:
#         await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
#         logger.info(f"[AgentEntry] Connected to room. Local Participant SID: {ctx.room.local_participant.sid}")
#     except Exception as e_connect:
#         logger.error(f"[AgentEntry] Fatal error during ctx.connect(): {e_connect}", exc_info=True); raise

#     logger.info("[AgentEntry] Initializing plugins...")
#     llm_plugin = None
#     try:
#         stt = deepgram.STT(model="nova-2", language="multi", interim_results=False)
#         if not OPENAI_API_KEY_ENV: raise ValueError("OpenAI API key not found for plugins.")
#         llm_plugin = openai.LLM(model="gpt-4o-mini", api_key=OPENAI_API_KEY_ENV) 
#         tts_plugin = openai.TTS(voice="ash", api_key=OPENAI_API_KEY_ENV)
#         vad = silero.VAD.load() 
#         logger.info("[AgentEntry] All plugins initialized successfully.")
#     except Exception as e_plugins:
#         logger.error(f"[AgentEntry] Error initializing plugins: {e_plugins}", exc_info=True); raise 

#     # --- LLM PLUGIN DIRECT TEST ---
#     logger.info("[AgentEntry] --- Testing LLM plugin directly START ---")
#     llm_test_passed = False
#     if not llm_plugin:
#         logger.error("[AgentEntry] LLM Plugin is None, cannot run direct test.")
#     else:
#         test_prompt_text = "Hello, LLM. Confirm you are working by saying OK."
        
#         # Step 1: Create a list of ChatMessage objects
#         # Based on previous logs, ChatMessage(role="user", content=[string]) is valid for ChatMessage creation.
#         chat_message_list: list[ChatMessage] = []
#         try:
#             chat_message_list = [ChatMessage(role="user", content=[test_prompt_text])]
#             logger.info(f"[AgentEntry] LLM Direct Test: Successfully created chat_message_list: {chat_message_list}")
#         except Exception as e_cm_create:
#             logger.error(f"[AgentEntry] LLM Direct Test: FAILED to create ChatMessage list: {e_cm_create}", exc_info=True)
#             chat_message_list = [] # Ensure it's an empty list if creation failed

#         if chat_message_list:
#             # Step 2: Try to create ChatContext
#             chat_ctx_for_test: ChatContext | None = None
#             try:
#                 logger.info(f"[AgentEntry] LLM Direct Test: Attempting ChatContext(chat_message_list) [POSITIONAL for messages]")
#                 chat_ctx_for_test = ChatContext(chat_message_list) # TRY POSITIONAL for messages list
#                 logger.info(f"[AgentEntry] LLM Direct Test: ChatContext created positionally: {chat_ctx_for_test}")
#             except TypeError as te_ctx_pos:
#                 logger.warning(f"[AgentEntry] LLM Direct Test: ChatContext(positional_list) FAILED (TypeError: {te_ctx_pos}). Trying ChatContext().")
#                 try:
#                     chat_ctx_for_test = ChatContext() # Try empty constructor
#                     logger.info(f"[AgentEntry] LLM Direct Test: ChatContext() created empty. Appending messages.")
#                     # If ChatContext has an 'messages' attribute that's a list, or an 'add_message' method
#                     if hasattr(chat_ctx_for_test, 'messages') and isinstance(chat_ctx_for_test.messages, list):
#                         for msg in chat_message_list:
#                             chat_ctx_for_test.messages.append(msg)
#                         logger.info(f"[AgentEntry] LLM Direct Test: Messages appended to chat_ctx_for_test.messages")
#                     elif hasattr(chat_ctx_for_test, 'add_message'):
#                          for msg in chat_message_list:
#                             chat_ctx_for_test.add_message(msg) # Assuming an add_message method
#                          logger.info(f"[AgentEntry] LLM Direct Test: Messages added via chat_ctx_for_test.add_message()")
#                     else:
#                         logger.warning("[AgentEntry] LLM Direct Test: Empty ChatContext created, but no obvious way to add messages to it.")
#                         chat_ctx_for_test = None # Mark as unusable
#                 except Exception as e_ctx_empty:
#                     logger.error(f"[AgentEntry] LLM Direct Test: Creating empty ChatContext or adding messages FAILED: {e_ctx_empty}", exc_info=True)
#                     chat_ctx_for_test = None
#             except Exception as e_ctx_other:
#                 logger.error(f"[AgentEntry] LLM Direct Test: Other error creating ChatContext: {e_ctx_other}", exc_info=True)
#                 chat_ctx_for_test = None

#             # Step 3: If ChatContext was created, try calling llm_plugin.chat()
#             if chat_ctx_for_test:
#                 try:
#                     logger.info(f"[AgentEntry] LLM Direct Test: Attempting llm_plugin.chat(chat_ctx=chat_ctx_for_test)")
#                     async for chunk in await llm_plugin.chat(chat_ctx=chat_ctx_for_test): 
#                         logger.info(f"[AgentEntry] LLM Direct Test Chunk (with chat_ctx=): '{getattr(chunk, 'text', str(chunk))}'")
#                         llm_test_passed = True; break 
#                     if llm_test_passed: logger.info("[AgentEntry] LLM plugin direct test with chat_ctx= SUCCEEDED.")
#                     else: logger.warning("[AgentEntry] LLM plugin direct test with chat_ctx=: no chunks.")
#                 except Exception as e_llm_call:
#                     logger.error(f"[AgentEntry] LLM plugin direct test with chat_ctx= FAILED: {e_llm_call}", exc_info=True)
#             else:
#                 logger.warning("[AgentEntry] LLM Direct Test: chat_ctx_for_test is None, skipping llm.chat() call.")
#         else:
#             logger.warning("[AgentEntry] LLM Direct Test: chat_message_list is empty, cannot proceed to create ChatContext.")
            
#     if not llm_test_passed: logger.critical("[AgentEntry] !!! LLM direct test FAILED. Agent may not function. !!!")
#     logger.info("[AgentEntry] --- Testing LLM plugin directly END ---")
#     # --- END OF LLM PLUGIN DIRECT TEST ---

#     session = AgentSession(stt=stt, llm=llm_plugin, tts=tts_plugin, vad=vad)
#     logger.info("[AgentEntry] AgentSession created.")
#     assistant_agent_instance = GUIFocusedAssistant() # Your class
#     logger.info("[AgentEntry] GUIFocusedAssistant instance created.")
#     logger.info("[AgentEntry] Starting AgentSession with GUIFocusedAssistant...")
#     main_agent_task = asyncio.create_task(session.start(room=ctx.room, agent=assistant_agent_instance))
#     logger.info("[AgentEntry] AgentSession.start() called. Session will manage greeting and turns.")

#     logger.info("[AgentEntry] Main agent logic running. Waiting for AgentSession task to complete.")
#     try:
#         await main_agent_task 
#     except asyncio.CancelledError: logger.info("[AgentEntry] Main agent_task was cancelled.")
#     except Exception as e_main_task: logger.error(f"[AgentEntry] Main agent_task exited with error: {e_main_task}", exc_info=True)
#     logger.info(f"[AgentEntry] Agent logic finished/terminated for job {ctx.job.id if ctx.job else 'N/A'}.")


# def check_and_free_port(port):
#     """Check if a port is in use and attempt to free it."""
#     for proc in psutil.process_iter(['pid', 'name']):
#         try:
#             for conn in proc.net_connections(kind='inet'):
#                 if conn.laddr.port == port:
#                     logger.warning(f"Port {port} is in use by PID {proc.pid} ({proc.name()}). Attempting to terminate.")
#                     proc.terminate()
#                     proc.wait(timeout=5)
#                     logger.info(f"Successfully terminated process on port {port}.")
#                     return
#         except (psutil.NoSuchProcess, psutil.AccessDenied):
#             continue
#     logger.debug(f"Port {port} is free or no process could be terminated.")

# def run_worker():
#     try:
#         logger.info("--- run_worker: Attempting to start LiveKit Agent Worker ---")
#         if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
#             logger.critical("!!! run_worker: CRITICAL - LiveKit connection credentials (URL, KEY, SECRET) NOT FOUND. Worker will NOT start.")
#             return
        
#         if not OPENAI_API_KEY_ENV: logger.warning("!!! run_worker: OpenAI API key not found. LLM/TTS plugins might fail.")
#         if not DEEPGRAM_API_KEY_ENV: logger.warning("!!! run_worker: Deepgram API key not found. STT plugin might fail.")

#         # Check and free port 8081 for LiveKit worker
#         check_and_free_port(8081)

#         worker_options = WorkerOptions(
#             entrypoint_fnc=entrypoint,
#         )
        
#         original_argv = list(sys.argv)
#         sys.argv = ['livekit_agent_script_embedded.py', 'start']
#         logger.info(f"run_worker: Modified sys.argv for cli.run_app: {sys.argv}")
        
#         logger.info("run_worker: Applying nest_asyncio and calling cli.run_app(worker_options)...")
#         try:
#             import nest_asyncio
#             nest_asyncio.apply()
#             logger.info("run_worker: Applied nest_asyncio.")
#             cli.run_app(worker_options)
#         finally:
#             sys.argv = original_argv
#             logger.info(f"run_worker: Restored original sys.argv: {sys.argv}")

#         logger.info("--- run_worker: LiveKit Agent Worker `cli.run_app` finished. ---")

#     except SystemExit as se:
#         logger.info(f"run_worker: cli.run_app exited with SystemExit (code: {se.code}). This is often normal.")
#     except Exception as e:
#         logger.error(f"!!! run_worker: CRITICAL ERROR during run_worker or cli.run_app: {e}", exc_info=True)
#     finally:
#         logger.info("--- run_worker: Function execution complete or terminated. ---")

# if __name__ == "__main__":
#     import faulthandler
#     faulthandler.enable()
#     logger.info("--- __main__: main.py script execution started ---")

#     if len(sys.argv) > 1 and sys.argv[1] == "download-files":
#         logger.info("--- __main__: 'download-files' argument detected. ---")
#         logger.info("--- __main__: Plugins will initialize when the worker starts, triggering downloads if needed.")
#         pass 

#     # Check and free port 8000 for Uvicorn
#     check_and_free_port(8000)

#     logger.info("--- __main__: Creating worker thread (LiveKitAgentWorker) ---")
#     worker_thread = threading.Thread(target=run_worker, name="LiveKitAgentWorker", daemon=True)
#     worker_thread.start()
#     logger.info("--- __main__: Worker thread started.")

#     logger.info("--- __main__: Waiting a few seconds for agent worker to initialize before starting Uvicorn...")
#     time.sleep(5)
    
#     if worker_thread.is_alive():
#         logger.info("--- __main__: Worker thread IS ALIVE. Proceeding to start Uvicorn.")
#     else:
#         logger.error("--- __main__: Worker thread IS NOT ALIVE. It likely exited. Check 'run_worker' logs for errors (e.g., missing credentials).")

#     logger.info("--- __main__: Starting Uvicorn server for FastAPI app ---")
#     try:
#         uvicorn.run(
#             "main:app",
#             host="0.0.0.0",
#             port=8000,
#             reload=False,
#             log_level="info"
#         )
#     except Exception as e_uvicorn:
#         logger.error(f"--- __main__: Uvicorn server failed: {e_uvicorn}", exc_info=True)
    
#     logger.info("--- __main__: Uvicorn server has stopped. ---")
#     logger.info("--- __main__: Script execution finished. ---")















 

#  hindi working model

# 15/05/25



# import asyncio
# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import logging
# import json
# from datetime import timedelta
# import threading
# import time
# import sys
# import psutil
# from typing import Any, AsyncIterator

# # LiveKit specific imports
# from livekit.api import AccessToken, VideoGrants
# from livekit import agents, rtc
# from livekit.agents import (
#     JobContext,
#     WorkerOptions,
#     cli,
#     Agent,
#     AgentSession,
#     AutoSubscribe,
# )
# from livekit.agents.llm import LLMStream
# from livekit.agents import voice
# from livekit.agents import llm as livekit_llm
# from livekit.agents.llm import ChatMessage, ChatContext
# import pydantic_core
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

# # LiveKit Plugins
# from livekit.plugins import deepgram, openai, silero

# # Load environment variables
# load_dotenv()

# app = FastAPI()

# origins = [
#     "http://localhost:3000",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Custom formatter to ensure Hindi text is not escaped
# class UnicodeSafeFormatter(logging.Formatter):
#     def format(self, record):
#         original_message = super().format(record)
#         try:
#             # If the message is JSON, re-dump with ensure_ascii=False
#             if record.name == "livekit.agents" and "{" in original_message:
#                 json_start = original_message.index("{")
#                 json_str = original_message[json_start:]
#                 json_data = json.loads(json_str)
#                 formatted_json = json.dumps(json_data, ensure_ascii=False)
#                 return original_message[:json_start] + formatted_json
#         except (ValueError, json.JSONDecodeError):
#             pass
#         return original_message

# # Configure logging with custom formatter
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler('agent_logs.log', encoding='utf-8')
#     ]
# )
# logger = logging.getLogger(__name__)
# agent_logger = logging.getLogger("livekit.agents")
# agent_logger.setLevel(logging.DEBUG)
# for handler in agent_logger.handlers:
#     handler.setFormatter(UnicodeSafeFormatter())

# def log_structured(event: str, content: str, language: str = "unknown"):
#     logger.info(json.dumps({
#         "event": event,
#         "language": language,
#         "content": content
#     }, ensure_ascii=False, indent=2))

# # Fetch environment variables
# LIVEKIT_URL = os.getenv("LIVEKIT_URL")
# LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
# LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
# OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY")
# ELEVENLABS_API_KEY_ENV = os.getenv("ELEVENLABS_API_KEY")
# DEEPGRAM_API_KEY_ENV = os.getenv("DEEPGRAM_API_KEY")

# logger.info(f"MODULE LEVEL: LIVEKIT_URL: {LIVEKIT_URL}")
# logger.info(f"MODULE LEVEL: LIVEKIT_API_KEY IS SET: {bool(LIVEKIT_API_KEY)}")
# logger.info(f"MODULE LEVEL: DEEPGRAM_API_KEY IS SET: {bool(DEEPGRAM_API_KEY_ENV)}")
# logger.info(f"MODULE LEVEL: OPENAI_API_KEY IS SET: {bool(OPENAI_API_KEY_ENV)}")

# @app.get("/token")
# async def get_token():
#     logger.info("[TokenSvc] /token endpoint hit.")
#     if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
#         logger.error("[TokenSvc] LiveKit credentials not configured properly.")
#         raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

#     video_grant_obj = VideoGrants(
#         room_join=True,
#         room="voice-assistant-room",
#         can_publish=True,
#         can_subscribe=True,
#         can_publish_data=True,
#     )

#     token_builder = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
#     token_builder.identity = "frontend_user"
#     token_builder.name = "Frontend User"
#     token_builder.ttl = timedelta(hours=1)
#     token_builder.with_grants(video_grant_obj)
#     token_jwt = token_builder.to_jwt()

#     logger.info(f"[TokenSvc] Generated token for frontend_user to join 'voice-assistant-room'")
#     return {"token": token_jwt, "url": LIVEKIT_URL}

# # --- Define the Assistant Agent ---
# class GUIFocusedAssistant(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             instructions=(
#                 "You are Echo, a friendly and helpful voice assistant. Respond concisely. "
#                 "When the user speaks, understand their query and provide a relevant answer."
#                 "**Always respond ONLY in the language the user used for their query.** "
#                 "आप एक सहायक AI ऑडियो असिस्टेंट हैं। केवल हिंदी में देवनागरी लिपि का उपयोग करके उत्तर दें। "
#                 "अपने उत्तर संक्षिप्त और मैत्रीपूर्ण रखें, और आपकी प्राथमिक भाषा हिंदी है। "
#                 "**If the user speaks in Hindi, respond ONLY in Hindi.** "
#                 "**If the user speaks in English, respond ONLY in English.** "
#                 "**Your primary language is Hindi, but strict language mirroring is required.**"
#             )
#         )
#         self.last_user_query_for_log = None
#         self.chat_history: list[ChatMessage] = []
#         logger.info("--- [GUIFocusedAssistant] __init__ CALLED ---")

#     async def on_transcript(self, transcript: str, is_final: bool) -> None:
#         logger.info(f"--- [GUIFocusedAssistant] on_transcript CALLED --- Final: {is_final}, Text: '{transcript}'")
        
#         if is_final and transcript:
#             self.last_user_query_for_log = transcript
#             log_structured("user_query", transcript, language="hi")
            
#             if not hasattr(self, 'llm') or not self.llm:
#                 logger.error("--- [GUIFocusedAssistant] on_transcript: LLM not available.")
#                 if hasattr(self, 'tts') and self.tts:
#                     async def gen_llm_error_fallback():
#                         yield "मुझे अभी भाषा मॉडल तक पहुंच नहीं है।"
#                     await self.tts.synthesize(gen_llm_error_fallback())
#                 return

#             if not hasattr(self, 'tts') or not self.tts:
#                 logger.error("--- [GUIFocusedAssistant] on_transcript: TTS not available.")
            
#             try:
#                 user_chat_message = ChatMessage(role="user", content=[transcript])
#                 current_chat_turn_history = list(self.chat_history)
#                 current_chat_turn_history.append(user_chat_message)
#                 if len(current_chat_turn_history) > 20:
#                     current_chat_turn_history = current_chat_turn_history[-20:]
                
#                 chat_ctx_for_llm = ChatContext(current_chat_turn_history)
#                 logger.debug(f"--- [GUIFocusedAssistant] on_transcript: Calling LLM with {len(current_chat_turn_history)} messages.")
                
#                 llm_stream = await self.llm.chat(chat_ctx_for_llm)
#                 self.chat_history = current_chat_turn_history
#                 await self.handle_llm_response(llm_stream)
                
#             except Exception as e_llm_call_or_handle:
#                 logger.error(f"--- [GUIFocusedAssistant] on_transcript: LLM processing error: {e_llm_call_or_handle}", exc_info=True)
#                 if hasattr(self, 'tts') and self.tts:
#                     async def gen_process_error_fallback():
#                         yield "आपके अनुरोध को संसाधित करने में त्रुटि हुई। कृपया पुनः प्रयास करें।"
#                     await self.tts.synthesize(gen_process_error_fallback())
            
#             if self.room and self.room.local_participant:
#                 try:
#                     await self.room.local_participant.publish_data(f"transcription:{transcript}", "transcription")
#                     logger.info(f"--- [GUIFocusedAssistant] on_transcript: Published user transcript: '{transcript}'")
#                 except Exception as e_pub_tx:
#                     logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error publishing transcript: {e_pub_tx}", exc_info=True)

#     async def handle_llm_response(self, llm_stream: LLMStream) -> None:
#         logger.info(f"--- [GUIFocusedAssistant] handle_llm_response CALLED ---")
#         collected_text_from_sentences = ""
#         sentences_for_tts_final = []
#         stream_processed_successfully = False

#         try:
#             logger.debug("--- [GUIFocusedAssistant] handle_llm_response: Starting sentence processing for TTS.")
#             async for sentence_obj in self.llm_stream_sentences(llm_stream):
#                 text_sentence = getattr(sentence_obj, 'text', str(sentence_obj))
#                 logger.debug(f"--- [GUIFocusedAssistant] handle_llm_response: Processed sentence chunk: '{text_sentence}'")
                
#                 if text_sentence:
#                     collected_text_from_sentences += text_sentence + " "
#                     sentences_for_tts_final.append(text_sentence)
#                     stream_processed_successfully = True
#                 else:
#                     logger.warning(f"--- [GUIFocusedAssistant] handle_llm_response: Empty sentence received from LLM stream.")

#         except Exception as e_llm_stream_processing:
#             logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error processing LLM stream: {e_llm_stream_processing}", exc_info=True)
#             error_text = "प्रतिक्रिया उत्पन्न करने में त्रुटि।"
#             if collected_text_from_sentences:
#                 error_text = collected_text_from_sentences + " (Stream processing error)"
#             collected_text_from_sentences = error_text
#             if not sentences_for_tts_final:
#                 sentences_for_tts_final.append(error_text)

#         finally:
#             logger.debug("--- [GUIFocusedAssistant] handle_llm_response: Finished LLM stream processing.")

#         final_collected_text = collected_text_from_sentences.strip()
#         logger.debug(f"--- [GUIFocusedAssistant] handle_llm_response: Final collected text: '{final_collected_text}'")
#         logger.debug(f"--- [GUIFocusedAssistant] handle_llm_response: Sentences for TTS: {sentences_for_tts_final}")

#         log_context_info = f"for user query '{self.last_user_query_for_log}'" if self.last_user_query_for_log else "(initial/agent-initiated)"

#         if final_collected_text:
#             assistant_chat_message = ChatMessage(role="assistant", content=[final_collected_text])
#             self.chat_history.append(assistant_chat_message)
#             if len(self.chat_history) > 20:
#                 self.chat_history = self.chat_history[-20:]
#             log_structured("assistant_response", final_collected_text, language="hi")
#             logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Logged assistant response: '{final_collected_text}'")
#         else:
#             fallback_msg = "कोई उत्तर उत्पन्न नहीं हुआ।"
#             log_structured("assistant_response", fallback_msg, language="hi")
#             logger.warning(f"--- [GUIFocusedAssistant] handle_llm_response: No response generated {log_context_info}.")

#         if self.room and self.room.local_participant:
#             payload_to_publish = f"response:{final_collected_text}" if final_collected_text else "response:कोई उत्तर उत्पन्न नहीं हुआ।"
#             logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Publishing response: '{payload_to_publish}'")
#             try:
#                 await self.room.local_participant.publish_data(payload=payload_to_publish, topic="response")
#             except Exception as e_pub_resp:
#                 logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error publishing response: {e_pub_resp}", exc_info=True)

#         if sentences_for_tts_final and any(s.strip() for s in sentences_for_tts_final):
#             logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Synthesizing TTS for {len(sentences_for_tts_final)} sentences.")
#             async def gen_tts():
#                 for s in sentences_for_tts_final:
#                     yield s.strip()
#             try:
#                 await self.tts.synthesize(gen_tts())
#                 logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: TTS synthesis completed.")
#             except Exception as e_tts:
#                 logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: TTS synthesis error: {e_tts}", exc_info=True)
#         else:
#             logger.warning("--- [GUIFocusedAssistant] handle_llm_response: No text for TTS synthesis.")

#         self.last_user_query_for_log = None
#         logger.info(f"--- [GUIFocusedAssistant] handle_llm_response FINISHED ---")

# # --- Agent Entrypoint ---
# async def entrypoint(ctx: JobContext):
#     logger.info(f"--- [AgentEntry] AGENT LOGIC ENTRYPOINT CALLED --- Room: {ctx.room.name}, Job: {ctx.job.id}")
    
#     try:
#         await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
#         logger.info(f"[AgentEntry] Connected to room. Local Participant SID: {ctx.room.local_participant.sid}")
#     except Exception as e_connect:
#         logger.critical(f"[AgentEntry] Fatal error during ctx.connect(): {e_connect}", exc_info=True)
#         raise
    
#     logger.info("[AgentEntry] Initializing plugins...")
#     stt_plugin = None
#     llm_plugin = None
#     tts_plugin = None
#     vad_plugin = None
#     turn_detector = None
    
#     try:
#         turn_detector = None
#         logger.info("[AgentEntry] Using no turn detector (temporary workaround for Hindi).")
        
#         stt_plugin = deepgram.STT(model="nova-2", language="hi", interim_results=True)
#         logger.info("[AgentEntry] Deepgram STT plugin initialized.")
        
#         if not OPENAI_API_KEY_ENV:
#             logger.error("[AgentEntry] OpenAI API key not found.")
#             raise ValueError("OPENAI_API_KEY environment variable is not set.")
#         llm_plugin = openai.LLM(model="gpt-4o-mini", api_key=OPENAI_API_KEY_ENV)
#         logger.info("[AgentEntry] OpenAI LLM plugin initialized.")
        
#         tts_plugin = openai.TTS(voice="ash", api_key=OPENAI_API_KEY_ENV)
#         logger.info("[AgentEntry] OpenAI TTS plugin initialized.")
        
#         vad_plugin = silero.VAD.load()
#         logger.info("[AgentEntry] Silero VAD plugin initialized.")
        
#         logger.info("[AgentEntry] All essential plugins initialized successfully.")
    
#     except Exception as e_plugins:
#         logger.critical(f"[AgentEntry] CRITICAL ERROR initializing plugins: {e_plugins}", exc_info=True)
#         if ctx.room and ctx.room.local_participant:
#             try:
#                 await ctx.room.local_participant.publish_data(payload=f"error:Agent failed to initialize plugins: {e_plugins}", topic="error")
#             except Exception:
#                 pass
#         raise
    
#     logger.info("[AgentEntry] Skipping problematic LLM direct test.")
    
#     if stt_plugin and llm_plugin and tts_plugin and vad_plugin:
#         session = AgentSession(stt=stt_plugin, llm=llm_plugin, tts=tts_plugin, vad=vad_plugin, turn_detection=turn_detector)
#         logger.info("[AgentEntry] AgentSession created with initialized plugins.")
        
#         assistant_agent_instance = GUIFocusedAssistant()
#         logger.info("[AgentEntry] GUIFocusedAssistant instance created.")
        
#         logger.info("[AgentEntry] Starting AgentSession with GUIFocusedAssistant...")
#         main_agent_task = asyncio.create_task(session.start(room=ctx.room, agent=assistant_agent_instance))
#         logger.info("[AgentEntry] AgentSession.start() called.")
        
#         try:
#             await main_agent_task
#             logger.info("[AgentEntry] Main agent_task completed normally.")
#         except asyncio.CancelledError:
#             logger.info("[AgentEntry] Main agent_task was cancelled.")
#         except Exception as e_main_task:
#             logger.error(f"[AgentEntry] Main agent_task error: {e_main_task}", exc_info=True)
#     else:
#         logger.critical("[AgentEntry] Cannot start AgentSession: missing plugins.")
    
#     logger.info(f"[AgentEntry] Agent logic finished for job {ctx.job.id if ctx.job else 'N/A'}.")

# def check_and_free_port(port):
#     for proc in psutil.process_iter(['pid', 'name']):
#         try:
#             for conn in proc.net_connections(kind='inet'):
#                 if conn.laddr.port == port:
#                     logger.warning(f"Port {port} is in use by PID {proc.pid} ({proc.name()}). Attempting to terminate.")
#                     try:
#                         proc.terminate()
#                         proc.wait(timeout=5)
#                         logger.info(f"Successfully terminated process on port {port}.")
#                     except psutil.TimeoutExpired:
#                         logger.warning(f"Process PID {proc.pid} did not terminate within timeout, killing.")
#                         proc.kill()
#                         logger.info(f"Successfully killed process on port {port}.")
#                     return
#         except (psutil.NoSuchProcess, psutil.AccessDenied):
#             continue
#     logger.debug(f"Port {port} is free or no process could be terminated.")

# def run_worker():
#     try:
#         logger.info("--- run_worker: Attempting to start LiveKit Agent Worker ---")
#         if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
#             logger.critical("!!! run_worker: CRITICAL - LiveKit connection credentials not found.")
#             return

#         if not OPENAI_API_KEY_ENV:
#             logger.warning("!!! run_worker: OpenAI API key not found.")
#         if not DEEPGRAM_API_KEY_ENV:
#             logger.warning("!!! run_worker: Deepgram API key not found.")

#         check_and_free_port(8081)

#         worker_options = WorkerOptions(
#             entrypoint_fnc=entrypoint,
#         )

#         original_argv = list(sys.argv)
#         sys.argv = [original_argv[0], 'start'] if len(original_argv) > 0 else ['livekit_agent_script_embedded.py', 'start']
#         logger.info(f"run_worker: Modified sys.argv for cli.run_app: {sys.argv}")

#         logger.info("run_worker: Applying nest_asyncio and calling cli.run_app...")
#         try:
#             import nest_asyncio
#             nest_asyncio.apply()
#             logger.info("run_worker: Applied nest_asyncio.")
#             cli.run_app(worker_options)
#         except SystemExit as se:
#             if se.code != 0:
#                 logger.error(f"run_worker: cli.run_app exited with non-zero status code {se.code}.", exc_info=True)
#             else:
#                 logger.info(f"run_worker: cli.run_app exited successfully (code: {se.code}).")
#         except Exception as e:
#             logger.error(f"!!! run_worker: CRITICAL ERROR during cli.run_app: {e}", exc_info=True)
#         finally:
#             sys.argv = original_argv
#             logger.info(f"run_worker: Restored original sys.argv.")

#         logger.info("--- run_worker: LiveKit Agent Worker `cli.run_app` finished. ---")

#     except Exception as e_outer:
#         logger.error(f"!!! run_worker: CRITICAL ERROR in outer run_worker block: {e_outer}", exc_info=True)
#     finally:
#         logger.info("--- run_worker: Function execution complete or terminated. ---")

# if __name__ == "__main__":
#     import faulthandler
#     faulthandler.enable()
#     logger.info("--- __main__: main.py script execution started ---")

#     check_and_free_port(8000)

#     logger.info("--- __main__: Creating worker thread (LiveKitAgentWorker) ---")
#     worker_thread = threading.Thread(target=run_worker, name="LiveKitAgentWorker", daemon=True)
#     worker_thread.start()
#     logger.info("--- __main__: Worker thread started.")

#     logger.info("--- __main__: Waiting a few seconds for agent worker to initialize before starting Uvicorn...")
#     time.sleep(5)

#     if worker_thread.is_alive():
#         logger.info("--- __main__: Worker thread IS ALIVE. Proceeding to start Uvicorn.")
#     else:
#         logger.critical("--- __main__: Worker thread IS NOT ALIVE. Agent will not function.")
#         # Consider exiting if critical
#         # sys.exit(1)

#     logger.info("--- __main__: Starting Uvicorn server for FastAPI app ---")
#     try:
#         uvicorn.run(
#             "main:app",
#             host="0.0.0.0",
#             port=8000,
#             reload=False,
#             log_level="info"
#         )
#     except Exception as e_uvicorn:
#         logger.error(f"--- __main__: Uvicorn server failed: {e_uvicorn}", exc_info=True)

#     logger.info("--- __main__: Uvicorn server has stopped. ---")
#     logger.info("--- __main__: Script execution finished. ---")
























# 17-05-25

#  multilanguage support support 



# --- START OF FILE main.py 

# import asyncio
# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import logging
# from datetime import timedelta
# import threading
# import time
# import sys
# import psutil
# from typing import Any, AsyncIterator, Dict, Optional

# # LiveKit specific imports
# from livekit.api import AccessToken, VideoGrants
# from livekit import agents, rtc
# from livekit.agents import (
#     JobContext,
#     WorkerOptions,
#     cli,
#     Agent,
#     AgentSession,
#     AutoSubscribe,
# )
# from livekit.agents.llm import LLMStream
# # Corrected: We still import `voice` but don't use TextPrompt or initial_prompts here
# from livekit.agents import voice
# from livekit.agents import llm as livekit_llm
# from livekit.agents.llm import ChatMessage, ChatContext
# import pydantic_core
# from livekit.plugins.turn_detector.multilingual import MultilingualModel


# # LiveKit Plugins
# from livekit.plugins import deepgram, openai, silero

# # Load environment variables
# load_dotenv()

# app = FastAPI()

# origins = [
#     "http://localhost:3000",
#     "http://127.0.0.1:3000", # Also allow localhost explicitly
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Configure logging with console and file output
# logging.basicConfig(
#     level=logging.DEBUG, # Keep at DEBUG for now to see all our debug logs
#     format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler('agent_logs.log', encoding='utf-8') # Ensure UTF-8 for Hindi
#     ]
# )
# logger = logging.getLogger(__name__)
# agent_logger = logging.getLogger("livekit.agents")
# agent_logger.setLevel(logging.DEBUG) # Ensure LiveKit's internal debug logs are also shown

# import json

# def log_structured(event: str, content: str, language: str = "unknown", **kwargs):
#     """Logs structured data as JSON."""
#     log_data: Dict[str, Any] = {
#         "event": event,
#         "language": language,
#         "content": content
#     }
#     log_data.update(kwargs)
#     try:
#         # Attempt to log JSON, handle potential non-serializable objects if any get into kwargs
#         logger.info(json.dumps(log_data, ensure_ascii=False, indent=2))
#     except Exception as e:
#         logger.error(f"Failed to log structured data: {e}. Data: {log_data}", exc_info=True)
#         # Fallback to basic log
#         logger.info(f"Event: {event}, Lang: {language}, Content: {content}, Details: {kwargs}")


# # Fetch environment variables
# LIVEKIT_URL = os.getenv("LIVEKIT_URL")
# LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
# LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
# OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY")
# ELEVENLABS_API_KEY_ENV = os.getenv("ELEVENLABS_API_KEY") # Not used in this version but kept
# DEEPGRAM_API_KEY_ENV = os.getenv("DEEPGRAM_API_KEY") # Required for multi-language STT

# # Basic check for essential keys
# if not LIVEKIT_URL or not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
#      logger.critical("!!! CRITICAL: LiveKit credentials (URL, KEY, SECRET) not fully configured. Agent Worker may fail.")
# if not OPENAI_API_KEY_ENV:
#     logger.warning("!!! WARNING: OpenAI API key not found. LLM/TTS plugins may fail.")
# if not DEEPGRAM_API_KEY_ENV:
#     # Deepgram is essential for multi-language STT
#     logger.critical("!!! CRITICAL: DEEPGRAM_API_KEY environment variable is not set. Multi-language STT will fail.")


# @app.get("/token")
# async def get_token():
#     """Generates a LiveKit token for frontend user."""
#     logger.info("[TokenSvc] /token endpoint hit.")
#     if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
#         logger.error("[TokenSvc] LiveKit credentials not configured properly.")
#         raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

#     video_grant_obj = VideoGrants(
#         room_join=True,
#         room="voice-assistant-room", # Keep room name consistent
#         can_publish=True,
#         can_subscribe=True,
#         can_publish_data=True,
#     )

#     token_builder = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
#     token_builder.identity = "frontend_user"
#     token_builder.name = "Frontend User"
#     token_builder.ttl = timedelta(hours=1)
#     token_builder.with_grants(video_grant_obj)
#     token_jwt = token_builder.to_jwt()

#     logger.info(f"[TokenSvc] Generated token for frontend_user to join 'voice-assistant-room'")
#     return {"token": token_jwt, "url": LIVEKIT_URL}

# # --- Define the Assistant Agent ---
# class MultilingualAssistant(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             # Updated instructions to rely on the LLM's ability to infer and mirror language
#             instructions=(
#                 "You are Echo, a friendly and helpful voice assistant for LiveKit. "
#                 "Your primary goal is to understand the user's query and provide a concise, relevant answer. "
#                 "**CRITICAL INSTRUCTION: Respond ONLY in the language that the user used for their query.** "
#                 "Infer the user's language from their message text and the conversation history. "
#                 "If the user speaks in Hindi, respond in Hindi (using Devnagari script). If the user speaks in English, respond in English. "
#                 "If they mix languages, try to respond primarily in the language they used most in their last message. "
#                 "Keep responses brief and natural for voice communication."
#                 "Example: If user says 'नमस्ते, कैसे हो?', respond in Hindi. If user says 'Hello, how are you?', respond in English."
#                 "Do NOT translate the user's query unless explicitly asked. Focus on answering the query in their language."
#                 "Ensure your Hindi responses use the Devnagari script correctly."
#             )
#         )
#         # No longer storing last_user_query_language here, as it's not directly available
#         # in the on_transcript signature. Rely on LLM inference + pipeline magic.
#         self.last_user_query_text: Optional[str] = None # Store text for logging context
#         self.chat_history: list[ChatMessage] = [] # Stores history for LLM context (text only)
#         logger.info("--- [MultilingualAssistant] __init__ CALLED ---")

#     # CORRECTED SIGNATURE: transcript is a string, not a Transcript object
#     async def on_transcript(self, transcript: str, is_final: bool) -> None:
#         # Log every transcript chunk for debugging
#         # Note: The *language code* is NOT available directly here with this signature.
#         # The voice pipeline handles passing the detected language to the LLM call internally.
#         logger.debug(f"--- [MultilingualAssistant] on_transcript CALLED --- Final: {is_final}, Text: '{transcript}'")

#         if is_final and transcript:
#             self.last_user_query_text = transcript # Store final text for logging context

#             # We log the final user query here. Log with 'detected'
#             log_structured("user_query_final", transcript, language="detected")


#             logger.info(f"--- [MultilingualAssistant] on_transcript: Final user query received: '{transcript}'. Attempting to generate LLM response.")

#             # Check if llm plugin is available
#             if not hasattr(self, 'llm') or not self.llm:
#                 logger.error("--- [MultilingualAssistant] on_transcript: self.llm is not available. Cannot generate LLM response.")
#                 await self._synthesize_fallback("I'm sorry, I cannot access the language model right now.") # Synthesize fallback
#                 return

#             # Check if tts plugin is available - needed for handle_llm_response
#             if not hasattr(self, 'tts') or not self.tts:
#                  logger.error("--- [MultilingualAssistant] on_transcript: self.tts is not available. Cannot synthesize response.")
#                  # Continue processing LLM if possible for logging/frontend, but speech will fail.


#             try:
#                 # Create ChatMessage for current user transcript (using the text string)
#                 user_chat_message = ChatMessage(role="user", content=[transcript])

#                 # Prepare history for the current LLM turn
#                 # The AgentSession's voice pipeline is responsible for potentially
#                 # adding language hints or structuring the prompt further based on
#                 # the STT-detected language *before* calling self.llm.chat.
#                 # We provide the base history + current user message.
#                 current_chat_turn_history = list(self.chat_history) # Work on a copy
#                 current_chat_turn_history.append(user_chat_message)

#                 # Prune history if it's getting too long (e.g., keep last 20 messages)
#                 if len(current_chat_turn_history) > 20:
#                     current_chat_turn_history = current_chat_turn_history[-20:]
#                 logger.debug(f"--- [MultilingualAssistant] on_transcript: Chat history length for LLM call: {len(current_chat_turn_history)}")

#                 # Create ChatContext from the prepared history
#                 logger.debug(f"--- [MultilingualAssistant] on_transcript: Creating ChatContext for LLM call...")
#                 chat_ctx_for_llm = ChatContext(current_chat_turn_history) # Pass messages list positionally
#                 logger.debug(f"--- [MultilingualAssistant] on_transcript: ChatContext created.")

#                 logger.debug(f"--- [MultilingualAssistant] on_transcript: Calling self.llm.chat(...) with context...")
#                 # Call self.llm.chat - LLM is expected to respect instructions and history language
#                 llm_stream = await self.llm.chat(chat_ctx_for_llm)
#                 logger.debug(f"--- [MultilingualAssistant] on_transcript: self.llm.chat() returned stream.")

#                 # Update agent's persistent chat history *after* a successful LLM chat call returns a stream
#                 self.chat_history.append(user_chat_message) # Add user message to persistent history
#                 if len(self.chat_history) > 20: # Keep persistent history reasonable
#                      self.chat_history = self.chat_history[-20:]

#                 # Pass the stream to the handler
#                 await self.handle_llm_response(llm_stream)

#             except Exception as e_llm_call_or_handle:
#                  # Catching broader Exception for safety in the main turn processing
#                  logger.error(f"--- [MultilingualAssistant] on_transcript: Error during LLM turn processing or handle_llm_response: {e_llm_call_or_handle}", exc_info=True)
#                  # Optionally, synthesize a fallback message on LLM processing error
#                  await self._synthesize_fallback("I encountered an error while processing your request. Please try again.")


#         # This publishes the user's transcript to the frontend
#         # We don't have the specific language code here anymore, sending text only.
#         if is_final and transcript:
#             if self.room and self.room.local_participant:
#                 try:
#                     # Payload format for frontend - include language as "detected"
#                     payload = json.dumps({"type": "transcription", "text": transcript, "language": "detected"}, ensure_ascii=False)
#                     await self.room.local_participant.publish_data(payload=payload, topic="transcript_update")
#                     logger.debug(f"--- [MultilingualAssistant] on_transcript: Published user transcript to frontend: {payload}")
#                 except Exception as e_pub_tx:
#                     logger.error(f"--- [MultilingualAssistant] on_transcript: Error publishing user transcript: {e_pub_tx}", exc_info=True)


#     async def handle_llm_response(self, llm_stream: LLMStream) -> None:
#         logger.info(f"--- [MultilingualAssistant] handle_llm_response CALLED ---")
#         collected_text_from_sentences = ""
#         sentences_for_tts_final = []
#         stream_processed_successfully = False # Flag to indicate if stream yielded content

#         try:
#             logger.debug("--- [MultilingualAssistant] handle_llm_response: Starting sentence processing for TTS.")
#             # Iterate over the stream broken into sentences by AgentSession/voice pipeline helper
#             async for sentence_obj in self.llm_stream_sentences(llm_stream):
#                  # The stream yields objects, access the 'text' attribute
#                  text_sentence = getattr(sentence_obj, 'text', '') # Get text safely

#                  if text_sentence:
#                     # Log each sentence chunk as DEBUG - this helps see text arriving
#                     logger.debug(f"--- [MultilingualAssistant] handle_llm_response: Processed sentence chunk for TTS: '{text_sentence}'")
#                     # Collect text for the final log and frontend data
#                     collected_text_from_sentences += text_sentence + " "
#                     # Collect sentences for the final TTS call
#                     sentences_for_tts_final.append(text_sentence)
#                     stream_processed_successfully = True # At least one sentence was processed
#                  else:
#                     logger.debug("--- [MultilingualAssistant] handle_llm_response: Received empty sentence chunk from stream.")


#         except Exception as e_llm_stream_processing:
#             logger.error(f"--- [MultilingualAssistant] handle_llm_response: Error processing LLM stream into sentences: {e_llm_stream_processing}", exc_info=True)
#             # If stream processing failed, add a fallback message to TTS/log list
#             error_text = "Error generating response."
#             if collected_text_from_sentences:
#                  error_text = collected_text_from_sentences + " (Error in stream processing)"

#             collected_text_from_sentences = error_text # Update collected text to include error info
#             # Add error text for TTS if nothing else was collected
#             if not sentences_for_tts_final:
#                  sentences_for_tts_final.append(error_text)


#         finally:
#              logger.debug("--- [MultilingualAssistant] handle_llm_response: Finished LLM stream processing.")


#         # This is the final text that should be logged and sent to the frontend
#         final_collected_text = collected_text_from_sentences.strip()

#         logger.debug(f"--- [MultilingualAssistant] handle_llm_response: Trimmed final collected text: '{final_collected_text}'")
#         logger.debug(f"--- [MultilingualAssistant] handle_llm_response: Sentences prepared for TTS (count {len(sentences_for_tts_final)}): {sentences_for_tts_final}")


#         log_context_info = f"for user query '{self.last_user_query_text}'" if self.last_user_query_text else "(initial/agent-initiated)"

#         # Check if we have *any* text to log/publish (could be a successful response or an error message)
#         if final_collected_text:
#             # Add the agent's response to chat history
#             assistant_chat_message = ChatMessage(role="assistant", content=[final_collected_text])
#             self.chat_history.append(assistant_chat_message)
#             if len(self.chat_history) > 20: # Keep persistent history reasonable
#                  self.chat_history = self.chat_history[-20:]

#             # Log the FINAL collected text. We don't have the specific output language code here easily.
#             # Assume it matches the intended language based on LLM instructions.
#             log_structured("assistant_response_final", final_collected_text,
#                            language="inferred", # Indicate language is inferred or based on LLM's choice
#                            user_query=self.last_user_query_text)
#         else:
#             # Log if no text was collected/generated
#             log_structured("assistant_response_empty", "No final text collected from LLM stream.",
#                            language="none",
#                            user_query=self.last_user_query_text)
#             logger.warning(f"--- [MultilingualAssistant] handle_llm_response: No final text collected from LLM stream {log_context_info}.")


#         # Attempt TTS synthesis ONLY if there's text prepared for it
#         if sentences_for_tts_final and any(s.strip() for s in sentences_for_tts_final):
#             logger.info(f"--- [MultilingualAssistant] handle_llm_response: Attempting TTS synthesis for {len(sentences_for_tts_final)} sentences.")
#             # The generator yields each sentence to the TTS plugin
#             async def gen_tts():
#                 for s in sentences_for_tts_final:
#                     stripped_s = s.strip()
#                     if stripped_s:
#                        yield stripped_s # Yield stripped sentences
#             try:
#                 # The TTS plugin (OpenAI with voice='shimmer') will attempt to synthesize
#                 # the text, handling different languages as best as possible with the chosen voice.
#                 await self.tts.synthesize(gen_tts())
#                 logger.info(f"--- [MultilingualAssistant] handle_llm_response: TTS synthesis initiated successfully.")
#             except Exception as e_tts:
#                 logger.error(f"--- [MultilingualAssistant] handle_llm_response: Error during TTS synthesis: {e_tts}", exc_info=True)
#         else:
#             logger.warning("--- [MultilingualAssistant] handle_llm_response: No text prepared for TTS synthesis.")

#         # Reset state for the next turn
#         self.last_user_query_text = None
#         logger.info(f"--- [MultilingualAssistant] handle_llm_response FINISHED ---")

#     async def _synthesize_fallback(self, message: str) -> None:
#         """Synthesizes a simple text message if something goes wrong."""
#         if hasattr(self, 'tts') and self.tts:
#             logger.info(f"--- [MultilingualAssistant] _synthesize_fallback: Attempting TTS for fallback message: '{message}'")
#             async def gen_fallback():
#                 yield message
#             try:
#                 # Use the fallback message in English, assuming that's the lowest common denominator
#                 await self.tts.synthesize(gen_fallback())
#                 logger.info("--- [MultilingualAssistant] _synthesize_fallback: Fallback TTS initiated.")
#             except Exception as e_tts_fb:
#                 logger.error(f"--- [MultilingualAssistant] _synthesize_fallback: Error during fallback TTS: {e_tts_fb}", exc_info=True)
#         else:
#             logger.warning(f"--- [MultilingualAssistant] _synthesize_fallback: TTS not available, cannot synthesize fallback message.")
#             # Log fallback message to frontend even if TTS fails
#             if self.room and self.room.local_participant:
#                  try:
#                     payload = json.dumps({"type": "response", "text": message, "language": "fallback_en"}, ensure_ascii=False) # Assume English for fallback
#                     await self.room.local_participant.publish_data(payload=payload, topic="response_update")
#                  except Exception: pass # Ignore publishing error


# # --- Agent Entrypoint ---
# async def entrypoint(ctx: JobContext):
#     logger.info(f"--- [AgentEntry] AGENT LOGIC ENTRYPOINT CALLED --- Room: {ctx.room.name}, Job: {ctx.job.id}")

#     try:
#         # Connect to the room - this needs to succeed for anything else to work
#         await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
#         logger.info(f"[AgentEntry] Connected to room. Local Participant SID: {ctx.room.local_participant.sid}")
#     except Exception as e_connect:
#         logger.critical(f"[AgentEntry] Fatal error during ctx.connect(): {e_connect}", exc_info=True);
#         # If connection fails, the job should terminate. Raise the exception.
#         raise

#     logger.info("[AgentEntry] Initializing plugins...")
#     stt_plugin = None
#     llm_plugin = None
#     tts_plugin = None
#     vad_plugin = None
#     turn_detector = None

#     try:
#         # Initialize STT - USE MULTI-LANGUAGE FOR DEEPGRAM
#         # interim_results=False is fine for this turn-based agent
#         # language="multi" tells Deepgram to auto-detect and include language in results
#         if not DEEPGRAM_API_KEY_ENV:
#              raise ValueError("DEEPGRAM_API_KEY environment variable is not set. Cannot initialize Deepgram STT.")
#         stt_plugin = deepgram.STT(model="nova-2", language="multi", interim_results=False)
#         logger.info("[AgentEntry] Deepgram STT plugin initialized with language='multi'.")

#         # Initialize LLM
#         if not OPENAI_API_KEY_ENV:
#              raise ValueError("OPENAI_API_KEY environment variable is not set. Cannot initialize LLM/TTS.")
#         llm_plugin = openai.LLM(model="gpt-4o-mini", api_key=OPENAI_API_KEY_ENV) # Using gpt-4o-mini as it's cost-effective and multilingual
#         logger.info(f"[AgentEntry] OpenAI LLM plugin initialized with model: gpt-4o-mini.")

#         # Initialize TTS - Use a voice that handles Hindi well, like 'shimmer'.
#         # The chosen voice will be used for ALL output languages.
#         # CORRECTED: Changed voice to 'shimmer' - confirmed valid by OpenAI API error message
#         tts_plugin = openai.TTS(voice="shimmer", api_key=OPENAI_API_KEY_ENV)
#         logger.info(f"[AgentEntry] OpenAI TTS plugin initialized with voice: shimmer.")

#         # Initialize VAD - Silero is good for voice activity detection
#         vad_plugin = silero.VAD.load()
#         logger.info("[AgentEntry] Silero VAD plugin initialized.")

#         # Initialize multilingual turn detector - Helps the pipeline know when turns end in different languages
#         turn_detector = MultilingualModel()
#         logger.info("[AgentEntry] Multilingual turn detector initialized.")


#         logger.info("[AgentEntry] All essential plugins initialized successfully.")

#     except Exception as e_plugins:
#         logger.critical(f"[AgentEntry] CRITICAL ERROR initializing plugins: {e_plugins}", exc_info=True);
#         # If essential plugins fail, the agent cannot function. Terminate the job.
#         if ctx.room and ctx.room.local_participant:
#              try:
#                  # Attempt to send error message to frontend
#                  payload = json.dumps({"type": "error", "text": f"Agent failed to initialize plugins: {e_plugins}"}, ensure_ascii=False)
#                  await ctx.room.local_participant.publish_data(payload=payload, topic="error")
#              except Exception:
#                  pass # Ignore errors publishing error messages
#         raise # Re-raise the exception to terminate the job


#     # Create and start the AgentSession with the initialized plugins
#     if stt_plugin and llm_plugin and tts_plugin and vad_plugin and turn_detector:
#         # Pass all necessary plugins to AgentSession. The AgentSession
#         # and its internal voice pipeline connect these plugins.
#         # The pipeline receives the STT result (including language) and
#         # uses it when feeding the LLM and TTS.
#         session = AgentSession(
#             stt=stt_plugin,
#             llm=llm_plugin,
#             tts=tts_plugin,
#             vad=vad_plugin,
#             turn_detection=turn_detector, # Pass the multilingual detector
#             # initial_prompts is not supported in this SDK version
#             # Manual greeting can be implemented in the Agent class if needed.
#         )
#         logger.info("[AgentEntry] AgentSession created with initialized plugins.")

#         # Instantiate your custom agent logic class
#         assistant_agent_instance = MultilingualAssistant() # Use the renamed class
#         logger.info("[AgentEntry] MultilingualAssistant instance created.")

#         logger.info("[AgentEntry] Starting AgentSession...")
#         # session.start awaits the completion of the agent's main loop.
#         # This is where the session takes control, receives audio, sends to STT,
#         # gets transcripts (passing the string to Agent.on_transcript),
#         # uses VAD/TurnDetector, sends context + detected language (internally) to LLM,
#         # gets LLM stream (passed to Agent.llm_stream_sentences helper),
#         # sends sentences to TTS.
#         main_agent_task = asyncio.create_task(session.start(room=ctx.room, agent=assistant_agent_instance))
#         logger.info("[AgentEntry] AgentSession.start() called. Session will manage turns.")

#         # Manual greeting alternative:
#         # You could potentially trigger the greeting here after the session starts.
#         # Example (requires self.tts on the agent instance, which might not be available immediately):
#         # await assistant_agent_instance._synthesize_fallback("नमस्ते! / Hello! मैं आपकी कैसे मदद कर सकती हूँ?")
#         # Or handle this within the Agent class's start lifecycle if available.


#         logger.info("[AgentEntry] Main agent logic running. Waiting for AgentSession task to complete.")
#         try:
#             # Await the main agent task. This will block until the session ends (e.g., room closes)
#             await main_agent_task
#             logger.info("[AgentEntry] Main agent_task completed normally.")
#         except asyncio.CancelledError:
#             logger.info("[AgentEntry] Main agent_task was cancelled.")
#         except Exception as e_main_task:
#             # This catches exceptions that might occur *during* the session's main loop (less common)
#             # Errors during plugin init or in agent methods are usually caught elsewhere.
#             logger.error(f"[AgentEntry] Main agent_task exited with unexpected error during session runtime: {e_main_task}", exc_info=True)
#     else:
#          # This case should be rare due to the 'raise' in plugin initialization,
#          # but logs if somehow plugins are None here.
#          logger.critical("[AgentEntry] Cannot start AgentSession because essential plugins were not initialized.")


#     logger.info(f"[AgentEntry] Agent logic finished/terminated for job {ctx.job.id if ctx.job else 'N/A'}.")


# # --- Helper function to check and free port (useful for development) ---
# def check_and_free_port(port):
#     """Check if a port is in use and attempt to free it."""
#     logger.debug(f"Checking port {port}...")
#     found_process = False
#     # Use proc.pid and proc.name() directly from the process object
#     for proc in psutil.process_iter(['pid', 'name']):
#         try:
#             # Call net_connections() method on the process object
#             # Filter for internet sockets (TCP/UDP) that are listening or established
#             connections = proc.net_connections(kind='inet')
#             if connections: # check if list is not empty
#                  for conn in connections:
#                     # Check if the local address matches the port
#                     # conn.status allows checking if it's actually listening or established
#                     if conn.laddr and conn.laddr.port == port and conn.status in (psutil.CONN_LISTEN, psutil.CONN_ESTABLISHED):
#                         logger.warning(f"Port {port} is in use by PID {proc.pid} ({proc.name()}). Attempting to terminate.")
#                         found_process = True
#                         try:
#                             proc.terminate()
#                             proc.wait(timeout=5)
#                             logger.info(f"Successfully terminated process on port {port}.")
#                         except psutil.TimeoutExpired:
#                              logger.warning(f"Process PID {proc.pid} did not terminate within timeout, killing.")
#                              try:
#                                  proc.kill()
#                                  logger.info(f"Successfully killed process on port {port}.")
#                              except psutil.AccessDenied:
#                                  logger.error(f"Access denied when trying to kill process PID {proc.pid} on port {port}. Manual intervention may be required.")
#                              except psutil.NoSuchProcess:
#                                  logger.warning(f"Process PID {proc.pid} vanished before kill.")
#                         except psutil.AccessDenied:
#                              logger.error(f"Access denied when trying to terminate process PID {proc.pid} on port {port}. Manual intervention may be required.")
#                         return # Exit after handling the first process found
#         except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
#             # Ignore processes that disappear or deny access or are zombies
#             continue
#         except Exception as e:
#              logger.error(f"Error checking process {proc.pid}: {e}", exc_info=True)

#     if not found_process:
#         logger.debug(f"Port {port} is free or no process found/could be terminated.")


# def run_worker():
#     """Function to run the LiveKit Agent Worker in a separate thread."""
#     try:
#         logger.info("--- run_worker: Attempting to start LiveKit Agent Worker ---")

#         if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
#             logger.critical("!!! run_worker: CRITICAL - LiveKit connection credentials NOT FULLY CONFIGUßRED. Worker will NOT start.")
#             return

#         # Plugin initialization checks moved to entrypoint, but log warnings here
#         # Now Deepgram is CRITICAL if not set.
#         if not OPENAI_API_KEY_ENV: logger.warning("!!! run_worker: OpenAI API key not found. LLM/TTS plugins might fail during entrypoint initialization.")
#         if not DEEPGRAM_API_KEY_ENV: logger.critical("!!! run_worker: DEEPGRAM_API_KEY not found. Deepgram STT will fail during entrypoint initialization.")


#         # Check and free port 8081, which LiveKit Agent Worker CLI might use
#         check_and_free_port(8081)

#         worker_options = WorkerOptions(
#             entrypoint_fnc=entrypoint,
#         )

#         # Modify sys.argv for cli.run_app, which expects a command like 'start'
#         original_argv = list(sys.argv)
#         # Ensure the first argument is the script name (can be dummy) and the second is 'start'
#         # Use the actual script name if possible, fallback otherwise.
#         script_name = os.path.basename(__file__) if '__file__' in locals() else 'main.py' # Assuming filename is main.py
#         sys.argv = [script_name, 'start']
#         logger.info(f"run_worker: Modified sys.argv for cli.run_app: {sys.argv}")

#         logger.info("run_worker: Applying nest_asyncio and calling cli.run_app(worker_options)...")
#         try:
#             # nest_asyncio is often needed when running asyncio event loops
#             # inside threads of applications like FastAPI/Uvicorn which already have one.
#             import nest_asyncio
#             nest_asyncio.apply()
#             logger.info("run_worker: Applied nest_asyncio.")
#             # This call blocks until the worker stops (e.g., via signal or error in entrypoint)
#             cli.run_app(worker_options)
#         except SystemExit as se:
#              # SystemExit with code 0 is usually a clean exit from the CLI app
#              if se.code != 0:
#                  logger.error(f"run_worker: cli.run_app exited with non-zero status code {se.code}.", exc_info=True)
#              else:
#                  logger.info(f"run_worker: cli.run_app exited successfully (code: {se.code}).")
#         except Exception as e:
#             # This will catch exceptions *within* cli.run_app itself, not exceptions within the async entrypoint task
#             logger.error(f"!!! run_worker: CRITICAL ERROR during cli.run_app execution: {e}", exc_info=True)
#         finally:
#             sys.argv = original_argv # Restore original argv when cli.run_app finishes
#             logger.info(f"run_worker: Restored original sys.argv.")

#         logger.info("--- run_worker: LiveKit Agent Worker `cli.run_app` finished. ---")

#     except Exception as e_outer:
#         # This catches exceptions that happen *outside* the cli.run_app call but inside run_worker
#         logger.error(f"!!! run_worker: CRITICAL ERROR in outer run_worker block: {e_outer}", exc_info=True)
#     finally:
#         logger.info("--- run_worker: Function execution complete or terminated. ---")


# # --- Main execution block ---
# if __name__ == "__main__":
#     import faulthandler
#     faulthandler.enable() # Enable fault handler for better debugging on crashes
#     logger.info("--- __main__: Script execution started ---")

#     # You can add argument parsing here if needed, e.g., for 'download-files'
#     # Example: if "download-files" in sys.argv: handle_download()

#     # Check and free port 8000 for Uvicorn (FastAPI server)
#     check_and_free_port(8000)

#     logger.info("--- __main__: Creating worker thread (LiveKitAgentWorker) ---")
#     # Daemon thread will exit automatically when the main process (uvicorn) exits
#     worker_thread = threading.Thread(target=run_worker, name="LiveKitAgentWorker", daemon=True)
#     worker_thread.start()
#     logger.info("--- __main__: Worker thread started.")

#     # Give the worker thread a moment to initialize plugins and connect.
#     # If entrypoint crashes instantly due to missing env vars etc., the worker thread might die here.
#     logger.info("--- __main__: Waiting a few seconds for agent worker to initialize before starting Uvicorn...")
#     time.sleep(5) # Adjust sleep duration if necessary

#     if worker_thread.is_alive():
#         logger.info("--- __main__: Worker thread IS ALIVE. Proceeding to start Uvicorn.")
#     else:
#         # If the worker thread isn't alive after the initial sleep, the agent won't work. Log this critically.
#         logger.critical("--- __main__: Worker thread IS NOT ALIVE after initial sleep. Agent will not function. Check 'agent_logs.log' for CRITICAL errors during plugin initialization (e.g., missing API keys).")
#         # Optionally, you could exit here if the worker is essential
#         # sys.exit(1)


#     logger.info("--- __main__: Starting Uvicorn server for FastAPI app ---")
#     try:
#         # Use "main:app" because the script is named main.py
#         uvicorn.run(
#             "main:app", # Ensure this matches the filename
#             host="0.0.0.0",
#             port=8000,
#             reload=False, # Keep reload=False, especially with threading/agent worker
#             log_level="info" # Keep this info or higher for less noise from uvicorn itself
#         )
#     except Exception as e_uvicorn:
#         logger.error(f"--- __main__: Uvicorn server failed: {e_uvicorn}", exc_info=True)

#     logger.info("--- __main__: Uvicorn server has stopped. ---")
#     logger.info("--- __main__: Script execution finished. ---")

# # --- END OF FILE main.py (Ensuring Correct TTS Voice) ---























# 17-05-2025

# whisper enable model working model

# import asyncio
# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import logging
# from datetime import timedelta
# import threading
# import time 
# import sys 
# import psutil
# from typing import Any, AsyncIterator

# # LiveKit specific imports
# from livekit.api import AccessToken, VideoGrants
# from livekit import agents, rtc
# from livekit.agents import (
#     JobContext,
#     WorkerOptions,
#     cli,
#     Agent, 
#     AgentSession,
#     AutoSubscribe,
# )
# from livekit.agents.llm import LLMStream
# from livekit.agents import voice
# from livekit.agents import llm as livekit_llm 
# from livekit.agents.llm import ChatMessage, ChatContext
# import pydantic_core

# # LiveKit Plugins
# # Import openai for STT, LLM, and TTS
# from livekit.plugins import openai 
# # Keep silero for VAD if needed, Deepgram is no longer needed for STT
# from livekit.plugins import silero 

# # Load environment variables
# load_dotenv()

# app = FastAPI()

# origins = [
#     "http://localhost:3000", 
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Configure logging with console and file output
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler('agent_logs.log')
#     ]
# )
# logger = logging.getLogger(__name__)
# agent_logger = logging.getLogger("livekit.agents")
# agent_logger.setLevel(logging.DEBUG)

# # Fetch environment variables
# LIVEKIT_URL = os.getenv("LIVEKIT_URL")
# LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
# LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
# OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY")
# ELEVENLABS_API_KEY_ENV = os.getenv("ELEVENLABS_API_KEY") # Note: ElevenLabs is not used in this code currently
# DEEPGRAM_API_KEY_ENV = os.getenv("DEEPGRAM_API_KEY") # Note: Deepgram is no longer used for STT

# logger.info(f"MODULE LEVEL: LIVEKIT_URL: {LIVEKIT_URL}")
# logger.info(f"MODULE LEVEL: LIVEKIT_API_KEY IS SET: {bool(LIVEKIT_API_KEY)}")
# logger.info(f"MODULE LEVEL: DEEPGRAM_API_KEY IS SET: {bool(DEEPGRAM_API_KEY_ENV)}") # Still logged, but Deepgram is not used
# logger.info(f"MODULE LEVEL: OPENAI_API_KEY IS SET: {bool(OPENAI_API_KEY_ENV)}")

# @app.get("/token")
# async def get_token():
#     logger.info("[TokenSvc] /token endpoint hit.")
#     if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
#         logger.error("[TokenSvc] LiveKit credentials not configured properly.")
#         raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

#     video_grant_obj = VideoGrants(
#         room_join=True,
#         room="voice-assistant-room", 
#         can_publish=True,
#         can_subscribe=True,
#         can_publish_data=True,
#     )

#     token_builder = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
#     token_builder.identity = "frontend_user" 
#     token_builder.name = "Frontend User"
#     token_builder.ttl = timedelta(hours=1)
#     token_builder.with_grants(video_grant_obj)
#     token_jwt = token_builder.to_jwt()

#     logger.info(f"[TokenSvc] Generated token for frontend_user to join 'voice-assistant-room'")
#     return {"token": token_jwt, "url": LIVEKIT_URL}

# # --- Define the Assistant Agent ---
# class GUIFocusedAssistant(Agent): 
#     def __init__(self) -> None:
#         super().__init__(
#             instructions=(
#                 "You are Echo, a friendly and helpful voice assistant. Respond concisely. "
#                 "When the user speaks, understand their query and provide a relevant answer."
#                 "Your primary language is English and Hindi, but you can understand multiple languages." # Added clarification
#             )
#         )
#         self.last_user_query_for_log = None 
#         self.chat_history: list[ChatMessage] = [] 
#         logger.info("--- [GUIFocusedAssistant] __init__ CALLED ---")

#     async def on_transcript(self, transcript: str, is_final: bool) -> None:
#         # This method receives the output directly from the STT engine (now OpenAI Whisper)
#         logger.info(f"--- [GUIFocusedAssistant] on_transcript CALLED --- Final: {is_final}, Text: '{transcript}'")
        
#         # The 'transcript' variable here will contain the text transcribed by Whisper,
#         # which includes Hindi if the user spoke Hindi.
#         # The line above already logs this transcript to the backend console.

#         if is_final and transcript:
#             self.last_user_query_for_log = transcript
            
#             logger.info(f"--- [GUIFocusedAssistant] on_transcript: User query received: '{transcript}'. Attempting to generate LLM response.")
            
#             if not hasattr(self, 'llm') or not self.llm:
#                 logger.error("--- [GUIFocusedAssistant] on_transcript: self.llm is not available. Cannot generate LLM response.")
#                 return

#             try:
#                 # Create ChatMessage for current user transcript
#                 # Content should be a list of strings
#                 user_chat_message = ChatMessage(role="user", content=[transcript]) 
                
#                 current_chat_turn_history = list(self.chat_history)
#                 current_chat_turn_history.append(user_chat_message)

#                 if len(current_chat_turn_history) > 20: 
#                     current_chat_turn_history = current_chat_turn_history[-20:]

#                 # Create ChatContext
#                 chat_ctx_for_llm = ChatContext(messages=current_chat_turn_history)

#                 logger.info(f"--- [GUIFocusedAssistant] on_transcript: Calling self.llm.chat(chat_ctx=...) with history (count: {len(current_chat_turn_history)})")
                
#                 llm_stream = await self.llm.chat(chat_ctx=chat_ctx_for_llm) 

#                 logger.info(f"--- [GUIFocusedAssistant] on_transcript: self.llm.chat() returned stream. Now calling self.handle_llm_response.")
#                 self.chat_history = current_chat_turn_history 
#                 await self.handle_llm_response(llm_stream)

#             except TypeError as te_chat: 
#                  logger.error(f"--- [GUIFocusedAssistant] on_transcript: TypeError calling self.llm.chat(chat_ctx=...): {te_chat}", exc_info=True)
#             except Exception as e_llm_call_or_handle:
#                  logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error in LLM call sequence or in handle_llm_response: {e_llm_call_or_handle}", exc_info=True)
        
#         if is_final and transcript: # Publish transcript to frontend
#             if self.room and self.room.local_participant:
#                 logger.info(f"--- [GUIFocusedAssistant] on_transcript: Publishing FINAL user transcript to frontend: '{transcript}'")
#                 try:
#                     # Ensure payload is bytes or str, encoding str to bytes if necessary
#                     payload_str = f"transcription:{transcript}"
#                     await self.room.local_participant.publish_data(payload=payload_str.encode('utf-8'), topic="transcription")
#                 except Exception as e_pub_tx:
#                     logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error publishing user transcript: {e_pub_tx}", exc_info=True)

#     async def handle_llm_response(self, llm_stream: LLMStream) -> None:
#         logger.info(f"--- [GUIFocusedAssistant] handle_llm_response CALLED ---")
#         collected_text_for_frontend = ""
#         temp_sentences_for_tts = [] 
#         # stream_processed_successfully = False # Variable not currently used, can be removed if not added later

#         try:
#             # Using the llm_stream_sentences helper from the base Agent class
#             async for sentence_obj in self.llm_stream_sentences(llm_stream):
#                 logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Processing sentence: '{sentence_obj.text}'")
#                 if sentence_obj.text:
#                     collected_text_for_frontend += sentence_obj.text + " "
#                     temp_sentences_for_tts.append(sentence_obj.text)
#                 # stream_processed_successfully = True # Variable not currently used
#         except Exception as e_llm_stream_processing:
#             logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error processing LLM stream: {e_llm_stream_processing}", exc_info=True)
#             if not collected_text_for_frontend: collected_text_for_frontend = "Error processing response."
#             else: collected_text_for_frontend += " (Error in stream)"
        
#         final_collected_text = collected_text_for_frontend.strip()

#         if final_collected_text: 
#             # Content should be a list of strings
#             assistant_chat_message = ChatMessage(role="assistant", content=[final_collected_text])
#             self.chat_history.append(assistant_chat_message)
#             if len(self.chat_history) > 20: self.chat_history = self.chat_history[-20:]

#         log_context_info = f"for user query '{self.last_user_query_for_log}'" if self.last_user_query_for_log else "(initial/agent-initiated)"
#         if final_collected_text: logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: FINAL MODEL_RESPONSE {log_context_info}: '{final_collected_text}'")
#         else: logger.warning(f"--- [GUIFocusedAssistant] handle_llm_response: No final text from LLM {log_context_info}.")

#         if final_collected_text and self.room and self.room.local_participant:
#              logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Publishing agent response to GUI: '{final_collected_text}'")
#              try:
#                 # Ensure payload is bytes or str, encoding str to bytes if necessary
#                 payload_str = f"response:{final_collected_text}"
#                 await self.room.local_participant.publish_data(payload=payload_str.encode('utf-8'), topic="response")
#              except Exception as e_pub_resp:
#                 logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error publishing agent response: {e_pub_resp}", exc_info=True)


#         if temp_sentences_for_tts and any(s.strip() for s in temp_sentences_for_tts) and hasattr(self, 'tts') and self.tts:
#             logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Attempting TTS.")
#             async def gen_tts():
#                 # Filter out empty strings just in case
#                 for s in temp_sentences_for_tts:
#                     if s.strip():
#                        yield s.strip()
#             try:
#                 await self.tts.synthesize(gen_tts())
#             except Exception as e_tts:
#                 logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error during TTS synthesis: {e_tts}", exc_info=True)

#         self.last_user_query_for_log = None 
#         logger.info(f"--- [GUIFocusedAssistant] handle_llm_response FINISHED ---")


# # --- Agent Entrypoint ---
# async def entrypoint(ctx: JobContext):
#     logger.info(f"--- [AgentEntry] AGENT LOGIC ENTRYPOINT CALLED (ChatContext Deep Dive) --- Room: {ctx.room.name}, Job: {ctx.job.id}")

#     try:
#         await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
#         logger.info(f"[AgentEntry] Connected to room. Local Participant SID: {ctx.room.local_participant.sid}")
#     except Exception as e_connect:
#         logger.error(f"[AgentEntry] Fatal error during ctx.connect(): {e_connect}", exc_info=True); raise

#     logger.info("[AgentEntry] Initializing plugins...")
#     llm_plugin = None
#     tts_plugin = None
#     stt_plugin = None # Renamed for clarity
#     vad_plugin = None # Renamed for clarity

#     # Check for required OpenAI API key
#     if not OPENAI_API_KEY_ENV:
#         logger.critical("!!! [AgentEntry] CRITICAL - OPENAI_API_KEY environment variable is NOT set. OpenAI STT, LLM, and TTS plugins cannot initialize. !!!")
#         raise EnvironmentError("OPENAI_API_KEY not set.")

#     try:
#         # --- Initialize OpenAI STT ---
#         # OpenAI Whisper supports multiple languages automatically.
#         # No need for specific language parameter like Deepgram.
#         logger.info("[AgentEntry] Initializing OpenAI STT plugin...")
#         stt_plugin = openai.STT(api_key=OPENAI_API_KEY_ENV) 
#         logger.info("[AgentEntry] OpenAI STT plugin initialized.")

#         # --- Initialize OpenAI LLM ---
#         logger.info("[AgentEntry] Initializing OpenAI LLM plugin...")
#         llm_plugin = openai.LLM(model="gpt-4o-mini", api_key=OPENAI_API_KEY_ENV) 
#         logger.info("[AgentEntry] OpenAI LLM plugin initialized.")

#         # --- Initialize OpenAI TTS ---
#         logger.info("[AgentEntry] Initializing OpenAI TTS plugin...")
#         tts_plugin = openai.TTS(voice="ash", api_key=OPENAI_API_KEY_ENV) # Use OpenAI TTS
#         logger.info("[AgentEntry] OpenAI TTS plugin initialized.")

#         # --- Initialize Silero VAD ---
#         logger.info("[AgentEntry] Initializing Silero VAD plugin...")
#         vad_plugin = silero.VAD.load() 
#         logger.info("[AgentEntry] Silero VAD plugin initialized.")

#         logger.info("[AgentEntry] All required plugins initialized successfully.")
#     except Exception as e_plugins:
#         logger.error(f"[AgentEntry] Error initializing plugins: {e_plugins}", exc_info=True); raise 

#     # --- LLM PLUGIN DIRECT TEST --- (Keep this test, it's good practice)
#     logger.info("[AgentEntry] --- Testing LLM plugin directly START ---")
#     llm_test_passed = False
#     if not llm_plugin:
#         logger.error("[AgentEntry] LLM Plugin is None, cannot run direct test.")
#     else:
#         test_prompt_text = "Hello, LLM. Confirm you are working by saying OK."
#         chat_message_list: list[ChatMessage] = []
#         try:
#             chat_message_list = [ChatMessage(role="user", content=[test_prompt_text])]
#             logger.info(f"[AgentEntry] LLM Direct Test: Successfully created chat_message_list.")
#         except Exception as e_cm_create:
#             logger.error(f"[AgentEntry] LLM Direct Test: FAILED to create ChatMessage list: {e_cm_create}", exc_info=True)

#         if chat_message_list:
#             chat_ctx_for_test: ChatContext | None = None
#             try:
#                 # Attempt to create ChatContext - using keyword arg 'messages' is safer
#                 logger.info(f"[AgentEntry] LLM Direct Test: Attempting ChatContext(messages=chat_message_list)")
#                 chat_ctx_for_test = ChatContext(messages=chat_message_list) 
#                 logger.info(f"[AgentEntry] LLM Direct Test: ChatContext created.")
#             except Exception as e_ctx_create:
#                 logger.error(f"[AgentEntry] LLM Direct Test: Error creating ChatContext: {e_ctx_create}", exc_info=True)
#                 chat_ctx_for_test = None

#             if chat_ctx_for_test:
#                 try:
#                     logger.info(f"[AgentEntry] LLM Direct Test: Attempting llm_plugin.chat(chat_ctx=chat_ctx_for_test)")
#                     # Use an iterator to get chunks from the async stream
#                     async for chunk in await llm_plugin.chat(chat_ctx=chat_ctx_for_test): 
#                         # Assuming chunk has a 'text' attribute for partial responses
#                         chunk_text = getattr(chunk, 'text', str(chunk))
#                         if chunk_text: # Log only if chunk has text
#                              logger.info(f"[AgentEntry] LLM Direct Test Chunk: '{chunk_text}'")
#                              llm_test_passed = True # Mark as passed on first text chunk
#                              break # Stop after receiving the first chunk
#                     if llm_test_passed: logger.info("[AgentEntry] LLM plugin direct test SUCCEEDED.")
#                     else: logger.warning("[AgentEntry] LLM plugin direct test: No text chunks received.")
#                 except Exception as e_llm_call:
#                     logger.error(f"[AgentEntry] LLM plugin direct test FAILED during chat call: {e_llm_call}", exc_info=True)
#             else:
#                 logger.warning("[AgentEntry] LLM Direct Test: chat_ctx_for_test is None, skipping llm.chat() call.")
#         else:
#             logger.warning("[AgentEntry] LLM Direct Test: chat_message_list is empty.")
            
#     if not llm_test_passed: logger.critical("[AgentEntry] !!! LLM direct test FAILED. Agent may not function correctly. !!!")
#     logger.info("[AgentEntry] --- Testing LLM plugin directly END ---")
#     # --- END OF LLM PLUGIN DIRECT TEST ---


#     # Pass the initialized plugins to AgentSession
#     session = AgentSession(stt=stt_plugin, llm=llm_plugin, tts=tts_plugin, vad=vad_plugin)
#     logger.info("[AgentEntry] AgentSession created with OpenAI STT, LLM, TTS and Silero VAD.")
    
#     assistant_agent_instance = GUIFocusedAssistant() # Your class
#     logger.info("[AgentEntry] GUIFocusedAssistant instance created.")
    
#     logger.info("[AgentEntry] Starting AgentSession with GUIFocusedAssistant...")
#     # The AgentSession will now use the configured STT (OpenAI Whisper)
#     main_agent_task = asyncio.create_task(session.start(room=ctx.room, agent=assistant_agent_instance))
#     logger.info("[AgentEntry] AgentSession.start() called. Session will manage greeting and turns.")

#     logger.info("[AgentEntry] Main agent logic running. Waiting for AgentSession task to complete.")
#     try:
#         await main_agent_task 
#     except asyncio.CancelledError: logger.info("[AgentEntry] Main agent_task was cancelled.")
#     except Exception as e_main_task: logger.error(f"[AgentEntry] Main agent_task exited with error: {e_main_task}", exc_info=True)
#     logger.info(f"[AgentEntry] Agent logic finished/terminated for job {ctx.job.id if ctx.job else 'N/A'}.")


# def check_and_free_port(port):
#     """Check if a port is in use and attempt to free it."""
#     for proc in psutil.process_iter(['pid', 'name']):
#         try:
#             for conn in proc.net_connections(kind='inet'):
#                 if conn.laddr.port == port:
#                     logger.warning(f"Port {port} is in use by PID {proc.pid} ({proc.name()}). Attempting to terminate.")
#                     try:
#                         proc.terminate()
#                         proc.wait(timeout=3) # Give it a few seconds
#                         if proc.is_running():
#                              logger.warning(f"Process {proc.pid} did not terminate gracefully. Attempting kill.")
#                              proc.kill()
#                              proc.wait(timeout=3)
#                         logger.info(f"Successfully handled process on port {port}.")
#                     except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as term_err:
#                          logger.error(f"Failed to terminate process {proc.pid} on port {port}: {term_err}")
#                     return # Assume we've handled the issue for this port
#         except (psutil.NoSuchProcess, psutil.AccessDenied):
#             continue
#     logger.debug(f"Port {port} is free or no process could be terminated.")

# def run_worker():
#     try:
#         logger.info("--- run_worker: Attempting to start LiveKit Agent Worker ---")
#         if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
#             logger.critical("!!! run_worker: CRITICAL - LiveKit connection credentials (URL, KEY, SECRET) NOT FOUND. Worker will NOT start.")
#             return
        
#         # Check for required OpenAI API key which is used by STT, LLM, and TTS
#         if not OPENAI_API_KEY_ENV: 
#             logger.critical("!!! run_worker: CRITICAL - OpenAI API key not found. OpenAI STT, LLM, and TTS plugins will fail.")
#             return

#         # Check and free port 8081 for LiveKit worker (if needed)
#         # check_and_free_port(8081) # Often the worker doesn't strictly require 8081, depends on config.

#         worker_options = WorkerOptions(
#             entrypoint_fnc=entrypoint,
#         )
        
#         # Modify sys.argv to trick cli.run_app into running the entrypoint
#         original_argv = list(sys.argv)
#         # Need to pass room and identity as cli args to entrypoint normally
#         # However, JobContext already provides room/job info.
#         # cli.run_app expects 'start' command.
#         sys.argv = ['livekit_agent_script_embedded.py', 'start'] 
#         logger.info(f"run_worker: Modified sys.argv for cli.run_app: {sys.argv}")
        
#         logger.info("run_worker: Applying nest_asyncio and calling cli.run_app(worker_options)...")
#         try:
#             # Use nest_asyncio to allow running the asyncio event loop within a thread
#             # Important when mixing asyncio (livekit agent) with blocking operations (threading/uvicorn)
#             import nest_asyncio
#             nest_asyncio.apply()
#             logger.info("run_worker: Applied nest_asyncio.")
            
#             # cli.run_app takes over the current thread and runs the worker
#             cli.run_app(worker_options)
            
#         finally:
#             # Restore original sys.argv after cli.run_app finishes
#             sys.argv = original_argv
#             logger.info(f"run_worker: Restored original sys.argv: {sys.argv}")

#         logger.info("--- run_worker: LiveKit Agent Worker `cli.run_app` finished. ---")

#     except SystemExit as se:
#         # cli.run_app might exit with SystemExit, which is normal shutdown
#         logger.info(f"run_worker: cli.run_app exited gracefully with SystemExit (code: {se.code}).")
#     except Exception as e:
#         logger.error(f"!!! run_worker: CRITICAL ERROR during run_worker or cli.run_app: {e}", exc_info=True)
#     finally:
#         logger.info("--- run_worker: Function execution complete or terminated. ---")

# if __name__ == "__main__":
#     # Enable faulthandler for debugging crashes
#     import faulthandler
#     faulthandler.enable()
#     logger.info("--- __main__: main.py script execution started ---")

#     # This block handles potential command line arguments if you were running
#     # this script directly with 'python main.py start', etc.
#     # In this setup, we manipulate sys.argv for cli.run_app.
#     if len(sys.argv) > 1 and sys.argv[1] in ["start", "download-files", "package"]:
#         logger.info(f"--- __main__: Command line argument '{sys.argv[1]}' detected. ---")
#         logger.info("--- __main__: Running cli.run_app directly without starting FastAPI server.")
#         # If you run 'python your_script.py start', it will just run the worker.
#         # This is useful for deploying the agent service separately.
#         # If you run 'python your_script.py download-files', it will download plugins.
#         # If you run 'python your_script.py package', it packages the agent.
#         # In the combined FastAPI+Agent setup, we usually DON'T run with these args.
#         # The threading approach calls run_worker() which handles the cli.run_app call.
#         # The 'download-files' step happens implicitly when plugins are initialized.
        
#         # We remove the FastAPI starting code if a cli argument is provided.
#         # This means you can run 'python main.py start' to run ONLY the agent worker,
#         # or 'python main.py' to run the FastAPI server AND the agent worker in a thread.
        
#         # Check and free port 8081 ONLY IF running agent directly? Usually not needed.
        
#         # Modify sys.argv and run cli.run_app directly in the main thread
#         original_argv = list(sys.argv)
#         logger.info(f"--- __main__: Running cli.run_app with sys.argv: {sys.argv}")
#         try:
#             # Apply nest_asyncio here too, just in case cli.run_app needs it directly
#             # though it's primarily needed when running an event loop *within* a thread.
#             # Keeping it applied in run_worker is sufficient for the threaded case.
#             # import nest_asyncio
#             # nest_asyncio.apply()
#             cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
#         except SystemExit as se:
#              logger.info(f"--- __main__: cli.run_app exited SystemExit (code: {se.code}).")
#         except Exception as e:
#              logger.error(f"--- __main__: Error during cli.run_app: {e}", exc_info=True)
#         finally:
#             sys.argv = original_argv # Restore sys.argv
#             logger.info(f"--- __main__: Restored sys.argv: {sys.argv}")
        
#         logger.info("--- __main__: Script execution finished after cli.run_app. ---")

#     else:
#         # Default case: no specific cli arguments, run FastAPI and the agent worker thread
#         logger.info("--- __main__: No specific command line argument. Starting FastAPI server and Agent Worker thread. ---")
        
#         # Check and free port 8000 for Uvicorn
#         check_and_free_port(8000)

#         logger.info("--- __main__: Creating worker thread (LiveKitAgentWorker) ---")
#         worker_thread = threading.Thread(target=run_worker, name="LiveKitAgentWorker", daemon=True) # daemon=True allows main process to exit even if thread is running
#         worker_thread.start()
#         logger.info("--- __main__: Worker thread started.")

#         logger.info("--- __main__: Waiting a few seconds for agent worker to potentially initialize...")
#         time.sleep(5) # Give the worker thread a moment to hit potential early errors
        
#         if worker_thread.is_alive():
#             logger.info("--- __main__: Worker thread IS ALIVE. Proceeding to start Uvicorn.")
#         else:
#             # If worker thread failed to start (e.g., missing env vars), log it prominently
#             logger.critical("!!! __main__: CRITICAL - Worker thread IS NOT ALIVE. It likely exited during startup. Check logs above for errors (e.g., missing credentials, plugin issues). Agent functionality will not be available. !!!")

#         logger.info("--- __main__: Starting Uvicorn server for FastAPI app ---")
#         try:
#             uvicorn.run(
#                 "main:app",
#                 host="0.0.0.0",
#                 port=8000,
#                 reload=False, # reload=True is not compatible with threading/nest_asyncio
#                 log_level="info"
#             )
#         except Exception as e_uvicorn:
#             logger.error(f"--- __main__: Uvicorn server failed: {e_uvicorn}", exc_info=True)
        
#         logger.info("--- __main__: Uvicorn server has stopped. ---")
        
#         # Wait for the worker thread to potentially finish (though daemon=True might prevent this if main exits)
#         # If main thread exits, daemon threads are abruptly stopped.
#         # If you needed graceful shutdown of the worker, you'd need a different mechanism (e.g., signals, shared event).
#         # For this example, daemon=True is simpler.
#         # worker_thread.join() # Don't join if daemon=True unless you have a specific shutdown signal

#         logger.info("--- __main__: Script execution finished. ---")





















# whisper trial model  -> working model in english and hindi perfectly





# import asyncio
# import os
# # Moved load_dotenv to the very top
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import logging
# from datetime import timedelta
# import threading
# import time 
# import sys 
# import psutil
# from typing import Any, AsyncIterator

# # LiveKit specific imports
# from livekit.api import AccessToken, VideoGrants
# from livekit import agents, rtc
# from livekit.agents import (
#     JobContext,
#     WorkerOptions,
#     cli,
#     Agent, 
#     AgentSession,
#     AutoSubscribe,
# )
# from livekit.agents.llm import LLMStream
# from livekit.agents import voice
# from livekit.agents import llm as livekit_llm 
# from livekit.agents.llm import ChatMessage, ChatContext
# import pydantic_core

# # LiveKit Plugins
# from livekit.plugins import openai 
# from livekit.plugins import silero 

# # --- Load environment variables and set up module-level variables immediately ---
# load_dotenv()

# LIVEKIT_URL = os.getenv("LIVEKIT_URL")
# LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
# LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
# OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY")
# ELEVENLABS_API_KEY_ENV = os.getenv("ELEVENLABS_API_KEY")
# DEEPGRAM_API_KEY_ENV = os.getenv("DEEPGRAM_API_KEY")

# # --- Configure logging with console and file output ---
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout), 
#         logging.FileHandler('agent_logs.log', encoding='utf-8') 
#     ]
# )
# logger = logging.getLogger(__name__)
# agent_logger = logging.getLogger("livekit.agents")
# agent_logger.setLevel(logging.DEBUG) 

# # Now log the module-level variables AFTER they have been defined
# logger.info(f"MODULE LEVEL: LIVEKIT_URL: {LIVEKIT_URL}")
# logger.info(f"MODULE LEVEL: LIVEKIT_API_KEY IS SET: {bool(LIVEKIT_API_KEY)}")
# logger.info(f"MODULE LEVEL: DEEPGRAM_API_KEY IS SET: {bool(DEEPGRAM_API_KEY_ENV)}") 
# logger.info(f"MODULE LEVEL: OPENAI_API_KEY IS SET: {bool(OPENAI_API_KEY_ENV)}")


# app = FastAPI()

# origins = [
#     "http://localhost:3000", 
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/token")
# async def get_token():
#     logger.info("[TokenSvc] /token endpoint hit.")
#     if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
#         logger.error("[TokenSvc] LiveKit credentials not configured properly.")
#         raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

#     video_grant_obj = VideoGrants(
#         room_join=True,
#         room="voice-assistant-room", 
#         can_publish=True,
#         can_subscribe=True,
#         can_publish_data=True,
#     )

#     token_builder = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
#     token_builder.identity = "frontend_user" 
#     token_builder.name = "Frontend User"
#     token_builder.ttl = timedelta(hours=1)
#     token_builder.with_grants(video_grant_obj)
#     token_jwt = token_builder.to_jwt()

#     logger.info(f"[TokenSvc] Generated token for frontend_user to join 'voice-assistant-room'")
#     return {"token": token_jwt, "url": LIVEKIT_URL}

# # --- Define the Assistant Agent ---
# class GUIFocusedAssistant(Agent): 
#     def __init__(self) -> None:
#         super().__init__(
#             instructions=(
#                 "You are Echo, a friendly and helpful voice assistant. Respond concisely. "
#                 "When the user speaks, understand their query and provide a relevant answer. "
#                 "Your primary language is Hindi and English, but you can understand multiple languages. "
#                 # Added explicit instruction to respond in Hindi using Devanagari script
#                 "If the user speaks in Hindi or a mix of Hindi, please respond in Hindi using Devanagari script."
#                 "Dont use urdu"
#             )
#         )
#         self.last_user_query_for_log = None 
#         self.chat_history: list[ChatMessage] = [] 
#         logger.info("--- [GUIFocusedAssistant] __init__ CALLED ---")

#     async def on_transcript(self, transcript: str, is_final: bool) -> None:
#         # Add a direct print statement to see the raw string value
#         print(f"--- DEBUG RAW TRANSCRIPT --- Final: {is_final}, Text: '{transcript}'")
        
#         # This logger.info call logs the `transcript` variable using your configured logger.
#         # This is where you should see readable Hindi if your terminal/file encoding is correct.
#         logger.info(f"--- [GUIFocusedAssistant] on_transcript CALLED --- Final: {is_final}, Text: '{transcript}'")
        
#         if is_final and transcript:
#             self.last_user_query_for_log = transcript
            
#             logger.info(f"--- [GUIFocusedAssistant] on_transcript: User query received: '{transcript}'. Attempting to generate LLM response.")
            
#             if not hasattr(self, 'llm') or not self.llm:
#                 logger.error("--- [GUIFocusedAssistant] on_transcript: self.llm is not available. Cannot generate LLM response.")
#                 return

#             try:
#                 user_chat_message = ChatMessage(role="user", content=[transcript]) 
                
#                 current_chat_turn_history = list(self.chat_history)
#                 current_chat_turn_history.append(user_chat_message)

#                 if len(current_chat_turn_history) > 20: 
#                     current_chat_turn_history = current_chat_turn_history[-20:]

#                 chat_ctx_for_llm = ChatContext(messages=current_chat_turn_history)

#                 logger.info(f"--- [GUIFocusedAssistant] on_transcript: Calling self.llm.chat(chat_ctx=...) with history (count: {len(current_chat_turn_history)})")
                
#                 llm_stream = await self.llm.chat(chat_ctx=chat_ctx_for_llm) 

#                 logger.info(f"--- [GUIFocusedAssistant] on_transcript: self.llm.chat() returned stream. Now calling self.handle_llm_response.")
#                 self.chat_history = current_chat_turn_history 
#                 await self.handle_llm_response(llm_stream)

#             except TypeError as te_chat: 
#                  logger.error(f"--- [GUIFocusedAssistant] on_transcript: TypeError calling self.llm.chat(chat_ctx=...): {te_chat}", exc_info=True)
#             except Exception as e_llm_call_or_handle:
#                  logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error in LLM call sequence or in handle_llm_response: {e_llm_call_or_handle}", exc_info=True)
        
#         if is_final and transcript: 
#             if self.room and self.room.local_participant:
#                 logger.info(f"--- [GUIFocusedAssistant] on_transcript: Publishing FINAL user transcript to frontend: '{transcript}'")
#                 try:
#                     payload_str = f"transcription:{transcript}"
#                     await self.room.local_participant.publish_data(payload=payload_str.encode('utf-8'), topic="transcription")
#                 except Exception as e_pub_tx:
#                     logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error publishing user transcript: {e_pub_tx}", exc_info=True)

#     async def handle_llm_response(self, llm_stream: LLMStream) -> None:
#         logger.info(f"--- [GUIFocusedAssistant] handle_llm_response CALLED ---")
#         collected_text_for_frontend = ""
#         temp_sentences_for_tts = [] 

#         try:
#             async for sentence_obj in self.llm_stream_sentences(llm_stream):
#                 logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Processing sentence: '{sentence_obj.text}'")
#                 if sentence_obj.text:
#                     collected_text_for_frontend += sentence_obj.text + " "
#                     temp_sentences_for_tts.append(sentence_obj.text)
#         except Exception as e_llm_stream_processing:
#             logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error processing LLM stream: {e_llm_stream_processing}", exc_info=True)
#             if not collected_text_for_frontend: collected_text_for_frontend = "Error processing response."
#             else: collected_text_for_frontend += " (Error in stream)"
        
#         final_collected_text = collected_text_for_frontend.strip()

#         if final_collected_text: 
#             assistant_chat_message = ChatMessage(role="assistant", content=[final_collected_text])
#             self.chat_history.append(assistant_chat_message)
#             if len(self.chat_history) > 20: self.chat_history = self.chat_history[-20:]

#         log_context_info = f"for user query '{self.last_user_query_for_log}'" if self.last_user_query_for_log else "(initial/agent-initiated)"
#         if final_collected_text: logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: FINAL MODEL_RESPONSE {log_context_info}: '{final_collected_text}'")
#         else: logger.warning(f"--- [GUIFocusedAssistant] handle_llm_response: No final text from LLM {log_context_info}.")

#         if final_collected_text and self.room and self.room.local_participant:
#              logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Publishing agent response to GUI: '{final_collected_text}'")
#              try:
#                 payload_str = f"response:{final_collected_text}"
#                 await self.room.local_participant.publish_data(payload=payload_str.encode('utf-8'), topic="response")
#              except Exception as e_pub_resp:
#                 logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error publishing agent response: {e_pub_resp}", exc_info=True)

#         if temp_sentences_for_tts and any(s.strip() for s in temp_sentences_for_tts) and hasattr(self, 'tts') and self.tts:
#             logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Attempting TTS.")
#             async def gen_tts():
#                 for s in temp_sentences_for_tts:
#                     if s.strip():
#                        yield s.strip()
#             try:
#                 await self.tts.synthesize(gen_tts())
#             except Exception as e_tts:
#                 logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error during TTS synthesis: {e_tts}", exc_info=True)

#         self.last_user_query_for_log = None 
#         logger.info(f"--- [GUIFocusedAssistant] handle_llm_response FINISHED ---")


# # --- Agent Entrypoint ---
# async def entrypoint(ctx: JobContext):
#     logger.info(f"--- [AgentEntry] AGENT LOGIC ENTRYPOINT CALLED (ChatContext Deep Dive) --- Room: {ctx.room.name}, Job: {ctx.job.id}")

#     try:
#         await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
#         logger.info(f"[AgentEntry] Connected to room. Local Participant SID: {ctx.room.local_participant.sid}")
#     except Exception as e_connect:
#         logger.error(f"[AgentEntry] Fatal error during ctx.connect(): {e_connect}", exc_info=True); raise

#     logger.info("[AgentEntry] Initializing plugins...")
#     llm_plugin = None
#     tts_plugin = None
#     stt_plugin = None 
#     vad_plugin = None 

#     if not OPENAI_API_KEY_ENV:
#         logger.critical("!!! [AgentEntry] CRITICAL - OPENAI_API_KEY environment variable is NOT set. OpenAI STT, LLM, and TTS plugins cannot initialize. !!!")
#         raise EnvironmentError("OPENAI_API_KEY not set.")

#     try:
#         logger.info("[AgentEntry] Initializing OpenAI STT plugin...")
#         stt_plugin = openai.STT(api_key=OPENAI_API_KEY_ENV) 
#         logger.info("[AgentEntry] OpenAI STT plugin initialized.")

#         logger.info("[AgentEntry] Initializing OpenAI LLM plugin...")
#         llm_plugin = openai.LLM(model="gpt-4o-mini", api_key=OPENAI_API_KEY_ENV) 
#         logger.info("[AgentEntry] OpenAI LLM plugin initialized.")

#         logger.info("[AgentEntry] Initializing OpenAI TTS plugin...")
#         tts_plugin = openai.TTS(voice="ash", api_key=OPENAI_API_KEY_ENV) 
#         logger.info("[AgentEntry] OpenAI TTS plugin initialized.")

#         logger.info("[AgentEntry] Initializing Silero VAD plugin...")
#         vad_plugin = silero.VAD.load() 
#         logger.info("[AgentEntry] Silero VAD plugin initialized.")

#         logger.info("[AgentEntry] All required plugins initialized successfully.")
#     except Exception as e_plugins:
#         logger.error(f"[AgentEntry] Error initializing plugins: {e_plugins}", exc_info=True); raise 

#     # --- LLM PLUGIN DIRECT TEST --- 
#     logger.info("[AgentEntry] --- Testing LLM plugin directly START ---")
#     llm_test_passed = False
#     if not llm_plugin:
#         logger.error("[AgentEntry] LLM Plugin is None, cannot run direct test.")
#     else:
#         # Test with a simple English prompt
#         test_prompt_text_en = "Hello, LLM. Confirm you are working by saying OK."
#         # Add a test with a Hindi prompt (Devanagari script) to see if it works
#         test_prompt_text_hi = "नमस्ते, आप कैसे हैं?" # "Namaste, aap kaise hain?" (Hello, how are you?)

#         chat_message_list: list[ChatMessage] = []
#         try:
#             chat_message_list = [
#                 ChatMessage(role="user", content=[test_prompt_text_en]),
#                 ChatMessage(role="user", content=[test_prompt_text_hi]) # Add Hindi test message
#             ]
#             logger.info(f"[AgentEntry] LLM Direct Test: Successfully created chat_message_list for English and Hindi tests.")
#         except Exception as e_cm_create:
#             logger.error(f"[AgentEntry] LLM Direct Test: FAILED to create ChatMessage list: {e_cm_create}", exc_info=True)

#         if chat_message_list:
#             chat_ctx_for_test: ChatContext | None = None
#             try:
#                 # Use the keyword argument 'messages'
#                 logger.info(f"[AgentEntry] LLM Direct Test: Attempting ChatContext(messages=chat_message_list)")
#                 chat_ctx_for_test = ChatContext(messages=chat_message_list) 
#                 logger.info(f"[AgentEntry] LLM Direct Test: ChatContext created.")
#             except Exception as e_ctx_create:
#                 logger.error(f"[AgentEntry] LLM Direct Test: Error creating ChatContext: {e_ctx_create}", exc_info=True)
#                 chat_ctx_for_test = None

#             if chat_ctx_for_test:
#                 try:
#                     logger.info(f"[AgentEntry] LLM Direct Test: Attempting llm_plugin.chat(chat_ctx=chat_ctx_for_test)")
#                     # Set a short timeout for the test chat call
#                     async for chunk in asyncio.wait_for(llm_plugin.chat(chat_ctx=chat_ctx_for_test), timeout=10): 
#                         chunk_text = getattr(chunk, 'text', str(chunk))
#                         if chunk_text: 
#                              logger.info(f"[AgentEntry] LLM Direct Test Chunk: '{chunk_text}'")
#                              # The LLM might respond in English or Hindi based on the prompts and instructions
#                              llm_test_passed = True 
#                              # Collect chunks for logging the full test response if needed
#                              # test_response_chunks.append(chunk_text) 
#                              # Break after the first chunk to confirm connectivity/response
#                              break 
                    
#                     # After breaking, check if the flag was set
#                     if llm_test_passed: 
#                         logger.info("[AgentEntry] LLM plugin direct test SUCCEEDED (received at least one chunk).")
#                         # Log full response if you uncommented collection
#                         # logger.info(f"[AgentEntry] LLM Direct Test Full Response: {''.join(test_response_chunks)}")
#                     else: 
#                         logger.warning("[AgentEntry] LLM plugin direct test: No text chunks received.")
#                 except asyncio.TimeoutError:
#                     logger.error("[AgentEntry] LLM plugin direct test FAILED: Timeout waiting for response.")
#                 except Exception as e_llm_call:
#                     logger.error(f"[AgentEntry] LLM plugin direct test FAILED during chat call: {e_llm_call}", exc_info=True)
#             else:
#                 logger.warning("[AgentEntry] LLM Direct Test: chat_ctx_for_test is None, skipping llm.chat() call.")
#         else:
#             logger.warning("[AgentEntry] LLM Direct Test: chat_message_list is empty.")
            
#     if not llm_test_passed: logger.critical("[AgentEntry] !!! LLM direct test FAILED. Agent may not function correctly. !!!")
#     logger.info("[AgentEntry] --- Testing LLM plugin directly END ---")
#     # --- END OF LLM PLUGIN DIRECT TEST ---

#     session = AgentSession(stt=stt_plugin, llm=llm_plugin, tts=tts_plugin, vad=vad_plugin)
#     logger.info("[AgentEntry] AgentSession created with OpenAI STT, LLM, TTS and Silero VAD.")
    
#     assistant_agent_instance = GUIFocusedAssistant() 
#     logger.info("[AgentEntry] GUIFocusedAssistant instance created.")
    
#     logger.info("[AgentEntry] Starting AgentSession with GUIFocusedAssistant...")
#     main_agent_task = asyncio.create_task(session.start(room=ctx.room, agent=assistant_agent_instance))
#     logger.info("[AgentEntry] AgentSession.start() called. Session will manage greeting and turns.")

#     logger.info("[AgentEntry] Main agent logic running. Waiting for AgentSession task to complete.")
#     try:
#         await main_agent_task 
#     except asyncio.CancelledError: logger.info("[AgentEntry] Main agent_task was cancelled.")
#     except Exception as e_main_task: logger.error(f"[AgentEntry] Main agent_task exited with error: {e_main_task}", exc_info=True)
#     logger.info(f"[AgentEntry] Agent logic finished/terminated for job {ctx.job.id if ctx.job else 'N/A'}.")


# def check_and_free_port(port):
#     logger.debug(f"Checking if port {port} is in use...")
#     for proc in psutil.process_iter(['pid', 'name']):
#         try:
#             for conn in proc.net_connections(kind='inet'):
#                 if conn.laddr.port == port:
#                     logger.warning(f"Port {port} is in use by PID {proc.pid} ({proc.name()}). Attempting to terminate.")
#                     try:
#                         proc.terminate()
#                         proc.wait(timeout=3) 
#                         if proc.is_running():
#                              logger.warning(f"Process {proc.pid} did not terminate gracefully. Attempting kill.")
#                              proc.kill()
#                              proc.wait(timeout=3)
#                         logger.info(f"Successfully handled potential process on port {port}.")
#                     except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as term_err:
#                          logger.error(f"Failed to terminate process {proc.pid} on port {port}: {term_err}")
#                     except Exception as e:
#                          logger.error(f"Unexpected error while handling process {proc.pid} on port {port}: {e}", exc_info=True)
#                     return 
#         except (psutil.NoSuchProcess, psutil.AccessDenied):
#             continue
#         except Exception as e:
#             logger.error(f"Error iterating process connections: {e}", exc_info=True)

#     logger.debug(f"Port {port} appears free.")

# def run_worker():
#     try:
#         logger.info("--- run_worker: Attempting to start LiveKit Agent Worker ---")
#         if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
#             logger.critical("!!! run_worker: CRITICAL - LiveKit connection credentials (URL, KEY, SECRET) NOT FOUND. Worker will NOT start.")
#             return
        
#         if not OPENAI_API_KEY_ENV: 
#             logger.critical("!!! run_worker: CRITICAL - OpenAI API key not found. OpenAI STT, LLM, and TTS plugins will fail.")
#             return

#         worker_options = WorkerOptions(
#             entrypoint_fnc=entrypoint,
#         )
        
#         original_argv = list(sys.argv)
#         sys.argv = ['livekit_agent_script_embedded.py', 'start'] 
#         logger.info(f"run_worker: Modified sys.argv for cli.run_app: {sys.argv}")
        
#         logger.info("run_worker: Applying nest_asyncio and calling cli.run_app(worker_options)...")
#         try:
#             import nest_asyncio
#             nest_asyncio.apply()
#             logger.info("run_worker: Applied nest_asyncio.")
            
#             cli.run_app(worker_options)
            
#         finally:
#             sys.argv = original_argv
#             logger.info(f"run_worker: Restored original sys.argv: {sys.argv}")

#         logger.info("--- run_worker: LiveKit Agent Worker `cli.run_app` finished. ---")

#     except SystemExit as se:
#         logger.info(f"run_worker: cli.run_app exited gracefully with SystemExit (code: {se.code}).")
#     except Exception as e:
#         logger.error(f"!!! run_worker: CRITICAL ERROR during run_worker or cli.run_app: {e}", exc_info=True)
#     finally:
#         logger.info("--- run_worker: Function execution complete or terminated. ---")

# if __name__ == "__main__":
#     import faulthandler
#     faulthandler.enable()
#     logger.info("--- __main__: main.py script execution started ---")

#     if len(sys.argv) > 1 and sys.argv[1] in ["start", "download-files", "package"]:
#         logger.info(f"--- __main__: Command line argument '{sys.argv[1]}' detected. Running cli.run_app directly. ---")
        
#         original_argv = list(sys.argv) 
#         logger.info(f"--- __main__: Running cli.run_app with current sys.argv: {sys.argv}")
#         try:
#             cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
#         except SystemExit as se:
#              logger.info(f"--- __main__: cli.run_app exited SystemExit (code: {se.code}).")
#         except Exception as e:
#              logger.error(f"--- __main__: Error during cli.run_app: {e}", exc_info=True)
#         finally:
#             sys.argv = original_argv 
#             logger.info(f"--- __main__: Restored sys.argv: {sys.argv}")
        
#         logger.info("--- __main__: Script execution finished after cli.run_app. ---")

#     else:
#         logger.info("--- __main__: No specific command line argument. Starting FastAPI server and Agent Worker thread. ---")
        
#         check_and_free_port(8000)

#         logger.info("--- __main__: Creating worker thread (LiveKitAgentWorker) ---")
#         worker_thread = threading.Thread(target=run_worker, name="LiveKitAgentWorker", daemon=True) 
#         worker_thread.start()
#         logger.info("--- __main__: Worker thread started.")

#         logger.info("--- __main__: Waiting a few seconds for agent worker to potentially initialize...")
#         time.sleep(5) 
        
#         if worker_thread.is_alive():
#             logger.info("--- __main__: Worker thread IS ALIVE. Proceeding to start Uvicorn.")
#         else:
#             logger.critical("!!! __main__: CRITICAL - Worker thread IS NOT ALIVE. It likely exited during startup. Check logs above for errors (e.g., missing credentials, plugin issues). Agent functionality will not be available. !!!")

#         logger.info("--- __main__: Starting Uvicorn server for FastAPI app ---")
#         try:
#             uvicorn.run(
#                 "main:app",
#                 host="0.0.0.0",
#                 port=8000,
#                 reload=False,
#                 log_level="info"
#             )
#         except Exception as e_uvicorn:
#             logger.error(f"--- __main__: Uvicorn server failed: {e_uvicorn}", exc_info=True)
        
#         logger.info("--- __main__: Uvicorn server has stopped. ---")
        
#         logger.info("--- __main__: Script execution finished. ---")



























# 19-05-2025









import asyncio
import os
import sys
import logging
from datetime import timedelta
import threading
import time
import psutil
from typing import Any, AsyncIterator

# Reconfigure sys.stdout to use UTF-8 encoding without replacing the stream
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    # Fallback for older Python versions or if reconfigure is not available
    pass

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
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
# Note: JSON logs from livekit.agents may show Unicode escapes (e.g., \u0939 for ह).
# Check agent_logs.log or console for readable Devanagari text.
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_logs.log', encoding='utf-8')
    ]
)

# Ensure logger uses UTF-8
logger = logging.getLogger(__name__)
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setStream(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

agent_logger = logging.getLogger("livekit.agents")
agent_logger.setLevel(logging.DEBUG)

# Log module-level variables
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
        logger.debug(f"--- [GUIFocusedAssistant] on_transcript ENTERED --- Final: {is_final}, Text: '{transcript}'")
        logger.debug(f"--- [GUIFocusedAssistant] on_transcript: Raw bytes: {transcript.encode('utf-8')}, Decoded: {transcript}")

        print(f"--- DEBUG RAW USER TRANSCRIPT --- Final: {is_final}, Text: '{transcript}'")
        logger.info(f"--- [GUIFocusedAssistant] on_transcript CALLED --- Final: {is_final}, Text: '{transcript}'")
        logger.info(f"--- [DEBUG STT OUTPUT] Received: '{transcript}'")

        if is_final and transcript and transcript.strip():
            self.last_user_query_for_log = transcript.strip()

            logger.info(f"--- [GUIFocusedAssistant] on_transcript: User query received: '{self.last_user_query_for_log}'. Attempting to generate LLM response.")

            if not hasattr(self, 'llm') or not self.llm:
                logger.error("--- [GUIFocusedAssistant] on_transcript: self.llm is not available. Cannot generate LLM response.")
                return

            try:
                user_chat_message = ChatMessage(role="user", content=[self.last_user_query_for_log])
                current_chat_turn_history = list(self.chat_history)
                current_chat_turn_history.append(user_chat_message)

                if len(current_chat_turn_history) > 20:
                    current_chat_turn_history = current_chat_turn_history[-20:]

                try:
                    chat_ctx_for_llm = ChatContext(current_chat_turn_history)
                    logger.debug("--- [GUIFocusedAssistant] on_transcript: Created ChatContext successfully (positional).")
                except TypeError as e_ctx_init:
                    logger.error(f"--- [GUIFocusedAssistant] on_transcript: Failed to create ChatContext positionally: {e_ctx_init}", exc_info=True)
                    return

                logger.info(f"--- [GUIFocusedAssistant] on_transcript: Calling self.llm.chat(...) with history (count: {len(current_chat_turn_history)})")

                llm_stream = self.llm.chat(chat_ctx=chat_ctx_for_llm)
                logger.debug(f"--- [on_transcript] Type of llm_stream: {type(llm_stream)}")
                if not isinstance(llm_stream, AsyncIterator):
                    logger.error(f"--- [GUIFocusedAssistant] on_transcript: llm.chat did NOT return an async iterator. Type: {type(llm_stream)}. Cannot process response.")
                    try: logger.error(f"--- [GUIFocusedAssistant] on_transcript: Received object from llm.chat: {llm_stream}")
                    except Exception: pass
                    return

                logger.info(f"--- [GUIFocusedAssistant] on_transcript: self.llm.chat() returned stream (AsyncIterator). Now calling self.handle_llm_response.")
                self.chat_history = current_chat_turn_history
                await self.handle_llm_response(llm_stream)

            except Exception as e_llm_call_or_handle:
                logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error in LLM interaction sequence: {e_llm_call_or_handle}", exc_info=True)

        if is_final:
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

        try:
            logger.debug("--- [GUIFocusedAssistant] handle_llm_response: Starting LLM stream iteration ---")
            async for sentence_obj in self.llm_stream_sentences(llm_stream):
                logger.debug(f"--- [GUIFocusedAssistant] LLM Stream Chunk --- Text: '{sentence_obj.text}', Type: {type(sentence_obj)}")
                if sentence_obj.text:
                    collected_text_for_frontend += sentence_obj.text + " "
                    temp_sentences_for_tts.append(sentence_obj.text)
            llm_stream_finished_successfully = True
            logger.info("--- [GUIFocusedAssistant] handle_llm_response: LLM stream iteration completed ---")
        except Exception as e_llm_stream_processing:
            logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error processing LLM stream: {e_llm_stream_processing}", exc_info=True)
            if not collected_text_for_frontend.strip():
                collected_text_for_frontend = "Error generating response."
            else:
                collected_text_for_frontend += " (Stream processing error)"
        finally:
            final_collected_text = collected_text_for_frontend.strip()
            log_context_info = f"for user query '{self.last_user_query_for_log}'" if self.last_user_query_for_log else "(initial/agent-initiated)"
            if final_collected_text:
                logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: FINAL MODEL_RESPONSE {log_context_info}: '{final_collected_text}'")
                assistant_chat_message = ChatMessage(role="assistant", content=[final_collected_text])
                self.chat_history.append(assistant_chat_message)
                if len(self.chat_history) > 20: self.chat_history = self.chat_history[-20:]
                if self.room and self.room.local_participant:
                    logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Publishing agent response to GUI: '{final_collected_text}'")
                    try:
                        payload_str = f"response:{final_collected_text}"
                        await self.room.local_participant.publish_data(payload=payload_str.encode('utf-8'), topic="response")
                    except Exception as e_pub_resp:
                        logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error publishing agent response: {e_pub_resp}", exc_info=True)
            elif llm_stream_finished_successfully:
                logger.warning(f"--- [GUIFocusedAssistant] handle_llm_response: LLM stream finished, but no text was collected {log_context_info}.")
            else:
                logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: LLM stream failed to produce text {log_context_info}.")

        if temp_sentences_for_tts and any(s.strip() for s in temp_sentences_for_tts) and hasattr(self, 'tts') and self.tts:
            logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Attempting TTS with {len(temp_sentences_for_tts)} sentences.")
            async def gen_tts():
                for s in temp_sentences_for_tts:
                    if s.strip():
                        yield s.strip()
            try:
                if asyncio.iscoroutinefunction(self.tts.synthesize):
                    await self.tts.synthesize(gen_tts())
                else:
                    logger.warning("self.tts.synthesize is not a coroutine function. Attempting direct call.")
                    self.tts.synthesize(gen_tts())
            except Exception as e_tts:
                logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error during TTS synthesis: {e_tts}", exc_info=True)

        self.last_user_query_for_log = None
        logger.info(f"--- [GUIFocusedAssistant] handle_llm_response FINISHED ---")

# --- Agent Entrypoint ---
async def entrypoint(ctx: JobContext):
    logger.info(f"--- [AgentEntry] AGENT LOGIC ENTRYPOINT CALLED --- Room: {ctx.room.name}, Job: {ctx.job.id}")

    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"[AgentEntry] Connected to room. Local Participant SID: {ctx.room.local_participant.sid}")
    except Exception as e_connect:
        logger.critical(f"!!! [AgentEntry] CRITICAL - Agent failed to connect to room: {e_connect}", exc_info=True)
        sys.exit(f"Fatal error during ctx.connect(): {e_connect}")

    logger.info("[AgentEntry] Initializing plugins...")
    llm_plugin = None
    tts_plugin = None
    stt_plugin = None
    vad_plugin = None

    if not OPENAI_API_KEY_ENV:
        logger.critical("!!! [AgentEntry] CRITICAL - OPENAI_API_KEY environment variable is NOT set. OpenAI STT, LLM, and TTS plugins cannot initialize. !!!")
        sys.exit("OPENAI_API_KEY environment variable is not set.")

    try:
        logger.info("[AgentEntry] Initializing OpenAI STT plugin...")
        stt_plugin = openai.STT(api_key=OPENAI_API_KEY_ENV, language="hi")
        logger.info("[AgentEntry] OpenAI STT plugin initialized.")

        logger.info("[AgentEntry] Initializing OpenAI LLM plugin...")
        llm_plugin = openai.LLM(model="gpt-4", api_key=OPENAI_API_KEY_ENV)
        logger.info("[AgentEntry] OpenAI LLM plugin initialized.")

        logger.info("[AgentEntry] Initializing OpenAI TTS plugin...")
        tts_plugin = openai.TTS(voice="nova", api_key=OPENAI_API_KEY_ENV)
        logger.info("[AgentEntry] OpenAI TTS plugin initialized.")

        logger.info("[AgentEntry] Initializing Silero VAD plugin...")
        vad_plugin = silero.VAD.load()
        logger.info("[AgentEntry] Silero VAD plugin initialized.")

        logger.info("[AgentEntry] All required plugins initialized successfully.")
    except Exception as e_plugins:
        logger.critical(f"!!! [AgentEntry] CRITICAL - Error initializing plugins: {e_plugins}", exc_info=True)
        sys.exit(f"Error initializing plugins: {e_plugins}")

    logger.info("[AgentEntry] --- Testing LLM plugin directly START ---")
    llm_test_passed = False
    llm_test_response_collected = ""

    if not llm_plugin:
        logger.error("[AgentEntry] LLM Plugin is None, cannot run direct test.")
    else:
        test_prompt_text_en = "Hello, LLM. Confirm you are working by saying OK."
        test_prompt_text_hi = "नमस्ते। क्या आप हिंदी में बात कर सकते हैं? कृपया हिंदी में जवाब दीजिए।"

        chat_message_list: list[ChatMessage] = []
        try:
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
                logger.info(f"[AgentEntry] LLM Direct Test: Attempting ChatContext(...) (positional)")
                chat_ctx_for_test = ChatContext(chat_message_list)
                logger.info(f"[AgentEntry] LLM Direct Test: ChatContext created.")
            except Exception as e_ctx_create:
                logger.error(f"[AgentEntry] LLM Direct Test: Error creating ChatContext: {e_ctx_create}", exc_info=True)
                chat_ctx_for_test = None

            if chat_ctx_for_test:
                try:
                    logger.info(f"[AgentEntry] LLM Direct Test: Attempting llm_plugin.chat(...)")
                    llm_test_stream = llm_plugin.chat(chat_ctx=chat_ctx_for_test)
                    async for chunk in llm_test_stream:
                        logger.info(f"[LLM Test] Got chunk: {chunk}")
                        chunk_text = getattr(chunk.delta, 'content', None)
                        if chunk_text:
                            llm_test_response_collected += chunk_text
                            logger.debug(f"[AgentEntry] LLM Direct Test Chunk: '{chunk_text}'")

                    logger.info(f"[LLM Test] Final collected response: '{llm_test_response_collected.strip()}'")
                    if llm_test_response_collected.strip():
                        llm_test_passed = True
                        logger.info(f"[AgentEntry] LLM plugin direct test SUCCEEDED. Collected response: '{llm_test_response_collected}'")
                    else:
                        logger.warning("[AgentEntry] LLM plugin direct test: Received stream, but no text chunks were collected.")
                except Exception as e_llm_call:
                    logger.error(f"[AgentEntry] LLM plugin direct test FAILED during chat call or stream processing: {e_llm_call}", exc_info=True)
            else:
                logger.warning("[AgentEntry] LLM Direct Test: chat_ctx_for_test is None, skipping llm.chat() call.")
        else:
            logger.warning("[AgentEntry] LLM Direct Test: chat_message_list is empty.")

    if not llm_test_passed:
        logger.critical("[AgentEntry] !!! LLM direct test FAILED. LLM might not function correctly. !!!")
    else:
        logger.info("[AgentEntry] --- Testing LLM plugin directly END (Successful) ---")

    session = AgentSession(stt=stt_plugin, llm=llm_plugin, tts=tts_plugin, vad=vad_plugin)
    logger.info("[AgentEntry] AgentSession created with OpenAI STT, LLM, TTS and Silero VAD.")

    assistant_agent_instance = GUIFocusedAssistant()
    logger.info("[AgentEntry] GUIFocusedAssistant instance created.")

    logger.info("[AgentEntry] Starting AgentSession with GUIFocusedAssistant...")
    try:
        logger.debug("[AgentEntry] Before session.start()")
        await session.start(room=ctx.room, agent=assistant_agent_instance)
        logger.info("[AgentEntry] AgentSession.start() completed normally.")
    except Exception as e_session_start:
        logger.error(f"[AgentEntry] Error in session.start(): {e_session_start}", exc_info=True)
        raise
    finally:
        logger.info("[AgentEntry] AgentSession.start() block exited.")

    logger.info(f"[AgentEntry] Agent logic finished/terminated for job {ctx.job.id if ctx.job else 'N/A'}.")

def check_and_free_port(port):
    logger.debug(f"Checking if port {port} is in use...")
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.net_connections(kind='inet'):
                    if conn.laddr.port == port:
                        logger.warning(f"Port {port} is in use by PID {proc.pid} ({proc.name()}). Attempting to terminate.")
                        try:
                            proc.terminate()
                            proc.wait(timeout=3)
                            if proc.is_running():
                                logger.warning(f"Process {proc.pid} did not terminate gracefully. Attempting kill.")
                                proc.kill()
                                proc.wait(timeout=3)
                            logger.info(f"Successfully handled potential process on port {port}.")
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as term_err:
                            logger.error(f"Failed to terminate process {proc.pid} on port {port}: {term_err}")
                        except Exception as e:
                            logger.error(f"Unexpected error while handling process {proc.pid} on port {port}: {e}", exc_info=True)
                        return
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception as e:
                logger.error(f"Error iterating process connections for PID {proc.pid}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error iterating processes: {e}", exc_info=True)
    logger.debug(f"Port {port} appears free.")

if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    logger.info("--- __main__: main.py script execution started ---")

    if len(sys.argv) > 1 and sys.argv[1] in ["start", "download-files", "package"]:
        logger.info(f"--- __main__: Command line argument '{sys.argv[1]}' detected. Running cli.run_app directly (Agent Worker mode). ---")
        try:
            cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
        except SystemExit as se:
            logger.info(f"--- __main__: cli.run_app exited SystemExit (code: {se.code}).")
        except Exception as e:
            logger.error(f"--- __main__: CRITICAL ERROR during cli.run_app (Agent Worker): {e}", exc_info=True)
            sys.exit(f"Agent Worker process failed: {e}")
        logger.info("--- __main__: Script execution finished (Agent Worker mode). ---")
    else:
        logger.info("--- __main__: No specific command line argument. Starting FastAPI server (Token Server mode). ---")
        check_and_free_port(8000)
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
            sys.exit(f"FastAPI server failed: {e_uvicorn}")
        logger.info("--- __main__: Uvicorn server has stopped. ---")
        logger.info("--- __main__: Script execution finished (Token Server mode). ---")












# //////////////////////////////////////////////////


# 26-05-25
# openai model complete code of ai voice assistant




# import asyncio
# import os
# import sys
# import logging
# from datetime import timedelta
# # import threading # Not used in the provided snippet, can be removed if not needed elsewhere
# # import time      # Not used in the provided snippet
# import psutil
# from typing import Any, AsyncIterator

# # Reconfigure sys.stdout to use UTF-8 encoding
# try:
#     # In Python 3.7+
#     if sys.stdout.encoding is None or 'utf-8' not in sys.stdout.encoding.lower():
#         sys.stdout.reconfigure(encoding='utf-8', errors='replace') # Use replace for robustness
#     if sys.stderr.encoding is None or 'utf-8' not in sys.stderr.encoding.lower():
#         sys.stderr.reconfigure(encoding='utf-8', errors='replace')
# except AttributeError:
#     # Fallback for older Python versions or specific environments
#     logging.warning("sys.stdout.reconfigure not available. Console output might not handle all Unicode characters perfectly.")
#     pass
# except Exception as e:
#     logging.error(f"Unexpected error during stdout/stderr reconfigure: {e}", exc_info=True)


# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# from livekit.api import AccessToken, VideoGrants
# from livekit import agents, rtc # Ensure rtc is imported
# from livekit.agents import (
#     JobContext,
#     WorkerOptions,
#     cli,
#     Agent,
#     AgentSession,
#     AutoSubscribe,
#     ConversationItemAddedEvent, # <<<<<<< ADDED IMPORT
# )
# from livekit.agents.llm import LLMStream
# from livekit.agents import voice
# from livekit.agents import llm as livekit_llm # Alias is fine
# from livekit.agents.llm import ChatMessage, ChatContext, ChatMessageRole # Import ChatMessageRole if needed for type hints
# import pydantic_core # Keep if required by SDK/plugins
# from livekit.plugins import openai # OpenAI plugins for STT, LLM, TTS
# from livekit.plugins import silero # Silero for VAD

# # --- Load environment variables ---
# load_dotenv()

# LIVEKIT_URL = os.getenv("LIVEKIT_URL")
# LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
# LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
# OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY")
# # ELEVENLABS_API_KEY_ENV = os.getenv("ELEVENLABS_API_KEY") # Not used if using OpenAI TTS
# # DEEPGRAM_API_KEY_ENV = os.getenv("DEEPGRAM_API_KEY")   # Not used if using OpenAI STT

# # --- Configure logging ---
# logging.basicConfig(
#     level=logging.INFO, # Changed to INFO for less verbosity, DEBUG for livekit.agents
#     format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout), # Ensures UTF-8 via reconfigure
#         logging.FileHandler('agent_openai_logs.log', encoding='utf-8') # Specific log file
#     ]
# )

# logger = logging.getLogger(__name__)
# # Formatter is set by basicConfig, no need to re-apply unless changing it for this specific logger

# agent_logger = logging.getLogger("livekit.agents")
# agent_logger.setLevel(logging.DEBUG) # For detailed agent logs
# # agent_logger uses handlers from root by default. If adding specific handlers, set formatters.

# logger.info(f"MODULE LEVEL: LIVEKIT_URL: {LIVEKIT_URL}")
# logger.info(f"MODULE LEVEL: LIVEKIT_API_KEY IS SET: {bool(LIVEKIT_API_KEY)}")
# logger.info(f"MODULE LEVEL: OPENAI_API_KEY IS SET: {bool(OPENAI_API_KEY_ENV)}")

# app = FastAPI()
# origins = ["http://localhost:3000"] # Frontend URL
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/token")
# async def get_token():
#     logger.info("[TokenSvc] /token endpoint hit.")
#     if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
#         logger.error("[TokenSvc] LiveKit credentials not configured properly.")
#         raise HTTPException(status_code=500, detail="LiveKit credentials not configured")
    
#     # Ensure this room name matches what the agent expects if it's hardcoded,
#     # or that the agent joins the room specified by the job request.
#     room_name = "voice-assistant-room-openai" # Using a different room name for clarity
#     participant_identity = "frontend_user"
#     participant_name = "Frontend User"

#     video_grant_obj = VideoGrants(
#         room_join=True,
#         room=room_name, # Use the defined room_name
#         can_publish=True,
#         can_subscribe=True,
#         can_publish_data=True,
#     )
#     token_builder = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
#     token_builder.identity = participant_identity
#     token_builder.name = participant_name
#     token_builder.ttl = timedelta(hours=1)
#     token_builder.with_grants(video_grant_obj)
#     token_jwt = token_builder.to_jwt()

#     logger.info(f"[TokenSvc] Generated token for '{participant_identity}' to join '{room_name}'")
#     return {"token": token_jwt, "url": LIVEKIT_URL, "roomName": room_name} # Return roomName to frontend

# # --- Define the Assistant Agent ---
# class GUIFocusedAssistant(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             instructions=(
#                 "You are Echo, a friendly and helpful voice assistant. Respond concisely. "
#                 "You understand and speak English primarily. " # Simplified instructions for clarity
#                 "If the user speaks in Hindi, try to understand and respond in simple English if possible, or indicate you primarily use English."
#                 # "When the user speaks, understand their query and provide a relevant answer. "
#                 # "Your primary language is Hindi and English, but you can understand multiple languages. "
#                 # "If the user speaks in Hindi or a mix of Hindi, please respond in Hindi using Devanagari script. "
#                 # "Do not use Urdu script or characters. Use only Devanagari for Hindi."
#             )
#         )
#         self.last_user_query_for_log = None
#         # Initialize chat history with system message if your LLM benefits from it
#         self.chat_history: list[ChatMessage] = [
#             ChatMessage(role=ChatMessageRole.SYSTEM, content=self.instructions)
#         ]
#         logger.info("--- [GUIFocusedAssistant] __init__ CALLED (OpenAI version) ---")

#     async def on_transcript(self, transcript: str, is_final: bool) -> None:
#         # logger.debug(f"--- [GUIFocusedAssistant] on_transcript --- Final: {is_final}, Text: '{transcript}'")
#         # print(f"--- DEBUG RAW USER TRANSCRIPT --- Final: {is_final}, Text: '{transcript}'")

#         # Publishing partial transcriptions for live feedback (optional)
#         if not is_final and transcript and self.room and self.room.local_participant:
#             try:
#                 payload_str_partial = f"transcription_update:{transcript}"
#                 # await self.room.local_participant.publish_data(payload=payload_str_partial.encode('utf-8'), topic="transcription_partial")
#             except Exception as e_pub_tx:
#                 logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error publishing partial transcript: {e_pub_tx}", exc_info=True)

#         if is_final and transcript and transcript.strip():
#             self.last_user_query_for_log = transcript.strip()
#             # logger.info(f"--- [GUIFocusedAssistant] on_transcript: Final user query: '{self.last_user_query_for_log}'.")

#             # NOTE: Final user query publishing to frontend is now handled by `_on_conversation_item_added`.
#             # If you were previously publishing "transcription:final_text" from here, it might be redundant.

#             if not hasattr(self, 'llm') or not self.llm:
#                 logger.error("--- [GUIFocusedAssistant] on_transcript: self.llm is not available.")
#                 await self._send_error_response("My LLM is currently unavailable.")
#                 return

#             try:
#                 user_chat_message = ChatMessage(role=ChatMessageRole.USER, content=self.last_user_query_for_log)
                
#                 # Add user message to this agent's internal history copy
#                 # The session history will be updated by the AgentSession framework
#                 current_chat_turn_history = list(self.chat_history)
#                 current_chat_turn_history.append(user_chat_message)

#                 # Simple history trimming
#                 max_history_len = 20 # Keep last N messages + system prompt
#                 if len(current_chat_turn_history) > (max_history_len +1): # +1 for system
#                     current_chat_turn_history = [current_chat_turn_history[0]] + current_chat_turn_history[-(max_history_len):]
                
#                 self.chat_history = current_chat_turn_history # Update agent's view of history

#                 chat_ctx_for_llm = ChatContext(messages=self.chat_history) # Use the agent's current history
#                 # logger.info(f"--- [GUIFocusedAssistant] on_transcript: Calling self.llm.chat() with history (count: {len(self.chat_history)})")

#                 llm_stream = self.llm.chat(chat_ctx=chat_ctx_for_llm)
#                 if not isinstance(llm_stream, AsyncIterator):
#                     logger.error(f"--- [GUIFocusedAssistant] on_transcript: llm.chat did NOT return an async iterator. Type: {type(llm_stream)}.")
#                     await self._send_error_response("Received invalid response from LLM.")
#                     return

#                 await self.handle_llm_response(llm_stream)

#             except Exception as e_llm_interaction:
#                 logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error in LLM interaction: {e_llm_interaction}", exc_info=True)
#                 await self._send_error_response("Error processing your request with the LLM.")

#     async def _send_error_response(self, error_msg: str):
#         logger.warning(f"--- [GUIFocusedAssistant] Sending error response: {error_msg}")
#         if self.room and self.room.local_participant:
#              try:
#                  # Use a distinct prefix or topic for errors if needed, or just "response:"
#                  await self.room.local_participant.publish_data(payload=f"response:Error: {error_msg}".encode('utf-8'), topic="response") # Old topic, fine for errors
#                  if hasattr(self, 'tts') and self.tts:
#                      async def error_gen(): yield error_msg
#                      await self.tts.synthesize(error_gen())
#              except Exception as e_pub_err:
#                  logger.error(f"--- [GUIFocusedAssistant] Failed to publish error message: {e_pub_err}", exc_info=True)

#     async def handle_llm_response(self, llm_stream: LLMStream) -> None:
#         # logger.info(f"--- [GUIFocusedAssistant] handle_llm_response CALLED ---")
#         collected_text_for_frontend = ""
#         temp_sentences_for_tts = []
#         llm_stream_finished_successfully = False

#         try:
#             async for sentence_obj in self.llm_stream_sentences(llm_stream): # Helper for sentence-based streaming
#                 if sentence_obj.text:
#                     collected_text_for_frontend += sentence_obj.text # OpenAI plugin usually sends complete sentences or chunks
#                     temp_sentences_for_tts.append(sentence_obj.text)
#                     # Optional: publish partial responses if desired for live typing effect
#                     # if self.room and self.room.local_participant and sentence_obj.text.strip():
#                     #    partial_payload = f"response_partial:{sentence_obj.text.strip()}"
#                     #    await self.room.local_participant.publish_data(payload=partial_payload.encode('utf-8'), topic="response_partial")
#             llm_stream_finished_successfully = True
#             # logger.info("--- [GUIFocusedAssistant] handle_llm_response: LLM stream iteration completed.")
#         except Exception as e_llm_stream_processing:
#             logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error processing LLM stream: {e_llm_stream_processing}", exc_info=True)
#             collected_text_for_frontend = collected_text_for_frontend.strip() + " (Error during stream processing)" if collected_text_for_frontend.strip() else "Error generating response."
#         finally:
#             final_collected_text = collected_text_for_frontend.strip()
#             log_context_info = f"for user query '{self.last_user_query_for_log}'" if self.last_user_query_for_log else "(agent-initiated)"

#             if final_collected_text:
#                 # logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: FINAL MODEL_RESPONSE {log_context_info}: '{final_collected_text}'")
                
#                 # Add assistant response to this agent's internal history
#                 # The AgentSession framework will handle adding it to the session's history, triggering `conversation_item_added`
#                 assistant_chat_message = ChatMessage(role=ChatMessageRole.ASSISTANT, content=final_collected_text) # OpenAI often returns just text
#                 self.chat_history.append(assistant_chat_message)
#                 if len(self.chat_history) > 21: # +1 for system prompt
#                     self.chat_history = [self.chat_history[0]] + self.chat_history[-20:]

#                 # NOTE: Final agent response publishing to frontend is now handled by `_on_conversation_item_added`.
#                 # If you were previously publishing "response:final_text" from here, it might be redundant.

#             elif llm_stream_finished_successfully:
#                 logger.warning(f"--- [GUIFocusedAssistant] handle_llm_response: LLM stream finished, but no text collected {log_context_info}.")
#                 await self._send_error_response("I couldn't generate a response that time.")
#             else: # Stream processing error and no text collected
#                 logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: LLM stream failed to produce text {log_context_info}.")
#                 # An error message might have already been set if the except block was hit and text was empty

#         if temp_sentences_for_tts and any(s.strip() for s in temp_sentences_for_tts) and hasattr(self, 'tts') and self.tts:
#             # logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Attempting TTS with {len(temp_sentences_for_tts)} text segments.")
#             async def gen_tts_stream():
#                 for s_chunk in temp_sentences_for_tts:
#                     if s_chunk.strip():
#                         yield s_chunk.strip()
#             try:
#                 await self.tts.synthesize(gen_tts_stream())
#             except Exception as e_tts:
#                 logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error during TTS synthesis: {e_tts}", exc_info=True)

#         self.last_user_query_for_log = None # Reset after processing
#         # logger.info(f"--- [GUIFocusedAssistant] handle_llm_response FINISHED ---")


# # --- Agent Entrypoint ---
# async def entrypoint(ctx: JobContext):
#     logger.info(f"--- [AgentEntry] AGENT LOGIC ENTRYPOINT (OpenAI) --- Room: {ctx.room.name}, Job: {ctx.job.id}")

#     try:
#         await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
#         logger.info(f"[AgentEntry] Connected to room '{ctx.room.name}'. Local SID: {ctx.room.local_participant.sid}")
#     except Exception as e_connect:
#         logger.critical(f"!!! [AgentEntry] CRITICAL - Agent failed to connect: {e_connect}", exc_info=True)
#         sys.exit(f"Fatal error during ctx.connect(): {e_connect}")

#     logger.info("[AgentEntry] Initializing plugins...")
#     stt_plugin, llm_plugin, tts_plugin, vad_plugin = None, None, None, None

#     if not OPENAI_API_KEY_ENV:
#         logger.critical("!!! [AgentEntry] CRITICAL - OPENAI_API_KEY env var is NOT set. Exiting. !!!")
#         sys.exit("OPENAI_API_KEY environment variable is not set.")

#     try:
#         logger.info("[AgentEntry] Initializing OpenAI STT plugin (language='en')...") # Defaulting to 'en' for OpenAI STT
#         stt_plugin = openai.STT(api_key=OPENAI_API_KEY_ENV, language="en") # Set language for STT
#         logger.info("[AgentEntry] OpenAI STT plugin initialized.")

#         logger.info("[AgentEntry] Initializing OpenAI LLM plugin (model='gpt-4o')...") # Using gpt-4o
#         llm_plugin = openai.LLM(model="gpt-4o", api_key=OPENAI_API_KEY_ENV)
#         logger.info("[AgentEntry] OpenAI LLM plugin initialized.")

#         logger.info("[AgentEntry] Initializing OpenAI TTS plugin (voice='nova')...")
#         tts_plugin = openai.TTS(voice="nova", api_key=OPENAI_API_KEY_ENV) # 'nova' is a good general voice
#         logger.info("[AgentEntry] OpenAI TTS plugin initialized.")

#         logger.info("[AgentEntry] Initializing Silero VAD plugin...")
#         vad_plugin = silero.VAD.load()
#         logger.info("[AgentEntry] Silero VAD plugin initialized.")
#         logger.info("[AgentEntry] All required plugins initialized successfully.")
#     except Exception as e_plugins:
#         logger.critical(f"!!! [AgentEntry] CRITICAL - Error initializing plugins: {e_plugins}", exc_info=True)
#         sys.exit(f"Error initializing plugins: {e_plugins}")

#     # LLM Direct Test (optional but good for sanity check)
#     # ... (You can adapt the LLM test from the previous script if needed) ...

#     session = AgentSession(stt=stt_plugin, llm=llm_plugin, tts=tts_plugin, vad=vad_plugin)
#     logger.info("[AgentEntry] AgentSession created with OpenAI STT, LLM, TTS and Silero VAD.")

#     # -------- START: Event Handler for conversation_item_added (same as before) --------
#     async def _publish_conversation_item_async(event_item: ChatMessage):
#         item_role_str = "unknown"
#         item_text_content = ""
#         if event_item:
#             if event_item.role:
#                 item_role_str = str(event_item.role.value) if hasattr(event_item.role, 'value') else str(event_item.role)
#             if hasattr(event_item, 'text_content') and event_item.text_content:
#                 item_text_content = event_item.text_content
#             elif hasattr(event_item, 'content'):
#                 if isinstance(event_item.content, list):
#                     str_parts = [str(part) for part in event_item.content if isinstance(part, str)]
#                     item_text_content = " ".join(str_parts).strip()
#                 elif isinstance(event_item.content, str):
#                     item_text_content = event_item.content
        
#         if ctx.room and ctx.room.local_participant and item_text_content:
#             payload_to_send = ""
#             topic_to_send = "lk_chat_history"
#             if item_role_str.lower() == "user":
#                 payload_to_send = f"user_msg:{item_text_content}"
#             elif item_role_str.lower() == "assistant":
#                 payload_to_send = f"agent_msg:{item_text_content}"
#                 logger.info(f"--- AGENT'S RESPONSE (via event, publishing) --- '{item_text_content}'")

#             if payload_to_send:
#                 logger.info(f"[AgentEntry] Publishing to frontend via '{topic_to_send}': '{payload_to_send}'")
#                 try:
#                     await ctx.room.local_participant.publish_data(
#                         payload=payload_to_send.encode('utf-8'), topic=topic_to_send)
#                 except Exception as e_pub:
#                     logger.error(f"[AgentEntry] Error publishing from _publish_conversation_item_async: {e_pub}", exc_info=True)

#     def _on_conversation_item_added_sync(event: ConversationItemAddedEvent):
#         item_role_str = "unknown"
#         item_text_content = ""
#         if event.item:
#             if event.item.role:
#                 item_role_str = str(event.item.role.value) if hasattr(event.item.role, 'value') else str(event.item.role)
#             if hasattr(event.item, 'text_content') and event.item.text_content:
#                 item_text_content = event.item.text_content
#             elif hasattr(event.item, 'content'):
#                 if isinstance(event.item.content, list):
#                     str_parts = [str(part) for part in event.item.content if isinstance(part, str)]
#                     item_text_content = " ".join(str_parts).strip()
#                 elif isinstance(event.item.content, str):
#                     item_text_content = event.item.content
#         logger.info(f"[AgentSession Event] Item Added (Sync Handler): Role='{item_role_str}', Text='{item_text_content}'")
#         if event.item:
#             asyncio.create_task(_publish_conversation_item_async(event.item))

#     session.on("conversation_item_added", _on_conversation_item_added_sync)
#     # -------- END: Event Handler for conversation_item_added --------

#     assistant_agent_instance = GUIFocusedAssistant()
#     logger.info("[AgentEntry] GUIFocusedAssistant (OpenAI) instance created.")

#     logger.info("[AgentEntry] Starting AgentSession with GUIFocusedAssistant (OpenAI)...")
#     try:
#         await session.start(room=ctx.room, agent=assistant_agent_instance)
#         logger.info("[AgentEntry] AgentSession.start() completed normally.")
#     except Exception as e_session_start:
#         logger.error(f"[AgentEntry] Error in session.start(): {e_session_start}", exc_info=True)
#         raise
#     finally:
#         logger.info("[AgentEntry] AgentSession.start() block exited.")

#     logger.info(f"[AgentEntry] Agent logic finished/terminated for job {ctx.job.id if ctx.job else 'N/A'}.")

# # (check_and_free_port function as before - it's a utility)
# def check_and_free_port(port):
#     logger.debug(f"Checking if port {port} is in use...")
#     try:
#         for proc in psutil.process_iter(['pid', 'name']):
#             try:
#                 for conn in proc.net_connections(kind='inet'):
#                     if conn.laddr.port == port:
#                         logger.warning(f"Port {port} is in use by PID {proc.pid} ({proc.name()}). Attempting to terminate.")
#                         try:
#                             proc.terminate()
#                             proc.wait(timeout=3)
#                             if proc.is_running():
#                                 logger.warning(f"Process {proc.pid} did not terminate gracefully. Attempting kill.")
#                                 proc.kill()
#                                 proc.wait(timeout=3)
#                             logger.info(f"Successfully handled potential process on port {port}.")
#                         except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as term_err:
#                             logger.error(f"Failed to terminate process {proc.pid} on port {port}: {term_err}")
#                         except Exception as e: # Catch more general exceptions during process handling
#                             logger.error(f"Unexpected error while handling process {proc.pid} on port {port}: {e}", exc_info=True)
#                         return # Port issue addressed, no need to check further for this port
#             except (psutil.NoSuchProcess, psutil.AccessDenied): # Process might have ended or access denied
#                 continue
#             except Exception as e: # Catch errors during net_connections iteration
#                 logger.error(f"Error iterating process connections for PID {proc.pid}: {e}", exc_info=True)
#     except Exception as e: # Catch errors during psutil.process_iter
#         logger.error(f"Error iterating processes to check port {port}: {e}", exc_info=True)
#     logger.debug(f"Port {port} appears free or could not be definitively checked/cleared.")


# if __name__ == "__main__":
#     import faulthandler
#     faulthandler.enable() # For better crash reports
    
#     # Determine script name for Uvicorn (assuming this file is 'main.py' or 'openai_assistant.py')
#     # If your script is named e.g. openai_agent.py, change SCRIPT_NAME_FOR_UVICORN
#     # This extracts the filename without the .py extension
#     script_filename = os.path.splitext(os.path.basename(__file__))[0]
#     SCRIPT_NAME_FOR_UVICORN = script_filename 
#     logger.info(f"--- __main__: Script '{SCRIPT_NAME_FOR_UVICORN}.py' execution started ---")


#     if len(sys.argv) > 1 and sys.argv[1] in ["start", "download-files", "package"]:
#         logger.info(f"--- __main__: Command '{sys.argv[1]}' detected. Running Agent Worker mode. ---")
#         try:
#             # WorkerOptions now correctly passed to cli.run_app
#             worker_opts = WorkerOptions(entrypoint_fnc=entrypoint)
#             cli.run_app(worker_opts)
#         except SystemExit as se: # Catch SystemExit from cli.run_app or sys.exit calls
#             if se.code == 0:
#                 logger.info(f"--- __main__: Agent Worker exited gracefully (code: {se.code}).")
#             else:
#                 logger.warning(f"--- __main__: Agent Worker exited with code: {se.code}.")
#         except Exception as e: # Catch any other unexpected errors during agent run
#             logger.critical(f"!!! __main__: CRITICAL ERROR in Agent Worker: {e}", exc_info=True)
#             sys.exit(f"Agent Worker process failed critically: {e}") # Ensure exit on critical failure
#         logger.info("--- __main__: Agent Worker mode finished. ---")
#     else:
#         logger.info("--- __main__: No specific command. Starting FastAPI server (Token Server mode). ---")
#         check_and_free_port(8000) # Check/free port 8000 for FastAPI
#         logger.info("--- __main__: Starting Uvicorn server for FastAPI app ---")
#         try:
#             uvicorn.run(
#                 f"{SCRIPT_NAME_FOR_UVICORN}:app", # Correctly uses the determined script name
#                 host="0.0.0.0",
#                 port=8000,
#                 reload=False, # reload=True can cause issues with agents/multiprocessing
#                 log_level="info" # Uvicorn's own log level
#             )
#         except Exception as e_uvicorn: # Catch errors during Uvicorn server startup/run
#             logger.critical(f"!!! __main__: Uvicorn server FAILED: {e_uvicorn}", exc_info=True)
#             sys.exit(f"FastAPI server failed to start: {e_uvicorn}") # Ensure exit
#         logger.info("--- __main__: Uvicorn server has stopped. ---")
#     logger.info(f"--- __main__: Script '{SCRIPT_NAME_FOR_UVICORN}.py' execution finished. ---")