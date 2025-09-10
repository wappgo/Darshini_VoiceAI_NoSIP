












# import asyncio
# import os
# import sys
# import logging
# import io # Import io for potential stdout reconfigure errors
# from datetime import timedelta
# import psutil
# from typing import Any, AsyncIterator

# # Attempt to reconfigure sys.stdout/stderr to use UTF-8
# # This helps ensure console output can display Hindi characters correctly,
# # although it depends on the terminal's own encoding settings.
# try:
#     if sys.stdout.encoding is None or 'utf-8' not in sys.stdout.encoding.lower():
#          sys.stdout.reconfigure(encoding='utf-8', errors='replace') # Use 'replace' for errors
#     if sys.stderr.encoding is None or 'utf-8' not in sys.stderr.encoding.lower():
#          sys.stderr.reconfigure(encoding='utf-8', errors='replace')
# except (AttributeError, io.UnsupportedOperation) as e:
#     # This can happen in some environments (e.g., certain IDEs, background tasks)
#     logging.warning(f"Failed to reconfigure stdout/stderr encoding to UTF-8: {e}. Console output might not display non-ASCII characters correctly.")
# except Exception as e:
#      logging.error(f"Unexpected error during stdout/stderr reconfigure: {e}", exc_info=True)


# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
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
# import pydantic_core # Keep if needed by plugins/SDK versions, but might be removable
# # Import the specific plugins needed: deepgram (for STT), groq (for LLM), elevenlabs (for TTS), silero (for VAD)
# from livekit.plugins import deepgram, groq, elevenlabs, silero

# # --- Load environment variables and set up module-level variables immediately ---
# load_dotenv()

# LIVEKIT_URL = os.getenv("LIVEKIT_URL")
# LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
# LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
# DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY") # Need this back for Deepgram STT
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# # OPENAI_API_KEY_ENV is no longer needed for STT, can remove or ignore

# # --- Configure logging with console and file output ---
# # Note: JSON logs from livekit.agents may show Unicode escapes (e.g., \u0939 for ह).
# # Check agent_logs.log or console for readable Devanagari text.
# # Setting level to INFO for general operation, DEBUG for more verbose agent logs
# logging.basicConfig(
#     level=logging.INFO, # Changed to INFO, agent_logger will be DEBUG
#     format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout), # Console output
#         logging.FileHandler('agent_logs.log', encoding='utf-8') # File output with UTF-8
#     ]
# )

# # Ensure main logger uses UTF-8 handlers (redundant with basicConfig above but good practice)
# logger = logging.getLogger(__name__)
# for handler in logger.handlers:
#     if isinstance(handler, logging.StreamHandler):
#          handler.setStream(sys.stdout) # Reset stream just in case
#     if hasattr(handler, 'encoding'):
#         handler.encoding = 'utf-8' # Ensure encoding is set
#     handler.setFormatter(logging.Formatter( # Apply formatter to all handlers
#         '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     ))


# # Set livekit.agents logger to DEBUG for detailed agent processing logs
# agent_logger = logging.getLogger("livekit.agents")
# agent_logger.setLevel(logging.DEBUG)
# # Ensure agent_logger also uses UTF-8 handlers from basicConfig
# for handler in agent_logger.handlers:
#      if isinstance(handler, logging.StreamHandler):
#          handler.setStream(sys.stdout)
#      if hasattr(handler, 'encoding'):
#         handler.encoding = 'utf-8'
#      # Use the same formatter for consistent look
#      handler.setFormatter(logging.Formatter(
#         '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     ))


# # Log module-level variables
# logger.info(f"MODULE LEVEL: LIVEKIT_URL: {LIVEKIT_URL}")
# logger.info(f"MODULE LEVEL: LIVEKIT_API_KEY IS SET: {bool(LIVEKIT_API_KEY)}")
# logger.info(f"MODULE LEVEL: DEEPGRAM_API_KEY IS SET: {bool(DEEPGRAM_API_KEY)}") # Log Deepgram key status
# logger.info(f"MODULE LEVEL: GROQ_API_KEY IS SET: {bool(GROQ_API_KEY)}")
# logger.info(f"MODULE LEVEL: ELEVENLABS_API_KEY IS SET: {bool(ELEVENLABS_API_KEY)}")

# app = FastAPI()

# origins = [
#     "http://localhost:3000", # Allow frontend running on 3000
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
#         raise HTTPException(
#             status_code=500,
#             detail="LiveKit credentials (URL, API Key, Secret) not configured"
#         )

#     room_name = "voice-assistant-room"
#     participant_identity = "frontend_user"
#     participant_name = "Frontend User"

#     video_grant_obj = VideoGrants(
#         room_join=True,
#         room=room_name,
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

#     logger.info(f"[TokenSvc] Generated token for identity '{participant_identity}' to join room '{room_name}'")
#     return {"token": token_jwt, "url": LIVEKIT_URL, "roomName": room_name}


# # --- Define the Assistant Agent ---
# class GUIFocusedAssistant(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             instructions=(
#                 "You are khushi, a friendly and helpful voice assistant. "
#                 "Respond concisely and directly answer the user's query. "
#                 "You understand languages including English and Hindi. which is your primary language "
#                 "If the user speaks primarily in Hindi or mixes Hindi, respond concisely in Hindi using the Devanagari script. Do not use Urdu script or characters."
#                 "If the user speaks primarily in English, respond concisely in English."
#                 "Maintain context from the conversation history."
#             )
#         )
#         self.last_user_query_for_log = None
#         self.chat_history: list[ChatMessage] = [
#              ChatMessage(role="system", content=[self.instructions])
#         ]
#         logger.info("--- [GUIFocusedAssistant] __init__ CALLED --- Agent initialized.")

#     async def on_transcript(self, transcript: str, is_final: bool) -> None:
#         # This method receives the output directly from the configured STT engine (Deepgram)
#         # Since Deepgram is configured with language="hi", the 'transcript' variable
#         # should contain Hindi text when the user speaks Hindi.

#         # Log the raw transcript received from the STT engine
#         # This is where you see the Hindi text logged
#         log_level = logging.INFO if is_final else logging.DEBUG
#         logger.log(log_level, f"--- [GUIFocusedAssistant] on_transcript --- Final: {is_final}, Text: '{transcript}'")

#         # Add extra logging specifically for the raw STT output, potentially bypassing formatting issues
#         print(f"--- DEBUG RAW STT OUTPUT --- Final: {is_final}, Text: '{transcript}'")


#         if is_final and transcript and transcript.strip():
#             # Store the final, non-empty transcript for logging context later
#             self.last_user_query_for_log = transcript.strip()

#             logger.info(f"--- [GUIFocusedAssistant] on_transcript: Final user query received: '{self.last_user_query_for_log}'. Attempting to generate LLM response.")

#             # Ensure LLM plugin is available
#             if not hasattr(self, 'llm') or not self.llm:
#                 logger.error("--- [GUIFocusedAssistant] on_transcript: self.llm is not available. Cannot generate LLM response.")
#                 # Potentially send an error message to the frontend
#                 await self._send_error_response("Sorry, my brain is not available right now.")
#                 return

#             try:
#                 # Create ChatMessage for current user transcript
#                 user_chat_message = ChatMessage(role="user", content=[self.last_user_query_for_log])

#                 # Build the chat context for the LLM, including history
#                 current_chat_turn_history = list(self.chat_history) # Create a copy
#                 current_chat_turn_history.append(user_chat_message)

#                 # Simple history trimming (keep system message + last N user/assistant turns)
#                 # Assuming system message is at index 0
#                 max_history_length = 21 # System message + 20 turns (10 user, 10 assistant)
#                 if len(current_chat_turn_history) > max_history_length:
#                      current_chat_turn_history = [current_chat_turn_history[0]] + current_chat_turn_history[-(max_history_length - 1):]


#                 # Create ChatContext - LiveKit SDK ChatContext expects list of messages positionally
#                 chat_ctx_for_llm = ChatContext(current_chat_turn_history)

#                 logger.debug(f"--- [GUIFocusedAssistant] on_transcript: Calling self.llm.chat(...) with history (count: {len(current_chat_turn_history)}). History: {current_chat_turn_history}")

#                 # Call the LLM asynchronously
#                 # Correct way to get the async iterator - NO await here
#                 llm_stream = self.llm.chat(chat_ctx=chat_ctx_for_llm)

#                 # Check if the returned object is an async iterator as expected
#                 if not isinstance(llm_stream, AsyncIterator):
#                     logger.error(f"--- [GUIFocusedAssistant] on_transcript: llm.chat did NOT return an async iterator. Type: {type(llm_stream)}. Cannot process response.")
#                     try: logger.error(f"--- [GUIFocusedAssistant] on_transcript: Received object representation: {llm_stream}")
#                     except Exception: pass # Avoid error looping
#                     await self._send_error_response("Sorry, I received an invalid response from the LLM.")
#                     return

#                 logger.info(f"--- [GUIFocusedAssistant] on_transcript: self.llm.chat() returned stream (AsyncIterator). Now calling self.handle_llm_response.")

#                 # Update persistent chat history with the user message BEFORE calling handle_llm_response
#                 self.chat_history = current_chat_turn_history # Save the history used for this turn

#                 # Process the LLM response stream
#                 await self.handle_llm_response(llm_stream)

#             except Exception as e_llm_interaction:
#                 logger.error(f"--- [GUIFocusedAssistant] on_transcript: Uncaught error in LLM interaction sequence: {e_llm_interaction}", exc_info=True)
#                 await self._send_error_response("Sorry, I encountered an error trying to generate a response.")


#         # Always publish the (potentially non-final) transcript to the frontend
#         if self.room and self.room.local_participant:
#             try:
#                 payload_str = f"transcription_update:{transcript}" if not is_final else f"transcription:{transcript}"
#                 await self.room.local_participant.publish_data(payload=payload_str.encode('utf-8'), topic="transcription")
#             except Exception as e_pub_tx:
#                 logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error publishing user transcript chunk: {e_pub_tx}", exc_info=True)

#     # Helper to send error message to frontend and optionally TTS
#     async def _send_error_response(self, error_msg: str):
#         logger.warning(f"--- [GUIFocusedAssistant] Sending error response: {error_msg}")
#         if self.room and self.room.local_participant:
#              try:
#                  await self.room.local_participant.publish_data(payload=f"response:{error_msg}".encode('utf-8'), topic="response")
#                  if hasattr(self, 'tts') and self.tts:
#                      # Using synthesize with an awaitable that yields strings
#                      async def error_gen(): yield error_msg
#                      await self.tts.synthesize(error_gen())
#              except Exception as e_pub_err:
#                  logger.error(f"--- [GUIFocusedAssistant] Failed to publish error message: {e_pub_err}", exc_info=True)

#     async def handle_llm_response(self, llm_stream: LLMStream) -> None:
#         logger.info(f"--- [GUIFocusedAssistant] handle_llm_response CALLED --- Starting stream processing.")
#         collected_text_for_frontend = ""
#         temp_sentences_for_tts = [] # Collect sentence chunks for TTS
#         llm_stream_finished_successfully = False

#         try:
#             logger.debug("--- [GUIFocusedAssistant] handle_llm_response: Starting LLM stream iteration via llm_stream_sentences ---")
#             sentence_count = 0
#             # llm_stream_sentences yields objects with a 'text' attribute
#             async for sentence_obj in self.llm_stream_sentences(llm_stream):
#                 logger.info(f"[PARTIAL] LLM sentence: '{sentence_obj.text}'")
#                 sentence_count += 1
#                 logger.debug(f"--- [GUIFocusedAssistant] LLM Stream Sentence ({sentence_count}) --- Text: '{sentence_obj.text}'")
#                 if sentence_obj.text:
#                     collected_text_for_frontend += sentence_obj.text + " "
#                     temp_sentences_for_tts.append(sentence_obj.text)


#                 # Publish partial text to frontend as sentences complete
#                 if self.room and self.room.local_participant and sentence_obj.text and sentence_obj.text.strip():
#                      try:
#                          partial_payload = f"response_partial:{sentence_obj.text.strip()}"
#                          await self.room.local_participant.publish_data(payload=partial_payload.encode('utf-8'), topic="response_partial") # Using a dedicated topic for partials
#                          logger.debug(f"--- [GUIFocusedAssistant] handle_llm_response: Published partial response: '{sentence_obj.text.strip()}'")
#                      except Exception as e_pub_partial:
#                          logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error publishing partial response: {e_pub_partial}", exc_info=True)


#             llm_stream_finished_successfully = True
#             logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: LLM stream iteration completed. Collected {sentence_count} sentences. ---")

#         except Exception as e_llm_stream_processing:
#             logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error processing LLM stream: {e_llm_stream_processing}", exc_info=True)
#             if not collected_text_for_frontend.strip():
#                 collected_text_for_frontend = "Error generating response."
#             else:
#                 collected_text_for_frontend += " (Stream processing error)"
#         finally:
#             final_collected_text = collected_text_for_frontend.strip()
#             logger.info(f"[DEBUG] Final response collected for query '{self.last_user_query_for_log}': '{final_collected_text}'")
#             log_context_info = f"for user query '{self.last_user_query_for_log}'" if self.last_user_query_for_log else "(initial/agent-initiated)"

#             if final_collected_text:
#                 logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: FINAL MODEL_RESPONSE {log_context_info}: '{final_collected_text}'")
#                 print(f"[DEBUG CONSOLE PRINT] MODEL FINAL RESPONSE: {final_collected_text}")
#                 import sys
#                 sys.stdout.flush()

#                 # Add assistant response to chat history
#                 # If the last message is from the assistant, append to its content, otherwise add new message
#                 if self.chat_history and self.chat_history[-1].role == "assistant":
#                      last_msg = self.chat_history[-1]
#                      if not isinstance(last_msg.content, list):
#                           last_msg.content = [str(last_msg.content)] if last_msg.content is not None else [""]
#                      last_msg.content[0] = (last_msg.content[0].strip() + " " + final_collected_text).strip()
#                      logger.debug("--- [GUIFocusedAssistant] handle_llm_response: Appended to last assistant message.")
#                 else:
#                     assistant_chat_message = ChatMessage(role="assistant", content=[final_collected_text])
#                     self.chat_history.append(assistant_chat_message)
#                     logger.debug("--- [GUIFocusedAssistant] handle_llm_response: Added new assistant message.")

#                 # Trim history again after adding assistant message
#                 max_history_length = 21 # System message + 20 turns
#                 if len(self.chat_history) > max_history_length:
#                      self.chat_history = [self.chat_history[0]] + self.chat_history[-(max_history_length - 1):]


#                 # Publish final response to frontend
#                 if self.room and self.room.local_participant:
#                     logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Publishing FINAL agent response to GUI: '{final_collected_text}'")
#                     try:
#                         payload_str = f"response:{final_collected_text}"
#                         # Use the main 'response' topic for the final complete text
#                         await self.room.local_participant.publish_data(payload=payload_str.encode('utf-8'), topic="response")
#                     except Exception as e_pub_resp:
#                         logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error publishing final agent response: {e_pub_resp}", exc_info=True)
#             elif llm_stream_finished_successfully:
#                 logger.warning(f"--- [GUIFocusedAssistant] handle_llm_response: LLM stream finished, but no text was collected {log_context_info}.")
#                 # Potentially send a "no response" message to frontend/user
#                 await self._send_error_response("Sorry, I couldn't generate a response.")

#             else:
#                 logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: LLM stream failed to produce text {log_context_info}.")
#                 # Error handling for stream processing might already have added text,
#                 # but if not, _send_error_response can be called here too if needed.


#         # Trigger TTS synthesis for collected sentences if TTS plugin is available
#         # Only synthesize if there are sentences AND the stream processed at least some text successfully
#         if temp_sentences_for_tts and any(s.strip() for s in temp_sentences_for_tts) and hasattr(self, 'tts') and self.tts:
#             logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Attempting TTS with {len(temp_sentences_for_tts)} text chunks.")
#             # Create an async generator for the text chunks
#             async def gen_tts_stream():
#                 for s in temp_sentences_for_tts:
#                     if s.strip():
#                         yield s.strip() # Yield sentence chunks

#             try:
#                  logger.debug("--- [GUIFocusedAssistant] Calling self.tts.synthesize(...)")
#                  # synthesize expects an async iterator that yields strings/bytes
#                  await self.tts.synthesize(gen_tts_stream())
#                  logger.info("--- [GUIFocusedAssistant] TTS synthesis call completed.")
#             except Exception as e_tts:
#                 logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error during TTS synthesis: {e_tts}", exc_info=True)
#                 # If TTS fails after getting LLM response, could log it or send a specific error message to frontend

#         self.last_user_query_for_log = None # Reset query log context after handling
#         logger.info(f"--- [GUIFocusedAssistant] handle_llm_response FINISHED ---")


# # --- Agent Entrypoint ---
# async def entrypoint(ctx: JobContext):
#     logger.info(f"--- [AgentEntry] AGENT LOGIC ENTRYPOINT CALLED --- Room: {ctx.room.name}, Job: {ctx.job.id}")

#     try:
#         # Connect the agent to the room, automatically subscribing to audio tracks.
#         # AUDIO_ONLY is often sufficient for voice assistants.
#         await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
#         logger.info(f"[AgentEntry] Successfully connected to room '{ctx.room.name}'. Local Participant SID: {ctx.room.local_participant.sid}")
#     except Exception as e_connect:
#         logger.critical(f"!!! [AgentEntry] CRITICAL - Agent failed to connect to room '{ctx.room.name}': {e_connect}", exc_info=True)
#         # Exit the process if connection fails - cannot proceed
#         sys.exit(f"Fatal error during ctx.connect(): {e_connect}")

#     logger.info("[AgentEntry] Initializing plugins...")
#     # Initialize plugin variables to None
#     stt_plugin = None
#     llm_plugin = None
#     tts_plugin = None
#     vad_plugin = None

#     # --- Environment Variable Checks ---
#     # Perform critical checks before attempting plugin initialization
#     missing_keys = []
#     if not DEEPGRAM_API_KEY: missing_keys.append("DEEPGRAM_API_KEY") # Need Deepgram key back
#     if not GROQ_API_KEY: missing_keys.append("GROQ_API_KEY")
#     if not ELEVENLABS_API_KEY: missing_keys.append("ELEVENLABS_API_KEY")
#     if not LIVEKIT_URL or not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET: missing_keys.append("LIVEKIT_URL/API_KEY/API_SECRET")

#     if missing_keys:
#         logger.critical(f"!!! [AgentEntry] CRITICAL - Missing required environment variables: {', '.join(missing_keys)}. Cannot initialize plugins or connect to LiveKit. !!!")
#         sys.exit(f"Missing required environment variables: {', '.join(missing_keys)}. Please check your .env file.")

#     logger.info("[AgentEntry] Initializing plugins...")

#     # --- Debug prints (move them here) ---
#     import os
#     import sys

#     # The correct import is already at the top: from livekit.plugins import deepgram

#     logger.debug(f"[AgentEntry] DEBUG: Script path: {os.path.abspath(__file__)}")
#     logger.debug(f"[AgentEntry] DEBUG: Python Executable: {sys.executable}")
#     try:
#         # This tries to get the package version from metadata
#         from importlib.metadata import version, PackageNotFoundError
#         try:
#             plugin_version = version("livekit-plugins-deepgram")
#             logger.debug(f"[AgentEntry] DEBUG: livekit-plugins-deepgram package version: {plugin_version}")
#         except PackageNotFoundError:
#             logger.debug("[AgentEntry] DEBUG: livekit-plugins-deepgram package version not found via metadata (maybe older pip or manual install).")
#     except ImportError:
#          logger.debug("[AgentEntry] DEBUG: importlib.metadata not available (Python < 3.8).")

#     try:
#         # This shows the path to the loaded module itself
#         # It relies on the 'from livekit.plugins import deepgram' being successful
#         if 'livekit.plugins.deepgram' in sys.modules:
#              logger.debug(f"[AgentEntry] DEBUG: livekit.plugins.deepgram module path: {livekit.plugins.deepgram.__file__}")
#         else:
#              logger.debug("[AgentEntry] DEBUG: livekit.plugins.deepgram module not found in sys.modules.")
#     except Exception as e:
#          logger.debug(f"[AgentEntry] DEBUG: Failed to get deepgram module path: {e}")
#     # --- END DEBUG PRINTS ---


#     # --- Plugin Initialization ---
#     try:
#         logger.info("[AgentEntry] Initializing Deepgram STT plugin (language='hi')...")

#         # Make SURE you have these parameters explicitly set to False
#         stt_plugin = deepgram.STT(
#             api_key=DEEPGRAM_API_KEY,
#             # language="hi",  # fallback to English
#             model="nova-2-general",  # more stable than nova-3 for now, optional

#         )
#         logger.info("[AgentEntry] Deepgram STT plugin initialized with language='hi'.")

#         logger.info("[AgentEntry] Initializing Groq LLM plugin (model='llama3-70b-8192')...")
#         # Using a Llama3 model from Groq
#         llm_plugin = groq.LLM(model="llama3-70b-8192", api_key=GROQ_API_KEY)
#         logger.info("[AgentEntry] Groq LLM plugin initialized.")

#         logger.info("[AgentEntry] Initializing ElevenLabs TTS plugin...")
#         # --- FIX: Removed 'voice' argument as per previous error ---
#         tts_plugin = elevenlabs.TTS(api_key=ELEVENLABS_API_KEY)
#         logger.info("[AgentEntry] ElevenLabs TTS plugin initialized (using default voice for key).")

#         logger.info("[AgentEntry] Initializing Silero VAD plugin...")
#         # Silero VAD is used for detecting speech activity
#         vad_plugin = silero.VAD.load()
#         logger.info("[AgentEntry] Silero VAD plugin initialized.")

#         logger.info("[AgentEntry] All required plugins initialized successfully.")
#     except Exception as e_plugins:
#         logger.critical(f"!!! [AgentEntry] CRITICAL - Error initializing one or more plugins: {e_plugins}", exc_info=True)
#         # Exit the process if plugins fail to initialize - agent cannot function
#         sys.exit(f"Error initializing plugins: {e_plugins}")


#     # --- Test LLM plugin directly ---
#     # This is a useful check to ensure the LLM is working and responding in expected ways
#     logger.info("[AgentEntry] --- Testing LLM plugin directly START ---")
#     llm_test_passed = False
#     llm_test_response_collected = ""

#     if not llm_plugin:
#         logger.error("[AgentEntry] LLM Plugin is None after initialization attempt, skipping direct test.")
#     else:
#         # Test prompt includes both English and Hindi to check multilingual capability
#         test_prompt_text_en = "Hello, LLM. Confirm you are working by saying OK."
#         test_prompt_text_hi = "नमस्ते। क्या आप हिंदी में बात कर सकते हैं? कृपया हिंदी में जवाब दीजिए।"

#         chat_message_list: list[ChatMessage] = []
#         try:
#             # Creating chat messages with content as a list, as per SDK structure
#             chat_message_list = [
#                 ChatMessage(role="system", content=["You are a helpful AI assistant. Respond concisely."]),
#                 ChatMessage(role="user", content=["Test 1 (EN): " + test_prompt_text_en]),
#                 ChatMessage(role="user", content=["Test 2 (HI): " + test_prompt_text_hi]) # Add Hindi test
#             ]
#             logger.info(f"[AgentEntry] LLM Direct Test: Created chat_message_list for tests (count: {len(chat_message_list)}).")
#         except Exception as e_cm_create:
#             logger.error(f"[AgentEntry] LLM Direct Test: FAILED to create ChatMessage list: {e_cm_create}", exc_info=True)

#         if chat_message_list:
#             chat_ctx_for_test: ChatContext | None = None
#             try:
#                 # Create ChatContext with the list of messages (positional argument)
#                 chat_ctx_for_test = ChatContext(chat_message_list)
#                 logger.info(f"[AgentEntry] LLM Direct Test: ChatContext created.")
#             except Exception as e_ctx_create:
#                 logger.error(f"[AgentEntry] LLM Direct Test: Error creating ChatContext: {e_ctx_create}", exc_info=True)
#                 chat_ctx_for_test = None

#             if chat_ctx_for_test:
#                 try:
#                     logger.info(f"[AgentEntry] LLM Direct Test: Attempting llm_plugin.chat(...)")
#                     # LLM chat returns an async iterator for streaming response
#                     # Correct way to get the async iterator - NO await here
#                     llm_test_stream = llm_plugin.chat(chat_ctx=chat_ctx_for_test)

#                     # Iterate through the stream to collect the response
#                     async for chunk in llm_test_stream:
#                         # Groq/OpenAI like streams often have text in chunk.delta.content
#                         chunk_text = getattr(getattr(chunk, 'delta', None), 'content', None)
#                         if chunk_text:
#                             llm_test_response_collected += chunk_text
#                             # Check if the response indicates success - Simplified check
#                             # Just check if we've received *any* non-whitespace text
#                             if llm_test_response_collected.strip():
#                                 logger.debug(f"[AgentEntry] LLM Direct Test: Received first non-empty chunk. Test considered passed.")
#                                 llm_test_passed = True # Mark test as passed
#                                 # Optional: break here if just confirming connectivity is enough
#                                 # break # Uncomment this line if you want the test to pass faster


#                     final_collected_text = llm_test_response_collected.strip()
#                     logger.info(f"[AgentEntry] LLM Test] Final collected response: '{final_collected_text}'")

#                     # The check for llm_test_passed after the loop handles success/failure
#                     # No need for a warning here based on specific indicators

#                 except Exception as e_llm_call:
#                     logger.error(f"[AgentEntry] LLM plugin direct test FAILED during chat call or stream processing: {e_llm_call}", exc_info=True)
#                     # If LLM test fails due to exception, it's usually critical. Exit.
#                     logger.critical(f"!!! [AgentEntry] LLM direct test failed: {e_llm_call}. Cannot proceed. !!!")
#                     sys.exit(f"LLM plugin direct test failed: {e_llm_call}")


#             else:
#                 logger.warning("[AgentEntry] LLM Direct Test: chat_ctx_for_test is None, skipping llm.chat() call.")
#         else:
#             logger.warning("[AgentEntry] LLM Direct Test: chat_message_list is empty, cannot proceed with test.")

#     # Final check after the test try/except block
#     if not llm_test_passed:
#          # This means the loop completed (or didn't even run if chat_ctx was None/empty)
#          # but llm_test_passed was never set to True (i.e., no text was received).
#          logger.critical("[AgentEntry] !!! LLM direct test FAILED. Agent will not function correctly. No text was received from the LLM stream or setup failed. !!!")
#          # Show the collected text again (should be empty if test failed this way)
#          logger.critical(f"[AgentEntry] Final collected response was: '{llm_test_response_collected.strip()}'")
#          sys.exit("LLM plugin direct test failed: No text received from stream or test setup failed.")

#     else:
#         # If we reach here, llm_test_passed is True, meaning we received at least some text.
#         logger.info("[AgentEntry] --- Testing LLM plugin directly END (Successful) ---")
#     # --- End of LLM Plugin Direct Test ---


#     # Create the AgentSession, passing the initialized plugins
#     # The AgentSession orchestrates the flow between audio input/output and the plugins
#     session = AgentSession(stt=stt_plugin, llm=llm_plugin, tts=tts_plugin, vad=vad_plugin)
#     logger.info("[AgentEntry] AgentSession created with Deepgram STT, Groq LLM, ElevenLabs TTS, and Silero VAD.")

#     # Create an instance of your custom Agent class
#     assistant_agent_instance = GUIFocusedAssistant()
#     logger.info("[AgentEntry] GUIFocusedAssistant instance created.")

#     # Start the AgentSession. This is a blocking call that runs the agent's lifecycle
#     # (connecting participants, handling audio, processing speech, interacting with LLM, etc.)
#     logger.info("[AgentEntry] Starting AgentSession with GUIFocusedAssistant...")
#     try:
#         # The session.start method handles the main loop of the agent
#         # This will block until the agent session ends (e.g., room closes, agent disconnects)
#         await session.start(room=ctx.room, agent=assistant_agent_instance)
#         logger.info("[AgentEntry] AgentSession.start() completed normally.")
#     except Exception as e_session_start:
#         logger.error(f"[AgentEntry] Error in session.start(): {e_session_start}", exc_info=True)
#         # Propagate the exception up if session start fails fundamentally
#         raise
#     finally:
#         logger.info("[AgentEntry] AgentSession.start() block exited.")

#     logger.info(f"[AgentEntry] Agent logic finished/terminated for job {ctx.job.id if ctx.job else 'N/A'}.")


# # Helper function to check and attempt to free a port
# def check_and_free_port(port):
#     """Check if a port is in use and attempt to terminate the process using it."""
#     logger.debug(f"Checking if port {port} is in use...")
#     try:
#         for proc in psutil.process_iter(['pid', 'name']):
#             try:
#                 for conn in proc.net_connections(kind='inet'):
#                     if conn.laddr.port == port:
#                         logger.warning(f"Port {port} is in use by PID {proc.pid} ({proc.name()}). Attempting to terminate.")
#                         try:
#                             # Attempt graceful termination first
#                             proc.terminate()
#                             # Wait for a short period for the process to exit
#                             proc.wait(timeout=3)
#                             if proc.is_running():
#                                  # If it's still running, attempt forceful kill
#                                 logger.warning(f"Process {proc.pid} did not terminate gracefully. Attempting kill.")
#                                 proc.kill()
#                                 proc.wait(timeout=3) # Wait again after kill
#                             logger.info(f"Successfully handled potential process on port {port} (PID {proc.pid}).")
#                         except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as term_err:
#                              # Log errors during termination attempt but continue
#                              logger.error(f"Failed to terminate process {proc.pid} on port {port}: {term_err}")
#                         except Exception as e:
#                             logger.error(f"Unexpected error while handling process {proc.pid} on port {port}: {e}", exc_info=True)
#                         return # Assume we've addressed the issue for this port and exit
#             except (psutil.NoSuchProcess, psutil.AccessDenied):
#                 # Ignore processes that no longer exist or are inaccessible
#                 continue
#             except Exception as e:
#                 logger.error(f"Error iterating process connections for PID {proc.pid}: {e}", exc_info=True)
#     except Exception as e:
#         logger.error(f"Error iterating processes to check port {port}: {e}", exc_info=True)
#     logger.debug(f"Port {port} appears free.")


# if __name__ == "__main__":
#     # Enable faulthandler to dump tracebacks on crashes
#     import faulthandler
#     faulthandler.enable()
#     logger.info("--- __main__: main.py script execution started ---")

#     if len(sys.argv) > 1 and sys.argv[1] in ["start", "download-files", "package"]:
#         logger.info(f"--- __main__: Command line argument '{sys.argv[1]}' detected. Running cli.run_app directly (Agent Worker mode). ---")
#         try:
#             worker_options = WorkerOptions(
#                 entrypoint_fnc=entrypoint,
#             )
#             logger.info("--- __main__: Calling cli.run_app(worker_options)...")
#             cli.run_app(worker_options)
#             logger.info("--- __main__: cli.run_app finished.")

#         except SystemExit as se:
#              if se.code != 0:
#                 logger.error(f"--- __main__: cli.run_app exited with non-zero SystemExit (code: {se.code}).")
#              else:
#                 logger.info(f"--- __main__: cli.run_app exited gracefully with SystemExit (code: {se.code}).")
#         except Exception as e:
#             logger.critical(f"!!! __main__: CRITICAL ERROR during cli.run_app (Agent Worker): {e}", exc_info=True)
#             sys.exit(f"Agent Worker process failed: {e}")

#         logger.info("--- __main__: Script execution finished (Agent Worker mode). ---")

#     else:
#         logger.info("--- __main__: No specific command line argument. Starting FastAPI server (Token Server mode). ---")

#         check_and_free_port(8000)

#         logger.info("--- __main__: Starting Uvicorn server for FastAPI app ---")
#         try:
#             uvicorn.run(
#                 "groq:app", # Assuming this script is named main.py
#                 host="0.0.0.0",
#                 port=8000,
#                 reload=False,
#                 log_level="info",
#             )
#         except Exception as e_uvicorn:
#             logger.critical(f"!!! __main__: CRITICAL - Uvicorn server failed: {e_uvicorn}", exc_info=True)
#             sys.exit(f"FastAPI server failed: {e_uvicorn}")

#         logger.info("--- __main__: Uvicorn server has stopped. ---")
#         logger.info("--- __main__: Script execution finished (Token Server mode). ---")


















# # ///////////////////////////////////////////////////////////

# full working code both response get logegd in console





import asyncio
import os
import sys
import logging
import io # Import io for potential stdout reconfigure errors
from datetime import timedelta
import psutil
from typing import Any, AsyncIterator

# Attempt to reconfigure sys.stdout/stderr to use UTF-8
# This helps ensure console output can display Hindi characters correctly,
# although it depends on the terminal's own encoding settings.
try:
    if sys.stdout.encoding is None or 'utf-8' not in sys.stdout.encoding.lower():
         sys.stdout.reconfigure(encoding='utf-8', errors='replace') # Use 'replace' for errors
    if sys.stderr.encoding is None or 'utf-8' not in sys.stderr.encoding.lower():
         sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, io.UnsupportedOperation) as e:
    # This can happen in some environments (e.g., certain IDEs, background tasks)
    logging.warning(f"Failed to reconfigure stdout/stderr encoding to UTF-8: {e}. Console output might not display non-ASCII characters correctly.")
except Exception as e:
     logging.error(f"Unexpected error during stdout/stderr reconfigure: {e}", exc_info=True)


from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from livekit.api import AccessToken, VideoGrants
from livekit import agents, rtc # Ensure rtc is imported if used by other parts or future SDK changes
from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    Agent,
    AgentSession,
    AutoSubscribe,
    ConversationItemAddedEvent, # <<<<<<< ADDED IMPORT
)
from livekit.agents.llm import LLMStream
from livekit.agents import voice
from livekit.agents import llm as livekit_llm # livekit_llm alias is good
from livekit.agents.llm import ChatMessage, ChatContext
import pydantic_core # Keep if needed by plugins/SDK versions, but might be removable
# Import the specific plugins needed: deepgram (for STT), groq (for LLM), elevenlabs (for TTS), silero (for VAD)
from livekit.plugins import deepgram, groq, elevenlabs, silero

# --- Load environment variables and set up module-level variables immediately ---
load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# --- Configure logging with console and file output ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_logs.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
         handler.setStream(sys.stdout)
    if hasattr(handler, 'encoding'):
        handler.encoding = 'utf-8'
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

agent_logger = logging.getLogger("livekit.agents")
agent_logger.setLevel(logging.DEBUG)
for handler in agent_logger.handlers:
     if isinstance(handler, logging.StreamHandler):
         handler.setStream(sys.stdout)
     if hasattr(handler, 'encoding'):
        handler.encoding = 'utf-8'
     handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

logger.info(f"MODULE LEVEL: LIVEKIT_URL: {LIVEKIT_URL}")
logger.info(f"MODULE LEVEL: LIVEKIT_API_KEY IS SET: {bool(LIVEKIT_API_KEY)}")
logger.info(f"MODULE LEVEL: DEEPGRAM_API_KEY IS SET: {bool(DEEPGRAM_API_KEY)}")
logger.info(f"MODULE LEVEL: GROQ_API_KEY IS SET: {bool(GROQ_API_KEY)}")
logger.info(f"MODULE LEVEL: ELEVENLABS_API_KEY IS SET: {bool(ELEVENLABS_API_KEY)}")

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
        raise HTTPException(
            status_code=500,
            detail="LiveKit credentials (URL, API Key, Secret) not configured"
        )

    room_name = "voice-assistant-room"
    participant_identity = "frontend_user"
    participant_name = "Frontend User"

    video_grant_obj = VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )

    token_builder = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token_builder.identity = participant_identity
    token_builder.name = participant_name
    token_builder.ttl = timedelta(hours=1)
    token_builder.with_grants(video_grant_obj)
    token_jwt = token_builder.to_jwt()

    logger.info(f"[TokenSvc] Generated token for identity '{participant_identity}' to join room '{room_name}'")
    return {"token": token_jwt, "url": LIVEKIT_URL, "roomName": room_name}


class GUIFocusedAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are khushi, a friendly and helpful voice assistant. "
                "Respond concisely and directly answer the user's query. "
                "You understand languages including English and Hindi. which is your primary language "
                "If the user speaks primarily in Hindi or mixes Hindi, respond concisely in Hindi using the Devanagari script. Do not use Urdu script or characters."
                "If the user speaks primarily in English, respond concisely in English."
                "Maintain context from the conversation history."
            )
        )
        self.last_user_query_for_log = None
        self.chat_history: list[ChatMessage] = [
             ChatMessage(role="system", content=[self.instructions])
        ]
        logger.info("--- [GUIFocusedAssistant] __init__ CALLED --- Agent initialized.")

    async def on_transcript(self, transcript: str, is_final: bool) -> None:
        log_level = logging.INFO if is_final else logging.DEBUG
        logger.log(log_level, f"--- [GUIFocusedAssistant] on_transcript --- Final: {is_final}, Text: '{transcript}'")
        print(f"--- DEBUG RAW STT OUTPUT --- Final: {is_final}, Text: '{transcript}'")

        if is_final and transcript and transcript.strip():
            self.last_user_query_for_log = transcript.strip()
            logger.info(f"--- [GUIFocusedAssistant] on_transcript: Final user query received: '{self.last_user_query_for_log}'. Attempting to generate LLM response.")

            if not hasattr(self, 'llm') or not self.llm:
                logger.error("--- [GUIFocusedAssistant] on_transcript: self.llm is not available. Cannot generate LLM response.")
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
                logger.debug(f"--- [GUIFocusedAssistant] on_transcript: Calling self.llm.chat(...) with history (count: {len(current_chat_turn_history)}).")

                llm_stream = self.llm.chat(chat_ctx=chat_ctx_for_llm)

                if not isinstance(llm_stream, AsyncIterator):
                    logger.error(f"--- [GUIFocusedAssistant] on_transcript: llm.chat did NOT return an async iterator. Type: {type(llm_stream)}. Cannot process response.")
                    await self._send_error_response("Sorry, I received an invalid response from the LLM.")
                    return

                logger.info(f"--- [GUIFocusedAssistant] on_transcript: self.llm.chat() returned stream. Now calling self.handle_llm_response.")
                self.chat_history = current_chat_turn_history
                await self.handle_llm_response(llm_stream)

            except Exception as e_llm_interaction:
                logger.error(f"--- [GUIFocusedAssistant] on_transcript: Uncaught error in LLM interaction sequence: {e_llm_interaction}", exc_info=True)
                await self._send_error_response("Sorry, I encountered an error trying to generate a response.")

        if self.room and self.room.local_participant:
            try:
                payload_str = f"transcription_update:{transcript}" if not is_final else f"transcription:{transcript}"
                await self.room.local_participant.publish_data(payload=payload_str.encode('utf-8'), topic="transcription")
            except Exception as e_pub_tx:
                logger.error(f"--- [GUIFocusedAssistant] on_transcript: Error publishing user transcript chunk: {e_pub_tx}", exc_info=True)

    async def _send_error_response(self, error_msg: str):
        logger.warning(f"--- [GUIFocusedAssistant] Sending error response: {error_msg}")
        if self.room and self.room.local_participant:
             try:
                 await self.room.local_participant.publish_data(payload=f"response:{error_msg}".encode('utf-8'), topic="response")
                 if hasattr(self, 'tts') and self.tts:
                     async def error_gen(): yield error_msg
                     await self.tts.synthesize(error_gen())
             except Exception as e_pub_err:
                 logger.error(f"--- [GUIFocusedAssistant] Failed to publish error message: {e_pub_err}", exc_info=True)

    async def handle_llm_response(self, llm_stream: LLMStream) -> None:
        logger.info(f"--- [GUIFocusedAssistant] handle_llm_response CALLED --- Starting stream processing.")
        collected_text_for_frontend = ""
        temp_sentences_for_tts = []
        llm_stream_finished_successfully = False

        try:
            logger.debug("--- [GUIFocusedAssistant] handle_llm_response: Starting LLM stream iteration via llm_stream_sentences ---")
            sentence_count = 0
            async for sentence_obj in self.llm_stream_sentences(llm_stream):
                logger.info(f"[PARTIAL] LLM sentence: '{sentence_obj.text}'")
                sentence_count += 1
                if sentence_obj.text:
                    collected_text_for_frontend += sentence_obj.text + " "
                    temp_sentences_for_tts.append(sentence_obj.text)

                if self.room and self.room.local_participant and sentence_obj.text and sentence_obj.text.strip():
                     try:
                         partial_payload = f"response_partial:{sentence_obj.text.strip()}"
                         await self.room.local_participant.publish_data(payload=partial_payload.encode('utf-8'), topic="response_partial")
                         logger.debug(f"--- [GUIFocusedAssistant] handle_llm_response: Published partial response: '{sentence_obj.text.strip()}'")
                     except Exception as e_pub_partial:
                         logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error publishing partial response: {e_pub_partial}", exc_info=True)

            llm_stream_finished_successfully = True
            logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: LLM stream iteration completed. Collected {sentence_count} sentences. ---")

        except Exception as e_llm_stream_processing:
            logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error processing LLM stream: {e_llm_stream_processing}", exc_info=True)
            collected_text_for_frontend = "Error generating response." if not collected_text_for_frontend.strip() else collected_text_for_frontend + " (Stream processing error)"
        finally:
            final_collected_text = collected_text_for_frontend.strip()
            log_context_info = f"for user query '{self.last_user_query_for_log}'" if self.last_user_query_for_log else "(initial/agent-initiated)"

            if final_collected_text:
                # THIS IS YOUR EXISTING LOGGING FOR THE AGENT'S RESPONSE
                logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: FINAL MODEL_RESPONSE {log_context_info}: '{final_collected_text}'")
                print(f"[DEBUG CONSOLE PRINT] MODEL FINAL RESPONSE (from handle_llm_response): {final_collected_text}")
                sys.stdout.flush()

                if self.chat_history and self.chat_history[-1].role == "assistant":
                     last_msg = self.chat_history[-1]
                     if not isinstance(last_msg.content, list): last_msg.content = [str(last_msg.content)] if last_msg.content is not None else [""]
                     last_msg.content[0] = (last_msg.content[0].strip() + " " + final_collected_text).strip()
                else:
                    self.chat_history.append(ChatMessage(role="assistant", content=[final_collected_text]))

                max_history_length = 21
                if len(self.chat_history) > max_history_length:
                     self.chat_history = [self.chat_history[0]] + self.chat_history[-(max_history_length - 1):]

                if self.room and self.room.local_participant:
                    logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Publishing FINAL agent response to GUI: '{final_collected_text}'")
                    try:
                        await self.room.local_participant.publish_data(payload=f"response:{final_collected_text}".encode('utf-8'), topic="response")
                    except Exception as e_pub_resp:
                        logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error publishing final agent response: {e_pub_resp}", exc_info=True)
            elif llm_stream_finished_successfully:
                logger.warning(f"--- [GUIFocusedAssistant] handle_llm_response: LLM stream finished, but no text was collected {log_context_info}.")
                await self._send_error_response("Sorry, I couldn't generate a response.")
            else:
                logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: LLM stream failed to produce text {log_context_info}.")

        if temp_sentences_for_tts and any(s.strip() for s in temp_sentences_for_tts) and hasattr(self, 'tts') and self.tts:
            logger.info(f"--- [GUIFocusedAssistant] handle_llm_response: Attempting TTS with {len(temp_sentences_for_tts)} text chunks.")
            async def gen_tts_stream():
                for s in temp_sentences_for_tts:
                    if s.strip(): yield s.strip()
            try:
                 logger.debug("--- [GUIFocusedAssistant] Calling self.tts.synthesize(...)")
                 await self.tts.synthesize(gen_tts_stream())
                 logger.info("--- [GUIFocusedAssistant] TTS synthesis call completed.")
            except Exception as e_tts:
                logger.error(f"--- [GUIFocusedAssistant] handle_llm_response: Error during TTS synthesis: {e_tts}", exc_info=True)

        self.last_user_query_for_log = None
        logger.info(f"--- [GUIFocusedAssistant] handle_llm_response FINISHED ---")


async def entrypoint(ctx: JobContext):
    logger.info(f"--- [AgentEntry] AGENT LOGIC ENTRYPOINT CALLED --- Room: {ctx.room.name}, Job: {ctx.job.id}")

    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        logger.info(f"[AgentEntry] Successfully connected to room '{ctx.room.name}'. Local Participant SID: {ctx.room.local_participant.sid}")
    except Exception as e_connect:
        logger.critical(f"!!! [AgentEntry] CRITICAL - Agent failed to connect to room '{ctx.room.name}': {e_connect}", exc_info=True)
        sys.exit(f"Fatal error during ctx.connect(): {e_connect}")

    logger.info("[AgentEntry] Initializing plugins...")
    stt_plugin, llm_plugin, tts_plugin, vad_plugin = None, None, None, None

    missing_keys = []
    if not DEEPGRAM_API_KEY: missing_keys.append("DEEPGRAM_API_KEY")
    if not GROQ_API_KEY: missing_keys.append("GROQ_API_KEY")
    if not ELEVENLABS_API_KEY: missing_keys.append("ELEVENLABS_API_KEY")
    if not LIVEKIT_URL or not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET: missing_keys.append("LIVEKIT_URL/API_KEY/API_SECRET")

    if missing_keys:
        logger.critical(f"!!! [AgentEntry] CRITICAL - Missing required environment variables: {', '.join(missing_keys)}. Cannot initialize plugins or connect to LiveKit. !!!")
        sys.exit(f"Missing required environment variables: {', '.join(missing_keys)}. Please check your .env file.")

    try:
        logger.info("[AgentEntry] Initializing Deepgram STT plugin...")
        stt_plugin = deepgram.STT(api_key=DEEPGRAM_API_KEY, model="nova-2-general")
        logger.info("[AgentEntry] Deepgram STT plugin initialized.")

        logger.info("[AgentEntry] Initializing Groq LLM plugin (model='llama3-70b-8192')...")
        llm_plugin = groq.LLM(model="llama3-70b-8192", api_key=GROQ_API_KEY)
        logger.info("[AgentEntry] Groq LLM plugin initialized.")

        logger.info("[AgentEntry] Initializing ElevenLabs TTS plugin...")
        tts_plugin = elevenlabs.TTS(api_key=ELEVENLABS_API_KEY)
        logger.info("[AgentEntry] ElevenLabs TTS plugin initialized.")

        logger.info("[AgentEntry] Initializing Silero VAD plugin...")
        vad_plugin = silero.VAD.load()
        logger.info("[AgentEntry] Silero VAD plugin initialized.")
        logger.info("[AgentEntry] All required plugins initialized successfully.")
    except Exception as e_plugins:
        logger.critical(f"!!! [AgentEntry] CRITICAL - Error initializing one or more plugins: {e_plugins}", exc_info=True)
        sys.exit(f"Error initializing plugins: {e_plugins}")

    logger.info("[AgentEntry] --- Testing LLM plugin directly START ---")
    llm_test_passed = False
    llm_test_response_collected = ""
    if llm_plugin:
        try:
            chat_message_list = [
                ChatMessage(role="system", content=["You are a helpful AI assistant. Respond concisely."]),
                ChatMessage(role="user", content=["Test 1 (EN): Hello, LLM. Confirm you are working by saying OK."]),
                ChatMessage(role="user", content=["Test 2 (HI): नमस्ते। क्या आप हिंदी में बात कर सकते हैं? कृपया हिंदी में जवाब दीजिए।"])
            ]
            chat_ctx_for_test = ChatContext(chat_message_list)
            llm_test_stream = llm_plugin.chat(chat_ctx=chat_ctx_for_test)
            async for chunk in llm_test_stream:
                chunk_text = getattr(getattr(chunk, 'delta', None), 'content', None)
                if chunk_text:
                    llm_test_response_collected += chunk_text
                    if llm_test_response_collected.strip():
                        llm_test_passed = True
            final_collected_text_test = llm_test_response_collected.strip()
            logger.info(f"[AgentEntry] LLM Test Final collected response: '{final_collected_text_test}'")
        except Exception as e_llm_call:
            logger.error(f"[AgentEntry] LLM plugin direct test FAILED: {e_llm_call}", exc_info=True)
            sys.exit(f"LLM plugin direct test failed: {e_llm_call}")
    else:
        logger.error("[AgentEntry] LLM Plugin is None, skipping direct test.")

    if not llm_test_passed:
         logger.critical("[AgentEntry] !!! LLM direct test FAILED. Agent will not function correctly. !!!")
         sys.exit("LLM plugin direct test failed: No text received or setup failed.")
    else:
        logger.info("[AgentEntry] --- Testing LLM plugin directly END (Successful) ---")

    session = AgentSession(stt=stt_plugin, llm=llm_plugin, tts=tts_plugin, vad=vad_plugin)
    logger.info("[AgentEntry] AgentSession created with Deepgram STT, Groq LLM, ElevenLabs TTS, and Silero VAD.")

    # -------- START: NEW EVENT HANDLER FOR LOGGING AGENT RESPONSE --------
    @session.on("conversation_item_added")
    def _on_conversation_item_added(event: ConversationItemAddedEvent):
        item_role = "UNKNOWN_ROLE"
        item_text_content = "NO_TEXT_CONTENT"
        item_interrupted = "N/A"

        if event.item:
            if hasattr(event.item, 'role'):
                item_role = event.item.role # This should be "user", "assistant", "system", etc.
            
            # event.item.text_content is a convenience property that gets text from event.item.content
            if hasattr(event.item, 'text_content') and event.item.text_content:
                item_text_content = event.item.text_content
            elif hasattr(event.item, 'content'): # Fallback if text_content is None or empty
                if isinstance(event.item.content, list):
                    str_parts = [part for part in event.item.content if isinstance(part, str)]
                    if str_parts:
                        item_text_content = " ".join(str_parts).strip()
                elif isinstance(event.item.content, str): # If content itself is a string
                    item_text_content = event.item.content
            
            if hasattr(event.item, 'interrupted') and event.item.interrupted is not None:
                item_interrupted = str(event.item.interrupted)

        # General log for any conversation item added to the session's history
        logger.info(f"[AgentSession Event] Conversation Item: Role='{item_role}', Interrupted='{item_interrupted}', Text='{item_text_content}'")

        # Specifically log the agent's (assistant's) response as requested
        if item_role == "assistant" and item_text_content != "NO_TEXT_CONTENT":
            logger.info(f"--- AGENT'S RESPONSE (logged via conversation_item_added event) --- '{item_text_content}'")
            # Optional: A direct print for easier spotting in console during development
            print(f"[CONSOLE PRINT - AGENT RESPONSE via event]: {item_text_content}")
            sys.stdout.flush() # Ensure it's flushed if printing
    # -------- END: NEW EVENT HANDLER FOR LOGGING AGENT RESPONSE --------

    assistant_agent_instance = GUIFocusedAssistant()
    logger.info("[AgentEntry] GUIFocusedAssistant instance created.")

    logger.info("[AgentEntry] Starting AgentSession with GUIFocusedAssistant...")
    try:
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
                            logger.info(f"Successfully handled potential process on port {port} (PID {proc.pid}).")
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
        logger.error(f"Error iterating processes to check port {port}: {e}", exc_info=True)
    logger.debug(f"Port {port} appears free.")


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    logger.info("--- __main__: main.py script execution started ---")

    # Corrected uvicorn run argument for the script name
    # Assuming your script is named 'main.py' and FastAPI app instance is 'app'
    # If your script is named 'groq.py' (as in uvicorn.run("groq:app", ...)), then it's correct
    # For clarity, I'll use a placeholder that you should adjust if your script name is different
    # e.g., if your script is main.py, it should be "main:app"
    SCRIPT_NAME_FOR_UVICORN = "groq" # Change this to "main" if your file is main.py

    if len(sys.argv) > 1 and sys.argv[1] in ["start", "download-files", "package"]:
        logger.info(f"--- __main__: Command line argument '{sys.argv[1]}' detected. Running cli.run_app directly (Agent Worker mode). ---")
        try:
            worker_options = WorkerOptions(entrypoint_fnc=entrypoint)
            logger.info("--- __main__: Calling cli.run_app(worker_options)...")
            cli.run_app(worker_options) # This is blocking
            logger.info("--- __main__: cli.run_app finished.")
        except SystemExit as se:
             if se.code != 0: logger.error(f"--- __main__: cli.run_app exited with non-zero SystemExit (code: {se.code}).")
             else: logger.info(f"--- __main__: cli.run_app exited gracefully with SystemExit (code: {se.code}).")
        except Exception as e:
            logger.critical(f"!!! __main__: CRITICAL ERROR during cli.run_app (Agent Worker): {e}", exc_info=True)
            sys.exit(f"Agent Worker process failed: {e}")
        logger.info("--- __main__: Script execution finished (Agent Worker mode). ---")
    else:
        logger.info("--- __main__: No specific command line argument. Starting FastAPI server (Token Server mode). ---")
        check_and_free_port(8000)
        logger.info("--- __main__: Starting Uvicorn server for FastAPI app ---")
        try:
            # Ensure the app string is correct. If your file is main.py, it should be "main:app"
            # If your file is indeed named "groq.py" and the FastAPI instance is 'app', then "groq:app" is correct.
            uvicorn.run(
                f"{SCRIPT_NAME_FOR_UVICORN}:app",
                host="0.0.0.0",
                port=8000,
                reload=False, # reload=True can be problematic with agent workers
                log_level="info",
            )
        except Exception as e_uvicorn:
            logger.critical(f"!!! __main__: CRITICAL - Uvicorn server failed: {e_uvicorn}", exc_info=True)
            sys.exit(f"FastAPI server failed: {e_uvicorn}")
        logger.info("--- __main__: Uvicorn server has stopped. ---")
        logger.info("--- __main__: Script execution finished (Token Server mode). ---")
