

# complete working code


# import asyncio
# import os
# import sys
# import logging
# import io
# from datetime import timedelta
# import psutil
# from typing import Any, AsyncIterator

# # Standard UTF-8 reconfiguration for console (good practice, though less critical if only English)
# try:
#     if sys.stdout.encoding is None or 'utf-8' not in sys.stdout.encoding.lower():
#          sys.stdout.reconfigure(encoding='utf-8', errors='replace')
#     if sys.stderr.encoding is None or 'utf-8' not in sys.stderr.encoding.lower():
#          sys.stderr.reconfigure(encoding='utf-8', errors='replace')
# except (AttributeError, io.UnsupportedOperation) as e:
#     logging.warning(f"Failed to reconfigure stdout/stderr encoding to UTF-8: {e}.")
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
#     ConversationItemAddedEvent,
# )
# from livekit.agents.llm import LLMStream
# from livekit.agents import voice
# from livekit.agents import llm as livekit_llm
# from livekit.agents.llm import ChatMessage, ChatContext
# import pydantic_core
# from livekit.plugins import deepgram, groq, elevenlabs, silero

# load_dotenv()

# LIVEKIT_URL = os.getenv("LIVEKIT_URL")
# LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
# LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
# DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler('agent_logs_english_only.log', encoding='utf-8') # Changed log file name
#     ]
# )

# logger = logging.getLogger(__name__)
# for handler in logger.handlers:
#     if isinstance(handler, logging.StreamHandler):
#          handler.setStream(sys.stdout)
#     if hasattr(handler, 'encoding'): # Should be utf-8 from basicConfig
#         handler.encoding = 'utf-8'
#     handler.setFormatter(logging.Formatter(
#         '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     ))

# agent_logger = logging.getLogger("livekit.agents")
# agent_logger.setLevel(logging.DEBUG)
# for handler in agent_logger.handlers: # Inherits handlers from root, re-apply formatting if needed
#      if isinstance(handler, logging.StreamHandler):
#          handler.setStream(sys.stdout)
#      if hasattr(handler, 'encoding'):
#         handler.encoding = 'utf-8'
#      handler.setFormatter(logging.Formatter(
#         '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     ))

# logger.info(f"MODULE LEVEL: LIVEKIT_URL: {LIVEKIT_URL}")
# logger.info(f"MODULE LEVEL: LIVEKIT_API_KEY IS SET: {bool(LIVEKIT_API_KEY)}")
# logger.info(f"MODULE LEVEL: DEEPGRAM_API_KEY IS SET: {bool(DEEPGRAM_API_KEY)}")
# logger.info(f"MODULE LEVEL: GROQ_API_KEY IS SET: {bool(GROQ_API_KEY)}")
# logger.info(f"MODULE LEVEL: ELEVENLABS_API_KEY IS SET: {bool(ELEVENLABS_API_KEY)}")

# app = FastAPI()

# origins = ["http://localhost:3000"]
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

#     room_name = "voice-assistant-room-english" # Slightly different room name
#     participant_identity = "frontend_user"
#     participant_name = "Frontend User"
#     video_grant_obj = VideoGrants(
#         room_join=True, room=room_name, can_publish=True, can_subscribe=True, can_publish_data=True,
#     )
#     token_builder = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
#     token_builder.identity = participant_identity
#     token_builder.name = participant_name
#     token_builder.ttl = timedelta(hours=1)
#     token_builder.with_grants(video_grant_obj)
#     token_jwt = token_builder.to_jwt()
#     logger.info(f"[TokenSvc] Generated token for identity '{participant_identity}' for room '{room_name}'")
#     return {"token": token_jwt, "url": LIVEKIT_URL, "roomName": room_name}

# class EnglishAssistant(Agent): # Renamed for clarity
#     def __init__(self) -> None:
#         super().__init__(
#             instructions=( # <<<<< MODIFIED INSTRUCTIONS
#                 "You are a friendly and helpful voice assistant named Khushi. "
#                 "You speak and understand only English. "
#                 "Respond concisely and directly answer the user's query in English. "
#                 "Maintain context from the conversation history."
#             )
#         )
#         self.last_user_query_for_log = None
#         self.chat_history: list[ChatMessage] = [
#              ChatMessage(role="system", content=[self.instructions])
#         ]
#         logger.info("--- [EnglishAssistant] __init__ CALLED --- Agent initialized for English only.")

#     async def on_transcript(self, transcript: str, is_final: bool) -> None:
#         log_level = logging.INFO if is_final else logging.DEBUG
#         logger.log(log_level, f"--- [EnglishAssistant] on_transcript --- Final: {is_final}, Text: '{transcript}'")
#         # print(f"--- DEBUG RAW STT OUTPUT --- Final: {is_final}, Text: '{transcript}'") # Kept for debugging

#         if is_final and transcript and transcript.strip():
#             self.last_user_query_for_log = transcript.strip()
#             logger.info(f"--- [EnglishAssistant] on_transcript: Final user query (English): '{self.last_user_query_for_log}'. Generating LLM response.")

#             if not hasattr(self, 'llm') or not self.llm:
#                 logger.error("--- [EnglishAssistant] on_transcript: self.llm is not available.")
#                 await self._send_error_response("Sorry, my brain is not available right now.")
#                 return

#             try:
#                 user_chat_message = ChatMessage(role="user", content=[self.last_user_query_for_log])
#                 current_chat_turn_history = list(self.chat_history)
#                 current_chat_turn_history.append(user_chat_message)

#                 max_history_length = 21
#                 if len(current_chat_turn_history) > max_history_length:
#                      current_chat_turn_history = [current_chat_turn_history[0]] + current_chat_turn_history[-(max_history_length - 1):]

#                 chat_ctx_for_llm = ChatContext(current_chat_turn_history)
#                 logger.debug(f"--- [EnglishAssistant] on_transcript: Calling self.llm.chat(...) with history (count: {len(current_chat_turn_history)}).")

#                 llm_stream = self.llm.chat(chat_ctx=chat_ctx_for_llm)

#                 if not isinstance(llm_stream, AsyncIterator):
#                     logger.error(f"--- [EnglishAssistant] on_transcript: llm.chat did NOT return an async iterator. Type: {type(llm_stream)}.")
#                     await self._send_error_response("Sorry, I received an invalid response from the LLM.")
#                     return

#                 logger.info(f"--- [EnglishAssistant] on_transcript: self.llm.chat() returned stream. Calling self.handle_llm_response.")
#                 self.chat_history = current_chat_turn_history
#                 await self.handle_llm_response(llm_stream)

#             except Exception as e_llm_interaction:
#                 logger.error(f"--- [EnglishAssistant] on_transcript: Uncaught error in LLM interaction: {e_llm_interaction}", exc_info=True)
#                 await self._send_error_response("Sorry, I encountered an error generating a response.")

#         if self.room and self.room.local_participant:
#             try:
#                 payload_str = f"transcription_update:{transcript}" if not is_final else f"transcription:{transcript}"
#                 await self.room.local_participant.publish_data(payload=payload_str.encode('utf-8'), topic="transcription")
#             except Exception as e_pub_tx:
#                 logger.error(f"--- [EnglishAssistant] on_transcript: Error publishing user transcript: {e_pub_tx}", exc_info=True)

#     async def _send_error_response(self, error_msg: str):
#         logger.warning(f"--- [EnglishAssistant] Sending error response: {error_msg}")
#         if self.room and self.room.local_participant:
#              try:
#                  await self.room.local_participant.publish_data(payload=f"response:{error_msg}".encode('utf-8'), topic="response")
#                  if hasattr(self, 'tts') and self.tts:
#                      async def error_gen(): yield error_msg
#                      await self.tts.synthesize(error_gen())
#              except Exception as e_pub_err:
#                  logger.error(f"--- [EnglishAssistant] Failed to publish error message: {e_pub_err}", exc_info=True)

#     async def handle_llm_response(self, llm_stream: LLMStream) -> None:
#         logger.info(f"--- [EnglishAssistant] handle_llm_response CALLED --- Processing English stream.")
#         collected_text_for_frontend = ""
#         temp_sentences_for_tts = []
#         llm_stream_finished_successfully = False

#         try:
#             sentence_count = 0
#             async for sentence_obj in self.llm_stream_sentences(llm_stream):
#                 # logger.info(f"[PARTIAL] LLM sentence (English): '{sentence_obj.text}'") # Can be verbose
#                 sentence_count += 1
#                 if sentence_obj.text:
#                     collected_text_for_frontend += sentence_obj.text + " "
#                     temp_sentences_for_tts.append(sentence_obj.text)

#                 if self.room and self.room.local_participant and sentence_obj.text and sentence_obj.text.strip():
#                      try:
#                          partial_payload = f"response_partial:{sentence_obj.text.strip()}"
#                          await self.room.local_participant.publish_data(payload=partial_payload.encode('utf-8'), topic="response_partial")
#                      except Exception as e_pub_partial:
#                          logger.error(f"--- [EnglishAssistant] handle_llm_response: Error publishing partial English response: {e_pub_partial}", exc_info=True)

#             llm_stream_finished_successfully = True
#             logger.info(f"--- [EnglishAssistant] handle_llm_response: English LLM stream processed. Sentences: {sentence_count}.")

#         except Exception as e_llm_stream_processing:
#             logger.error(f"--- [EnglishAssistant] handle_llm_response: Error processing English LLM stream: {e_llm_stream_processing}", exc_info=True)
#             collected_text_for_frontend = "Error generating English response." if not collected_text_for_frontend.strip() else collected_text_for_frontend + " (Stream error)"
#         finally:
#             final_collected_text = collected_text_for_frontend.strip()
#             log_context_info = f"for user query '{self.last_user_query_for_log}'" if self.last_user_query_for_log else "(agent-initiated)"

#             if final_collected_text:
#                 logger.info(f"--- [EnglishAssistant] handle_llm_response: FINAL ENGLISH MODEL_RESPONSE {log_context_info}: '{final_collected_text}'")
#                 # print(f"[CONSOLE DEBUG] MODEL FINAL ENGLISH RESPONSE: {final_collected_text}") # Kept for debugging
#                 # sys.stdout.flush()

#                 if self.chat_history and self.chat_history[-1].role == "assistant":
#                      last_msg = self.chat_history[-1]
#                      if not isinstance(last_msg.content, list): last_msg.content = [str(last_msg.content)] if last_msg.content is not None else [""]
#                      last_msg.content[0] = (last_msg.content[0].strip() + " " + final_collected_text).strip()
#                 else:
#                     self.chat_history.append(ChatMessage(role="assistant", content=[final_collected_text]))

#                 max_history_length = 21
#                 if len(self.chat_history) > max_history_length:
#                      self.chat_history = [self.chat_history[0]] + self.chat_history[-(max_history_length - 1):]

#                 if self.room and self.room.local_participant:
#                     try:
#                         await self.room.local_participant.publish_data(payload=f"response:{final_collected_text}".encode('utf-8'), topic="response")
#                     except Exception as e_pub_resp:
#                         logger.error(f"--- [EnglishAssistant] handle_llm_response: Error publishing final English response: {e_pub_resp}", exc_info=True)
#             elif llm_stream_finished_successfully:
#                 logger.warning(f"--- [EnglishAssistant] handle_llm_response: LLM stream finished, no English text collected {log_context_info}.")
#                 await self._send_error_response("Sorry, I couldn't generate an English response.")
#             else:
#                 logger.error(f"--- [EnglishAssistant] handle_llm_response: LLM stream failed for English text {log_context_info}.")

#         if temp_sentences_for_tts and any(s.strip() for s in temp_sentences_for_tts) and hasattr(self, 'tts') and self.tts:
#             logger.info(f"--- [EnglishAssistant] handle_llm_response: Attempting English TTS with {len(temp_sentences_for_tts)} chunks.")
#             async def gen_tts_stream():
#                 for s in temp_sentences_for_tts:
#                     if s.strip(): yield s.strip()
#             try:
#                  await self.tts.synthesize(gen_tts_stream())
#                  logger.info("--- [EnglishAssistant] English TTS synthesis call completed.")
#             except Exception as e_tts:
#                 logger.error(f"--- [EnglishAssistant] handle_llm_response: Error during English TTS synthesis: {e_tts}", exc_info=True)

#         self.last_user_query_for_log = None
#         logger.info(f"--- [EnglishAssistant] handle_llm_response FINISHED (English) ---")

# async def entrypoint(ctx: JobContext):
#     logger.info(f"--- [AgentEntry] AGENT ENTRYPOINT (English Only) --- Room: {ctx.room.name}")

#     try:
#         await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
#         logger.info(f"[AgentEntry] Connected to room '{ctx.room.name}'.")
#     except Exception as e_connect:
#         logger.critical(f"!!! [AgentEntry] CRITICAL - Agent failed to connect: {e_connect}", exc_info=True)
#         sys.exit(f"Fatal error during ctx.connect(): {e_connect}")

#     stt_plugin, llm_plugin, tts_plugin, vad_plugin = None, None, None, None
#     missing_keys = []
#     if not DEEPGRAM_API_KEY: missing_keys.append("DEEPGRAM_API_KEY")
#     if not GROQ_API_KEY: missing_keys.append("GROQ_API_KEY")
#     if not ELEVENLABS_API_KEY: missing_keys.append("ELEVENLABS_API_KEY")
#     if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]): missing_keys.append("LIVEKIT_URL/KEY/SECRET")

#     if missing_keys:
#         logger.critical(f"!!! [AgentEntry] Missing env vars: {', '.join(missing_keys)}. Exiting.")
#         sys.exit(f"Missing env vars: {', '.join(missing_keys)}.")

#     try:
#         logger.info("[AgentEntry] Initializing Deepgram STT plugin (language='en')...") # <<<<< MODIFIED STT
#         stt_plugin = deepgram.STT(
#             api_key=DEEPGRAM_API_KEY,
#             language="en", # Explicitly set to English
#             model="nova-2-general" # Good general model for English
#         )
#         logger.info("[AgentEntry] Deepgram STT plugin initialized for English.")

#         logger.info("[AgentEntry] Initializing Groq LLM plugin (model='llama3-70b-8192')...")
#         llm_plugin = groq.LLM(model="llama3-70b-8192", api_key=GROQ_API_KEY)
#         logger.info("[AgentEntry] Groq LLM plugin initialized.")

#         logger.info("[AgentEntry] Initializing ElevenLabs TTS plugin...")
#         tts_plugin = elevenlabs.TTS(api_key=ELEVENLABS_API_KEY) # Default voice is usually English
#         logger.info("[AgentEntry] ElevenLabs TTS plugin initialized.")

#         logger.info("[AgentEntry] Initializing Silero VAD plugin...")
#         vad_plugin = silero.VAD.load()
#         logger.info("[AgentEntry] Silero VAD plugin initialized.")
#         logger.info("[AgentEntry] All plugins initialized successfully for English operation.")
#     except Exception as e_plugins:
#         logger.critical(f"!!! [AgentEntry] CRITICAL - Error initializing plugins: {e_plugins}", exc_info=True)
#         sys.exit(f"Error initializing plugins: {e_plugins}")

#     logger.info("[AgentEntry] --- Testing LLM plugin (English) START ---")
#     llm_test_passed = False
#     llm_test_response_collected = ""
#     if llm_plugin:
#         try:
#             chat_message_list = [
#                 ChatMessage(role="system", content=["You are a helpful AI assistant. Respond concisely in English."]),
#                 ChatMessage(role="user", content=["Hello, LLM. Confirm you are working by responding 'OK, I am ready.' in English."]) # <<<<< SIMPLIFIED LLM TEST
#             ]
#             chat_ctx_for_test = ChatContext(chat_message_list)
#             llm_test_stream = llm_plugin.chat(chat_ctx=chat_ctx_for_test)
#             async for chunk in llm_test_stream:
#                 chunk_text = getattr(getattr(chunk, 'delta', None), 'content', None)
#                 if chunk_text:
#                     llm_test_response_collected += chunk_text
#                     if llm_test_response_collected.strip():
#                         llm_test_passed = True # Pass if any text received
#             final_collected_text_test = llm_test_response_collected.strip()
#             logger.info(f"[AgentEntry] LLM Test (English) - Final collected response: '{final_collected_text_test}'")
#             if not "ok" in final_collected_text_test.lower() and not "ready" in final_collected_text_test.lower(): # Basic check
#                  logger.warning(f"[AgentEntry] LLM Test (English) - Response '{final_collected_text_test}' did not contain expected 'OK' or 'ready'. Check LLM if issues arise.")

#         except Exception as e_llm_call:
#             logger.error(f"[AgentEntry] LLM plugin direct test FAILED: {e_llm_call}", exc_info=True)
#             sys.exit(f"LLM plugin direct test failed: {e_llm_call}")
#     else:
#         logger.error("[AgentEntry] LLM Plugin is None, skipping direct test.")

#     if not llm_test_passed:
#          logger.critical("[AgentEntry] !!! LLM direct test FAILED. Agent will not function correctly. No text received or setup failed. !!!")
#          sys.exit("LLM plugin direct test failed: No text received or test setup failed.")
#     else:
#         logger.info("[AgentEntry] --- Testing LLM plugin (English) END (Successful) ---")

#     session = AgentSession(stt=stt_plugin, llm=llm_plugin, tts=tts_plugin, vad=vad_plugin)
#     logger.info("[AgentEntry] AgentSession created for English operation.")

#     @session.on("conversation_item_added")
#     def _on_conversation_item_added(event: ConversationItemAddedEvent):
#         item_role = "UNKNOWN_ROLE"
#         item_text_content = "NO_TEXT_CONTENT"
#         item_interrupted = "N/A"

#         if event.item:
#             if hasattr(event.item, 'role'): item_role = event.item.role
#             if hasattr(event.item, 'text_content') and event.item.text_content:
#                 item_text_content = event.item.text_content
#             elif hasattr(event.item, 'content'):
#                 if isinstance(event.item.content, list):
#                     str_parts = [part for part in event.item.content if isinstance(part, str)]
#                     if str_parts: item_text_content = " ".join(str_parts).strip()
#                 elif isinstance(event.item.content, str): item_text_content = event.item.content
#             if hasattr(event.item, 'interrupted') and event.item.interrupted is not None:
#                 item_interrupted = str(event.item.interrupted)

#         logger.info(f"[AgentSession Event] Item: Role='{item_role}', Interrupt='{item_interrupted}', Text='{item_text_content}'")
#         if item_role == "assistant" and item_text_content != "NO_TEXT_CONTENT":
#             logger.info(f"--- AGENT'S ENGLISH RESPONSE (via event) --- '{item_text_content}'")
#             # print(f"[CONSOLE DEBUG - AGENT RESPONSE via event]: {item_text_content}") # Kept for debugging
#             # sys.stdout.flush()

#     assistant_agent_instance = EnglishAssistant() # Use the English-specific assistant
#     logger.info("[AgentEntry] EnglishAssistant instance created.")

#     logger.info("[AgentEntry] Starting AgentSession with EnglishAssistant...")
#     try:
#         await session.start(room=ctx.room, agent=assistant_agent_instance)
#         logger.info("[AgentEntry] AgentSession.start() completed.")
#     except Exception as e_session_start:
#         logger.error(f"[AgentEntry] Error in session.start(): {e_session_start}", exc_info=True)
#         raise
#     finally:
#         logger.info("[AgentEntry] AgentSession.start() block exited.")

#     logger.info(f"[AgentEntry] Agent logic finished for job {ctx.job.id if ctx.job else 'N/A'}.")

# def check_and_free_port(port):
#     # (This function remains the same, useful for Uvicorn)
#     logger.debug(f"Checking if port {port} is in use...")
#     try:
#         for proc in psutil.process_iter(['pid', 'name']):
#             try:
#                 for conn in proc.net_connections(kind='inet'):
#                     if conn.laddr.port == port:
#                         logger.warning(f"Port {port} in use by PID {proc.pid} ({proc.name()}). Terminating.")
#                         try:
#                             proc.terminate()
#                             proc.wait(timeout=3)
#                             if proc.is_running():
#                                 logger.warning(f"Process {proc.pid} did not terminate gracefully. Killing.")
#                                 proc.kill()
#                                 proc.wait(timeout=3)
#                             logger.info(f"Handled process on port {port} (PID {proc.pid}).")
#                         except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as term_err:
#                              logger.error(f"Failed to terminate PID {proc.pid} on port {port}: {term_err}")
#                         except Exception as e:
#                             logger.error(f"Unexpected error handling PID {proc.pid} on port {port}: {e}", exc_info=True)
#                         return
#             except (psutil.NoSuchProcess, psutil.AccessDenied): continue
#             except Exception as e: logger.error(f"Error iterating conns for PID {proc.pid}: {e}", exc_info=True)
#     except Exception as e: logger.error(f"Error iterating procs for port {port}: {e}", exc_info=True)
#     logger.debug(f"Port {port} appears free.")

# if __name__ == "__main__":
#     import faulthandler
#     faulthandler.enable()
#     logger.info("--- __main__: Script execution started (English Only Mode) ---")

#     # Ensure SCRIPT_NAME_FOR_UVICORN is correctly set if your file is not named 'main.py' or 'groq.py'
#     # For this example, I'll assume it's 'main.py'. If your file is `english_assistant.py`, change it.
#     SCRIPT_NAME_FOR_UVICORN = "main" # Or the actual name of this script file without .py

#     if len(sys.argv) > 1 and sys.argv[1] in ["start", "download-files", "package"]:
#         logger.info(f"--- __main__: Running Agent Worker mode (English Only). ---")
#         try:
#             worker_options = WorkerOptions(entrypoint_fnc=entrypoint)
#             cli.run_app(worker_options)
#         except SystemExit as se:
#              logger.info(f"--- __main__: cli.run_app exited with SystemExit (code: {se.code}).")
#         except Exception as e:
#             logger.critical(f"!!! __main__: CRITICAL ERROR in Agent Worker: {e}", exc_info=True)
#             sys.exit(f"Agent Worker failed: {e}")
#         logger.info("--- __main__: Script finished (Agent Worker mode). ---")
#     else:
#         logger.info("--- __main__: Starting FastAPI server (Token Server mode). ---")
#         check_and_free_port(8000) # Port for FastAPI
#         try:
#             uvicorn.run(
#                 f"{SCRIPT_NAME_FOR_UVICORN}:app",
#                 host="0.0.0.0", port=8000, reload=False, log_level="info",
#             )
#         except Exception as e_uvicorn:
#             logger.critical(f"!!! __main__: CRITICAL - Uvicorn server failed: {e_uvicorn}", exc_info=True)
#             sys.exit(f"FastAPI server failed: {e_uvicorn}")
#         logger.info("--- __main__: Uvicorn server stopped. Script finished (Token Server mode). ---")






















# /////////////////////////////////////////////////////////////










# trial code for frontend logging - it is working perfectly






# import asyncio
# import os
# import sys
# import logging
# import io
# from datetime import timedelta
# import psutil
# from typing import Any, AsyncIterator

# # Standard UTF-8 reconfiguration for console
# try:
#     if sys.stdout.encoding is None or 'utf-8' not in sys.stdout.encoding.lower():
#          sys.stdout.reconfigure(encoding='utf-8', errors='replace')
#     if sys.stderr.encoding is None or 'utf-8' not in sys.stderr.encoding.lower():
#          sys.stderr.reconfigure(encoding='utf-8', errors='replace')
# except (AttributeError, io.UnsupportedOperation) as e:
#     logging.warning(f"Failed to reconfigure stdout/stderr encoding to UTF-8: {e}.")
# except Exception as e:
#      logging.error(f"Unexpected error during stdout/stderr reconfigure: {e}", exc_info=True)

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
#     ConversationItemAddedEvent, # Ensure this is imported
# )
# from livekit.agents.llm import LLMStream
# from livekit.agents import voice
# from livekit.agents import llm as livekit_llm
# from livekit.agents.llm import ChatMessage, ChatContext
# import pydantic_core
# from livekit.plugins import deepgram, groq, elevenlabs, silero

# load_dotenv()

# LIVEKIT_URL = os.getenv("LIVEKIT_URL")
# LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
# LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
# DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler('agent_logs_english_only.log', encoding='utf-8')
#     ]
# )
# logger = logging.getLogger(__name__)
# # (Formatter setup for logger and agent_logger as before) ...
# for handler in logger.handlers:
#     if isinstance(handler, logging.StreamHandler):
#          handler.setStream(sys.stdout)
#     if hasattr(handler, 'encoding'): # Should be utf-8 from basicConfig
#         handler.encoding = 'utf-8'
#     handler.setFormatter(logging.Formatter(
#         '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     ))

# agent_logger = logging.getLogger("livekit.agents")
# agent_logger.setLevel(logging.DEBUG) # Or INFO if DEBUG is too verbose
# for handler_ref in logging.getLogger().handlers: # Get root handlers configured by basicConfig
#     # Create a new handler for agent_logger if you want separate filtering or formatting
#     # For simplicity, let agent_logger use the same handlers and formatters
#     agent_logger.addHandler(handler_ref)
#     # If you specifically want agent_logger to have its own formatting on existing handlers:
#     # (though basicConfig applies to root, and child loggers usually propagate to root handlers)
#     # if isinstance(handler_ref, logging.StreamHandler):
#     #      handler_ref.setStream(sys.stdout) # Already done
#     # if hasattr(handler_ref, 'encoding'):
#     #     handler_ref.encoding = 'utf-8'   # Already done
#     # handler_ref.setFormatter(logging.Formatter(
#     #    '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
#     #    datefmt='%Y-%m-%d %H:%M:%S'
#     # ))
# agent_logger.propagate = False # Prevent double logging if handlers are added directly and also propagate


# logger.info(f"MODULE LEVEL: LIVEKIT_URL: {LIVEKIT_URL}")
# # ... (other env var logs)

# app = FastAPI()
# origins = ["http://localhost:3000"]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/token")
# async def get_token():
#     # ... (get_token implementation as before, ensure room_name is consistent or managed)
#     logger.info("[TokenSvc] /token endpoint hit.")
#     if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
#         logger.error("[TokenSvc] LiveKit credentials not configured properly.")
#         raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

#     room_name = "voice-assistant-room-english"
#     participant_identity = "frontend_user" # This is the identity the frontend user will use
#     # The agent will have its own identity, typically auto-generated or set in WorkerOptions
#     participant_name = "Frontend User"
#     video_grant_obj = VideoGrants(
#         room_join=True, room=room_name, can_publish=True, can_subscribe=True, can_publish_data=True,
#     )
#     token_builder = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
#     token_builder.identity = participant_identity
#     token_builder.name = participant_name
#     token_builder.ttl = timedelta(hours=1)
#     token_builder.with_grants(video_grant_obj)
#     token_jwt = token_builder.to_jwt()
#     logger.info(f"[TokenSvc] Generated token for identity '{participant_identity}' for room '{room_name}'")
#     return {"token": token_jwt, "url": LIVEKIT_URL, "roomName": room_name}


# class EnglishAssistant(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             instructions=(
#                 "You are a friendly and helpful voice assistant named Khushi. "
#                 "You speak and understand only English. "
#                 "Respond concisely and directly answer the user's query in English. "
#                 "Maintain context from the conversation history."
#             )
#         )
#         self.last_user_query_for_log = None
#         self.chat_history: list[ChatMessage] = [
#              ChatMessage(role="system", content=[self.instructions])
#         ]
#         logger.info("--- [EnglishAssistant] __init__ CALLED --- Agent initialized for English only.")

#     async def on_transcript(self, transcript: str, is_final: bool) -> None:
#         log_level = logging.INFO if is_final else logging.DEBUG
#         # logger.log(log_level, f"--- [EnglishAssistant] on_transcript --- Final: {is_final}, Text: '{transcript}'")

#         # Publishing partial transcriptions for live feedback (optional)
#         if self.room and self.room.local_participant and not is_final and transcript:
#             try:
#                 payload_str_partial = f"transcription_update:{transcript}"
#                 # await self.room.local_participant.publish_data(payload=payload_str_partial.encode('utf-8'), topic="transcription_partial")
#             except Exception as e_pub_tx:
#                 logger.error(f"--- [EnglishAssistant] on_transcript: Error publishing partial user transcript: {e_pub_tx}", exc_info=True)


#         if is_final and transcript and transcript.strip():
#             self.last_user_query_for_log = transcript.strip()
#             # logger.info(f"--- [EnglishAssistant] on_transcript: Final user query (English): '{self.last_user_query_for_log}'. Generating LLM response.")

#             # NOTE: We are moving the primary publishing of the *final* user query
#             # to the `_on_conversation_item_added` event handler to ensure it aligns with session history.
#             # If you still want to publish it from here for some reason, you can, but be mindful of duplicates.
#             # Example of publishing the *final* user transcript from here (potentially redundant now):
#             # if self.room and self.room.local_participant:
#             #     try:
#             #         payload_str_final = f"transcription:{self.last_user_query_for_log}"
#             #         await self.room.local_participant.publish_data(payload=payload_str_final.encode('utf-8'), topic="transcription")
#             #     except Exception as e_pub_tx:
#             #         logger.error(f"--- [EnglishAssistant] on_transcript: Error publishing final user transcript: {e_pub_tx}", exc_info=True)


#             if not hasattr(self, 'llm') or not self.llm:
#                 logger.error("--- [EnglishAssistant] on_transcript: self.llm is not available.")
#                 await self._send_error_response("Sorry, my brain is not available right now.")
#                 return

#             try:
#                 user_chat_message = ChatMessage(role="user", content=[self.last_user_query_for_log])
#                 current_chat_turn_history = list(self.chat_history)
#                 current_chat_turn_history.append(user_chat_message)

#                 max_history_length = 21
#                 if len(current_chat_turn_history) > max_history_length:
#                      current_chat_turn_history = [current_chat_turn_history[0]] + current_chat_turn_history[-(max_history_length - 1):]

#                 chat_ctx_for_llm = ChatContext(current_chat_turn_history)
#                 # logger.debug(f"--- [EnglishAssistant] on_transcript: Calling self.llm.chat(...) with history (count: {len(current_chat_turn_history)}).")
#                 llm_stream = self.llm.chat(chat_ctx=chat_ctx_for_llm)

#                 if not isinstance(llm_stream, AsyncIterator):
#                     logger.error(f"--- [EnglishAssistant] on_transcript: llm.chat did NOT return an async iterator. Type: {type(llm_stream)}.")
#                     await self._send_error_response("Sorry, I received an invalid response from the LLM.")
#                     return

#                 # logger.info(f"--- [EnglishAssistant] on_transcript: self.llm.chat() returned stream. Calling self.handle_llm_response.")
#                 self.chat_history = current_chat_turn_history # Update history *before* processing response
#                 await self.handle_llm_response(llm_stream)

#             except Exception as e_llm_interaction:
#                 logger.error(f"--- [EnglishAssistant] on_transcript: Uncaught error in LLM interaction: {e_llm_interaction}", exc_info=True)
#                 await self._send_error_response("Sorry, I encountered an error generating a response.")

#     async def _send_error_response(self, error_msg: str):
#         # (This function remains the same)
#         logger.warning(f"--- [EnglishAssistant] Sending error response: {error_msg}")
#         if self.room and self.room.local_participant:
#              try:
#                  # Publishing error messages to the frontend using the "response" topic
#                  await self.room.local_participant.publish_data(payload=f"response:Error: {error_msg}".encode('utf-8'), topic="response")
#                  if hasattr(self, 'tts') and self.tts:
#                      async def error_gen(): yield error_msg
#                      await self.tts.synthesize(error_gen())
#              except Exception as e_pub_err:
#                  logger.error(f"--- [EnglishAssistant] Failed to publish error message: {e_pub_err}", exc_info=True)

#     async def handle_llm_response(self, llm_stream: LLMStream) -> None:
#         # logger.info(f"--- [EnglishAssistant] handle_llm_response CALLED --- Processing English stream.")
#         collected_text_for_frontend = ""
#         temp_sentences_for_tts = []
#         llm_stream_finished_successfully = False

#         try:
#             sentence_count = 0
#             async for sentence_obj in self.llm_stream_sentences(llm_stream):
#                 sentence_count += 1
#                 if sentence_obj.text:
#                     collected_text_for_frontend += sentence_obj.text + " "
#                     temp_sentences_for_tts.append(sentence_obj.text)

#                     # Publishing partial agent responses (optional)
#                     if self.room and self.room.local_participant and sentence_obj.text.strip():
#                          try:
#                              partial_payload = f"response_partial:{sentence_obj.text.strip()}"
#                              # await self.room.local_participant.publish_data(payload=partial_payload.encode('utf-8'), topic="response_partial")
#                          except Exception as e_pub_partial:
#                              logger.error(f"--- [EnglishAssistant] handle_llm_response: Error publishing partial English response: {e_pub_partial}", exc_info=True)

#             llm_stream_finished_successfully = True
#             # logger.info(f"--- [EnglishAssistant] handle_llm_response: English LLM stream processed. Sentences: {sentence_count}.")

#         except Exception as e_llm_stream_processing:
#             logger.error(f"--- [EnglishAssistant] handle_llm_response: Error processing English LLM stream: {e_llm_stream_processing}", exc_info=True)
#             collected_text_for_frontend = "Error generating English response." if not collected_text_for_frontend.strip() else collected_text_for_frontend + " (Stream error)"
#         finally:
#             final_collected_text = collected_text_for_frontend.strip()
#             log_context_info = f"for user query '{self.last_user_query_for_log}'" if self.last_user_query_for_log else "(agent-initiated)"

#             if final_collected_text:
#                 # logger.info(f"--- [EnglishAssistant] handle_llm_response: FINAL ENGLISH MODEL_RESPONSE {log_context_info}: '{final_collected_text}'")

#                 # NOTE: We are moving the primary publishing of the *final* agent response
#                 # to the `_on_conversation_item_added` event handler.
#                 # If you still want to publish it from here:
#                 # if self.room and self.room.local_participant:
#                 #     try:
#                 #         await self.room.local_participant.publish_data(payload=f"response:{final_collected_text}".encode('utf-8'), topic="response")
#                 #     except Exception as e_pub_resp:
#                 #         logger.error(f"--- [EnglishAssistant] handle_llm_response: Error publishing final English response: {e_pub_resp}", exc_info=True)

#                 # Add to internal agent history (this agent instance's history)
#                 # This will eventually trigger conversation_item_added if the session's agent is this instance
#                 if self.chat_history and self.chat_history[-1].role == "assistant":
#                      last_msg = self.chat_history[-1]
#                      if not isinstance(last_msg.content, list): last_msg.content = [str(last_msg.content)] if last_msg.content is not None else [""]
#                      last_msg.content[0] = (last_msg.content[0].strip() + " " + final_collected_text).strip()
#                 else: # This is usually how it works if session.say or generate_reply is used implicitly
#                     self.chat_history.append(ChatMessage(role="assistant", content=[final_collected_text]))

#                 max_history_length = 21
#                 if len(self.chat_history) > max_history_length:
#                      self.chat_history = [self.chat_history[0]] + self.chat_history[-(max_history_length - 1):]

#             elif llm_stream_finished_successfully:
#                 logger.warning(f"--- [EnglishAssistant] handle_llm_response: LLM stream finished, no English text collected {log_context_info}.")
#                 await self._send_error_response("Sorry, I couldn't generate an English response.")
#             else: # Stream processing error and no text collected
#                 logger.error(f"--- [EnglishAssistant] handle_llm_response: LLM stream failed for English text {log_context_info}.")
#                 # Error already sent if collected_text_for_frontend was empty from the except block

#         if temp_sentences_for_tts and any(s.strip() for s in temp_sentences_for_tts) and hasattr(self, 'tts') and self.tts:
#             # logger.info(f"--- [EnglishAssistant] handle_llm_response: Attempting English TTS with {len(temp_sentences_for_tts)} chunks.")
#             async def gen_tts_stream():
#                 for s in temp_sentences_for_tts:
#                     if s.strip(): yield s.strip()
#             try:
#                  await self.tts.synthesize(gen_tts_stream())
#                  # logger.info("--- [EnglishAssistant] English TTS synthesis call completed.")
#             except Exception as e_tts:
#                 logger.error(f"--- [EnglishAssistant] handle_llm_response: Error during English TTS synthesis: {e_tts}", exc_info=True)

#         self.last_user_query_for_log = None # Reset after processing
#         # logger.info(f"--- [EnglishAssistant] handle_llm_response FINISHED (English) ---")


# async def entrypoint(ctx: JobContext):
#     logger.info(f"--- [AgentEntry] AGENT ENTRYPOINT (English Only) --- Room: {ctx.room.name}")

#     try:
#         await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
#         logger.info(f"[AgentEntry] Connected to room '{ctx.room.name}'. Local SID: {ctx.room.local_participant.sid}")
#     except Exception as e_connect:
#         logger.critical(f"!!! [AgentEntry] CRITICAL - Agent failed to connect: {e_connect}", exc_info=True)
#         sys.exit(f"Fatal error during ctx.connect(): {e_connect}")

#     # ... (Plugin initialization and LLM test as before - ENSURE KEYS ARE SET) ...
#     stt_plugin, llm_plugin, tts_plugin, vad_plugin = None, None, None, None
#     # ... (rest of plugin init) ...
#     try:
#         stt_plugin = deepgram.STT(api_key=DEEPGRAM_API_KEY, language="en", model="nova-2-general")
#         llm_plugin = groq.LLM(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
#         tts_plugin = elevenlabs.TTS(api_key=ELEVENLABS_API_KEY)
#         vad_plugin = silero.VAD.load()
#         logger.info("[AgentEntry] All plugins initialized successfully for English operation.")
#     except Exception as e_plugins:
#         logger.critical(f"!!! [AgentEntry] CRITICAL - Error initializing plugins: {e_plugins}", exc_info=True)
#         sys.exit(f"Error initializing plugins: {e_plugins}")


#     session = AgentSession(stt=stt_plugin, llm=llm_plugin, tts=tts_plugin, vad=vad_plugin)
#     logger.info("[AgentEntry] AgentSession created for English operation.")

#     # -------- CORRECTED EVENT HANDLER REGISTRATION --------
#     async def _publish_conversation_item_async(event_item: ChatMessage):
#         """Helper async function to do the actual publishing"""
#         item_role_str = "unknown"
#         item_text_content = ""

#         if event_item: # Check if event_item (which is event.item) is not None
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
        
#         # Publish to frontend using a new topic
#         if ctx.room and ctx.room.local_participant and item_text_content:
#             payload_to_send = ""
#             topic_to_send = "lk_chat_history"

#             if item_role_str.lower() == "user":
#                 payload_to_send = f"user_msg:{item_text_content}"
#             elif item_role_str.lower() == "assistant":
#                 payload_to_send = f"agent_msg:{item_text_content}"
#                 logger.info(f"--- AGENT'S RESPONSE (via event, to be published) --- '{item_text_content}'")

#             if payload_to_send:
#                 logger.info(f"[AgentEntry] Publishing to frontend via '{topic_to_send}': '{payload_to_send}'")
#                 try:
#                     await ctx.room.local_participant.publish_data(
#                         payload=payload_to_send.encode('utf-8'),
#                         topic=topic_to_send
#                     )
#                 except Exception as e_pub:
#                     logger.error(f"[AgentEntry] Error publishing from _publish_conversation_item_async: {e_pub}", exc_info=True)

#     # This is the synchronous callback registered with .on()
#     def _on_conversation_item_added_sync(event: ConversationItemAddedEvent):
#         item_role_str = "unknown"
#         item_text_content = "" # Default to empty string

#         if event.item: # Ensure event.item exists
#             if event.item.role:
#                 # Handle if role is an enum (like ChatMessageRole.USER) or just a string
#                 item_role_str = str(event.item.role.value) if hasattr(event.item.role, 'value') else str(event.item.role)
            
#             # Get text content robustly
#             if hasattr(event.item, 'text_content') and event.item.text_content:
#                 item_text_content = event.item.text_content
#             elif hasattr(event.item, 'content'): # Fallback for content list
#                 if isinstance(event.item.content, list):
#                     str_parts = [str(part) for part in event.item.content if isinstance(part, str)] # Ensure parts are strings
#                     item_text_content = " ".join(str_parts).strip()
#                 elif isinstance(event.item.content, str): # If content itself is a string
#                     item_text_content = event.item.content
            
#         # Original backend logging (this part is synchronous)
#         logger.info(f"[AgentSession Event] Item Added (Sync Handler): Role='{item_role_str}', Text='{item_text_content}'")

#         # Create a task to run the async publishing logic
#         if event.item: # Only try to publish if there's an item
#             asyncio.create_task(_publish_conversation_item_async(event.item))

#     session.on("conversation_item_added", _on_conversation_item_added_sync)
#     # -------- END CORRECTED EVENT HANDLER REGISTRATION --------

#     assistant_agent_instance = EnglishAssistant() # Or your relevant assistant class
#     logger.info("[AgentEntry] EnglishAssistant instance created.")

#     logger.info("[AgentEntry] Starting AgentSession with EnglishAssistant...")
#     try:
#         await session.start(room=ctx.room, agent=assistant_agent_instance)
#         logger.info("[AgentEntry] AgentSession.start() completed.")
#     except Exception as e_session_start:
#         logger.error(f"[AgentEntry] Error in session.start(): {e_session_start}", exc_info=True)
#         raise
#     finally:
#         logger.info("[AgentEntry] AgentSession.start() block exited.")
#     logger.info(f"[AgentEntry] Agent logic finished for job {ctx.job.id if ctx.job else 'N/A'}.")


# # (check_and_free_port function as before)
# def check_and_free_port(port):
#     logger.debug(f"Checking if port {port} is in use...")
#     try:
#         for proc in psutil.process_iter(['pid', 'name']):
#             try:
#                 for conn in proc.net_connections(kind='inet'):
#                     if conn.laddr.port == port:
#                         logger.warning(f"Port {port} in use by PID {proc.pid} ({proc.name()}). Terminating.")
#                         try:
#                             proc.terminate()
#                             proc.wait(timeout=3)
#                             if proc.is_running():
#                                 logger.warning(f"Process {proc.pid} did not terminate gracefully. Killing.")
#                                 proc.kill()
#                                 proc.wait(timeout=3)
#                             logger.info(f"Handled process on port {port} (PID {proc.pid}).")
#                         except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as term_err:
#                              logger.error(f"Failed to terminate PID {proc.pid} on port {port}: {term_err}")
#                         except Exception as e:
#                             logger.error(f"Unexpected error handling PID {proc.pid} on port {port}: {e}", exc_info=True)
#                         return # Addressed the port, exit check
#             except (psutil.NoSuchProcess, psutil.AccessDenied): continue
#             except Exception as e: logger.error(f"Error iterating conns for PID {proc.pid}: {e}", exc_info=True)
#     except Exception as e: logger.error(f"Error iterating procs for port {port}: {e}", exc_info=True)
#     logger.debug(f"Port {port} appears free.")


# if __name__ == "__main__":
#     import faulthandler
#     faulthandler.enable()
#     logger.info("--- __main__: Script execution started (English Only Mode) ---")
#     SCRIPT_NAME_FOR_UVICORN = "main" # CHANGE IF YOUR FILENAME IS DIFFERENT (e.g., "english_assistant")

#     if len(sys.argv) > 1 and sys.argv[1] in ["start", "download-files", "package"]:
#         logger.info(f"--- __main__: Running Agent Worker mode (English Only). ---")
#         try:
#             worker_options = WorkerOptions(entrypoint_fnc=entrypoint) # Ensure this points to your entrypoint
#             cli.run_app(worker_options)
#         except SystemExit as se:
#              logger.info(f"--- __main__: cli.run_app exited with SystemExit (code: {se.code}).")
#         except Exception as e:
#             logger.critical(f"!!! __main__: CRITICAL ERROR in Agent Worker: {e}", exc_info=True)
#             sys.exit(f"Agent Worker failed: {e}")
#         logger.info("--- __main__: Script finished (Agent Worker mode). ---")
#     else:
#         logger.info("--- __main__: Starting FastAPI server (Token Server mode). ---")
#         check_and_free_port(8000)
#         try:
#             uvicorn.run(
#                 f"{SCRIPT_NAME_FOR_UVICORN}:app",
#                 host="0.0.0.0", port=8000, reload=False, log_level="info",
#             )
#         except Exception as e_uvicorn:
#             logger.critical(f"!!! __main__: CRITICAL - Uvicorn server failed: {e_uvicorn}", exc_info=True)
#             sys.exit(f"FastAPI server failed: {e_uvicorn}")
#         logger.info("--- __main__: Uvicorn server stopped. Script finished (Token Server mode). ---")
























# ////////////darshini voice assistanace



import asyncio
import os
import sys
import logging
import io
from datetime import timedelta
import psutil
from typing import AsyncIterator, Optional
import json
import psycopg2
from urllib.parse import quote_plus
import pydantic
from livekit.agents import function_tool, Agent, RunContext
from livekit.agents.llm import ChatContext, ChatMessage

# Standard UTF-8 reconfiguration (No changes)
# ...

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
    AgentSession,
    AutoSubscribe,
    ConversationItemAddedEvent,
)
from livekit.plugins import deepgram, groq, elevenlabs, silero

load_dotenv()

# --- Load Environment Variables (No changes) ---
# ...
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
try:
    db_user = os.getenv('DB_USER')
    db_password_raw = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')
    if not all([db_user, db_password_raw, db_host, db_port, db_name]):
        raise ValueError("One or more database environment variables are missing.")
    db_password_encoded = quote_plus(db_password_raw)
    DB_CONNECTION_STRING = (
        f"postgresql://{db_user}:{db_password_encoded}"
        f"@{db_host}:{db_port}/{db_name}"
    )
except (TypeError, AttributeError, ValueError) as e:
    print(f"FATAL ERROR: Database environment variables are not set correctly. Details: {e}")
    sys.exit(1)


# --- Logging Setup (No changes) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent_logs_detailed.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# --- FastAPI App for Token Generation (No changes) ---
app = FastAPI()
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/token")
async def get_token():
    # Using the corrected builder pattern
    logger.info("[TokenSvc] /token endpoint hit.")
    room_name = "voice-assistant-room-db"
    participant_identity = "frontend_user"
    participant_name = "Frontend User"
    video_grant_obj = VideoGrants(room_join=True, room=room_name, can_publish=True, can_subscribe=True, can_publish_data=True)
    token_builder = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token_builder.identity = participant_identity
    token_builder.name = participant_name
    token_builder.ttl = timedelta(hours=1)
    token_builder.with_grants(video_grant_obj)
    token_jwt = token_builder.to_jwt()
    return {"token": token_jwt, "url": LIVEKIT_URL, "roomName": room_name}


class EnglishAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "you always have to do greeting first with hello my self Darshini"
                "You are a friendly and helpful voice assistant named Darshini."
                "You have access to a tool to search a database for lost items."
                "ONLY use this tool if the user's query is clearly about finding a lost or found item."
                "To use the tool, you MUST have at least TWO of the following details: product_type, brand, or color."
                "If the user provides only one detail, you MUST ask for more information."
                # --- THIS IS THE CRITICAL NEW INSTRUCTION ---
                "When you receive results from the tool, your ONLY job is to present that summary to the user. Do not analyze it, do not try to search again. Simply state the summary provided by the tool."
                "For all other topics, respond conversationally."
            )
        )
        logger.info("--- [EnglishAssistant] __init__ CALLED --- Agent initialized.")

    async def generate_reply(self, chat_ctx: ChatContext) -> AsyncIterator[str]:
        # (No changes in this function)
        user_query = chat_ctx.messages[-1].text_content
        logger.info(f"--- [generate_reply] Processing user query: '{user_query}'")

        llm_stream = self.llm.chat(chat_ctx=chat_ctx)

        full_response = ""
        async for chunk in llm_stream:
            if chunk.choices[0].delta.content:
                text_chunk = chunk.choices[0].delta.content
                full_response += text_chunk
                yield text_chunk
        
        logger.info(f"--- [generate_reply] Final model response: '{full_response.strip()}'")


    @function_tool()
    async def find_lost_item(
        self,
        context: RunContext,
        product_type: Optional[str] = None,
        brand: Optional[str] = None,
        color: Optional[str] = None,
    ) -> str:
        # (No changes in this function)
        logger.info(f"--- [DB Tool Called] LLM requested search with args: type='{product_type}', brand='{brand}', color='{color}'")
        result = await asyncio.to_thread(self._find_matching_items_sync, product_type, brand, color)
        logger.info(f"--- [DB Tool Result] Result sent back to LLM: '{result}'")
        return result

    def _find_matching_items_sync(self, product_type: Optional[str], brand: Optional[str], color: Optional[str]) -> str:
        provided_params = sum(1 for p in [product_type, brand, color] if p is not None)
        if provided_params < 2:
            return json.dumps({"summary": "Tool execution failed: The user did not provide at least two search criteria. You must ask the user for more details."})

        try:
            conn = psycopg2.connect(DB_CONNECTION_STRING)
        except psycopg2.OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            return json.dumps({"summary": "I'm sorry, I couldn't connect to the item database right now."})

        try:
            with conn.cursor() as cur:
                # This part dynamically builds the scoring logic for the SQL query
                fields = {"product_type": product_type, "brand": brand, "color": color}
                score_parts, params = [], []
                for column, value in fields.items():
                    if value:
                        score_parts.append(f'(CASE WHEN "{column}" ILIKE %s THEN 1 ELSE 0 END)')
                        params.append(f"%{value}%")

                # --- THIS IS THE FIX ---
                # This line was missing before. It joins the parts together.
                score_calculation = " + ".join(score_parts)
                # --- END OF FIX ---

                final_query = f"""
                    SELECT "ID", brand, model, color, product_type, image_url
                    FROM scrapingdata."Darshini"
                    WHERE ({score_calculation}) >= 2
                    ORDER BY ({score_calculation}) DESC, "ID"
                    LIMIT 5;
                """
                
                logger.info(f"--- [DB Query] EXECUTING SQL: {final_query}")
                logger.info(f"--- [DB Query] WITH PARAMS: {tuple(params + params)}")
                
                # The parameters need to be duplicated because they are used twice in the query
                cur.execute(final_query, tuple(params + params))
                
                columns = [desc[0] for desc in cur.description]
                results = [dict(zip(columns, row)) for row in cur.fetchall()]
                logger.info("result666", results)
                if not results:
                    return json.dumps({"summary": "I searched the database, but no items were found matching your criteria."})

                summary_text = f"I found {len(results)} matching items for you. Here they are."
                
                payload = {
                    "summary": summary_text,
                    "items": results,
                    "images": [item["image_url"] for item in results if "image_url" in item]
                }
                return json.dumps(payload)

        except Exception as e:
            logger.error(f"An error occurred during database query: {e}", exc_info=True)
            return json.dumps({"summary": "I encountered an unexpected error while searching for the item."})
        finally:
            if conn:
                conn.close()

async def entrypoint(ctx: JobContext):
    logger.info(f"--- [AgentEntry] AGENT ENTRYPOINT --- Room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    stt_plugin = deepgram.STT(api_key=DEEPGRAM_API_KEY, language="en", model="nova-2-general")
    llm_plugin = groq.LLM(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
    tts_plugin = elevenlabs.TTS(api_key=ELEVENLABS_API_KEY)
    vad_plugin = silero.VAD.load()

    session = AgentSession(stt=stt_plugin, llm=llm_plugin, tts=tts_plugin, vad=vad_plugin)



    # def log_conversation_item(event: ConversationItemAddedEvent):
    #     item = event.item
    #     if not item:
    #         return

    #     log_prefix = f"[CONVO_HISTORY - {item.role}]"
        
    #     if item.text_content:
    #         logger.info(f"{log_prefix} Text: '{item.text_content}'")
        
    #     # --- THIS IS THE FIX ---
    #     # Safely check if 'tool_calls' attribute exists before trying to access it
    #     if hasattr(item, 'tool_calls') and item.tool_calls:
    #         for tc in item.tool_calls:
    #             logger.info(f"{log_prefix} Tool Call Request: Name='{tc.function.name}', Args='{tc.function.arguments_str}'")
        
    #     if item.role == 'tool' and hasattr(item, 'content'):
    #         logger.info(f"{log_prefix} Tool Execution Result: '{item.content}'")

    # session.on("conversation_item_added", log_conversation_item)
    
    # --- MODIFIED: The event listener function is now fixed ---

    async def publish_item_to_frontend(item: ChatMessage):
        topic_to_send = "lk_chat_history"
        
        if not (item and ctx.room and ctx.room.local_participant):
            return

        # --- NEW: Logic to handle both plain text and rich JSON messages ---
        payload_to_send = ""
        if item.role == 'user' and item.text_content:
            payload_to_send = f"user_msg:{item.text_content}"
        elif item.role == 'assistant' and item.text_content:
            try:
                # Try to find JSON inside the text (handles cases where LLM adds words)
                match = re.search(r'\{.*\}', item.text_content, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    if 'summary' in data and 'items' in data:
                        payload_to_send = f"agent_rich_msg:{json.dumps(data)}"
                    else:
                        payload_to_send = f"agent_msg:{item.text_content}"
                else:
                    payload_to_send = f"agent_msg:{item.text_content}"
            except Exception:
                payload_to_send = f"agent_msg:{item.text_content}"
        # --- END NEW LOGIC ---

        if payload_to_send:
            try:
                logger.info(f"--- [Data Publishing] Sending to frontend: '{payload_to_send}'")
                await ctx.room.local_participant.publish_data(
                    payload=payload_to_send.encode('utf-8'),
                    topic=topic_to_send
                )
            except Exception as e:
                logger.error(f"--- [Data Publishing] Failed to publish data: {e}", exc_info=True)

    # This is the synchronous event handler that gets called by the SDK
    def handle_conversation_item(event: ConversationItemAddedEvent):
        item = event.item
        if not item:
            return

        # 1. Log to backend console (your existing logic)
        log_prefix = f"[CONVO_HISTORY - {item.role}]"
        if item.text_content:
            logger.info(f"{log_prefix} Text: '{item.text_content}'")
        if hasattr(item, 'tool_calls') and item.tool_calls:
            for tc in item.tool_calls:
                logger.info(f"{log_prefix} Tool Call Request: Name='{tc.function.name}', Args='{tc.function.arguments_str}'")
        
        # 2. NEW: Publish the item to the frontend in the background
        asyncio.create_task(publish_item_to_frontend(item))

    session.on("conversation_item_added", handle_conversation_item)
    # --- END MODIFIED SECTION ---

    assistant_agent_instance = EnglishAssistant()
    await session.start(room=ctx.room, agent=assistant_agent_instance)
    logger.info("[AgentEntry] AgentSession.start() completed.")

def check_and_free_port(port):
    logger.debug(f"Checking if port {port} is in use...")
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.net_connections(kind='inet'):
                    if conn.laddr.port == port:
                        logger.warning(f"Port {port} in use by PID {proc.pid} ({proc.name()}). Terminating.")
                        p = psutil.Process(proc.pid)
                        p.terminate()
                        p.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception as e:
        logger.error(f"Error checking port {port}: {e}")

if __name__ == "__main__":
    SCRIPT_NAME_FOR_UVICORN = os.path.splitext(os.path.basename(__file__))[0]

    if len(sys.argv) > 1 and sys.argv[1] in ["start", "download-files", "package"]:
        logger.info("--- __main__: Running Agent Worker mode ---")
        try:
            worker_options = WorkerOptions(entrypoint_fnc=entrypoint)
            cli.run_app(worker_options)
        except Exception as e:
            logger.critical(f"!!! __main__: CRITICAL ERROR in Agent Worker: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.info("--- __main__: Starting FastAPI server (Token Server mode) ---")
        check_and_free_port(8000)
        try:
            uvicorn.run(
                f"{SCRIPT_NAME_FOR_UVICORN}:app",
                host="0.0.0.0", port=8000, reload=False, log_level="info",
            )
        except Exception as e_uvicorn:
            logger.critical(f"!!! __main__: CRITICAL - Uvicorn server failed: {e_uvicorn}", exc_info=True)
            sys.exit(1)