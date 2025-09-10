# minimal_agent_test.py
import asyncio # Still needed for async def
import logging
import os
import sys
from dotenv import load_dotenv
from datetime import datetime
import faulthandler

from livekit.agents import JobContext, WorkerOptions, cli

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(threadName)s: %(message)s')
logger = logging.getLogger("minimal_agent_test")
livekit_sdk_logger = logging.getLogger("livekit.agents")
livekit_sdk_logger.setLevel(logging.DEBUG)

async def minimal_entry(ctx: JobContext):
    entry_time = datetime.now().isoformat()
    room_name_from_ctx = ctx.room.name if ctx.room else "N/A_ROOM_IN_CTX"
    job_id_from_ctx = ctx.job.id if ctx.job else "N/A_JOB_IN_CTX"
    try:
        with open("minimal_agent_entrypoint_touched.txt", "a") as f:
            f.write(f"{entry_time} - MINIMAL_AGENT_ENTRYPOINT_TOUCHED! Room: {room_name_from_ctx}, Job: {job_id_from_ctx}\n")
    except Exception as e_file:
        print(f"DEBUG PRINT: Failed to write to minimal_agent_entrypoint_touched.txt: {e_file}")
    logger.info(f"--- MINIMAL_ENTRYPOINT_REACHED --- Room: {room_name_from_ctx}, Job ID: {job_id_from_ctx}, Entry Time: {entry_time}")
    try:
        logger.info(f"[MinimalEntry] Attempting ctx.connect() for job {job_id_from_ctx} in room {room_name_from_ctx}.")
        await ctx.connect()
        logger.info(f"[MinimalEntry] Successfully called ctx.connect(). Participant SID: {ctx.room.local_participant.sid if ctx.room and ctx.room.local_participant else 'N/A'}")
        logger.info("[MinimalEntry] Waiting indefinitely (simulating being busy).")
        await asyncio.Future()
    except Exception as e_connect:
        logger.error(f"[MinimalEntry] Error during ctx.connect() or asyncio.Future(): {e_connect}", exc_info=True)
        # ... (file logging for error) ...
        raise
    finally:
        logger.info(f"[MinimalEntry] Finished or terminated for job {job_id_from_ctx}.")
        # ... (file logging for finally) ...

# This function is now synchronous as cli.run_app will handle async
def main_agent_runner(): # Renamed to avoid confusion, and no longer async
    logger.info("Minimal agent main_agent_runner() started.")
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        logger.critical("LiveKit credentials missing in .env file. Exiting.")
        return

    opts = WorkerOptions(
        entrypoint_fnc=minimal_entry,
    )

    original_argv = list(sys.argv)
    sys.argv = ['minimal_agent_test.py', 'start']
    logger.info(f"Running cli.run_app with sys.argv: {sys.argv}")
    logger.info(f"Credentials for cli.run_app will be taken from environment variables: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET")

    try:
        # cli.run_app is a blocking call that manages its own asyncio loop
        cli.run_app(opts)
    except SystemExit as se:
        logger.warning(f"cli.run_app exited with SystemExit code: {se.code}")
    except Exception as e:
        logger.error(f"Exception from cli.run_app: {e}", exc_info=True)
    finally:
        sys.argv = original_argv
        logger.info(f"Restored original sys.argv: {sys.argv}")

    logger.info("Minimal agent main_agent_runner() finished.")


if __name__ == "__main__":
    faulthandler.enable()
    logger.info("--- faulthandler enabled for minimal_agent_test.py ---")
    
    # Directly call the function that contains cli.run_app
    # DO NOT use asyncio.run() here.
    main_agent_runner()