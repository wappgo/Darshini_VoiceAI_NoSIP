# api_server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants # From livekit server SDK
from datetime import timedelta
import logging

load_dotenv()
app = FastAPI()
# ... (all your FastAPI app, /token endpoint logic from main.py) ...
# Ensure LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET are accessible here

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s')
logger = logging.getLogger(__name__)

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")


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
    logger.info("[FastAPI] /token endpoint hit.")
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        logger.error("[FastAPI] /token: LiveKit credentials not configured.")
        raise HTTPException(status_code=500, detail="LiveKit credentials not configured")

    grants = VideoGrants(
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
    token_builder.with_grants(grants)
    token_jwt = token_builder.to_jwt()
    logger.info(f"[FastAPI] /token: Generated token for 'frontend_user'.")
    return {"token": token_jwt, "url": LIVEKIT_URL}


if __name__ == "__main__":
    logger.info("--- Starting FastAPI server (api_server.py) ---")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")