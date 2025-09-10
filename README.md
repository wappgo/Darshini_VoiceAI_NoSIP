# AI-Powered Darshini Voice AI Assistant

This project is a real-time, conversational voice assistant built with the LiveKit Agent Framework. It helps users find lost items by verbally querying a PostgreSQL database and displaying the results visually in a web interface.

The agent uses an advanced "Tool Use" pattern where the Large Language Model (LLM) intelligently decides when to search the database, gathers the necessary information from the user's natural language, and then formulates a response based on the search results.

## Key Features

-   **Real-time Voice Conversation**: Low-latency, bidirectional audio streaming.
-   **Natural Language Understanding (NLU)**: The agent understands user requests in natural language (e.g., "I lost my silver Apple watch").
-   **AI Tool Use**: The LLM is equipped with a `lost_and_found_search` tool, which it decides to use when a user reports a lost item.
-   **Dynamic & Secure SQL Generation**: The agent's tool uses the LLM to dynamically generate a secure, parameterized PostgreSQL query to search the database, preventing SQL injection.
-   **Real-time UI Updates**: The agent sends commands over a data channel to a web frontend to display and highlight search results.
-   **Powered by Leading AI Services**:
    -   **LLM**: Groq (Llama 3) for fast reasoning and tool use.
    -   **STT**: Deepgram for accurate real-time transcription.
    -   **TTS**: ElevenLabs for natural-sounding voice responses.

## Technology Stack

-   **Backend**: Python
-   **Agent Framework**: LiveKit Agent Framework
-   **Web Server**: FastAPI & Uvicorn (for the token server)
-   **Database**: PostgreSQL
-   **AI Services**:
    -   Groq (LLM)
    -   Deepgram (Speech-to-Text)
    -   ElevenLabs (Text-to-Speech)

## Setup and Installation

Follow these steps to get the backend server and agent worker running.

#### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-directory>
2. Create and Activate a Python Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.
Windows:
code
Bash
python -m venv venv
.\venv\Scripts\activate
macOS / Linux:
code
Bash
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Create a requirements.txt file with the following content:
requirements.txt
code
Code
livekit-agents
psycopg2-binary
python-dotenv
fastapi
uvicorn[standard]
groq
elevenlabs
deepgram-sdk[async]
Then, install the packages using pip:
code
Bash
pip install -r requirements.txt
4. Configure Environment Variables
Create a file named .env in the root of your project directory. This file will store all your secret keys and configuration details.
Copy the following template into your .env file and fill in your actual credentials.
.env
code
Env
# LiveKit Server Details
LIVEKIT_URL="ws://localhost:7880" # Replace with your LiveKit server URL
LIVEKIT_API_KEY="YOUR_LIVEKIT_API_KEY"
LIVEKIT_API_SECRET="YOUR_LIVEKIT_API_SECRET"

# AI Service API Keys
GROQ_API_KEY="YOUR_GROQ_API_KEY"
DEEPGRAM_API_KEY="YOUR_DEEPGRAM_API_KEY"
ELEVENLABS_API_KEY="YOUR_ELEVENLABS_API_KEY"

# PostgreSQL Database Connection Details
DB_HOST="localhost"           # Your database host
DB_PORT="5432"                # Your database port
DB_NAME="your_database_name"
DB_USER="your_database_user"
DB_PASSWORD="your_database_password"
Running the Application
The backend runs in two separate processes, which should be started in two separate terminals.
Terminal 1: Start the FastAPI Token Server
This server provides the connection token that your frontend needs to join the LiveKit room.
code
Bash
python main.py
You should see output from Uvicorn indicating the server is running on http://0.0.0.0:8000.
Terminal 2: Start the LiveKit Agent Worker
This is the main process for the voice assistant. It connects to LiveKit and waits to be assigned a room.
code
Bash
python main.py start
You will see logs indicating that the worker has registered and is ready for jobs.
3. Start Your Frontend
Navigate to your frontend project directory (e.g., your React app) and start its development server.
code
Bash
# Example for a React app
npm start
