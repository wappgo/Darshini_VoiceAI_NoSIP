import livekit.agents
import sys

print(f"Python executable: {sys.executable}")
print(f"livekit.agents module path: {livekit.agents.__file__}")
try:
    print(f"livekit.agents version: {livekit.agents.__version__}")
except AttributeError:
    print("livekit.agents version attribute not found.")

print("\nAttempting to find VoiceAssistant:")

try:
    from livekit.agents import VoiceAssistant
    print("Found VoiceAssistant in livekit.agents")
    print(VoiceAssistant)
except ImportError:
    print("Could NOT find VoiceAssistant in livekit.agents")
    print("Checking livekit.agents.voice_assistant...")
    try:
        from livekit.agents.voice_assistant import VoiceAssistant
        print("Found VoiceAssistant in livekit.agents.voice_assistant")
        print(VoiceAssistant)
    except ImportError:
        print("Could NOT find VoiceAssistant in livekit.agents.voice_assistant either.")
        print("\nContents of livekit.agents:")
        for name in dir(livekit.agents):
            if not name.startswith("_"):
                print(name)
        # If livekit.agents.voice_assistant might exist as a submodule:
        try:
            import livekit.agents.voice_assistant
            print("\nContents of livekit.agents.voice_assistant:")
            for name in dir(livekit.agents.voice_assistant):
                if not name.startswith("_"):
                    print(name)
        except ImportError:
            print("\nSubmodule livekit.agents.voice_assistant not found.")