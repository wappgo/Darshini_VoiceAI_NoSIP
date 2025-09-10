import React, { useState, useEffect } from 'react';
import { RoomEvent, LocalParticipant, Track } from 'livekit-client'; // Added Track for source check
import {
    RoomAudioRenderer,
    useRoomContext,
    useLocalParticipant,
    LiveKitRoom
} from '@livekit/components-react';
import './index.css'; // Your Tailwind CSS import

// Updated Guard: Check isMicrophoneEnabled is boolean, setMicrophoneEnabled is function
function participantHasMicControls(participant) {
  // Check if participant exists and has the expected controls with correct types
  return participant &&
         typeof participant.isMicrophoneEnabled === 'boolean' && // Check for boolean property
         typeof participant.setMicrophoneEnabled === 'function'; // Check for function method
}


function AppInner() {
  const [isConnected, setIsConnected] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [isMicOn, setIsMicOn] = useState(false); // Tracks the UI/desired mic state
  const [statusMessage, setStatusMessage] = useState('Initializing...');

  const { localParticipant } = useLocalParticipant(); // Hook providing the local participant object
  const room = useRoomContext(); // Hook providing the room object

  // Function to toggle the microphone
  const toggleMic = async () => {
    console.log('[AppInner:toggleMic] Attempting to toggle microphone.');
    // Use the updated guard
    if (!participantHasMicControls(localParticipant) || !room) {
        console.warn('[AppInner:toggleMic] Local participant/room not ready or mic controls invalid.');
        setStatusMessage('Cannot toggle mic: participant/room not ready or controls invalid.');
        return; // Exit if controls aren't available/valid
    }

    // Resume AudioContext if needed (essential for browser audio)
    if (room.audioContext && room.audioContext.state === "suspended") {
      try {
        await room.audioContext.resume();
        console.log("[AppInner:toggleMic] LiveKit Room.audioContext resumed.");
        setStatusMessage('AudioContext Resumed.');
      } catch (e) { console.error("[AppInner:toggleMic] Error resuming AudioContext:", e); }
    }
    // Ensure playback capability
    if (room.canPlaybackAudio === false) {
        try {
            await room.startAudio();
            console.log('[AppInner:toggleMic] room.startAudio() called.');
        } catch (e) { console.error('[AppInner:toggleMic] Error calling room.startAudio():', e); }
    }

    // Determine desired state based on current UI state
    const currentlyOn = isMicOn;
    const enableMic = !currentlyOn; // The desired new state

    try {
        console.log(`[AppInner:toggleMic] Requesting microphone ${enableMic ? 'enable' : 'disable'}.`);
        setStatusMessage(`Microphone turning ${enableMic ? 'On' : 'Off'}...`);
        // Call the setMicrophoneEnabled method (which IS a function)
        await localParticipant.setMicrophoneEnabled(enableMic, room.audioContext ? { audioContext: room.audioContext } : undefined);
        console.log(`[AppInner:toggleMic] setMicrophoneEnabled(${enableMic}) called successfully.`);
        // The actual UI state (isMicOn) will be updated by the useEffect below
        // when it detects the change in the localParticipant.isMicrophoneEnabled property or receives a mute event.
    } catch (e) {
        console.error(`[AppInner:toggleMic] Error calling setMicrophoneEnabled(${enableMic}):`, e);
        setStatusMessage(`Error ${enableMic ? 'enabling' : 'disabling'} mic.`);
    }
  };

  // Effect for Room Event Listeners & Connection State
  useEffect(() => {
    if (!room) {
        setStatusMessage('Waiting for room context...');
        console.log('[AppInner:RoomEffect] Room context not yet available.');
        return;
    }
    console.log('[AppInner:RoomEffect] Room available. Setting up listeners. Initial state:', room.state);
    setStatusMessage(`Room: ${room.state}`);

    // Handler for connection state changes
    const handleConnectionStateChange = (newState) => {
        console.log('[AppInner:Event] ConnectionStateChanged:', newState);
        setStatusMessage(`Connection: ${newState}`);
        if (newState === 'connected') {
             setIsConnected(true);
             console.log('[AppInner:Connected] Successfully connected to room (via ConnectionStateChanged).');
             // Initial mic state check is deferred to the LocalParticipantEffect
        } else if (newState === 'disconnected' || newState === 'failed') {
            setIsConnected(false);
            setIsMicOn(false); // Reset mic state on disconnect
            setTranscript('');
            setResponse('');
            console.log(`[AppInner:Disconnected] Disconnected/Failed. State: ${newState}`);
        }
    };

    // Other event handlers
    const onParticipantConnected = (participant) => {
      console.log('[AppInner:Event] ParticipantConnected:', { identity: participant.identity, sid: participant.sid, isAgent: participant.isAgent });
      setStatusMessage(`Participant ${participant.identity} joined.`);
    };
    const onParticipantDisconnected = (participant) => {
      console.log('[AppInner:Event] ParticipantDisconnected:', { identity: participant.identity, sid: participant.sid });
      setStatusMessage(`Participant ${participant.identity} left.`);
    };
    const onDataReceived = (payload, p, kind, topic) => {
      const str = new TextDecoder().decode(payload);
      console.log(`[AppInner:Event] DataReceived: Topic "${topic}", From "${p?.identity}", Payload "${str}"`);
      if (topic === "transcription" && str.startsWith('transcript:')) {
        setTranscript(str.replace('transcript:', ''));
      } else if (topic === "response" && str.startsWith('response:')) {
        setResponse(str.replace('response:', ''));
      }
    };

    // Register listeners using ConnectionStateChanged event
    room.on(RoomEvent.ConnectionStateChanged, handleConnectionStateChange);
    room.on(RoomEvent.DataReceived, onDataReceived);
    room.on(RoomEvent.ParticipantConnected, onParticipantConnected);
    room.on(RoomEvent.ParticipantDisconnected, onParticipantDisconnected);

    // Set initial state based on current room state when effect runs
    handleConnectionStateChange(room.state);

    // Cleanup function to remove listeners
    return () => {
      console.log('[AppInner:RoomEffectCleanup] Cleaning up room listeners.');
      room.off(RoomEvent.ConnectionStateChanged, handleConnectionStateChange);
      room.off(RoomEvent.DataReceived, onDataReceived);
      room.off(RoomEvent.ParticipantConnected, onParticipantConnected);
      room.off(RoomEvent.ParticipantDisconnected, onParticipantDisconnected);
    };
  }, [room]); // Depends only on the room object reference


  // Effect specifically for handling the Local Participant object updates
  useEffect(() => {
    if (localParticipant) {
        console.log('[AppInner:LocalParticipantEffect] localParticipant updated:', localParticipant);

        // --- Read isMicrophoneEnabled as a PROPERTY ---
        let micIsEnabledProperty = false;
        if (typeof localParticipant.isMicrophoneEnabled === 'boolean') {
             micIsEnabledProperty = localParticipant.isMicrophoneEnabled; // Read property
             console.log(`[AppInner:LocalParticipantEffect] Mic state read from property: ${micIsEnabledProperty}`);
             // Update status message only when connected and property is valid
             if (isConnected) {
                 setStatusMessage(micIsEnabledProperty ? 'Microphone On.' : 'Microphone Off.');
             }
             setIsMicOn(micIsEnabledProperty); // Update UI state based on property
        } else {
            console.warn('[AppInner:LocalParticipantEffect] localParticipant.isMicrophoneEnabled IS NOT a boolean property.');
            setStatusMessage('Mic status unavailable (type error).');
             // Keep previous isMicOn state or default to false if necessary
             // setIsMicOn(false); // Optionally reset if property is invalid
        }
        // --- End of property reading change ---

        // Safely log audio tracks
        if (localParticipant.audioTracks && typeof localParticipant.audioTracks.values === 'function') {
             const trackPublications = Array.from(localParticipant.audioTracks.values());
             console.log('[AppInner:LocalParticipantEffect] Local participant audioTracks:',
                 trackPublications.map(pub => ({ sid: pub.trackSid, kind: pub.kind, source: pub.source, muted: pub.isMuted ? pub.isMuted() : 'N/A' }))
             );
        } else {
            console.warn('[AppInner:LocalParticipantEffect] localParticipant.audioTracks not available or not a Map.');
        }

        // Listen for mute/unmute events to confirm state changes
        const handleMicTrackMuteChange = (trackPublication) => {
             // Ensure it's the microphone track changing state
             if (trackPublication.source === Track.Source.Microphone) {
                console.log(`[AppInner:LocalParticipantEvent] Mic Track ${trackPublication.isMuted ? 'Muted' : 'Unmuted'} event.`);
                // Re-read the property when event occurs to ensure sync
                let currentMicState = false;
                if (typeof localParticipant.isMicrophoneEnabled === 'boolean') {
                     currentMicState = localParticipant.isMicrophoneEnabled;
                     setIsMicOn(currentMicState); // Update UI state
                     if(isConnected) {
                        setStatusMessage(currentMicState ? 'Microphone On.' : 'Microphone Off.');
                     }
                     console.log(`[AppInner:LocalParticipantEvent] Mic state updated via event to: ${currentMicState}`);
                 } else {
                     console.warn('[AppInner:LocalParticipantEvent] isMicrophoneEnabled not boolean on mute/unmute event.');
                 }
             }
        };

        localParticipant.on(RoomEvent.TrackMuted, handleMicTrackMuteChange);
        localParticipant.on(RoomEvent.TrackUnmuted, handleMicTrackMuteChange);

        // Cleanup listeners for this participant
        return () => {
            console.log('[AppInner:LocalParticipantEffectCleanup] Cleaning up participant listeners.');
            // Check if localParticipant still exists before removing listeners
            if(localParticipant) {
              localParticipant.off(RoomEvent.TrackMuted, handleMicTrackMuteChange);
              localParticipant.off(RoomEvent.TrackUnmuted, handleMicTrackMuteChange);
            }
        };

    } else {
        // Handle case where localParticipant becomes null/undefined
        console.log('[AppInner:LocalParticipantEffect] localParticipant is null/undefined.');
        setIsMicOn(false); // Ensure mic state is off
    }
  // Depend on localParticipant reference and connection status
  }, [localParticipant, isConnected]);


  // Effect for logging Remote Participant Tracks
  useEffect(() => {
    if (!room) {
        console.log('[AppInner:RemoteTracksEffect] Room is null.');
        return;
    }

    const logRemoteTracks = (context = "Update") => {
        const remoteParticipants = Array.from(room.remoteParticipants.values());
        console.log(`[AppInner:RemoteTracksEffect:${context}] Remote participants count: ${remoteParticipants.length}`);
        remoteParticipants.forEach(p => {
          if (p.audioTracks && typeof p.audioTracks.values === 'function') {
            const audioPubs = Array.from(p.audioTracks.values());
            console.log(`  [RemoteTracks] For ${p.identity} (isAgent: ${p.isAgent}): ${audioPubs.length} audio track pubs. SIDs: ${audioPubs.map(pub => pub.trackSid).join(', ')} Muted: ${audioPubs.map(pub => pub.isMuted ? pub.isMuted() : 'N/A').join(', ')}`);
          } else {
            console.warn(`  [RemoteTracks] For ${p.identity}: p.audioTracks not available or not a Map.`);
          }
        });
         const hasRemoteAudio = remoteParticipants.some(p => p.audioTracks && p.audioTracks.size > 0);
         console.log(`[AppInner:RemoteTracksEffect:${context}] Has Remote Audio Tracks? ${hasRemoteAudio}`);
    };

    const handleTrackSubscribed = (track, publication, participant) => {
        console.log(`[AppInner:Event] TrackSubscribed: ${track.kind} SID ${track.sid} for ${participant.identity}`);
        logRemoteTracks("TrackSubscribed");
    };
    const handleTrackUnsubscribed = (track, publication, participant) => {
        console.log(`[AppInner:Event] TrackUnsubscribed: ${track.kind} SID ${track.sid} for ${participant.identity}`);
        logRemoteTracks("TrackUnsubscribed");
    };
    const handleTrackPublished = (publication, participant) => {
        if (!participant.isLocal) {
            console.log(`[AppInner:Event] Remote TrackPublished by ${participant.identity}: ${publication.kind} source ${publication.source}`);
            logRemoteTracks("TrackPublished");
        }
    };

    logRemoteTracks("Initial");
    room.on(RoomEvent.TrackSubscribed, handleTrackSubscribed);
    room.on(RoomEvent.TrackUnsubscribed, handleTrackUnsubscribed);
    room.on(RoomEvent.TrackPublished, handleTrackPublished);

    return () => {
        console.log('[AppInner:RemoteTracksEffectCleanup] Cleaning up track listeners.');
        room.off(RoomEvent.TrackSubscribed, handleTrackSubscribed);
        room.off(RoomEvent.TrackUnsubscribed, handleTrackUnsubscribed);
        room.off(RoomEvent.TrackPublished, handleTrackPublished);
    };
  }, [room]);

  // Render the UI
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
      <h1 className="text-3xl font-bold mb-6">Voice Assistant</h1>
      <div className="bg-white p-6 rounded-lg shadow-md w-full max-w-md">
        {/* UI reflects connection and mic state */}
        <div className="mb-4">
          <p className="text-lg">Status: {statusMessage}</p>
          <p className="text-sm">LiveKit Connected: {isConnected ? 'Yes' : 'No'} | Mic Active: {isMicOn ? 'On' : 'Off'}</p>
          {room?.audioContext && <p className="text-sm">Browser AudioContext: {room.audioContext.state}</p>}
        </div>
        {/* Microphone toggle button - uses updated guard */}
        <button
          onClick={toggleMic}
          disabled={!isConnected || !participantHasMicControls(localParticipant)}
          className={`w-full py-2 px-4 rounded-md text-white font-semibold ${
            isMicOn ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'
          } ${(!isConnected || !participantHasMicControls(localParticipant)) ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          {isMicOn ? 'Stop Microphone' : 'Start Microphone'}
        </button>
        {/* Transcript display */}
        <div className="mt-6">
          <h2 className="text-lg font-semibold">Transcript</h2>
          <p className="mt-2 p-2 bg-gray-100 rounded min-h-[2em]">{transcript || 'Waiting for input...'}</p>
        </div>
        {/* Response display */}
        <div className="mt-4">
          <h2 className="text-lg font-semibold">Response</h2>
          <p className="mt-2 p-2 bg-gray-100 rounded min-h-[2em]">{response || 'Waiting for response...'}</p>
        </div>
      </div>
      {/* Component responsible for playing remote audio tracks */}
      <RoomAudioRenderer />
    </div>
  );
}


// Wrapper component to fetch token and provide LiveKitRoom context
export default function AppWrapper() {
  const [token, setToken] = useState('');
  const [url, setUrl] = useState('');
  const [loadingMessage, setLoadingMessage] = useState('Loading configuration...');

  // Effect to fetch token from backend on component mount
  useEffect(() => {
    console.log("[AppWrapper:useEffect] Fetching token...");
    setLoadingMessage('Fetching token from backend...');
    fetch('http://localhost:8000/token')
      .then(res => {
        if (!res.ok) {
          return res.text().then(text => {
            throw new Error(`HTTP error! status: ${res.status}, body: ${text}`);
          });
        }
        return res.json();
      })
      .then(data => {
        if (data.token && data.url) {
          setToken(data.token);
          setUrl(data.url);
          console.log("[AppWrapper:useEffect] Token and URL fetched successfully.");
          setLoadingMessage('Configuration loaded. Connecting to LiveKit...');
        } else {
          console.error('[AppWrapper:useEffect] Error: Token or URL missing in response', data);
          setLoadingMessage('Error: Token or URL missing in server response.');
        }
      })
      .catch(err => {
        console.error('[AppWrapper:useEffect] Error fetching token:', err);
        setLoadingMessage(`Error fetching token: ${err.message}. Is backend running?`);
      });
  }, []); // Empty dependency array means run only once on mount

  // Display loading message until token/url are ready
  if (!token || !url) {
    return (
        <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
            <p className="text-xl mb-4">{loadingMessage}</p>
            <p className="text-sm">Ensure backend server is running on port 8000.</p>
        </div>
    );
  }

  // Render the LiveKitRoom context provider
  return (
    <LiveKitRoom
      token={token}
      serverUrl={url}
      connect={true} // Automatically connect on mount
      // roomName is NOT needed here; it's defined in the token.
      audio={true} // Enable audio capabilities
      video={false} // Disable video capabilities
      // Event listeners directly on LiveKitRoom for top-level feedback/debugging
      onConnected={() => {
        console.log("[AppWrapper:LiveKitRoomEvent] Connected");
        // setLoadingMessage('Successfully connected to LiveKit room!'); // Can update status if needed
      }}
      onDisconnected={(reason) => {
        console.log(`[AppWrapper:LiveKitRoomEvent] Disconnected. Reason: ${reason}`);
        setLoadingMessage(`Disconnected from LiveKit: ${reason || 'No reason provided'}`);
      }}
      onError={(error) => {
        console.error("[AppWrapper:LiveKitRoomEvent] Connection error:", error);
        setLoadingMessage(`LiveKit connection error: ${error.message}. Check console.`);
      }}
      onParticipantConnected={(participant) => console.log(`[AppWrapper:LiveKitRoomEvent] Participant connected: ${participant.identity}`)}
      onParticipantDisconnected={(participant) => console.log(`[AppWrapper:LiveKitRoomEvent] Participant disconnected: ${participant.identity}`)}
      onMediaDeviceFailure={(error) => console.error("[AppWrapper:LiveKitRoomEvent] Media device failure:", error)}
    >
      {/* AppInner consumes the context provided by LiveKitRoom */}
      <AppInner />
    </LiveKitRoom>
  );
}







// 15-05-2025






import React, { useState, useEffect, useRef } from 'react'; // Added useRef for scrolling
import { RoomEvent, LocalParticipant, Track } from 'livekit-client';
import {
    RoomAudioRenderer,
    useRoomContext,
    useLocalParticipant,
    LiveKitRoom
} from '@livekit/components-react';
import './index.css'; // Your Tailwind CSS import

// (participantHasMicControls function remains the same)
function participantHasMicControls(participant) {
  return participant &&
         typeof participant.isMicrophoneEnabled === 'boolean' &&
         typeof participant.setMicrophoneEnabled === 'function';
}


function AppInner() {
  const [isConnected, setIsConnected] = useState(false);
  // const [transcript, setTranscript] = useState(''); // REMOVE
  // const [response, setResponse] = useState('');   // REMOVE
  const [chatMessages, setChatMessages] = useState([]); // NEW: Array to store chat messages
  const [isMicOn, setIsMicOn] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Initializing...');

  const { localParticipant } = useLocalParticipant();
  const room = useRoomContext();
  const chatEndRef = useRef(null); // Ref for auto-scrolling

  // Auto-scroll to the bottom of the chat messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]); // Dependency on chatMessages

  // Function to toggle the microphone (remains largely the same logic)
  const toggleMic = async () => {
    console.log('[AppInner:toggleMic] Attempting to toggle microphone.');
    if (!participantHasMicControls(localParticipant) || !room) {
        console.warn('[AppInner:toggleMic] Local participant/room not ready or mic controls invalid.');
        setStatusMessage('Cannot toggle mic: participant/room not ready or controls invalid.');
        return;
    }
    if (room.audioContext && room.audioContext.state === "suspended") {
      try {
        await room.audioContext.resume();
        console.log("[AppInner:toggleMic] LiveKit Room.audioContext resumed.");
        setStatusMessage('AudioContext Resumed.');
      } catch (e) { console.error("[AppInner:toggleMic] Error resuming AudioContext:", e); }
    }
    if (room.canPlaybackAudio === false) {
        try {
            await room.startAudio();
            console.log('[AppInner:toggleMic] room.startAudio() called.');
        } catch (e) { console.error('[AppInner:toggleMic] Error calling room.startAudio():', e); }
    }
    const currentlyOn = isMicOn;
    const enableMic = !currentlyOn;
    try {
        console.log(`[AppInner:toggleMic] Requesting microphone ${enableMic ? 'enable' : 'disable'}.`);
        setStatusMessage(`Microphone turning ${enableMic ? 'On' : 'Off'}...`);
        await localParticipant.setMicrophoneEnabled(enableMic, room.audioContext ? { audioContext: room.audioContext } : undefined);
        console.log(`[AppInner:toggleMic] setMicrophoneEnabled(${enableMic}) called successfully.`);
    } catch (e) {
        console.error(`[AppInner:toggleMic] Error calling setMicrophoneEnabled(${enableMic}):`, e);
        setStatusMessage(`Error ${enableMic ? 'enabling' : 'disabling'} mic.`);
    }
  };

  // Effect for Room Event Listeners & Connection State
  useEffect(() => {
    if (!room) {
        setStatusMessage('Waiting for room context...');
        return;
    }
    console.log('[AppInner:RoomEffect] Room available. Initial state:', room.state);
    setStatusMessage(`Room: ${room.state}`);

    const handleConnectionStateChange = (newState) => {
        console.log('[AppInner:Event] ConnectionStateChanged:', newState);
        setStatusMessage(`Connection: ${newState}`);
        if (newState === 'connected') {
             setIsConnected(true);
             console.log('[AppInner:Connected] Successfully connected to room.');
             // Add initial agent greeting when connected (if not already added)
             // This assumes the agent sends its greeting via "response" topic
        } else if (newState === 'disconnected' || newState === 'failed') {
            setIsConnected(false);
            setIsMicOn(false);
            setChatMessages([]); // Clear chat on disconnect
            console.log(`[AppInner:Disconnected] Disconnected/Failed. State: ${newState}`);
        }
    };

    const onParticipantConnected = (participant) => {
      console.log('[AppInner:Event] ParticipantConnected:', { identity: participant.identity, sid: participant.sid, isAgent: participant.isAgent });
      setStatusMessage(`Participant ${participant.identity} joined.`);
    };
    const onParticipantDisconnected = (participant) => {
      console.log('[AppInner:Event] ParticipantDisconnected:', { identity: participant.identity, sid: participant.sid });
      setStatusMessage(`Participant ${participant.identity} left.`);
    };

    const onDataReceived = (payload, p, kind, topic) => {
      const str = new TextDecoder().decode(payload);
      console.log(`[AppInner:Event] DataReceived: Topic "${topic}", From "${p?.identity}", Payload "${str}"`);
      
      let sender = '';
      let messageText = '';

      if (topic === "transcription" && str.startsWith('transcript:')) {
        sender = 'You';
        messageText = str.replace('transcript:', '').trim();
        console.log('[AppInner:Debug] Transcription processed:', { sender, messageText });
      } else if (topic === "response" && str.startsWith('response:')) {
        sender = 'Agent';
        messageText = str.replace('response:', '').trim();
        console.log('[AppInner:Debug] Response processed:', { sender, messageText });
      }

      console.log('[AppInner:Debug] Before adding to chat:', { sender, messageText, topic, str });

      if (sender && messageText) {
        setChatMessages(prevMessages => [...prevMessages, { sender, text: messageText, id: Date.now() + Math.random() }]);
      }
    };

    room.on(RoomEvent.ConnectionStateChanged, handleConnectionStateChange);
    room.on(RoomEvent.DataReceived, onDataReceived);
    room.on(RoomEvent.ParticipantConnected, onParticipantConnected);
    room.on(RoomEvent.ParticipantDisconnected, onParticipantDisconnected);

    handleConnectionStateChange(room.state);

    return () => {
      console.log('[AppInner:RoomEffectCleanup] Cleaning up room listeners.');
      room.off(RoomEvent.ConnectionStateChanged, handleConnectionStateChange);
      room.off(RoomEvent.DataReceived, onDataReceived);
      room.off(RoomEvent.ParticipantConnected, onParticipantConnected);
      room.off(RoomEvent.ParticipantDisconnected, onParticipantDisconnected);
    };
  }, [room]);


  // Effect specifically for handling the Local Participant object updates
  useEffect(() => {
    if (localParticipant) {
        console.log('[AppInner:LocalParticipantEffect] localParticipant updated:', localParticipant);
        let micIsEnabledProperty = false;
        if (typeof localParticipant.isMicrophoneEnabled === 'boolean') {
             micIsEnabledProperty = localParticipant.isMicrophoneEnabled;
             if (isConnected) {
                 setStatusMessage(micIsEnabledProperty ? 'Microphone On.' : 'Microphone Off.');
             }
             setIsMicOn(micIsEnabledProperty);
        } else {
            console.warn('[AppInner:LocalParticipantEffect] localParticipant.isMicrophoneEnabled IS NOT a boolean property.');
            setStatusMessage('Mic status unavailable (type error).');
        }

        const handleMicTrackMuteChange = (trackPublication) => {
             if (trackPublication.source === Track.Source.Microphone) {
                console.log(`[AppInner:LocalParticipantEvent] Mic Track ${trackPublication.isMuted ? 'Muted' : 'Unmuted'} event.`);
                let currentMicState = false;
                if (typeof localParticipant.isMicrophoneEnabled === 'boolean') {
                     currentMicState = localParticipant.isMicrophoneEnabled;
                     setIsMicOn(currentMicState);
                     if(isConnected) {
                        setStatusMessage(currentMicState ? 'Microphone On.' : 'Microphone Off.');
                     }
                 }
             }
        };
        localParticipant.on(RoomEvent.TrackMuted, handleMicTrackMuteChange);
        localParticipant.on(RoomEvent.TrackUnmuted, handleMicTrackMuteChange);
        return () => {
            if(localParticipant) {
              localParticipant.off(RoomEvent.TrackMuted, handleMicTrackMuteChange);
              localParticipant.off(RoomEvent.TrackUnmuted, handleMicTrackMuteChange);
            }
        };
    } else {
        setIsMicOn(false);
    }
  }, [localParticipant, isConnected]);


  // Effect for logging Remote Participant Tracks (remains the same)
  useEffect(() => {
    if (!room) { return; }
    const logRemoteTracks = (context = "Update") => { /* ... */ };
    const handleTrackSubscribed = (track, publication, participant) => { /* ... */ logRemoteTracks("TrackSubscribed"); };
    const handleTrackUnsubscribed = (track, publication, participant) => { /* ... */ logRemoteTracks("TrackUnsubscribed"); };
    const handleTrackPublished = (publication, participant) => { if (!participant.isLocal) { /* ... */ logRemoteTracks("TrackPublished"); }};
    logRemoteTracks("Initial");
    room.on(RoomEvent.TrackSubscribed, handleTrackSubscribed);
    room.on(RoomEvent.TrackUnsubscribed, handleTrackUnsubscribed);
    room.on(RoomEvent.TrackPublished, handleTrackPublished);
    return () => {
        room.off(RoomEvent.TrackSubscribed, handleTrackSubscribed);
        room.off(RoomEvent.TrackUnsubscribed, handleTrackUnsubscribed);
        room.off(RoomEvent.TrackPublished, handleTrackPublished);
    };
  }, [room]);

  // Render the UI
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4 font-sans">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-lg flex flex-col" style={{height: '80vh', maxHeight: '700px'}}>
        {/* Header */}
        <div className="p-4 border-b border-gray-200">
            <h1 className="text-xl font-semibold text-center text-gray-700">Voice Assistant</h1>
            <p className="text-xs text-center text-gray-500">{statusMessage} | Mic: {isMicOn ? 'On' : 'Off'} | Connected: {isConnected ? 'Yes' : 'No'}</p>
        </div>

        {/* Chat Messages Area */}
        <div className="flex-grow p-4 overflow-y-auto space-y-4">
          {chatMessages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.sender === 'You' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg shadow ${
                  msg.sender === 'You'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 text-gray-800'
                }`}
              >
                <p className="text-sm font-semibold mb-1">{msg.sender}</p>
                <p className="text-sm">{msg.text}</p>
              </div>
            </div>
          ))}
          <div ref={chatEndRef} /> {/* For auto-scrolling */}
        </div>

        {/* Controls Area */}
        <div className="p-4 border-t border-gray-200">
          <button
            onClick={toggleMic}
            disabled={!isConnected || !participantHasMicControls(localParticipant)}
            className={`w-full py-3 px-4 rounded-lg text-white font-semibold transition-colors duration-150
              ${!isConnected || !participantHasMicControls(localParticipant)
                ? 'bg-gray-400 cursor-not-allowed'
                : isMicOn
                  ? 'bg-red-500 hover:bg-red-600'
                  : 'bg-green-500 hover:bg-green-600'
              }`}
          >
             <div className="flex items-center justify-center">
                  <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                      {isMicOn ? (
                          // Icon for "Microphone is ON" / "Stop Microphone" (Microphone Slash)
                          <>
                              <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8h-1a6 6 0 11-5.445-2.916l.904.904A4.999 4.999 0 1011 14.93V18a1 1 0 102 0v-3.07zM12 3a1 1 0 011 1v4a1 1 0 11-2 0V4a1 1 0 011-1z" clipRule="evenodd" />
                              <path fillRule="evenodd" d="M3.05 3.05a1 1 0 011.414 0L16.95 15.536a1 1 0 01-1.414 1.414L3.05 4.464a1 1 0 010-1.414z" clipRule="evenodd" />
                          </>
                      ) : (
                          // Icon for "Microphone is OFF" / "Start Microphone" (Standard Microphone)
                          <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8h-1a6 6 0 11-5.445-2.916l.904.904A4.999 4.999 0 1011 14.93V18a1 1 0 102 0v-3.07zM12 3a1 1 0 011 1v4a1 1 0 11-2 0V4a1 1 0 011-1z" clipRule="evenodd" />
                      )}
                  </svg>
                  {isMicOn ? 'Stop Microphone' : 'Start Microphone'}
              </div>
          </button>
        </div>
      </div>
      <RoomAudioRenderer />
    </div>
  );
}


// AppWrapper remains the same as your provided version
export default function AppWrapper() {
  const [token, setToken] = useState('');
  const [url, setUrl] = useState('');
  const [loadingMessage, setLoadingMessage] = useState('Loading configuration...');

  useEffect(() => {
    console.log("[AppWrapper:useEffect] Fetching token...");
    setLoadingMessage('Fetching token from backend...');
    fetch('http://localhost:8000/token')
      .then(res => {
        if (!res.ok) {
          return res.text().then(text => {
            throw new Error(`HTTP error! status: ${res.status}, body: ${text}`);
          });
        }
        return res.json();
      })
      .then(data => {
        if (data.token && data.url) {
          setToken(data.token);
          setUrl(data.url);
          console.log("[AppWrapper:useEffect] Token and URL fetched successfully.");
          setLoadingMessage('Configuration loaded. Connecting to LiveKit...');
        } else {
          console.error('[AppWrapper:useEffect] Error: Token or URL missing in response', data);
          setLoadingMessage('Error: Token or URL missing in server response.');
        }
      })
      .catch(err => {
        console.error('[AppWrapper:useEffect] Error fetching token:', err);
        setLoadingMessage(`Error fetching token: ${err.message}. Is backend running?`);
      });
  }, []);

  if (!token || !url) {
    return (
        <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
            <p className="text-xl mb-4">{loadingMessage}</p>
            <p className="text-sm">Ensure backend server is running on port 8000.</p>
        </div>
    );
  }

  return (
    <LiveKitRoom
      token={token}
      serverUrl={url}
      connect={true}
      audio={true}
      video={false}
      onConnected={() => { console.log("[AppWrapper:LiveKitRoomEvent] Connected"); }}
      onDisconnected={(reason) => {
        console.log(`[AppWrapper:LiveKitRoomEvent] Disconnected. Reason: ${reason}`);
        setLoadingMessage(`Disconnected from LiveKit: ${reason || 'No reason provided'}`);
      }}
      onError={(error) => {
        console.error("[AppWrapper:LiveKitRoomEvent] Connection error:", error);
        setLoadingMessage(`LiveKit connection error: ${error.message}. Check console.`);
      }}
      onParticipantConnected={(participant) => console.log(`[AppWrapper:LiveKitRoomEvent] Participant connected: ${participant.identity}`)}
      onParticipantDisconnected={(participant) => console.log(`[AppWrapper:LiveKitRoomEvent] Participant disconnected: ${participant.identity}`)}
      onMediaDeviceFailure={(error) => console.error("[AppWrapper:LiveKitRoomEvent] Media device failure:", error)}
    >
      <AppInner />
    </LiveKitRoom>
  );
}


















//  oepn source model complete working code 24-05-2025 saturday
// app.jsx file of englog file 






// Complete content for app.jsx

import React, { useState, useEffect, useRef } from 'react';
import { RoomEvent, Track, LocalParticipant } from 'livekit-client'; // Explicit imports
import {
    RoomAudioRenderer,
    useRoomContext,
    useLocalParticipant,
    LiveKitRoom
} from '@livekit/components-react';
import './index.css'; // Your Tailwind CSS import (ensure this file exists and is configured)

// Helper function to check if participant has microphone controls
function participantHasMicControls(participant) {
  return participant &&
         typeof participant.isMicrophoneEnabled === 'boolean' &&
         typeof participant.setMicrophoneEnabled === 'function';
}

// Inner component that operates within the LiveKitRoom context
function AppInner() {
  const [isConnected, setIsConnected] = useState(false);
  const [chatMessages, setChatMessages] = useState([]); // Stores chat messages: { sender, text, id }
  const [isMicOn, setIsMicOn] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Initializing...');

  const { localParticipant } = useLocalParticipant(); // Hook to get the local participant
  const room = useRoomContext(); // Hook to get the room context
  const chatEndRef = useRef(null); // Ref for auto-scrolling the chat

  // Effect to auto-scroll chat to the latest message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  // Function to toggle the microphone
  const toggleMic = async () => {
    console.log('[AppInner:toggleMic] Attempting to toggle microphone.');
    if (!participantHasMicControls(localParticipant) || !room) {
        console.warn('[AppInner:toggleMic] Local participant/room not ready or mic controls invalid.');
        setStatusMessage('Cannot toggle mic: participant/room not ready.');
        return;
    }
    // Attempt to resume AudioContext if suspended (common browser behavior)
    if (room.audioContext && room.audioContext.state === "suspended") {
      try {
        await room.audioContext.resume();
        console.log("[AppInner:toggleMic] AudioContext resumed successfully.");
      } catch (e) {
        console.error("[AppInner:toggleMic] Error resuming AudioContext:", e);
      }
    }
    // Attempt to start audio playback if not already allowed
    if (room.canPlaybackAudio === false) {
        try {
            await room.startAudio();
            console.log('[AppInner:toggleMic] room.startAudio() called successfully.');
        } catch (e) {
            console.error('[AppInner:toggleMic] Error calling room.startAudio():', e);
        }
    }

    const enableMic = !isMicOn; // Determine target microphone state
    try {
        setStatusMessage(`Microphone turning ${enableMic ? 'On' : 'Off'}...`);
        await localParticipant.setMicrophoneEnabled(enableMic, room.audioContext ? { audioContext: room.audioContext } : undefined);
        // Actual state (isMicOn, statusMessage for mic) will be updated by LocalParticipantEffect via events
    } catch (e) {
        console.error(`[AppInner:toggleMic] Error calling setMicrophoneEnabled(${enableMic}):`, e);
        setStatusMessage(`Error ${enableMic ? 'enabling' : 'disabling'} mic.`);
    }
  };

  // Effect for handling Room events and connection state
  useEffect(() => {
    if (!room) {
        setStatusMessage('Waiting for room context...');
        return;
    }
    setStatusMessage(`Room: ${room.state}`); // Initial room state

    // Handler for connection state changes
    const handleConnectionStateChange = (newState) => {
        console.log('[AppInner:Event] ConnectionStateChanged:', newState);
        setStatusMessage(`Connection: ${newState}`);
        setIsConnected(newState === 'connected');
        if (newState === 'disconnected' || newState === 'failed') {
            setIsMicOn(false); // Reset mic state on disconnect
        }
    };

    // Handler for new participant connections
    const onParticipantConnected = (participant) => {
      console.log('[AppInner:Event] ParticipantConnected:', { identity: participant.identity, sid: participant.sid, isAgent: participant.isAgent });
      setStatusMessage(`Participant ${participant.identity} joined.`);
    };

    // Handler for participant disconnections
    const onParticipantDisconnected = (participant) => {
      console.log('[AppInner:Event] ParticipantDisconnected:', { identity: participant.identity, sid: participant.sid });
      setStatusMessage(`Participant ${participant.identity} left.`);
    };

    // Handler for receiving data messages from the backend
    const onDataReceived = (payload, p, kind, topic) => {
      const str = new TextDecoder().decode(payload);
      console.log(`[AppInner:Event] DataReceived: Topic "${topic}", From "${p?.identity}", Payload "${str}"`);

      let sender = '';
      let messageText = '';
      let messageSource = ''; // For debugging which path added the message

      // Primary logic for displaying chat messages from 'lk_chat_history' topic
      if (topic === "lk_chat_history") {
        messageSource = 'lk_chat_history';
        if (str.startsWith('user_msg:')) {
          sender = 'You';
          messageText = str.replace('user_msg:', '').trim();
        } else if (str.startsWith('agent_msg:')) {
          sender = 'Agent';
          messageText = str.replace('agent_msg:', '').trim();
        }
      }
      // Optional: Handle older topics or partial updates if needed for other UI elements
      // else if (topic === "transcription" && str.startsWith('transcript:')) { /* ... */ }
      // else if (topic === "response" && str.startsWith('response:')) { /* ... */ }
      else if (topic === "transcription_partial" && str.startsWith("transcription_update:")) {
          // const partialText = str.replace("transcription_update:", "").trim();
          // console.log("Live transcript (not added to chat history):", partialText);
          return; // Do not add partials to the main chat history display
      } else if (topic === "response_partial" && str.startsWith("response_partial:")) {
          // const partialResponse = str.replace("response_partial:", "").trim();
          // console.log("Live agent response (not added to chat history):", partialResponse);
          return; // Do not add partials to the main chat history display
      }

      // If a valid sender and message text were parsed, add to chat messages
      if (sender && messageText) {
        console.log(`[AppInner:Debug] Adding to chat from ${messageSource}: Sender='${sender}', Text='${messageText}'`);
        setChatMessages(prevMessages => {
          // Basic de-duplication: check if a very similar message was added recently
          const isRecentDuplicate = prevMessages.some(
            (msg) => msg.text === messageText && msg.sender === sender && (Date.now() - msg.id) < 3000 // Check within last 3 seconds
          );
          if (isRecentDuplicate) {
            console.log("[AppInner:Debug] Recent duplicate message skipped:", {sender, messageText});
            return prevMessages;
          }
          return [...prevMessages, { sender, text: messageText, id: Date.now() }]; // Use Date.now() as a simple unique ID
        });
      }
    };

    // Subscribe to room events
    room.on(RoomEvent.ConnectionStateChanged, handleConnectionStateChange);
    room.on(RoomEvent.DataReceived, onDataReceived);
    room.on(RoomEvent.ParticipantConnected, onParticipantConnected);
    room.on(RoomEvent.ParticipantDisconnected, onParticipantDisconnected);

    handleConnectionStateChange(room.state); // Set initial state

    // Cleanup function to unsubscribe from events when component unmounts
    return () => {
        console.log('[AppInner:RoomEffectCleanup] Cleaning up room listeners.');
        room.off(RoomEvent.ConnectionStateChanged, handleConnectionStateChange);
        room.off(RoomEvent.DataReceived, onDataReceived);
        room.off(RoomEvent.ParticipantConnected, onParticipantConnected);
        room.off(RoomEvent.ParticipantDisconnected, onParticipantDisconnected);
    };
  }, [room]); // Re-run effect if room object changes

  // Effect for handling Local Participant microphone state updates
  useEffect(() => {
    if (localParticipant) {
        const updateMicStateBasedOnParticipant = () => {
            if (typeof localParticipant.isMicrophoneEnabled === 'boolean') {
                const micEnabled = localParticipant.isMicrophoneEnabled;
                setIsMicOn(micEnabled);
                if(isConnected) { // Only update status message if connected
                    setStatusMessage(micEnabled ? 'Microphone On.' : 'Microphone Off.');
                }
            } else {
                 console.warn('[AppInner:LocalParticipantEffect] localParticipant.isMicrophoneEnabled is not a boolean.');
            }
        };
        
        updateMicStateBasedOnParticipant(); // Initial check of mic state

        // Handler for track muted/unmuted events (more reliable for mic state)
        const handleMicTrackMuteChange = (trackPublication) => {
             if (trackPublication.source === Track.Source.Microphone) {
                console.log(`[AppInner:LocalParticipantEvent] Mic Track ${trackPublication.isMuted ? 'Muted' : 'Unmuted'} (isMicrophoneEnabled: ${localParticipant.isMicrophoneEnabled}).`);
                updateMicStateBasedOnParticipant();
             }
        };
        localParticipant.on(RoomEvent.TrackMuted, handleMicTrackMuteChange);
        localParticipant.on(RoomEvent.TrackUnmuted, handleMicTrackMuteChange);
        
        // Cleanup
        return () => {
            if(localParticipant) {
              localParticipant.off(RoomEvent.TrackMuted, handleMicTrackMuteChange);
              localParticipant.off(RoomEvent.TrackUnmuted, handleMicTrackMuteChange);
            }
        };
    } else { // If no local participant, default mic to off
        setIsMicOn(false);
        if (isConnected) setStatusMessage('Microphone Off (No local participant).');
    }
  }, [localParticipant, isConnected]); // Re-run if localParticipant or isConnected changes

  // UI Rendering
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4 font-sans">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-lg flex flex-col" style={{height: '80vh', maxHeight: '700px'}}>
        {/* Header Section */}
        <div className="p-4 border-b border-gray-200">
            <h1 className="text-xl font-semibold text-center text-gray-700">Voice Assistant</h1>
            <p className="text-xs text-center text-gray-500">{statusMessage} | Mic: {isMicOn ? 'On' : 'Off'} | Connected: {isConnected ? 'Yes' : 'No'}</p>
        </div>

        {/* Chat Messages Area */}
        <div className="flex-grow p-4 overflow-y-auto space-y-4 bg-slate-50">
          {chatMessages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.sender === 'You' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-[80%] lg:max-w-[75%] px-4 py-2 rounded-xl shadow ${
                  msg.sender === 'You'
                    ? 'bg-blue-500 text-white' // User's messages
                    : 'bg-gray-200 text-gray-800' // Agent's messages
                }`}
              >
                <p className="text-sm break-words">{msg.text}</p>
              </div>
            </div>
          ))}
          <div ref={chatEndRef} /> {/* Invisible element for auto-scrolling */}
        </div>

        {/* Controls Area (Microphone Button) */}
        <div className="p-4 border-t border-gray-200">
          <button
            onClick={toggleMic}
            disabled={!isConnected || !participantHasMicControls(localParticipant)}
            className={`w-full py-3 px-4 rounded-lg text-white font-semibold transition-colors duration-150
              ${!isConnected || !participantHasMicControls(localParticipant)
                ? 'bg-gray-400 cursor-not-allowed' // Disabled state
                : isMicOn
                  ? 'bg-red-500 hover:bg-red-600 focus:ring-2 focus:ring-red-400' // Mic is On
                  : 'bg-green-500 hover:bg-green-600 focus:ring-2 focus:ring-green-400' // Mic is Off
              }`}
          >
             <div className="flex items-center justify-center">
                  {/* Microphone Icon */}
                  <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                      {isMicOn ? ( // Icon for "Stop Microphone" (Mic Slash)
                          <>
                            <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8h-1a6 6 0 11-5.445-2.916l.904.904A4.999 4.999 0 1011 14.93V18a1 1 0 102 0v-3.07zM12 3a1 1 0 011 1v4a1 1 0 11-2 0V4a1 1 0 011-1z" clipRule="evenodd" />
                            <path fillRule="evenodd" d="M3.05 3.05a1 1 0 011.414 0L16.95 15.536a1 1 0 01-1.414 1.414L3.05 4.464a1 1 0 010-1.414z" clipRule="evenodd" />
                          </>
                      ) : ( // Icon for "Start Microphone" (Standard Mic)
                          <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8h-1a6 6 0 11-5.445-2.916l.904.904A4.999 4.999 0 1011 14.93V18a1 1 0 102 0v-3.07zM12 3a1 1 0 011 1v4a1 1 0 11-2 0V4a1 1 0 011-1z" clipRule="evenodd" />
                      )}
                  </svg>
                  {isMicOn ? 'Stop Microphone' : 'Start Microphone'}
              </div>
          </button>
        </div>
      </div>
      <RoomAudioRenderer /> {/* Essential for hearing remote audio (like the agent's voice) */}
    </div>
  );
}

// Main AppWrapper component that handles token fetching and LiveKitRoom setup
export default function AppWrapper() {
  const [token, setToken] = useState('');
  const [url, setUrl] = useState('');
  const [loadingMessage, setLoadingMessage] = useState('Initializing application...');

  // Effect to fetch the connection token and server URL from the backend
  useEffect(() => {
    console.log("[AppWrapper:useEffect] Attempting to fetch token...");
    setLoadingMessage('Fetching connection details from server...');
    fetch('http://localhost:8000/token') // Ensure this URL matches your backend endpoint
      .then(res => {
        if (!res.ok) { // Check for HTTP errors
          return res.text().then(text => { // Try to get error message from response body
            throw new Error(`HTTP error! Status: ${res.status}, Body: ${text}`);
          });
        }
        return res.json();
      })
      .then(data => {
        if (data.token && data.url) {
          setToken(data.token);
          setUrl(data.url);
          console.log("[AppWrapper:useEffect] Token and URL fetched successfully.");
          setLoadingMessage('Configuration loaded. Connecting to LiveKit room...');
        } else {
          console.error('[AppWrapper:useEffect] Error: Token or URL missing in server response.', data);
          setLoadingMessage('Error: Connection details (token/URL) missing in server response.');
        }
      })
      .catch(err => {
        console.error('[AppWrapper:useEffect] Error fetching token:', err);
        setLoadingMessage(`Error fetching connection details: ${err.message}. Please ensure the backend server is running and accessible.`);
      });
  }, []); // Empty dependency array means this effect runs once when the component mounts

  // Display loading message or error if token/URL are not yet available
  if (!token || !url) {
    return (
        <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4 text-center">
            <p className="text-xl mb-4 text-red-600 font-semibold">{loadingMessage}</p>
            {loadingMessage.toLowerCase().includes("error") && ( // Show additional help if it's an error
                 <p className="text-sm text-gray-700">
                    Please check if your backend server is running (e.g., on `http://localhost:8000`)
                    and that the `/token` endpoint is correctly configured and accessible.
                 </p>
            )}
        </div>
    );
  }

  // Render the LiveKitRoom provider once token and URL are available
  return (
    <LiveKitRoom
      token={token}
      serverUrl={url}
      connect={true} // Automatically connect to the room
      audio={true}   // Request audio permissions and enable local audio
      video={false}  // Video is not needed for a voice assistant
      // connectOptions={{ autoSubscribe: true }} // autoSubscribe is true by default
      onConnected={() => {
        console.log("[AppWrapper:LiveKitRoomEvent] Successfully connected to LiveKit room.");
        setLoadingMessage('Connected!'); // Update status on successful connection
      }}
      onDisconnected={(reason) => {
        console.log(`[AppWrapper:LiveKitRoomEvent] Disconnected from LiveKit. Reason: ${reason}`);
        setLoadingMessage(`Disconnected: ${reason || 'Connection lost'}. Please refresh or check connection.`);
        // Optionally, clear token/URL to show loading screen again or implement reconnect logic
        // setToken('');
        // setUrl('');
      }}
      onError={(error) => {
        console.error("[AppWrapper:LiveKitRoomEvent] LiveKit connection error:", error);
        setLoadingMessage(`Connection Error: ${error.message}. Check console, network, and LiveKit server status.`);
      }}
    >
      <AppInner /> {/* The AppInner component is rendered within the LiveKitRoom context */}
    </LiveKitRoom>
  );
}
