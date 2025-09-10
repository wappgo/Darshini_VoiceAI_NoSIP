


// Minimal App.jsx for testing connection and agent job dispatch
import React, { useState, useEffect } from 'react';
import { RoomEvent } from 'livekit-client';
import { 
    useRoomContext, 
    LiveKitRoom, 
    useParticipants,
    RoomAudioRenderer // Keep this for potential agent audio
} from '@livekit/components-react';
import './index.css'; // Assuming you have this for basic styling

function MinimalAppInner() {
  const room = useRoomContext();
  const participants = useParticipants(); // Hook to get all participants (local and remote)
  const [roomState, setRoomState] = useState(room?.state || 'initial');
  const [participantList, setParticipantList] = useState([]);

  useEffect(() => {
    if (!room) {
      console.log('[MinimalInner] Room context not yet available.');
      return;
    }
    console.log('[MinimalInner] Room context available. Initial state:', room.state);
    setRoomState(room.state); // Set initial state

    const handleStateChange = (newState) => {
      console.log('[MinimalInner] ConnectionStateChanged:', newState);
      setRoomState(newState);
      if (newState === 'connected') {
        console.log('[MinimalInner] SUCCESSFULLY CONNECTED TO ROOM:', room.name, room.sid);
      } else if (newState === 'disconnected' || newState === 'failed') {
        console.log(`[MinimalInner] Disconnected or Failed. State: ${newState}`);
      }
    };

    room.on(RoomEvent.ConnectionStateChanged, handleStateChange);
    handleStateChange(room.state); 

    return () => {
      console.log('[MinimalInner] Cleaning up room state listener.');
      if (room) { // Ensure room exists before calling off
        room.off(RoomEvent.ConnectionStateChanged, handleStateChange);
      }
    };
  }, [room]); 

  useEffect(() => {
    const currentParticipants = participants.map(p => {
      let audioTrackCount = 0;
      if (p.audioTracks && typeof p.audioTracks.size === 'number') { // Check if it's a Map-like object with .size
        audioTrackCount = p.audioTracks.size;
      } else if (p.audioTracks && Array.isArray(p.audioTracks)) { // Fallback if it's an array
        audioTrackCount = p.audioTracks.length;
      } else if (p.audioTracks) { // If it exists but isn't a Map or Array with size/length
        console.warn(`[MinimalInner] Participant ${p.identity} p.audioTracks is an object but not a recognized collection:`, p.audioTracks);
        // Could try Object.keys(p.audioTracks).length if it's a plain object of tracks
      } else {
        console.warn(`[MinimalInner] Participant ${p.identity} p.audioTracks is undefined or null.`);
      }

      return {
        identity: p.identity, 
        sid: p.sid, 
        isLocal: p.isLocal, 
        isAgent: p.isAgent === undefined ? 'N/A' : String(p.isAgent), 
        audioTracks: audioTrackCount 
      };
    });
    setParticipantList(currentParticipants);
    console.log('[MinimalInner] Participants updated:', currentParticipants);
  }, [participants]); 

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>Minimal LiveKit Test</h1>
      <p><strong>Room Connection State:</strong> {roomState}</p>
      <p><strong>Room Name:</strong> {room?.name || 'N/A'}</p>
      <p><strong>Room SID:</strong> {room?.sid || 'N/A'}</p>
      <hr />
      <p><strong>Participants ({participantList.length}):</strong></p>
      {participantList.length === 0 && <p>No participants yet.</p>}
      <ul>
        {participantList.map(p => (
          <li key={p.sid}>
            Identity: {p.identity} (isLocal: {String(p.isLocal)}) (isAgent: {p.isAgent}) (Audio Tracks: {p.audioTracks})
          </li>
        ))}
      </ul>
      <RoomAudioRenderer /> 
    </div>
  );
}

export default function AppWrapper() {
  const [token, setToken] = useState('');
  const [url, setUrl] = useState('');
  const [loadingMessage, setLoadingMessage] = useState('Fetching token...');

  useEffect(() => {
    console.log('[MinimalAppWrapper] Fetching token...');
    setLoadingMessage('Fetching token...');
    fetch('http://localhost:8000/token')
      .then(res => {
        if (!res.ok) {
            return res.text().then(text => {
                throw new Error(`HTTP error! Status: ${res.status}, Body: ${text}`);
            });
        }
        return res.json();
      })
      .then(data => {
        if (data.token && data.url) {
          setToken(data.token);
          setUrl(data.url);
          console.log('[MinimalAppWrapper] Token and URL fetched.');
          setLoadingMessage('Token fetched. Connecting to LiveKit...');
        } else { 
          console.error('[MinimalAppWrapper] Token/URL missing in response:', data);
          setLoadingMessage('Error: Token/URL missing from server.');
        }
      })
      .catch(err => {
        console.error('[MinimalAppWrapper] Token fetch error:', err);
        setLoadingMessage(`Error fetching token: ${err.message}. Is backend running?`);
      });
  }, []);

  if (!token || !url) {
    return (
      <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
        <p>{loadingMessage}</p>
      </div>
    );
  }

  return (
    <LiveKitRoom
      token={token}
      serverUrl={url}
      connect={true} 
      audio={false} 
      video={false}
      onConnected={() => console.log("[MinimalAppWrapper:LiveKitRoomEvent] Event: Connected")}
      onDisconnected={(reason) => console.log("[MinimalAppWrapper:LiveKitRoomEvent] Event: Disconnected. Reason:", reason)}
      onError={(e) => console.error("[MinimalAppWrapper:LiveKitRoomEvent] Event: Error", e)}
      onParticipantConnected={(p) => console.log("[MinimalAppWrapper:LiveKitRoomEvent] Event: Participant Connected", p.identity)}
      onParticipantDisconnected={(p) => console.log("[MinimalAppWrapper:LiveKitRoomEvent] Event: Participant Disconnected", p.identity)}
    >
      <MinimalAppInner />
    </LiveKitRoom>
  );
}