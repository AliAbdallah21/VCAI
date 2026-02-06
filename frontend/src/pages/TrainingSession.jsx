import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { createWebSocket } from '../services/api';

export default function TrainingSession() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [emotion, setEmotion] = useState({ mood: 0, risk: 'low', tip: null });

  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const messagesEndRef = useRef(null);

  // ══════════════════════════════════════════════════════════════════════
  // STREAMING AUDIO QUEUE
  // ══════════════════════════════════════════════════════════════════════
  const audioContextRef = useRef(null);
  const audioQueueRef = useRef([]);
  const isPlayingRef = useRef(false);
  const hasReceivedChunksRef = useRef(false);
  
  // ══════════════════════════════════════════════════════════════════════
  // PREVENT DUPLICATE CONNECTIONS
  // ══════════════════════════════════════════════════════════════════════
  const isConnectingRef = useRef(false);
  const hasConnectedRef = useRef(false);

  // Get or create AudioContext (reuse across chunks for gapless playback)
  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }
    return audioContextRef.current;
  }, []);

  // Play next chunk from queue - using ref to avoid dependency issues
  const playNextChunkRef = useRef(null);
  playNextChunkRef.current = () => {
    if (audioQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      return;
    }

    isPlayingRef.current = true;
    const chunk = audioQueueRef.current.shift();

    try {
      const ctx = getAudioContext();
      const buffer = ctx.createBuffer(1, chunk.data.length, chunk.sampleRate);
      buffer.getChannelData(0).set(chunk.data);

      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      
      source.onended = () => {
        playNextChunkRef.current();
      };

      source.start();
    } catch (err) {
      console.error('Chunk playback error:', err);
      playNextChunkRef.current();
    }
  };

  // Queue audio chunk - stable function using refs
  const queueAudioChunkRef = useRef(null);
  queueAudioChunkRef.current = (base64Audio, sampleRate) => {
    try {
      const binaryString = atob(base64Audio);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      const float32Array = new Float32Array(bytes.buffer);

      audioQueueRef.current.push({
        data: float32Array,
        sampleRate: sampleRate || 24000
      });

      if (!isPlayingRef.current) {
        playNextChunkRef.current();
      }
    } catch (err) {
      console.error('Audio chunk decode error:', err);
    }
  };

  // Play full audio - stable function using refs
  const playAudioRef = useRef(null);
  playAudioRef.current = async (base64Audio, sampleRate = 24000) => {
    try {
      const binaryString = atob(base64Audio);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      const float32Array = new Float32Array(bytes.buffer);
      const ctx = getAudioContext();
      const audioBuffer = ctx.createBuffer(1, float32Array.length, sampleRate);
      audioBuffer.getChannelData(0).set(float32Array);
      
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      source.start();
    } catch (err) {
      console.error('Audio playback error:', err);
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ══════════════════════════════════════════════════════════════════════
  // WEBSOCKET CONNECTION - with duplicate prevention
  // ══════════════════════════════════════════════════════════════════════
  useEffect(() => {
    // Prevent duplicate connections (React StrictMode calls useEffect twice)
    if (isConnectingRef.current || hasConnectedRef.current) {
      console.log('[WS] Skipping duplicate connection attempt');
      return;
    }
    
    // Check if we already have an open connection
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      console.log('[WS] Already connected, skipping');
      return;
    }

    isConnectingRef.current = true;
    console.log('[WS] Creating new WebSocket connection...');
    
    const ws = createWebSocket(sessionId);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[WS] Connected successfully');
      setConnected(true);
      hasConnectedRef.current = true;
      isConnectingRef.current = false;
    };

    ws.onclose = (event) => {
      console.log('[WS] Connection closed:', event.code, event.reason);
      setConnected(false);
      isConnectingRef.current = false;
      // Only reset hasConnected if this was a normal close (user ended session)
      if (event.code === 1000) {
        hasConnectedRef.current = false;
      }
    };

    ws.onerror = (error) => {
      console.error('[WS] Connection error:', error);
      setConnected(false);
      isConnectingRef.current = false;
    };

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);

      switch (data.type) {
        case 'connected':
          setConnected(true);
          break;

        case 'transcription':
          if (data.data.text && data.data.text.trim() !== '...' && data.data.text.trim() !== '') {
            setMessages(m => [...m, { speaker: 'you', text: data.data.text }]);
          }
          break;

        case 'response':
          setMessages(m => [...m, { speaker: 'customer', text: data.data.text }]);
          break;

        case 'emotion':
          setEmotion({
            mood: data.data.mood_score,
            risk: data.data.risk_level,
            tip: data.data.tip
          });
          break;

        case 'processing':
          if (data.data.status === 'started') {
            setIsProcessing(true);
            hasReceivedChunksRef.current = false;
            audioQueueRef.current = [];
          } else {
            setIsProcessing(false);
          }
          break;

        case 'session_ended':
          navigate(`/evaluation/${sessionId}`);
          break;

        case 'audio_chunk':
          if (data.data.is_final) {
            console.log(`🔊 Streaming complete: ${data.data.total_chunks} chunks`);
          } else {
            hasReceivedChunksRef.current = true;
            queueAudioChunkRef.current(data.data.audio_base64, data.data.sample_rate);
            console.log(`🔊 Chunk ${data.data.chunk_index}: "${data.data.text?.substring(0, 30)}..."`);
          }
          break;

        case 'audio':
          if (!hasReceivedChunksRef.current) {
            playAudioRef.current(data.data.audio_base64, data.data.sample_rate);
          }
          break;
      }
    };

    // Cleanup function
    return () => {
      console.log('[WS] Cleanup - closing connection');
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close(1000, 'Component unmounting');
      }
      // Reset refs on unmount
      isConnectingRef.current = false;
      hasConnectedRef.current = false;
      
      // Clean up AudioContext
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
    };
  }, [sessionId, navigate]); // Removed queueAudioChunk from dependencies!

  const startRecording = async () => {
    try {
      getAudioContext();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        }
      });

      audioChunksRef.current = [];

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const arrayBuffer = await audioBlob.arrayBuffer();
        const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));

        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({
            type: 'audio_complete',
            data: { audio_base64: base64, format: 'webm' }
          }));
        }

        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(100);
      setIsRecording(true);

    } catch (err) {
      console.error('Microphone error:', err);
      alert('Could not access microphone. Please allow microphone access.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
    setIsProcessing(true);
  };

  const endSession = () => {
    wsRef.current?.send(JSON.stringify({ type: 'end_session' }));
  };

  const getMoodColor = () => {
    if (emotion.mood > 20) return 'bg-emerald-500';
    if (emotion.mood < -20) return 'bg-red-500';
    return 'bg-amber-500';
  };

  const getRiskBadge = () => {
    const styles = {
      low: 'bg-emerald-100 text-emerald-700',
      medium: 'bg-amber-100 text-amber-700',
      high: 'bg-red-100 text-red-700',
    };
    return styles[emotion.risk] || styles.low;
  };

  return (
    <div className="min-h-screen bg-slate-100 flex flex-col">
      <header className="bg-white px-6 py-4 flex justify-between items-center shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center text-xl">👤</div>
          <div>
            <p className="font-medium text-slate-800">Training Session</p>
            <p className="text-xs text-slate-500">{connected ? '🟢 Connected' : '🔴 Disconnected'}</p>
          </div>
        </div>
        <button
          onClick={endSession}
          className="px-4 py-2 bg-red-100 text-red-600 rounded-lg hover:bg-red-200 transition font-medium"
        >
          End Session
        </button>
      </header>

      <div className="bg-white border-b px-6 py-4">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-center gap-4 mb-2">
            <span className="text-sm text-slate-500 w-28">Customer Mood</span>
            <div className="flex-1 h-3 bg-slate-200 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${getMoodColor()}`}
                style={{ width: `${50 + emotion.mood / 2}%` }}
              />
            </div>
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskBadge()}`}>
              {emotion.risk.toUpperCase()}
            </span>
          </div>
          {emotion.tip && (
            <div className="mt-2 p-3 bg-amber-50 border border-amber-200 rounded-lg text-amber-800 text-sm">
              💡 {emotion.tip}
            </div>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl mx-auto space-y-4">
          {messages.length === 0 && !isProcessing && (
            <div className="text-center py-12 text-slate-400">
              <div className="text-5xl mb-4">🎤</div>
              <p>Click the microphone button and start speaking</p>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.speaker === 'you' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[75%] p-4 rounded-2xl ${m.speaker === 'you'
                  ? 'bg-blue-600 text-white rounded-br-sm'
                  : 'bg-white shadow-sm rounded-bl-sm'
                }`}>
                <p className="text-xs opacity-70 mb-1">{m.speaker === 'you' ? 'You' : 'Customer'}</p>
                <p>{m.text}</p>
              </div>
            </div>
          ))}

          {isProcessing && (
            <div className="flex justify-start">
              <div className="bg-white shadow-sm p-4 rounded-2xl rounded-bl-sm">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="bg-white border-t p-6">
        <div className="flex flex-col items-center">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={!connected || isProcessing}
            className={`w-20 h-20 rounded-full flex items-center justify-center transition-all shadow-lg ${isRecording
                ? 'bg-red-500 recording-pulse'
                : 'bg-blue-600 hover:bg-blue-700'
              } ${(!connected || isProcessing) ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {isRecording ? (
              <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
            ) : (
              <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
              </svg>
            )}
          </button>
          <p className="mt-3 text-sm text-slate-500">
            {isRecording ? 'Click to stop recording' : isProcessing ? 'Processing...' : 'Click to start speaking'}
          </p>
        </div>
      </div>
    </div>
  );
}