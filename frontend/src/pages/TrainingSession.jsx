import { useState, useEffect, useRef } from 'react';
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

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    const ws = createWebSocket(sessionId);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      
      switch(data.type) {
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
          setIsProcessing(data.data.status === 'started');
          break;
        case 'session_ended':
          navigate('/dashboard');
          break;
        case 'audio':
          playAudio(data.data.audio_base64, data.data.sample_rate);
          break;
      }
    };

    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [sessionId, navigate]);

  const playAudio = async (base64Audio, sampleRate = 22050) => {
    try {
      const binaryString = atob(base64Audio);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
      
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);
      source.start();
    } catch (err) {
      console.log('Audio playback error:', err);
    }
  };

  const startRecording = async () => {
    try {
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
              <div className={`max-w-[75%] p-4 rounded-2xl ${
                m.speaker === 'you' 
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
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
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
            className={`w-20 h-20 rounded-full flex items-center justify-center transition-all shadow-lg ${
              isRecording 
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
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
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
