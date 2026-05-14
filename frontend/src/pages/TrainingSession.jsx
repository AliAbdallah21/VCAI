import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { createWebSocket, sessionsAPI } from '../services/api';

const MicIcon = () => (
  <svg width="28" height="28" fill="none" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
    <path d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
  </svg>
);

const StopIcon = () => (
  <svg width="26" height="26" fill="white" viewBox="0 0 24 24">
    <rect x="6" y="6" width="12" height="12" rx="2.5" />
  </svg>
);

export default function TrainingSession() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [connected, setConnected]       = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const [messages, setMessages]         = useState([]);
  const [isRecording, setIsRecording]   = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [emotion, setEmotion]           = useState({ mood: 0, risk: 'low', tip: null });

  const wsRef                = useRef(null);
  const mediaRecorderRef     = useRef(null);
  const audioChunksRef       = useRef([]);
  const messagesEndRef       = useRef(null);
  const audioContextRef      = useRef(null);
  const audioQueueRef        = useRef([]);
  const isPlayingRef         = useRef(false);
  const hasReceivedChunksRef = useRef(false);
  const isConnectingRef      = useRef(false);
  const reconnectAttemptsRef = useRef(0);
  const unmountedRef         = useRef(false);
  const processingWatchdogRef = useRef(null);
  const [stuckRecovery, setStuckRecovery] = useState(null);

  // Watchdog: if processing doesn't complete within 60s, force-unlock the UI.
  // Prevents the user from being permanently locked out when the server drops
  // the processing:completed message for any reason.
  const PROCESSING_TIMEOUT_MS = 60000;
  const armProcessingWatchdog = () => {
    if (processingWatchdogRef.current) clearTimeout(processingWatchdogRef.current);
    processingWatchdogRef.current = setTimeout(() => {
      setIsProcessing(false);
      setStuckRecovery('Server took too long to respond — you can record again.');
      setTimeout(() => setStuckRecovery(null), 5000);
    }, PROCESSING_TIMEOUT_MS);
  };
  const clearProcessingWatchdog = () => {
    if (processingWatchdogRef.current) {
      clearTimeout(processingWatchdogRef.current);
      processingWatchdogRef.current = null;
    }
  };

  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContextRef.current.state === 'suspended') audioContextRef.current.resume();
    return audioContextRef.current;
  }, []);

  const playNextChunkRef = useRef(null);
  playNextChunkRef.current = () => {
    if (audioQueueRef.current.length === 0) { isPlayingRef.current = false; return; }
    isPlayingRef.current = true;
    const chunk = audioQueueRef.current.shift();
    try {
      const ctx = getAudioContext();
      const buffer = ctx.createBuffer(1, chunk.data.length, chunk.sampleRate);
      buffer.getChannelData(0).set(chunk.data);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.onended = () => playNextChunkRef.current();
      source.start();
    } catch { playNextChunkRef.current(); }
  };

  const queueAudioChunkRef = useRef(null);
  queueAudioChunkRef.current = (base64Audio, sampleRate) => {
    try {
      const bytes = new Uint8Array(atob(base64Audio).split('').map(c => c.charCodeAt(0)));
      audioQueueRef.current.push({ data: new Float32Array(bytes.buffer), sampleRate: sampleRate || 24000 });
      if (!isPlayingRef.current) playNextChunkRef.current();
    } catch {}
  };

  const playAudioRef = useRef(null);
  playAudioRef.current = async (base64Audio, sampleRate = 24000) => {
    try {
      const bytes = new Uint8Array(atob(base64Audio).split('').map(c => c.charCodeAt(0)));
      const float32 = new Float32Array(bytes.buffer);
      const ctx = getAudioContext();
      const buf = ctx.createBuffer(1, float32.length, sampleRate);
      buf.getChannelData(0).set(float32);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      src.start();
    } catch {}
  };

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  // Client-side ping every 15s — prevents browser/proxy from closing the WebSocket
  // during long LLM generation (which can take 10-35s with no data flowing)
  useEffect(() => {
    if (!connected) return;
    const id = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 15000);
    return () => clearInterval(id);
  }, [connected]);

  // Stable connect function stored in a ref so the timeout callback always
  // sees the latest version without stale closure issues.
  const doConnectRef = useRef(null);
  doConnectRef.current = () => {
    if (isConnectingRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    isConnectingRef.current = true;

    const ws = createWebSocket(sessionId);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setReconnecting(false);
      reconnectAttemptsRef.current = 0;
      isConnectingRef.current = false;
    };

    ws.onclose = (e) => {
      setConnected(false);
      isConnectingRef.current = false;
      if (e.code !== 1000 && reconnectAttemptsRef.current < 3 && !unmountedRef.current) {
        reconnectAttemptsRef.current += 1;
        setReconnecting(true);
        setTimeout(() => doConnectRef.current?.(), 2000);
      } else {
        setReconnecting(false);
      }
    };

    ws.onerror = () => {
      setConnected(false);
      isConnectingRef.current = false;
    };

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      switch (data.type) {
        case 'connected':
          setConnected(true);
          setIsProcessing(false);
          clearProcessingWatchdog();
          break;
        case 'transcription':
          if (data.data.text?.trim() && data.data.text.trim() !== '...') {
            setMessages(m => [...m, { speaker: 'you', text: data.data.text }]);
          }
          break;
        case 'response':
          setMessages(m => [...m, { speaker: 'customer', text: data.data.text }]);
          break;
        case 'emotion':
          setEmotion({ mood: data.data.mood_score, risk: data.data.risk_level, tip: data.data.tip });
          break;
        case 'processing':
          if (data.data.status === 'started') {
            setIsProcessing(true);
            hasReceivedChunksRef.current = false;
            audioQueueRef.current = [];
            armProcessingWatchdog();
          } else {
            setIsProcessing(false);
            clearProcessingWatchdog();
          }
          break;
        case 'info':
          // Server says it can't process right now (e.g. "busy"). Unlock the UI
          // so the user isn't silently stuck.
          if (data.data?.message === 'busy') {
            setIsProcessing(false);
            clearProcessingWatchdog();
            setStuckRecovery('Server is busy — try again in a moment.');
            setTimeout(() => setStuckRecovery(null), 4000);
          }
          break;
        case 'error':
          setIsProcessing(false);
          clearProcessingWatchdog();
          setStuckRecovery(data.data?.message || 'Something went wrong — try again.');
          setTimeout(() => setStuckRecovery(null), 5000);
          break;
        case 'session_ended':
          clearProcessingWatchdog();
          navigate(`/evaluation/${sessionId}`);
          break;
        case 'audio_chunk':
          if (!data.data.is_final) {
            hasReceivedChunksRef.current = true;
            queueAudioChunkRef.current(data.data.audio_base64, data.data.sample_rate);
          }
          break;
        case 'audio':
          if (!hasReceivedChunksRef.current) playAudioRef.current(data.data.audio_base64, data.data.sample_rate);
          break;
        // server keepalive ping — no reply needed, connection stays alive
        case 'ping': break;
      }
    };
  };

  useEffect(() => {
    unmountedRef.current = false;
    reconnectAttemptsRef.current = 0;

    // Load existing messages so resumed sessions show previous conversation
    sessionsAPI.getMessages(sessionId)
      .then(msgs => {
        if (msgs?.length > 0) {
          setMessages(msgs.map(m => ({
            speaker: m.speaker === 'salesperson' ? 'you' : 'customer',
            text: m.text,
          })));
        }
      })
      .catch(() => {});

    doConnectRef.current();

    return () => {
      unmountedRef.current = true;
      isConnectingRef.current = false;
      clearProcessingWatchdog();
      if (wsRef.current?.readyState <= WebSocket.OPEN) wsRef.current.close(1000, 'Component unmounting');
      if (audioContextRef.current?.state !== 'closed') audioContextRef.current?.close();
    };
  }, [sessionId]);

  const startRecording = async () => {
    try {
      getAudioContext();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      audioChunksRef.current = [];
      const mr = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      mr.ondataavailable = e => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      mr.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });

        // Guard accidental quick taps — opus webm of < ~0.4s of speech is
        // typically under 3KB. Lets "أوك" / "تمام" / "لا" through, blocks
        // 0.1s clicks. Don't send, don't lock the UI, just nudge the user.
        if (blob.size < 3000) {
          stream.getTracks().forEach(t => t.stop());
          setIsProcessing(false);
          clearProcessingWatchdog();
          setStuckRecovery('التسجيل قصير جداً — اضغط واتكلم لثانية على الأقل.');
          setTimeout(() => setStuckRecovery(null), 3000);
          return;
        }

        // Use FileReader for base64 encoding — spreading a 60KB Uint8Array
        // into String.fromCharCode(...bytes) overflows the JS call stack on
        // longer recordings. FileReader handles arbitrary sizes natively.
        const reader = new FileReader();
        reader.onloadend = () => {
          try {
            // result is "data:audio/webm;base64,XXXX" — strip the prefix
            const base64 = String(reader.result).split(',')[1] || '';
            if (wsRef.current?.readyState === WebSocket.OPEN && base64) {
              wsRef.current.send(JSON.stringify({
                type: 'audio_complete',
                data: { audio_base64: base64, format: 'webm' },
              }));
            }
          } finally {
            stream.getTracks().forEach(t => t.stop());
          }
        };
        reader.onerror = () => {
          stream.getTracks().forEach(t => t.stop());
          setIsProcessing(false);
          clearProcessingWatchdog();
          setStuckRecovery('Failed to encode audio — try again.');
          setTimeout(() => setStuckRecovery(null), 4000);
        };
        reader.readAsDataURL(blob);
      };
      mediaRecorderRef.current = mr;
      mr.start(100);
      setIsRecording(true);
    } catch {
      alert('Could not access microphone. Please allow microphone access.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state !== 'inactive') mediaRecorderRef.current.stop();
    setIsRecording(false);
    setIsProcessing(true);
    armProcessingWatchdog();
  };

  const endSession = () => wsRef.current?.send(JSON.stringify({ type: 'end_session' }));

  const moodPercent = 50 + emotion.mood / 2;
  const moodColor   = emotion.mood > 20 ? '#10b981' : emotion.mood < -20 ? '#ef4444' : '#f59e0b';
  const riskStyle   = {
    low:    { bg: 'rgba(16,185,129,0.1)',  text: '#34d399', border: 'rgba(16,185,129,0.2)'  },
    medium: { bg: 'rgba(245,158,11,0.1)', text: '#fbbf24', border: 'rgba(245,158,11,0.2)' },
    high:   { bg: 'rgba(239,68,68,0.1)',   text: '#f87171', border: 'rgba(239,68,68,0.2)'   },
  }[emotion.risk] || { bg: 'rgba(16,185,129,0.1)', text: '#34d399', border: 'rgba(16,185,129,0.2)' };

  const statusDotClass = connected
    ? 'bg-emerald-400'
    : reconnecting
    ? 'bg-amber-400 animate-pulse'
    : 'bg-red-400';

  const statusText = connected
    ? 'Connected'
    : reconnecting
    ? `Reconnecting (${reconnectAttemptsRef.current}/3)…`
    : 'Disconnected';

  return (
    <div className="min-h-screen flex flex-col" style={{ background: '#030712' }}>
      {/* Header */}
      <header
        className="flex justify-between items-center px-6 py-4 flex-shrink-0"
        style={{ background: 'rgba(8,14,28,0.95)', borderBottom: '1px solid rgba(255,255,255,0.05)', backdropFilter: 'blur(20px)' }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-9 h-9 rounded-xl flex items-center justify-center"
            style={{ background: 'rgba(37,99,235,0.15)', border: '1px solid rgba(37,99,235,0.2)' }}
          >
            <svg width="16" height="16" fill="none" stroke="#60a5fa" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
              <path d="M17.982 18.725A7.488 7.488 0 0012 15.75a7.488 7.488 0 00-5.982 2.975m11.963 0a9 9 0 10-11.963 0m11.963 0A8.966 8.966 0 0112 21a8.966 8.966 0 01-5.982-2.275M15 9.75a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </div>
          <div>
            <p className="font-semibold text-white text-sm">Training Session</p>
            <div className="flex items-center gap-1.5 mt-0.5">
              <div
                className={`w-1.5 h-1.5 rounded-full ${statusDotClass}`}
                style={connected ? { boxShadow: '0 0 6px #34d399' } : {}}
              />
              <p className="text-xs" style={{ color: 'rgba(148,163,184,0.5)' }}>{statusText}</p>
            </div>
          </div>
        </div>
        <button
          onClick={endSession}
          className="px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200"
          style={{ background: 'rgba(239,68,68,0.08)', color: '#f87171', border: '1px solid rgba(239,68,68,0.15)' }}
        >
          End Session
        </button>
      </header>

      {/* Reconnecting banner */}
      {reconnecting && (
        <div
          className="px-6 py-2.5 text-center text-xs font-medium"
          style={{ background: 'rgba(245,158,11,0.08)', borderBottom: '1px solid rgba(245,158,11,0.15)', color: '#fbbf24' }}
        >
          Connection lost — reconnecting automatically… ({reconnectAttemptsRef.current}/3)
        </div>
      )}

      {/* Stuck/error recovery banner */}
      {stuckRecovery && !reconnecting && (
        <div
          className="px-6 py-2.5 text-center text-xs font-medium"
          style={{ background: 'rgba(239,68,68,0.08)', borderBottom: '1px solid rgba(239,68,68,0.15)', color: '#f87171' }}
        >
          {stuckRecovery}
        </div>
      )}

      {/* Emotion Bar */}
      <div
        className="px-6 py-4 flex-shrink-0"
        style={{ background: 'rgba(8,14,28,0.8)', borderBottom: '1px solid rgba(255,255,255,0.04)' }}
      >
        <div className="max-w-2xl mx-auto">
          <div className="flex items-center gap-4">
            <span className="text-xs font-medium w-28 flex-shrink-0" style={{ color: 'rgba(148,163,184,0.5)' }}>
              Customer Mood
            </span>
            <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{ width: `${moodPercent}%`, background: moodColor, boxShadow: `0 0 8px ${moodColor}60` }}
              />
            </div>
            <span
              className="px-2.5 py-1 rounded-lg text-xs font-semibold uppercase tracking-wide flex-shrink-0"
              style={{ background: riskStyle.bg, color: riskStyle.text, border: `1px solid ${riskStyle.border}` }}
            >
              {emotion.risk}
            </span>
          </div>
          {emotion.tip && (
            <div
              className="mt-3 px-4 py-2.5 rounded-xl text-sm"
              style={{
                background: 'rgba(245,158,11,0.07)',
                border: '1px solid rgba(245,158,11,0.15)',
                color: '#fcd34d',
              }}
            >
              <span style={{ opacity: 0.6 }}>Tip · </span>{emotion.tip}
            </div>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="max-w-2xl mx-auto space-y-4">
          {messages.length === 0 && !isProcessing && (
            <div className="text-center py-16">
              <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-5 mic-glow"
                style={{ background: 'rgba(37,99,235,0.12)', border: '1px solid rgba(37,99,235,0.2)' }}
              >
                <MicIcon />
              </div>
              <p className="font-medium text-white mb-1">Ready to practice</p>
              <p className="text-sm" style={{ color: 'rgba(148,163,184,0.45)' }}>
                Press the button below and start speaking
              </p>
            </div>
          )}

          {messages.map((m, i) => {
            const isYou = m.speaker === 'you';
            return (
              <div key={i} className={`flex ${isYou ? 'justify-end' : 'justify-start'} slide-up`}>
                <div
                  className="max-w-[76%] px-4 py-3 rounded-2xl text-sm leading-relaxed"
                  style={isYou ? {
                    background: 'linear-gradient(135deg, rgba(37,99,235,0.5), rgba(124,58,237,0.4))',
                    border: '1px solid rgba(37,99,235,0.3)',
                    color: '#e0e7ff',
                    borderBottomRightRadius: '6px',
                  } : {
                    background: 'rgba(20,30,55,0.9)',
                    border: '1px solid rgba(255,255,255,0.07)',
                    color: '#cbd5e1',
                    borderBottomLeftRadius: '6px',
                  }}
                >
                  <p className="text-xs mb-1.5 font-medium" style={{ opacity: 0.5 }}>
                    {isYou ? 'You' : 'Customer'}
                  </p>
                  {m.text}
                </div>
              </div>
            );
          })}

          {isProcessing && (
            <div className="flex justify-start">
              <div
                className="px-4 py-3 rounded-2xl"
                style={{
                  background: 'rgba(20,30,55,0.9)',
                  border: '1px solid rgba(255,255,255,0.07)',
                  borderBottomLeftRadius: '6px',
                }}
              >
                <p className="text-xs mb-2 font-medium" style={{ opacity: 0.4, color: '#94a3b8' }}>Customer</p>
                <div className="flex items-center gap-1.5">
                  {[0, 1, 2].map(n => (
                    <div
                      key={n}
                      className="w-2 h-2 rounded-full animate-bounce"
                      style={{ background: '#60a5fa', animationDelay: `${n * 0.12}s` }}
                    />
                  ))}
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Controls */}
      <div
        className="flex-shrink-0 px-6 py-6"
        style={{ background: 'rgba(8,14,28,0.95)', borderTop: '1px solid rgba(255,255,255,0.05)' }}
      >
        <div className="flex flex-col items-center">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={!connected || isProcessing}
            className={`w-18 h-18 rounded-full flex items-center justify-center transition-all duration-300 ${
              !connected || isProcessing ? 'opacity-30 cursor-not-allowed' : ''
            }`}
            style={{
              width: 72, height: 72,
              ...(isRecording ? {
                background: '#dc2626',
                boxShadow: '0 0 0 0 rgba(239,68,68,0.4)',
              } : {
                background: 'linear-gradient(135deg, #2563eb, #7c3aed)',
                ...(connected && !isProcessing ? {
                  boxShadow: '0 0 20px rgba(59,130,246,0.35), 0 0 60px rgba(124,58,237,0.15)',
                } : {}),
              }),
            }}
          >
            {isRecording ? <StopIcon /> : <MicIcon />}
          </button>
          <p className="mt-3 text-xs font-medium" style={{ color: 'rgba(148,163,184,0.45)' }}>
            {isRecording ? 'Tap to stop' : isProcessing ? 'Processing…' : !connected ? (reconnecting ? 'Reconnecting…' : 'Disconnected') : 'Tap to speak'}
          </p>
        </div>
      </div>
    </div>
  );
}
