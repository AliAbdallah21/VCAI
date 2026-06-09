import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { createWebSocket, sessionsAPI, personasAPI } from '../services/api';

export default function TrainingSession() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [connected, setConnected]       = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const [messages, setMessages]         = useState([]);
  const [isRecording, setIsRecording]   = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [emotion, setEmotion]           = useState({ mood: 0, risk: 'low', tip: null });
  const [persona, setPersona]           = useState({ name: '', subtitle: '', avatar: null });
  const [elapsed, setElapsed]           = useState(0); // call timer, seconds

  const wsRef                = useRef(null);
  const mediaRecorderRef     = useRef(null);
  const audioChunksRef       = useRef([]);
  const cancelRecordingRef   = useRef(false); // when true, onstop discards instead of sending
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

  // Replay audio (one shared element so only one plays at a time)
  const replayAudioRef = useRef(null);
  const [playingId, setPlayingId] = useState(null);

  const playReplay = useCallback((id, url) => {
    if (replayAudioRef.current) {
      try { replayAudioRef.current.pause(); } catch {}
      replayAudioRef.current = null;
    }
    if (playingId === id) { setPlayingId(null); return; }
    const el = new Audio(url);
    el.onended = () => { setPlayingId(null); replayAudioRef.current = null; };
    el.onerror = () => { setPlayingId(null); replayAudioRef.current = null; };
    el.play().catch(() => { setPlayingId(null); replayAudioRef.current = null; });
    replayAudioRef.current = el;
    setPlayingId(id);
  }, [playingId]);

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

  // Call timer — counts up once connected, like a live call.
  useEffect(() => {
    if (!connected) return;
    const t = setInterval(() => setElapsed(e => e + 1), 1000);
    return () => clearInterval(t);
  }, [connected]);
  const fmtClock = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

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
            // After a turn completes, refetch the persisted message list so
            // the latest two messages get their real ids + audio_paths
            // (needed for the inline replay buttons).
            sessionsAPI.getMessages(sessionId)
              .then(msgs => {
                if (!msgs?.length) return;
                setMessages(msgs.map(m => ({
                  id: m.id,
                  speaker: m.speaker === 'salesperson' ? 'you' : 'customer',
                  text: m.text,
                  audioPath: m.audio_path,
                })));
              })
              .catch(() => {});
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

    // Load the persona + scenario for the call-screen header card.
    Promise.all([
      sessionsAPI.getById(sessionId).catch(() => null),
      personasAPI.getAll().catch(() => null),
    ]).then(([session, personaList]) => {
      if (!session) return;
      const list = personaList?.personas || personaList || [];
      const p = list.find(x => x.id === session.persona_id);
      const scenario = session.scenario || {};
      setPersona({
        name: p?.name_en || p?.name_ar || session.persona_id || 'Customer',
        subtitle: scenario.label_en || scenario.label || (p?.difficulty ? `${p.difficulty} difficulty` : ''),
        avatar: p?.avatar_url || null,
      });
    });

    // Load existing messages so resumed sessions show previous conversation.
    // Keep id+audioPath so the replay buttons work.
    sessionsAPI.getMessages(sessionId)
      .then(msgs => {
        if (msgs?.length > 0) {
          setMessages(msgs.map(m => ({
            id: m.id,
            speaker: m.speaker === 'salesperson' ? 'you' : 'customer',
            text: m.text,
            audioPath: m.audio_path,
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
      if (replayAudioRef.current) {
        try { replayAudioRef.current.pause(); } catch {}
        replayAudioRef.current = null;
      }
    };
  }, [sessionId]);

  const startRecording = async () => {
    try {
      getAudioContext();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      audioChunksRef.current = [];
      cancelRecordingRef.current = false;
      const mr = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      mr.ondataavailable = e => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      mr.onstop = () => {
        // User discarded this take — drop the audio, don't send, don't lock UI.
        if (cancelRecordingRef.current) {
          cancelRecordingRef.current = false;
          audioChunksRef.current = [];
          stream.getTracks().forEach(t => t.stop());
          setIsProcessing(false);
          clearProcessingWatchdog();
          return;
        }

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

  // Discard the in-progress take without sending it (for stutters / mistakes).
  const cancelRecording = () => {
    cancelRecordingRef.current = true;
    if (mediaRecorderRef.current?.state !== 'inactive') mediaRecorderRef.current.stop();
    setIsRecording(false);
    // onstop handles cleanup; never enters processing.
  };

  // Replay the customer's most recent audio reply (Speaker button).
  const replayLastCustomer = () => {
    const last = [...messages].reverse().find(m => m.speaker === 'customer' && m.id && m.audioPath);
    if (last) playReplay(`msg-${last.id}`, sessionsAPI.messageAudioUrl(sessionId, last.id));
  };
  const hasCustomerAudio = messages.some(m => m.speaker === 'customer' && m.id && m.audioPath);

  const endSession = () => wsRef.current?.send(JSON.stringify({ type: 'end_session' }));


  // Human-readable emotional state for the call screen (mood + risk).
  const emotionLabel = (() => {
    const m = emotion.mood;
    const tone = m > 35 ? 'Happy' : m > 10 ? 'Interested' : m < -35 ? 'Frustrated' : m < -10 ? 'Skeptical' : 'Calm';
    const calm = emotion.risk === 'high' ? 'Tense' : emotion.risk === 'medium' ? 'Guarded' : 'Calm';
    return `${calm} / ${tone}`;
  })();

  return (
    <div
      className="min-h-screen flex flex-col relative overflow-hidden"
      style={{ background: 'radial-gradient(ellipse 90% 70% at 50% 45%, #15233f 0%, #0b1220 70%)' }}
    >
      {/* Top bar: call status + timer */}
      <div className="flex justify-between items-start px-6 md:px-10 pt-6 md:pt-8 flex-shrink-0">
        <div>
          <div className="flex items-center gap-2">
            <span
              className="w-2.5 h-2.5 rounded-full"
              style={{ background: connected ? '#ef4444' : '#f59e0b', boxShadow: connected ? '0 0 8px #ef4444' : 'none' }}
            />
            <p className="font-bold text-base md:text-lg" style={{ color: '#e8edf5' }}>
              {connected ? 'Call in progress' : reconnecting ? 'Reconnecting…' : 'Disconnected'}
            </p>
          </div>
          <p className="text-sm mt-0.5" style={{ color: '#6b7a93' }}>Secure connection</p>
        </div>
        <p
          className="font-bold tracking-[0.15em] tabular-nums"
          style={{ fontSize: 'clamp(28px, 5vw, 44px)', color: '#c9d6e8', fontVariantNumeric: 'tabular-nums' }}
        >
          {fmtClock(elapsed)}
        </p>
      </div>

      {/* Recovery banner (kept — surfaces short-recording / busy / error nudges) */}
      {stuckRecovery && (
        <div className="px-6 mt-3 flex-shrink-0">
          <div
            className="mx-auto max-w-md px-4 py-2.5 rounded-xl text-center text-xs font-medium"
            style={{ background: 'rgba(255,180,171,0.1)', border: '1px solid rgba(255,180,171,0.25)', color: '#ffb4ab' }}
          >
            {stuckRecovery}
          </div>
        </div>
      )}

      {/* Center: emotional state + mic with vibrance + persona card */}
      <div className="flex-1 flex flex-col items-center justify-center px-6">
        <p
          className="text-xs font-semibold tracking-[0.18em] uppercase mb-2"
          style={{ color: '#b08d57' }}
        >
          Customer Emotional State
        </p>
        <p className="font-bold mb-10" style={{ fontSize: 'clamp(26px, 4vw, 38px)', color: '#f1f5fb' }}>
          {emotionLabel}
        </p>

        {/* Mic zone — dashed frame + pulsing vibrance while recording */}
        <div className="relative flex items-center justify-center" style={{ width: 260, height: 260 }}>
          <div
            className="absolute inset-0 rounded-3xl"
            style={{ border: '1px dashed rgba(255,255,255,0.12)' }}
          />

          {/* vibrance rings — only while the agent is speaking (recording) */}
          {isRecording && (
            <>
              <span className="vibrance-ring" style={{ animationDelay: '0s' }} />
              <span className="vibrance-ring" style={{ animationDelay: '0.6s' }} />
              <span className="vibrance-ring" style={{ animationDelay: '1.2s' }} />
            </>
          )}

          {/* mic / stop / processing button */}
          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={!connected || isProcessing}
            className="relative flex items-center justify-center transition-all duration-300"
            style={{
              width: 132, height: 132, borderRadius: 26,
              cursor: !connected || isProcessing ? 'not-allowed' : 'pointer',
              background: isRecording
                ? 'linear-gradient(135deg, rgba(80,130,220,0.9), rgba(120,170,250,0.8))'
                : 'rgba(40,58,92,0.55)',
              border: '1px solid rgba(255,255,255,0.08)',
              boxShadow: isRecording
                ? '0 0 40px rgba(90,150,250,0.5), 0 0 90px rgba(90,150,250,0.25)'
                : '0 8px 40px rgba(0,0,0,0.4)',
              opacity: !connected || isProcessing ? 0.4 : 1,
            }}
          >
            {isProcessing ? (
              <div className="flex items-center gap-1.5">
                {[0, 1, 2].map(n => (
                  <span key={n} className="w-2 h-2 rounded-full animate-bounce"
                    style={{ background: '#9db8e8', animationDelay: `${n * 0.12}s` }} />
                ))}
              </div>
            ) : isRecording ? (
              <svg width="30" height="30" fill="#fff" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="3" /></svg>
            ) : (
              <svg width="34" height="34" fill="none" stroke="#aebfdc" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
                <path d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
              </svg>
            )}
          </button>

          {/* persona card — floating, overlaps the mic frame */}
          {persona.name && (
            <div
              className="absolute flex items-center gap-3 px-4 py-2.5 rounded-2xl"
              style={{
                bottom: 26, left: -34,
                background: 'rgba(20,32,56,0.72)',
                border: '1px solid rgba(255,255,255,0.08)',
                backdropFilter: 'blur(12px)',
                boxShadow: '0 10px 30px rgba(0,0,0,0.35)',
              }}
            >
              {persona.avatar ? (
                <img src={persona.avatar} alt="" className="w-10 h-10 rounded-xl object-cover flex-shrink-0"
                  style={{ border: '1px solid rgba(255,255,255,0.1)' }} />
              ) : (
                <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                  style={{ background: 'rgba(120,150,210,0.2)' }}>
                  <svg width="18" height="18" fill="none" stroke="#9db8e8" strokeWidth="1.6" viewBox="0 0 24 24">
                    <path d="M15 9.75a3 3 0 11-6 0 3 3 0 016 0zM18 18.7A7.5 7.5 0 0012 15.75 7.5 7.5 0 006 18.7" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
              )}
              <div className="min-w-0">
                <p className="font-bold text-sm leading-tight" style={{ color: '#f1f5fb' }}>{persona.name}</p>
                {persona.subtitle && (
                  <p className="text-xs leading-tight mt-0.5" style={{ color: '#7e8ca4' }}>{persona.subtitle}</p>
                )}
              </div>
            </div>
          )}
        </div>

        {/* mic hint */}
        <p className="mt-8 text-sm font-medium" style={{ color: '#6b7a93' }}>
          {isRecording ? 'Recording — tap to send, or cancel to discard'
            : isProcessing ? 'Processing…'
            : !connected ? (reconnecting ? 'Reconnecting…' : 'Disconnected')
            : 'Tap the mic and speak'}
        </p>
      </div>

      {/* Bottom controls — Cancel · End (red) · Speaker */}
      <div className="flex-shrink-0 pb-10 md:pb-14 flex items-end justify-center gap-8 md:gap-10">
        {/* Cancel / discard current take */}
        <div className="flex flex-col items-center gap-2">
          <button
            onClick={cancelRecording}
            disabled={!isRecording}
            className="flex items-center justify-center rounded-2xl transition-all duration-200"
            style={{
              width: 60, height: 60,
              background: isRecording ? 'rgba(40,58,92,0.6)' : 'rgba(40,58,92,0.25)',
              border: '1px solid rgba(255,255,255,0.08)',
              cursor: isRecording ? 'pointer' : 'not-allowed',
              opacity: isRecording ? 1 : 0.4,
            }}
            title="Discard this recording without sending"
          >
            <svg width="22" height="22" fill="none" stroke="#c9d6e8" strokeWidth="1.8" strokeLinecap="round" viewBox="0 0 24 24">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
          <span className="text-xs" style={{ color: '#6b7a93' }}>Cancel</span>
        </div>

        {/* End session */}
        <div className="flex flex-col items-center gap-2">
          <button
            onClick={endSession}
            className="flex items-center justify-center rounded-2xl transition-all duration-200"
            style={{ width: 66, height: 66, background: '#e23b3b', boxShadow: '0 8px 28px rgba(226,59,59,0.45)' }}
            title="End the call & evaluate"
          >
            <svg width="26" height="26" fill="#fff" viewBox="0 0 24 24" style={{ transform: 'rotate(135deg)' }}>
              <path d="M21 15.46l-5.27-.61-2.52 2.52a15.05 15.05 0 01-6.59-6.59l2.53-2.53L8.54 3H3.03C2.45 13.18 10.82 21.55 21 20.97v-5.51z" />
            </svg>
          </button>
          <span className="text-xs" style={{ color: '#6b7a93' }}>End</span>
        </div>

        {/* Speaker — replay customer's last reply */}
        <div className="flex flex-col items-center gap-2">
          <button
            onClick={replayLastCustomer}
            disabled={!hasCustomerAudio}
            className="flex items-center justify-center rounded-2xl transition-all duration-200"
            style={{
              width: 60, height: 60,
              background: playingId ? 'rgba(90,150,250,0.25)' : 'rgba(40,58,92,0.6)',
              border: '1px solid rgba(255,255,255,0.08)',
              cursor: hasCustomerAudio ? 'pointer' : 'not-allowed',
              opacity: hasCustomerAudio ? 1 : 0.4,
            }}
            title="Replay the customer's last reply"
          >
            <svg width="22" height="22" fill="#c9d6e8" viewBox="0 0 24 24">
              <path d="M3 10v4h4l5 5V5L7 10H3zm13.5 2a4.5 4.5 0 00-2.5-4.03v8.06A4.5 4.5 0 0016.5 12z"/>
            </svg>
          </button>
          <span className="text-xs" style={{ color: '#6b7a93' }}>Speaker</span>
        </div>
      </div>

      <div ref={messagesEndRef} className="hidden" />
    </div>
  );
}
