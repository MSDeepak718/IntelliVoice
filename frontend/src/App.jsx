import { useState, useCallback, useEffect, useRef } from 'react';
import styles from './App.module.css';

import Header from './components/Header';
import Visualizer from './components/Visualizer';
import MicButton from './components/MicButton';
import Conversation from './components/Conversation';
import Sidebar from './components/Sidebar';

import useWebSocket from './hooks/useWebSocket';
import useAudioCapture from './hooks/useAudioCapture';
import useAudioPlayback from './hooks/useAudioPlayback';

const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.hostname || 'localhost'}:8000/ws/audio`;

function formatDuration(startTime) {
  if (!startTime) return '00:00';
  const elapsed = Math.floor((Date.now() - startTime) / 1000);
  const m = Math.floor(elapsed / 60).toString().padStart(2, '0');
  const s = (elapsed % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

function capitalize(s) {
  if (!s) return '';
  return s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
}

export default function App() {
  // --- State ---
  const [messages, setMessages] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [autoPlayback, setAutoPlayback] = useState(true);
  const [language, setLanguage] = useState(null);
  const [emotion, setEmotion] = useState(null);
  const [activeLayers, setActiveLayers] = useState({});
  const [timings, setTimings] = useState({});
  const [duration, setDuration] = useState('00:00');
  const sessionStartRef = useRef(null);
  const durationTimerRef = useRef(null);

  // --- Hooks ---
  const ws = useWebSocket(WS_URL);
  const playback = useAudioPlayback(22050);

  const audioCapture = useAudioCapture({
    sampleRate: 16000,
    onAudioData: useCallback((buffer) => {
      ws.sendAudio(buffer);
    }, [ws]),
  });

  // --- Duration timer ---
  useEffect(() => {
    if (ws.sessionId && !sessionStartRef.current) {
      sessionStartRef.current = Date.now();
      durationTimerRef.current = setInterval(() => {
        setDuration(formatDuration(sessionStartRef.current));
      }, 1000);
    }
    return () => {
      if (durationTimerRef.current) clearInterval(durationTimerRef.current);
    };
  }, [ws.sessionId]);

  // --- Message helpers ---
  const addMessage = useCallback((role, text, meta = {}) => {
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    setMessages(prev => [...prev, { role, text, time, emotion: meta.emotion }]);
  }, []);

  // --- WebSocket message handlers ---
  useEffect(() => {
    ws.setHandler('vad', (msg) => {
      if (msg.status === 'speech_start') {
        setActiveLayers(prev => ({ ...prev, vad: 'running' }));
      } else if (msg.status === 'speech_end') {
        setActiveLayers(prev => ({ ...prev, vad: 'done' }));
      }
    });

    ws.setHandler('processing', (msg) => {
      if (msg.status === 'started') {
        setIsProcessing(true);
        setActiveLayers(prev => ({
          ...prev,
          encoder: 'running',
          semantic: 'running',
          emotion: 'running',
        }));
      } else if (msg.status === 'complete') {
        setIsProcessing(false);
        setActiveLayers({});
      }
    });

    ws.setHandler('transcription', (msg) => {
      addMessage('user', msg.text, { emotion: msg.emotion });
      if (msg.language) setLanguage(capitalize(msg.language));
      if (msg.emotion) setEmotion(capitalize(msg.emotion));
      setActiveLayers(prev => ({
        ...prev,
        encoder: 'done',
        semantic: 'done',
        emotion: 'done',
        reasoning: 'running',
      }));
    });

    ws.setHandler('response', (msg) => {
      addMessage('assistant', msg.text);
      if (msg.metadata) {
        setTimings(msg.metadata);
      }
      setActiveLayers(prev => ({
        ...prev,
        reasoning: 'done',
        tts: 'running',
      }));
    });

    ws.setHandler('audio_response', (msg) => {
      if (autoPlayback) {
        playback.playBase64(msg.audio, msg.sample_rate);
      }
      setActiveLayers(prev => ({ ...prev, tts: 'done' }));
    });

    ws.setHandler('reset', () => {
      setMessages([]);
      setLanguage(null);
      setEmotion(null);
      setTimings({});
      setActiveLayers({});
      setIsProcessing(false);
    });

    ws.setHandler('error', (msg) => {
      addMessage('assistant', msg.message || 'An error occurred');
    });
  }, [ws, addMessage, autoPlayback, playback]);

  // --- Recording toggle ---
  const toggleRecording = useCallback(async () => {
    if (isProcessing) return;

    if (audioCapture.isRecording) {
      audioCapture.stop();
    } else {
      if (!ws.isConnected) {
        ws.connect();
        await new Promise(r => setTimeout(r, 500));
      }
      await audioCapture.start();
    }
  }, [audioCapture, ws, isProcessing]);

  // --- Send text ---
  const handleSendText = useCallback((text) => {
    if (!ws.isConnected) {
      ws.connect();
    }
    addMessage('user', text);
    setIsProcessing(true);
    ws.sendText(text);
  }, [ws, addMessage]);

  // --- Reset ---
  const handleReset = useCallback(() => {
    if (audioCapture.isRecording) audioCapture.stop();
    ws.sendReset();
    setMessages([]);
    setLanguage(null);
    setEmotion(null);
    setTimings({});
    setActiveLayers({});
    setIsProcessing(false);
  }, [audioCapture, ws]);

  // --- Keyboard shortcut (Space) ---
  useEffect(() => {
    const handler = (e) => {
      if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
        e.preventDefault();
        toggleRecording();
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [toggleRecording]);

  // --- Auto-connect ---
  useEffect(() => {
    ws.connect();
  }, [ws]);

  // --- Mute sync ---
  useEffect(() => {
    playback.setMuted(isMuted);
  }, [isMuted, playback]);

  return (
    <>
      <div className={styles.bgGrid} />
      <div className={styles.orbOne} />
      <div className={styles.orbTwo} />

      <div className={styles.container}>
        <Header connectionStatus={ws.status} onReset={handleReset} />

        <main className={styles.main}>
          <div className={styles.voicePanel}>
            <Visualizer
              analyserNode={audioCapture.getAnalyserNode()}
              isActive={audioCapture.isRecording}
            />

            <MicButton
              isRecording={audioCapture.isRecording}
              isProcessing={isProcessing}
              onClick={toggleRecording}
              onMuteToggle={() => setIsMuted(!isMuted)}
              isMuted={isMuted}
              autoPlayback={autoPlayback}
              onPlaybackToggle={() => setAutoPlayback(!autoPlayback)}
            />

            {/* Status Bar */}
            <div className={styles.statusBar}>
              {['VAD', 'Encoder', 'Reasoning', 'TTS'].map(label => {
                const key = label.toLowerCase();
                const isActive = activeLayers[key] === 'running';
                return (
                  <div key={key} className={`${styles.statusItem} ${isActive ? styles.statusActive : ''}`}>
                    <span className={styles.indicator} />
                    <span>{label}</span>
                  </div>
                );
              })}
            </div>

            <Conversation
              messages={messages}
              isProcessing={isProcessing}
              onSendText={handleSendText}
            />
          </div>

          <Sidebar
            sessionId={ws.sessionId}
            duration={duration}
            language={language}
            emotion={emotion}
            activeLayers={activeLayers}
            timings={timings}
          />
        </main>
      </div>
    </>
  );
}
