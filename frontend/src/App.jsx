import React, { useState, useCallback, useEffect, useRef } from 'react';
import useWebSocket from './hooks/useWebSocket';
import useAudioCapture from './hooks/useAudioCapture';
import useAudioPlayback from './hooks/useAudioPlayback';
import styles from './App.module.css';

const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.hostname || 'localhost'}:8000/ws/audio`;

export default function App() {
  const [messages, setMessages] = useState([]);
  const [isUserSpeaking, setIsUserSpeaking] = useState(false);
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  
  // Hooks
  const ws = useWebSocket(WS_URL);
  const playback = useAudioPlayback(22050);
  
  const audioCapture = useAudioCapture({
    sampleRate: 16000,
    onAudioData: useCallback((buffer) => {
      ws.sendAudio(buffer);
    }, [ws]),
  });

  // WebSocket Handlers
  useEffect(() => {
    ws.setHandler('vad', (msg) => {
      if (msg.status === 'speech_start') {
        setIsUserSpeaking(true);
      } else if (msg.status === 'speech_end') {
        setIsUserSpeaking(false);
      }
    });

    ws.setHandler('transcription', (msg) => {
      setIsUserSpeaking(false);
      setMessages(prev => [...prev, { role: 'user', text: msg.text }]);
    });

    ws.setHandler('response_start', () => {
      setMessages(prev => [...prev, { role: 'assistant', text: '' }]);
    });

    ws.setHandler('response_chunk', (msg) => {
      setMessages(prev => {
        const newMessages = [...prev];
        const lastIndex = newMessages.findLastIndex(m => m.role === 'assistant');
        if (lastIndex !== -1) {
          newMessages[lastIndex] = {
            ...newMessages[lastIndex],
            text: newMessages[lastIndex].text + msg.text
          };
        } else {
          newMessages.push({ role: 'assistant', text: msg.text });
        }
        return newMessages;
      });
    });

    ws.setHandler('audio_response', (msg) => {
      playback.playBase64(msg.audio, msg.sample_rate);
    });

    ws.setHandler('interrupt', () => {
      playback.stopPlayback();
    });

    ws.setHandler('reset', () => {
      setMessages([]);
    });

    ws.setHandler('error', (msg) => {
      setMessages(prev => [...prev, { role: 'assistant', text: msg.message || 'Error occurred' }]);
    });
  }, [ws, playback]);

  // Auto-scroll logic
  const listRef = useRef(null);
  
  const scrollToBottom = () => {
    if (listRef.current) {
      listRef.current.scrollTo({
        top: listRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isUserSpeaking]);

  // Bright Volumetric Wave Logic (AI output bound)
  const drawWave = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const analyser = playback.getAnalyserNode();
    
    const bufferLength = analyser ? analyser.frequencyBinCount : 256;
    const dataArray = new Uint8Array(bufferLength);
    
    let time = 0;

    const render = () => {
      animationRef.current = requestAnimationFrame(render);
      
      if (analyser) {
        analyser.getByteFrequencyData(dataArray);
      } else {
        dataArray.fill(0);
      }

      const subBass = dataArray.slice(0, 5).reduce((a, b) => a + b, 0) / 5;
      const bass = dataArray.slice(5, 15).reduce((a, b) => a + b, 0) / 10;
      const mids = dataArray.slice(15, 40).reduce((a, b) => a + b, 0) / 25;
      const highMids = dataArray.slice(40, 80).reduce((a, b) => a + b, 0) / 40;
      const treble = dataArray.slice(80, 150).reduce((a, b) => a + b, 0) / 70;

      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = '#050505';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const centerY = canvas.height / 2;

      const calculateY = (nx, complexity, speed, phaseOffset, amplitude, taper, direction = -1) => {
        let waveY = 0;
        waveY += Math.sin(nx * complexity + time * speed + phaseOffset) * 1.0;
        waveY += Math.sin(nx * complexity * 2.2 - time * speed * 1.3) * 0.5;
        waveY += Math.sin(nx * complexity * 3.5 + time * speed * 0.8) * 0.25;

        return centerY + (waveY * amplitude * taper * direction);
      };

      const drawFilledReflection = (rgbGlow, amplitude, speed, phaseOffset, complexity) => {
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        for (let i = 0; i <= canvas.width; i += 2) {
          const nx = (i / canvas.width) * 2 - 1; 
          const taper = Math.exp(-Math.pow(nx * 2.5, 2)); 
          const y = calculateY(nx, complexity, speed, phaseOffset, amplitude, taper, 1);
          ctx.lineTo(i, y);
        }
        ctx.lineTo(canvas.width, centerY);
        ctx.closePath();

        const fillGradient = ctx.createLinearGradient(0, centerY, 0, centerY + (amplitude * 2));
        fillGradient.addColorStop(0, `rgba(${rgbGlow}, 0.4)`); 
        fillGradient.addColorStop(1, `rgba(${rgbGlow}, 0.0)`);  
        
        ctx.fillStyle = fillGradient;
        ctx.fill();

        ctx.beginPath();
        for (let i = 0; i <= canvas.width; i += 2) {
          const nx = (i / canvas.width) * 2 - 1; 
          const taper = Math.exp(-Math.pow(nx * 2.5, 2)); 
          const y = calculateY(nx, complexity, speed, phaseOffset, amplitude, taper, 1);
          if (i === 0) ctx.moveTo(i, y);
          else ctx.lineTo(i, y);
        }
        ctx.lineWidth = 2; 
        ctx.strokeStyle = `rgba(${rgbGlow}, 0.3)`; 
        ctx.stroke();
      };

      const drawFilledWave = (baseColor, rgbGlow, amplitude, speed, phaseOffset, complexity) => {
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        for (let i = 0; i <= canvas.width; i += 2) {
          const nx = (i / canvas.width) * 2 - 1; 
          const taper = Math.exp(-Math.pow(nx * 2.5, 2)); 
          const y = calculateY(nx, complexity, speed, phaseOffset, amplitude, taper, -1);
          ctx.lineTo(i, y);
        }
        ctx.lineTo(canvas.width, centerY);
        ctx.closePath();

        const fillGradient = ctx.createLinearGradient(0, centerY - (amplitude * 2), 0, centerY);
        fillGradient.addColorStop(0, `rgba(${rgbGlow}, 0.8)`); 
        fillGradient.addColorStop(1, `rgba(${rgbGlow}, 0.1)`); 
        
        ctx.fillStyle = fillGradient;
        ctx.fill();

        ctx.beginPath();
        for (let i = 0; i <= canvas.width; i += 2) {
          const nx = (i / canvas.width) * 2 - 1; 
          const taper = Math.exp(-Math.pow(nx * 2.5, 2)); 
          const y = calculateY(nx, complexity, speed, phaseOffset, amplitude, taper, -1);
          if (i === 0) ctx.moveTo(i, y);
          else ctx.lineTo(i, y);
        }

        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        
        ctx.lineWidth = 12;
        ctx.strokeStyle = `rgba(${rgbGlow}, 0.3)`; 
        ctx.stroke();

        ctx.lineWidth = 4;
        ctx.strokeStyle = `rgba(${rgbGlow}, 0.7)`; 
        ctx.stroke();

        ctx.lineWidth = 1.5;
        ctx.strokeStyle = baseColor;
        ctx.stroke();
      };

      ctx.globalCompositeOperation = 'screen';

      const a1 = 30 + (subBass / 255) * 100;
      const a2 = 25 + (bass / 255) * 85;
      const a3 = 20 + (mids / 255) * 70;
      const a4 = 15 + (highMids / 255) * 60;
      const a5 = 10 + (treble / 255) * 50;

      const purple = '138, 43, 226';
      const pink = '255, 20, 147';
      const blue = '65, 105, 225';
      const cyan = '0, 255, 255';
      const orange = '255, 140, 0';

      drawFilledReflection(purple, a1, 2.0, 0, 2.5);
      drawFilledReflection(pink, a2, 2.8, Math.PI / 4, 3.2);
      drawFilledReflection(blue, a3, 3.5, Math.PI, 4.0);
      drawFilledReflection(cyan, a4, 4.2, Math.PI * 1.5, 4.8);
      drawFilledReflection(orange, a5, 5.0, Math.PI / 2, 5.5);

      drawFilledWave('#FFFFFF', purple, a1, 2.0, 0, 2.5); 
      drawFilledWave('#FFFFFF', pink, a2, 2.8, Math.PI / 4, 3.2);
      drawFilledWave('#FFFFFF', blue, a3, 3.5, Math.PI, 4.0);
      drawFilledWave('#FFFFFF', cyan, a4, 4.2, Math.PI * 1.5, 4.8);
      drawFilledWave('#FFFFFF', orange, a5, 5.0, Math.PI / 2, 5.5);

      time += 0.012; 
    };

    render();
  }, [playback]);

  useEffect(() => {
    drawWave();
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [drawWave]);

  // Auto-connect
  useEffect(() => {
    ws.connect();
  }, [ws]);

  const toggleConnection = () => {
    if (ws.isConnected) {
      ws.disconnect();
    } else {
      ws.connect();
    }
  };

  const toggleMic = async () => {
    if (audioCapture.isRecording) {
      audioCapture.stop();
    } else {
      if (!ws.isConnected) {
        ws.connect();
      }
      await audioCapture.start();
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.glowBg} />
      
      {/* LEFT SIDE: Visuals and Controls */}
      <div className={styles.leftPanel}>
        
        <div className={styles.controls}>
          <div className={styles.controlGroup}>
            <button 
              className={`${styles.btn} ${ws.isConnected ? styles.btnConnectActive : styles.btnConnect}`}
              onClick={toggleConnection}
              title={ws.isConnected ? "Disconnect" : "Connect"}
            >
              {ws.isConnected ? (
                <svg className={styles.icon} viewBox="0 0 24 24">
                  <path d="M18.36 6.64a9 9 0 1 1-12.73 0"></path>
                  <line x1="12" y1="2" x2="12" y2="12"></line>
                </svg>
              ) : (
                <svg className={styles.icon} viewBox="0 0 24 24">
                  <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
                  <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
                </svg>
              )}
            </button>
            <span className={styles.label}>{ws.isConnected ? 'Connected' : 'Connect'}</span>
          </div>
          
          <div className={styles.controlGroup}>
            <button 
              className={`${styles.btn} ${audioCapture.isRecording ? styles.btnMicActive : styles.btnMic}`}
              onClick={toggleMic}
              title={audioCapture.isRecording ? "Turn off microphone" : "Turn on microphone"}
            >
              {audioCapture.isRecording ? (
                <svg className={styles.icon} viewBox="0 0 24 24">
                  <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                  <line x1="12" y1="19" x2="12" y2="22"></line>
                </svg>
              ) : (
                <svg className={styles.icon} viewBox="0 0 24 24">
                  <line x1="1" y1="1" x2="23" y2="23"></line>
                  <path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"></path>
                  <path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23"></path>
                  <line x1="12" y1="19" x2="12" y2="22"></line>
                </svg>
              )}
            </button>
            <span className={styles.label}>{audioCapture.isRecording ? 'Listening' : 'Mic Off'}</span>
          </div>
        </div>

        <div className={styles.canvasWrapper}>
          <canvas 
            ref={canvasRef} 
            width={800} 
            height={400} 
            className={styles.canvas}
          />
        </div>
      </div>

      {/* RIGHT SIDE: Transcription */}
      <div className={styles.rightPanel}>
        <div className={styles.convHeader}>Live Transcript</div>
        
        <div className={styles.messageList} ref={listRef}>
          {messages.length === 0 ? (
            <div className={styles.emptyState}>
              <div className={styles.emptyStateIcon}>🎙️</div>
              <span>No messages yet. Turn on the mic and speak!</span>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div 
                key={idx} 
                className={`${styles.message} ${msg.role === 'user' ? styles.msgUser : styles.msgAssistant}`}
              >
                {msg.text}
              </div>
            ))
          )}
          
          {isUserSpeaking && (
            <div className={`${styles.message} ${styles.msgUser} ${styles.speakingBubble}`}>
              <div className={styles.dotFlashing}></div>
            </div>
          )}
        </div>
      </div>

    </div>
  );
}
