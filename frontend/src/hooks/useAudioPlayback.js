import { useRef, useCallback } from 'react';

/**
 * Hook for playing PCM audio responses from the server.
 */
export default function useAudioPlayback(defaultSampleRate = 22050) {
  const ctxRef = useRef(null);
  const mutedRef = useRef(false);
  const activeSourcesRef = useRef([]);
  const nextPlayTimeRef = useRef(0);
  const analyserRef = useRef(null);

  const playBase64 = useCallback((base64Data, sampleRate) => {
    if (mutedRef.current) return;

    const binary = atob(base64Data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }

    const int16 = new Int16Array(bytes.buffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768;
    }

    const rate = sampleRate || defaultSampleRate;
    if (!ctxRef.current) {
      ctxRef.current = new AudioContext({ sampleRate: rate });
      analyserRef.current = ctxRef.current.createAnalyser();
      analyserRef.current.fftSize = 512;
      analyserRef.current.connect(ctxRef.current.destination);
      nextPlayTimeRef.current = ctxRef.current.currentTime;
    }

    // Ensure we don't schedule in the past
    if (nextPlayTimeRef.current < ctxRef.current.currentTime) {
      nextPlayTimeRef.current = ctxRef.current.currentTime;
    }

    const buffer = ctxRef.current.createBuffer(1, float32.length, rate);
    buffer.getChannelData(0).set(float32);

    const source = ctxRef.current.createBufferSource();
    source.buffer = buffer;
    source.connect(analyserRef.current);
    source.onended = () => {
      activeSourcesRef.current = activeSourcesRef.current.filter(s => s !== source);
    };
    activeSourcesRef.current.push(source);
    
    source.start(nextPlayTimeRef.current);
    nextPlayTimeRef.current += buffer.duration;
  }, [defaultSampleRate]);

  const stopPlayback = useCallback(() => {
    activeSourcesRef.current.forEach(source => {
      try {
        source.stop();
      } catch (e) {
        // ignore
      }
    });
    activeSourcesRef.current = [];
    nextPlayTimeRef.current = 0;
  }, []);

  const setMuted = useCallback((muted) => {
    mutedRef.current = muted;
  }, []);

  const getAnalyserNode = useCallback(() => {
    return analyserRef.current;
  }, []);

  return { playBase64, setMuted, stopPlayback, getAnalyserNode };
}
