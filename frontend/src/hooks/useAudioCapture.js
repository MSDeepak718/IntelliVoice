import { useRef, useState, useCallback } from 'react';

/**
 * Custom hook for microphone audio capture.
 * Outputs raw PCM int16 mono at 16kHz.
 */
export default function useAudioCapture({ sampleRate = 16000, onAudioData }) {
  const streamRef = useRef(null);
  const audioCtxRef = useRef(null);
  const sourceRef = useRef(null);
  const analyserRef = useRef(null);
  const scriptRef = useRef(null);

  const [isRecording, setIsRecording] = useState(false);
  const [hasPermission, setHasPermission] = useState(null);

  const onAudioDataRef = useRef(onAudioData);
  onAudioDataRef.current = onAudioData;

  const float32ToInt16 = (float32Array) => {
    const int16 = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return int16;
  };

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      streamRef.current = stream;
      setHasPermission(true);

      const audioCtx = new AudioContext({ sampleRate });
      audioCtxRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      sourceRef.current = source;

      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.85;
      analyserRef.current = analyser;
      source.connect(analyser);

      const scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
      scriptNode.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        const pcm = float32ToInt16(input);
        if (onAudioDataRef.current) {
          onAudioDataRef.current(pcm.buffer);
        }
      };
      scriptRef.current = scriptNode;

      analyser.connect(scriptNode);
      scriptNode.connect(audioCtx.destination);

      setIsRecording(true);
    } catch (err) {
      console.error('[AudioCapture] Permission denied:', err);
      setHasPermission(false);
    }
  }, [sampleRate]);

  const stop = useCallback(() => {
    if (scriptRef.current && analyserRef.current) {
      try {
        analyserRef.current.disconnect(scriptRef.current);
        scriptRef.current.disconnect();
      } catch (e) { /* may already be disconnected */ }
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }

    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }

    analyserRef.current = null;
    scriptRef.current = null;
    setIsRecording(false);
  }, []);

  const getAnalyserNode = useCallback(() => analyserRef.current, []);

  return { isRecording, hasPermission, start, stop, getAnalyserNode };
}
