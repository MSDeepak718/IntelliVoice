import { useEffect, useRef } from 'react';
import styles from './Visualizer.module.css';

const ACCENT = '#6366f1';
const ACCENT_FADED = 'rgba(99, 102, 241, 0.15)';
const GRID_COLOR = 'rgba(99, 102, 241, 0.04)';

export default function Visualizer({ analyserNode, isActive }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const analyserRef = useRef(analyserNode);
  analyserRef.current = analyserNode;

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    const resize = () => {
      const rect = canvas.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';
      ctx.scale(dpr, dpr);
    };

    const observer = new ResizeObserver(resize);
    observer.observe(canvas.parentElement);
    resize();

    const drawGrid = (w, h) => {
      ctx.strokeStyle = GRID_COLOR;
      ctx.lineWidth = 0.5;
      for (let y = 0; y < h; y += 32) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
      for (let x = 0; x < w; x += 32) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
    };

    const drawIdle = () => {
      const rect = canvas.parentElement.getBoundingClientRect();
      const w = rect.width, h = rect.height;
      ctx.clearRect(0, 0, w, h);
      drawGrid(w, h);
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.strokeStyle = ACCENT_FADED;
      ctx.lineWidth = 1;
      ctx.stroke();
    };

    const drawWaveform = () => {
      const analyser = analyserRef.current;
      if (!analyser) { drawIdle(); return; }

      const bufLen = analyser.frequencyBinCount;
      const data = new Uint8Array(bufLen);
      analyser.getByteTimeDomainData(data);

      const rect = canvas.parentElement.getBoundingClientRect();
      const w = rect.width, h = rect.height;
      ctx.clearRect(0, 0, w, h);
      drawGrid(w, h);

      const sliceW = w / bufLen;

      // Glow
      ctx.beginPath();
      ctx.lineWidth = 4;
      ctx.strokeStyle = ACCENT_FADED;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      let x = 0;
      for (let i = 0; i < bufLen; i++) {
        const y = (data[i] / 128.0) * (h / 2);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        x += sliceW;
      }
      ctx.stroke();

      // Crisp
      ctx.beginPath();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = ACCENT;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      x = 0;
      for (let i = 0; i < bufLen; i++) {
        const y = (data[i] / 128.0) * (h / 2);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        x += sliceW;
      }
      ctx.stroke();
    };

    const loop = () => {
      if (isActive && analyserRef.current) {
        drawWaveform();
      } else {
        drawIdle();
      }
      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(animRef.current);
      observer.disconnect();
    };
  }, [isActive]);

  return (
    <div className={styles.container}>
      <canvas ref={canvasRef} className={styles.canvas} />
      {!isActive && (
        <div className={styles.overlay}>
          <span className={styles.label}>Audio Visualizer</span>
        </div>
      )}
    </div>
  );
}
