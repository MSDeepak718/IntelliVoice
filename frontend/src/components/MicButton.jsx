import styles from './MicButton.module.css';

export default function MicButton({ isRecording, isProcessing, onClick, onMuteToggle, isMuted, autoPlayback, onPlaybackToggle }) {
  const btnClass = [
    styles.micBtn,
    isRecording && styles.recording,
    isProcessing && styles.processing,
  ].filter(Boolean).join(' ');

  return (
    <div className={styles.section}>
      <button
        className={styles.ctrlBtn}
        onClick={onMuteToggle}
        title={isMuted ? 'Unmute' : 'Mute'}
        style={isMuted ? { color: 'var(--error)' } : {}}
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          {isMuted ? (
            <>
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
              <line x1="23" x2="17" y1="9" y2="15"/>
              <line x1="17" x2="23" y1="9" y2="15"/>
            </>
          ) : (
            <>
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
              <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
              <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
            </>
          )}
        </svg>
      </button>

      <button className={btnClass} onClick={onClick} title="Click to toggle recording" disabled={isProcessing}>
        {isRecording ? (
          <svg viewBox="0 0 24 24" fill="currentColor" stroke="none">
            <rect x="6" y="6" width="12" height="12" rx="2"/>
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            <line x1="12" x2="12" y1="19" y2="22"/>
          </svg>
        )}
      </button>

      <button
        className={styles.ctrlBtn}
        onClick={onPlaybackToggle}
        title={autoPlayback ? 'Auto-playback ON' : 'Auto-playback OFF'}
        style={autoPlayback ? { color: 'var(--success)' } : {}}
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polygon points="6 3 20 12 6 21 6 3"/>
        </svg>
      </button>
    </div>
  );
}
