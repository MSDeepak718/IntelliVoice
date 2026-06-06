import styles from './Header.module.css';

export default function Header({ connectionStatus, onReset }) {
  const badgeClass =
    connectionStatus === 'connected' ? styles.connected :
    connectionStatus === 'error' ? styles.error : '';

  const label =
    connectionStatus === 'connected' ? 'Connected' :
    connectionStatus === 'connecting' ? 'Connecting...' :
    connectionStatus === 'error' ? 'Error' : 'Disconnected';

  return (
    <header className={styles.header}>
      <div className={styles.logo}>
        <div className={styles.logoMark}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            <line x1="12" x2="12" y1="19" y2="22"/>
          </svg>
        </div>
        <span className={styles.logoText}>IntelliVoice</span>
      </div>

      <div className={styles.controls}>
        <div className={`${styles.badge} ${badgeClass}`}>
          <span className={styles.dot} />
          <span>{label}</span>
        </div>
        <button className={styles.iconBtn} onClick={onReset} title="Reset conversation">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>
            <path d="M21 3v5h-5"/>
            <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>
            <path d="M8 16H3v5"/>
          </svg>
        </button>
      </div>
    </header>
  );
}
