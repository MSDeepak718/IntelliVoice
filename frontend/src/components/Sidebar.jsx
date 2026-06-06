import styles from './Sidebar.module.css';

const PIPELINE_LAYERS = [
  { key: 'vad', num: '1', name: 'Voice Activity' },
  { key: 'encoder', num: '2', name: 'Acoustic Encoder' },
  { key: 'semantic', num: '3', name: 'Semantic' },
  { key: 'emotion', num: '4', name: 'Emotion' },
  { key: 'reasoning', num: '6', name: 'Reasoning' },
  { key: 'tts', num: '10', name: 'Speech Gen' },
];

function Card({ title, children }) {
  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <span className={styles.cardTitle}>{title}</span>
      </div>
      <div className={styles.cardBody}>
        {children}
      </div>
    </div>
  );
}

function SessionCard({ sessionId, duration, language, emotion }) {
  return (
    <Card title="Session">
      <div className={styles.infoGrid}>
        <div className={styles.infoItem}>
          <span className={styles.infoLabel}>Session ID</span>
          <span className={styles.infoValue}>{sessionId ? sessionId.substring(0, 8) : '--'}</span>
        </div>
        <div className={styles.infoItem}>
          <span className={styles.infoLabel}>Duration</span>
          <span className={styles.infoValue}>{duration}</span>
        </div>
        <div className={styles.infoItem}>
          <span className={styles.infoLabel}>Language</span>
          <span className={styles.infoValue}>{language || '--'}</span>
        </div>
        <div className={styles.infoItem}>
          <span className={styles.infoLabel}>Emotion</span>
          <span className={styles.infoValue}>{emotion || '--'}</span>
        </div>
      </div>
    </Card>
  );
}

function PipelineCard({ activeLayers }) {
  return (
    <Card title="Pipeline">
      <div className={styles.pipelineLayers}>
        {PIPELINE_LAYERS.map(layer => {
          const state = activeLayers[layer.key] || 'idle';
          const isActive = state === 'running';
          const statusClass = state === 'running' ? styles.statusRunning :
                              state === 'done' ? styles.statusLoaded : '';

          return (
            <div key={layer.key} className={`${styles.pipelineLayer} ${isActive ? styles.layerActive : ''}`}>
              <div className={styles.layerInfo}>
                <span className={`${styles.layerNum} ${isActive ? styles.layerNumActive : ''}`}>
                  {layer.num}
                </span>
                <span className={styles.layerName}>{layer.name}</span>
              </div>
              <span className={`${styles.layerStatus} ${statusClass}`}>{state}</span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function TimingCard({ timings }) {
  const maxMs = 5000;
  const items = [
    { label: 'Preprocessing', key: 'preprocess_ms' },
    { label: 'Understanding', key: 'understanding_ms' },
    { label: 'Reasoning', key: 'reasoning_ms' },
    { label: 'Generation', key: 'generation_ms' },
  ];

  return (
    <Card title="Latency">
      <div className={styles.timingList}>
        {items.map(item => {
          const val = timings[item.key];
          const pct = val !== undefined ? Math.min(100, (val / maxMs) * 100) : 0;
          return (
            <div key={item.key}>
              <div className={styles.timingRow}>
                <span className={styles.timingLabel}>{item.label}</span>
                <span className={styles.timingValue}>{val !== undefined ? val + 'ms' : '--'}</span>
              </div>
              <div className={styles.timingBarBg}>
                <div className={styles.timingBar} style={{ width: pct + '%' }} />
              </div>
            </div>
          );
        })}
        <div className={styles.timingTotal}>
          <span className={styles.timingTotalLabel}>Total</span>
          <span className={styles.timingValue}>
            {timings.total_ms !== undefined ? timings.total_ms + 'ms' : '--'}
          </span>
        </div>
      </div>
    </Card>
  );
}

export default function Sidebar({ sessionId, duration, language, emotion, activeLayers, timings }) {
  return (
    <aside className={styles.sidebar}>
      <SessionCard sessionId={sessionId} duration={duration} language={language} emotion={emotion} />
      <PipelineCard activeLayers={activeLayers} />
      <TimingCard timings={timings} />
    </aside>
  );
}
