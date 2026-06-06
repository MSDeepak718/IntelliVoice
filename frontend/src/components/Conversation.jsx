import { useRef, useEffect, useState } from 'react';
import styles from './Conversation.module.css';

function ProcessingDots() {
  return (
    <div className={styles.message + ' ' + styles.assistant}>
      <div className={styles.bubble + ' ' + styles.assistantBubble}>
        <div className={styles.dots}>
          <span /><span /><span />
        </div>
      </div>
    </div>
  );
}

function Message({ role, text, emotion, time }) {
  const isUser = role === 'user';
  return (
    <div className={`${styles.message} ${isUser ? styles.user : styles.assistant}`}>
      <div className={`${styles.bubble} ${isUser ? styles.userBubble : styles.assistantBubble}`}>
        {text}
      </div>
      <div className={styles.meta}>
        <span>{time}</span>
        {emotion && emotion !== 'neutral' && (
          <span className={styles.emotionTag}>{emotion}</span>
        )}
      </div>
    </div>
  );
}

export default function Conversation({ messages, isProcessing, onSendText }) {
  const scrollRef = useRef(null);
  const [inputValue, setInputValue] = useState('');

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isProcessing]);

  const handleSend = () => {
    const text = inputValue.trim();
    if (!text) return;
    onSendText(text);
    setInputValue('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.title}>Conversation</span>
        <span className={styles.badge}>{messages.length} message{messages.length !== 1 ? 's' : ''}</span>
      </div>

      <div className={styles.messages} ref={scrollRef}>
        {messages.length === 0 && !isProcessing && (
          <div className={styles.empty}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" x2="12" y1="19" y2="22"/>
            </svg>
            <span className={styles.emptyTitle}>Start a conversation</span>
            <span className={styles.emptyDesc}>
              Click the microphone button and speak, or type a message below.
            </span>
          </div>
        )}

        {messages.map((msg, i) => (
          <Message key={i} {...msg} />
        ))}

        {isProcessing && <ProcessingDots />}
      </div>

      <div className={styles.inputBar}>
        <input
          type="text"
          className={styles.textInput}
          placeholder="Type a message..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          autoComplete="off"
        />
        <button
          className={styles.sendBtn}
          onClick={handleSend}
          disabled={!inputValue.trim()}
          title="Send message"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" x2="11" y1="2" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>
    </div>
  );
}
