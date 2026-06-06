import { useRef, useState, useCallback, useEffect } from 'react';

/**
 * Custom hook for managing the WebSocket connection to the IntelliVoice backend.
 */
export default function useWebSocket(url) {
  const wsRef = useRef(null);
  const heartbeatRef = useRef(null);
  const reconnectRef = useRef(null);
  const attemptsRef = useRef(0);
  const closedIntentionallyRef = useRef(false);

  const [status, setStatus] = useState('disconnected'); // 'disconnected' | 'connecting' | 'connected' | 'error'
  const [sessionId, setSessionId] = useState(null);

  const handlersRef = useRef({});

  const setHandler = useCallback((type, handler) => {
    handlersRef.current[type] = handler;
  }, []);

  const clearTimers = useCallback(() => {
    if (heartbeatRef.current) {
      clearInterval(heartbeatRef.current);
      heartbeatRef.current = null;
    }
    if (reconnectRef.current) {
      clearTimeout(reconnectRef.current);
      reconnectRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState !== WebSocket.CLOSED) return;

    closedIntentionallyRef.current = false;
    setStatus('connecting');

    try {
      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('connected');
        attemptsRef.current = 0;
        heartbeatRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
          }
        }, 15000);
      };

      ws.onmessage = (event) => {
        if (typeof event.data !== 'string') return;
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'pong') return;
          if (msg.type === 'connected') {
            setSessionId(msg.session_id);
          }
          const handler = handlersRef.current[msg.type];
          if (handler) handler(msg);
        } catch (e) {
          console.error('[WS] Parse error:', e);
        }
      };

      ws.onclose = (event) => {
        setStatus('disconnected');
        clearTimers();
        if (!closedIntentionallyRef.current && attemptsRef.current < 10) {
          attemptsRef.current += 1;
          const delay = 3000 * Math.min(attemptsRef.current, 5);
          reconnectRef.current = setTimeout(connect, delay);
        }
      };

      ws.onerror = () => {
        setStatus('error');
      };
    } catch (err) {
      setStatus('error');
    }
  }, [url, clearTimers]);

  const disconnect = useCallback(() => {
    closedIntentionallyRef.current = true;
    clearTimers();
    if (wsRef.current) {
      wsRef.current.close(1000, 'Client disconnect');
      wsRef.current = null;
    }
    setStatus('disconnected');
  }, [clearTimers]);

  const sendAudio = useCallback((arrayBuffer) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(arrayBuffer);
      return true;
    }
    return false;
  }, []);

  const sendJSON = useCallback((data) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
      return true;
    }
    return false;
  }, []);

  const sendText = useCallback((text) => {
    return sendJSON({ type: 'text_message', text });
  }, [sendJSON]);

  const sendReset = useCallback(() => {
    return sendJSON({ type: 'reset' });
  }, [sendJSON]);

  useEffect(() => {
    return () => {
      closedIntentionallyRef.current = true;
      clearTimers();
      if (wsRef.current) wsRef.current.close();
    };
  }, [clearTimers]);

  return {
    status,
    sessionId,
    connect,
    disconnect,
    sendAudio,
    sendJSON,
    sendText,
    sendReset,
    setHandler,
    isConnected: status === 'connected',
  };
}
