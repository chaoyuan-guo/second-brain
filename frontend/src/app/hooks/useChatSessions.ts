import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  getApiBaseUrl,
  type ApiMessagePayload,
  type ChatMessage,
  type ChatSession,
  type StreamEvent,
  STORAGE_KEY,
  STREAM_ENDPOINT,
  TITLE_ENDPOINT,
} from '../lib/chat-types';
import {
  createEmptySession,
  createId,
  deriveTitle,
  serializeMessagesForApi,
} from '../lib/chat-helpers';

interface UseChatSessionsResult {
  sessions: ChatSession[];
  activeSession?: ChatSession;
  activeSessionId: string;
  setActiveSessionId: (id: string) => void;
  inputValue: string;
  setInputValue: (value: string) => void;
  pendingSessions: Record<string, boolean>;
  hydrated: boolean;
  isActivePending: boolean;
  isAnyPending: boolean;
  createNewSession: () => void;
  deleteSession: (sessionId: string) => void;
  renameSession: (sessionId: string, value: string) => void;
  clearActiveSession: () => void;
  handleSubmit: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  abortSessionRequest: (sessionId: string) => void;
  refreshSessionTitle: (sessionId: string, messagesOverride?: ApiMessagePayload[]) => Promise<void>;
}

export function useChatSessions(): UseChatSessionsResult {
  const defaultSession = useMemo(() => createEmptySession(), []);
  const [sessions, setSessions] = useState<ChatSession[]>([defaultSession]);
  const [activeSessionId, setActiveSessionId] = useState(defaultSession.id);
  const [inputValue, setInputValue] = useState('');
  const [pendingSessions, setPendingSessions] = useState<Record<string, boolean>>({});
  const [hydrated, setHydrated] = useState(false);

  const streamControllersRef = useRef(new Map<string, AbortController>());
  const apiBaseUrlRef = useRef<string>(typeof window === 'undefined' ? '' : getApiBaseUrl());
  const titleRefreshTrackerRef = useRef<Map<string, number>>(new Map());

  const activeSession =
    sessions.find((session) => session.id === activeSessionId) ?? sessions[0] ?? defaultSession;
  const isActivePending = Boolean(activeSession && pendingSessions[activeSession.id]);
  const isAnyPending = Object.values(pendingSessions).some(Boolean);

  const setSessionPending = useCallback((sessionId: string, pending: boolean) => {
    setPendingSessions((prev) => {
      if (pending) {
        if (prev[sessionId]) {
          return prev;
        }
        return { ...prev, [sessionId]: true };
      }
      if (!prev[sessionId]) {
        return prev;
      }
      const { [sessionId]: _discarded, ...rest } = prev;
      return rest;
    });
  }, []);

  const abortSessionRequest = useCallback(
    (sessionId: string) => {
      const controller = streamControllersRef.current.get(sessionId);
      if (controller) {
        controller.abort();
        streamControllersRef.current.delete(sessionId);
      }
      setSessionPending(sessionId, false);
    },
    [setSessionPending],
  );

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed: ChatSession[] = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setSessions(parsed);
          setActiveSessionId(parsed[0].id);
        }
      }
      document.documentElement.dataset.theme = 'light';
    } catch (error) {
      console.error('Failed to load sessions', error);
    } finally {
      setHydrated(true);
    }
  }, [defaultSession.id]);

  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  }, [sessions, hydrated]);

  useEffect(() => {
    if (!hydrated) return;
    document.documentElement.dataset.theme = 'light';
  }, [hydrated]);

  useEffect(() => {
    return () => {
      streamControllersRef.current.forEach((controller) => controller.abort());
      streamControllersRef.current.clear();
    };
  }, []);

  const upsertSession = useCallback(
    (sessionId: string, updater: (session: ChatSession) => ChatSession) => {
      setSessions((prev) => {
        const existing =
          prev.find((item) => item.id === sessionId) ?? createEmptySession(sessionId);
        const updated = updater(existing);
        const others = prev.filter((item) => item.id !== sessionId);
        return [updated, ...others];
      });
    },
    [],
  );

  const createNewSession = useCallback(() => {
    const newSession = createEmptySession();
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
    setInputValue('');
  }, []);

  const deleteSession = useCallback(
    (sessionId: string) => {
      abortSessionRequest(sessionId);
      setSessions((prev) => {
        const remaining = prev.filter((session) => session.id !== sessionId);
        if (remaining.length === 0) {
          const fallback = createEmptySession();
          setActiveSessionId(fallback.id);
          return [fallback];
        }
        if (sessionId === activeSessionId) {
          setActiveSessionId(remaining[0].id);
        }
        return remaining;
      });
    },
    [abortSessionRequest, activeSessionId],
  );

  const renameSession = useCallback(
    (sessionId: string, value: string) => {
      const title = value.trim() || '新的对话';
      upsertSession(sessionId, (session) => ({
        ...session,
        title,
        isCustomTitle: true,
      }));
    },
    [upsertSession],
  );

  const refreshSessionTitle = useCallback(
    async (sessionId: string, messagesOverride?: ApiMessagePayload[]) => {
      const session = sessions.find((item) => item.id === sessionId);
      if (!session || session.isCustomTitle) {
        return;
      }

      const tracker = titleRefreshTrackerRef.current;
      const now = Date.now();
      const lastRequested = tracker.get(sessionId) ?? 0;
      if (now - lastRequested < 5000) {
        return;
      }
      tracker.set(sessionId, now);

      try {
        const baseUrl = apiBaseUrlRef.current || getApiBaseUrl();
        apiBaseUrlRef.current = baseUrl;
        const messagesForTitle: ApiMessagePayload[] = messagesOverride?.length
          ? messagesOverride
          : serializeMessagesForApi(session);
        if (!messagesForTitle.length) {
          return;
        }

        const response = await fetch(`${baseUrl}${TITLE_ENDPOINT}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ messages: messagesForTitle }),
        });

        if (!response.ok) {
          throw new Error(`生成标题失败: ${response.status}`);
        }

        const data = (await response.json()) as { title?: string };
        const generatedTitle = data.title?.trim();
        if (generatedTitle) {
          upsertSession(sessionId, (current) => {
            if (current.isCustomTitle) {
              return current;
            }
            return {
              ...current,
              title: generatedTitle,
            };
          });
        }
      } catch (error) {
        console.warn('自动生成标题失败', error);
      }
    },
    [sessions, upsertSession],
  );

  const clearActiveSession = useCallback(() => {
    if (!activeSession) return;
    abortSessionRequest(activeSession.id);
    upsertSession(activeSession.id, (session) => ({
      ...session,
      messages: [],
    }));
  }, [abortSessionRequest, activeSession, upsertSession]);

  const updateAssistantMessage = useCallback(
    (
      sessionId: string,
      messageId: string,
      updater: (prev: ChatMessage) => ChatMessage,
    ) => {
      upsertSession(sessionId, (session) => {
        if (!session.messages.some((message) => message.id === messageId)) {
          return session;
        }
        return {
          ...session,
          messages: session.messages.map((message) =>
            message.id === messageId ? updater(message) : message,
          ),
        };
      });
    },
    [upsertSession],
  );

  const parseNdjsonStream = useCallback(
    async (
      reader: ReadableStreamDefaultReader<Uint8Array>,
      onEvent: (event: StreamEvent) => void,
    ) => {
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        if (!value) {
          continue;
        }
        buffer += decoder.decode(value, { stream: true });
        let newlineIndex = buffer.indexOf('\n');
        while (newlineIndex !== -1) {
          const line = buffer.slice(0, newlineIndex).trim();
          buffer = buffer.slice(newlineIndex + 1);
          if (line) {
            try {
              onEvent(JSON.parse(line) as StreamEvent);
            } catch {
              // ignore malformed chunks
            }
          }
          newlineIndex = buffer.indexOf('\n');
        }
      }

      buffer += decoder.decode();
      const tail = buffer.trim();
      if (tail) {
        try {
          onEvent(JSON.parse(tail) as StreamEvent);
        } catch {
          // ignore
        }
      }
    },
    [],
  );

  const handleSubmit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const content = inputValue.trim();
      if (!content || !activeSession) {
        return;
      }

      const targetSessionId = activeSession.id;
      if (pendingSessions[targetSessionId]) {
        return;
      }

      const payloadMessages: ApiMessagePayload[] = [
        ...serializeMessagesForApi(activeSession),
        {
          role: 'user',
          content,
        },
      ];

      const triggerTitleUpdate = (assistantContent: string) => {
        if (!assistantContent.trim()) {
          return;
        }
        const messagesForTitle: ApiMessagePayload[] = [
          ...payloadMessages,
          {
            role: 'assistant',
            content: assistantContent,
          },
        ];
        refreshSessionTitle(targetSessionId, messagesForTitle).catch(() => undefined);
      };

      const userMessage: ChatMessage = {
        id: createId(),
        role: 'user',
        content,
        timestamp: Date.now(),
      };

      const assistantPlaceholder: ChatMessage = {
        id: createId(),
        role: 'assistant',
        content: '',
        isThinking: false,
        statusText: '',
        timestamp: Date.now(),
      };

      upsertSession(targetSessionId, (session) => ({
        ...session,
        title: session.messages.length ? session.title : deriveTitle(content),
        messages: [...session.messages, userMessage, assistantPlaceholder],
      }));

      setInputValue('');
      setSessionPending(targetSessionId, true);

      const controller = new AbortController();
      streamControllersRef.current.set(targetSessionId, controller);

      let thinkingTimer: number | undefined;
      let statusTimer: number | undefined;

      try {
        const baseUrl = apiBaseUrlRef.current || getApiBaseUrl();
        apiBaseUrlRef.current = baseUrl;

        // 轻量等待：避免短请求闪烁
        thinkingTimer = window.setTimeout(() => {
          updateAssistantMessage(targetSessionId, assistantPlaceholder.id, (prev) => ({
            ...prev,
            isThinking: true,
          }));
        }, 300);

        // 复杂场景：若迟迟无文本输出，则展示状态文案
        let latestStatusText = '';
        let statusVisible = false;
        statusTimer = window.setTimeout(() => {
          statusVisible = true;
          updateAssistantMessage(targetSessionId, assistantPlaceholder.id, (prev) => ({
            ...prev,
            isThinking: true,
            statusText: prev.statusText || latestStatusText || '正在思考…',
          }));
        }, 700);

        const response = await fetch(`${baseUrl}${STREAM_ENDPOINT}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Stream-Format': 'ndjson',
            Accept: 'application/x-ndjson',
          },
          body: JSON.stringify({ messages: payloadMessages }),
          signal: controller.signal,
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(errorText || `请求失败: ${response.status}`);
        }

        const contentType = response.headers.get('content-type') ?? '';
        if (!contentType.includes('application/x-ndjson')) {
          const reader = response.body?.getReader();
          if (!reader) {
            const fallbackText = (await response.text()).trim() || '助手暂时没有回复。';
            updateAssistantMessage(targetSessionId, assistantPlaceholder.id, (prev) => ({
              ...prev,
              content: fallbackText,
              isThinking: false,
              statusText: '',
              timestamp: Date.now(),
            }));
            triggerTitleUpdate(fallbackText);
            return;
          }

          const decoder = new TextDecoder('utf-8');
          let aggregated = '';
          let receivedFirstChunk = false;
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            if (!value) continue;
            if (!receivedFirstChunk) {
              receivedFirstChunk = true;
              if (thinkingTimer) {
                window.clearTimeout(thinkingTimer);
                thinkingTimer = undefined;
              }
              if (statusTimer) {
                window.clearTimeout(statusTimer);
                statusTimer = undefined;
              }
            }
            const chunk = decoder.decode(value, { stream: true });
            if (!chunk) continue;
            aggregated += chunk;
            updateAssistantMessage(targetSessionId, assistantPlaceholder.id, (prev) => ({
              ...prev,
              content: aggregated,
              isThinking: true,
              statusText: '',
              timestamp: prev.timestamp ?? Date.now(),
            }));
          }
          aggregated += decoder.decode();
          const finalText = (aggregated || '助手暂时没有回复。').trim();
          updateAssistantMessage(targetSessionId, assistantPlaceholder.id, (prev) => ({
            ...prev,
            content: finalText,
            isThinking: false,
            statusText: '',
            timestamp: Date.now(),
          }));
          triggerTitleUpdate(finalText);
          return;
        }

        if (!response.body) {
          const fallbackText = (await response.text()).trim() || '助手暂时没有回复。';
          updateAssistantMessage(targetSessionId, assistantPlaceholder.id, (prev) => ({
            ...prev,
            content: fallbackText,
            isThinking: false,
            timestamp: Date.now(),
          }));
          triggerTitleUpdate(fallbackText);
          return;
        }

        const reader = response.body.getReader();
        let aggregated = '';
        let hasDelta = false;

        await parseNdjsonStream(reader, (event) => {
          if (event.type === 'delta') {
            hasDelta = true;
            statusVisible = false;
            if (thinkingTimer) {
              window.clearTimeout(thinkingTimer);
              thinkingTimer = undefined;
            }
            if (statusTimer) {
              window.clearTimeout(statusTimer);
              statusTimer = undefined;
            }
            aggregated += event.delta;
            updateAssistantMessage(targetSessionId, assistantPlaceholder.id, (prev) => ({
              ...prev,
              content: aggregated,
              isThinking: true,
              statusText: '',
              timestamp: prev.timestamp ?? Date.now(),
            }));
            return;
          }

          if (event.type === 'tool' || event.type === 'status') {
            latestStatusText = event.message;
            const shouldShow = event.type === 'tool' || (statusVisible && !hasDelta);
            if (!shouldShow) {
              return;
            }
            if (thinkingTimer) {
              window.clearTimeout(thinkingTimer);
              thinkingTimer = undefined;
            }
            updateAssistantMessage(targetSessionId, assistantPlaceholder.id, (prev) => ({
              ...prev,
              isThinking: true,
              statusText: event.message,
              timestamp: prev.timestamp ?? Date.now(),
            }));
            return;
          }

          if (event.type === 'done') {
            return;
          }
        });

        const finalText = (aggregated || (hasDelta ? '' : '助手暂时没有回复。')).trim();
        updateAssistantMessage(targetSessionId, assistantPlaceholder.id, (prev) => ({
          ...prev,
          content: finalText,
          isThinking: false,
          statusText: '',
          timestamp: Date.now(),
        }));
        triggerTitleUpdate(finalText);
      } catch (error) {
        // 确保定时器被清理
        if ((error as DOMException)?.name === 'AbortError') {
          updateAssistantMessage(targetSessionId, assistantPlaceholder.id, (prev) => ({
            ...prev,
            content: prev.content || '（请求已取消）',
            isThinking: false,
            statusText: '',
            timestamp: Date.now(),
          }));
          return;
        }
        const errorText = error instanceof Error ? error.message : '发生未知错误';
        updateAssistantMessage(targetSessionId, assistantPlaceholder.id, (prev) => ({
          ...prev,
          content: errorText,
          isThinking: false,
          isError: true,
          statusText: '',
          timestamp: Date.now(),
        }));
      } finally {
        if (thinkingTimer) window.clearTimeout(thinkingTimer);
        if (statusTimer) window.clearTimeout(statusTimer);
        const storedController = streamControllersRef.current.get(targetSessionId);
        if (storedController === controller) {
          streamControllersRef.current.delete(targetSessionId);
        }
        setSessionPending(targetSessionId, false);
      }
    },
    [
      inputValue,
      activeSession,
      pendingSessions,
      upsertSession,
      setSessionPending,
      setInputValue,
    ],
  );

  return {
    sessions,
    activeSession,
    activeSessionId,
    setActiveSessionId,
    inputValue,
    setInputValue,
    pendingSessions,
    hydrated,
    isActivePending,
    isAnyPending,
    createNewSession,
    deleteSession,
    renameSession,
    clearActiveSession,
    handleSubmit,
    abortSessionRequest,
    refreshSessionTitle,
  };
}
