import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  getApiBaseUrl,
  type ApiMessagePayload,
  type ChatMessage,
  type ChatSession,
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
        isThinking: true,
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

      try {
        const baseUrl = apiBaseUrlRef.current || getApiBaseUrl();
        apiBaseUrlRef.current = baseUrl;

        const response = await fetch(`${baseUrl}${STREAM_ENDPOINT}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ messages: payloadMessages }),
          signal: controller.signal,
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(errorText || `请求失败: ${response.status}`);
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
        const decoder = new TextDecoder('utf-8');
        let aggregated = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            break;
          }
          if (value) {
            const chunk = decoder.decode(value, { stream: true });
            if (!chunk) {
              continue;
            }
            aggregated += chunk;
            updateAssistantMessage(targetSessionId, assistantPlaceholder.id, (prev) => ({
              ...prev,
              content: aggregated,
              isThinking: true,
              timestamp: prev.timestamp ?? Date.now(),
            }));
          }
        }

        aggregated += decoder.decode();
        const finalText = (aggregated || '助手暂时没有回复。').trim();
        updateAssistantMessage(targetSessionId, assistantPlaceholder.id, (prev) => ({
          ...prev,
          content: finalText,
          isThinking: false,
          timestamp: Date.now(),
        }));
        triggerTitleUpdate(finalText);
      } catch (error) {
        if ((error as DOMException)?.name === 'AbortError') {
          updateAssistantMessage(targetSessionId, assistantPlaceholder.id, (prev) => ({
            ...prev,
            content: prev.content || '（请求已取消）',
            isThinking: false,
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
          timestamp: Date.now(),
        }));
      } finally {
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
