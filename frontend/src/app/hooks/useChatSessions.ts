import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  EVENT_ENDPOINT,
  getApiBaseUrl,
  type ChatMessage,
  type ChatSession,
  SESSION_ENDPOINT,
  sessionMessageEndpoint,
  type SourceRef,
  STORAGE_KEY,
} from '../lib/chat-types';
import { createEmptySession, createId, deriveTitle } from '../lib/chat-helpers';

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
  refreshSessionTitle: () => Promise<void>;
}

type JsonRecord = Record<string, unknown>;
type ToolStatus = 'pending' | 'running' | 'completed' | 'error';

const LEGACY_STORAGE_KEY = 'second_brain_sessions_v1';
const DEFAULT_OPENCODE_SESSION_PATH =
  process.env.NEXT_PUBLIC_OPENCODE_SESSION_PATH?.trim() || '/app';

const asRecord = (value: unknown): JsonRecord | undefined =>
  value && typeof value === 'object' ? (value as JsonRecord) : undefined;

const asString = (value: unknown): string | undefined =>
  typeof value === 'string' ? value : undefined;

const buildTraceId = (sessionId: string): string => {
  const suffix = createId().slice(0, 8);
  return `chat-${sessionId}-${suffix}`;
};

const extractSourceRefs = (metadata: unknown): SourceRef[] => {
  const record = asRecord(metadata);
  const raw = record?.source_refs;
  if (!Array.isArray(raw)) {
    return [];
  }

  const refs: SourceRef[] = [];
  raw.forEach((item) => {
    const entry = asRecord(item);
    const path = asString(entry?.path)?.trim();
    if (!path) {
      return;
    }

    const heading = asString(entry?.heading)?.trim() ?? '';
    const snippet = asString(entry?.snippet);
    const charOffsetRaw = entry?.char_offset;
    const char_offset = typeof charOffsetRaw === 'number' ? charOffsetRaw : undefined;
    refs.push({ path, heading, snippet, char_offset });
  });

  return refs;
};

export function useChatSessions(): UseChatSessionsResult {
  const defaultSession = useMemo(() => createEmptySession(), []);
  const [sessions, setSessions] = useState<ChatSession[]>([defaultSession]);
  const [activeSessionId, setActiveSessionId] = useState(defaultSession.id);
  const [inputValue, setInputValue] = useState('');
  const [pendingSessions, setPendingSessions] = useState<Record<string, boolean>>({});
  const [hydrated, setHydrated] = useState(false);

  const sessionsRef = useRef<ChatSession[]>([defaultSession]);
  const streamControllersRef = useRef(new Map<string, AbortController>());
  const apiBaseUrlRef = useRef<string>(typeof window === 'undefined' ? '' : getApiBaseUrl());

  const activeSession =
    sessions.find((session) => session.id === activeSessionId) ?? sessions[0] ?? defaultSession;
  const isActivePending = Boolean(activeSession && pendingSessions[activeSession.id]);
  const isAnyPending = Object.values(pendingSessions).some(Boolean);

  useEffect(() => {
    sessionsRef.current = sessions;
  }, [sessions]);

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
      const stored = localStorage.getItem(STORAGE_KEY) ?? localStorage.getItem(LEGACY_STORAGE_KEY);
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
  }, []);

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
        const existing = prev.find((item) => item.id === sessionId) ?? createEmptySession(sessionId);
        const updated = updater(existing);
        const others = prev.filter((item) => item.id !== sessionId);
        return [updated, ...others];
      });
    },
    [],
  );

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

  const parseSseStream = useCallback(
    async (
      reader: ReadableStreamDefaultReader<Uint8Array>,
      onEvent: (eventName: string, data: string) => void,
    ) => {
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let eventName = 'message';
      let dataLines: string[] = [];

      const dispatch = () => {
        if (!dataLines.length) {
          eventName = 'message';
          return;
        }
        onEvent(eventName, dataLines.join('\n'));
        eventName = 'message';
        dataLines = [];
      };

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
          let line = buffer.slice(0, newlineIndex);
          buffer = buffer.slice(newlineIndex + 1);
          if (line.endsWith('\r')) {
            line = line.slice(0, -1);
          }

          if (!line) {
            dispatch();
            newlineIndex = buffer.indexOf('\n');
            continue;
          }

          if (line.startsWith(':')) {
            newlineIndex = buffer.indexOf('\n');
            continue;
          }

          if (line.startsWith('event:')) {
            eventName = line.slice(6).trim() || 'message';
            newlineIndex = buffer.indexOf('\n');
            continue;
          }

          if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).trimStart());
          }

          newlineIndex = buffer.indexOf('\n');
        }
      }

      buffer += decoder.decode();
      if (buffer) {
        let tail = buffer;
        if (tail.endsWith('\r')) {
          tail = tail.slice(0, -1);
        }
        if (tail.startsWith('data:')) {
          dataLines.push(tail.slice(5).trimStart());
        }
      }
      dispatch();
    },
    [],
  );

  const ensureUpstreamSessionId = useCallback(
    async (sessionId: string, baseUrl: string, traceId: string): Promise<string> => {
      const existing = sessionsRef.current.find((item) => item.id === sessionId)?.upstreamSessionId;
      if (existing) {
        return existing;
      }

      const response = await fetch(`${baseUrl}${SESSION_ENDPOINT}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Request-Id': traceId,
          'X-Trace-Id': traceId,
        },
        body: JSON.stringify({ path: DEFAULT_OPENCODE_SESSION_PATH }),
      });

      if (!response.ok) {
        const detail = (await response.text()).trim();
        throw new Error(detail || `创建 OpenCode 会话失败: ${response.status}`);
      }

      const payload = (await response.json().catch(() => null)) as { id?: string } | null;
      const upstreamSessionId = payload?.id?.trim();
      if (!upstreamSessionId) {
        throw new Error('创建 OpenCode 会话失败：响应缺少 session id');
      }

      upsertSession(sessionId, (session) => ({
        ...session,
        upstreamSessionId,
      }));

      return upstreamSessionId;
    },
    [upsertSession],
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

  const clearActiveSession = useCallback(() => {
    if (!activeSession) return;
    abortSessionRequest(activeSession.id);
    upsertSession(activeSession.id, (session) => ({
      ...session,
      messages: [],
    }));
  }, [abortSessionRequest, activeSession, upsertSession]);

  const refreshSessionTitle = useCallback(async () => undefined, []);

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
        statusText: '准备连接 OpenCode...',
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

      let completionTimer: number | undefined;
      let timeoutTimer: number | undefined;
      let autoCompletedAbort = false;
      let timeoutAbort = false;
      let sawStepFinish = false;
      let sawStepStart = false;
      let currentStepMessageId: string | undefined;
      let assistantContent = '';

      const sourceRefMap = new Map<string, SourceRef>();
      const toolStatusByCall = new Map<string, ToolStatus>();
      const normalizedUserInput = content.trim();

      const clearCompletionTimer = () => {
        if (completionTimer) {
          window.clearTimeout(completionTimer);
          completionTimer = undefined;
        }
      };

      const activeToolCount = (): number => {
        let count = 0;
        toolStatusByCall.forEach((status) => {
          if (status === 'pending' || status === 'running') {
            count += 1;
          }
        });
        return count;
      };

      const buildSourceRefs = (): SourceRef[] => Array.from(sourceRefMap.values());

      const mergeSourceRefs = (refs: SourceRef[]) => {
        refs.forEach((ref) => {
          const key = [ref.path, ref.heading ?? '', ref.char_offset ?? '', ref.snippet ?? ''].join('|');
          if (!sourceRefMap.has(key)) {
            sourceRefMap.set(key, ref);
          }
        });
      };

      const scheduleCompletionAbort = () => {
        clearCompletionTimer();
        completionTimer = window.setTimeout(() => {
          autoCompletedAbort = true;
          controller.abort();
        }, 1800);
      };

      const maybeScheduleCompletion = () => {
        const normalizedAssistant = assistantContent.trim();
        const hasMeaningfulContent =
          normalizedAssistant.length > 0 && normalizedAssistant !== normalizedUserInput;
        if (sawStepFinish && activeToolCount() === 0 && hasMeaningfulContent) {
          scheduleCompletionAbort();
        }
      };

      const updateAssistantState = (overrides: Partial<ChatMessage>) => {
        updateAssistantMessage(targetSessionId, assistantPlaceholder.id, (prev) => ({
          ...prev,
          ...overrides,
          timestamp: Date.now(),
        }));
      };

      try {
        const baseUrl = apiBaseUrlRef.current || getApiBaseUrl();
        apiBaseUrlRef.current = baseUrl;
        const traceId = buildTraceId(targetSessionId);

        const upstreamSessionId = await ensureUpstreamSessionId(targetSessionId, baseUrl, traceId);

        const streamResponse = await fetch(`${baseUrl}${EVENT_ENDPOINT}`, {
          method: 'GET',
          headers: {
            Accept: 'text/event-stream',
            'X-Request-Id': traceId,
            'X-Trace-Id': traceId,
            'X-Session-Id': upstreamSessionId,
          },
          cache: 'no-store',
          signal: controller.signal,
        });

        if (!streamResponse.ok || !streamResponse.body) {
          const detail = (await streamResponse.text()).trim();
          throw new Error(detail || `连接 OpenCode 事件流失败: ${streamResponse.status}`);
        }

        updateAssistantState({
          isThinking: true,
          statusText: '已连接 OpenCode，等待响应...',
        });

        const promptResponse = await fetch(`${baseUrl}${sessionMessageEndpoint(upstreamSessionId)}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Request-Id': traceId,
            'X-Trace-Id': traceId,
          },
          body: JSON.stringify({
            parts: [{ type: 'text', text: content }],
          }),
          signal: controller.signal,
        });

        if (!promptResponse.ok) {
          const detail = (await promptResponse.text()).trim();
          throw new Error(detail || `发送消息失败: ${promptResponse.status}`);
        }

        updateAssistantState({
          isThinking: true,
          statusText: '正在思考...',
        });

        timeoutTimer = window.setTimeout(() => {
          timeoutAbort = true;
          controller.abort();
        }, 240000);

        const reader = streamResponse.body.getReader();
        await parseSseStream(reader, (eventName, data) => {
          let parsedValue: unknown;
          try {
            parsedValue = JSON.parse(data);
          } catch {
            return;
          }
          const parsed = asRecord(parsedValue);
          if (!parsed) {
            return;
          }

          const eventType = asString(parsed.type) ?? eventName;
          if (eventType !== 'message.part.updated') {
            return;
          }

          const properties = asRecord(parsed.properties);
          const part = asRecord(properties?.part ?? parsed.part);
          if (!part) {
            return;
          }

          const partSessionId = asString(part.sessionID);
          if (!partSessionId || partSessionId !== upstreamSessionId) {
            return;
          }

          const partType = asString(part.type);
          const partMessageId = asString(part.messageID);
          if (!partType) {
            return;
          }

          if (partType === 'text') {
            if (!sawStepStart) {
              return;
            }

            if (currentStepMessageId && partMessageId && partMessageId !== currentStepMessageId) {
              return;
            }

            const delta = asString(properties?.delta ?? parsed.delta);
            const partText = asString(part.text);
            if (delta && delta.length > 0) {
              assistantContent += delta;
            } else if (partText !== undefined && partText.length > 0) {
              assistantContent = partText;
            }

            clearCompletionTimer();
            updateAssistantState({
              content: assistantContent,
              isThinking: true,
              statusText: '',
              sourceRefs: buildSourceRefs(),
            });
            return;
          }

          if (partType === 'tool') {
            const toolName = asString(part.tool) ?? 'tool';
            const callId = asString(part.callID) ?? `${toolName}-${toolStatusByCall.size}`;
            const state = asRecord(part.state);
            const status = asString(state?.status) as ToolStatus | undefined;

            if (!status) {
              return;
            }

            toolStatusByCall.set(callId, status);

            if (status === 'pending' || status === 'running') {
              clearCompletionTimer();
              updateAssistantState({
                isThinking: true,
                statusText: `正在调用工具：${toolName}`,
              });
              return;
            }

            if (status === 'completed') {
              mergeSourceRefs(extractSourceRefs(state?.metadata));
              updateAssistantState({
                isThinking: true,
                statusText: activeToolCount() > 0 ? '等待其他工具完成...' : '',
                sourceRefs: buildSourceRefs(),
              });
              maybeScheduleCompletion();
              return;
            }

            if (status === 'error') {
              const message = asString(state?.error) ?? '工具执行失败';
              updateAssistantState({
                isThinking: true,
                statusText: `${toolName} 失败：${message}`,
              });
              maybeScheduleCompletion();
            }
            return;
          }

          if (partType === 'step-start') {
            sawStepStart = true;
            if (partMessageId) {
              currentStepMessageId = partMessageId;
            }
            clearCompletionTimer();
            updateAssistantState({
              isThinking: true,
              statusText: assistantContent.trim() ? '' : '正在思考...',
            });
            return;
          }

          if (partType === 'step-finish') {
            if (currentStepMessageId && partMessageId && partMessageId !== currentStepMessageId) {
              return;
            }
            sawStepFinish = true;
            maybeScheduleCompletion();
          }
        });

        const finalText = assistantContent.trim() || '助手暂时没有回复。';
        const normalizedFinal = finalText.trim();
        const safeFinalText =
          normalizedFinal && normalizedFinal !== normalizedUserInput
            ? finalText
            : '本次请求未产出可展示的最终回复，请重试或缩小问题范围。';
        updateAssistantState({
          content: safeFinalText,
          isThinking: false,
          statusText: '',
          sourceRefs: buildSourceRefs(),
        });
      } catch (error) {
        if ((error as DOMException)?.name === 'AbortError') {
          if (autoCompletedAbort) {
            const finalText = assistantContent.trim();
            const safeFinalText =
              finalText && finalText !== normalizedUserInput
                ? finalText
                : '本次请求未产出可展示的最终回复，请重试或缩小问题范围。';
            updateAssistantState({
              content: safeFinalText,
              isThinking: false,
              statusText: '',
              sourceRefs: buildSourceRefs(),
            });
            return;
          }

          if (timeoutAbort) {
            const normalizedAssistant = assistantContent.trim();
            const timeoutContent =
              normalizedAssistant && normalizedAssistant !== normalizedUserInput
                ? normalizedAssistant
                : '响应超时，请重试（建议缩小问题范围或分步提问）。';
            updateAssistantState({
              content: timeoutContent,
              isThinking: false,
              isError: true,
              statusText: '',
              sourceRefs: buildSourceRefs(),
            });
            return;
          }

          updateAssistantState({
            content: assistantContent || '（请求已取消）',
            isThinking: false,
            statusText: '',
            sourceRefs: buildSourceRefs(),
          });
          return;
        }

        const errorText = error instanceof Error ? error.message : '发生未知错误';
        updateAssistantState({
          content: errorText,
          isThinking: false,
          isError: true,
          statusText: '',
          sourceRefs: buildSourceRefs(),
        });
      } finally {
        if (completionTimer) {
          window.clearTimeout(completionTimer);
        }
        if (timeoutTimer) {
          window.clearTimeout(timeoutTimer);
        }
        const storedController = streamControllersRef.current.get(targetSessionId);
        if (storedController === controller) {
          streamControllersRef.current.delete(targetSessionId);
        }
        setSessionPending(targetSessionId, false);
      }
    },
    [
      activeSession,
      ensureUpstreamSessionId,
      inputValue,
      parseSseStream,
      pendingSessions,
      setSessionPending,
      updateAssistantMessage,
      upsertSession,
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
