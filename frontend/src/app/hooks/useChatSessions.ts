import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  EVENT_ENDPOINT,
  getApiBaseUrl,
  type ChatMessage,
  type ChatSession,
  type ThinkingStep,
  SESSION_ENDPOINT,
  sessionMessageEndpoint,
  type SourceRef,
  STORAGE_KEY,
  // Answer-first 新增类型
  type DecisionSummary,
  type ProcessOverview,
  type CompletionState,
  type EvidenceItem,
  type RunPhase,
  type CitationRef,
  type ProcessStepSummary,
  type EvidenceRef,
  // 证据与透明性新增类型
  type HonestySignals,
} from '../lib/chat-types';
import {
  computeHonestySignals,
  enrichCitationsWithEvidence,
  extractCitations,
  generateProcessSummary,
  splitDirectAnswer,
  type ToolCallRecord,
} from '../api/_lib/event-adapter';
import {
  deriveSourceTitle,
  inferSourceDateLabel,
  isWeakRetrievalScore,
  normalizeHonestySignalsWithReferences,
  sanitizeCitationSnippet,
} from '../lib/citation-utils';
import { createEmptySession, createId, deriveTitle } from '../lib/chat-helpers';

interface UseChatSessionsResult {
  sessions: ChatSession[];
  activeSession?: ChatSession;
  activeSessionId: string;
  setActiveSessionId: (id: string) => void;
  inputValue: string;
  setInputValue: (value: string) => void;
  hydrated: boolean;
  isActivePending: boolean;
  createNewSession: () => void;
  deleteSession: (sessionId: string) => void;
  renameSession: (sessionId: string, value: string) => void;
  handleSubmit: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  abortSessionRequest: (sessionId: string) => void;
}

type JsonRecord = Record<string, unknown>;
type ToolStatus = 'pending' | 'running' | 'completed' | 'error';

const LEGACY_STORAGE_KEY = 'second_brain_sessions_v1';
const DEFAULT_OPENCODE_SESSION_PATH =
  process.env.NEXT_PUBLIC_OPENCODE_SESSION_PATH?.trim() || '/app';
const FINAL_EVENT_ABORT_DELAY_MS = 3500;
const CHAT_STREAM_ENDPOINT = '/api/chat/stream/';

const asRecord = (value: unknown): JsonRecord | undefined =>
  value && typeof value === 'object' ? (value as JsonRecord) : undefined;

const asString = (value: unknown): string | undefined =>
  typeof value === 'string' ? value : undefined;

const getToolArguments = (state: JsonRecord | undefined): Record<string, unknown> | undefined => {
  if (!state) {
    return undefined;
  }
  return asRecord(state.arguments) || asRecord(state.input);
};

const getToolResult = (state: JsonRecord | undefined): unknown => state?.result ?? state?.output;

const parseDecisionSummary = (value: unknown): DecisionSummary | undefined => {
  const record = asRecord(value);
  const conclusion = asString(record?.conclusion);
  if (conclusion === undefined) {
    return undefined;
  }

  return {
    conclusion,
    actions: Array.isArray(record?.actions)
      ? record.actions.filter((item): item is string => typeof item === 'string')
      : [],
    confidence: (['high', 'medium', 'low', 'unknown'].includes(asString(record?.confidence) || '')
      ? record?.confidence
      : 'unknown') as DecisionSummary['confidence'],
    assumptions: Array.isArray(record?.assumptions)
      ? record.assumptions.filter((item): item is string => typeof item === 'string')
      : [],
    risks: Array.isArray(record?.risks)
      ? record.risks.filter((item): item is string => typeof item === 'string')
      : [],
    failureReason: asString(record?.failureReason),
  };
};

const parseProcessOverview = (value: unknown): ProcessOverview | undefined => {
  const record = asRecord(value);
  const phase = asString(record?.phase);
  if (!phase || !['retrieving', 'validating', 'synthesizing', 'completed'].includes(phase)) {
    return undefined;
  }

  return {
    phase: phase as RunPhase,
    durationMs: typeof record?.durationMs === 'number' ? record.durationMs : 0,
    warningCount: typeof record?.warningCount === 'number' ? record.warningCount : 0,
    blockingErrorCount: typeof record?.blockingErrorCount === 'number' ? record.blockingErrorCount : 0,
    impact: (['none', 'partial', 'blocking'].includes(asString(record?.impact) || '')
      ? record?.impact
      : 'none') as ProcessOverview['impact'],
  };
};

const parseCompletionState = (value: unknown): CompletionState | undefined => {
  if (typeof value !== 'string') {
    return undefined;
  }
  return ['completed', 'partial_completed', 'failed'].includes(value)
    ? (value as CompletionState)
    : undefined;
};

const parseReferences = (value: unknown): CitationRef[] | undefined => {
  if (!Array.isArray(value)) {
    return undefined;
  }
  return value.filter((item): item is CitationRef => {
    const record = asRecord(item);
    return Boolean(record && typeof record.id === 'string' && typeof record.sourcePath === 'string');
  });
};

const parseProcessSummary = (value: unknown): ProcessStepSummary[] | undefined =>
  Array.isArray(value) ? (value as ProcessStepSummary[]) : undefined;

const buildCitationMap = (references?: CitationRef[]): Record<string, CitationRef> => {
  const citationMap: Record<string, CitationRef> = {};
  references?.forEach((ref) => {
    citationMap[ref.id] = ref;
  });
  return citationMap;
};

const splitFallbackAnswer = (content: string): { directAnswer: string; fullAnalysis: string } => {
  const normalized = content.trim();
  if (!normalized) {
    return { directAnswer: '', fullAnalysis: '' };
  }

  const separatorMatch = normalized.match(/\n---\s*\n/);
  if (separatorMatch) {
    return {
      directAnswer: normalized.slice(0, separatorMatch.index).trim(),
      fullAnalysis: normalized.slice(separatorMatch.index! + separatorMatch[0].length).trim(),
    };
  }

  const paragraphs = normalized.split(/\n\s*\n/).map((part) => part.trim()).filter(Boolean);
  if (paragraphs.length <= 1) {
    return { directAnswer: normalized, fullAnalysis: normalized };
  }

  return {
    directAnswer: paragraphs[0],
    fullAnalysis: normalized,
  };
};

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

const extractEvidenceSourceRefs = (metadata: unknown): EvidenceRef[] => {
  const record = asRecord(metadata);
  const raw = record?.source_refs;
  if (!Array.isArray(raw)) {
    return [];
  }

  const refs: EvidenceRef[] = [];
  raw.forEach((item) => {
    const entry = asRecord(item);
    const sourcePath = asString(entry?.path)?.trim();
    if (!sourcePath) {
      return;
    }

    const heading = asString(entry?.heading)?.trim();
    const snippet = sanitizeCitationSnippet(asString(entry?.snippet));
    const sourceTitle = asString(entry?.source_title)?.trim();
    const charOffset = typeof entry?.char_offset === 'number' ? entry.char_offset : undefined;
    const score = typeof entry?.score === 'number' ? entry.score : undefined;
    const citationId = asString(entry?.citation_id)?.trim();

    refs.push({
      sourcePath,
      sourceTitle: sourceTitle || deriveSourceTitle(sourcePath, sourceTitle, heading),
      sourceDateLabel: inferSourceDateLabel(sourcePath, sourceTitle, heading),
      heading,
      snippet,
      charOffsetStart: charOffset,
      retrievalScore: score,
      citationId,
    });
  });

  return refs;
};

const stripUnresolvedCitations = (
  content: string,
  evidenceRefs: EvidenceRef[],
): string => {
  if (!content) {
    return content;
  }

  const resolvedIds = new Set<string>();
  evidenceRefs.forEach((ref) => {
    const citationId = ref.citationId?.trim();
    if (!citationId) {
      return;
    }
    const normalized = citationId.startsWith('c') ? citationId.slice(1) : citationId;
    resolvedIds.add(normalized);
    resolvedIds.add(`c${normalized}`);
  });

  return content.replace(/\[c(\d{2,3})\]/g, (match, id: string) =>
    resolvedIds.has(id) || resolvedIds.has(`c${id}`) ? match : '',
  );
};

const hasExplicitNoHitStatement = (content: string): boolean =>
  /没有关于.*直接记录|没有相关记录|未找到.*记录|没有检索到.*记录/.test(content);

const didAllRetrievalStepsMiss = (steps: ProcessStepSummary[]): boolean => {
  const retrievalSteps = steps.filter((step) => step.phase === 'retrieving');
  if (retrievalSteps.length === 0) {
    return false;
  }

  return retrievalSteps.every((step) => {
    const haystack = [step.summary, step.detail, step.resultSummary].filter(Boolean).join(' ');
    return /命中\s*0\s*条/.test(haystack);
  });
};

const buildNoHitAnswer = (hasWeakReferences: boolean): string =>
  hasWeakReferences
    ? '你的笔记中没有关于该问题的直接记录。当前只检索到弱相关内容，不能据此给出可靠结论，请优先核对下方原文引用。'
    : '你的笔记中没有关于该问题的直接记录，目前也没有可支撑回答的相关引用。';

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

  const parseNdjsonStream = useCallback(
    async (
      reader: ReadableStreamDefaultReader<Uint8Array>,
      onEvent: (event: JsonRecord) => void,
    ): Promise<void> => {
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        let newlineIndex = buffer.indexOf('\n');
        while (newlineIndex !== -1) {
          const rawLine = buffer.slice(0, newlineIndex);
          buffer = buffer.slice(newlineIndex + 1);
          const line = rawLine.trim();
          if (line) {
            try {
              const parsed = JSON.parse(line);
              const record = asRecord(parsed);
              if (record) {
                onEvent(record);
              }
            } catch {
              // Ignore malformed partial lines from upstream.
            }
          }
          newlineIndex = buffer.indexOf('\n');
        }
      }

      const trailing = buffer.trim();
      if (!trailing) {
        return;
      }

      try {
        const parsed = JSON.parse(trailing);
        const record = asRecord(parsed);
        if (record) {
          onEvent(record);
        }
      } catch {
        // Ignore malformed trailing data.
      }
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
      let sawStepStart = false;
      let currentStepMessageId: string | undefined;
      let assistantContent = '';

      const sourceRefMap = new Map<string, SourceRef>();
      const evidenceSourceRefMap = new Map<string, EvidenceRef>();
      const toolStatusByCall = new Map<string, ToolStatus>();
      const normalizedUserInput = content.trim();
      let finalizedEventVersion: number | undefined;
      let decisionSummary: DecisionSummary | undefined;
      let processOverview: ProcessOverview | undefined;
      let completionState: CompletionState | undefined;
      let evidence: EvidenceItem[] | undefined;

      // 步骤管理
      const steps: ThinkingStep[] = [];
      let currentStepId: string | undefined;

      const addStep = (step: ThinkingStep) => {
        steps.push(step);
        currentStepId = step.id;
        updateAssistantState({
          thinkingSteps: [...steps],
          currentStepId,
        });
      };

      const updateStep = (stepId: string, updates: Partial<ThinkingStep>) => {
        const index = steps.findIndex(s => s.id === stepId);
        if (index !== -1) {
          steps[index] = { ...steps[index], ...updates };
          updateAssistantState({
            thinkingSteps: [...steps],
          });
        }
      };

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

      const mergeEvidenceSourceRefs = (refs: EvidenceRef[]) => {
        refs.forEach((ref) => {
          const key = [
            ref.sourcePath,
            ref.heading ?? '',
            ref.charOffsetStart ?? '',
            ref.citationId ?? '',
          ].join('|');
          if (!evidenceSourceRefMap.has(key)) {
            evidenceSourceRefMap.set(key, ref);
          }
        });
      };

      const scheduleCompletionAbort = () => {
        clearCompletionTimer();
        completionTimer = window.setTimeout(() => {
          autoCompletedAbort = true;
          controller.abort();
        }, FINAL_EVENT_ABORT_DELAY_MS);
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

        const sessionSnapshot =
          sessionsRef.current.find((session) => session.id === targetSessionId) ?? activeSession;
        const requestMessages = [...(sessionSnapshot?.messages ?? []), userMessage]
          .filter((message) => message.id !== assistantPlaceholder.id)
          .map((message) => ({
            role: message.role,
            content: message.content,
          }));

        if (requestMessages.length === 0) {
          requestMessages.push({ role: 'user', content });
        }

        const streamResponse = await fetch(`${baseUrl}${CHAT_STREAM_ENDPOINT}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/x-ndjson',
            'X-Stream-Format': 'ndjson',
            'X-Request-Id': traceId,
            'X-Trace-Id': traceId,
          },
          body: JSON.stringify({
            messages: requestMessages,
          }),
          signal: controller.signal,
        });

        if (!streamResponse.ok || !streamResponse.body) {
          const detail = (await streamResponse.text()).trim();
          throw new Error(detail || `连接后端流式接口失败: ${streamResponse.status}`);
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

        await parseNdjsonStream(reader, (parsed) => {
          const eventType = asString(parsed.type);
          if (!eventType) {
            return;
          }

          clearCompletionTimer();
          autoCompletedAbort = false;

          if (eventType === 'delta') {
            const delta = asString(parsed.delta);
            if (!delta) {
              return;
            }
            assistantContent += delta;
            updateAssistantState({
              content: assistantContent,
              isThinking: true,
              statusText: '',
              sourceRefs: buildSourceRefs(),
            });
            return;
          }

          if (eventType === 'status') {
            const message = asString(parsed.message) || '正在思考...';
            updateAssistantState({
              isThinking: true,
              statusText: message,
            });
            if (!steps.some((step) => step.type === 'thought')) {
              addStep({
                id: createId(),
                type: 'thought',
                content: message,
                timestamp: Date.now(),
              });
            }
            return;
          }

          if (eventType === 'tool') {
            const toolName = asString(parsed.tool_name) ?? 'tool';
            const callId = asString(parsed.tool_call_id) ?? `${toolName}-${toolStatusByCall.size}`;
            const stage = asString(parsed.stage);
            const toolArguments = asRecord(parsed.arguments);
            const status: ToolStatus =
              stage === 'start'
                ? 'running'
                : stage === 'end'
                  ? 'completed'
                  : 'error';

            toolStatusByCall.set(callId, status);
            let toolStep = steps.find((step) => step.tool?.id === callId);

            if (!toolStep) {
              const stepId = createId();
              toolStep = {
                id: stepId,
                type: 'tool',
                tool: {
                  id: callId,
                  name: toolName,
                  status,
                  arguments: toolArguments,
                  startedAt: Date.now(),
                },
                timestamp: Date.now(),
              };
              addStep(toolStep);
            } else {
              updateStep(toolStep.id, {
                tool: {
                  ...toolStep.tool!,
                  status,
                  arguments: toolArguments ?? toolStep.tool?.arguments,
                  error: status === 'error' ? asString(parsed.error) ?? toolStep.tool?.error : toolStep.tool?.error,
                  completedAt: status === 'running' ? toolStep.tool?.completedAt : Date.now(),
                },
              });
            }

            updateAssistantState({
              isThinking: true,
              statusText:
                asString(parsed.message) ||
                (status === 'error' ? `${toolName} 失败` : `正在执行：${toolName}`),
            });
            return;
          }

          if (eventType === 'sources') {
            const previewRefs = extractSourceRefs(parsed);
            const evidenceRefs = extractEvidenceSourceRefs(parsed);
            mergeSourceRefs(previewRefs);
            mergeEvidenceSourceRefs(evidenceRefs);
            updateAssistantState({
              sourceRefs: buildSourceRefs(),
            });
            return;
          }
        });

        const filteredAssistantContent = stripUnresolvedCitations(
          assistantContent,
          Array.from(evidenceSourceRefMap.values()),
        );
        const finalText = filteredAssistantContent.trim() || '助手暂时没有回复。';
        const normalizedFinal = finalText.trim();
        const safeFinalText =
          normalizedFinal && normalizedFinal !== normalizedUserInput
            ? finalText
            : '本次请求未产出可展示的最终回复，请重试或缩小问题范围。';

        const evidenceRefs = Array.from(evidenceSourceRefMap.values());
        const evidenceMap = new Map<string, EvidenceRef>();
        evidenceRefs.forEach((ref) => {
          const key = [
            ref.sourcePath,
            ref.heading ?? '',
            ref.charOffsetStart ?? '',
            ref.citationId ?? '',
          ].join('|');
          evidenceMap.set(key, ref);
        });

        const references = enrichCitationsWithEvidence(
          extractCitations(safeFinalText),
          evidenceMap,
        ).filter((ref) => Boolean(ref.sourcePath));

        const fallbackReferences = evidenceRefs
          .filter((ref) => Boolean(ref.sourcePath))
          .map((ref, index) => ({
            id: String(index + 1).padStart(2, '0'),
            sourcePath: ref.sourcePath,
            sourceTitle: ref.sourceTitle || deriveSourceTitle(ref.sourcePath, ref.sourceTitle, ref.heading),
            sourceDateLabel: ref.sourceDateLabel || inferSourceDateLabel(ref.sourcePath, ref.sourceTitle, ref.heading),
            heading: ref.heading,
            charOffsetStart: ref.charOffsetStart,
            snippet: sanitizeCitationSnippet(ref.snippet),
            retrievalScore: ref.retrievalScore,
          }));

        const resolvedReferences = references.length > 0 ? references : fallbackReferences;

        const completedCalls: ToolCallRecord[] = [];
        const errorCalls: ToolCallRecord[] = [];
        const activeCalls: ToolCallRecord[] = [];
        steps
          .filter((step) => step.type === 'tool' && step.tool)
          .forEach((step) => {
            const tool = step.tool!;
            const toolRecord = {
              id: tool.id,
              name: tool.name,
              status: tool.status,
              arguments: tool.arguments,
              result: tool.result,
              error: tool.error,
              startedAt: tool.startedAt ?? step.timestamp,
              completedAt: tool.completedAt,
              sourceRefs: evidenceRefs.filter((ref) => Boolean(ref.sourcePath)),
            };
            if (tool.status === 'completed') {
              completedCalls.push(toolRecord);
            } else if (tool.status === 'error') {
              errorCalls.push(toolRecord);
            } else {
              activeCalls.push(toolRecord);
            }
          });

        const generatedProcessSummary = generateProcessSummary(
          completedCalls,
          errorCalls,
          activeCalls,
        );
        const computedHonestySignals = normalizeHonestySignalsWithReferences(
          computeHonestySignals(resolvedReferences, errorCalls.length > 0, errorCalls.length),
          resolvedReferences,
        );
        const allRetrievalStepsMissed = didAllRetrievalStepsMiss(generatedProcessSummary);
        const hasWeakReferences = resolvedReferences.some((ref) =>
          isWeakRetrievalScore(ref.retrievalScore),
        );
        const onlyWeakReferences =
          resolvedReferences.length > 0 &&
          resolvedReferences.every((ref) => isWeakRetrievalScore(ref.retrievalScore));
        const shouldForceNoHitAnswer =
          allRetrievalStepsMissed &&
          (resolvedReferences.length === 0 || onlyWeakReferences) &&
          !hasExplicitNoHitStatement(safeFinalText);
        const finalAnswerText = shouldForceNoHitAnswer
          ? buildNoHitAnswer(hasWeakReferences)
          : safeFinalText;
        const answerParts = splitDirectAnswer(finalAnswerText);

        decisionSummary = {
          conclusion: (answerParts.directAnswer || finalAnswerText).slice(0, 200),
          actions: [],
          confidence: computedHonestySignals?.hasSufficientEvidence ? 'high' : 'unknown',
          assumptions: [],
          risks: computedHonestySignals?.honestyWarnings ?? [],
        };
        processOverview = {
          phase: 'completed',
          durationMs: Date.now() - assistantPlaceholder.timestamp!,
          warningCount: errorCalls.length,
          blockingErrorCount: 0,
          impact: errorCalls.length > 0 ? 'partial' : 'none',
        };
        completionState = 'completed';
        evidence = [];

        updateAssistantState({
          content: finalAnswerText,
          isThinking: false,
          statusText: '',
          sourceRefs: buildSourceRefs(),
          directAnswer: answerParts.directAnswer || finalAnswerText,
          fullAnalysis: answerParts.fullAnalysis || finalAnswerText,
          references: resolvedReferences.length > 0 ? resolvedReferences : undefined,
          citationMap: resolvedReferences.length > 0 ? buildCitationMap(resolvedReferences) : undefined,
          processSummary: generatedProcessSummary.length > 0 ? generatedProcessSummary : undefined,
          honestySignals:
            computedHonestySignals && !computedHonestySignals.hasSufficientEvidence
              ? computedHonestySignals
              : undefined,
          decisionSummary,
          processOverview,
          completionState,
          evidence,
        });
      } catch (error) {
        if ((error as DOMException)?.name === 'AbortError') {
	          if (autoCompletedAbort) {
	            const finalText = assistantContent.trim();
	            const safeFinalText =
	              finalText && finalText !== normalizedUserInput
	                ? finalText
	                : '本次请求未产出可展示的最终回复，请重试或缩小问题范围。';
	            const finalState: Partial<ChatMessage> = {
	              content: safeFinalText,
	              isThinking: false,
	              statusText: '',
	              sourceRefs: buildSourceRefs(),
	            };
              if (finalizedEventVersion === undefined) {
	              const fallbackAnswer = splitFallbackAnswer(safeFinalText);
                finalState.directAnswer = fallbackAnswer.directAnswer || safeFinalText;
                finalState.fullAnalysis = fallbackAnswer.fullAnalysis || safeFinalText;
              }
	            updateAssistantState(finalState);
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
      inputValue,
      parseNdjsonStream,
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
    hydrated,
    isActivePending,
    createNewSession,
    deleteSession,
    renameSession,
    handleSubmit,
    abortSessionRequest,
  };
}
