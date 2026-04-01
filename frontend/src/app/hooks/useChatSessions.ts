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
  type DisplayState,
  type EvidenceItem,
  type RunPhase,
  type CitationRef,
  type ProcessStepSummary,
  // 证据与透明性新增类型
  type HonestySignals,
} from '../lib/chat-types';
import {
  createEmptySession,
  createId,
  deriveTitle,
  getUserFacingAssistantStatusText,
  shouldIgnoreComposerSubmitAfterAbort,
} from '../lib/chat-helpers';
import { isPreciseCitationRef, normalizeCitationId } from '../lib/citation-utils';

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
interface KnownPartMeta {
  type?: string;
  messageID?: string;
  sessionID?: string;
}

const LEGACY_STORAGE_KEY = 'second_brain_sessions_v1';
const DEFAULT_OPENCODE_SESSION_PATH =
  process.env.NEXT_PUBLIC_OPENCODE_SESSION_PATH?.trim() || '/app';
const FINAL_EVENT_ABORT_DELAY_MS = 3500;
const LONG_RUNNING_THRESHOLD_MS = 5000;

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
  return value.flatMap((item) => {
    const record = asRecord(item);
    if (!record || typeof record.id !== 'string' || typeof record.sourcePath !== 'string') {
      return [];
    }
    return [{
      ...(record as unknown as CitationRef),
      kind: record.kind === 'precise' || record.kind === 'file' ? record.kind : 'file',
      provenance:
        record.provenance === 'native' ||
        record.provenance === 'synthetic_read' ||
        record.provenance === 'content_path'
          ? record.provenance
          : 'content_path',
    }];
  });
};

const parseProcessSummary = (value: unknown): ProcessStepSummary[] | undefined =>
  Array.isArray(value) ? (value as ProcessStepSummary[]) : undefined;

const buildCitationMap = (references?: CitationRef[]): Record<string, CitationRef> => {
  const citationMap: Record<string, CitationRef> = {};
  references?.forEach((ref) => {
    if (!isPreciseCitationRef(ref)) {
      return;
    }
    citationMap[ref.id] = ref;
    const normalized = normalizeCitationId(ref.id);
    citationMap[normalized] = ref;
    citationMap[`c${normalized}`] = ref;
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
  const lastAbortAtRef = useRef(0);

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
      lastAbortAtRef.current = Date.now();
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

  const handleSubmit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (shouldIgnoreComposerSubmitAfterAbort(lastAbortAtRef.current)) {
        return;
      }
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
        statusText: '正在准备回答',
        displayState: 'running',
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
      let longRunningTimer: number | undefined;
      let timeoutTimer: number | undefined;
      let autoCompletedAbort = false;
      let timeoutAbort = false;
      let sawStepStart = false;
      let currentStepMessageId: string | undefined;
      let assistantContent = '';
      let displayState: DisplayState = 'running';

      const sourceRefMap = new Map<string, SourceRef>();
      const toolStatusByCall = new Map<string, ToolStatus>();
      const normalizedUserInput = content.trim();
      let finalizedEventVersion: number | undefined;
      let decisionSummary: DecisionSummary | undefined;
      let processOverview: ProcessOverview | undefined;
      let completionState: CompletionState | undefined;
      let evidence: EvidenceItem[] | undefined;
      const knownParts = new Map<string, KnownPartMeta>();

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

      const clearLongRunningTimer = () => {
        if (longRunningTimer) {
          window.clearTimeout(longRunningTimer);
          longRunningTimer = undefined;
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
        }, FINAL_EVENT_ABORT_DELAY_MS);
      };

      const getUserFacingStatusText = (options?: {
        forceLongRunning?: boolean;
        isToolWork?: boolean;
      }): string =>
        getUserFacingAssistantStatusText({
          assistantContent,
          displayState,
          forceLongRunning: options?.forceLongRunning,
          isToolWork: options?.isToolWork,
        });

      const scheduleLongRunningState = () => {
        clearLongRunningTimer();
        longRunningTimer = window.setTimeout(() => {
          displayState = 'long_running';
          updateAssistantState({
            displayState,
            statusText: getUserFacingStatusText({ forceLongRunning: true }),
          });
        }, LONG_RUNNING_THRESHOLD_MS);
      };

      const updateAssistantState = (overrides: Partial<ChatMessage>) => {
        updateAssistantMessage(targetSessionId, assistantPlaceholder.id, (prev) => ({
          ...prev,
          ...overrides,
          displayState: overrides.displayState ?? prev.displayState,
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
          displayState,
          statusText: '正在准备回答',
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
          displayState,
          statusText: getUserFacingStatusText(),
        });

        scheduleLongRunningState();
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

          if (eventName === 'final') {
            const eventVersion = typeof parsed.event_version === 'number' ? parsed.event_version : 0;
            if (finalizedEventVersion !== undefined && eventVersion <= finalizedEventVersion) {
              return;
            }

            finalizedEventVersion = eventVersion;

            const parsedDecisionSummary = parseDecisionSummary(parsed.decisionSummary);
            const parsedProcessOverview = parseProcessOverview(parsed.processOverview);
            const parsedCompletionState = parseCompletionState(parsed.completionState);
            const parsedEvidence = Array.isArray(parsed.evidence)
              ? (parsed.evidence as EvidenceItem[])
              : undefined;
            const directAnswer = asString(parsed.directAnswer);
            const fullAnalysis = asString(parsed.fullAnalysis);
            const references = parseReferences(parsed.references);
            const processSummary = parseProcessSummary(parsed.processSummary);
            const honestySignals =
              parsed.honestySignals && typeof parsed.honestySignals === 'object'
                ? (parsed.honestySignals as HonestySignals)
                : undefined;

            decisionSummary = parsedDecisionSummary ?? decisionSummary;
            processOverview = parsedProcessOverview ?? processOverview;
            completionState = parsedCompletionState ?? completionState;
            evidence = parsedEvidence ?? evidence;
            displayState = parsedCompletionState ?? 'completed';

            updateAssistantState({
              isThinking: false,
              statusText: '',
              displayState: parsedCompletionState ?? 'completed',
              sourceRefs: buildSourceRefs(),
              finalizedEventVersion,
              ...(parsedDecisionSummary ? { decisionSummary: parsedDecisionSummary } : {}),
              ...(parsedProcessOverview ? { processOverview: parsedProcessOverview } : {}),
              ...(parsedCompletionState ? { completionState: parsedCompletionState } : {}),
              ...(parsedEvidence ? { evidence: parsedEvidence } : {}),
              ...(directAnswer ? { directAnswer } : {}),
              ...(fullAnalysis ? { fullAnalysis } : {}),
              ...(references ? { references, citationMap: buildCitationMap(references) } : {}),
              ...(processSummary ? { processSummary } : {}),
              ...(honestySignals ? { honestySignals } : {}),
            });
            scheduleCompletionAbort();
            return;
          }

          const eventType = asString(parsed.type) ?? eventName;
          const properties = asRecord(parsed.properties);
          if (eventType === 'message.part.delta') {
            const partId = asString(properties?.partID);
            const delta = asString(properties?.delta);
            const field = asString(properties?.field);
            const meta = partId ? knownParts.get(partId) : undefined;
            const partType = meta?.type;
            const partMessageId = asString(properties?.messageID) ?? meta?.messageID;
            const partSessionId = asString(properties?.sessionID) ?? meta?.sessionID;

            if (!partSessionId || partSessionId !== upstreamSessionId) {
              return;
            }

            if (field !== 'text' || !delta || partType !== 'text') {
              return;
            }

            if (!sawStepStart) {
              return;
            }

            if (currentStepMessageId && partMessageId && partMessageId !== currentStepMessageId) {
              return;
            }

            clearCompletionTimer();
            clearLongRunningTimer();
            autoCompletedAbort = false;
            assistantContent += delta;
            updateAssistantState({
              content: assistantContent,
              isThinking: true,
              displayState,
              statusText: getUserFacingStatusText(),
              sourceRefs: buildSourceRefs(),
            });
            return;
          }

          if (eventType !== 'message.part.updated') {
            return;
          }

          clearCompletionTimer();
          autoCompletedAbort = false;

          const part = asRecord(properties?.part ?? parsed.part);
          if (!part) {
            return;
          }

          const partId = asString(part.id);
          const partSessionId = asString(part.sessionID);
          const partMessageId = asString(part.messageID);
          const partType = asString(part.type);
          if (!partSessionId || partSessionId !== upstreamSessionId || !partType) {
            return;
          }

          if (partId) {
            knownParts.set(partId, {
              type: partType,
              messageID: partMessageId,
              sessionID: partSessionId,
            });
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

            updateAssistantState({
              content: assistantContent,
              isThinking: true,
              displayState,
              statusText: getUserFacingStatusText(),
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

            let toolStep = steps.find((step) => step.tool?.id === callId);
            const toolArguments = getToolArguments(state);
            const toolResult = getToolResult(state);

            if (status === 'pending' || status === 'running') {
              updateAssistantState({
                isThinking: true,
                displayState,
                statusText: getUserFacingStatusText({ isToolWork: true }),
              });

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
                  },
                });
              }
              return;
            }

            if (status === 'completed') {
              mergeSourceRefs(extractSourceRefs(state?.metadata));
              updateAssistantState({
                isThinking: true,
                displayState,
                statusText: activeToolCount() > 0 ? getUserFacingStatusText({ isToolWork: true }) : getUserFacingStatusText(),
                sourceRefs: buildSourceRefs(),
              });

              if (toolStep) {
                updateStep(toolStep.id, {
                  tool: {
                    ...toolStep.tool!,
                    status: 'completed',
                    arguments: toolArguments ?? toolStep.tool?.arguments,
                    result: toolResult,
                    completedAt: Date.now(),
                  },
                });
              }
              return;
            }

            if (status === 'error') {
              const message = asString(state?.error) ?? '工具执行失败';
              updateAssistantState({
                isThinking: true,
                displayState,
                statusText: '正在整理依据',
              });

              if (toolStep) {
                updateStep(toolStep.id, {
                  tool: {
                    ...toolStep.tool!,
                    status: 'error',
                    arguments: toolArguments ?? toolStep.tool?.arguments,
                    error: message,
                    completedAt: Date.now(),
                  },
                });
              }
            }
            return;
          }

          if (partType === 'step-start') {
            sawStepStart = true;
            if (partMessageId) {
              currentStepMessageId = partMessageId;
            }
            updateAssistantState({
              isThinking: true,
              displayState,
              statusText: getUserFacingStatusText(),
            });

            const stepId = asString(part.stepID) || createId();
            addStep({
              id: stepId,
              type: 'thought',
              content: asString(part.content) || '正在思考...',
              timestamp: Date.now(),
            });
            return;
          }

          if (partType === 'step-finish') {
            if (currentStepMessageId && partMessageId && partMessageId !== currentStepMessageId) {
              return;
            }
          }
        });

        const finalText = assistantContent.trim() || '助手暂时没有回复。';
        const normalizedFinal = finalText.trim();
        const safeFinalText =
          normalizedFinal && normalizedFinal !== normalizedUserInput
            ? finalText
            : '本次请求未产出可展示的最终回复，请重试或缩小问题范围。';

        const finalUpdate: Partial<ChatMessage> = {
          content: safeFinalText,
          isThinking: false,
          statusText: '',
          displayState: completionState ?? 'completed',
          sourceRefs: buildSourceRefs(),
        };

        const fallbackAnswer = splitFallbackAnswer(safeFinalText);
        if (finalizedEventVersion === undefined) {
          finalUpdate.directAnswer = fallbackAnswer.directAnswer || safeFinalText;
          finalUpdate.fullAnalysis = fallbackAnswer.fullAnalysis || safeFinalText;
        }

        if (decisionSummary === undefined) {
          finalUpdate.decisionSummary = {
            conclusion: safeFinalText.slice(0, 200),
            actions: [],
            confidence: 'unknown',
            assumptions: [],
            risks: [],
          };
          finalUpdate.processOverview = processOverview || {
            phase: 'completed',
            durationMs: 0,
            warningCount: 0,
            blockingErrorCount: 0,
            impact: 'none',
          };
          finalUpdate.completionState = 'completed';
          finalUpdate.evidence = [];
        }

        updateAssistantState(finalUpdate);
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
                displayState: completionState ?? 'completed',
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
              isError: !normalizedAssistant,
              statusText: '',
              displayState: normalizedAssistant ? 'partial_completed' : 'failed',
              completionState: normalizedAssistant ? 'partial_completed' : 'failed',
              sourceRefs: buildSourceRefs(),
            });
            return;
          }

          updateAssistantState({
            content: assistantContent || '（请求已取消）',
            isThinking: false,
            statusText: '',
            displayState: assistantContent.trim() ? 'partial_completed' : 'failed',
            completionState: assistantContent.trim() ? 'partial_completed' : 'failed',
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
          displayState: 'failed',
          completionState: 'failed',
          sourceRefs: buildSourceRefs(),
        });
      } finally {
        if (completionTimer) {
          window.clearTimeout(completionTimer);
        }
        if (longRunningTimer) {
          window.clearTimeout(longRunningTimer);
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
    hydrated,
    isActivePending,
    createNewSession,
    deleteSession,
    renameSession,
    handleSubmit,
    abortSessionRequest,
  };
}
