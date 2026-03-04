import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  EVENT_ENDPOINT,
  getApiBaseUrl,
  type ChatMessage,
  type ChatSession,
  type ThinkingStep,
  type ToolInvocation,
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
  // 证据与透明性新增类型
  type HonestySignals,
} from '../lib/chat-types';
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

        // Answer-first: 结构化数据存储
        let finalizedEventVersion: number | undefined;
        let decisionSummary: DecisionSummary | undefined;
        let processOverview: ProcessOverview | undefined;
        let completionState: CompletionState | undefined;
        let evidence: EvidenceItem[] | undefined;

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

          // 处理终态事件 (event: final)
          if (eventName === 'final') {
            const eventVersion = typeof parsed.event_version === 'number' ? parsed.event_version : 0;

            // 终态竞态检查：只接受更高版本
            if (finalizedEventVersion !== undefined && eventVersion <= finalizedEventVersion) {
              return;
            }

            finalizedEventVersion = eventVersion;

            // 提取结构化字段
            const ds = asRecord(parsed.decisionSummary);
            const po = asRecord(parsed.processOverview);
            const cs = asString(parsed.completionState) as CompletionState | undefined;
            const ev = parsed.evidence;

            if (ds && po && cs && Array.isArray(ev)) {
              decisionSummary = {
                conclusion: asString(ds.conclusion) || '',
                actions: Array.isArray(ds.actions) ? ds.actions.filter((a): a is string => typeof a === 'string') : [],
                confidence: (['high', 'medium', 'low', 'unknown'].includes(asString(ds.confidence) || '') ? ds.confidence : 'unknown') as 'high' | 'medium' | 'low' | 'unknown',
                assumptions: Array.isArray(ds.assumptions) ? ds.assumptions.filter((a): a is string => typeof a === 'string') : [],
                risks: Array.isArray(ds.risks) ? ds.risks.filter((a): a is string => typeof a === 'string') : [],
                failureReason: asString(ds.failureReason),
              };

              processOverview = {
                phase: (['retrieving', 'validating', 'synthesizing', 'completed'].includes(asString(po.phase) || '') ? po.phase : 'retrieving') as RunPhase,
                durationMs: typeof po.durationMs === 'number' ? po.durationMs : 0,
                warningCount: typeof po.warningCount === 'number' ? po.warningCount : 0,
                blockingErrorCount: typeof po.blockingErrorCount === 'number' ? po.blockingErrorCount : 0,
                impact: (['none', 'partial', 'blocking'].includes(asString(po.impact) || '') ? po.impact : 'none') as 'none' | 'partial' | 'blocking',
              };

              completionState = cs;
              evidence = ev as EvidenceItem[];

              // 提取新增字段（证据与透明性）
              const directAnswer = asString(parsed.directAnswer);
              const fullAnalysis = asString(parsed.fullAnalysis);
              const references = Array.isArray(parsed.references) ? parsed.references : undefined;
              const processSummary = Array.isArray(parsed.processSummary) ? parsed.processSummary : undefined;
              const honestySignals = parsed.honestySignals && typeof parsed.honestySignals === 'object' 
                ? parsed.honestySignals as HonestySignals 
                : undefined;

              // 构建 citationMap
              const citationMap: Record<string, any> = {};
              if (references && Array.isArray(references)) {
                references.forEach((ref: any) => {
                  if (ref && typeof ref.id === 'string') {
                    citationMap[ref.id] = ref;
                  }
                });
              }

              // 更新状态
              updateAssistantState({
                isThinking: false,
                statusText: '',
                sourceRefs: buildSourceRefs(),
                decisionSummary,
                processOverview,
                completionState,
                evidence,
                finalizedEventVersion,
                directAnswer,
                fullAnalysis,
                references,
                citationMap,
                processSummary,
                honestySignals,
              });
            }
            return;
          }

          // 处理 OpenCode 原始事件 (message.part.updated)
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

            // 查找或创建工具步骤
            let toolStep = steps.find(s => s.tool?.id === callId);

            if (status === 'pending' || status === 'running') {
              clearCompletionTimer();
              updateAssistantState({
                isThinking: true,
                statusText: `正在调用工具：${toolName}`,
              });

              // 创建新的工具步骤
              if (!toolStep) {
                const stepId = createId();
                toolStep = {
                  id: stepId,
                  type: 'tool',
                  tool: {
                    id: callId,
                    name: toolName,
                    status: status,
                    arguments: state?.arguments as Record<string, unknown>,
                    startedAt: Date.now(),
                  },
                  timestamp: Date.now(),
                };
                addStep(toolStep);
              } else {
                // 更新状态
                updateStep(toolStep.id, {
                  tool: {
                    ...toolStep.tool!,
                    status: status,
                  }
                });
              }
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

              // 更新工具步骤为完成状态
              if (toolStep) {
                updateStep(toolStep.id, {
                  tool: {
                    ...toolStep.tool!,
                    status: 'completed',
                    result: state?.result,
                    completedAt: Date.now(),
                  }
                });
              }
              return;
            }

            if (status === 'error') {
              const message = asString(state?.error) ?? '工具执行失败';
              updateAssistantState({
                isThinking: true,
                statusText: `${toolName} 失败：${message}`,
              });
              maybeScheduleCompletion();

              // 更新工具步骤为错误状态
              if (toolStep) {
                updateStep(toolStep.id, {
                  tool: {
                    ...toolStep.tool!,
                    status: 'error',
                    error: message,
                    completedAt: Date.now(),
                  }
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
            clearCompletionTimer();
            updateAssistantState({
              isThinking: true,
              statusText: assistantContent.trim() ? '' : '正在思考...',
            });

            // 添加思考步骤
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
            sawStepFinish = true;
            maybeScheduleCompletion();
          }
        });

          // 最终步骤：合并思考步骤
        const finalText = assistantContent.trim() || '助手暂时没有回复。';
        const normalizedFinal = finalText.trim();
        const safeFinalText =
          normalizedFinal && normalizedFinal !== normalizedUserInput
            ? finalText
            : '本次请求未产出可展示的最终回复，请重试或缩小问题范围。';

        // 如果没有收到 final 事件，使用降级状态
        const finalUpdate: Partial<ChatMessage> = {
          content: safeFinalText,
          isThinking: false,
          statusText: '',
          sourceRefs: buildSourceRefs(),
        };

        // 仅在未收到终态事件时才设置默认值
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
    hydrated,
    isActivePending,
    createNewSession,
    deleteSession,
    renameSession,
    handleSubmit,
    abortSessionRequest,
  };
}
