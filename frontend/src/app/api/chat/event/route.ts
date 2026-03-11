import {
  OPENCODE_BASE_URL,
  buildUpstreamUrl,
  getOrCreateRequestId,
  logProxy,
} from '../../_lib/upstream';
import {
  createProcessAccumulator,
  deriveSyntheticSourceRefsFromCall,
  updateToolCall,
  mergeSourceRefs,
  synthesizeFinalEvent,
  type ToolCallRecord,
} from '../../_lib/event-adapter';
import type { EvidenceRef } from '../../../lib/chat-types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const LOG_EVENT_SUMMARY = process.env.OPENCODE_EVENT_SUMMARY_LOG !== '0';

// ============================================================================
// 辅助函数
// ============================================================================

const asRecord = (value: unknown): Record<string, unknown> | undefined =>
  value && typeof value === 'object' ? (value as Record<string, unknown>) : undefined;

const asString = (value: unknown): string | undefined =>
  typeof value === 'string' ? value : undefined;

const asNumber = (value: unknown): number | undefined =>
  typeof value === 'number' ? value : undefined;

const getToolInput = (state: Record<string, unknown>): Record<string, unknown> | undefined =>
  asRecord(state.arguments) || asRecord(state.input);

const getToolOutput = (state: Record<string, unknown>): unknown =>
  state.result ?? state.output;

interface KnownPartMeta {
  type?: string;
  messageID?: string;
  sessionID?: string;
}

/**
 * 从工具 metadata 中提取 source_refs
 */
const extractSourceRefs = (metadata: unknown): EvidenceRef[] => {
  const record = asRecord(metadata);
  const raw = record?.source_refs;
  if (!Array.isArray(raw)) {
    return [];
  }

  const refs: EvidenceRef[] = [];
  raw.forEach((item) => {
    const entry = asRecord(item);
    const path = asString(entry?.path)?.trim();
    if (!path) {
      return;
    }

    const heading = asString(entry?.heading)?.trim();
    const snippet = asString(entry?.snippet);
    const charOffset = asNumber(entry?.char_offset);
    const score = asNumber(entry?.score);
    const citationId = asString(entry?.citation_id);
    const sourceTitle = asString(entry?.source_title)?.trim();

    refs.push({
      sourcePath: path,
      sourceTitle,
      heading,
      snippet,
      charOffsetStart: charOffset,
      retrievalScore: score,
      citationId,
    });
  });

  return refs;
};

/**
 * 解析 OpenCode 事件
 */
const parseOpenCodeEvent = (
  data: string
): {
  type: string;
  part?: Record<string, unknown>;
  properties: Record<string, unknown>;
} | null => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(data);
  } catch {
    return null;
  }

  const record = asRecord(parsed);
  if (!record) return null;

  const type = asString(record.type) || 'message';
  if (type !== 'message.part.updated' && type !== 'message.part.delta') {
    return null;
  }

  const properties = asRecord(record.properties) || {};
  const part = asRecord(properties.part ?? record.part);

  return { type, part, properties };
};

// ============================================================================
// 事件日志摘要
// ============================================================================

const summarizeEvent = (
  requestId: string,
  traceId: string,
  targetSessionId: string,
  payload: string,
): void => {
  if (!LOG_EVENT_SUMMARY) {
    return;
  }

  const parsed = parseOpenCodeEvent(payload);
  if (!parsed) {
    // 检查是否是 session.error
    let errorParsed: Record<string, unknown> | undefined;
    try {
      errorParsed = JSON.parse(payload) as Record<string, unknown>;
    } catch {
      return;
    }
    const type = typeof errorParsed.type === 'string' ? errorParsed.type : '';
    if (type === 'session.error') {
      logProxy('error', {
        route: '/api/chat/event',
        requestId,
        traceId,
        phase: 'event',
        eventType: 'session.error',
        payload: errorParsed.properties ?? errorParsed,
      });
    }
    return;
  }

  const { type, part, properties } = parsed;
  const partType = asString(part?.type);
  const sessionID = asString(part?.sessionID) ?? asString(properties.sessionID);
  const messageID = asString(part?.messageID) ?? asString(properties.messageID);

  if (targetSessionId && sessionID && sessionID !== targetSessionId) {
    return;
  }

  if (type === 'message.part.delta') {
    logProxy('info', {
      route: '/api/chat/event',
      requestId,
      traceId,
      phase: 'event',
      eventType: type,
      partType,
      sessionID,
      messageID,
      partID: properties.partID,
      field: properties.field,
    });
    return;
  }

  if (partType === 'tool') {
    const state = asRecord(part?.state) || {};
    logProxy('info', {
      route: '/api/chat/event',
      requestId,
      traceId,
      phase: 'event',
      eventType: 'message.part.updated',
      partType,
      sessionID,
      messageID,
      tool: part?.tool,
      callID: part?.callID,
      status: state.status,
    });
    return;
  }

  if (partType === 'step-start' || partType === 'step-finish') {
    logProxy('info', {
      route: '/api/chat/event',
      requestId,
      traceId,
      phase: 'event',
      eventType: 'message.part.updated',
      partType,
      sessionID,
      messageID,
    });
  }
};

// ============================================================================
// 主处理逻辑
// ============================================================================

export async function GET(request: Request): Promise<Response> {
  const requestId = getOrCreateRequestId(request);
  const traceId = request.headers.get('x-trace-id')?.trim() || requestId;
  const targetSessionId = request.headers.get('x-session-id')?.trim() || '';
  const route = '/api/chat/event';
  const upstreamUrl = buildUpstreamUrl(OPENCODE_BASE_URL, '/event');
  const start = Date.now();

  logProxy('info', {
    route,
    requestId,
    traceId,
    targetSessionId,
    phase: 'start',
    method: 'GET',
    upstreamUrl,
  });

  let upstream: Response;
  try {
    upstream = await fetch(upstreamUrl, {
      method: 'GET',
      headers: {
        Accept: request.headers.get('accept') ?? 'text/event-stream',
        'X-Request-Id': requestId,
        'X-Trace-Id': traceId,
      },
      cache: 'no-store',
      signal: request.signal,
    });
  } catch (error) {
    logProxy('error', {
      route,
      requestId,
      traceId,
      targetSessionId,
      phase: 'fetch_error',
      method: 'GET',
      upstreamUrl,
      durationMs: Date.now() - start,
      error: error instanceof Error ? error.message : String(error),
    });
    throw error;
  }

  const durationMs = Date.now() - start;
  logProxy('info', {
    route,
    requestId,
    traceId,
    targetSessionId,
    phase: 'done',
    method: 'GET',
    upstreamUrl,
    status: upstream.status,
    durationMs,
  });

  if (!upstream.body) {
    return new Response(await upstream.text(), {
      status: upstream.status,
      headers: {
        'Content-Type': upstream.headers.get('content-type') ?? 'text/plain; charset=utf-8',
        'X-Request-Id': requestId,
      },
    });
  }

  // 创建过程累加器
  const acc = createProcessAccumulator();
  let currentStepMessageId: string | undefined;
  let sawStepStart = false;
  const knownParts = new Map<string, KnownPartMeta>();

  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = '';

  const emitFinalEvent = (
    controller: TransformStreamDefaultController<Uint8Array>,
    trigger: 'step_finish' | 'flush',
  ): void => {
    const finalEvent = synthesizeFinalEvent(acc);
    const finalPayload = `event: final\ndata: ${JSON.stringify(finalEvent)}\n\n`;
    controller.enqueue(encoder.encode(finalPayload));

    logProxy('info', {
      route: '/api/chat/event',
      requestId,
      traceId,
      phase: 'final_event',
      trigger,
      completionState: finalEvent.completionState,
      eventVersion: finalEvent.event_version,
      durationMs: Date.now() - start,
    });
  };

  /**
   * 处理单个事件并返回是否应立即合成 final 事件
   */
  const processEvent = (data: string): boolean => {
    // 日志摘要
    summarizeEvent(requestId, traceId, targetSessionId, data);

    const parsed = parseOpenCodeEvent(data);
    if (!parsed) {
      return false;
    }

    const { type, part, properties } = parsed;

    if (type === 'message.part.delta') {
      const partID = asString(properties.partID);
      const delta = asString(properties.delta);
      const field = asString(properties.field);
      const meta = partID ? knownParts.get(partID) : undefined;
      const partType = meta?.type;
      const partMessageId = asString(properties.messageID) ?? meta?.messageID;
      const sessionID = asString(properties.sessionID) ?? meta?.sessionID;

      if (targetSessionId && sessionID && sessionID !== targetSessionId) {
        return false;
      }

      if (field !== 'text' || !delta || partType !== 'text') {
        return false;
      }

      if (!sawStepStart) {
        return false;
      }

      if (currentStepMessageId && partMessageId && partMessageId !== currentStepMessageId) {
        return false;
      }

      acc.assistantContent += delta;
      return false;
    }

    const partType = asString(part?.type);
    const partMessageId = asString(part?.messageID);
    const sessionID = asString(part?.sessionID);
    const partId = asString(part?.id);

    // 会话过滤
    if (targetSessionId && sessionID && sessionID !== targetSessionId) {
      return false;
    }

    if (partId) {
      knownParts.set(partId, {
        type: partType,
        messageID: partMessageId,
        sessionID,
      });
    }

    // 处理不同 partType
    if (partType === 'text') {
      if (!sawStepStart) {
        return false;
      }

      if (currentStepMessageId && partMessageId && partMessageId !== currentStepMessageId) {
        return false;
      }

      const delta = asString(properties?.delta ?? parsed.properties?.delta);
      const partText = asString(part?.text);
      if (delta && delta.length > 0) {
        acc.assistantContent += delta;
      } else if (partText !== undefined && partText.length > 0) {
        acc.assistantContent = partText;
      }
      return false;
    }

    if (partType === 'tool') {
      const toolName = asString(part?.tool) ?? 'tool';
      const callId = asString(part?.callID) ?? `${toolName}-${Date.now()}`;
      const state = asRecord(part?.state) || {};
      const status = asString(state.status) as ToolCallRecord['status'] | undefined;

      if (status) {
        updateToolCall(acc, callId, toolName, status, {
          arguments: getToolInput(state),
          error: asString(state.error),
          result: getToolOutput(state),
        });

        // completed 时提取 source_refs
        if (status === 'completed') {
          const refsFromMetadata = extractSourceRefs(state.metadata);
          const refs =
            refsFromMetadata.length > 0
              ? refsFromMetadata
              : deriveSyntheticSourceRefsFromCall(
                  {
                    name: toolName,
                    arguments: getToolInput(state),
                    result: getToolOutput(state),
                  },
                  acc.sourceRefMap,
                );
          if (refs.length > 0) {
            mergeSourceRefs(acc, refs);
            // 更新工具调用记录
            const record = acc.completedCalls.find((c) => c.id === callId);
            if (record) {
              record.sourceRefs = refs;
            }
          }
        }
      }
      return false;
    }

    if (partType === 'step-start') {
      sawStepStart = true;
      if (partMessageId) {
        currentStepMessageId = partMessageId;
      }
      return false;
    }

    if (partType === 'step-finish') {
      if (currentStepMessageId && partMessageId && partMessageId !== currentStepMessageId) {
        return false;
      }
      return acc.activeCalls.size === 0 && acc.assistantContent.trim().length > 0;
    }

    return false;
  };

  const observed = upstream.body.pipeThrough(
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        controller.enqueue(chunk);

        buffer += decoder.decode(chunk, { stream: true });
        let newlineIndex = buffer.indexOf('\n');
        while (newlineIndex !== -1) {
          let line = buffer.slice(0, newlineIndex);
          buffer = buffer.slice(newlineIndex + 1);
          if (line.endsWith('\r')) {
            line = line.slice(0, -1);
          }

          if (!line || line.startsWith(':')) {
            newlineIndex = buffer.indexOf('\n');
            continue;
          }

          if (line.startsWith('event:')) {
            newlineIndex = buffer.indexOf('\n');
            continue;
          }

          if (line.startsWith('data:')) {
            const data = line.slice(5).trimStart();
            const shouldEmitFinal = processEvent(data);
            if (shouldEmitFinal) {
              emitFinalEvent(controller, 'step_finish');
            }
          }

          newlineIndex = buffer.indexOf('\n');
        }
      },
      flush(controller) {
        buffer += decoder.decode();
        const tail = buffer.trim();
        if (tail.startsWith('data:')) {
          const data = tail.slice(5).trimStart();
          const shouldEmitFinal = processEvent(data);
          if (shouldEmitFinal) {
            emitFinalEvent(controller, 'step_finish');
          }
        }

        emitFinalEvent(controller, 'flush');
      },
    }),
  );

  return new Response(observed, {
    status: upstream.status,
    headers: {
      'Content-Type': upstream.headers.get('content-type') ?? 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-Request-Id': requestId,
    },
  });
}
