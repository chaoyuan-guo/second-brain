import {
  OPENCODE_BASE_URL,
  buildUpstreamUrl,
  getOrCreateRequestId,
  logProxy,
} from '../../_lib/upstream';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const LOG_EVENT_SUMMARY = process.env.OPENCODE_EVENT_SUMMARY_LOG !== '0';

const summarizeEvent = (
  requestId: string,
  traceId: string,
  targetSessionId: string,
  payload: string,
): void => {
  if (!LOG_EVENT_SUMMARY) {
    return;
  }
  let parsed: Record<string, unknown> | undefined;
  try {
    parsed = JSON.parse(payload) as Record<string, unknown>;
  } catch {
    return;
  }
  const type = typeof parsed.type === 'string' ? parsed.type : '';
  if (type !== 'message.part.updated' && type !== 'session.error') {
    return;
  }

  if (type === 'session.error') {
    logProxy('error', {
      route: '/api/chat/event',
      requestId,
      traceId,
      phase: 'event',
      eventType: 'session.error',
      payload: parsed.properties ?? parsed,
    });
    return;
  }

  const properties =
    parsed.properties && typeof parsed.properties === 'object'
      ? (parsed.properties as Record<string, unknown>)
      : undefined;
  const part =
    properties?.part && typeof properties.part === 'object'
      ? (properties.part as Record<string, unknown>)
      : undefined;
  if (!part) {
    return;
  }
  const partType = typeof part.type === 'string' ? part.type : '';
  const sessionID = typeof part.sessionID === 'string' ? part.sessionID : '';
  const messageID = typeof part.messageID === 'string' ? part.messageID : '';
  if (targetSessionId && sessionID && sessionID !== targetSessionId) {
    return;
  }

  if (partType === 'tool') {
    const state = part.state && typeof part.state === 'object' ? (part.state as Record<string, unknown>) : {};
    logProxy('info', {
      route: '/api/chat/event',
      requestId,
      traceId,
      phase: 'event',
      eventType: type,
      partType,
      sessionID,
      messageID,
      tool: part.tool,
      callID: part.callID,
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
      eventType: type,
      partType,
      sessionID,
      messageID,
    });
  }
};

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

  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = '';
  const observed = upstream.body.pipeThrough(
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        controller.enqueue(chunk);
        if (!LOG_EVENT_SUMMARY) {
          return;
        }
        buffer += decoder.decode(chunk, { stream: true });
        let newlineIndex = buffer.indexOf('\n');
        while (newlineIndex !== -1) {
          let line = buffer.slice(0, newlineIndex);
          buffer = buffer.slice(newlineIndex + 1);
          if (line.endsWith('\r')) {
            line = line.slice(0, -1);
          }
          if (line.startsWith('data:')) {
            summarizeEvent(requestId, traceId, targetSessionId, line.slice(5).trimStart());
          }
          newlineIndex = buffer.indexOf('\n');
        }
      },
      flush(controller) {
        if (!LOG_EVENT_SUMMARY) {
          return;
        }
        buffer += decoder.decode();
        const tail = buffer.trim();
        if (tail.startsWith('data:')) {
          summarizeEvent(requestId, traceId, targetSessionId, tail.slice(5).trimStart());
        }
        const remaining = encoder.encode('');
        if (remaining.length > 0) {
          controller.enqueue(remaining);
        }
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
