import {
  OPENCODE_BASE_URL,
  buildUpstreamUrl,
  getOrCreateRequestId,
  logProxy,
} from '../../_lib/upstream';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(request: Request): Promise<Response> {
  const requestId = getOrCreateRequestId(request);
  const route = '/api/chat/event';
  const upstreamUrl = buildUpstreamUrl(OPENCODE_BASE_URL, '/event');
  const start = Date.now();
  logProxy('info', { route, requestId, phase: 'start', method: 'GET', upstreamUrl });

  let upstream: Response;
  try {
    upstream = await fetch(upstreamUrl, {
      method: 'GET',
      headers: {
        Accept: request.headers.get('accept') ?? 'text/event-stream',
        'X-Request-Id': requestId,
      },
      cache: 'no-store',
      signal: request.signal,
    });
  } catch (error) {
    logProxy('error', {
      route,
      requestId,
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

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      'Content-Type': upstream.headers.get('content-type') ?? 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-Request-Id': requestId,
    },
  });
}
