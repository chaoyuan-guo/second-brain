import {
  OPENCODE_BASE_URL,
  buildUpstreamUrl,
  copySelectedResponseHeaders,
  forwardCommonHeaders,
  getOrCreateRequestId,
  logProxy,
} from '../../_lib/upstream';

export const runtime = 'nodejs';

export async function POST(request: Request): Promise<Response> {
  const requestId = getOrCreateRequestId(request);
  const route = '/api/chat/session';
  const upstreamUrl = buildUpstreamUrl(OPENCODE_BASE_URL, '/session');
  const start = Date.now();
  logProxy('info', { route, requestId, phase: 'start', method: 'POST', upstreamUrl });

  const body = await request.arrayBuffer();
  let upstream: Response;
  try {
    upstream = await fetch(upstreamUrl, {
      method: 'POST',
      headers: forwardCommonHeaders(request, requestId),
      body,
      cache: 'no-store',
    });
  } catch (error) {
    logProxy('error', {
      route,
      requestId,
      phase: 'fetch_error',
      method: 'POST',
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
    method: 'POST',
    upstreamUrl,
    status: upstream.status,
    durationMs,
  });

  const headers = copySelectedResponseHeaders(upstream.headers, ['content-type']);
  headers.set('x-request-id', requestId);
  return new Response(await upstream.text(), {
    status: upstream.status,
    headers,
  });
}
