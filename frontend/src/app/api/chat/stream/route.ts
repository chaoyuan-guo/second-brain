import {
  RAG_BASE_URL,
  buildUpstreamUrl,
  copySelectedResponseHeaders,
  forwardCommonHeaders,
  getOrCreateRequestId,
  logProxy,
} from '../../_lib/upstream';

export const runtime = 'nodejs';

export async function POST(request: Request): Promise<Response> {
  const requestId = getOrCreateRequestId(request);
  const route = '/api/chat/stream';
  const upstreamUrl = buildUpstreamUrl(RAG_BASE_URL, '/chat/stream');
  const start = Date.now();

  logProxy('info', { route, requestId, phase: 'start', method: 'POST', upstreamUrl });

  const body = await request.arrayBuffer();
  let upstream: Response;
  try {
    upstream = await fetch(upstreamUrl, {
      method: 'POST',
      headers: {
        ...forwardCommonHeaders(request, requestId),
        Accept: 'application/x-ndjson',
        'X-Stream-Format': 'ndjson',
      },
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

  logProxy('info', {
    route,
    requestId,
    phase: 'done',
    method: 'POST',
    upstreamUrl,
    status: upstream.status,
    durationMs: Date.now() - start,
  });

  const headers = copySelectedResponseHeaders(upstream.headers, ['content-type']);
  headers.set('x-request-id', requestId);

  if (!upstream.body) {
    return new Response(await upstream.text(), {
      status: upstream.status,
      headers,
    });
  }

  return new Response(upstream.body, {
    status: upstream.status,
    headers,
  });
}
