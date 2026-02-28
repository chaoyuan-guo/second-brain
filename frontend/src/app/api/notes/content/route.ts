import {
  RAG_BASE_URL,
  buildUpstreamUrl,
  copySelectedResponseHeaders,
  getOrCreateRequestId,
  logProxy,
} from '../../_lib/upstream';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(request: Request): Promise<Response> {
  const requestId = getOrCreateRequestId(request);
  const route = '/api/notes/content';
  const sourceUrl = new URL(request.url);
  const upstreamUrl = buildUpstreamUrl(RAG_BASE_URL, '/notes/content', sourceUrl.searchParams);
  const start = Date.now();
  logProxy('info', { route, requestId, phase: 'start', method: 'GET', upstreamUrl });

  let upstream: Response;
  try {
    upstream = await fetch(upstreamUrl, {
      method: 'GET',
      headers: {
        Accept: request.headers.get('accept') ?? 'application/json',
        'X-Request-Id': requestId,
      },
      cache: 'no-store',
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
  const headers = copySelectedResponseHeaders(upstream.headers, ['content-type']);
  headers.set('x-request-id', requestId);
  return new Response(await upstream.text(), {
    status: upstream.status,
    headers,
  });
}
