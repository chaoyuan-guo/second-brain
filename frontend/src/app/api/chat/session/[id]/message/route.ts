import {
  OPENCODE_BASE_URL,
  buildUpstreamUrl,
  copySelectedResponseHeaders,
  forwardCommonHeaders,
  getOrCreateRequestId,
  logProxy,
} from '../../../../_lib/upstream';

export const runtime = 'nodejs';

interface RouteContext {
  params: { id: string } | Promise<{ id: string }>;
}

export async function POST(request: Request, context: RouteContext): Promise<Response> {
  const { id } = await context.params;
  const requestId = getOrCreateRequestId(request);
  const route = '/api/chat/session/:id/message';
  const upstreamUrl = buildUpstreamUrl(
    OPENCODE_BASE_URL,
    `/session/${encodeURIComponent(id)}/prompt_async`,
  );
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
  if (upstream.status === 204 || upstream.status === 205 || upstream.status === 304) {
    return new Response(null, {
      status: upstream.status,
      headers,
    });
  }

  return new Response(await upstream.text(), {
    status: upstream.status,
    headers,
  });
}
