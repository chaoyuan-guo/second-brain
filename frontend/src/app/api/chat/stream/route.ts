import {
  LEGACY_BACKEND_BASE_URL,
  buildUpstreamUrl,
  copySelectedResponseHeaders,
  forwardCommonHeaders,
} from '../../_lib/upstream';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: Request): Promise<Response> {
  const body = await request.arrayBuffer();
  const upstream = await fetch(buildUpstreamUrl(LEGACY_BACKEND_BASE_URL, '/chat/stream'), {
    method: 'POST',
    headers: forwardCommonHeaders(request),
    body,
    cache: 'no-store',
  });

  if (!upstream.body) {
    return new Response(await upstream.text(), {
      status: upstream.status,
      headers: copySelectedResponseHeaders(upstream.headers, ['content-type']),
    });
  }

  return new Response(upstream.body, {
    status: upstream.status,
    headers: copySelectedResponseHeaders(upstream.headers, [
      'content-type',
      'cache-control',
      'x-request-id',
    ]),
  });
}

