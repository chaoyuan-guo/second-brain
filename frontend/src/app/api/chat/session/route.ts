import {
  OPENCODE_BASE_URL,
  buildUpstreamUrl,
  copySelectedResponseHeaders,
  forwardCommonHeaders,
} from '../../_lib/upstream';

export const runtime = 'nodejs';

export async function POST(request: Request): Promise<Response> {
  const body = await request.arrayBuffer();
  const upstream = await fetch(buildUpstreamUrl(OPENCODE_BASE_URL, '/session'), {
    method: 'POST',
    headers: forwardCommonHeaders(request),
    body,
    cache: 'no-store',
  });

  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: copySelectedResponseHeaders(upstream.headers, ['content-type']),
  });
}

