import {
  OPENCODE_BASE_URL,
  buildUpstreamUrl,
  copySelectedResponseHeaders,
  forwardCommonHeaders,
} from '../../../../_lib/upstream';

export const runtime = 'nodejs';

interface RouteContext {
  params: { id: string } | Promise<{ id: string }>;
}

export async function POST(request: Request, context: RouteContext): Promise<Response> {
  const { id } = await context.params;
  const body = await request.arrayBuffer();
  const upstream = await fetch(
    buildUpstreamUrl(OPENCODE_BASE_URL, `/session/${encodeURIComponent(id)}/prompt_async`),
    {
      method: 'POST',
      headers: forwardCommonHeaders(request),
      body,
      cache: 'no-store',
    },
  );

  const headers = copySelectedResponseHeaders(upstream.headers, ['content-type']);
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
