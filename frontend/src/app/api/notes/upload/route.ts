import {
  RAG_BASE_URL,
  buildUpstreamUrl,
  copySelectedResponseHeaders,
} from '../../_lib/upstream';

export const runtime = 'nodejs';

export async function POST(request: Request): Promise<Response> {
  const formData = await request.formData();
  const upstream = await fetch(buildUpstreamUrl(RAG_BASE_URL, '/notes/upload'), {
    method: 'POST',
    body: formData,
    cache: 'no-store',
  });

  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: copySelectedResponseHeaders(upstream.headers, ['content-type']),
  });
}

