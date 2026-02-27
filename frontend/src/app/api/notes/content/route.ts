import {
  RAG_BASE_URL,
  buildUpstreamUrl,
  copySelectedResponseHeaders,
} from '../../_lib/upstream';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(request: Request): Promise<Response> {
  const sourceUrl = new URL(request.url);
  const upstream = await fetch(
    buildUpstreamUrl(RAG_BASE_URL, '/notes/content', sourceUrl.searchParams),
    {
      method: 'GET',
      headers: {
        Accept: request.headers.get('accept') ?? 'application/json',
      },
      cache: 'no-store',
    },
  );

  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: copySelectedResponseHeaders(upstream.headers, ['content-type']),
  });
}

