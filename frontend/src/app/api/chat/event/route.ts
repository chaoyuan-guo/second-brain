import { OPENCODE_BASE_URL, buildUpstreamUrl } from '../../_lib/upstream';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(request: Request): Promise<Response> {
  const upstream = await fetch(buildUpstreamUrl(OPENCODE_BASE_URL, '/event'), {
    method: 'GET',
    headers: {
      Accept: request.headers.get('accept') ?? 'text/event-stream',
    },
    cache: 'no-store',
    signal: request.signal,
  });

  if (!upstream.body) {
    return new Response(await upstream.text(), {
      status: upstream.status,
      headers: {
        'Content-Type': upstream.headers.get('content-type') ?? 'text/plain; charset=utf-8',
      },
    });
  }

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      'Content-Type': upstream.headers.get('content-type') ?? 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
    },
  });
}

