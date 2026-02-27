const trimTrailingSlash = (value: string): string => value.replace(/\/+$/, '');

const getBaseUrl = (envName: string, fallback: string): string => {
  const value = process.env[envName]?.trim();
  return trimTrailingSlash(value || fallback);
};

export const LEGACY_BACKEND_BASE_URL = getBaseUrl(
  'SECOND_BRAIN_LEGACY_BACKEND_BASE_URL',
  'http://127.0.0.1:9000',
);

export const OPENCODE_BASE_URL = getBaseUrl(
  'SECOND_BRAIN_OPENCODE_BASE_URL',
  'http://127.0.0.1:9090',
);

export const RAG_BASE_URL = getBaseUrl(
  'SECOND_BRAIN_RAG_BASE_URL',
  LEGACY_BACKEND_BASE_URL,
);

export const buildUpstreamUrl = (
  baseUrl: string,
  path: string,
  searchParams?: URLSearchParams,
): string => {
  const url = new URL(path, `${baseUrl}/`);
  if (searchParams) {
    searchParams.forEach((value, key) => {
      url.searchParams.set(key, value);
    });
  }
  return url.toString();
};

export const forwardCommonHeaders = (request: Request): HeadersInit => {
  const headers: Record<string, string> = {};
  const accept = request.headers.get('accept');
  const contentType = request.headers.get('content-type');
  const authorization = request.headers.get('authorization');
  const xStreamFormat = request.headers.get('x-stream-format');

  if (accept) headers.Accept = accept;
  if (contentType) headers['Content-Type'] = contentType;
  if (authorization) headers.Authorization = authorization;
  if (xStreamFormat) headers['X-Stream-Format'] = xStreamFormat;

  return headers;
};

export const copySelectedResponseHeaders = (
  upstreamHeaders: Headers,
  keys: string[],
): Headers => {
  const headers = new Headers();
  keys.forEach((key) => {
    const value = upstreamHeaders.get(key);
    if (value) {
      headers.set(key, value);
    }
  });
  return headers;
};

