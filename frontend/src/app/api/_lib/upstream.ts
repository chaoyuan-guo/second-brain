import { randomUUID } from 'crypto';

const trimTrailingSlash = (value: string): string => value.replace(/\/+$/, '');

const getBaseUrl = (envName: string, fallback: string): string => {
  const value = process.env[envName]?.trim();
  return trimTrailingSlash(value || fallback);
};

export const OPENCODE_BASE_URL = getBaseUrl(
  'SECOND_BRAIN_OPENCODE_BASE_URL',
  'http://127.0.0.1:9090',
);

export const RAG_BASE_URL = getBaseUrl(
  'SECOND_BRAIN_RAG_BASE_URL',
  'http://127.0.0.1:9070',
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

export const forwardCommonHeaders = (request: Request, requestId?: string): HeadersInit => {
  const headers: Record<string, string> = {};
  const accept = request.headers.get('accept');
  const contentType = request.headers.get('content-type');
  const authorization = request.headers.get('authorization');
  const xStreamFormat = request.headers.get('x-stream-format');
  const traceId = request.headers.get('x-trace-id');
  const resolvedRequestId = requestId || getOrCreateRequestId(request);

  if (accept) headers.Accept = accept;
  if (contentType) headers['Content-Type'] = contentType;
  if (authorization) headers.Authorization = authorization;
  if (xStreamFormat) headers['X-Stream-Format'] = xStreamFormat;
  if (traceId) headers['X-Trace-Id'] = traceId;
  headers['X-Request-Id'] = resolvedRequestId;

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

export const getOrCreateRequestId = (request: Request): string => {
  const existing = request.headers.get('x-request-id')?.trim();
  if (existing) {
    return existing;
  }
  return randomUUID();
};

export const logProxy = (
  level: 'info' | 'error',
  payload: Record<string, unknown>,
): void => {
  const line = JSON.stringify({
    scope: 'next-proxy',
    ...payload,
    ts: new Date().toISOString(),
  });
  if (level === 'error') {
    console.error(line);
    return;
  }
  console.info(line);
};
