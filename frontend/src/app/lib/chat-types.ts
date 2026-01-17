export type ChatRole = 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  isThinking?: boolean;
  isError?: boolean;
  statusText?: string;
  tool_call_id?: string;
  timestamp?: number;
}

export type StreamEvent =
  | { type: 'delta'; delta: string; ts?: number }
  | {
      type: 'status';
      phase?: 'thinking' | 'synthesize' | string;
      message: string;
      tool_invocations?: number;
      ts?: number;
    }
  | {
      type: 'tool';
      stage: 'start' | 'end' | 'error';
      tool_name: string;
      tool_call_id?: string;
      tool_count?: number;
      latency_ms?: number;
      message: string;
      error?: string | null;
      ts?: number;
    }
  | { type: 'done'; ts?: number };

export type ApiRole = 'system' | 'user' | 'assistant' | 'tool' | 'developer';

export interface ApiMessagePayload {
  role: ApiRole;
  content: string;
  tool_call_id?: string;
  name?: string;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
  isCustomTitle?: boolean;
}

export interface MessageSegment {
  type: 'text' | 'code';
  content: string;
  language?: string;
}

const DEFAULT_BACKEND_PORT = process.env.NEXT_PUBLIC_BACKEND_PORT ?? '9000';
const LOCAL_FALLBACK_BASE = 'http://127.0.0.1:9000';

const trimTrailingSlash = (value: string) => value.replace(/\/+$/, '');

const inferBrowserApiBase = (): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }

  const { protocol, hostname, origin } = window.location;
  const codespaceMatch = hostname.match(/^(\d+)-(.+)$/);
  if (codespaceMatch) {
    const [, , remainder] = codespaceMatch;
    return `${protocol}//${DEFAULT_BACKEND_PORT}-${remainder}`;
  }

  if (process.env.NODE_ENV !== 'development') {
    return origin;
  }

  const shouldOmitPort =
    !DEFAULT_BACKEND_PORT ||
    (protocol === 'https:' && DEFAULT_BACKEND_PORT === '443') ||
    (protocol === 'http:' && DEFAULT_BACKEND_PORT === '80');

  const portSegment = shouldOmitPort ? '' : `:${DEFAULT_BACKEND_PORT}`;
  return `${protocol}//${hostname}${portSegment}`;
};

export const getApiBaseUrl = (): string => {
  const explicit = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  if (explicit) {
    return trimTrailingSlash(explicit);
  }

  const inferred = inferBrowserApiBase();
  if (inferred) {
    return trimTrailingSlash(inferred);
  }

  return LOCAL_FALLBACK_BASE;
};

export const STORAGE_KEY = 'second_brain_sessions_v1';
export const STREAM_ENDPOINT = '/chat/stream';
export const TITLE_ENDPOINT = '/chat/title';
