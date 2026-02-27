export type ChatRole = 'user' | 'assistant';

export interface SourceRef {
  path: string;
  heading: string;
  char_offset?: number;
  snippet?: string;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  isThinking?: boolean;
  isError?: boolean;
  statusText?: string;
  tool_call_id?: string;
  timestamp?: number;
  sources?: string[];
  sourceRefs?: SourceRef[];
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
  | {
      type: 'sources';
      sources: string[];
      source_refs?: SourceRef[];
      expected_sources?: string[];
      question_id?: string;
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
  upstreamSessionId?: string;
}

export interface MessageSegment {
  type: 'text' | 'code';
  content: string;
  language?: string;
}

const trimTrailingSlash = (value: string) => value.replace(/\/+$/, '');

export const getApiBaseUrl = (): string => {
  const explicit = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  if (explicit) {
    return trimTrailingSlash(explicit);
  }
  return '';
};

export const STORAGE_KEY = 'second_brain_sessions_v2';
export const STREAM_ENDPOINT = '/api/chat/stream/';
export const TITLE_ENDPOINT = '/api/chat/title/';
export const SESSION_ENDPOINT = '/api/chat/session/';
export const EVENT_ENDPOINT = '/api/chat/event/';
export const sessionMessageEndpoint = (sessionId: string): string =>
  `/api/chat/session/${encodeURIComponent(sessionId)}/message/`;
export const UPLOAD_ENDPOINT = '/api/notes/upload/';
export const NOTE_CONTENT_ENDPOINT = '/api/notes/content/';
