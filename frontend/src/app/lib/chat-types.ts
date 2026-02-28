export type ChatRole = 'user' | 'assistant';

export interface SourceRef {
  path: string;
  heading: string;
  char_offset?: number;
  snippet?: string;
}

// 工具调用状态
type ToolStatus = 'pending' | 'running' | 'completed' | 'error';

// 工具调用记录
export interface ToolInvocation {
  id: string;                    // 调用ID
  name: string;                  // 工具名称
  status: ToolStatus;
  arguments?: Record<string, unknown>;  // 调用参数
  result?: unknown;              // 返回结果（执行完成后）
  startedAt?: number;            // 开始时间戳
  completedAt?: number;        // 完成时间戳
  error?: string;                // 错误信息
}

// 思考步骤
export interface ThinkingStep {
  id: string;
  type: 'thought' | 'tool' | 'synthesize';
  content?: string;              // 思考内容
  tool?: ToolInvocation;         // 工具调用详情
  timestamp: number;
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
  // 新增：保存完整的思考步骤
  thinkingSteps?: ThinkingStep[];
  currentStepId?: string;  // 当前正在执行的步骤
}

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
export const SESSION_ENDPOINT = '/api/chat/session/';
export const EVENT_ENDPOINT = '/api/chat/event/';
export const sessionMessageEndpoint = (sessionId: string): string =>
  `/api/chat/session/${encodeURIComponent(sessionId)}/message/`;
export const UPLOAD_ENDPOINT = '/api/notes/upload/';
export const NOTE_CONTENT_ENDPOINT = '/api/notes/content/';
