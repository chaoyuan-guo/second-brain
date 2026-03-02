export type ChatRole = 'user' | 'assistant';

// ============================================================================
// Answer-first 重构新增类型（2026-03-02）
// ============================================================================

/** 运行阶段 */
export type RunPhase = 'retrieving' | 'validating' | 'synthesizing' | 'completed';

/** 完成状态 */
export type CompletionState = 'completed' | 'partial_completed' | 'failed';

/** 置信度等级 */
export type ConfidenceLevel = 'high' | 'medium' | 'low' | 'unknown';

/** 证据引用 */
export interface EvidenceRef {
  sourcePath: string;
  sourceTitle?: string;
  heading?: string;
  charOffsetStart?: number;
  charOffsetEnd?: number;
  snippet?: string;
}

/** 证据项：断言-来源映射 */
export interface EvidenceItem {
  claimId: string;
  claimText: string;
  refs: EvidenceRef[];
}

/** 决策摘要 */
export interface DecisionSummary {
  conclusion: string;
  actions: string[];
  confidence: ConfidenceLevel;
  assumptions: string[];
  risks: string[];
  failureReason?: string;
}

/** 过程概览 */
export interface ProcessOverview {
  phase: RunPhase;
  durationMs: number;
  warningCount: number;
  blockingErrorCount: number;
  impact: 'none' | 'partial' | 'blocking';
}

// ============================================================================
// 原有类型定义
// ============================================================================

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
  // ============================================================================
  // Answer-first 重构新增字段（2026-03-02）
  // ============================================================================
  /** 决策摘要（结论、行动、置信度等） */
  decisionSummary?: DecisionSummary;
  /** 过程概览（阶段、耗时、异常数等） */
  processOverview?: ProcessOverview;
  /** 完成状态 */
  completionState?: CompletionState;
  /** 证据与引用 */
  evidence?: EvidenceItem[];
  /** 终态事件版本号，用于防止乱序/重复 */
  finalizedEventVersion?: number;
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
