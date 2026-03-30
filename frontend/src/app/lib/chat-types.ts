export type ChatRole = 'user' | 'assistant';
export type CitationKind = 'precise' | 'file';
export type CitationProvenance = 'native' | 'synthetic_read' | 'content_path';

// ============================================================================
// 证据可追溯性与过程透明度新增类型（2026-03-04）
// ============================================================================

/** 引用引用类型 */
export interface CitationRef {
  id: string;
  sourcePath: string;
  sourceTitle?: string;
  sourceDateLabel?: string;
  heading?: string;
  charOffsetStart?: number;
  charOffsetEnd?: number;
  snippet?: string;
  retrievalScore?: number;
  weakMatch?: boolean;
  kind?: CitationKind;
  provenance?: CitationProvenance;
}

/** 过程步骤语义摘要 */
export interface ProcessStepSummary {
  /** 步骤序号 */
  stepNumber: number;
  /** 运行阶段 */
  phase: 'retrieving' | 'validating' | 'synthesizing' | 'completed';
  /** 步骤摘要（用户友好的描述） */
  summary: string;
  /** 详细说明 */
  detail?: string;
  /** 工具名称 */
  toolName?: string;
  /** 执行耗时（毫秒） */
  durationMs?: number;
  /** 步骤 ID（兼容性） */
  stepId?: string;
  /** 语义类型（兼容性） */
  semanticType?: 'retrieve' | 'validate' | 'synthesize_helper' | 'read' | 'web' | 'execute' | 'skill' | 'other';
  /** 输入摘要（兼容性） */
  inputSummary?: string;
  /** 结果摘要（兼容性） */
  resultSummary?: string;
  /** 状态（兼容性） */
  status?: 'completed' | 'error' | 'running';
}

/** 诚实性信号 */
export interface HonestySignals {
  /** 触发原因码 */
  reasonCodes: Array<'no_hit' | 'weak_match' | 'insufficient_hits'>;
  /** 证据质量等级 */
  evidenceQuality: 'strong' | 'partial' | 'weak' | 'none';
  /** 弱匹配引用 ID 列表（分数 < 0.8） */
  weakMatches: string[];
  /** 无分数引用 ID 列表 */
  unscoredMatches: string[];
  /** 诚实性警告信息 */
  honestyWarnings: string[];
  /** 局限性说明（当证据不足时） */
  limitationNote?: string;
  /** 是否有足够证据（强匹配 >= 2 或 强匹配 >=1 + 弱匹配 >=1） */
  hasSufficientEvidence: boolean;
  /** 是否有直接证据 */
  hasDirectEvidence?: boolean;
  /** 检索命中数 */
  retrievalHitCount?: number;
  /** 最佳分数 */
  bestScore?: number;
}

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
  sourceDateLabel?: string;
  heading?: string;
  charOffsetStart?: number;
  charOffsetEnd?: number;
  snippet?: string;
  citationId?: string;      // 新增：引用标记 ID
  retrievalScore?: number;  // 新增：检索相关性分数
  kind?: CitationKind;
  provenance?: CitationProvenance;
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
  thinkingSteps?: ThinkingStep[];
  currentStepId?: string;
  decisionSummary?: DecisionSummary;
  processOverview?: ProcessOverview;
  completionState?: CompletionState;
  evidence?: EvidenceItem[];
  finalizedEventVersion?: number;
  directAnswer?: string;
  fullAnalysis?: string;
  references?: CitationRef[];
  citationMap?: Record<string, CitationRef>;
  processSummary?: ProcessStepSummary[];
  honestySignals?: HonestySignals;
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
