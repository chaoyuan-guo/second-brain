/**
 * 事件适配器工具模块
 * 用于将 OpenCode 原始事件映射为语义化事件，并在流结束时合成终态
 */

import type {
  RunPhase,
  CompletionState,
  ConfidenceLevel,
  DecisionSummary,
  ProcessOverview,
  EvidenceItem,
  EvidenceRef,
} from '../../lib/chat-types';

// ============================================================================
// 类型定义
// ============================================================================

export type EventKind = 'intermediate' | 'final';

export type ToolSemantic = 'retrieve' | 'validate' | 'synthesize_helper' | 'other';

export interface SemanticEvent {
  event_kind: EventKind;
  event_version: number;
  phase: RunPhase;
  severity: 'info' | 'warning' | 'blocking_error';
  impact: 'none' | 'partial' | 'blocking';
  user_message: string;
  debug?: Record<string, unknown>;
}

export interface FinalEventPayload {
  event_kind: 'final';
  event_version: number;
  decisionSummary: DecisionSummary;
  processOverview: ProcessOverview;
  completionState: CompletionState;
  evidence: EvidenceItem[];
}

// ============================================================================
// 工具名到语义阶段的映射
// ============================================================================

const TOOL_SEMANTIC_MAP: Record<string, ToolSemantic> = {
  query_my_notes: 'retrieve',
  web_search: 'retrieve',
  read_page: 'retrieve',
  read_note_file: 'retrieve',
  grep: 'retrieve',
  glob: 'retrieve',
  run_code_interpreter: 'validate',
  load_skill: 'synthesize_helper',
};

export const getToolSemantic = (toolName: string): ToolSemantic => {
  return TOOL_SEMANTIC_MAP[toolName] || 'other';
};

// ============================================================================
// 阶段判定逻辑
// ============================================================================

const PHASE_PRIORITY: RunPhase[] = ['retrieving', 'validating', 'synthesizing'];

const SEMANTIC_TO_PHASE: Record<ToolSemantic, RunPhase | null> = {
  retrieve: 'retrieving',
  validate: 'validating',
  synthesize_helper: 'synthesizing',
  other: null,
};

/**
 * 根据活跃工具集合判定当前阶段
 */
export const determinePhase = (
  activeToolSemantics: Set<ToolSemantic>,
  hasAssistantOutput: boolean
): RunPhase => {
  // 按优先级判定阶段
  for (const phase of PHASE_PRIORITY) {
    const semantic = phase === 'retrieving' ? 'retrieve' :
                     phase === 'validating' ? 'validate' : 'synthesize_helper';
    if (activeToolSemantics.has(semantic)) {
      return phase;
    }
  }

  // 若无活跃工具但有助手输出，则为 synthesizing
  if (hasAssistantOutput) {
    return 'synthesizing';
  }

  return 'retrieving';
};

// ============================================================================
// 过程概览计算
// ============================================================================

export interface ToolCallRecord {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  arguments?: Record<string, unknown>;
  result?: unknown;
  startedAt?: number;
  completedAt?: number;
  error?: string;
  sourceRefs?: EvidenceRef[];
}

export interface ProcessAccumulator {
  startTime: number;
  activeCalls: Map<string, ToolCallRecord>;
  completedCalls: ToolCallRecord[];
  errorCalls: ToolCallRecord[];
  assistantContent: string;
  eventVersion: number;
  sourceRefMap: Map<string, EvidenceRef>;
}

export const createProcessAccumulator = (): ProcessAccumulator => ({
  startTime: Date.now(),
  activeCalls: new Map(),
  completedCalls: [],
  errorCalls: [],
  assistantContent: '',
  eventVersion: 0,
  sourceRefMap: new Map(),
});

/**
 * 更新工具调用状态
 */
export const updateToolCall = (
  acc: ProcessAccumulator,
  callId: string,
  toolName: string,
  status: ToolCallRecord['status'],
  extras?: Partial<ToolCallRecord>
): void => {
  if (status === 'pending' || status === 'running') {
    // 添加或更新活跃调用
    const existing = acc.activeCalls.get(callId);
    acc.activeCalls.set(callId, {
      id: callId,
      name: toolName,
      status,
      ...existing,
      ...extras,
    });
  } else {
    // completed 或 error：从活跃移到完成/错误列表
    const existing = acc.activeCalls.get(callId);
    acc.activeCalls.delete(callId);

    const record: ToolCallRecord = {
      id: callId,
      name: toolName,
      status,
      ...existing,
      ...extras,
    };

    if (status === 'error') {
      acc.errorCalls.push(record);
    } else {
      acc.completedCalls.push(record);
    }
  }
};

/**
 * 合并来源引用
 */
export const mergeSourceRefs = (acc: ProcessAccumulator, refs: EvidenceRef[]): void => {
  refs.forEach((ref) => {
    const key = [ref.sourcePath, ref.heading ?? '', ref.charOffsetStart ?? '', ref.snippet ?? ''].join('|');
    if (!acc.sourceRefMap.has(key)) {
      acc.sourceRefMap.set(key, ref);
    }
  });
};

/**
 * 计算过程概览
 */
export const computeProcessOverview = (acc: ProcessAccumulator): ProcessOverview => {
  const activeSemantics = new Set<ToolSemantic>();
  acc.activeCalls.forEach((call) => {
    activeSemantics.add(getToolSemantic(call.name));
  });

  const phase = determinePhase(activeSemantics, acc.assistantContent.length > 0);
  const durationMs = Date.now() - acc.startTime;
  const warningCount = acc.errorCalls.length;
  const blockingErrorCount = acc.errorCalls.filter((call) =>
    call.status === 'error' && call.name === 'run_code_interpreter'
  ).length;

  const impact = blockingErrorCount > 0 ? 'blocking' :
                 warningCount > 0 ? 'partial' : 'none';

  return {
    phase,
    durationMs,
    warningCount,
    blockingErrorCount,
    impact,
  };
};

// ============================================================================
// 终态事件合成
// ============================================================================

/**
 * 计算 claimId
 */
export const computeClaimId = (claimText: string, refs: EvidenceRef[]): string => {
  const normalized = claimText.trim().toLowerCase();
  const sortedSources = refs
    .map((r) => `${r.sourcePath}#${r.charOffsetStart ?? 0}`)
    .sort()
    .join('|');
  // 简单 hash（非加密用途）
  const str = `${normalized}|${sortedSources}`;
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return `claim-${Math.abs(hash).toString(16)}`;
};

/**
 * 从工具调用记录生成证据列表
 */
export const generateEvidenceFromCalls = (calls: ToolCallRecord[]): EvidenceItem[] => {
  const evidence: EvidenceItem[] = [];

  calls.forEach((call) => {
    if (call.sourceRefs && call.sourceRefs.length > 0) {
      // 为每个来源创建证据项
      const claimText = `来源：${call.name}`;
      const claimId = computeClaimId(claimText, call.sourceRefs);

      evidence.push({
        claimId,
        claimText,
        refs: call.sourceRefs,
      });
    }
  });

  // 最多返回 10 条
  return evidence.slice(0, 10);
};

/**
 * 决策摘要生成规则
 * 注意：这是规则兜底实现，实际应使用 LLM 抽取
 */
export const generateDecisionSummaryFallback = (
  assistantContent: string,
  hasError: boolean,
  errorCount: number
): DecisionSummary => {
  // 从内容中提取首句作为结论
  const lines = assistantContent.split('\n').filter((line) => line.trim());
  const firstLine = lines[0] || '';

  // 简单截取（最多 200 字符）
  const conclusion = firstLine.slice(0, 200);

  // 根据错误情况设置置信度
  const confidence: ConfidenceLevel = hasError
    ? (errorCount > 2 ? 'low' : 'medium')
    : 'high';

  return {
    conclusion,
    actions: [],
    confidence,
    assumptions: [],
    risks: hasError ? ['部分工具执行失败，可能影响答案完整性'] : [],
    failureReason: hasError && !conclusion ? '未能生成有效回复' : undefined,
  };
};

/**
 * 计算完成状态
 */
export const computeCompletionState = (
  hasContent: boolean,
  errorCount: number,
  hasBlockingError: boolean
): CompletionState => {
  if (hasBlockingError || !hasContent) {
    return 'failed';
  }
  if (errorCount > 0) {
    return 'partial_completed';
  }
  return 'completed';
};

/**
 * 合成终态事件
 */
export const synthesizeFinalEvent = (acc: ProcessAccumulator): FinalEventPayload => {
  const processOverview = computeProcessOverview(acc);
  const evidence = generateEvidenceFromCalls(acc.completedCalls);

  const hasBlockingError = processOverview.blockingErrorCount > 0;
  const completionState = computeCompletionState(
    acc.assistantContent.length > 0,
    acc.errorCalls.length,
    hasBlockingError
  );

  const decisionSummary = generateDecisionSummaryFallback(
    acc.assistantContent,
    acc.errorCalls.length > 0,
    acc.errorCalls.length
  );

  // 递增事件版本
  acc.eventVersion += 1;

  return {
    event_kind: 'final',
    event_version: acc.eventVersion,
    decisionSummary,
    processOverview,
    completionState,
    evidence,
  };
};
