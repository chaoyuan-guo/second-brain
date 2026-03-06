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
  CitationRef,
  ProcessStepSummary,
  HonestySignals,
} from '../../lib/chat-types';
import {
  deriveSourceTitle,
  extractFileLevelReferencesFromContent,
  inferSourceDateLabel,
  isWeakRetrievalScore,
  normalizeHonestySignalsWithReferences,
  normalizeCitationId,
  sanitizeCitationSnippet,
} from '../../lib/citation-utils';

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
  // 证据与透明性新增字段
  directAnswer?: string;
  fullAnalysis?: string;
  references?: CitationRef[];
  processSummary?: ProcessStepSummary[];
  honestySignals?: HonestySignals;
}

// ============================================================================
// 工具名到语义阶段的映射
// ============================================================================

const TOOL_SEMANTIC_MAP: Record<string, ToolSemantic> = {
  query_my_notes: 'retrieve',
  web_search: 'retrieve',
  read_page: 'retrieve',
  read_note_file: 'retrieve',
  read: 'retrieve',
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

const getResultCount = (result: unknown): number => {
  if (Array.isArray(result)) {
    return result.length;
  }
  if (result && typeof result === 'object' && Array.isArray((result as { results?: unknown[] }).results)) {
    return (result as { results: unknown[] }).results.length;
  }
  return 0;
};

const getUniqueFileCount = (call: ToolCallRecord): number => {
  if (call.sourceRefs && call.sourceRefs.length > 0) {
    return new Set(call.sourceRefs.map((ref) => ref.sourcePath).filter(Boolean)).size;
  }
  if (call.result && typeof call.result === 'object' && Array.isArray((call.result as { results?: Array<{ source_path?: string }> }).results)) {
    const files = (call.result as { results: Array<{ source_path?: string }> }).results
      .map((item) => item?.source_path)
      .filter((value): value is string => Boolean(value));
    return new Set(files).size;
  }
  return 0;
};

const summarizeInterpreterResult = (result: unknown): string => {
  if (!result || typeof result !== 'object') {
    return '已完成验证';
  }

  const record = result as Record<string, unknown>;
  const status = typeof record.status === 'number' ? record.status : undefined;
  const stdout = typeof record.stdout === 'string' ? record.stdout.trim() : '';
  const stderr = typeof record.stderr === 'string' ? record.stderr.trim() : '';
  const content = stdout || stderr;

  if (content) {
    return content.split('\n')[0].slice(0, 72);
  }
  if (status === 0) {
    return '验证通过';
  }
  if (typeof status === 'number') {
    return `退出码 ${status}`;
  }
  return '已完成验证';
};

const getStringValue = (
  record: Record<string, unknown> | undefined,
  keys: string[],
): string | undefined => {
  if (!record) {
    return undefined;
  }
  for (const key of keys) {
    const value = record[key];
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }
  return undefined;
};

const getNumberValue = (
  record: Record<string, unknown> | undefined,
  keys: string[],
): number | undefined => {
  if (!record) {
    return undefined;
  }
  for (const key of keys) {
    const value = record[key];
    if (typeof value === 'number' && !Number.isNaN(value)) {
      return value;
    }
    if (typeof value === 'string' && value.trim()) {
      const parsed = Number.parseInt(value, 10);
      if (!Number.isNaN(parsed)) {
        return parsed;
      }
    }
  }
  return undefined;
};

const extractSnippetPreview = (result: unknown): string | undefined => {
  if (typeof result === 'string') {
    const normalized = sanitizeCitationSnippet(result);
    return normalized ? normalized.slice(0, 220) : undefined;
  }

  if (!result || typeof result !== 'object') {
    return undefined;
  }

  const record = result as Record<string, unknown>;
  const raw = getStringValue(record, ['content', 'text', 'snippet', 'output']);
  if (!raw) {
    return undefined;
  }

  const normalized = sanitizeCitationSnippet(raw);
  return normalized ? normalized.slice(0, 220) : undefined;
};

const extractPathFromCall = (call: ToolCallRecord): string | undefined => {
  const fromArgs = getStringValue(call.arguments, [
    'path',
    'filePath',
    'file_path',
    'sourcePath',
    'source_path',
  ]);
  if (fromArgs) {
    return fromArgs;
  }

  if (!call.result || typeof call.result !== 'object') {
    return undefined;
  }

  const record = call.result as Record<string, unknown>;
  return getStringValue(record, [
    'path',
    'filePath',
    'file_path',
    'sourcePath',
    'source_path',
  ]);
};

const deriveReferencesFromCalls = (calls: ToolCallRecord[]): CitationRef[] => {
  const refs: CitationRef[] = [];
  const seen = new Set<string>();

  calls.forEach((call) => {
    if (call.name !== 'read' && call.name !== 'read_note_file') {
      return;
    }

    const sourcePath = extractPathFromCall(call);
    if (!sourcePath) {
      return;
    }

    const charOffsetStart = getNumberValue(call.arguments, ['offset', 'char_offset', 'charOffset']);
    const dedupeKey = `${sourcePath}#${charOffsetStart ?? ''}`;
    if (seen.has(dedupeKey)) {
      return;
    }
    seen.add(dedupeKey);

    refs.push({
      id: String(refs.length + 1).padStart(2, '0'),
      sourcePath,
      sourceTitle: deriveSourceTitle(sourcePath),
      sourceDateLabel: inferSourceDateLabel(sourcePath),
      charOffsetStart,
      snippet: extractSnippetPreview(call.result),
    });
  });

  return refs;
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

// ============================================================================
// 证据与透明性：核心辅助函数
// ============================================================================

/**
 * 从 assistantContent 中提取 [cxx] 引用标记
 * 支持格式：[c01]、[c12] 等
 */
export const extractCitations = (content: string): CitationRef[] => {
  const citations: CitationRef[] = [];
  const seen = new Set<string>();

  // 匹配 [cxx] 格式，x 为数字，仅提取数字部分
  const regex = /\[c(\d{2,3})\]/g;
  let match;

  while ((match = regex.exec(content)) !== null) {
    const id = normalizeCitationId(match[1]);
    if (!seen.has(id)) {
      seen.add(id);
      citations.push({
        id,
        sourcePath: '', // 临时值，后续 enrichCitationsWithEvidence 会填充
        retrievalScore: undefined,
        snippet: undefined,
      });
    }
  }

  return citations;
};

/**
 * 将 citation 与 sourceRefMap 中的证据信息关联
 */
export const enrichCitationsWithEvidence = (
  citations: CitationRef[],
  sourceRefMap: Map<string, EvidenceRef>
): CitationRef[] => {
  return citations.map((citation) => {
    // 查找匹配的 source ref（兼容 "c01" 和 "01" 两种 citationId 格式）
    const citationIdWithPrefix = citation.id.startsWith('c') ? citation.id : `c${citation.id}`;
    const citationIdWithoutPrefix = citation.id.startsWith('c') ? citation.id.slice(1) : citation.id;
    
    for (const [, ref] of sourceRefMap) {
      if (ref.citationId === citationIdWithPrefix || ref.citationId === citationIdWithoutPrefix || ref.citationId === citation.id) {
        return {
          ...citation,
          sourcePath: ref.sourcePath,
          sourceTitle: ref.sourceTitle,
          sourceDateLabel: ref.sourceDateLabel ?? inferSourceDateLabel(ref.sourcePath, ref.sourceTitle, ref.heading),
          heading: ref.heading,
          snippet: sanitizeCitationSnippet(ref.snippet),
          retrievalScore: ref.retrievalScore,
          weakMatch: isWeakRetrievalScore(ref.retrievalScore),
        };
      }
    }
    return citation;
  });
};

/**
 * 将内容拆分为直接回答和完整分析
 * 
 * 策略：
 * 1. 查找 "---" 分隔符
 * 2. 查找 "完整分析" 或 "详细分析" 等标记
 * 3. 默认：前 2-3 句为直接回答，剩余为完整分析
 */
export const splitDirectAnswer = (content: string): { directAnswer: string; fullAnalysis: string } => {
  // 尝试查找分隔符
  const separatorMatch = content.match(/\n---\s*\n/);
  if (separatorMatch) {
    const directAnswer = content.slice(0, separatorMatch.index).trim();
    const fullAnalysis = content.slice(separatorMatch.index! + separatorMatch[0].length).trim();
    return { directAnswer, fullAnalysis };
  }

  // 尝试查找 "完整分析" 或 "详细分析" 标记
  const analysisMarkers = ['完整分析', '详细分析', '详细说明', '展开说明', '详细过程'];
  for (const marker of analysisMarkers) {
    const markerIndex = content.indexOf(marker);
    if (markerIndex > 0) { // 只要找到标记就可以分割
      const directAnswer = content.slice(0, markerIndex).trim();
      const fullAnalysis = content.slice(markerIndex).trim();
      return { directAnswer, fullAnalysis };
    }
  }

  // 默认策略：按句子分割，前 2-3 句为直接回答。
  // 仅在英文句号/问号/感叹号后存在空白或已到结尾时分割，避免把 ".md" 这类文件扩展名误切开。
  const sentences = content
    .split(/(?<=[。！？])\s*|(?<=[.!?])(?=\s|$)/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
  if (sentences.length <= 2) {
    return { directAnswer: content.trim(), fullAnalysis: '' };
  }

  // 取前 2-3 句（确保直接回答在合理长度内）
  let directAnswerSentences = 2;
  let directAnswer = sentences.slice(0, directAnswerSentences).join('');
  
  // 如果太短，增加一句
  if (directAnswer.length < 80 && sentences.length > 2) {
    directAnswerSentences = 3;
    directAnswer = sentences.slice(0, directAnswerSentences).join('');
  }

  const fullAnalysis = sentences.slice(directAnswerSentences).join('');
  
  return { directAnswer: directAnswer.trim(), fullAnalysis: fullAnalysis.trim() };
};

/**
 * 从工具调用记录生成语义化过程摘要
 */
export const generateProcessSummary = (
  completedCalls: ToolCallRecord[],
  errorCalls: ToolCallRecord[]
): ProcessStepSummary[] => {
  const summaries: ProcessStepSummary[] = [];

  // 按时间排序
  const allCalls = [...completedCalls, ...errorCalls].sort((a, b) => 
    (a.startedAt || 0) - (b.startedAt || 0)
  );

  allCalls.forEach((call, index) => {
    const stepNumber = index + 1;
    const semantic = getToolSemantic(call.name);
    
    // 根据工具语义生成摘要
    let summary = '';
    let detail = '';
    let inputSummary = '';
    let resultSummary = '';

    switch (semantic) {
      case 'retrieve':
        if (call.name === 'query_my_notes') {
          const query = (call.arguments?.query as string) || '相关笔记';
          const resultCount = getResultCount(call.result);
          const fileCount = getUniqueFileCount(call);
          summary = fileCount > 0
            ? `检索笔记 "${query}" -> 命中 ${resultCount} 条，来自 ${fileCount} 个文件`
            : `检索笔记 "${query}" -> 命中 ${resultCount} 条`;
          inputSummary = `检索词 "${query}"`;
          resultSummary = fileCount > 0
            ? `命中 ${resultCount} 条，涉及 ${fileCount} 个文件`
            : `命中 ${resultCount} 条`;
          detail = `输入：${inputSummary}；结果：${resultSummary}`;
        } else if (call.name === 'web_search') {
          const query = (call.arguments?.query as string) || '';
          summary = query ? `联网搜索 "${query}"` : '联网搜索';
          inputSummary = query ? `搜索词 "${query}"` : '无输入';
          resultSummary = call.status === 'error' ? '搜索失败' : '已返回结果';
          detail = `输入：${inputSummary}；结果：${resultSummary}`;
        } else if (call.name === 'read_note_file' || call.name === 'read') {
          const path = extractPathFromCall(call) || '文件';
          const offsetValue = call.arguments?.offset;
          const offset = typeof offsetValue === 'number'
            ? offsetValue
            : typeof offsetValue === 'string'
              ? Number.parseInt(offsetValue, 10)
              : undefined;
          summary = offset && !Number.isNaN(offset)
            ? `读取文件 ${path.split('/').pop() || path}（偏移 ${offset} 附近）`
            : `读取文件 ${path.split('/').pop() || path}`;
          inputSummary = `读取 ${path}`;
          resultSummary = call.status === 'error' ? '读取失败' : '已载入原文片段';
          detail = `输入：${inputSummary}；结果：${resultSummary}`;
        } else if (call.name === 'grep') {
          const pattern = (call.arguments?.pattern as string) || (call.arguments?.query as string) || '关键词';
          summary = `搜索文件内容 "${pattern}" -> 命中 ${getResultCount(call.result)} 条`;
          inputSummary = `搜索词 "${pattern}"`;
          resultSummary = call.status === 'error' ? '搜索失败' : `命中 ${getResultCount(call.result)} 条`;
          detail = `输入：${inputSummary}；结果：${resultSummary}`;
        } else if (call.name === 'glob') {
          const pattern = (call.arguments?.pattern as string) || (call.arguments?.query as string) || '文件模式';
          summary = `定位候选文件 "${pattern}" -> 命中 ${getResultCount(call.result)} 条`;
          inputSummary = `文件模式 "${pattern}"`;
          resultSummary = call.status === 'error' ? '枚举失败' : `命中 ${getResultCount(call.result)} 条`;
          detail = `输入：${inputSummary}；结果：${resultSummary}`;
        } else {
          summary = `检索信息：${call.name}`;
          inputSummary = '执行检索';
          resultSummary = call.status === 'error' ? '执行失败' : '执行完成';
          detail = `输入：${inputSummary}；结果：${resultSummary}`;
        }
        break;
      
      case 'validate':
        summary = call.name === 'run_code_interpreter'
          ? `执行代码验证 -> ${summarizeInterpreterResult(call.result)}`
          : `验证步骤：${call.name}`;
        inputSummary = '执行验证';
        resultSummary = call.status === 'error'
          ? '验证失败'
          : summarizeInterpreterResult(call.result);
        detail = `输入：${inputSummary}；结果：${resultSummary}`;
        break;
      
      case 'synthesize_helper':
        summary = `辅助分析：${call.name}`;
        inputSummary = '执行辅助分析';
        resultSummary = call.status === 'error' ? '执行失败' : '执行完成';
        detail = `输入：${inputSummary}；结果：${resultSummary}`;
        break;
      
      default:
        summary = `执行操作：${call.name}`;
        inputSummary = '执行操作';
        resultSummary = call.status === 'error' ? '执行失败' : '执行完成';
        detail = `输入：${inputSummary}；结果：${resultSummary}`;
    }

    // 处理错误状态
    if (call.status === 'error') {
      summary = `${summary}（失败）`;
      const errorText = call.error || '执行出错';
      detail = detail ? `${detail}；错误：${errorText}` : errorText;
    }

    summaries.push({
      stepNumber,
      phase: SEMANTIC_TO_PHASE[semantic] || 'synthesizing',
      summary,
      detail,
      toolName: call.name,
      stepId: call.id,
      semanticType: semantic,
      inputSummary,
      resultSummary,
      status: call.status === 'error' ? 'error' : 'completed',
      durationMs: call.completedAt && call.startedAt 
        ? call.completedAt - call.startedAt 
        : undefined,
    });
  });

  return summaries;
};

/**
 * 计算诚实性信号
 * 基于文档第9节：诚实性优先原则
 */
export const computeHonestySignals = (
  citations: CitationRef[],
  hasErrorCalls: boolean,
  errorCount: number
): HonestySignals => {
  // 强匹配：retrievalScore < 0.8 的引用（L2 距离越小越相似）
  const strongMatches = citations.filter(
    (c) => c.retrievalScore !== undefined && c.retrievalScore < 0.8
  );

  // 弱匹配：retrievalScore >= 0.8 的引用（距离大，相关性低）
  const weakMatches = citations.filter(
    (c) => c.retrievalScore !== undefined && c.retrievalScore >= 0.8
  );

  // 无分数的引用（无法验证）
  const unscoredMatches = citations.filter((c) => c.retrievalScore === undefined);

  // 证据质量：基于强匹配比例
  let evidenceQuality: 'strong' | 'partial' | 'weak' | 'none' = 'none';
  if (citations.length === 0) {
    evidenceQuality = 'none';
  } else if (strongMatches.length >= 3) {
    evidenceQuality = 'strong';
  } else if (strongMatches.length >= 1) {
    // 有强匹配但不足 3 个
    evidenceQuality = 'partial';
  } else if (weakMatches.length >= 1 || unscoredMatches.length >= 1) {
    // 无强匹配，只有弱匹配或无分数
    evidenceQuality = 'weak';
  }

  const hasSufficientEvidence = strongMatches.length >= 2;

  const reasonCodes: Array<'no_hit' | 'weak_match' | 'insufficient_hits'> = [];
  if (citations.length === 0) {
    reasonCodes.push('no_hit');
  } else if (strongMatches.length === 0) {
    reasonCodes.push('weak_match');
  } else if (!hasSufficientEvidence) {
    reasonCodes.push('insufficient_hits');
  }

  // 诚实性提示
  const honestyWarnings: string[] = [];
  
  if (weakMatches.length > 0) {
    honestyWarnings.push(
      `${weakMatches.length} 条引用来自相关性较低的检索结果（距离 >= 0.8），建议进一步核实`
    );
  }
  
  if (unscoredMatches.length > 0) {
    honestyWarnings.push(
      `${unscoredMatches.length} 条引用缺少相关性评分`
    );
  }
  
  if (hasErrorCalls) {
    honestyWarnings.push(
      `${errorCount} 个工具执行失败，可能影响答案完整性`
    );
  }

  // 局限性说明
  let limitationNote: string | undefined;
  if (reasonCodes.includes('no_hit')) {
    limitationNote = '笔记中没有检索到直接相关记录，回答只能基于有限线索推断。';
  } else if (reasonCodes.includes('weak_match')) {
    limitationNote = '检索结果相关性偏弱，请优先核对原文后再采纳结论。';
  } else if (reasonCodes.includes('insufficient_hits')) {
    limitationNote = '已命中部分证据，但数量不足以形成高置信结论。';
  } else if (weakMatches.length > strongMatches.length) {
    limitationNote = '主要引用来源相关性较低，请谨慎采纳。';
  }

  return {
    reasonCodes,
    evidenceQuality,
    weakMatches: weakMatches.map((c) => c.id),
    unscoredMatches: unscoredMatches.map((c) => c.id),
    honestyWarnings,
    limitationNote,
    hasSufficientEvidence,
    hasDirectEvidence: strongMatches.length > 0,
    retrievalHitCount: citations.length,
    bestScore: citations
      .map((c) => c.retrievalScore)
      .filter((v): v is number => typeof v === 'number')
      .sort((a, b) => a - b)[0],
  };
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

  // 提取引用
  const citations = extractCitations(acc.assistantContent);

  // 丰富引用信息（关联 sourceRefMap 中的证据详情）
  const enrichedCitations = enrichCitationsWithEvidence(citations, acc.sourceRefMap);
  const fallbackReferences = deriveReferencesFromCalls(acc.completedCalls);
  const contentFallbackReferences = extractFileLevelReferencesFromContent(acc.assistantContent);
  const references = enrichedCitations.length > 0
    ? enrichedCitations
    : fallbackReferences.length > 0
      ? fallbackReferences
      : contentFallbackReferences;

  // 拆分直接回答和完整分析
  const { directAnswer, fullAnalysis } = splitDirectAnswer(acc.assistantContent);

  // 生成过程摘要
  const processSummary = generateProcessSummary(acc.completedCalls, acc.errorCalls);

  // 计算诚实性信号
  const honestySignals = normalizeHonestySignalsWithReferences(
    computeHonestySignals(
      references,
      acc.errorCalls.length > 0,
      acc.errorCalls.length
    ),
    references,
  );

  // 递增事件版本
  acc.eventVersion += 1;

  return {
    event_kind: 'final',
    event_version: acc.eventVersion,
    decisionSummary,
    processOverview: {
      ...processOverview,
      phase: 'completed',
    },
    completionState,
    evidence,
    // 新增字段
    directAnswer: directAnswer || undefined,
    fullAnalysis: fullAnalysis || undefined,
    references: references.length > 0 ? references : undefined,
    processSummary: processSummary.length > 0 ? processSummary : undefined,
    honestySignals: !honestySignals || honestySignals.hasSufficientEvidence ? undefined : honestySignals,
  };
};
