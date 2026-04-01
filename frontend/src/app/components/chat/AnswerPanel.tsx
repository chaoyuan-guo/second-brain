'use client';

import { useState, useMemo, type ReactNode } from 'react';
import type { ChatMessage, CitationRef } from '../../lib/chat-types';
import { sanitizeFailureReason } from '../../lib/chat-helpers';
import { FullResponseMarkdown } from './FullResponseMarkdown';
import { HonestyBanner } from './HonestyBanner';
import { InlineCitation } from './InlineCitation';
import {
  buildDisplayReferences,
  isPreciseCitationRef,
  normalizeCitationId,
  normalizeHonestySignalsWithReferences,
  sanitizeCitationSnippet,
} from '../../lib/citation-utils';

// ============================================================================
// 辅助函数
// ============================================================================

const hasLegacyStructuredData = (message: ChatMessage): boolean => {
  const { decisionSummary, processOverview, completionState, evidence } = message;
  if (!decisionSummary || typeof decisionSummary !== 'object') return false;
  if (typeof decisionSummary.conclusion !== 'string') return false;
  if (!processOverview || typeof processOverview !== 'object') return false;
  if (typeof processOverview.phase !== 'string') return false;
  if (!completionState || typeof completionState !== 'string') return false;
  if (!Array.isArray(evidence)) return false;
  return true;
};

const hasAnswerFirstPayload = (message: ChatMessage): boolean => {
  if (typeof message.directAnswer === 'string' && message.directAnswer.trim().length > 0) {
    return true;
  }
  if (typeof message.fullAnalysis === 'string' && message.fullAnalysis.trim().length > 0) {
    return true;
  }
  if (Array.isArray(message.references) && message.references.length > 0) {
    return true;
  }
  if (message.honestySignals) {
    return true;
  }
  return false;
};

/**
 * 渲染带行内引用的内容
 * 解析 [cxx] 标记并替换为 InlineCitation 组件
 */
const renderCitedContent = (
  content: string,
  citationMap: Record<string, CitationRef>,
  onOpenPreview?: (path: string, title: string, ref?: any) => void
): ReactNode => {
  const parts: ReactNode[] = [];
  let lastIndex = 0;
  
  // 匹配 [cxx] 格式
  const regex = /\[c(\d{2,3})\]/g;
  let match;
  
  while ((match = regex.exec(content)) !== null) {
    // 添加匹配前的文本
    if (match.index > lastIndex) {
      parts.push(content.slice(lastIndex, match.index));
    }
    
    const citationId = match[1]; // 统一使用数字格式 "01"
    const normalizedId = normalizeCitationId(citationId);
    const ref = citationMap[citationId] || citationMap[normalizedId] || citationMap[`c${normalizedId}`];
    
    if (isPreciseCitationRef(ref)) {
      parts.push(
        <InlineCitation
          key={`citation-${match.index}`}
          citationId={citationId}
          citationMap={citationMap}
          onOpenPreview={onOpenPreview}
        />
      );
    } else {
      parts.push(match[0]);
    }
    
    lastIndex = match.index + match[0].length;
  }
  
  // 添加剩余文本
  if (lastIndex < content.length) {
    parts.push(content.slice(lastIndex));
  }
  
  return parts;
};

// ============================================================================
// 组件 Props
// ============================================================================

interface AnswerPanelProps {
  message: ChatMessage;
  copiedKey: string | null;
  onCopyCode: (value: string, key: string) => void;
  onOpenPreview?: (path: string, title: string, ref?: { char_offset?: number; snippet?: string }) => void;
  onRetry?: () => void;
}

// ============================================================================
// AnswerPanel 组件
// ============================================================================

/**
 * AnswerPanel - 三段式答案面板
 * 
 * 布局结构：
 * 1. 直接回答（Direct Answer）- 最优先展示
 * 2. 完整分析（Full Analysis）- 可折叠的详细分析
 * 
 * 诚实性原则：当证据不足时，显式展示 HonestyBanner
 */
export function AnswerPanel({
  message,
  copiedKey,
  onCopyCode,
  onOpenPreview,
  onRetry,
}: AnswerPanelProps) {
  const [isAnalysisExpanded, setIsAnalysisExpanded] = useState(false);
  
  // 构建引用映射表
  const citationMap = useMemo(() => {
    const map: Record<string, CitationRef> = {};
    if (message.citationMap) {
      Object.entries(message.citationMap).forEach(([id, ref]) => {
        if (!isPreciseCitationRef(ref)) {
          return;
        }
        const normalizedRef = {
          ...ref,
          snippet: sanitizeCitationSnippet(ref.snippet),
        };
        map[id] = normalizedRef;
        const normalized = id.startsWith('c') ? id.slice(1) : id;
        map[normalized] = normalizedRef;
        map[`c${normalized}`] = normalizedRef;
      });
      return map;
    }
    if (message.references) {
      message.references.forEach(ref => {
        if (!isPreciseCitationRef(ref)) {
          return;
        }
        const normalizedRef = {
          ...ref,
          snippet: sanitizeCitationSnippet(ref.snippet),
        };
        map[ref.id] = normalizedRef;
        const normalized = ref.id.startsWith('c') ? ref.id.slice(1) : ref.id;
        map[normalized] = normalizedRef;
        map[`c${normalized}`] = normalizedRef;
      });
    }
    return map;
  }, [message.citationMap, message.references]);
  
  const hasStructured = useMemo(
    () => hasAnswerFirstPayload(message) || hasLegacyStructuredData(message),
    [message],
  );
  const displayReferences = useMemo(() => buildDisplayReferences(message), [message]);
  const effectiveHonestySignals = useMemo(
    () => normalizeHonestySignalsWithReferences(message.honestySignals, displayReferences),
    [displayReferences, message.honestySignals],
  );
  const isFailedState = message.completionState === 'failed' || message.displayState === 'failed';

  if (isFailedState) {
    const safeFailureDetail = sanitizeFailureReason(message.content);
    return (
      <div className="answer-panel failed">
        <div className="failure-card">
          <p className="failure-title">未能形成可靠结论</p>
          <p className="failure-reason">
            {sanitizeFailureReason(message.decisionSummary?.failureReason ?? message.content)}
          </p>
          {onRetry && (
            <button type="button" className="pill-btn primary" onClick={onRetry}>
              重试这个问题
            </button>
          )}
        </div>
        <div className="direct-answer">{safeFailureDetail}</div>
      </div>
    );
  }

  // 降级渲染：缺失结构化字段时，展示原始回答正文和降级提示
  if (!hasStructured) {
    return (
      <div className="answer-panel fallback-answer">
        {effectiveHonestySignals && !effectiveHonestySignals.hasSufficientEvidence && (
          <HonestyBanner signals={effectiveHonestySignals} />
        )}
        <section className="answer-section direct-answer-section fallback-answer-section">
          <h2 className="section-title">回答</h2>
          <FullResponseMarkdown
            content={message.content}
            messageId={message.id}
            copiedKey={copiedKey}
            onCopyCode={onCopyCode}
            citationMap={citationMap}
            onOpenPreview={onOpenPreview}
            title={null}
          />
        </section>
      </div>
    );
  }

  const {
    decisionSummary,
    directAnswer,
    fullAnalysis,
  } = message;

  // 使用直接回答或从完整内容中提取
  const answerContent = directAnswer || decisionSummary?.conclusion || message.content.split('\n')[0];
  const analysisContent = fullAnalysis || message.content;
  const hasDistinctAnalysis =
    analysisContent.trim().length > 0 &&
    analysisContent.trim() !== answerContent.trim();

  return (
    <div className="answer-panel evidence-traceable">
      {/* 1. 诚实性提示横幅（当证据不足时） */}
      {effectiveHonestySignals && !effectiveHonestySignals.hasSufficientEvidence && (
        <HonestyBanner signals={effectiveHonestySignals} />
      )}

      {/* 2. 直接回答（Direct Answer）- 最优先展示 */}
      <section className="answer-section direct-answer-section">
        <h2 className="section-title">回答</h2>
        <div className="direct-answer-content">
          <FullResponseMarkdown
            content={answerContent}
            messageId={`${message.id}-direct-answer`}
            copiedKey={copiedKey}
            onCopyCode={onCopyCode}
            citationMap={citationMap}
            onOpenPreview={onOpenPreview}
            title={null}
          />
        </div>
      </section>

      {/* 3. 完整分析（Full Analysis）- 可折叠 */}
      {hasDistinctAnalysis && (
        <section className="answer-section analysis-section">
          <button
            className={`analysis-toggle ${isAnalysisExpanded ? 'expanded' : ''}`}
            onClick={() => setIsAnalysisExpanded(!isAnalysisExpanded)}
          >
            <span className="toggle-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d={isAnalysisExpanded ? "M6 15l6-6 6 6" : "M6 9l6 6 6-6"} />
              </svg>
            </span>
            <span className="toggle-label">
              {isAnalysisExpanded ? '收起完整分析' : '展开完整分析'}
            </span>
            {analysisContent.length > 500 && (
              <span className="toggle-hint">（详细推理过程）</span>
            )}
          </button>

          {isAnalysisExpanded && (
            <div className="full-analysis-content">
              <FullResponseMarkdown
                content={analysisContent}
                messageId={message.id}
                copiedKey={copiedKey}
                onCopyCode={onCopyCode}
                citationMap={citationMap}
                onOpenPreview={onOpenPreview}
              />
            </div>
          )}
        </section>
      )}

      {/* 4. 旧版兼容：证据面板（如果存在 legacy evidence 数据） */}
      {message.evidence && message.evidence.length > 0 && !message.references?.length && (
        <section className="answer-section legacy-evidence">
          <div className="legacy-notice">以下是旧版证据展示</div>
        </section>
      )}
    </div>
  );
}
