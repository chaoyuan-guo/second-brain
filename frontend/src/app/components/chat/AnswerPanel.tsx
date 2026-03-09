'use client';

import { useState, useMemo, type ReactNode } from 'react';
import type { ChatMessage, CitationRef } from '../../lib/chat-types';
import { FullResponseMarkdown } from './FullResponseMarkdown';
import { ReferencesPanel } from './ReferencesPanel';
import { HonestyBanner } from './HonestyBanner';
import { InlineCitation } from './InlineCitation';
import {
  deriveSourceTitle,
  extractFileLevelReferencesFromContent,
  inferSourceDateLabel,
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
    
    // 使用 InlineCitation 组件
    parts.push(
      <InlineCitation
        key={`citation-${match.index}`}
        citationId={citationId}
        citationMap={citationMap}
        onOpenPreview={onOpenPreview}
      />
    );
    
    lastIndex = match.index + match[0].length;
  }
  
  // 添加剩余文本
  if (lastIndex < content.length) {
    parts.push(content.slice(lastIndex));
  }
  
  return parts;
};

const extractContentFallbackReferences = (message: ChatMessage): CitationRef[] | null => {
  const contentPool = [message.directAnswer, message.fullAnalysis, message.content]
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
    .join('\n');
  if (!contentPool) {
    return null;
  }
  const refs = extractFileLevelReferencesFromContent(contentPool);
  return refs.length > 0 ? refs : null;
};

const buildFallbackReferences = (message: ChatMessage): CitationRef[] | null => {
  if (message.references && message.references.length > 0) {
    return message.references.map((ref) => ({
      ...ref,
      snippet: sanitizeCitationSnippet(ref.snippet),
    }));
  }
  if (message.sourceRefs && message.sourceRefs.length > 0) {
    return message.sourceRefs.map((ref, idx) => ({
      id: String(idx + 1).padStart(2, '0'),
      sourcePath: ref.path,
      sourceTitle: deriveSourceTitle(ref.path, undefined, ref.heading),
      sourceDateLabel: inferSourceDateLabel(ref.path, ref.heading),
      heading: ref.heading,
      snippet: sanitizeCitationSnippet(ref.snippet),
      charOffsetStart: ref.char_offset,
    }));
  }
  return extractContentFallbackReferences(message);
};

// ============================================================================
// 组件 Props
// ============================================================================

interface AnswerPanelProps {
  message: ChatMessage;
  copiedKey: string | null;
  onCopyCode: (value: string, key: string) => void;
  onOpenPreview?: (path: string, title: string, ref?: { char_offset?: number; snippet?: string }) => void;
}

// ============================================================================
// AnswerPanel 组件
// ============================================================================

/**
 * AnswerPanel - 三段式答案面板
 * 
 * 布局结构：
 * 1. 直接回答（Direct Answer）- 最优先展示
 * 2. 来自你的笔记（References Panel）- 展示引用来源
 * 3. 完整分析（Full Analysis）- 可折叠的详细分析
 * 
 * 诚实性原则：当证据不足时，显式展示 HonestyBanner
 */
export function AnswerPanel({
  message,
  copiedKey,
  onCopyCode,
  onOpenPreview,
}: AnswerPanelProps) {
  const [isAnalysisExpanded, setIsAnalysisExpanded] = useState(false);
  
  // 构建引用映射表
  const citationMap = useMemo(() => {
    const map: Record<string, CitationRef> = {};
    if (message.citationMap) {
      Object.entries(message.citationMap).forEach(([id, ref]) => {
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
  const displayReferences = useMemo(() => buildFallbackReferences(message), [message]);
  const effectiveHonestySignals = useMemo(
    () => normalizeHonestySignalsWithReferences(message.honestySignals, displayReferences),
    [displayReferences, message.honestySignals],
  );

  // 降级渲染：缺失结构化字段时，展示原始回答正文和降级提示
  if (!hasStructured) {
    return (
      <div className="answer-panel fallback-answer">
        {effectiveHonestySignals && !effectiveHonestySignals.hasSufficientEvidence && (
          <HonestyBanner signals={effectiveHonestySignals} />
        )}
        <section className="answer-section direct-answer-section fallback-answer-section">
          <h2 className="section-title">完整回答</h2>
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
        {displayReferences && displayReferences.length > 0 && (
          <section className="answer-section references-section">
            <ReferencesPanel references={displayReferences} onOpenPreview={onOpenPreview} />
          </section>
        )}
      </div>
    );
  }

  const {
    decisionSummary,
    completionState,
    directAnswer,
    fullAnalysis,
  } = message;

  // 失败状态渲染
  if (completionState === 'failed') {
    return (
      <div className="answer-panel failed">
        <div className="failure-card">
          <p className="failure-title">未能形成可靠结论</p>
          <p className="failure-reason">
            {decisionSummary?.failureReason || '请重试或缩小问题范围'}
          </p>
        </div>
        <div className="direct-answer">
          {renderCitedContent(message.content, citationMap, onOpenPreview)}
        </div>
      </div>
    );
  }

  // 使用直接回答或从完整内容中提取
  const answerContent = directAnswer || decisionSummary?.conclusion || message.content.split('\n')[0];
  const analysisContent = fullAnalysis || message.content;

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

      {/* 3. 来自你的笔记（References Panel）- 支持 sourceRefs 降级 */}
      {displayReferences && displayReferences.length > 0 && (
        <section className="answer-section references-section">
          <ReferencesPanel 
            references={displayReferences} 
            onOpenPreview={onOpenPreview}
          />
        </section>
      )}

      {/* 4. 完整分析（Full Analysis）- 可折叠 */}
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

      {/* 5. 旧版兼容：证据面板（如果存在 legacy evidence 数据） */}
      {message.evidence && message.evidence.length > 0 && !message.references?.length && (
        <section className="answer-section legacy-evidence">
          <div className="legacy-notice">以下是旧版证据展示</div>
        </section>
      )}
    </div>
  );
}
