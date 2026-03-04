'use client';

import { useState, useMemo } from 'react';
import type { ChatMessage, CitationRef } from '../../lib/chat-types';
import { FullResponseMarkdown } from './FullResponseMarkdown';
import { ReferencesPanel } from './ReferencesPanel';
import { HonestyBanner } from './HonestyBanner';
import { InlineCitation } from './InlineCitation';

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 判断是否具备有效结构化数据
 * - null/undefined/错误类型视为缺失
 * - 空数组/空字符串是合法值
 */
const hasValidStructuredData = (message: ChatMessage): boolean => {
  const { decisionSummary, processOverview, completionState, evidence } = message;

  // 检查 decisionSummary
  if (!decisionSummary || typeof decisionSummary !== 'object') return false;
  if (typeof decisionSummary.conclusion !== 'string') return false;

  // 检查 processOverview
  if (!processOverview || typeof processOverview !== 'object') return false;
  if (typeof processOverview.phase !== 'string') return false;

  // 检查 completionState
  if (!completionState || typeof completionState !== 'string') return false;

  // 检查 evidence（可以是空数组，但必须是数组）
  if (!Array.isArray(evidence)) return false;

  return true;
};

/**
 * 渲染带行内引用的内容
 * 解析 [cxx] 标记并替换为 InlineCitation 组件
 */
const renderCitedContent = (
  content: string,
  citationMap: Record<string, CitationRef>,
  onOpenPreview?: (path: string, title: string, ref?: any) => void
): React.ReactNode => {
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  
  // 匹配 [cxx] 格式
  const regex = /\[c(\d{2,3})\]/g;
  let match;
  
  while ((match = regex.exec(content)) !== null) {
    // 添加匹配前的文本
    if (match.index > lastIndex) {
      parts.push(content.slice(lastIndex, match.index));
    }
    
    const citationId = `c${match[1]}`; // 完整格式 "c01"
    
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
    const map: Record<string, any> = {};
    if (message.citationMap) {
      return message.citationMap;
    }
    if (message.references) {
      message.references.forEach(ref => {
        map[ref.id] = ref;
      });
    }
    return map;
  }, [message.citationMap, message.references]);
  
  const hasStructured = useMemo(() => hasValidStructuredData(message), [message]);

  // 降级渲染：缺失结构化字段时，展示原始回答正文和降级提示
  if (!hasStructured) {
    return (
      <div className="answer-panel degraded">
        <div className="degraded-notice">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span>结构化数据暂不可用，以下为原始回答</span>
        </div>
        <div className="direct-answer">
          {renderCitedContent(message.content, citationMap, onOpenPreview)}
        </div>
      </div>
    );
  }

  const { 
    decisionSummary, 
    completionState, 
    directAnswer, 
    fullAnalysis,
    references,
    honestySignals 
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

  // 构建引用列表（优先使用 references，降级使用 sourceRefs）
  const displayReferences = useMemo(() => {
    if (references && references.length > 0) {
      return references;
    }
    // 降级：从 sourceRefs 构建文件级引用
    if (message.sourceRefs && message.sourceRefs.length > 0) {
      return message.sourceRefs.map((ref, idx) => ({
        id: `fallback-${idx}`,
        sourcePath: ref.path,
        sourceTitle: ref.heading || ref.path.split('/').pop(),
        heading: ref.heading,
        snippet: ref.snippet,
        charOffsetStart: ref.char_offset,
      }));
    }
    return null;
  }, [references, message.sourceRefs]);

  return (
    <div className="answer-panel evidence-traceable">
      {/* 1. 诚实性提示横幅（当证据不足时） */}
      {honestySignals && !honestySignals.hasSufficientEvidence && (
        <HonestyBanner signals={honestySignals} />
      )}

      {/* 2. 直接回答（Direct Answer）- 最优先展示 */}
      <section className="answer-section direct-answer-section">
        <h2 className="section-title">回答</h2>
        <div className="direct-answer-content">
          {renderCitedContent(answerContent, citationMap, onOpenPreview)}
        </div>
      </section>

      {/* 3. 来自你的笔记（References Panel）- 支持 sourceRefs 降级 */}
      {displayReferences && displayReferences.length > 0 && (
        <section className="answer-section references-section">
          {references && references.length > 0 ? (
            <ReferencesPanel 
              references={references} 
              onOpenPreview={onOpenPreview}
            />
          ) : (
            <div className="fallback-references">
              <h3 className="fallback-title">相关文件来源</h3>
              <ul className="fallback-list">
                {displayReferences.map((ref, idx) => (
                  <li key={idx} className="fallback-item">
                    <button
                      className="fallback-link"
                      onClick={() => onOpenPreview?.(
                        ref.sourcePath,
                        ref.sourceTitle || ref.sourcePath.split('/').pop() || '来源',
                        { char_offset: ref.charOffsetStart, snippet: ref.snippet }
                      )}
                    >
                      {ref.sourceTitle || ref.sourcePath.split('/').pop()}
                    </button>
                    {ref.snippet && (
                      <p className="fallback-snippet">{ref.snippet.slice(0, 120)}...</p>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
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
            />
          </div>
        )}
      </section>

      {/* 5. 旧版兼容：证据面板（如果存在 legacy evidence 数据） */}
      {message.evidence && message.evidence.length > 0 && !references && (
        <section className="answer-section legacy-evidence">
          <div className="legacy-notice">以下是旧版证据展示</div>
        </section>
      )}
    </div>
  );
}