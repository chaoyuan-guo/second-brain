'use client';

import { useState } from 'react';
import type { CitationRef } from '../../lib/chat-types';

interface InlineCitationProps {
  citationId: string;
  citationMap: Record<string, CitationRef>;
  onOpenPreview?: (path: string, title: string, ref?: { char_offset?: number; snippet?: string }) => void;
}

/**
 * 行内引用标记组件
 * 显示 [cxx] 格式标记，hover 显示来源详情
 */
export function InlineCitation({ citationId, citationMap, onOpenPreview }: InlineCitationProps) {
  const [isHovered, setIsHovered] = useState(false);

  const normalizedId = citationId.startsWith('c') ? citationId.slice(1) : citationId;
  const ref = citationMap[citationId] || citationMap[normalizedId] || citationMap[`c${normalizedId}`];
  const hasRef = !!ref;
  
  // 弱匹配样式（分数 < 0.8）
  const isWeakMatch = ref?.retrievalScore !== undefined && ref.retrievalScore < 0.8;
  
  const handleClick = () => {
    if (hasRef && onOpenPreview) {
      onOpenPreview(
        ref.sourcePath,
        ref.sourceTitle || ref.sourcePath.split('/').pop() || '来源',
        { char_offset: ref.charOffsetStart, snippet: ref.snippet }
      );
    }
  };

  return (
    <span
      className={`inline-citation ${hasRef ? 'has-ref' : ''} ${isWeakMatch ? 'weak-match' : ''}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={handleClick}
    >
      <span className="citation-bracket">[</span>
      <span className="citation-id">c{normalizedId}</span>
      <span className="citation-bracket">]</span>
      
      {isHovered && hasRef && (
        <div className="citation-tooltip">
          <div className="tooltip-header">
            <span className="tooltip-title">{ref.sourceTitle || ref.sourcePath.split('/').pop()}</span>
            {ref.retrievalScore !== undefined && (
              <span className={`tooltip-score ${isWeakMatch ? 'weak' : 'strong'}`}>
                {Math.round(ref.retrievalScore * 100)}%
              </span>
            )}
          </div>
          {ref.heading && (
            <div className="tooltip-heading">{ref.heading}</div>
          )}
          {ref.snippet && (
            <div className="tooltip-snippet">{ref.snippet.substring(0, 100)}...</div>
          )}
          <div className="tooltip-hint">点击查看详情</div>
        </div>
      )}
    </span>
  );
}
