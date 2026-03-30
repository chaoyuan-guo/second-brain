'use client';

import { useState } from 'react';
import type { CitationRef } from '../../lib/chat-types';
import {
  deriveSourceTitle,
  formatRetrievalDistance,
  getReferenceKindLabel,
  getCitationSuperscript,
  isWeakRetrievalScore,
  normalizeCitationId,
  sanitizeCitationSnippet,
  shouldUsePrecisePreviewTarget,
} from '../../lib/citation-utils';

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

  const normalizedId = normalizeCitationId(citationId);
  const rawRef = citationMap[citationId] || citationMap[normalizedId] || citationMap[`c${normalizedId}`];
  const ref = rawRef
    ? {
        ...rawRef,
        snippet: sanitizeCitationSnippet(rawRef.snippet),
      }
    : undefined;
  const hasRef = !!ref;
  const isWeakMatch = isWeakRetrievalScore(ref?.retrievalScore);
  const sourceTitle = ref ? deriveSourceTitle(ref.sourcePath, ref.sourceTitle, ref.heading) : '';
  const sourceMeta = [
    ref ? getReferenceKindLabel(ref) : undefined,
    ref?.sourceDateLabel,
    isWeakMatch ? '弱匹配' : undefined,
  ].filter(Boolean).join(' · ');
  const distanceLabel = formatRetrievalDistance(ref?.retrievalScore);
  
  const handleClick = () => {
    if (hasRef && onOpenPreview) {
      onOpenPreview(
        ref.sourcePath,
        sourceTitle || '来源',
        shouldUsePrecisePreviewTarget(ref)
          ? { char_offset: ref.charOffsetStart, snippet: ref.snippet }
          : undefined
      );
    }
  };

  return (
    <span
      className={`inline-citation ${hasRef ? 'has-ref' : ''} ${isWeakMatch ? 'weak-match' : ''}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={handleClick}
      role={hasRef ? 'button' : undefined}
      tabIndex={hasRef ? 0 : undefined}
      aria-label={hasRef ? `查看引用 ${normalizedId}` : undefined}
      onKeyDown={(event) => {
        if (hasRef && (event.key === 'Enter' || event.key === ' ')) {
          event.preventDefault();
          handleClick();
        }
      }}
    >
      <sup className="citation-sup">{getCitationSuperscript(normalizedId)}</sup>
      
      {isHovered && hasRef && (
        <div className="citation-tooltip">
          <div className="tooltip-header">
            <span className="tooltip-title">{sourceTitle}</span>
            {distanceLabel && (
              <span className={`tooltip-score ${isWeakMatch ? 'weak' : 'strong'}`}>
                {distanceLabel}
              </span>
            )}
          </div>
          {sourceMeta && (
            <div className="tooltip-meta">{sourceMeta}</div>
          )}
          {ref.heading && (
            <div className="tooltip-heading">{ref.heading}</div>
          )}
          {ref.snippet && (
            <div className="tooltip-snippet">{ref.snippet.substring(0, 100)}...</div>
          )}
          <div className="tooltip-hint">
            {ref.provenance === 'native' ? '点击定位支撑片段' : '点击查看补偿定位'}
          </div>
        </div>
      )}
    </span>
  );
}
