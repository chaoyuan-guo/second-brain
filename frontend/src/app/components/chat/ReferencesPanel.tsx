'use client';

import { useState } from 'react';
import type { CitationRef } from '../../lib/chat-types';

interface ReferencesPanelProps {
  references: CitationRef[];
  onOpenPreview?: (path: string, title: string, ref?: { char_offset?: number; snippet?: string }) => void;
}

/**
 * 引用来源详情面板
 * 展示所有引用的详细信息，按相关性排序
 */
export function ReferencesPanel({ references, onOpenPreview }: ReferencesPanelProps) {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [sortBy, setSortBy] = useState<'relevance' | 'order'>('relevance');

  if (!references || references.length === 0) {
    return (
      <div className="references-panel empty">
        <div className="empty-state">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M9 12h.01M15 12h.01M10 16c.5.3 1.2.5 2 .5s1.5-.2 2-.5M22 12c0 5.523-4.477 10-10 10S2 17.523 2 12 6.477 2 12 2s10 4.477 10 10z" />
          </svg>
          <span>未找到引用来源</span>
        </div>
      </div>
    );
  }

  // 排序引用
  const sortedRefs = [...references].sort((a, b) => {
    if (sortBy === 'relevance') {
      // 按分数降序，无分数的排在最后
      const scoreA = a.retrievalScore ?? -1;
      const scoreB = b.retrievalScore ?? -1;
      return scoreB - scoreA;
    }
    // 按引用 ID 顺序
    return parseInt(a.id) - parseInt(b.id);
  });

  const toggleExpand = (id: string) => {
    const newSet = new Set(expandedIds);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    setExpandedIds(newSet);
  };

  const handleOpenSource = (ref: CitationRef) => {
    if (onOpenPreview) {
      onOpenPreview(
        ref.sourcePath,
        ref.sourceTitle || ref.sourcePath.split('/').pop() || '来源',
        { char_offset: ref.charOffsetStart, snippet: ref.snippet }
      );
    }
  };

  // 统计信息
  const strongMatches = references.filter(r => r.retrievalScore !== undefined && r.retrievalScore >= 0.8).length;
  const weakMatches = references.filter(r => r.retrievalScore !== undefined && r.retrievalScore < 0.8).length;
  const unscored = references.filter(r => r.retrievalScore === undefined).length;

  return (
    <div className="references-panel">
      <div className="panel-header">
        <h3 className="panel-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
          </svg>
          引用来源
          <span className="ref-count">({references.length})</span>
        </h3>
        
        <div className="sort-controls">
          <button
            className={`sort-btn ${sortBy === 'relevance' ? 'active' : ''}`}
            onClick={() => setSortBy('relevance')}
          >
            按相关性
          </button>
          <button
            className={`sort-btn ${sortBy === 'order' ? 'active' : ''}`}
            onClick={() => setSortBy('order')}
          >
            按顺序
          </button>
        </div>
      </div>

      <div className="quality-summary">
        {strongMatches > 0 && (
          <span className="quality-badge strong">{strongMatches} 条高相关</span>
        )}
        {weakMatches > 0 && (
          <span className="quality-badge weak">{weakMatches} 条弱相关</span>
        )}
        {unscored > 0 && (
          <span className="quality-badge unscored">{unscored} 条未评分</span>
        )}
      </div>

      <div className="references-list">
        {sortedRefs.map((ref, index) => {
          const isExpanded = expandedIds.has(ref.id);
          const isWeak = ref.retrievalScore !== undefined && ref.retrievalScore < 0.8;
          const hasScore = ref.retrievalScore !== undefined;

          return (
            <div
              key={ref.id}
              className={`reference-item ${isWeak ? 'weak' : ''} ${isExpanded ? 'expanded' : ''}`}
            >
              <div className="ref-header" onClick={() => toggleExpand(ref.id)}>
                <span className="ref-index">[c{ref.id}]</span>
                <span className="ref-title">
                  {ref.sourceTitle || ref.sourcePath.split('/').pop() || '未知来源'}
                </span>
                {hasScore && (
                  <span className={`ref-score ${isWeak ? 'weak' : 'strong'}`}>
                    {Math.round(ref.retrievalScore! * 100)}%
                  </span>
                )}
                <svg
                  className="expand-icon"
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path d={isExpanded ? "M18 15l-6-6-6 6" : "M6 9l6 6 6-6"} />
                </svg>
              </div>

              {isExpanded && (
                <div className="ref-details">
                  {ref.heading && (
                    <div className="ref-heading">章节：{ref.heading}</div>
                  )}
                  {ref.snippet && (
                    <div className="ref-snippet">
                      <div className="snippet-label">内容片段：</div>
                      <blockquote>{ref.snippet}</blockquote>
                    </div>
                  )}
                  <div className="ref-path">路径：{ref.sourcePath}</div>
                  <button
                    className="open-source-btn"
                    onClick={() => handleOpenSource(ref)}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                    查看原文
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
