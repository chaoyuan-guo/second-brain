'use client';

import type { EvidenceItem, EvidenceRef } from '../../lib/chat-types';

interface EvidencePanelProps {
  evidence: EvidenceItem[];
  onOpenPreview?: (path: string, title: string, ref?: { char_offset?: number; snippet?: string }) => void;
}

const formatSourcePath = (path: string): string => {
  const normalized = path.replace(/\\/g, '/');
  const absoluteMarker = '/data/notes/my_markdowns/';
  const relativeMarker = 'data/notes/my_markdowns/';
  if (normalized.includes(absoluteMarker)) {
    return normalized.split(absoluteMarker).pop() ?? path;
  }
  if (normalized.startsWith(relativeMarker)) {
    return normalized.slice(relativeMarker.length);
  }
  return path;
};

/**
 * EvidencePanel - 证据与引用面板
 * 展示断言到来源片段的映射，支持来源预览
 */
export function EvidencePanel({ evidence, onOpenPreview }: EvidencePanelProps) {
  if (!evidence?.length) {
    return null;
  }

  // 最多展示 3 条关键证据
  const displayEvidence = evidence.slice(0, 3);

  const handleRefClick = (ref: EvidenceRef) => {
    if (!onOpenPreview) return;
    const title = ref.sourceTitle || ref.sourcePath.split('/').pop() || ref.sourcePath;
    onOpenPreview(ref.sourcePath, title, {
      char_offset: ref.charOffsetStart,
      snippet: ref.snippet,
    });
  };

  return (
    <div className="evidence-panel">
      <h4 className="evidence-title">关键证据</h4>
      <ul className="evidence-list">
        {displayEvidence.map((item) => (
          <li key={item.claimId} className="evidence-item">
            <p className="claim-text">{item.claimText}</p>
            {item.refs?.length > 0 && (
              <ul className="evidence-refs">
                {item.refs.map((ref, refIndex) => (
                  <li key={`${item.claimId}-ref-${refIndex}`} className="evidence-ref">
                    <button
                      type="button"
                      className="evidence-ref-btn"
                      onClick={() => handleRefClick(ref)}
                      disabled={!onOpenPreview}
                    >
                      <span className="ref-source">{formatSourcePath(ref.sourcePath)}</span>
                      {ref.heading && (
                        <span className="ref-heading">#{ref.heading}</span>
                      )}
                    </button>
                    {ref.snippet && (
                      <p className="ref-snippet">{ref.snippet}</p>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}