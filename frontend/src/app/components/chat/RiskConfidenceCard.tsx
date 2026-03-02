'use client';

import type { ConfidenceLevel } from '../../lib/chat-types';

interface RiskConfidenceCardProps {
  confidence: ConfidenceLevel;
  risks: string[];
  assumptions: string[];
}

const confidenceLabels: Record<ConfidenceLevel, { text: string; className: string }> = {
  high: { text: '高', className: 'confidence-high' },
  medium: { text: '中', className: 'confidence-medium' },
  low: { text: '低', className: 'confidence-low' },
  unknown: { text: '未知', className: 'confidence-unknown' },
};

/**
 * RiskConfidenceCard - 风险与置信度卡片
 * 展示置信度等级、主要风险和关键假设
 * 使用语义色，不以红绿唯一编码（兼容色弱）
 */
export function RiskConfidenceCard({ confidence, risks, assumptions }: RiskConfidenceCardProps) {
  const confidenceInfo = confidenceLabels[confidence] || confidenceLabels.unknown;

  return (
    <div className="risk-confidence-card">
      {/* 置信度 */}
      <div className="confidence-section">
        <span className="confidence-label">置信度</span>
        <span className={`confidence-value ${confidenceInfo.className}`}>
          {confidenceInfo.text}
        </span>
      </div>

      {/* 主要风险 */}
      {risks?.length > 0 && (
        <div className="risks-section">
          <h5 className="risks-title">主要风险</h5>
          <ul className="risks-list">
            {risks.map((risk, index) => (
              <li key={`risk-${index}`} className="risk-item">
                <span className="risk-icon" aria-hidden>⚠</span>
                <span className="risk-text">{risk}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 关键假设 */}
      {assumptions?.length > 0 && (
        <div className="assumptions-section">
          <h5 className="assumptions-title">关键假设</h5>
          <ul className="assumptions-list">
            {assumptions.map((assumption, index) => (
              <li key={`assumption-${index}`} className="assumption-item">
                <span className="assumption-icon" aria-hidden>💡</span>
                <span className="assumption-text">{assumption}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}