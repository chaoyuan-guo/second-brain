'use client';

import type { ConfidenceLevel } from '../../lib/chat-types';

interface RiskConfidenceCardProps {
  confidence: ConfidenceLevel;
  risks: string[];
  assumptions: string[];
}

const confidenceConfig: Record<ConfidenceLevel, { text: string; className: string }> = {
  high: { text: '高', className: 'high' },
  medium: { text: '中', className: 'medium' },
  low: { text: '低', className: 'low' },
  unknown: { text: '未知', className: 'unknown' },
};

/**
 * RiskConfidenceCard - 风险与置信度卡片
 * 展示置信度等级、主要风险和关键假设
 * 使用语义色，不以红绿唯一编码（兼容色弱）
 */
export function RiskConfidenceCard({ confidence, risks, assumptions }: RiskConfidenceCardProps) {
  const config = confidenceConfig[confidence] || confidenceConfig.unknown;

  // 合并风险和假设一起展示
  const allRisks = [
    ...(risks || []),
    ...(assumptions || []).map(a => `假设: ${a}`),
  ];

  return (
    <div className="risk-confidence-card">
      {/* 置信度 */}
      <div className="confidence-section">
        <span className="section-label">置信度</span>
        <span className={`confidence-badge ${config.className}`}>
          {config.text}
        </span>
      </div>

      {/* 风险与假设 */}
      <div className="risk-section">
        <span className="section-label">风险与假设</span>
        <ul className="risk-list">
          {allRisks.length > 0 ? (
            allRisks.map((risk, index) => (
              <li key={`risk-${index}`} className="risk-item">
                {risk}
              </li>
            ))
          ) : (
            <li className="risk-item empty">无已知风险</li>
          )}
        </ul>
      </div>
    </div>
  );
}