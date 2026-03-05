'use client';

import { useState } from 'react';
import type { HonestySignals } from '../../lib/chat-types';

interface HonestyBannerProps {
  signals: HonestySignals;
}

/**
 * 诚实性提示横幅
 * 当证据不足或弱匹配时，显式告知用户局限性
 */
export function HonestyBanner({ signals }: HonestyBannerProps) {
  const [isDismissed, setIsDismissed] = useState(false);
  const reasonCodes = Array.isArray(signals.reasonCodes) ? signals.reasonCodes : [];

  if (isDismissed) {
    return (
      <button
        className="honesty-banner-trigger"
        onClick={() => setIsDismissed(false)}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        显示局限性说明
      </button>
    );
  }

  // 根据证据质量选择样式
  const getSeverityStyles = () => {
    switch (signals.evidenceQuality) {
      case 'none':
      case 'weak':
        return 'severity-high';
      case 'partial':
        return 'severity-medium';
      default:
        return 'severity-low';
    }
  };

  // 构建主提示信息
  const getMainMessage = (): string => {
    if (signals.limitationNote) {
      return signals.limitationNote;
    }

    if (reasonCodes.includes('no_hit')) {
      return '未检索到可直接支持答案的记录，请将结论视为待验证假设。';
    }
    if (reasonCodes.includes('weak_match')) {
      return '已命中的引用相关性偏弱，建议先核对来源再采用结论。';
    }
    if (reasonCodes.includes('insufficient_hits')) {
      return '目前命中的有效证据数量不足，结论置信度受限。';
    }

    switch (signals.evidenceQuality) {
      case 'none':
        return '未找到直接相关证据，本回答主要基于推理生成。';
      case 'weak':
        return '检索到的证据相关性较低，请谨慎采纳。';
      case 'partial':
        return '部分引用来源相关性不足，建议进一步核实。';
      default:
        return '证据质量良好，但仍建议您核实关键信息。';
    }
  };

  return (
    <div className={`honesty-banner ${getSeverityStyles()}`}>
      <div className="banner-header">
        <div className="banner-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <div className="banner-title">
          {signals.evidenceQuality === 'strong' ? '证据提示' : '局限性说明'}
        </div>
        <button
          className="banner-dismiss"
          onClick={() => setIsDismissed(true)}
          aria-label="关闭"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="banner-content">
        <p className="main-message">{getMainMessage()}</p>

        {signals.honestyWarnings.length > 0 && (
          <ul className="warning-list">
            {signals.honestyWarnings.map((warning, index) => (
              <li key={index} className="warning-item">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                {warning}
              </li>
            ))}
          </ul>
        )}

        {signals.weakMatches.length > 0 && (
          <div className="weak-matches-info">
            <div className="info-label">弱相关引用：</div>
            <div className="citation-tags">
              {signals.weakMatches.slice(0, 5).map(id => (
                <span key={id} className="citation-tag weak">[c{id.startsWith('c') ? id.slice(1) : id}]</span>
              ))}
              {signals.weakMatches.length > 5 && (
                <span className="more-tag">+{signals.weakMatches.length - 5}</span>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="banner-footer">
        <span className="evidence-quality-label">
          证据质量：
          <span className={`quality-value ${signals.evidenceQuality}`}>
            {signals.evidenceQuality === 'strong' && '高'}
            {signals.evidenceQuality === 'partial' && '中'}
            {signals.evidenceQuality === 'weak' && '低'}
            {signals.evidenceQuality === 'none' && '无'}
          </span>
        </span>
      </div>
    </div>
  );
}
