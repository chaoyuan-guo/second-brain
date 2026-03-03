'use client';

import { useMemo } from 'react';
import {
  type ChatMessage,
  type DecisionSummary,
  type ProcessOverview,
  type CompletionState,
  type EvidenceItem,
} from '../../lib/chat-types';
import { ConclusionCard } from './ConclusionCard';
import { ActionChecklist } from './ActionChecklist';
import { RiskConfidenceCard } from './RiskConfidenceCard';
import { EvidencePanel } from './EvidencePanel';
import { FullResponseMarkdown } from './FullResponseMarkdown';

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

interface AnswerPanelProps {
  message: ChatMessage;
  copiedKey: string | null;
  onCopyCode: (value: string, key: string) => void;
  onOpenPreview?: (path: string, title: string, ref?: { char_offset?: number; snippet?: string }) => void;
}

/**
 * AnswerPanel - 主答案面板
 * 遵循 Answer-first 原则，默认展示结论与行动，细节按需展开
 */
export function AnswerPanel({
  message,
  copiedKey,
  onCopyCode,
  onOpenPreview,
}: AnswerPanelProps) {
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
        <FullResponseMarkdown
          content={message.content}
          messageId={message.id}
          copiedKey={copiedKey}
          onCopyCode={onCopyCode}
        />
      </div>
    );
  }

  const { decisionSummary, processOverview, completionState, evidence } = message;

  // 根据完成状态渲染不同的布局
  if (completionState === 'failed') {
    return (
      <div className="answer-panel failed">
        <div className="failure-card">
          <p className="failure-title">未能形成可靠结论</p>
          <p className="failure-reason">
            {decisionSummary?.failureReason || '请重试或缩小问题范围'}
          </p>
        </div>
        {/* 即使失败也展示完整内容 */}
        <FullResponseMarkdown
          content={message.content}
          messageId={message.id}
          copiedKey={copiedKey}
          onCopyCode={onCopyCode}
        />
      </div>
    );
  }

  return (
    <div className="answer-panel">
      {/* 1. 一句话结论 */}
      <ConclusionCard conclusion={decisionSummary!.conclusion} />

      {/* 2. 下一步行动 */}
      {decisionSummary!.actions.length > 0 && (
        <ActionChecklist actions={decisionSummary!.actions} />
      )}

      {/* 3. 风险与置信度 */}
      <RiskConfidenceCard
        confidence={decisionSummary!.confidence}
        risks={decisionSummary!.risks}
        assumptions={decisionSummary!.assumptions}
      />

      {/* 4. 关键证据 */}
      {evidence!.length > 0 && (
        <EvidencePanel evidence={evidence!} onOpenPreview={onOpenPreview} />
      )}

      {/* 5. 完整展开内容 */}
      <FullResponseMarkdown
        content={message.content}
        messageId={message.id}
        copiedKey={copiedKey}
        onCopyCode={onCopyCode}
      />
    </div>
  );
}