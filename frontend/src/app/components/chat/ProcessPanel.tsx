'use client';

import { useState, useMemo } from 'react';
import type { ChatMessage, ProcessOverview, ThinkingStep } from '../../lib/chat-types';
import { ProcessOverviewBar } from './ProcessOverviewBar';
import { ProcessGroupList } from './ProcessGroupList';
import { ProcessDebugDrawer } from './ProcessDebugDrawer';

/**
 * 判断是否具备有效的过程数据
 */
const hasValidProcessData = (message: ChatMessage): boolean => {
  const { processOverview, thinkingSteps } = message;

  // 需要有 processOverview 或 thinkingSteps
  if (processOverview && typeof processOverview === 'object' && typeof processOverview.phase === 'string') {
    return true;
  }

  if (Array.isArray(thinkingSteps) && thinkingSteps.length > 0) {
    return true;
  }

  return false;
};

interface ProcessPanelProps {
  message: ChatMessage;
}

/**
 * ProcessPanel - 过程面板
 * 默认折叠为摘要条，展开后显示语义分组与调试信息
 */
export function ProcessPanel({ message }: ProcessPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isDebugOpen, setIsDebugOpen] = useState(false);

  const hasProcess = useMemo(() => hasValidProcessData(message), [message]);

  if (!hasProcess) {
    return null;
  }

  const { processOverview, thinkingSteps } = message;

  // 从 thinkingSteps 中提取工具调用
  const toolCalls = useMemo(() => {
    if (!thinkingSteps) return [];
    return thinkingSteps
      .filter((step) => step.type === 'tool' && step.tool)
      .map((step) => step.tool!);
  }, [thinkingSteps]);

  return (
    <div className="process-panel">
      {/* 过程摘要条 */}
      <ProcessOverviewBar
        processOverview={processOverview}
        toolCalls={toolCalls}
        isExpanded={isExpanded}
        onToggle={() => setIsExpanded(!isExpanded)}
      />

      {/* 展开后的过程分组列表 */}
      {isExpanded && (
        <ProcessGroupList
          thinkingSteps={thinkingSteps}
          onOpenDebug={() => setIsDebugOpen(true)}
        />
      )}

      {/* 调试抽屉（专家层） */}
      {isDebugOpen && (
        <ProcessDebugDrawer
          thinkingSteps={thinkingSteps}
          onClose={() => setIsDebugOpen(false)}
        />
      )}
    </div>
  );
}