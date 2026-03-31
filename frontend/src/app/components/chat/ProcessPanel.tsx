'use client';

import { useState, useMemo } from 'react';
import type { ChatMessage, ProcessStepSummary } from '../../lib/chat-types';
import { ProcessOverviewBar } from './ProcessOverviewBar';
import { ProcessGroupList } from './ProcessGroupList';
import { ProcessDebugDrawer } from './ProcessDebugDrawer';
import { generateProcessSummary, type ToolCallRecord } from '../../api/_lib/event-adapter';

/**
 * 判断是否具备有效的过程数据
 */
const hasValidProcessData = (message: ChatMessage): boolean => {
  const { processOverview, thinkingSteps, processSummary } = message;

  // 优先使用语义化的 processSummary
  if (Array.isArray(processSummary) && processSummary.length > 0) {
    return true;
  }

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
 * 
 * 新增：支持语义化的 processSummary 展示
 */
export function ProcessPanel({ message }: ProcessPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isDebugOpen, setIsDebugOpen] = useState(false);
  const { processOverview, thinkingSteps, processSummary } = message;

  const hasProcess = useMemo(() => hasValidProcessData(message), [message]);

  // 从 thinkingSteps 中提取工具调用
  const toolCalls = useMemo(() => {
    if (!thinkingSteps) return [];
    return thinkingSteps
      .filter((step) => step.type === 'tool' && step.tool)
      .map((step) => step.tool!);
  }, [thinkingSteps]);

  const liveProcessSummary = useMemo(() => {
    if (processSummary && processSummary.length > 0) {
      return processSummary;
    }
    if (!thinkingSteps || thinkingSteps.length === 0) {
      return [];
    }

    const activeCalls: ToolCallRecord[] = [];
    const completedCalls: ToolCallRecord[] = [];
    const errorCalls: ToolCallRecord[] = [];

    thinkingSteps
      .filter((step) => step.type === 'tool' && step.tool)
      .forEach((step) => {
        const tool = step.tool!;
        const call: ToolCallRecord = {
          id: tool.id,
          name: tool.name,
          status: tool.status,
          arguments: tool.arguments,
          result: tool.result,
          error: tool.error,
          startedAt: tool.startedAt ?? step.timestamp,
          completedAt: tool.completedAt,
        };

        if (tool.status === 'completed') {
          completedCalls.push(call);
          return;
        }
        if (tool.status === 'error') {
          errorCalls.push(call);
          return;
        }
        activeCalls.push(call);
      });

    return generateProcessSummary(completedCalls, errorCalls, activeCalls);
  }, [processSummary, thinkingSteps]);

  if (!hasProcess) {
    return null;
  }

  return (
    <div className="process-panel">
      {/* 过程摘要条 */}
      <ProcessOverviewBar
        processOverview={processOverview}
        toolCalls={toolCalls}
        isThinking={message.isThinking}
        isExpanded={isExpanded}
        onToggle={() => setIsExpanded(!isExpanded)}
      />

      {/* 展开后的语义化过程摘要 */}
      {isExpanded && liveProcessSummary.length > 0 && (
        <>
          <SemanticProcessSummary steps={liveProcessSummary} />
        </>
      )}

      {/* 展开后的过程分组列表（旧版兼容） */}
      {isExpanded && liveProcessSummary.length === 0 && (
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

// ============================================================================
// 语义化过程摘要组件
// ============================================================================

interface SemanticProcessSummaryProps {
  steps: ProcessStepSummary[];
}

const phaseIcons: Record<string, string> = {
  retrieving: '🔍',
  validating: '✓',
  synthesizing: '💡',
  completed: '✓',
};

const phaseLabels: Record<string, string> = {
  retrieving: '检索',
  validating: '验证',
  synthesizing: '合成',
  completed: '完成',
};

/**
 * SemanticProcessSummary - 语义化过程摘要
 * 展示用户友好的过程步骤描述
 */
function SemanticProcessSummary({ steps }: SemanticProcessSummaryProps) {
  return (
    <div className="semantic-process-summary">
      <div className="process-timeline">
        {steps.map((step) => {
          const phase = step.phase || 'synthesizing';
          const icon = phaseIcons[phase] || '•';
          const phaseLabel = phaseLabels[phase] || phase;
          
          const isRunning = step.status === 'running';

          return (
            <div key={step.stepNumber} className={`timeline-item status-${step.status || 'completed'}`}>
              <div className="timeline-marker">
                <span className="step-number">{step.stepNumber}</span>
              </div>
              <div className="timeline-content">
                <div className="step-header">
                  <span className="step-icon">{icon}</span>
                  <span className="step-phase">{phaseLabel}</span>
                  <span className="step-summary">{step.summary}</span>
                  {isRunning && <span className="step-live-badge">进行中</span>}
                </div>
                {step.detail && (
                  <div className="step-detail">{step.detail}</div>
                )}
                {step.durationMs && step.durationMs > 100 && (
                  <div className="step-duration">{Math.round(step.durationMs / 100) / 10}s</div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
