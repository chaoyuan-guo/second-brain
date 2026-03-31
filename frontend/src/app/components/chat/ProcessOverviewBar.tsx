'use client';

import type { ProcessOverview, ToolInvocation, RunPhase } from '../../lib/chat-types';

interface ProcessOverviewBarProps {
  processOverview?: ProcessOverview;
  toolCalls: ToolInvocation[];
  isThinking?: boolean;
  isExpanded: boolean;
  onToggle: () => void;
}

const phaseLabels: Record<RunPhase, string> = {
  retrieving: '正在整理依据',
  validating: '正在核对依据',
  synthesizing: '正在补全回答',
  completed: '已完成',
};

/**
 * ProcessOverviewBar - 过程摘要条
 * 默认折叠态，只展示当前状态与展开入口
 */
export function ProcessOverviewBar({
  processOverview,
  toolCalls,
  isThinking = false,
  isExpanded,
  onToggle,
}: ProcessOverviewBarProps) {
  // 从工具调用中计算活跃数量
  const activeCount = toolCalls.filter(
    (tool) => tool.status === 'pending' || tool.status === 'running'
  ).length;

  const errorCount = toolCalls.filter((tool) => tool.status === 'error').length;

  // 从 processOverview 或默认值获取阶段
  const phase: RunPhase = processOverview?.phase
    ? (isThinking && processOverview.phase === 'completed' ? 'synthesizing' : processOverview.phase)
    : activeCount > 0
      ? 'retrieving'
      : isThinking
        ? 'synthesizing'
        : 'completed';

  const blockingErrorCount = processOverview?.blockingErrorCount || 0;
  const impact = processOverview?.impact || (blockingErrorCount > 0 ? 'blocking' : errorCount > 0 ? 'partial' : 'none');

  return (
    <div className="process-overview-bar">
      <button
        type="button"
        className="process-toggle-btn"
        onClick={onToggle}
        aria-expanded={isExpanded}
        aria-label={isExpanded ? '收起过程详情' : '展开过程详情'}
      >
        <span className="process-phase-label">{phaseLabels[phase]}</span>
        <span className={`process-chevron ${isExpanded ? 'expanded' : ''}`}>›</span>
      </button>

      {/* 影响判定提示 */}
      {impact !== 'none' && (
        <div className={`process-impact impact-${impact}`}>
          {impact === 'blocking' && '未形成可靠结论，请重试或缩小问题范围'}
          {impact === 'partial' && '部分步骤失败，但结论可用'}
        </div>
      )}
    </div>
  );
}
