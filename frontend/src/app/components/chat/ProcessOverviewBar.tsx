'use client';

import type { ProcessOverview, ToolInvocation, RunPhase } from '../../lib/chat-types';

interface ProcessOverviewBarProps {
  processOverview?: ProcessOverview;
  toolCalls: ToolInvocation[];
  isExpanded: boolean;
  onToggle: () => void;
}

const phaseLabels: Record<RunPhase, string> = {
  retrieving: '正在检索相关信息',
  validating: '正在验证关键证据',
  synthesizing: '正在生成最终结论',
  completed: '已完成',
};

/**
 * ProcessOverviewBar - 过程摘要条
 * 默认折叠态，展示阶段、耗时、异常数
 */
export function ProcessOverviewBar({
  processOverview,
  toolCalls,
  isExpanded,
  onToggle,
}: ProcessOverviewBarProps) {
  // 从工具调用中计算活跃数量
  const activeCount = toolCalls.filter(
    (tool) => tool.status === 'pending' || tool.status === 'running'
  ).length;

  const errorCount = toolCalls.filter((tool) => tool.status === 'error').length;

  // 从 processOverview 或默认值获取阶段
  const phase: RunPhase = processOverview?.phase || (activeCount > 0 ? 'retrieving' : 'completed');

  // 计算耗时
  const durationMs = processOverview?.durationMs || 0;
  const durationSec = (durationMs / 1000).toFixed(1);

  // 异常数量
  const warningCount = processOverview?.warningCount || errorCount;
  const blockingErrorCount = processOverview?.blockingErrorCount || 0;

  // 影响判定
  const impact = processOverview?.impact || (blockingErrorCount > 0 ? 'blocking' : warningCount > 0 ? 'partial' : 'none');

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
        <span className="process-meta">
          {durationMs > 0 && <span className="process-duration">{durationSec}s</span>}
          {warningCount > 0 && (
            <span className="process-warning" title={`${warningCount} 个警告`}>
              ⚠ {warningCount}
            </span>
          )}
          {blockingErrorCount > 0 && (
            <span className="process-error" title={`${blockingErrorCount} 个阻断性错误`}>
              ✕ {blockingErrorCount}
            </span>
          )}
        </span>
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