'use client';

import { useEffect, useRef } from 'react';
import type { ThinkingStep } from '../../lib/chat-types';

interface ProcessDebugDrawerProps {
  thinkingSteps?: ThinkingStep[];
  onClose: () => void;
}

/**
 * ProcessDebugDrawer - 调试抽屉（专家层）
 * 展示完整的工具参数、结果、错误信息
 */
export function ProcessDebugDrawer({ thinkingSteps, onClose }: ProcessDebugDrawerProps) {
  const drawerRef = useRef<HTMLDivElement>(null);

  // ESC 关闭
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  // 点击外部关闭
  const handleOverlayClick = () => {
    onClose();
  };

  const handleDrawerClick = (event: React.MouseEvent) => {
    event.stopPropagation();
  };

  return (
    <div className="debug-drawer-overlay" onClick={handleOverlayClick}>
      <div
        ref={drawerRef}
        className="debug-drawer"
        onClick={handleDrawerClick}
        role="dialog"
        aria-modal="true"
        aria-label="调试详情"
      >
        <div className="debug-drawer-header">
          <h4 className="debug-drawer-title">调试详情</h4>
          <button
            type="button"
            className="debug-drawer-close"
            onClick={onClose}
            aria-label="关闭调试面板"
          >
            ✕
          </button>
        </div>

        <div className="debug-drawer-body">
          {thinkingSteps?.map((step, index) => (
            <div key={step.id} className="debug-step">
              <div className="debug-step-header">
                <span className="debug-step-index">#{index + 1}</span>
                <span className="debug-step-type">{step.type}</span>
                <span className="debug-step-time">
                  {new Date(step.timestamp).toLocaleTimeString()}
                </span>
              </div>

              {step.tool ? (
                <div className="debug-tool">
                  <div className="debug-tool-row">
                    <span className="debug-label">工具名称:</span>
                    <span className="debug-value">{step.tool.name}</span>
                  </div>
                  <div className="debug-tool-row">
                    <span className="debug-label">调用ID:</span>
                    <span className="debug-value mono">{step.tool.id}</span>
                  </div>
                  <div className="debug-tool-row">
                    <span className="debug-label">状态:</span>
                    <span className={`debug-value status-${step.tool.status}`}>
                      {step.tool.status}
                    </span>
                  </div>
                  {step.tool.arguments && (
                    <div className="debug-tool-row">
                      <span className="debug-label">参数:</span>
                      <pre className="debug-json">
                        {JSON.stringify(step.tool.arguments, null, 2)}
                      </pre>
                    </div>
                  )}
                  {step.tool.result !== undefined && (
                    <div className="debug-tool-row">
                      <span className="debug-label">结果:</span>
                      <pre className="debug-json">
                        {typeof step.tool.result === 'string'
                          ? step.tool.result
                          : JSON.stringify(step.tool.result, null, 2)}
                      </pre>
                    </div>
                  )}
                  {step.tool.error && (
                    <div className="debug-tool-row error">
                      <span className="debug-label">错误:</span>
                      <span className="debug-value">{step.tool.error}</span>
                    </div>
                  )}
                </div>
              ) : (
                <div className="debug-thought">
                  <span className="debug-label">思考内容:</span>
                  <p className="debug-thought-content">{step.content}</p>
                </div>
              )}
            </div>
          ))}

          {(!thinkingSteps || thinkingSteps.length === 0) && (
            <div className="debug-empty">无调试信息</div>
          )}
        </div>
      </div>
    </div>
  );
}