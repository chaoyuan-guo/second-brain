"use client";

import { useState } from 'react';
import type { ThinkingStep, ToolInvocation } from '../lib/chat-types';
import { CheckIcon, ErrorIcon, LoaderIcon, ChevronDownIcon, ChevronRightIcon } from './icons';

interface ThinkingTimelineProps {
  steps: ThinkingStep[];
  currentStepId?: string;
  isComplete?: boolean;
}

const formatDuration = (start?: number, end?: number): string => {
  if (!start) return '';
  const endTime = end || Date.now();
  const duration = endTime - start;
  if (duration < 1000) return `${duration}ms`;
  return `${(duration / 1000).toFixed(1)}s`;
};

const ToolInvocationCard = ({ tool }: { tool: ToolInvocation }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const statusIcon = {
    pending: <span className="tool-status-icon pending"><LoaderIcon /></span>,
    running: <span className="tool-status-icon running"><LoaderIcon /></span>,
    completed: <span className="tool-status-icon completed"><CheckIcon /></span>,
    error: <span className="tool-status-icon error"><ErrorIcon /></span>,
  }[tool.status];

  const duration = formatDuration(tool.startedAt, tool.completedAt);

  return (
    <div className={`tool-invocation ${tool.status}`}>
      <button
        className="tool-header"
        onClick={() => setIsExpanded(!isExpanded)}
        type="button"
      >
        {statusIcon}
        <span className="tool-name">{tool.name}</span>
        {duration && <span className="tool-duration">{duration}</span>}
        <span className="tool-expand-icon">
          {isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
        </span>
      </button>

      {isExpanded && (
        <div className="tool-details">
          {tool.arguments && (
            <div className="tool-section">
              <span className="tool-section-label">参数：</span>
              <pre className="tool-code">
                {JSON.stringify(tool.arguments, null, 2)}
              </pre>
            </div>
          )}
          {tool.result != null && (
            <div className="tool-section">
              <span className="tool-section-label">结果：</span>
              <pre className="tool-code result">
                {typeof tool.result === 'string'
                  ? tool.result
                  : JSON.stringify(tool.result, null, 2)}
              </pre>
            </div>
          )}
          {tool.error && (
            <div className="tool-section error">
              <span className="tool-section-label">错误：</span>
              <span className="tool-error-message">{tool.error}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export const ThinkingTimeline = ({ steps, currentStepId, isComplete }: ThinkingTimelineProps) => {
  if (!steps.length) return null;

  return (
    <div className="thinking-timeline">
      <div className="timeline-header">
        <span className="timeline-title">思考过程</span>
        {isComplete && <span className="timeline-complete">已完成</span>}
      </div>
      <div className="timeline-steps">
        {steps.map((step, index) => {
          const isCurrent = step.id === currentStepId;
          const stepNumber = index + 1;

          return (
            <div
              key={step.id}
              className={`timeline-step ${step.type} ${isCurrent ? 'current' : ''}`}
            >
              <div className="step-marker">
                <span className="step-number">{stepNumber}</span>
                {step.type === 'tool' && step.tool && (
                  <span className={`step-status ${step.tool.status}`} />
                )}
              </div>
              <div className="step-content">
                {step.type === 'thought' && (
                  <div className="thought-content">{step.content}</div>
                )}
                {step.type === 'tool' && step.tool && (
                  <ToolInvocationCard tool={step.tool} />
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ThinkingTimeline;
