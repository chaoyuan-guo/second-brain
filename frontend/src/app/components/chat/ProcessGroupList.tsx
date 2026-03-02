'use client';

import { useMemo } from 'react';
import type { ThinkingStep, ToolInvocation } from '../../lib/chat-types';

interface ProcessGroupListProps {
  thinkingSteps?: ThinkingStep[];
  onOpenDebug: () => void;
}

type SemanticGroup = {
  id: string;
  title: string;
  items: ThinkingStep[];
};

/**
 * 工具名到语义阶段的映射
 */
const toolSemanticMap: Record<string, 'retrieve' | 'validate' | 'synthesize_helper' | 'other'> = {
  query_my_notes: 'retrieve',
  web_search: 'retrieve',
  read_page: 'retrieve',
  read_note_file: 'retrieve',
  run_code_interpreter: 'validate',
  load_skill: 'synthesize_helper',
};

const getToolSemantic = (toolName: string): 'retrieve' | 'validate' | 'synthesize_helper' | 'other' => {
  return toolSemanticMap[toolName] || 'other';
};

/**
 * ProcessGroupList - 过程分组列表
 * 将工具调用按语义分组展示：信息检索、证据验证、结果合成
 */
export function ProcessGroupList({ thinkingSteps, onOpenDebug }: ProcessGroupListProps) {
  const groups = useMemo(() => {
    if (!thinkingSteps?.length) return [];

    const retrieveItems: ThinkingStep[] = [];
    const validateItems: ThinkingStep[] = [];
    const synthesizeItems: ThinkingStep[] = [];

    thinkingSteps.forEach((step) => {
      if (step.type === 'tool' && step.tool) {
        const semantic = getToolSemantic(step.tool.name);
        if (semantic === 'retrieve') {
          retrieveItems.push(step);
        } else if (semantic === 'validate') {
          validateItems.push(step);
        } else if (semantic === 'synthesize_helper') {
          synthesizeItems.push(step);
        } else {
          // other 归入信息检索
          retrieveItems.push(step);
        }
      } else if (step.type === 'thought') {
        synthesizeItems.push(step);
      } else if (step.type === 'synthesize') {
        synthesizeItems.push(step);
      }
    });

    const result: SemanticGroup[] = [];

    if (retrieveItems.length > 0) {
      result.push({ id: 'retrieve', title: '信息检索', items: retrieveItems });
    }
    if (validateItems.length > 0) {
      result.push({ id: 'validate', title: '证据验证', items: validateItems });
    }
    if (synthesizeItems.length > 0) {
      result.push({ id: 'synthesize', title: '结果合成', items: synthesizeItems });
    }

    return result;
  }, [thinkingSteps]);

  if (!groups.length) {
    return null;
  }

  return (
    <div className="process-group-list">
      {groups.map((group) => (
        <div key={group.id} className="process-group">
          <h5 className="process-group-title">{group.title}</h5>
          <ul className="process-group-items">
            {group.items.map((item) => (
              <li key={item.id} className={`process-item status-${item.tool?.status || 'completed'}`}>
                {item.tool ? (
                  <>
                    <span className="tool-name">{item.tool.name}</span>
                    <span className="tool-status">
                      {item.tool.status === 'completed' && '✓'}
                      {item.tool.status === 'error' && '✕'}
                      {item.tool.status === 'running' && '...'}
                      {item.tool.status === 'pending' && '○'}
                    </span>
                  </>
                ) : (
                  <span className="thought-content">{item.content}</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      ))}

      {/* 调试入口 */}
      <button
        type="button"
        className="debug-toggle-btn"
        onClick={onOpenDebug}
      >
        查看调试详情
      </button>
    </div>
  );
}