'use client';

import { useMemo } from 'react';
import type { ThinkingStep } from '../../lib/chat-types';

interface ProcessGroupListProps {
  thinkingSteps?: ThinkingStep[];
  onOpenDebug: () => void;
}

type SemanticGroup = {
  id: string;
  title: string;
  items: ThinkingStep[];
  toolCount: number;
  thoughtCount: number;
};

/**
 * 工具名到语义阶段的映射
 * 基于规范文档 6.2 节主工具集合
 */
const toolSemanticMap: Record<string, 'retrieve' | 'validate' | 'synthesize_helper' | 'other'> = {
  read: 'retrieve',
  grep: 'retrieve',
  glob: 'retrieve',
  bash: 'validate',
};

const getToolSemantic = (toolName: string): 'retrieve' | 'validate' | 'synthesize_helper' | 'other' => {
  return toolSemanticMap[toolName] || 'other';
};

/**
 * 判断两个 thought 内容是否相似（用于去重）
 */
const isSimilarThought = (a: string, b: string): boolean => {
  const normalize = (s: string) => s.toLowerCase().replace(/\s+/g, '').slice(0, 50);
  return normalize(a) === normalize(b);
};

/**
 * ProcessGroupList - 过程分组列表
 * 将工具调用按语义分组展示：信息检索、证据验证、结果合成
 * 默认摘要化，合并重复 thought，工具调用为主要展示内容
 */
export function ProcessGroupList({ thinkingSteps, onOpenDebug }: ProcessGroupListProps) {
  const groups = useMemo(() => {
    if (!thinkingSteps?.length) return [];

    const retrieveItems: ThinkingStep[] = [];
    const validateItems: ThinkingStep[] = [];
    const synthesizeItems: ThinkingStep[] = [];
    let retrieveThoughtCount = 0;
    let validateThoughtCount = 0;
    let synthesizeThoughtCount = 0;

    // 用于 thought 去重
    const seenThoughts: string[] = [];

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
        // 去重：只记录不相似的 thought
        const isDuplicate = seenThoughts.some(t => isSimilarThought(t, step.content || ''));
        if (!isDuplicate && step.content && step.content.length > 10) {
          seenThoughts.push(step.content);
          // 默认只保留最新的一条 thought 在合成组
          synthesizeItems.push(step);
          synthesizeThoughtCount++;
        }
      } else if (step.type === 'synthesize') {
        synthesizeItems.push(step);
      }
    });

    const result: SemanticGroup[] = [];

    if (retrieveItems.length > 0 || retrieveThoughtCount > 0) {
      result.push({ 
        id: 'retrieve', 
        title: '信息检索', 
        items: retrieveItems.filter(i => i.type === 'tool'),
        toolCount: retrieveItems.filter(i => i.type === 'tool').length,
        thoughtCount: retrieveThoughtCount
      });
    }
    if (validateItems.length > 0 || validateThoughtCount > 0) {
      result.push({ 
        id: 'validate', 
        title: '证据验证', 
        items: validateItems.filter(i => i.type === 'tool'),
        toolCount: validateItems.filter(i => i.type === 'tool').length,
        thoughtCount: validateThoughtCount
      });
    }
    if (synthesizeItems.length > 0 || synthesizeThoughtCount > 0) {
      // 合成组最多只保留 2 条 thought
      const limitedSynthItems = synthesizeItems.filter(i => i.type === 'tool').slice(0, 10);
      const synthThoughts = synthesizeItems.filter(i => i.type === 'thought').slice(-1);
      result.push({ 
        id: 'synthesize', 
        title: '结果合成', 
        items: [...limitedSynthItems, ...synthThoughts],
        toolCount: limitedSynthItems.length,
        thoughtCount: synthThoughts.length
      });
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
          <h5 className="process-group-title">
            {group.title}
            {group.thoughtCount > 0 && (
              <span className="process-group-meta">（{group.toolCount} 个步骤）</span>
            )}
          </h5>
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
