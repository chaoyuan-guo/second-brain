'use client';

interface ActionChecklistProps {
  actions: string[];
}

/**
 * ActionChecklist - 下一步行动清单
 * 展示 3-5 条可执行行动项
 */
export function ActionChecklist({ actions }: ActionChecklistProps) {
  if (!actions?.length) {
    return null;
  }

  // 最多展示 5 条
  const displayActions = actions.slice(0, 5);

  return (
    <div className="action-checklist">
      <h4 className="action-title">下一步行动</h4>
      <ul className="action-list">
        {displayActions.map((action, index) => (
          <li key={`action-${index}`} className="action-item">
            <span className="action-number">{index + 1}</span>
            <span className="action-text">{action}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}