import { render, screen } from '@testing-library/react';

import { AnswerPanel } from '../app/components/chat/AnswerPanel';
import type { ChatMessage } from '../app/lib/chat-types';

const createMessage = (overrides: Partial<ChatMessage> = {}): ChatMessage => ({
  id: 'assistant-1',
  role: 'assistant',
  content: '默认回答内容',
  ...overrides,
});

describe('AnswerPanel', () => {
  it('renders answer-first layout when new final-event fields are present without legacy structured fields', () => {
    render(
      <AnswerPanel
        message={createMessage({
          content: '完整分析内容',
          directAnswer: '这是直接回答。',
          fullAnalysis: '这是完整分析内容。',
          references: [
            {
              id: '01',
              sourcePath: 'data/notes/my_markdowns/动态规划.md',
              sourceTitle: '动态规划',
              snippet: '动态规划是一种保存中间结果的算法设计方法。',
            },
          ],
        })}
        copiedKey={null}
        onCopyCode={() => {}}
      />,
    );

    expect(screen.queryByText('结构化数据暂不可用，以下为原始回答')).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '回答' })).toBeInTheDocument();
    expect(screen.getByText('这是直接回答。')).toBeInTheDocument();
    expect(screen.getByText('引用来源')).toBeInTheDocument();
    expect(screen.getByText('展开完整分析')).toBeInTheDocument();
  });

  it('renders markdown in degraded mode instead of raw markdown syntax', () => {
    render(
      <AnswerPanel
        message={createMessage({
          content: '**完整回答**\n\n- 第一条',
        })}
        copiedKey={null}
        onCopyCode={() => {}}
      />,
    );

    expect(screen.getByText('结构化数据暂不可用，以下为原始回答')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '完整回答' })).toBeInTheDocument();
    expect(screen.getByText('完整回答', { selector: 'strong' })).toBeInTheDocument();
    expect(screen.getByText('第一条')).toBeInTheDocument();
    expect(screen.queryByText('**完整回答**')).not.toBeInTheDocument();
  });
});
