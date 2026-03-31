import { fireEvent, render, screen } from '@testing-library/react';

import { ProcessPanel } from '../app/components/chat/ProcessPanel';
import type { ChatMessage } from '../app/lib/chat-types';

const createMessage = (overrides: Partial<ChatMessage> = {}): ChatMessage => ({
  id: 'assistant-process',
  role: 'assistant',
  content: '',
  ...overrides,
});

describe('ProcessPanel', () => {
  it('stays collapsed by default while message is still thinking', () => {
    render(
      <ProcessPanel
        message={createMessage({
          isThinking: true,
          thinkingSteps: [
            {
              id: 'thought-1',
              type: 'thought',
              content: '正在思考...',
              timestamp: 1000,
            },
            {
              id: 'tool-1',
              type: 'tool',
              timestamp: 1100,
              tool: {
                id: 'call-1',
                name: 'grep',
                status: 'running',
                arguments: { pattern: '动态规划' },
                startedAt: 1100,
              },
            },
          ],
        })}
      />,
    );

    expect(screen.getByText('正在整理依据')).toBeInTheDocument();
    expect(screen.queryByText(/搜索文件内容 "动态规划"/)).not.toBeInTheDocument();
    expect(screen.queryByText('进行中')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '展开过程详情' }));

    expect(screen.getByText(/搜索文件内容 "动态规划"/)).toBeInTheDocument();
    expect(screen.getByText('进行中')).toBeInTheDocument();
  });
});
