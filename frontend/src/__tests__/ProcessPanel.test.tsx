import { render, screen } from '@testing-library/react';

import { ProcessPanel } from '../app/components/chat/ProcessPanel';
import type { ChatMessage } from '../app/lib/chat-types';

const createMessage = (overrides: Partial<ChatMessage> = {}): ChatMessage => ({
  id: 'assistant-process',
  role: 'assistant',
  content: '',
  ...overrides,
});

describe('ProcessPanel', () => {
  it('renders live semantic process summary while message is still thinking', () => {
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

    expect(screen.getByText('正在检索相关信息')).toBeInTheDocument();
    expect(screen.getByText(/搜索文件内容 "动态规划"/)).toBeInTheDocument();
    expect(screen.getByText('进行中')).toBeInTheDocument();
    expect(screen.queryByText(/^grep$/)).not.toBeInTheDocument();
  });
});
