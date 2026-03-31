import { fireEvent, render, screen } from '@testing-library/react';

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
    expect(screen.getByText('展开完整分析')).toBeInTheDocument();
  });

  it('renders markdown in fallback mode instead of raw markdown syntax', () => {
    render(
      <AnswerPanel
        message={createMessage({
          content: '**完整回答**\n\n- 第一条',
        })}
        copiedKey={null}
        onCopyCode={() => {}}
      />,
    );

    expect(screen.queryByText('结构化数据暂不可用，以下为原始回答')).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '回答' })).toBeInTheDocument();
    expect(screen.getByText('完整回答', { selector: 'strong' })).toBeInTheDocument();
    expect(screen.getByText('第一条')).toBeInTheDocument();
    expect(screen.queryByText('**完整回答**')).not.toBeInTheDocument();
  });

  it('hides the analysis toggle when there is no analysis content beyond the direct answer', () => {
    render(
      <AnswerPanel
        message={createMessage({
          content: '这是直接回答。',
          directAnswer: '这是直接回答。',
        })}
        copiedKey={null}
        onCopyCode={() => {}}
      />,
    );

    expect(screen.queryByText('展开完整分析')).not.toBeInTheDocument();
  });

  it('derives file-level references from content paths and suppresses contradictory no_hit wording', () => {
    render(
      <AnswerPanel
        message={createMessage({
          content:
            '完整分析内容\n\ndata/notes/my_markdowns/动态规划.md:645\n' +
            'data/notes/my_markdowns/爬楼梯动态规划思路解析.md:49',
          directAnswer: '动态规划的核心思想是先定义状态，再写状态转移。',
          fullAnalysis:
            'data/notes/my_markdowns/动态规划.md:645\n' +
            'data/notes/my_markdowns/爬楼梯动态规划思路解析.md:49',
          honestySignals: {
            reasonCodes: ['no_hit'],
            evidenceQuality: 'none',
            weakMatches: [],
            unscoredMatches: [],
            honestyWarnings: [],
            limitationNote: '笔记中没有检索到直接相关记录，回答只能基于有限线索推断。',
            hasSufficientEvidence: false,
          },
        })}
        copiedKey={null}
        onCopyCode={() => {}}
      />,
    );

    expect(screen.getByText('当前仅拿到文件级来源，还没有稳定的精准片段证据。')).toBeInTheDocument();
  });

  it('does not fabricate inline citations from raw file paths in fallback mode', () => {
    render(
      <AnswerPanel
        message={createMessage({
          directAnswer: '结论如下。引用：data/notes/my_markdowns/动态规划.md:632',
          fullAnalysis: '完整分析见 /app/data/notes/my_markdowns/动态规划.md:632',
          references: [
            {
              id: '02',
              sourcePath: '/app/data/notes/my_markdowns/动态规划.md',
              sourceTitle: '动态规划.md',
              snippet: '动态规划是一种通过状态转移复用子问题结果的方法。',
            },
          ],
        })}
        copiedKey={null}
        onCopyCode={() => {}}
      />,
    );

    expect(screen.getByText(/结论如下。引用：data\/notes\/my_markdowns\/动态规划\.md:632/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /查看引用 02/ })).not.toBeInTheDocument();
  });

  it('does not bind inline citation tokens to file-level references', () => {
    render(
      <AnswerPanel
        message={createMessage({
          directAnswer: '这里只拿到了文件级来源 [c01]。',
          references: [
            {
              id: '01',
              sourcePath: '/app/data/notes/my_markdowns/动态规划.md',
              sourceTitle: '动态规划.md',
              snippet: '回答正文提到了该文件中的相关内容。',
              kind: 'file',
              provenance: 'content_path',
            },
          ],
        })}
        copiedKey={null}
        onCopyCode={() => {}}
      />,
    );

    expect(screen.queryByRole('button', { name: /查看引用 01/ })).not.toBeInTheDocument();
    expect(screen.getByText('这里只拿到了文件级来源 [c01]。')).toBeInTheDocument();
  });

  it('renders inline citations inside markdown list items with mixed children', () => {
    render(
      <AnswerPanel
        message={createMessage({
          directAnswer: '- **适用前提**：满足最优子结构。引用：[c01]',
          references: [
            {
              id: '01',
              sourcePath: '/app/data/notes/my_markdowns/动态规划.md',
              sourceTitle: '动态规划.md',
              snippet: '动态规划要求问题具备最优子结构。',
              charOffsetStart: 320,
              kind: 'precise',
              provenance: 'native',
            },
          ],
        })}
        copiedKey={null}
        onCopyCode={() => {}}
      />,
    );

    expect(screen.getByRole('button', { name: /查看引用 01/ })).toBeInTheDocument();
    expect(screen.queryByText('[c01]')).not.toBeInTheDocument();
  });
});
