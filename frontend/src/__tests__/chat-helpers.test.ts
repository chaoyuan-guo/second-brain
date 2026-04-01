import { describe, expect, it } from 'vitest';

import {
  deriveTitle,
  getUserFacingAssistantStatusText,
  hasRenderableAssistantAnswer,
  shouldIgnoreComposerSubmitAfterAbort,
} from '../app/lib/chat-helpers';

describe('chat-helpers', () => {
  it('deriveTitle trims and truncates long content', () => {
    expect(deriveTitle('   ')).toBe('新的对话');
    expect(deriveTitle('hello')).toBe('hello');
    expect(deriveTitle('abcdefghijklmnopqrstuvwxyz')).toBe('abcdefghijklmnopqrstuvwx...');
  });

  it('ignores submit briefly after abort to avoid accidental resend', () => {
    expect(shouldIgnoreComposerSubmitAfterAbort(0, 1000)).toBe(false);
    expect(shouldIgnoreComposerSubmitAfterAbort(1000, 1200)).toBe(true);
    expect(shouldIgnoreComposerSubmitAfterAbort(1000, 1400)).toBe(false);
  });

  it('returns long-running user-facing status text without leaking internals', () => {
    expect(
      getUserFacingAssistantStatusText({
        assistantContent: '',
        displayState: 'long_running',
      }),
    ).toBe('正在整理依据');

    expect(
      getUserFacingAssistantStatusText({
        assistantContent: '已经有部分回答',
        displayState: 'long_running',
      }),
    ).toBe('正在继续整理依据');
  });

  it('identifies when assistant answer should remain visible during streaming or fallback states', () => {
    expect(
      hasRenderableAssistantAnswer({
        role: 'assistant',
        content: '已有正文',
        directAnswer: undefined,
        fullAnalysis: undefined,
        completionState: undefined,
      }),
    ).toBe(true);

    expect(
      hasRenderableAssistantAnswer({
        role: 'assistant',
        content: '',
        directAnswer: '',
        fullAnalysis: '',
        completionState: 'failed',
      }),
    ).toBe(true);

    expect(
      hasRenderableAssistantAnswer({
        role: 'assistant',
        content: '',
        directAnswer: '',
        fullAnalysis: '',
        completionState: undefined,
      }),
    ).toBe(false);
  });
});
