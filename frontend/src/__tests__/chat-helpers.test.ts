import { describe, expect, it } from 'vitest';

import { deriveTitle, shouldIgnoreComposerSubmitAfterAbort } from '../app/lib/chat-helpers';

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
});
