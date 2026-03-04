import { describe, expect, it } from 'vitest';

import { deriveTitle } from '../app/lib/chat-helpers';

describe('chat-helpers', () => {
  it('deriveTitle trims and truncates long content', () => {
    expect(deriveTitle('   ')).toBe('新的对话');
    expect(deriveTitle('hello')).toBe('hello');
    expect(deriveTitle('abcdefghijklmnopqrstuvwxyz')).toBe('abcdefghijklmnopqrstuvwx...');
  });
});
