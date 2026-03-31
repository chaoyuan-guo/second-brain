import { describe, expect, it } from 'vitest';

import { sanitizeCitationSnippet } from '../app/lib/citation-utils';

describe('sanitizeCitationSnippet', () => {
  it('removes xml wrappers and standalone instruction lines', () => {
    const snippet = [
      '<path>/notes/dp.md</path>',
      '<instruction>When citing facts from this read result, use the exact citation_id above and do not renumber it.</instruction>',
      '<content>620: 动态规划的核心是状态转移。</content>',
    ].join('\n');

    expect(sanitizeCitationSnippet(snippet)).toBe('动态规划的核心是状态转移。');
  });

  it('removes standalone assistant and user markers without touching正文', () => {
    const snippet = [
      '## 🤖 Assistant',
      '',
      'path.pop() 的作用是撤销上一步选择。',
      '',
      '## 🧑‍💻 User',
    ].join('\n');

    expect(sanitizeCitationSnippet(snippet)).toBe('path.pop() 的作用是撤销上一步选择。');
  });

  it('preserves normal content containing Assistant or User words inline', () => {
    const snippet = '这里讨论 Assistant API 和 User Prompt 的区别。';

    expect(sanitizeCitationSnippet(snippet)).toBe(snippet);
  });
});

