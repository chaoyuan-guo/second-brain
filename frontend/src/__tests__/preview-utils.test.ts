import { describe, expect, it } from 'vitest';

import {
  findPreviewHighlightRange,
  getPreviewLoadOffset,
} from '../app/lib/preview-utils';

describe('preview-utils', () => {
  it('loads preview window around the evidence anchor', () => {
    expect(getPreviewLoadOffset(500)).toBe(340);
    expect(getPreviewLoadOffset(100)).toBe(0);
  });

  it('prefers full sanitized snippet matching when possible', () => {
    const content = '前文\npath.pop() 的作用是撤销上一步选择，恢复到当前层的干净状态。\n后文';
    const range = findPreviewHighlightRange(content, 0, {
      charOffset: 3,
      snippet: '<content>path.pop() 的作用是撤销上一步选择，恢复到当前层的干净状态。</content>',
    });

    expect(range).not.toBeNull();
    expect(range?.matchedBy).toBe('snippet');
    expect(content.slice(range!.start, range!.end)).toContain('path.pop()');
  });

  it('falls back to shorter snippet prefixes before offset fallback', () => {
    const snippet = 'prefix '.repeat(25) + 'tail only in snippet';
    const content = 'prefix '.repeat(17);
    const range = findPreviewHighlightRange(content, 0, {
      charOffset: 0,
      snippet,
    });

    expect(range).not.toBeNull();
    expect(range?.matchedBy).toBe('snippet');
    expect(range?.end).toBe(60);
  });

  it('falls back to the anchor line when snippet matching fails', () => {
    const content = '第一行\n第二行关键证据\n第三行';
    const range = findPreviewHighlightRange(content, 0, {
      charOffset: 5,
      snippet: '完全不存在的片段',
    });

    expect(range).not.toBeNull();
    expect(range?.matchedBy).toBe('offset');
    expect(content.slice(range!.start, range!.end)).toBe('第二行关键证据');
  });
});
