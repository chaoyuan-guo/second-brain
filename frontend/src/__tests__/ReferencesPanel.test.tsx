import { fireEvent, render, screen } from '@testing-library/react';
import { vi } from 'vitest';

import { ReferencesPanel } from '../app/components/chat/ReferencesPanel';

const references = [
  {
    id: '01',
    sourcePath: '/notes/a.md',
    sourceTitle: 'A.md',
    snippet: '<path>/notes/a.md</path><content>100: 关键结论 A。</content>',
    charOffsetStart: 100,
    kind: 'precise' as const,
    provenance: 'native' as const,
    retrievalScore: 0.11,
  },
  {
    id: '02',
    sourcePath: '/notes/a.md',
    sourceTitle: 'A.md',
    snippet: '同一文件的另一条片段，不应默认重复占位。',
    charOffsetStart: 160,
    kind: 'precise' as const,
    provenance: 'native' as const,
    retrievalScore: 0.21,
  },
  {
    id: '03',
    sourcePath: '/notes/b.md',
    sourceTitle: 'B.md',
    snippet: '关键结论 B。',
    charOffsetStart: 42,
    kind: 'precise' as const,
    provenance: 'synthetic_read' as const,
    retrievalScore: 0.2,
  },
  {
    id: '04',
    sourcePath: '/notes/c.md',
    sourceTitle: 'C.md',
    snippet: '文件级来源 C。',
    kind: 'file' as const,
    provenance: 'content_path' as const,
  },
  {
    id: '05',
    sourcePath: '/notes/d.md',
    sourceTitle: 'D.md',
    snippet: '文件级来源 D。',
    kind: 'file' as const,
    provenance: 'content_path' as const,
  },
];

describe('ReferencesPanel', () => {
  it('shows a simplified default list without sorting or quality badges', () => {
    render(<ReferencesPanel references={references} onOpenPreview={() => {}} />);

    expect(screen.getByText('引用来源')).toBeInTheDocument();
    expect(screen.getByText('(4)')).toBeInTheDocument();
    expect(screen.queryByText('按相关性')).not.toBeInTheDocument();
    expect(screen.queryByText(/原生精准/)).not.toBeInTheDocument();
    expect(screen.queryByText(/高相关/)).not.toBeInTheDocument();
    expect(screen.getByText('A.md')).toBeInTheDocument();
    expect(screen.getByText('B.md')).toBeInTheDocument();
    expect(screen.getByText('C.md')).toBeInTheDocument();
    expect(screen.getByText('D.md')).toBeInTheDocument();
    expect(screen.queryByText('同一文件的另一条片段，不应默认重复占位。')).not.toBeInTheDocument();
  });

  it('expands a reference and keeps preview action available', () => {
    const onOpenPreview = vi.fn();

    render(<ReferencesPanel references={references} onOpenPreview={onOpenPreview} />);

    fireEvent.click(screen.getByText('A.md', { selector: '.ref-title' }));

    expect(screen.getByText('关键结论 A。')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '定位原文' }));

    expect(onOpenPreview).toHaveBeenCalledWith(
      '/notes/a.md',
      'A.md',
      expect.objectContaining({ char_offset: 100, snippet: '关键结论 A。' }),
    );
  });
});
