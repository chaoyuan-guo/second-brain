import { describe, expect, it } from 'vitest';
import {
  synthesizeFinalEvent,
  extractCitations,
  enrichCitationsWithEvidence,
  splitDirectAnswer,
  generateProcessSummary,
  computeHonestySignals,
  createProcessAccumulator,
  deriveSyntheticSourceRefsFromCall,
  mergeSourceRefs,
  updateToolCall,
  type ToolCallRecord,
} from '../app/api/_lib/event-adapter';
import type { CitationRef, EvidenceRef } from '../app/lib/chat-types';

// ============================================================================
// extractCitations 测试
// ============================================================================

describe('extractCitations', () => {
  it('should extract [cxx] citations from content', () => {
    const content = '这是回答 [c01] 包含引用 [c02] 多个引用 [c12]';
    const citations = extractCitations(content);
    
    expect(citations).toHaveLength(3);
    expect(citations[0].id).toBe('01');
    expect(citations[1].id).toBe('02');
    expect(citations[2].id).toBe('12');
  });

  it('should handle duplicate citations (deduplicate)', () => {
    const content = '重复引用 [c01] 再次出现 [c01] 和 [c01]';
    const citations = extractCitations(content);
    
    expect(citations).toHaveLength(1);
    expect(citations[0].id).toBe('01');
  });

  it('should return empty array for content without citations', () => {
    const content = '这是没有引用的纯文本回答';
    const citations = extractCitations(content);
    
    expect(citations).toHaveLength(0);
  });

  it('should handle edge case citation IDs', () => {
    const content = '边界测试 [c00] [c99] [c001]';
    const citations = extractCitations(content);
    
    // 只匹配 2-3 位数字
    expect(citations).toHaveLength(3);
    expect(citations.map(c => c.id)).toContain('00');
    expect(citations.map(c => c.id)).toContain('99');
  });
});

// ============================================================================
// enrichCitationsWithEvidence 测试
// ============================================================================

describe('enrichCitationsWithEvidence', () => {
  it('should enrich citations with evidence info', () => {
    const citations: CitationRef[] = [
      { id: '01', sourcePath: '', retrievalScore: undefined, snippet: undefined },
      { id: '02', sourcePath: '', retrievalScore: undefined, snippet: undefined },
    ];
    
    const sourceRefMap = new Map<string, EvidenceRef>([
      ['path1|heading|0|snippet', {
        sourcePath: '/notes/doc1.md',
        sourceTitle: '文档1',
        heading: '章节1',
        charOffsetStart: 0,
        snippet: '内容片段1',
        citationId: '01',
        retrievalScore: 0.85,
      }],
    ]);

    const enriched = enrichCitationsWithEvidence(citations, sourceRefMap);
    
    expect(enriched[0].sourcePath).toBe('/notes/doc1.md');
    expect(enriched[0].sourceTitle).toBe('文档1');
    expect(enriched[0].sourceDateLabel).toBeUndefined();
    expect(enriched[0].retrievalScore).toBe(0.85);
    expect(enriched[0].weakMatch).toBe(true);
    expect(enriched[0].snippet).toBe('内容片段1');
    expect(enriched[1].sourcePath).toBe(''); // 未找到匹配
  });

  it('should infer source date label from path when possible', () => {
    const citations: CitationRef[] = [
      { id: '01', sourcePath: '', retrievalScore: undefined, snippet: undefined },
    ];

    const sourceRefMap = new Map<string, EvidenceRef>([
      ['path1|heading|0|snippet', {
        sourcePath: '/notes/2024-09-15-review.md',
        charOffsetStart: 0,
        snippet: '内容片段1',
        citationId: '01',
      }],
    ]);

    const enriched = enrichCitationsWithEvidence(citations, sourceRefMap);
    expect(enriched[0].sourceDateLabel).toBe('2024-09-15');
  });

  it('should handle empty sourceRefMap', () => {
    const citations: CitationRef[] = [
      { id: '01', sourcePath: '', retrievalScore: undefined, snippet: undefined },
    ];
    
    const enriched = enrichCitationsWithEvidence(citations, new Map());
    
    expect(enriched[0].sourcePath).toBe('');
  });

  it('should sanitize xml-like snippet wrappers from evidence refs', () => {
    const citations: CitationRef[] = [
      { id: '01', sourcePath: '', retrievalScore: undefined, snippet: undefined },
    ];

    const sourceRefMap = new Map<string, EvidenceRef>([
      ['path1|heading|0|snippet', {
        sourcePath: '/notes/dp.md',
        charOffsetStart: 620,
        snippet: '<path>/notes/dp.md</path><content>620: 动态规划的核心是状态转移。</content>',
        citationId: '01',
      }],
    ]);

    const enriched = enrichCitationsWithEvidence(citations, sourceRefMap);
    expect(enriched[0].snippet).toBe('动态规划的核心是状态转移。');
  });
});

describe('deriveSyntheticSourceRefsFromCall', () => {
  it('assigns stable synthetic citation ids for read calls in read order', () => {
    const sourceRefMap = new Map<string, EvidenceRef>();

    const firstRefs = deriveSyntheticSourceRefsFromCall(
      {
        name: 'read',
        arguments: { filePath: '/notes/dp_notes.md', offset: 42 },
        result: { content: '动态规划先定义状态，再写状态转移。' },
      },
      sourceRefMap,
    );
    mergeSourceRefs(
      {
        startTime: Date.now(),
        activeCalls: new Map(),
        completedCalls: [],
        errorCalls: [],
        assistantContent: '',
        eventVersion: 0,
        sourceRefMap,
      },
      firstRefs,
    );

    const secondRefs = deriveSyntheticSourceRefsFromCall(
      {
        name: 'read',
        arguments: { filePath: '/notes/greedy.md', offset: 12 },
        result: { content: '贪心依赖局部最优的可证明性。' },
      },
      sourceRefMap,
    );

    expect(firstRefs[0].citationId).toBe('c01');
    expect(secondRefs[0].citationId).toBe('c02');
    expect(firstRefs[0].snippet).toContain('动态规划先定义状态');
  });
});

// ============================================================================
// splitDirectAnswer 测试
// ============================================================================

describe('splitDirectAnswer', () => {
  it('should split content by separator', () => {
    const content = '这是直接回答\n---\n这是详细分析';
    const { directAnswer, fullAnalysis } = splitDirectAnswer(content);
    
    expect(directAnswer).toBe('这是直接回答');
    expect(fullAnalysis).toBe('这是详细分析');
  });

  it('should split by "完整分析" marker', () => {
    const content = '这是直接回答。完整分析：这是详细分析内容';
    const { directAnswer, fullAnalysis } = splitDirectAnswer(content);
    
    expect(directAnswer).toBe('这是直接回答。');
    expect(fullAnalysis).toContain('完整分析');
  });

  it('should default to first 2-3 sentences', () => {
    const content = '第一句。第二句。第三句。第四句。第五句。';
    const { directAnswer, fullAnalysis } = splitDirectAnswer(content);
    
    expect(directAnswer.split('。').length).toBeGreaterThanOrEqual(2);
    expect(fullAnalysis.length).toBeGreaterThan(0);
  });

  it('should handle short content', () => {
    const content = '只有一句话';
    const { directAnswer, fullAnalysis } = splitDirectAnswer(content);
    
    expect(directAnswer).toBe('只有一句话');
    expect(fullAnalysis).toBe('');
  });

  it('should not split on markdown file extensions inside citations', () => {
    const content =
      '动态规划核心思想（基于你的笔记）：先定义状态，再写状态转移（data/notes/my_markdowns/动态规划.md:632）。然后再考虑初始化。';
    const { directAnswer, fullAnalysis } = splitDirectAnswer(content);

    expect(directAnswer).toContain('动态规划.md:632');
    expect(directAnswer).toContain('然后再考虑初始化');
    expect(fullAnalysis).toBe('');
  });
});

// ============================================================================
// generateProcessSummary 测试
// ============================================================================

describe('generateProcessSummary', () => {
  it('should generate semantic summaries for tool calls', () => {
    const completedCalls: ToolCallRecord[] = [
      {
        id: '1',
        name: 'grep',
        status: 'completed',
        arguments: { query: '搜索关键词' },
        result: [{ file: 'test.md' }, { file: 'other.md' }],
        sourceRefs: [
          { sourcePath: 'test.md', citationId: '01' },
          { sourcePath: 'other.md', citationId: '02' },
        ],
        startedAt: 1000,
        completedAt: 2000,
      },
    ];

    const summaries = generateProcessSummary(completedCalls, []);
    
    expect(summaries).toHaveLength(1);
    expect(summaries[0].summary).toContain('搜索文件内容');
    expect(summaries[0].summary).toContain('命中 2 条');
    expect(summaries[0].phase).toBe('retrieving');
    expect(summaries[0].durationMs).toBe(1000);
    expect(summaries[0].detail).toContain('输入：');
    expect(summaries[0].detail).toContain('结果：');
    expect(summaries[0].status).toBe('completed');
  });

  it('should handle error calls', () => {
    const errorCalls: ToolCallRecord[] = [
      {
        id: '1',
        name: 'read_note_file',
        status: 'error',
        arguments: { path: 'missing.md' },
        error: '文件不存在',
        startedAt: 1000,
        completedAt: 1500,
      },
    ];

    const summaries = generateProcessSummary([], errorCalls);
    
    expect(summaries[0].summary).toContain('失败');
    expect(summaries[0].detail).toContain('文件不存在');
  });

  it('should sort calls by start time', () => {
    const calls: ToolCallRecord[] = [
      { id: '2', name: 'read_note_file', status: 'completed', startedAt: 2000, completedAt: 2500 },
      { id: '1', name: 'grep', status: 'completed', startedAt: 1000, completedAt: 1500 },
    ];

    const summaries = generateProcessSummary(calls, []);
    
    expect(summaries[0].stepNumber).toBe(1);
    expect(summaries[1].stepNumber).toBe(2);
  });

  it('should summarize generic read tool as file reading', () => {
    const calls: ToolCallRecord[] = [
      {
        id: '1',
        name: 'read',
        status: 'completed',
        arguments: { filePath: '/notes/dp_notes.md', offset: 120 },
        startedAt: 1000,
        completedAt: 1800,
      },
    ];

    const summaries = generateProcessSummary(calls, []);

    expect(summaries[0].summary).toContain('读取文件');
    expect(summaries[0].summary).toContain('dp_notes.md');
  });

  it('should summarize bash note discovery commands semantically', () => {
    const calls: ToolCallRecord[] = [
      {
        id: '1',
        name: 'bash',
        status: 'completed',
        arguments: {
          command: "find /app/data/notes/my_markdowns -name '*.md' -maxdepth 2",
          description: 'Find markdown notes files',
        },
        result: '/app/data/notes/my_markdowns/a.md\n/app/data/notes/my_markdowns/b.md\n',
        startedAt: 1000,
        completedAt: 1300,
      },
      {
        id: '2',
        name: 'bash',
        status: 'completed',
        arguments: {
          command: "rg -n --glob '*.md' '动态规划|DP|dp' /app/data/notes/my_markdowns",
          description: 'Search DP keywords in notes',
        },
        result:
          '/app/data/notes/my_markdowns/动态规划.md:1:# 动态规划\n' +
          '/app/data/notes/my_markdowns/爬楼梯动态规划思路解析.md:1:# 爬楼梯动态规划思路解析\n',
        startedAt: 1400,
        completedAt: 1700,
      },
    ];

    const summaries = generateProcessSummary(calls, []);

    expect(summaries[0].summary).toContain('定位候选文件');
    expect(summaries[0].summary).toContain('命中 2 个文件');
    expect(summaries[1].summary).toContain('检索笔记');
    expect(summaries[1].summary).toContain('来自 2 个文件');
  });
});

// ============================================================================
// computeHonestySignals 测试
// ============================================================================

describe('computeHonestySignals', () => {
  it('should identify strong evidence quality', () => {
    // L2 距离越小越相似，< 0.8 为强匹配
    const citations: CitationRef[] = [
      { id: '01', sourcePath: '/a.md', retrievalScore: 0.5 },
      { id: '02', sourcePath: '/b.md', retrievalScore: 0.6 },
      { id: '03', sourcePath: '/c.md', retrievalScore: 0.7 },
    ];

    const signals = computeHonestySignals(citations, false, 0);
    
    expect(signals.evidenceQuality).toBe('strong');
    expect(signals.hasSufficientEvidence).toBe(true);
    expect(signals.honestyWarnings).toHaveLength(0);
    expect(signals.reasonCodes).toEqual([]);
  });

  it('should identify weak evidence quality', () => {
    // L2 距离 >= 0.8 为弱匹配（相关性低）
    const citations: CitationRef[] = [
      { id: '01', sourcePath: '/a.md', retrievalScore: 0.9 },
      { id: '02', sourcePath: '/b.md', retrievalScore: 0.85 },
    ];

    const signals = computeHonestySignals(citations, false, 0);
    
    expect(signals.evidenceQuality).toBe('weak');
    expect(signals.hasSufficientEvidence).toBe(false);
    expect(signals.honestyWarnings.length).toBeGreaterThan(0);
    expect(signals.limitationNote).toBeDefined();
    expect(signals.reasonCodes).toContain('weak_match');
  });

  it('should identify partial evidence quality', () => {
    // 混合：0.6 < 0.8 为强匹配，0.85 >= 0.8 为弱匹配
    const citations: CitationRef[] = [
      { id: '01', sourcePath: '/a.md', retrievalScore: 0.85 },
      { id: '02', sourcePath: '/b.md', retrievalScore: 0.6 },
    ];

    const signals = computeHonestySignals(citations, false, 0);
    
    expect(signals.evidenceQuality).toBe('partial');
    expect(signals.weakMatches).toContain('01'); // 0.85 >= 0.8 是弱匹配
    expect(signals.reasonCodes).toContain('insufficient_hits');
  });

  it('should handle error calls', () => {
    const citations: CitationRef[] = [
      { id: '01', sourcePath: '/a.md', retrievalScore: 0.5 },
    ];

    const signals = computeHonestySignals(citations, true, 2);
    
    expect(signals.honestyWarnings.some(w => w.includes('工具执行失败'))).toBe(true);
  });

  it('should handle unscored citations', () => {
    const citations: CitationRef[] = [
      { id: '01', sourcePath: '/a.md' },
      { id: '02', sourcePath: '/b.md' },
    ];

    const signals = computeHonestySignals(citations, false, 0);
    
    expect(signals.evidenceQuality).toBe('weak');
    expect(signals.unscoredMatches).toEqual(['01', '02']);
    expect(signals.reasonCodes).toContain('weak_match');
  });

  it('should emit no_hit reason when there are no citations', () => {
    const signals = computeHonestySignals([], false, 0);
    expect(signals.reasonCodes).toEqual(['no_hit']);
    expect(signals.hasSufficientEvidence).toBe(false);
  });
});

// ============================================================================
// synthesizeFinalEvent 测试
// ============================================================================

describe('synthesizeFinalEvent', () => {
  it('should preserve completed status when a tool transitions from running to completed', () => {
    const acc = createProcessAccumulator();

    updateToolCall(acc, 'call-1', 'bash', 'running', {
      arguments: { command: "find /app/data/notes/my_markdowns -name '*.md'" },
      startedAt: 1000,
    });
    updateToolCall(acc, 'call-1', 'bash', 'completed', {
      result: '/app/data/notes/my_markdowns/动态规划.md\n',
      completedAt: 1200,
    });
    acc.assistantContent = '已完成总结。';

    const finalEvent = synthesizeFinalEvent(acc);

    expect(finalEvent.processSummary?.[0].status).toBe('completed');
    expect(finalEvent.processSummary?.[0].summary).toContain('定位候选文件');
    expect(finalEvent.processSummary?.[0].summary).not.toContain('进行中');
  });

  it('should derive fallback references from read calls when inline citations are missing', () => {
    const finalEvent = synthesizeFinalEvent({
      startTime: Date.now() - 1200,
      activeCalls: new Map(),
      completedCalls: [
        {
          id: 'read-1',
          name: 'read',
          status: 'completed',
          arguments: { filePath: '/notes/dp_notes.md', offset: 42 },
          result: { content: '动态规划的核心是定义状态并写出状态转移方程。' },
          startedAt: 1000,
          completedAt: 1800,
        },
      ],
      errorCalls: [],
      assistantContent: '动态规划的核心是先定义状态，再设计状态转移。',
      eventVersion: 0,
      sourceRefMap: new Map(),
    });

    expect(finalEvent.references).toHaveLength(1);
    expect(finalEvent.references?.[0].sourcePath).toBe('/notes/dp_notes.md');
    expect(finalEvent.references?.[0].charOffsetStart).toBeUndefined();
    expect(finalEvent.honestySignals).toBeUndefined();
  });

  it('should resolve inline citations through synthetic read refs when metadata source_refs are absent', () => {
    const sourceRefMap = new Map<string, EvidenceRef>();
    const syntheticRefs = deriveSyntheticSourceRefsFromCall(
      {
        name: 'read',
        arguments: { filePath: '/notes/dp_notes.md', offset: 42 },
        result: { content: '动态规划的核心是先定义状态，再设计状态转移。' },
      },
      sourceRefMap,
    );

    const acc = {
      startTime: Date.now() - 1200,
      activeCalls: new Map(),
      completedCalls: [
        {
          id: 'read-1',
          name: 'read',
          status: 'completed' as const,
          arguments: { filePath: '/notes/dp_notes.md', offset: 42 },
          result: { content: '动态规划的核心是先定义状态，再设计状态转移。' },
          startedAt: 1000,
          completedAt: 1800,
          sourceRefs: syntheticRefs,
        },
      ],
      errorCalls: [],
      assistantContent: '动态规划的核心是先定义状态，再设计状态转移。[c01]',
      eventVersion: 0,
      sourceRefMap,
    };
    mergeSourceRefs(acc, syntheticRefs);

    const finalEvent = synthesizeFinalEvent(acc);

    expect(finalEvent.references).toHaveLength(1);
    expect(finalEvent.references?.[0].id).toBe('01');
    expect(finalEvent.references?.[0].sourcePath).toBe('/notes/dp_notes.md');
    expect(finalEvent.references?.[0].snippet).toContain('动态规划的核心');
  });

  it('should collapse repeated read calls from the same file into one file-level fallback reference', () => {
    const finalEvent = synthesizeFinalEvent({
      startTime: Date.now() - 1200,
      activeCalls: new Map(),
      completedCalls: [
        {
          id: 'read-1',
          name: 'read',
          status: 'completed',
          arguments: { filePath: '/notes/dp_notes.md', offset: 42 },
          result: { content: '第一段：动态规划先定义状态。' },
          startedAt: 1000,
          completedAt: 1400,
        },
        {
          id: 'read-2',
          name: 'read',
          status: 'completed',
          arguments: { filePath: '/notes/dp_notes.md', offset: 360 },
          result: { content: '第二段：再写状态转移与初始化。' },
          startedAt: 1500,
          completedAt: 1800,
        },
      ],
      errorCalls: [],
      assistantContent: '动态规划的核心是先定义状态，再设计状态转移。',
      eventVersion: 0,
      sourceRefMap: new Map(),
    });

    expect(finalEvent.references).toHaveLength(1);
    expect(finalEvent.references?.[0].sourcePath).toBe('/notes/dp_notes.md');
    expect(finalEvent.references?.[0].snippet?.length ?? 0).toBeGreaterThan(0);
  });

  it('should derive content fallback references and normalize no_hit when answer mentions note paths', () => {
    const finalEvent = synthesizeFinalEvent({
      startTime: Date.now() - 1200,
      activeCalls: new Map(),
      completedCalls: [],
      errorCalls: [],
      assistantContent:
        '这是结论。引用：data/notes/my_markdowns/动态规划.md:632\n' +
        '更多说明见 data/notes/my_markdowns/爬楼梯动态规划思路解析.md:49',
      eventVersion: 0,
      sourceRefMap: new Map(),
    });

    expect(finalEvent.references).toHaveLength(2);
    expect(finalEvent.references?.[0].sourcePath).toBe('data/notes/my_markdowns/动态规划.md');
    expect(finalEvent.honestySignals).toBeUndefined();
  });

  it('should mark final process phase as completed', () => {
    const finalEvent = synthesizeFinalEvent({
      startTime: Date.now() - 500,
      activeCalls: new Map(),
      completedCalls: [],
      errorCalls: [],
      assistantContent: '这是最终回答。',
      eventVersion: 0,
      sourceRefMap: new Map(),
    });

    expect(finalEvent.processOverview.phase).toBe('completed');
  });
});
