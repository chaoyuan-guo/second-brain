import type { CitationRef, HonestySignals } from './chat-types';

const DATE_PATTERNS: RegExp[] = [
  /\b(20\d{2})[-_/\.](0[1-9]|1[0-2])[-_/\.](0[1-9]|[12]\d|3[01])\b/,
  /\b(20\d{2})[-_/\.](0[1-9]|1[0-2])\b/,
  /\b(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\b/,
  /\b(20\d{2})(0[1-9]|1[0-2])\b/,
  /\b(20\d{2})\b/,
];

export const normalizeCitationId = (id: string): string => (
  id.startsWith('c') ? id.slice(1) : id
);

export const isWeakRetrievalScore = (score?: number): boolean => (
  typeof score === 'number' && score >= 0.8
);

export const getCitationSuperscript = (id: string): string => {
  const normalized = normalizeCitationId(id);
  const numeric = Number.parseInt(normalized, 10);
  if (Number.isNaN(numeric)) {
    return normalized;
  }
  return String(numeric);
};

export const formatRetrievalDistance = (score?: number): string | undefined => {
  if (typeof score !== 'number') {
    return undefined;
  }
  return `L2 ${score.toFixed(2)}`;
};

export const inferSourceDateLabel = (...parts: Array<string | undefined>): string | undefined => {
  const joined = parts
    .filter((part): part is string => Boolean(part && part.trim()))
    .join(' ');

  for (const pattern of DATE_PATTERNS) {
    const match = joined.match(pattern);
    if (!match) {
      continue;
    }
    if (match.length >= 4) {
      return `${match[1]}-${match[2]}-${match[3]}`;
    }
    if (match.length >= 3) {
      return `${match[1]}-${match[2]}`;
    }
    if (match.length >= 2) {
      return match[1];
    }
  }
  return undefined;
};

export const deriveSourceTitle = (
  sourcePath: string,
  sourceTitle?: string,
  heading?: string,
): string => {
  if (sourceTitle && sourceTitle.trim()) {
    return sourceTitle.trim();
  }
  if (heading && heading.trim()) {
    return heading.trim();
  }
  const fallback = sourcePath.split('/').pop() || sourcePath;
  return fallback.trim();
};

export const sanitizeCitationSnippet = (snippet?: string): string | undefined => {
  if (!snippet || !snippet.trim()) {
    return undefined;
  }

  const contentMatch = snippet.match(/<content>([\s\S]*?)<\/content>/i);
  const extracted = contentMatch?.[1] ?? snippet;
  const normalized = extracted
    .replace(/<[^>]+>/g, ' ')
    .replace(/\b\d+:\s*/g, '')
    .replace(/\s+/g, ' ')
    .trim();

  return normalized || undefined;
};

export const extractFileLevelReferencesFromContent = (content: string): CitationRef[] => {
  if (!content.trim()) {
    return [];
  }

  const pathPattern = /((?:[\w.\-\u4e00-\u9fa5]+\/)+[\w.\-\u4e00-\u9fa5]+\.md)(?::(\d+))?/g;
  const refs: CitationRef[] = [];
  const seen = new Set<string>();
  let match: RegExpExecArray | null;

  while ((match = pathPattern.exec(content)) !== null) {
    const sourcePath = match[1];
    if (seen.has(sourcePath)) {
      continue;
    }
    seen.add(sourcePath);

    refs.push({
      id: String(refs.length + 1).padStart(2, '0'),
      sourcePath,
      sourceTitle: deriveSourceTitle(sourcePath),
      sourceDateLabel: inferSourceDateLabel(sourcePath),
      snippet: match[2] ? `回答正文引用了该文件的第 ${match[2]} 行附近内容。` : '回答正文引用了该文件中的内容。',
    });
  }

  return refs;
};

export const normalizeHonestySignalsWithReferences = (
  signals: HonestySignals | undefined,
  references: CitationRef[] | null | undefined,
): HonestySignals | undefined => {
  if (!signals || !references || references.length === 0) {
    return signals;
  }

  const hasOnlyUnscoredReferences =
    !signals.hasDirectEvidence &&
    signals.unscoredMatches.length >= references.length &&
    signals.reasonCodes.includes('weak_match');

  if (!signals.reasonCodes.includes('no_hit') && !hasOnlyUnscoredReferences) {
    return signals;
  }

  const nextReasonCodes = signals.reasonCodes.filter((code) => code !== 'no_hit');
  return {
    ...signals,
    reasonCodes: nextReasonCodes.length > 0 ? nextReasonCodes : ['weak_match'],
    evidenceQuality: signals.evidenceQuality === 'none' ? 'weak' : signals.evidenceQuality,
    limitationNote: '回答正文已显式引用相关笔记文件，但上游事件未返回精确检索分数，请优先核对原文。',
    hasDirectEvidence: true,
    retrievalHitCount: Math.max(signals.retrievalHitCount ?? 0, references.length),
    unscoredMatches: Array.from(new Set([...signals.unscoredMatches, ...references.map((ref) => ref.id)])),
  };
};
