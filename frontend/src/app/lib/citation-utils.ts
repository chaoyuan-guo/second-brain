import type {
  CitationKind,
  CitationProvenance,
  CitationRef,
  EvidenceRef,
  HonestySignals,
} from './chat-types';

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

export const hasPreciseEvidenceFields = (
  ref: Pick<EvidenceRef, 'citationId' | 'snippet' | 'charOffsetStart'>,
): boolean => (
  typeof ref.citationId === 'string' &&
  ref.citationId.trim().length > 0 &&
  typeof ref.charOffsetStart === 'number' &&
  !Number.isNaN(ref.charOffsetStart) &&
  typeof ref.snippet === 'string' &&
  ref.snippet.trim().length > 0
);

export const classifyEvidenceRef = (
  ref: Pick<EvidenceRef, 'citationId' | 'snippet' | 'charOffsetStart'>,
  provenance: CitationProvenance,
): { kind: CitationKind; provenance: CitationProvenance } => ({
  kind: hasPreciseEvidenceFields(ref) ? 'precise' : 'file',
  provenance,
});

export const isPreciseCitationRef = (
  ref?: Pick<CitationRef, 'kind'> | null,
): boolean => ref?.kind === 'precise';

export const getReferenceKindLabel = (ref: Pick<CitationRef, 'kind' | 'provenance'>): string => {
  if (ref.kind === 'precise' && ref.provenance === 'native') {
    return '精准片段';
  }
  if (ref.kind === 'precise' && ref.provenance === 'synthetic_read') {
    return '补偿定位';
  }
  if (ref.provenance === 'content_path') {
    return '文件级来源';
  }
  return ref.kind === 'precise' ? '精准片段' : '文件级来源';
};

export const shouldUsePrecisePreviewTarget = (
  ref?: Pick<CitationRef, 'kind'> | null,
): boolean => ref?.kind === 'precise';

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

const stripHeadingMarkers = (value: string): string => value.replace(/^#{1,6}\s+/, '').trim();

const isConversationMarkerLine = (value: string): boolean => {
  const normalized = stripHeadingMarkers(value)
    .replace(/^[>*\-\s]+/, '')
    .trim();
  return /^(?:🤖\s*)?assistant[:：]?$/i.test(normalized)
    || /^(?:🧑‍💻\s*)?user[:：]?$/i.test(normalized)
    || /^(?:assistant|user)[:：]?$/i.test(normalized);
};

const isNoiseSnippetLine = (value: string): boolean => {
  const trimmed = value.trim();
  if (!trimmed) {
    return true;
  }
  if (trimmed === '-' || /^[-*_]{3,}$/.test(trimmed)) {
    return true;
  }
  if (isConversationMarkerLine(trimmed)) {
    return true;
  }
  if (/^when citing facts from this read result/i.test(trimmed)) {
    return true;
  }
  if (/^(?:citation_id|path|type|char_offset|instruction|heading)\s*:/i.test(trimmed)) {
    return true;
  }
  return false;
};

export const sanitizeCitationSnippet = (snippet?: string): string | undefined => {
  if (!snippet || !snippet.trim()) {
    return undefined;
  }

  const extracted = snippet
    .replace(/\r\n?/g, '\n')
    .replace(/<(citation_id|path|type|char_offset|instruction|heading)>[\s\S]*?<\/\1>/gi, '\n')
    .replace(/<content>([\s\S]*?)<\/content>/gi, '$1')
    .replace(/<\/?[^>]+>/g, ' ')
    .trim();

  if (!extracted) {
    return undefined;
  }

  const lines = extracted
    .split('\n')
    .map((line) => line.replace(/\s+$/g, ''))
    .filter((line) => !isNoiseSnippetLine(line));

  if (lines.length === 0) {
    return undefined;
  }

  const normalized = lines
    .join('\n')
    .replace(/^\d+:\s*/, '')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]{2,}/g, ' ')
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
      snippet: match[2] ? `回答正文提到了该文件的第 ${match[2]} 行附近内容。` : '回答正文提到了该文件中的相关内容。',
      kind: 'file',
      provenance: 'content_path',
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

  const preciseNativeRefs = references.filter(
    (ref) => ref.kind === 'precise' && ref.provenance === 'native',
  );
  if (preciseNativeRefs.length > 0) {
    return signals;
  }

  const preciseSyntheticRefs = references.filter(
    (ref) => ref.kind === 'precise' && ref.provenance === 'synthetic_read',
  );
  const fileRefs = references.filter((ref) => ref.kind === 'file');
  const nextReasonCodes = signals.reasonCodes.filter((code) => code !== 'no_hit');
  const limitationNote = preciseSyntheticRefs.length > 0
    ? '当前可点击定位主要来自读取补偿，不是上游稳定返回的原生证据链。'
    : fileRefs.length > 0
      ? '当前仅拿到文件级来源，还没有稳定的精准片段证据。'
      : signals.limitationNote;

  return {
    ...signals,
    reasonCodes: nextReasonCodes.length > 0 ? nextReasonCodes : ['insufficient_hits'],
    evidenceQuality:
      preciseSyntheticRefs.length > 0
        ? 'partial'
        : fileRefs.length > 0
          ? 'weak'
          : signals.evidenceQuality,
    limitationNote,
    hasDirectEvidence: preciseSyntheticRefs.length > 0,
    hasSufficientEvidence: false,
    retrievalHitCount: Math.max(signals.retrievalHitCount ?? 0, references.length),
    unscoredMatches: Array.from(new Set([...signals.unscoredMatches, ...references.map((ref) => ref.id)])),
  };
};
