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
