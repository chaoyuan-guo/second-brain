import { sanitizeCitationSnippet } from './citation-utils';

export interface PreviewScrollTarget {
  charOffset: number;
  snippet?: string;
}

export interface PreviewHighlightRange {
  start: number;
  end: number;
  matchedBy: 'snippet' | 'offset';
}

export const normalizeWhitespace = (input: string): string => {
  let result = '';
  let lastWasSpace = false;
  for (let i = 0; i < input.length; i += 1) {
    const ch = input[i];
    if (/\s/.test(ch)) {
      if (!lastWasSpace) {
        result += ' ';
        lastWasSpace = true;
      }
    } else {
      result += ch;
      lastWasSpace = false;
    }
  }
  return result.trim();
};

export const normalizeWhitespaceForSearch = (input: string): { normalized: string; map: number[] } => {
  let normalized = '';
  const map: number[] = [];
  let lastWasSpace = false;
  for (let i = 0; i < input.length; i += 1) {
    const ch = input[i];
    if (/\s/.test(ch)) {
      if (!lastWasSpace) {
        normalized += ' ';
        map.push(i);
        lastWasSpace = true;
      }
    } else {
      normalized += ch;
      map.push(i);
      lastWasSpace = false;
    }
  }

  let start = 0;
  while (start < normalized.length && normalized[start] === ' ') {
    start += 1;
  }

  let end = normalized.length;
  while (end > start && normalized[end - 1] === ' ') {
    end -= 1;
  }

  return {
    normalized: normalized.slice(start, end),
    map: map.slice(start, end),
  };
};

const buildSnippetSearchKeys = (snippet: string): string[] => {
  const keys = [snippet];
  if (snippet.length > 120) {
    keys.push(snippet.slice(0, 120).trim());
  }
  if (snippet.length > 60) {
    keys.push(snippet.slice(0, 60).trim());
  }
  return Array.from(new Set(keys.filter(Boolean)));
};

const findSnippetRange = (content: string, snippet: string): PreviewHighlightRange | null => {
  for (const searchKey of buildSnippetSearchKeys(snippet)) {
    const directIndex = content.indexOf(searchKey);
    if (directIndex !== -1) {
      return {
        start: directIndex,
        end: Math.min(content.length, directIndex + searchKey.length),
        matchedBy: 'snippet',
      };
    }

    const normalizedKey = normalizeWhitespace(searchKey);
    if (!normalizedKey) {
      continue;
    }

    const { normalized, map } = normalizeWhitespaceForSearch(content);
    const normalizedIndex = normalized.indexOf(normalizedKey);
    if (normalizedIndex === -1) {
      continue;
    }

    const start = map[normalizedIndex] ?? 0;
    const endIndex = normalizedIndex + normalizedKey.length - 1;
    const end = endIndex >= 0 && endIndex < map.length ? map[endIndex] + 1 : start + normalizedKey.length;
    return { start, end: Math.min(content.length, end), matchedBy: 'snippet' };
  }
  return null;
};

export const getPreviewLoadOffset = (charOffset?: number): number => {
  if (typeof charOffset !== 'number' || Number.isNaN(charOffset)) {
    return 0;
  }
  return Math.max(0, charOffset - 160);
};

export const findPreviewHighlightRange = (
  content: string,
  loadedOffset: number,
  scrollTarget?: PreviewScrollTarget,
): PreviewHighlightRange | null => {
  if (!scrollTarget) {
    return null;
  }

  const snippet = sanitizeCitationSnippet(scrollTarget.snippet)?.trim();
  if (snippet) {
    const snippetRange = findSnippetRange(content, snippet);
    if (snippetRange) {
      return snippetRange;
    }
  }

  const relativeOffset = scrollTarget.charOffset - loadedOffset;
  if (relativeOffset >= 0 && relativeOffset < content.length) {
    const lineStart = content.lastIndexOf('\n', relativeOffset) + 1;
    const lineEnd = content.indexOf('\n', relativeOffset);
    return {
      start: lineStart,
      end: lineEnd === -1 ? content.length : lineEnd,
      matchedBy: 'offset',
    };
  }

  return null;
};
