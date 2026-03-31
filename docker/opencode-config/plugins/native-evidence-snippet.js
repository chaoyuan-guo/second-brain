const MAX_SNIPPET_CHARS = 240;
const MIN_SNIPPET_CHARS = 80;

const normalizeLineEndings = (value) => value.replace(/\r\n?/g, "\n");

const isFenceLine = (line) => /^```/.test(line.trim());

const isHeadingLine = (line) => /^#{1,6}\s+/.test(line.trim());

const getHeadingLevel = (line) => {
  const match = line.trim().match(/^(#{1,6})\s+/);
  return match ? match[1].length : null;
};

const isDividerLine = (line) => {
  const trimmed = line.trim();
  return trimmed === "-" || /^[-*_]{3,}$/.test(trimmed);
};

const stripHeadingMarkers = (value) => value.replace(/^#{1,6}\s+/, "").trim();

const parseSubmissionHeading = (value) => {
  const normalized = stripHeadingMarkers(value).trim();
  const match = normalized.match(
    /^提交\s*(\d+)\s*[·•]\s*([^·•\n]+?)\s*[·•]\s*(\d{4}-\d{2}-\d{2})\b/i,
  );
  if (!match) {
    return null;
  }
  return {
    id: match[1],
    result: match[2].trim(),
    date: match[3],
    text: normalized,
  };
};

const isConversationMarker = (value) => {
  const normalized = stripHeadingMarkers(value)
    .replace(/^[>*\-\s]+/, "")
    .trim();
  return /^(?:🤖\s*)?assistant[:：]?$/i.test(normalized)
    || /^(?:🧑‍💻\s*)?user[:：]?$/i.test(normalized)
    || /^(?:assistant|user)[:：]?$/i.test(normalized);
};

const isMetaLine = (line) => {
  const trimmed = line.trim();
  if (!trimmed) {
    return true;
  }
  if (isDividerLine(trimmed) || isConversationMarker(trimmed)) {
    return true;
  }
  if (/^when citing facts from this read result/i.test(trimmed)) {
    return true;
  }
  if (/^<(?:\/)?(?:citation_id|path|type|char_offset|instruction|content|heading)>/i.test(trimmed)) {
    return true;
  }
  if (/^(?:citation_id|path|type|char_offset|instruction|heading)\s*:/i.test(trimmed)) {
    return true;
  }
  return false;
};

const buildBlocks = (content) => {
  const normalized = normalizeLineEndings(content);
  const lines = normalized.split("\n");
  const blocks = [];
  let cursor = 0;
  let current = null;
  let inFence = false;

  const flush = () => {
    if (!current) {
      return;
    }
    blocks.push({
      kind: current.kind,
      text: current.lines.join("\n"),
      start: current.start,
      end: current.end,
    });
    current = null;
  };

  lines.forEach((line, index) => {
    const lineStart = cursor;
    const lineEnd = lineStart + line.length;
    cursor = lineEnd + 1;
    const isLastLine = index === lines.length - 1;
    const effectiveEnd = isLastLine ? lineEnd : lineEnd + 1;
    const trimmed = line.trim();

    if (inFence) {
      current.lines.push(line);
      current.end = effectiveEnd;
      if (isFenceLine(line)) {
        inFence = false;
        flush();
      }
      return;
    }

    if (!trimmed) {
      flush();
      return;
    }

    if (isDividerLine(line)) {
      flush();
      return;
    }

    if (isFenceLine(line)) {
      flush();
      current = {
        kind: "code",
        lines: [line],
        start: lineStart,
        end: effectiveEnd,
      };
      inFence = true;
      return;
    }

    if (isHeadingLine(line)) {
      flush();
      blocks.push({
        kind: "heading",
        text: line,
        start: lineStart,
        end: effectiveEnd,
      });
      return;
    }

    if (!current) {
      current = {
        kind: "text",
        lines: [line],
        start: lineStart,
        end: effectiveEnd,
      };
      return;
    }

    current.lines.push(line);
    current.end = effectiveEnd;
  });

  flush();
  return blocks;
};

const annotateBlocks = (blocks) => {
  const headingsByLevel = new Map();

  return blocks.map((block) => {
    if (block.kind === "heading") {
      const level = getHeadingLevel(block.text);
      if (level) {
        for (let current = level; current <= 6; current += 1) {
          headingsByLevel.delete(current);
        }
        headingsByLevel.set(level, stripHeadingMarkers(block.text));
      }
    }

    return {
      ...block,
      context: {
        h2: headingsByLevel.get(2),
        h3: headingsByLevel.get(3),
        h4: headingsByLevel.get(4),
      },
    };
  });
};

const trimBlockForDisplay = (value) => {
  const normalized = normalizeLineEndings(value).trim();
  if (!normalized) {
    return "";
  }

  const lines = normalized
    .split("\n")
    .map((line) => line.replace(/\s+$/g, ""))
    .filter((line) => !isMetaLine(line));

  return lines.join("\n").trim();
};

const isNoiseBlock = (block) => {
  if (block.kind === "heading") {
    return true;
  }
  const cleaned = trimBlockForDisplay(block.text);
  return !cleaned;
};

const looksLikeCodeishBlock = (value) => {
  const trimmed = value.trim();
  if (!trimmed) {
    return false;
  }
  if (/^```/.test(trimmed) || /```$/.test(trimmed)) {
    return true;
  }
  const lines = trimmed.split("\n").filter(Boolean);
  return lines.some((line) =>
    /^\s*(?:class |def |for |while |if |elif |return\b|const |let |function\b|public |private )/.test(line),
  );
};

const looksLikeTableBlock = (value) => {
  const lines = value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.length > 0 && lines.every((line) => line.startsWith("|") || /^\|?\s*[-:| ]+\|?\s*$/.test(line));
};

const looksLikeMetadataList = (value) => {
  const lines = value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.length > 0 && lines.every((line) => {
    if (!/^[-*]\s+/.test(line)) {
      return false;
    }
    return /[：:]/.test(line);
  });
};

const isTruncatedCodeBlock = (block) => {
  if (!block || block.kind !== "code") {
    return false;
  }
  return !normalizeLineEndings(block.text).trimEnd().endsWith("```");
};

const buildSubmissionCandidates = (blocks) => {
  const candidates = [];

  for (let index = 0; index < blocks.length; index += 1) {
    const block = blocks[index];
    if (block.kind !== "heading") {
      continue;
    }

    const submission = parseSubmissionHeading(block.text);
    if (!submission) {
      continue;
    }

    let body = null;
    for (let nextIndex = index + 1; nextIndex < blocks.length; nextIndex += 1) {
      const nextBlock = blocks[nextIndex];
      if (nextBlock.kind === "heading") {
        break;
      }
      if (isNoiseBlock(nextBlock)) {
        continue;
      }
      body = {
        index: nextIndex,
        block: nextBlock,
        cleaned: trimBlockForDisplay(nextBlock.text),
      };
      break;
    }

    if (!body?.cleaned) {
      continue;
    }

    candidates.push({
      id: `submission:${submission.id}:${block.start}`,
      type: "submission",
      submission,
      index,
      bodyIndex: body.index,
      cleaned: `${block.text.trim()}\n\n${body.cleaned}`.trim(),
      start: block.start,
      end: body.block.end,
      context: block.context,
      isTruncated: isTruncatedCodeBlock(body.block),
    });
  }

  return candidates;
};

const findCrossDateSubmissionSection = (content, currentSubmission) => {
  if (!currentSubmission) {
    return null;
  }

  const matches = Array.from(
    normalizeLineEndings(content).matchAll(
      /^####\s+提交\s*(\d+)\s*[·•]\s*([^·•\n]+?)\s*[·•]\s*(\d{4}-\d{2}-\d{2})[^\n]*$/gm,
    ),
  );

  const candidates = matches
    .map((match, index) => {
      const start = match.index ?? 0;
      const end = index + 1 < matches.length ? (matches[index + 1].index ?? content.length) : content.length;
      const text = content.slice(start, end).trim();
      const fenceCount = (text.match(/^```/gm) || []).length;
      return {
        start,
        date: match[3],
        text,
        isComplete: fenceCount >= 2,
      };
    })
    .filter((candidate) => candidate.isComplete && candidate.date !== currentSubmission.date);

  return candidates.at(-1) || null;
};

const getCandidateScore = (candidate, cleaned, index, options = {}) => {
  let score = 0;
  const block = candidate.block;
  const context = candidate.context || block?.context || {};
  const text = candidate.text || block?.text || "";
  const currentSubmission = parseSubmissionHeading(options.currentHeading || "");
  const allowLeadingPartialBoost = !options.preferredSubmissionDate
    || !currentSubmission
    || currentSubmission.date === options.preferredSubmissionDate;

  if (candidate.type === "submission") {
    score += 22;
  }
  if (block?.kind === "code" || /```/.test(text)) {
    score += 8;
  }
  if (/提交\s*\d+/.test(cleaned) || /Runtime Error|Wrong Answer|Accepted/.test(cleaned)) {
    score += 6;
  }
  if (/path\.pop|backtrack|return\b|dp\[|queue|deque|minutes|fresh|grid\[|travel_days|cost_/.test(cleaned)) {
    score += 4;
  }
  if (/[。！？.!?]/.test(cleaned)) {
    score += 1;
  }
  if (cleaned.length >= MIN_SNIPPET_CHARS) {
    score += 1;
  }
  if (context.h3 === "未通过提交代码") {
    score += 10;
  }
  if (context.h3 === "题目笔记") {
    score -= 8;
  }
  if (/^笔记\b/.test(context.h4 || "")) {
    score -= 4;
  }
  if (cleaned.length < 24) {
    score -= 3;
  }
  if (looksLikeTableBlock(cleaned)) {
    score -= 6;
  }
  if (looksLikeMetadataList(cleaned)) {
    score -= 4;
  }
  if (/^(?:统计时间范围|总提交数|题目链接|难度|标签|总提交次数|最近提交时间)/m.test(cleaned)) {
    score -= 4;
  }
  if (/^\d{1,2}:\d{2}:\d{2}\b/.test(cleaned)) {
    score -= 3;
  }
  if (options.startedMidLine && index === 0 && allowLeadingPartialBoost && looksLikeCodeishBlock(cleaned)) {
    score += 10;
  }
  if (options.startedMidLine && index === 0 && allowLeadingPartialBoost && currentSubmission) {
    score += 6;
  }
  if (
    options.startedMidLine
    && index === 0
    && options.preferredSubmissionDate
    && currentSubmission
    && currentSubmission.date !== options.preferredSubmissionDate
  ) {
    score -= 20;
  }
  if (options.preferredCandidateId && candidate.id === options.preferredCandidateId) {
    score += 24;
  }
  if (candidate.type === "submission" && candidate.isTruncated) {
    score -= 4;
  }

  return score - index * 0.2;
};

const truncateSnippet = (value) => {
  if (value.length <= MAX_SNIPPET_CHARS) {
    return value;
  }

  const searchStart = Math.min(MIN_SNIPPET_CHARS, MAX_SNIPPET_CHARS);
  let cut = -1;
  for (let i = Math.min(value.length, MAX_SNIPPET_CHARS) - 1; i >= searchStart; i -= 1) {
    const ch = value[i];
    if (ch === "\n" || ch === "。" || ch === "！" || ch === "？" || ch === "；" || ch === "." || ch === "!" || ch === "?") {
      cut = i + 1;
      break;
    }
  }

  if (cut === -1) {
    for (let i = Math.min(value.length, MAX_SNIPPET_CHARS) - 1; i >= searchStart; i -= 1) {
      if (/\s/.test(value[i])) {
        cut = i;
        break;
      }
    }
  }

  if (cut === -1) {
    cut = MAX_SNIPPET_CHARS;
  }

  return value.slice(0, cut).trim();
};

const findSnippetOffset = (block, cleanedBlock, snippet) => {
  const source = normalizeLineEndings(block.text);
  const cleanedIndex = source.indexOf(cleanedBlock);
  const baseIndex = cleanedIndex >= 0 ? cleanedIndex : source.search(/\S/);
  const snippetIndex = source.indexOf(snippet, Math.max(0, baseIndex));
  if (snippetIndex >= 0) {
    return block.start + snippetIndex;
  }
  return block.start + Math.max(0, baseIndex);
};

export const extractEvidenceAnchor = (content, baseOffset = 0, options = {}) => {
  const normalized = normalizeLineEndings(content);
  if (!normalized.trim()) {
    return {
      snippet: "",
      charOffset: Math.max(0, Math.floor(baseOffset)),
      localOffset: 0,
    };
  }

  const blocks = annotateBlocks(buildBlocks(normalized));
  const submissionCandidates = buildSubmissionCandidates(blocks);
  const currentSubmission = parseSubmissionHeading(options.currentHeading || "");
  const crossDateSubmission = options.startedMidLine
    ? findCrossDateSubmissionSection(normalized, currentSubmission)
    : null;

  if (crossDateSubmission) {
    const snippet = truncateSnippet(trimBlockForDisplay(crossDateSubmission.text));
    return {
      snippet,
      charOffset: Math.max(0, Math.floor(baseOffset)) + crossDateSubmission.start,
      localOffset: crossDateSubmission.start,
    };
  }

  if (
    currentSubmission
    && Number.isFinite(options.currentHeadingDistance)
    && options.currentHeadingDistance >= 0
    && options.currentHeadingDistance <= 700
  ) {
    const leadingBlock = blocks.find((block) => !isNoiseBlock(block));
    const cleanedLeadingBlock = leadingBlock ? trimBlockForDisplay(leadingBlock.text) : "";
    if (leadingBlock && cleanedLeadingBlock && looksLikeCodeishBlock(cleanedLeadingBlock)) {
      const snippet = truncateSnippet(cleanedLeadingBlock);
      const localOffset = findSnippetOffset(leadingBlock, cleanedLeadingBlock, snippet);
      return {
        snippet,
        charOffset: Math.max(0, Math.floor(baseOffset)) + localOffset,
        localOffset,
      };
    }
  }

  const preferredSubmission = options.startedMidLine && submissionCandidates.length >= 2
    ? submissionCandidates.filter((candidate) => !candidate.isTruncated).at(-1)
    : null;
  const preferredCandidateId = preferredSubmission?.id || null;
  const preferredSubmissionDate = preferredSubmission?.submission?.date || null;

  const blockCandidates = blocks
    .map((block, index) => {
      if (isNoiseBlock(block)) {
        return null;
      }
      const cleaned = trimBlockForDisplay(block.text);
      const candidate = {
        id: `block:${index}`,
        type: "block",
        block,
        text: block.text,
        context: block.context,
        cleaned,
        index,
      };
      return {
        ...candidate,
        score: getCandidateScore(candidate, cleaned, index, {
          ...options,
          preferredCandidateId,
          preferredSubmissionDate,
        }),
      };
    })
    .filter(Boolean);

  const syntheticSubmissionCandidates = submissionCandidates.map((candidate) => {
    const syntheticBlock = {
      kind: "synthetic",
      text: candidate.cleaned,
      start: candidate.start,
      end: candidate.end,
      context: candidate.context,
    };
    const scoredCandidate = {
      ...candidate,
      block: syntheticBlock,
      text: candidate.cleaned,
      cleaned: candidate.cleaned,
    };
    return {
      ...scoredCandidate,
      score: getCandidateScore(scoredCandidate, candidate.cleaned, candidate.index, {
        ...options,
        preferredCandidateId,
        preferredSubmissionDate,
      }),
    };
  });

  const candidates = blockCandidates.concat(syntheticSubmissionCandidates);

  if (candidates.length === 0) {
    const fallback = truncateSnippet(normalized.trim());
    return {
      snippet: fallback,
      charOffset: Math.max(0, Math.floor(baseOffset)),
      localOffset: 0,
    };
  }

  const bestCandidate = options.startedMidLine
    && currentSubmission
    && preferredSubmission
    && currentSubmission.date !== preferredSubmissionDate
    ? syntheticSubmissionCandidates.find((candidate) => candidate.id === preferredCandidateId)
    : candidates.reduce((best, current) => {
        if (!best) {
          return current;
        }
        if (current.score > best.score) {
          return current;
        }
        if (current.score === best.score && current.index < best.index) {
          return current;
        }
        return best;
      }, null);

  const selected = [bestCandidate];
  let totalLength = bestCandidate.cleaned.length;
  for (let i = bestCandidate.index + 1; i < blocks.length; i += 1) {
    const block = blocks[i];
    if (block.kind === "heading") {
      break;
    }
    if (isNoiseBlock(block)) {
      continue;
    }

    const cleaned = trimBlockForDisplay(block.text);
    const candidate = {
      id: `block:${i}`,
      type: "block",
      block,
      text: block.text,
      context: block.context,
    };
    const score = getCandidateScore(candidate, cleaned, i, {
      ...options,
      preferredCandidateId,
      preferredSubmissionDate,
    });
    if (score < 0 && totalLength >= MIN_SNIPPET_CHARS) {
      break;
    }
    selected.push({ ...candidate, cleaned, index: i, score });
    totalLength += cleaned.length;
    if (totalLength >= MIN_SNIPPET_CHARS || totalLength >= MAX_SNIPPET_CHARS) {
      break;
    }
  }

  const combined = truncateSnippet(selected.map((entry) => entry.cleaned).join("\n\n"));
  const first = selected[0];
  const localOffset = findSnippetOffset(first.block, first.cleaned, combined);
  return {
    snippet: combined,
    charOffset: Math.max(0, Math.floor(baseOffset)) + localOffset,
    localOffset,
  };
};
