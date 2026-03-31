import { readFile } from "node:fs/promises";
import path from "node:path";
import { tool } from "@opencode-ai/plugin";
import { extractEvidenceAnchor } from "./native-evidence-snippet.js";

const sessionState = new Map();

const getSessionState = (sessionID) => {
  let state = sessionState.get(sessionID);
  if (!state) {
    state = {
      nextCitationIndex: 1,
      citationByKey: new Map(),
    };
    sessionState.set(sessionID, state);
  }
  return state;
};

const formatCitationId = (index) => `c${String(index).padStart(2, "0")}`;

const assignCitationId = (sessionID, evidenceKey) => {
  const state = getSessionState(sessionID);
  const existing = state.citationByKey.get(evidenceKey);
  if (existing) {
    return existing;
  }
  const next = formatCitationId(state.nextCitationIndex);
  state.nextCitationIndex += 1;
  state.citationByKey.set(evidenceKey, next);
  return next;
};

const clampOffset = (value) => {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.floor(value));
};

const clampLimit = (value) => {
  if (!Number.isFinite(value)) {
    return 1200;
  }
  return Math.max(1, Math.min(8000, Math.floor(value)));
};

const resolveFilePath = (filePath, directory) => {
  if (path.isAbsolute(filePath)) {
    return path.normalize(filePath);
  }
  return path.normalize(path.resolve(directory, filePath));
};

const escapeXml = (value) =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");

const detectHeadingInfo = (content, offset) => {
  const prefix = content.slice(0, Math.min(offset, content.length));
  const matches = Array.from(prefix.matchAll(/^#{1,6}\s+(.+)$/gm));
  const last = matches.at(-1);
  const heading = last?.[1]?.trim();
  if (!heading) {
    return undefined;
  }
  return {
    text: heading,
    offset: Number.isFinite(last?.index) ? last.index : undefined,
  };
};

const deriveSourceTitle = (resolvedPath) => path.basename(resolvedPath).replace(/\.[^.]+$/, "");

const extractTagValue = (output, tagName) => {
  const match = output.match(new RegExp(`<${tagName}>([\\s\\S]*?)</${tagName}>`));
  return match?.[1]?.trim() || "";
};

const decodeXml = (value) =>
  value
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&apos;", "'")
    .replaceAll("&amp;", "&");

const buildCompletedMetadata = ({ output, existingMetadata }) => {
  const citationId = decodeXml(extractTagValue(output, "citation_id"));
  const resolvedPath = decodeXml(extractTagValue(output, "path"));
  const heading = decodeXml(extractTagValue(output, "heading"));
  const content = decodeXml(extractTagValue(output, "content"));
  const charOffset = Number.parseInt(extractTagValue(output, "char_offset"), 10);
  const existingSourceRef = Array.isArray(existingMetadata?.source_refs)
    ? existingMetadata.source_refs[0]
    : undefined;
  const existingLoaded = Array.isArray(existingMetadata?.loaded)
    ? existingMetadata.loaded
    : undefined;

  const anchor = extractEvidenceAnchor(content, 0);
  const snippet = typeof existingSourceRef?.snippet === "string"
    ? existingSourceRef.snippet
    : anchor.snippet;
  const sourceCharOffset = Number.isFinite(existingSourceRef?.char_offset)
    ? existingSourceRef.char_offset
    : Number.isFinite(charOffset)
      ? charOffset
      : anchor.charOffset;
  const loadedOffset = Number.isFinite(existingLoaded?.[0]?.offset)
    ? existingLoaded[0].offset
    : sourceCharOffset;

  if (!citationId || !resolvedPath || !Number.isFinite(charOffset)) {
    return existingMetadata || {};
  }

  const merged = {
    ...(existingMetadata || {}),
    preview: content.slice(0, 240),
    loaded: existingLoaded?.length
      ? existingLoaded
      : [
          {
            path: resolvedPath,
            offset: loadedOffset,
            limit: content.length,
            citation_id: citationId,
          },
        ],
    source_refs: Array.isArray(existingMetadata?.source_refs) && existingMetadata.source_refs.length > 0
      ? existingMetadata.source_refs
      : [
          {
            path: resolvedPath,
            citation_id: citationId,
            snippet,
            char_offset: sourceCharOffset,
            heading: heading || undefined,
            source_title: deriveSourceTitle(resolvedPath),
          },
        ],
  };

  if (typeof merged.truncated !== "boolean") {
    merged.truncated = false;
  }

  return merged;
};

const buildReadOutput = ({ citationId, resolvedPath, charOffset, heading, content }) => {
  const parts = [
    `<citation_id>${escapeXml(citationId)}</citation_id>`,
    `<path>${escapeXml(resolvedPath)}</path>`,
    "<type>file</type>",
    `<char_offset>${charOffset}</char_offset>`,
  ];
  if (heading) {
    parts.push(`<heading>${escapeXml(heading)}</heading>`);
  }
  parts.push(
    "<instruction>When citing facts from this read result, use the exact citation_id above and do not renumber it.</instruction>",
    `<content>${escapeXml(content)}</content>`,
  );
  return parts.join("\n");
};

export const NativeEvidenceReadPlugin = async () => {
  return {
    "tool.execute.after": async (input, output) => {
      if (input.tool !== "read" || typeof output.output !== "string") {
        return;
      }
      const completedMetadata = buildCompletedMetadata({
        output: output.output,
        existingMetadata: output.metadata,
      });
      output.metadata = completedMetadata;
      const resolvedPath = extractTagValue(output.output, "path");
      const citationId = extractTagValue(output.output, "citation_id");
      if (resolvedPath && citationId) {
        output.title = `read ${path.basename(resolvedPath)} [${citationId}]`;
      }
    },
    tool: {
      read: tool({
        description:
          "Read file contents from the project or an absolute path. Each successful read returns a stable citation_id; when citing facts from this read result, use that exact [cxx] id and do not renumber it.",
        args: {
          filePath: tool.schema.string().describe("Absolute or project-relative file path to read."),
          offset: tool.schema.number().int().nonnegative().optional().describe("Character offset to start reading from."),
          limit: tool.schema.number().int().positive().optional().describe("Maximum number of characters to read."),
        },
        async execute(args, context) {
          const resolvedPath = resolveFilePath(args.filePath, context.directory);
          const charOffset = clampOffset(args.offset);
          const limit = clampLimit(args.limit);
          const raw = await readFile(resolvedPath, "utf8");
          const content = raw.slice(charOffset, charOffset + limit);
          const headingInfo = detectHeadingInfo(raw, charOffset);
          const anchor = extractEvidenceAnchor(content, charOffset, {
            startedMidLine: charOffset > 0 && raw[charOffset - 1] !== "\n",
            currentHeading: headingInfo?.text,
            currentHeadingDistance: Number.isFinite(headingInfo?.offset) ? charOffset - headingInfo.offset : undefined,
          });
          const snippet = anchor.snippet;
          const sourceCharOffset = Number.isFinite(anchor.charOffset) ? anchor.charOffset : charOffset;
          const heading = detectHeadingInfo(raw, sourceCharOffset)?.text;
          const evidenceKey = `${resolvedPath}::${charOffset}`;
          const citationId = assignCitationId(context.sessionID, evidenceKey);

          context.metadata({
            title: `read ${path.basename(resolvedPath)} [${citationId}]`,
            metadata: {
              preview: content.slice(0, 240),
              truncated: charOffset + limit < raw.length,
              loaded: [
                {
                  path: resolvedPath,
                  offset: charOffset,
                  limit,
                  citation_id: citationId,
                },
              ],
              source_refs: [
                {
                  path: resolvedPath,
                  citation_id: citationId,
                  snippet,
                  char_offset: sourceCharOffset,
                  heading,
                  source_title: deriveSourceTitle(resolvedPath),
                },
              ],
            },
          });

          return buildReadOutput({
            citationId,
            resolvedPath,
            charOffset: sourceCharOffset,
            heading,
            content,
          });
        },
      }),
    },
  };
};
