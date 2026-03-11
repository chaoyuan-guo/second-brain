import fs from 'fs';
import { readFile } from 'fs/promises';
import path from 'path';

import { getOrCreateRequestId, logProxy } from '../../_lib/upstream';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type NoteContentResponse = {
  content: string;
  source_file: string;
  done: boolean;
  next_offset: number | null;
  total_chars: number;
  offset: number;
  limit_chars: number;
};

const DEFAULT_LIMIT_CHARS = 12000;
const MAX_LIMIT_CHARS = 60000;

const findProjectRoot = (): string => {
  let current = process.cwd();
  while (true) {
    if (fs.existsSync(path.join(current, 'data', 'notes', 'my_markdowns'))) {
      return current;
    }

    const parent = path.dirname(current);
    if (parent === current) {
      return process.cwd();
    }
    current = parent;
  }
};

const PROJECT_ROOT = findProjectRoot();
const NOTES_ROOT = path.resolve(PROJECT_ROOT, 'data', 'notes', 'my_markdowns');

const parsePositiveInt = (value: string | null, fallback: number): number => {
  const parsed = Number.parseInt(value || '', 10);
  if (Number.isNaN(parsed)) {
    return fallback;
  }
  return parsed;
};

const resolveNotePath = (inputPath: string): string => {
  const normalized = inputPath.replace(/\\/g, '/').trim();
  if (!normalized) {
    throw Object.assign(new Error('Missing path'), { status: 400 });
  }

  const candidate = path.isAbsolute(normalized)
    ? path.resolve(normalized)
    : normalized.startsWith('data/notes/my_markdowns/')
      ? path.resolve(PROJECT_ROOT, normalized)
      : path.resolve(NOTES_ROOT, normalized);

  if (candidate !== NOTES_ROOT && !candidate.startsWith(`${NOTES_ROOT}${path.sep}`)) {
    throw Object.assign(new Error('只允许读取 data/notes/my_markdowns/ 下的文件。'), { status: 403 });
  }

  if (!fs.existsSync(candidate)) {
    throw Object.assign(new Error(`文件不存在: ${candidate}`), { status: 404 });
  }

  const stat = fs.statSync(candidate);
  if (stat.isDirectory()) {
    throw Object.assign(new Error(`路径是目录，无法读取: ${candidate}`), { status: 400 });
  }

  return candidate;
};

const buildResponse = (
  sourceFile: string,
  text: string,
  offset: number,
  limitChars: number,
): NoteContentResponse => {
  const totalChars = text.length;
  const start = Math.min(offset, totalChars);
  const end = Math.min(start + limitChars, totalChars);

  return {
    content: text.slice(start, end),
    source_file: sourceFile,
    done: end >= totalChars,
    next_offset: end < totalChars ? end : null,
    total_chars: totalChars,
    offset: start,
    limit_chars: limitChars,
  };
};

export async function GET(request: Request): Promise<Response> {
  const requestId = getOrCreateRequestId(request);
  const route = '/api/notes/content';
  const sourceUrl = new URL(request.url);
  const requestedPath = sourceUrl.searchParams.get('path')?.trim() || '';
  const offset = Math.max(0, parsePositiveInt(sourceUrl.searchParams.get('offset'), 0));
  const requestedLimit = parsePositiveInt(
    sourceUrl.searchParams.get('limit_chars'),
    DEFAULT_LIMIT_CHARS,
  );
  const limitChars = Math.min(
    MAX_LIMIT_CHARS,
    requestedLimit > 0 ? requestedLimit : DEFAULT_LIMIT_CHARS,
  );
  const start = Date.now();

  logProxy('info', {
    route,
    requestId,
    phase: 'start',
    method: 'GET',
    path: requestedPath,
    offset,
    limitChars,
  });

  try {
    const sourceFile = resolveNotePath(requestedPath);
    const text = await readFile(sourceFile, 'utf-8');
    const payload = buildResponse(sourceFile, text, offset, limitChars);

    logProxy('info', {
      route,
      requestId,
      phase: 'done',
      method: 'GET',
      status: 200,
      durationMs: Date.now() - start,
      path: requestedPath,
      returnedChars: payload.content.length,
      done: payload.done,
    });

    return Response.json(payload, {
      status: 200,
      headers: { 'x-request-id': requestId },
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : '读取失败，请稍后重试';
    const status =
      typeof error === 'object' &&
      error !== null &&
      'status' in error &&
      typeof (error as { status?: unknown }).status === 'number'
        ? (error as { status: number }).status
        : 500;

    logProxy('error', {
      route,
      requestId,
      phase: 'error',
      method: 'GET',
      status,
      durationMs: Date.now() - start,
      path: requestedPath,
      error: detail,
    });

    return Response.json({ detail }, { status, headers: { 'x-request-id': requestId } });
  }
}
