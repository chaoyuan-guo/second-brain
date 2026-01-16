import type {
  ApiMessagePayload,
  ChatMessage,
  ChatSession,
  MessageSegment,
} from './chat-types';

export const createId = () =>
  typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);

export const createEmptySession = (id?: string): ChatSession => ({
  id: id ?? createId(),
  title: '新的对话',
  messages: [],
  createdAt: Date.now(),
  isCustomTitle: false,
});

export const deriveTitle = (text: string) => {
  const sanitized = text.trim().replace(/\s+/g, ' ');
  if (!sanitized) {
    return '新的对话';
  }
  return sanitized.length > 24 ? `${sanitized.slice(0, 24)}...` : sanitized;
};

export const formatTimestamp = (timestamp?: number) => {
  if (!timestamp) {
    return '';
  }
  try {
    return new Intl.DateTimeFormat('zh-CN', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    }).format(timestamp);
  } catch {
    return '';
  }
};

export const summarizeText = (text: string, maxLength = 80) => {
  const sanitized = text.replace(/\s+/g, ' ').trim();
  if (!sanitized) {
    return '';
  }
  return sanitized.length > maxLength ? `${sanitized.slice(0, maxLength)}…` : sanitized;
};

const getSessionTimestampValue = (session?: ChatSession) => {
  if (!session) {
    return undefined;
  }
  return session.messages[session.messages.length - 1]?.timestamp ?? session.createdAt;
};

export const deriveSessionTimestamp = (session?: ChatSession) => {
  const value = getSessionTimestampValue(session);
  return value ? formatTimestamp(value) : '';
};

export const deriveSessionSubtitle = (session?: ChatSession) => {
  if (!session) {
    return '';
  }
  const latestUserContent = [...session.messages]
    .reverse()
    .find((message) => message.role === 'user' && message.content.trim())?.content;
  const fallbackContent = session.messages[session.messages.length - 1]?.content ?? '';
  return summarizeText(latestUserContent ?? fallbackContent, 72);
};

export const serializeMessagesForApi = (session: ChatSession): ApiMessagePayload[] =>
  session.messages
    .filter((message) => !message.isThinking && message.content.trim().length > 0)
    .map((message) => {
      const payload: ApiMessagePayload = {
        role: message.role,
        content: message.content,
      };
      if (message.tool_call_id) {
        payload.tool_call_id = message.tool_call_id;
      }
      return payload;
    });

export const parseMessageSegments = (content: string): MessageSegment[] => {
  if (!content) {
    return [{ type: 'text', content: '' }];
  }
  const segments: MessageSegment[] = [];
  const regex = /```(\w+)?\n?([\s\S]*?)```/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      segments.push({ type: 'text', content: content.slice(lastIndex, match.index) });
    }
    segments.push({
      type: 'code',
      language: match[1]?.trim() || 'code',
      content: match[2]?.replace(/^\n/, '').replace(/\n$/, '') ?? '',
    });
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < content.length) {
    segments.push({ type: 'text', content: content.slice(lastIndex) });
  }

  return segments.length ? segments : [{ type: 'text', content }];
};

export const isStandaloneUrl = (text: string) => {
  try {
    const candidate = text.trim();
    if (!candidate) return false;
    const url = new URL(candidate);
    return Boolean(url.hostname);
  } catch {
    return false;
  }
};

export const formatLinkLabel = (href: string) => {
  try {
    const url = new URL(href);
    return url.hostname.replace(/^www\./, '');
  } catch {
    return href;
  }
};
