import type {
  ChatSession,
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
