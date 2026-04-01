import type {
  ChatSession,
  ChatMessage,
  DisplayState,
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

export const shouldIgnoreComposerSubmitAfterAbort = (
  lastAbortAt: number,
  now = Date.now(),
  cooldownMs = 400,
) => {
  if (!lastAbortAt) {
    return false;
  }
  return now - lastAbortAt < cooldownMs;
};

export const getUserFacingAssistantStatusText = (options: {
  assistantContent: string;
  displayState: DisplayState;
  forceLongRunning?: boolean;
  isToolWork?: boolean;
}) => {
  const { assistantContent, displayState, forceLongRunning = false, isToolWork = false } = options;
  const hasContent = assistantContent.trim().length > 0;
  if (forceLongRunning || displayState === 'long_running') {
    return hasContent ? '正在继续整理依据' : '正在整理依据';
  }
  if (isToolWork) {
    return '正在整理依据';
  }
  if (!hasContent) {
    return '正在整理回答';
  }
  return '';
};

export const hasRenderableAssistantAnswer = (
  message: Pick<ChatMessage, 'role' | 'content' | 'directAnswer' | 'fullAnalysis' | 'completionState'>,
) =>
  message.role === 'assistant'
  && Boolean(
    message.content.trim()
    || message.directAnswer?.trim()
    || message.fullAnalysis?.trim()
    || message.completionState === 'partial_completed'
    || message.completionState === 'failed'
  );

export const sanitizeFailureReason = (failureReason?: string) => {
  const value = failureReason?.trim();
  if (!value) {
    return '请重试或缩小问题范围';
  }
  if (/步骤上限|tool|工具失败|调用工具|prompt_async|session/i.test(value)) {
    return '当前未能形成可靠结论，请重试或缩小问题范围';
  }
  return value;
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
