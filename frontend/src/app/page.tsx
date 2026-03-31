"use client";

import {
  ChangeEvent,
  KeyboardEvent as ReactKeyboardEvent,
  isValidElement,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import './styles/thinking-timeline.css';

import { ThinkingTimeline } from './components/ThinkingTimeline';
import {
  BotIcon,
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CloseIcon,
  CopyIcon,
  PencilIcon,
  SendIcon,
  SparklesIcon,
  StopIcon,
  TrashIcon,
  UploadIcon,
  UserIcon,
} from './components/icons';
import { AnswerPanel, ProcessPanel } from './components/chat';
import { useChatSessions } from './hooks/useChatSessions';
import {
  deriveSessionTimestamp,
  formatTimestamp,
} from './lib/chat-helpers';
import {
  getApiBaseUrl,
  NOTE_CONTENT_ENDPOINT,
  UPLOAD_ENDPOINT,
  type ChatSession,
  type SourceRef,
} from './lib/chat-types';
import {
  findPreviewHighlightRange,
  getPreviewLoadOffset,
  type PreviewScrollTarget,
} from './lib/preview-utils';

// Feature Flag: Answer-first 重构开关（默认开启，可通过环境变量关闭）
const parseFeatureFlag = (value: string | undefined, defaultValue = true): boolean => {
  if (value === undefined) {
    return defaultValue;
  }
  const normalized = value.trim().toLowerCase();
  if (['0', 'false', 'off', 'no'].includes(normalized)) {
    return false;
  }
  if (['1', 'true', 'on', 'yes'].includes(normalized)) {
    return true;
  }
  return defaultValue;
};

const FEATURE_ANSWER_FIRST_REDESIGN = parseFeatureFlag(
  process.env.NEXT_PUBLIC_FEATURE_ANSWER_FIRST
    ?? process.env.NEXT_PUBLIC_FEATURE_ANSWER_FIRST_REDESIGN,
  true,
);

type NoteContentResponse = {
  content: string;
  source_file?: string;
  done?: boolean;
  next_offset?: number | null;
  total_chars?: number;
  offset?: number;
  limit_chars?: number;
};

type PreviewCacheEntry = {
  content: string;
  done: boolean;
  nextOffset: number | null;
  totalChars: number;
  loadedOffset: number;
};

type PreviewState = PreviewCacheEntry & {
  path: string;
  title: string;
  scrollTarget?: PreviewScrollTarget;
};

type SourceGroup = {
  path: string;
  fileName: string;
  headingRefs: Map<string, SourceRef>;
};

const formatSourcePath = (path: string): string => {
  const normalized = path.replace(/\\/g, '/');
  const absoluteMarker = '/data/notes/my_markdowns/';
  const relativeMarker = 'data/notes/my_markdowns/';
  if (normalized.includes(absoluteMarker)) {
    return normalized.split(absoluteMarker).pop() ?? path;
  }
  if (normalized.startsWith(relativeMarker)) {
    return normalized.slice(relativeMarker.length);
  }
  return path;
};

const PreviewContent = ({
  content,
  loadedOffset,
  scrollTarget,
}: {
  content: string;
  loadedOffset: number;
  scrollTarget?: PreviewScrollTarget;
}) => {
  const highlightRef = useRef<HTMLElement | null>(null);

  const highlightRange = useMemo(() => {
    return findPreviewHighlightRange(content, loadedOffset, scrollTarget);
  }, [content, loadedOffset, scrollTarget]);

  useEffect(() => {
    if (highlightRef.current) {
      highlightRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [highlightRange]);

  if (!highlightRange) {
    return <pre className="preview-content">{content || '（空文件）'}</pre>;
  }

  const before = content.slice(0, highlightRange.start);
  const highlighted = content.slice(highlightRange.start, highlightRange.end);
  const after = content.slice(highlightRange.end);

  return (
    <pre className="preview-content">
      {before}
      <mark className="preview-highlight" ref={highlightRef}>
        {highlighted}
      </mark>
      {after}
    </pre>
  );
};

const MarkdownMessage = ({
  content,
  messageId,
  copiedKey,
  onCopyCode,
}: {
  content: string;
  messageId: string;
  copiedKey: string | null;
  onCopyCode: (value: string, key: string) => void;
}) => {
  let codeBlockIndex = 0;

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ href, children, ...props }) => {
          const normalizedHref = typeof href === 'string' ? href : '';
          if (!normalizedHref) {
            return <span>{children}</span>;
          }
          return (
            <a
              {...props}
              className="inline-link"
              href={normalizedHref}
              target="_blank"
              rel="noopener noreferrer"
            >
              {children}
            </a>
          );
        },
        pre: ({ children }) => {
          const firstChild = Array.isArray(children) ? children[0] : children;
          if (!isValidElement(firstChild)) {
            return <pre>{children}</pre>;
          }

          const codeProps = firstChild.props as { className?: string; children?: unknown };
          const className = codeProps.className ?? '';
          const codeText = String(codeProps.children ?? '').replace(/\n$/, '');
          const languageMatch = /language-([\w-]+)/.exec(className);
          const language = languageMatch?.[1] ?? 'code';
          const codeKey = `${messageId}-code-${codeBlockIndex}`;
          codeBlockIndex += 1;

          return (
            <div className="code-block">
              <div className="code-header">
                <span>{language}</span>
                <button type="button" onClick={() => onCopyCode(codeText, codeKey)} aria-label="复制代码">
                  {copiedKey === codeKey ? '已复制' : '复制'}
                </button>
              </div>
              <pre>
                <code className={className}>{codeText}</code>
              </pre>
            </div>
          );
        },
        code: ({ className, children, ...props }) => {
          return (
            <code className={className} {...props}>
              {children}
            </code>
          );
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
};

export default function HomePage() {
  const {
    sessions,
    activeSession,
    activeSessionId,
    setActiveSessionId,
    inputValue,
    setInputValue,
    hydrated,
    isActivePending,
    createNewSession,
    deleteSession,
    renameSession,
    handleSubmit,
    abortSessionRequest,
  } = useChatSessions();

  const [renamingSessionId, setRenamingSessionId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState('');
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<{
    tone: 'success' | 'error';
    message: string;
  } | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [previewState, setPreviewState] = useState<PreviewState | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});
  const historyEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const copyTimerRef = useRef<number | null>(null);
  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const uploadTimerRef = useRef<number | null>(null);
  const previewCacheRef = useRef<Map<string, PreviewCacheEntry>>(new Map());
  const previewPathRef = useRef<string | null>(null);

  const messages = activeSession?.messages ?? [];
  const hasContent = messages.length > 0;
  const isActivePendingFlag = isActivePending;

  useEffect(() => {
    historyEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, activeSessionId]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 240)}px`;
  }, [inputValue]);

  useEffect(
    () => () => {
      if (copyTimerRef.current) {
        window.clearTimeout(copyTimerRef.current);
      }
      if (uploadTimerRef.current) {
        window.clearTimeout(uploadTimerRef.current);
      }
    },
    [],
  );

  const beginRename = (session: ChatSession) => {
    setRenamingSessionId(session.id);
    setRenameDraft(session.title);
  };

  const cancelRename = () => {
    setRenamingSessionId(null);
    setRenameDraft('');
  };

  const commitRename = (sessionId: string, value?: string) => {
    const finalValue = value ?? renameDraft;
    renameSession(sessionId, finalValue);
    cancelRename();
  };

  const handleRenameKey = (event: ReactKeyboardEvent<HTMLInputElement>, sessionId: string) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      commitRename(sessionId);
    } else if (event.key === 'Escape') {
      event.preventDefault();
      cancelRename();
    }
  };

  const confirmDeleteSession = (sessionId: string) => {
    if (typeof window === 'undefined' || window.confirm('确定删除该会话吗？')) {
      deleteSession(sessionId);
    }
  };


  const handleCopy = useCallback((text: string, key: string) => {
    if (!text) return;
    navigator.clipboard
      ?.writeText(text)
      .then(() => {
        setCopiedKey(key);
        if (copyTimerRef.current) {
          window.clearTimeout(copyTimerRef.current);
        }
        copyTimerRef.current = window.setTimeout(() => setCopiedKey(null), 2000);
      })
      .catch((error) => {
        console.error('复制失败', error);
      });
  }, []);

  const showUploadStatus = useCallback((tone: 'success' | 'error', message: string) => {
    setUploadStatus({ tone, message });
    if (uploadTimerRef.current) {
      window.clearTimeout(uploadTimerRef.current);
    }
    const timeout = tone === 'error' ? 8000 : 4500;
    uploadTimerRef.current = window.setTimeout(() => setUploadStatus(null), timeout);
  }, []);

  const getPreviewErrorMessage = useCallback((error: unknown) => {
    if (error instanceof TypeError) {
      return '网络异常，请检查连接后重试';
    }
    if (error instanceof Error) {
      return error.message;
    }
    return '读取失败，请稍后重试';
  }, []);

  const closePreview = useCallback(() => {
    previewPathRef.current = null;
    setPreviewState(null);
    setPreviewError(null);
    setPreviewLoading(false);
  }, []);

  const normalizePreviewEntry = useCallback(
    (data: NoteContentResponse, fallbackOffset = 0): PreviewCacheEntry => {
      const content = typeof data.content === 'string' ? data.content : '';
      const totalChars = typeof data.total_chars === 'number' ? data.total_chars : content.length;
      const nextOffset = typeof data.next_offset === 'number' ? data.next_offset : null;
      const done = typeof data.done === 'boolean' ? data.done : nextOffset === null;
      const loadedOffset = typeof data.offset === 'number' ? data.offset : fallbackOffset;
      return { content, done, nextOffset, totalChars, loadedOffset };
    },
    [],
  );

  const fetchNoteContent = useCallback(async (filePath: string, offset = 0) => {
    const url = `${getApiBaseUrl()}${NOTE_CONTENT_ENDPOINT}?path=${encodeURIComponent(
      filePath,
    )}&offset=${offset}`;
    const response = await fetch(url);
    if (!response.ok) {
      const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
      const detail = payload?.detail || `读取失败: ${response.status}`;
      throw new Error(detail);
    }
    return (await response.json()) as NoteContentResponse;
  }, []);

  const handleOpenPreview = useCallback(
    async (filePath: string, title: string, ref?: SourceRef) => {
      previewPathRef.current = filePath;
      setPreviewError(null);

      const hasOffset = typeof ref?.char_offset === 'number' && !Number.isNaN(ref.char_offset);
      const scrollTarget = hasOffset
        ? { charOffset: ref?.char_offset ?? 0, snippet: ref?.snippet }
        : undefined;
      const loadOffset = hasOffset ? getPreviewLoadOffset(ref?.char_offset) : 0;

      const cached = previewCacheRef.current.get(filePath);
      if (cached) {
        const cacheStart = cached.loadedOffset;
        const cacheEnd = cached.loadedOffset + cached.content.length;
        if (loadOffset >= cacheStart && loadOffset < cacheEnd) {
          setPreviewState({ path: filePath, title, ...cached, scrollTarget });
          setPreviewLoading(false);
          return;
        }
      }

      setPreviewState({
        path: filePath,
        title,
        content: '',
        done: false,
        nextOffset: null,
        totalChars: 0,
        loadedOffset: loadOffset,
        scrollTarget,
      });
      setPreviewLoading(true);
      try {
        const data = await fetchNoteContent(filePath, loadOffset);
        if (previewPathRef.current !== filePath) {
          return;
        }
        const entry = normalizePreviewEntry(data, loadOffset);
        setPreviewState({ path: filePath, title, ...entry, scrollTarget });
        previewCacheRef.current.set(filePath, entry);
      } catch (error) {
        if (previewPathRef.current === filePath) {
          setPreviewError(getPreviewErrorMessage(error));
        }
      } finally {
        if (previewPathRef.current === filePath) {
          setPreviewLoading(false);
        }
      }
    },
    [fetchNoteContent, getPreviewErrorMessage, normalizePreviewEntry],
  );

  const handleLoadMorePreview = useCallback(async () => {
    if (!previewState || previewLoading || previewState.done || previewState.nextOffset === null) {
      return;
    }
    const currentPath = previewState.path;
    const currentNextOffset = previewState.nextOffset;
    setPreviewError(null);
    setPreviewLoading(true);
    try {
      const data = await fetchNoteContent(currentPath, currentNextOffset);
      const entry = normalizePreviewEntry(data, currentNextOffset);
      setPreviewState((prev) => {
        if (!prev || prev.path !== currentPath) {
          return prev;
        }
        const merged: PreviewCacheEntry = {
          content: `${prev.content}${entry.content}`,
          done: entry.done,
          nextOffset: entry.nextOffset,
          totalChars: entry.totalChars || prev.totalChars,
          loadedOffset: prev.loadedOffset,
        };
        previewCacheRef.current.set(currentPath, merged);
        return { ...prev, ...merged };
      });
    } catch (error) {
      if (previewPathRef.current === currentPath) {
        setPreviewError(getPreviewErrorMessage(error));
      }
    } finally {
      if (previewPathRef.current === currentPath) {
        setPreviewLoading(false);
      }
    }
  }, [fetchNoteContent, getPreviewErrorMessage, normalizePreviewEntry, previewLoading, previewState]);

  const toggleGroup = useCallback((key: string) => {
    setExpandedGroups((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  useEffect(() => {
    if (!previewState) {
      return;
    }
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') {
        closePreview();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [closePreview, previewState]);

  useEffect(() => {
    setExpandedGroups({});
  }, [activeSessionId]);

  const handleUploadClick = () => {
    uploadInputRef.current?.click();
  };

  const handleUploadChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.md')) {
      showUploadStatus('error', '仅支持上传 .md 文件');
      event.target.value = '';
      return;
    }

    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${getApiBaseUrl()}${UPLOAD_ENDPOINT}`, {
        method: 'POST',
        body: formData,
      });
      const payload = (await response.json().catch(() => null)) as
        | {
            message?: string;
            detail?: string;
            file_name?: string;
            chunks_added?: number;
            removed_vectors?: number;
          }
        | null;
      if (!response.ok) {
        throw new Error(payload?.detail || `上传失败: ${response.status}`);
      }
      const fileLabel = payload?.file_name ?? file.name;
      const chunkLabel =
        typeof payload?.chunks_added === 'number' ? `，新增 ${payload.chunks_added} chunks` : '';
      const removedLabel =
        typeof payload?.removed_vectors === 'number' && payload.removed_vectors > 0
          ? `，移除 ${payload.removed_vectors} 条旧向量`
          : '';
      const message = `上传成功：${fileLabel}${chunkLabel}${removedLabel}`;
      previewCacheRef.current.clear();
      showUploadStatus('success', message);
    } catch (error) {
      const message = error instanceof Error ? error.message : '上传失败，请稍后重试';
      showUploadStatus('error', message);
    } finally {
      setIsUploading(false);
      event.target.value = '';
    }
  };

  const handleInputKey = (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    const isComposing = (event.nativeEvent as ReactKeyboardEvent['nativeEvent'])?.isComposing;
    if (event.key === 'Enter' && !event.shiftKey && !isComposing) {
      event.preventDefault();
      event.currentTarget.form?.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    }
  };

  const toggleSidebar = () => setIsSidebarCollapsed((prev) => !prev);

  return (
    <main className="screen">
      <div className={`chat-app ${isSidebarCollapsed ? 'collapsed' : ''}`}>
        <aside className={`history-panel ${isSidebarCollapsed ? 'collapsed' : ''}`} aria-label="历史会话">
          <div className="history-header">
            <button className="pill-btn primary" onClick={createNewSession}>
              <SparklesIcon /> 新对话
            </button>
          </div>
          <div className="history-scroll" role="list">
            {sessions.map((session) => {
              const createdAtLabel = hydrated ? deriveSessionTimestamp(session) : '';
              const isActive = session.id === activeSession?.id;
              const isRenaming = renamingSessionId === session.id;
              const itemClassName = ['history-item', isActive ? 'active' : '', isRenaming ? 'renaming' : '']
                .filter(Boolean)
                .join(' ');

              return (
                <div
                  key={session.id}
                  className={itemClassName}
                  onClick={() => setActiveSessionId(session.id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      setActiveSessionId(session.id);
                    }
                  }}
                >
                  <div className="history-text">
                    {isRenaming ? (
                      <input
                        className="session-title-input"
                        autoFocus
                        value={renameDraft}
                        onClick={(event) => event.stopPropagation()}
                        onChange={(event) => setRenameDraft(event.target.value)}
                        onBlur={() => commitRename(session.id, renameDraft)}
                        onKeyDown={(event) => handleRenameKey(event, session.id)}
                      />
                    ) : (
                      <div className="history-lines">
                        <p className="session-title" title={session.title}>
                          {session.title}
                        </p>
                        {createdAtLabel && <time className="session-time">{createdAtLabel}</time>}
                      </div>
                    )}
                  </div>
                  <div className="session-actions" onClick={(event) => event.stopPropagation()}>
                    {isRenaming ? (
                      <>
                        <button
                          type="button"
                          className="icon-btn"
                          onMouseDown={(event) => event.preventDefault()}
                          onClick={() => commitRename(session.id, renameDraft)}
                          aria-label="保存名称"
                        >
                          <CheckIcon />
                        </button>
                        <button
                          type="button"
                          className="icon-btn"
                          onMouseDown={(event) => event.preventDefault()}
                          onClick={cancelRename}
                          aria-label="取消重命名"
                        >
                          <CloseIcon />
                        </button>
                      </>
                    ) : (
                      <>
                        <button
                          type="button"
                          className="icon-btn"
                          onClick={() => beginRename(session)}
                          title="重命名"
                          aria-label="重命名"
                        >
                          <PencilIcon />
                        </button>
                        <button
                          type="button"
                          className="icon-btn danger"
                          onClick={() => confirmDeleteSession(session.id)}
                          title="删除"
                          aria-label="删除"
                        >
                          <TrashIcon />
                        </button>
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="history-footer">
            <button
              type="button"
              className="collapse-toggle"
              onClick={toggleSidebar}
              aria-label={isSidebarCollapsed ? '展开侧边栏' : '收起侧边栏'}
            >
              {isSidebarCollapsed ? <ChevronRightIcon /> : <ChevronLeftIcon />}
            </button>
          </div>
        </aside>

        <section className="conversation-shell">
          <div className="conversation-inner">
            <nav className="top-nav">
              <div className="title-stack">
                <div className="logo-mark" aria-hidden>
                  <SparklesIcon />
                </div>
                <div className="title-text">
                  <h1>Second Brain</h1>
                </div>
              </div>
              <div className="nav-actions">
              </div>
            </nav>

            <div className="conversation-body">
              <section className="chat-feed" aria-live="polite">
                {hasContent ? (
                  messages.map((message) => {
                    if (message.isError) {
                      return (
                        <div key={message.id} className="message-error" role="alert">
                          <span>{message.content}</span>
                        </div>
                      );
                    }

                    const showThinking = message.role === 'assistant' && message.isThinking;
                    const hasTextContent = Boolean(message.content.trim());
                    const shouldRenderBubble =
                      message.role !== 'assistant' || hasTextContent || (showThinking && !message.statusText);
                    const timestampLabel = hydrated ? formatTimestamp(message.timestamp) : '';
                    const timestampIso =
                      message.timestamp && !Number.isNaN(message.timestamp)
                        ? new Date(message.timestamp).toISOString()
                        : undefined;
                    const sourceEntries: SourceRef[] =
                      message.sourceRefs ?? message.sources?.map((path) => ({ path, heading: '' })) ?? [];
                    const hasSources = sourceEntries.length > 0;
                    const groupedSources = new Map<string, SourceGroup>();
                    const urlSources: string[] = [];
                    const urlSet = new Set<string>();

                    sourceEntries.forEach((ref) => {
                      const { path, heading } = ref;
                      if (!path) return;
                      const isUrl = path.startsWith('http://') || path.startsWith('https://');
                      if (isUrl) {
                        if (!urlSet.has(path)) {
                          urlSources.push(path);
                          urlSet.add(path);
                        }
                        return;
                      }
                      const fileName = path.split('/').pop() ?? path;
                      const existing = groupedSources.get(path);
                      const group = existing ?? { path, fileName, headingRefs: new Map<string, SourceRef>() };
                      const trimmedHeading = heading?.trim();
                      if (trimmedHeading) {
                        const stored = group.headingRefs.get(trimmedHeading);
                        if (
                          !stored ||
                          (!stored.snippet && ref.snippet) ||
                          (stored.char_offset === undefined && ref.char_offset !== undefined)
                        ) {
                          group.headingRefs.set(trimmedHeading, ref);
                        }
                      }
                      if (!existing) {
                        groupedSources.set(path, group);
                      }
                    });

                    return (
                      <article key={message.id} className={`message-row ${message.role}`}>
                        <div className="message-avatar" aria-hidden>
                          <div className={`avatar ${message.role}`}>
                            {message.role === 'user' ? <UserIcon /> : <BotIcon />}
                          </div>
                        </div>
                        <div className="message-stack">
                          {/* Answer-first 渲染模式 */}
                          {FEATURE_ANSWER_FIRST_REDESIGN && message.role === 'assistant' ? (
                            <>
                              {!message.isThinking && (
                                <AnswerPanel
                                  message={message}
                                  copiedKey={copiedKey}
                                  onCopyCode={handleCopy}
                                  onOpenPreview={(path, title, ref) => {
                                    const sourceRef: SourceRef | undefined = ref ? {
                                      path,
                                      heading: '',
                                      char_offset: ref.char_offset,
                                      snippet: ref.snippet,
                                    } : undefined;
                                    handleOpenPreview(path, title, sourceRef);
                                  }}
                                />
                              )}
                              <ProcessPanel message={message} />
                              {message.statusText && (
                                <div className="message-status" role="status" aria-live="polite">
                                  <span className="status-spinner" aria-hidden />
                                  <span>{message.statusText}</span>
                                </div>
                              )}
                            </>
                          ) : (
                            <>
                              {/* 旧版渲染模式 */}
                              {shouldRenderBubble && (
                                <div className="message-bubble">
                                  <div className="message-content">
                                    {hasTextContent && (
                                      <MarkdownMessage
                                        content={message.content}
                                        messageId={message.id}
                                        copiedKey={copiedKey}
                                        onCopyCode={handleCopy}
                                      />
                                    )}
                                    {!hasTextContent && showThinking && <span>&nbsp;</span>}
                                    {showThinking && !message.statusText && <ThinkingDots />}
                                  </div>
                                </div>
                              )}
                              {message.thinkingSteps && message.thinkingSteps.length > 0 && (
                                <ThinkingTimeline
                                  steps={message.thinkingSteps}
                                  currentStepId={message.currentStepId}
                                  isComplete={!message.isThinking}
                                />
                              )}
                              {message.statusText && (
                                <div className="message-status" role="status" aria-live="polite">
                                  <span className="status-spinner" aria-hidden />
                                  <span>{message.statusText}</span>
                                </div>
                              )}
                              {message.role === 'assistant' && hasSources && !message.isThinking && (
                                <div className="sources-panel">
                                  <p className="sources-label">来源文件</p>
                                  <ul className="sources-list">
                                    {Array.from(groupedSources.values()).map((group) => {
                                      const headingEntries = Array.from(group.headingRefs.entries());
                                      const groupKey = `${message.id}:${group.path}`;
                                      const isExpanded = Boolean(expandedGroups[groupKey]);
                                      const displayPath = formatSourcePath(group.path);
                                      return (
                                        <li key={groupKey} className="source-group">
                                          <div className="source-group-header">
                                            <div className="source-group-meta">
                                              <button
                                                type="button"
                                                className="source-filename-btn"
                                                onClick={() => handleOpenPreview(group.path, group.fileName)}
                                                aria-label={`预览 ${group.fileName}`}
                                              >
                                                {group.fileName}
                                              </button>
                                              <span className="source-path" title={group.path}>
                                                {displayPath}
                                              </span>
                                            </div>
                                            {headingEntries.length > 0 && (
                                              <button
                                                type="button"
                                                className="source-expand-btn"
                                                onClick={() => toggleGroup(groupKey)}
                                                aria-expanded={isExpanded}
                                              >
                                                {isExpanded ? <ChevronLeftIcon /> : <ChevronRightIcon />}
                                                {headingEntries.length} 个章节
                                              </button>
                                            )}
                                          </div>
                                          {headingEntries.length > 0 && isExpanded && (
                                            <ul className="source-headings-list">
                                              {headingEntries.map(([heading, ref], headingIndex) => (
                                                <li
                                                  key={`${groupKey}-heading-${headingIndex}`}
                                                  className="source-heading-item"
                                                >
                                                  <button
                                                    type="button"
                                                    className="source-heading-btn"
                                                    onClick={() => handleOpenPreview(group.path, group.fileName, ref)}
                                                  >
                                                    {heading}
                                                  </button>
                                                </li>
                                              ))}
                                            </ul>
                                          )}
                                        </li>
                                      );
                                    })}
                                    {urlSources.map((path, sourceIndex) => (
                                      <li key={`${message.id}-url-${sourceIndex}`} className="source-item">
                                        <a
                                          href={path}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          className="source-link"
                                        >
                                          {path}
                                        </a>
                                      </li>
                                    ))}
                                  </ul>
                                </div>
                              )}
                            </>
                          )}
                          <div className="message-meta">
                            <div className="bubble-actions">
                              <button
                                type="button"
                                className="bubble-action"
                                onClick={() => handleCopy(message.content, message.id)}
                                disabled={!message.content}
                                aria-label="复制消息"
                              >
                                <CopyIcon />
                              </button>
                            </div>
                            {timestampLabel && (
                              <time className="message-time" dateTime={timestampIso}>
                                {timestampLabel}
                              </time>
                            )}
                          </div>
                        </div>
                      </article>
                    );
                  })
                ) : (
                  <div className="empty-state">
                    <p>提出你的第一个问题，让 Second Brain 协助你梳理想法。</p>
                    <button className="text-btn" type="button" onClick={createNewSession}>
                      <SparklesIcon /> 发起对话
                    </button>
                  </div>
                )}
                <div ref={historyEndRef} />
              </section>

              <form className="composer" onSubmit={handleSubmit}>
                <div className="composer-field">
                  <div className="composer-tools">
                    <input
                      ref={uploadInputRef}
                      className="file-input"
                      type="file"
                      accept=".md"
                      onChange={handleUploadChange}
                      aria-hidden="true"
                      tabIndex={-1}
                    />
                    <button
                      type="button"
                      className="upload-btn"
                      onClick={handleUploadClick}
                      disabled={isUploading}
                      aria-label="上传 Markdown 文档"
                      title="上传 Markdown 文档"
                    >
                      <UploadIcon />
                    </button>
                  </div>
                  <textarea
                    ref={textareaRef}
                    placeholder={
                      isActivePendingFlag ? '等待当前回复完成...' : '输入你的问题，Shift+Enter 换行'
                    }
                    value={inputValue}
                    onChange={(event) => setInputValue(event.target.value)}
                    onKeyDown={handleInputKey}
                    disabled={isActivePendingFlag && !inputValue}
                  />
                  {isActivePendingFlag && activeSession ? (
                    <button
                      type="button"
                      className="send-btn stop-btn"
                      onClick={() => abortSessionRequest(activeSession.id)}
                      aria-label="停止生成"
                      title="停止生成"
                    >
                      <StopIcon />
                    </button>
                  ) : (
                    <button
                      type="submit"
                      className="send-btn"
                      data-ready={Boolean(inputValue.trim()) && !isActivePendingFlag}
                      disabled={!inputValue.trim() || isActivePendingFlag}
                      aria-label="发送"
                    >
                      <SendIcon />
                    </button>
                  )}
                </div>
                {uploadStatus && (
                  <div className={`composer-status ${uploadStatus.tone}`} role="status">
                    <span>{uploadStatus.message}</span>
                  </div>
                )}
              </form>
            </div>
          </div>
        </section>
      </div>
      {previewState && (
        <div className="preview-overlay" onClick={closePreview} role="presentation">
          <div
            className="preview-panel"
            role="dialog"
            aria-modal="true"
            aria-label={`预览 ${previewState.title}`}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="preview-header">
              <span className="preview-title">{previewState.title}</span>
              <button
                type="button"
                className="preview-close"
                onClick={closePreview}
                aria-label="关闭预览"
              >
                <CloseIcon />
              </button>
            </div>
            <div className="preview-body">
              {previewLoading && !previewState.content ? (
                <div className="preview-loading">
                  <ThinkingDots />
                </div>
              ) : (
                <>
                  {previewError && (
                    <div className="preview-error" role="alert">
                      {previewError}
                    </div>
                  )}
                  <PreviewContent
                    content={previewState.content}
                    loadedOffset={previewState.loadedOffset}
                    scrollTarget={previewState.scrollTarget}
                  />
                  <div className="preview-footer">
                    <span className="preview-meta">
                      {previewState.totalChars > 0
                        ? `已加载 ${Math.min(
                            previewState.loadedOffset + previewState.content.length,
                            previewState.totalChars,
                          )} / ${previewState.totalChars} 字符`
                        : '已加载内容'}
                    </span>
                    {!previewState.done && (
                      <button
                        type="button"
                        className="preview-load-btn"
                        onClick={handleLoadMorePreview}
                        disabled={previewLoading}
                      >
                        {previewLoading ? '加载中...' : '加载更多'}
                      </button>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

const ThinkingDots = () => (
  <span className="thinking" aria-label="回答生成中" role="status">
    <span className="thinking-dot" />
    <span className="thinking-dot" />
    <span className="thinking-dot" />
  </span>
);
