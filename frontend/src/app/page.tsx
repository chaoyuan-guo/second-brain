"use client";

import { KeyboardEvent, ReactNode, useCallback, useEffect, useRef, useState } from 'react';

import { LinkCard } from './components/LinkCard';
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
  UserIcon,
} from './components/icons';
import { useChatSessions } from './hooks/useChatSessions';
import {
  deriveSessionTimestamp,
  formatTimestamp,
  isStandaloneUrl,
  parseMessageSegments,
} from './lib/chat-helpers';
import type { ChatSession } from './lib/chat-types';

const urlRegex = /(https?:\/\/[^\s]+)/gi;

const renderTextWithLinks = (text: string): ReactNode[] => {
  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  const regex = new RegExp(urlRegex);

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    const href = match[0];
    nodes.push(
      <a
        key={`inline-link-${href}-${match.index}`}
        className="inline-link"
        href={href}
        target="_blank"
        rel="noreferrer"
      >
        {href.replace(/^https?:\/\//, '').replace(/^www\./, '')}
      </a>,
    );
    lastIndex = match.index + href.length;
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }

  return nodes.length ? nodes : [text];
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
  const historyEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const copyTimerRef = useRef<number | null>(null);

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

  const handleRenameKey = (event: KeyboardEvent<HTMLInputElement>, sessionId: string) => {
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

  const handleInputKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    const isComposing = (event.nativeEvent as KeyboardEvent['nativeEvent'])?.isComposing;
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
                    const segments = hasTextContent ? parseMessageSegments(message.content) : [];
                    const shouldRenderBubble =
                      message.role !== 'assistant' || hasTextContent || (showThinking && !message.statusText);
                    const timestampLabel = hydrated ? formatTimestamp(message.timestamp) : '';
                    const timestampIso =
                      message.timestamp && !Number.isNaN(message.timestamp)
                        ? new Date(message.timestamp).toISOString()
                        : undefined;

                    return (
                      <article key={message.id} className={`message-row ${message.role}`}>
                        <div className="message-avatar" aria-hidden>
                          <div className={`avatar ${message.role}`}>
                            {message.role === 'user' ? <UserIcon /> : <BotIcon />}
                          </div>
                        </div>
                        <div className="message-stack">
                          {shouldRenderBubble && (
                            <div className="message-bubble">
                              <div className="message-content">
                                {segments.map((segment, index) => {
                                  if (segment.type === 'code') {
                                    return (
                                      <div key={`${message.id}-code-${index}`} className="code-block">
                                        <div className="code-header">
                                          <span>{segment.language}</span>
                                          <button
                                            type="button"
                                            onClick={() =>
                                              handleCopy(segment.content, `${message.id}-code-${index}`)
                                            }
                                            aria-label="复制代码"
                                          >
                                            {copiedKey === `${message.id}-code-${index}`
                                              ? '已复制'
                                              : '复制'}
                                          </button>
                                        </div>
                                        <pre>
                                          <code>{segment.content}</code>
                                        </pre>
                                      </div>
                                    );
                                  }

                                  const paragraphs = segment.content.split(/\n{2,}/);
                                  return paragraphs.map((paragraph, paragraphIndex) => {
                                    const trimmed = paragraph.trim();
                                    if (isStandaloneUrl(trimmed)) {
                                      return (
                                        <LinkCard
                                          key={`${message.id}-link-${index}-${paragraphIndex}`}
                                          href={trimmed}
                                        />
                                      );
                                    }
                                    return (
                                      <p key={`${message.id}-p-${index}-${paragraphIndex}`}>
                                        {paragraph.split('\n').map((line, lineIndex) => (
                                          <span
                                            key={`${message.id}-line-${index}-${paragraphIndex}-${lineIndex}`}
                                          >
                                            {renderTextWithLinks(line)}
                                            {lineIndex < paragraph.split('\n').length - 1 && <br />}
                                          </span>
                                        ))}
                                      </p>
                                    );
                                  });
                                })}
                                {!segments.length && showThinking && <span>&nbsp;</span>}
                                {showThinking && !message.statusText && <ThinkingDots />}
                              </div>
                            </div>
                          )}
                          {message.statusText && (
                            <div className="message-status" role="status" aria-live="polite">
                              <span className="status-spinner" aria-hidden />
                              <span>{message.statusText}</span>
                            </div>
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
              </form>
            </div>
          </div>
        </section>
      </div>
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
