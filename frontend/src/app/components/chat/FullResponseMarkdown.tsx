'use client';

import { isValidElement } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface FullResponseMarkdownProps {
  content: string;
  messageId: string;
  copiedKey: string | null;
  onCopyCode: (value: string, key: string) => void;
}

/**
 * FullResponseMarkdown - 完整 Markdown 回答区块
 * 作为兜底展示完整回答内容
 */
export function FullResponseMarkdown({
  content,
  messageId,
  copiedKey,
  onCopyCode,
}: FullResponseMarkdownProps) {
  let codeBlockIndex = 0;

  if (!content?.trim()) {
    return null;
  }

  return (
    <div className="full-response-markdown">
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
    </div>
  );
}