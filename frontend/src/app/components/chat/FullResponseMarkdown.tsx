'use client';

import { Children, cloneElement, isValidElement, type ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { CitationRef } from '../../lib/chat-types';
import { isPreciseCitationRef, normalizeCitationId } from '../../lib/citation-utils';
import { InlineCitation } from './InlineCitation';

interface FullResponseMarkdownProps {
  content: string;
  messageId: string;
  copiedKey: string | null;
  onCopyCode: (value: string, key: string) => void;
  citationMap?: Record<string, CitationRef>;
  onOpenPreview?: (path: string, title: string, ref?: { char_offset?: number; snippet?: string }) => void;
  title?: string | null;
}

const CITE_RE = /\[c(\d{2,3})\]/g;

const renderTextWithCitations = (
  text: string,
  citationMap: Record<string, CitationRef>,
  onOpenPreview?: (path: string, title: string, ref?: { char_offset?: number; snippet?: string }) => void,
): ReactNode[] => {
  const parts: ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = CITE_RE.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    const normalizedId = normalizeCitationId(match[1]);
    const ref = citationMap[match[1]] || citationMap[normalizedId] || citationMap[`c${normalizedId}`];
    if (!isPreciseCitationRef(ref)) {
      parts.push(match[0]);
      lastIndex = match.index + match[0].length;
      continue;
    }
    parts.push(
      <InlineCitation
        key={`full-citation-${match.index}`}
        citationId={match[1]}
        citationMap={citationMap}
        onOpenPreview={onOpenPreview}
      />,
    );
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  CITE_RE.lastIndex = 0;
  return parts;
};

const renderCitationNodes = (
  node: ReactNode,
  citationMap: Record<string, CitationRef>,
  onOpenPreview?: (path: string, title: string, ref?: { char_offset?: number; snippet?: string }) => void,
): ReactNode => {
  if (Array.isArray(node)) {
    return Children.map(node, (child) => renderCitationNodes(child, citationMap, onOpenPreview));
  }

  if (typeof node === 'string') {
    return renderTextWithCitations(node, citationMap, onOpenPreview);
  }

  if (!isValidElement(node)) {
    return node;
  }

  if (typeof node.type === 'string' && (node.type === 'code' || node.type === 'pre')) {
    return node;
  }

  const props = node.props as { children?: ReactNode };
  if (props.children === undefined) {
    return node;
  }

  const children = Children.map(props.children, (child) =>
    renderCitationNodes(child, citationMap, onOpenPreview),
  );

  return cloneElement(node, undefined, children);
};

/**
 * FullResponseMarkdown - 完整 Markdown 回答区块
 * 作为兜底展示完整回答内容
 */
export function FullResponseMarkdown({
  content,
  messageId,
  copiedKey,
  onCopyCode,
  citationMap = {},
  onOpenPreview,
  title = '完整分析',
}: FullResponseMarkdownProps) {
  let codeBlockIndex = 0;

  if (!content?.trim()) {
    return null;
  }

  return (
    <div className="full-response">
      {title ? <h4 className="full-response-title">{title}</h4> : null}
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p>{renderCitationNodes(children, citationMap, onOpenPreview)}</p>,
          li: ({ children }) => <li>{renderCitationNodes(children, citationMap, onOpenPreview)}</li>,
          blockquote: ({ children }) => <blockquote>{renderCitationNodes(children, citationMap, onOpenPreview)}</blockquote>,
          h1: ({ children }) => <h1>{renderCitationNodes(children, citationMap, onOpenPreview)}</h1>,
          h2: ({ children }) => <h2>{renderCitationNodes(children, citationMap, onOpenPreview)}</h2>,
          h3: ({ children }) => <h3>{renderCitationNodes(children, citationMap, onOpenPreview)}</h3>,
          h4: ({ children }) => <h4>{renderCitationNodes(children, citationMap, onOpenPreview)}</h4>,
          h5: ({ children }) => <h5>{renderCitationNodes(children, citationMap, onOpenPreview)}</h5>,
          h6: ({ children }) => <h6>{renderCitationNodes(children, citationMap, onOpenPreview)}</h6>,
          a: ({ href, children, ...props }) => {
            const normalizedHref = typeof href === 'string' ? href : '';
            if (!normalizedHref) {
              return <span>{renderCitationNodes(children, citationMap, onOpenPreview)}</span>;
            }
            return (
              <a
                {...props}
                className="inline-link"
                href={normalizedHref}
                target="_blank"
                rel="noopener noreferrer"
              >
                {renderCitationNodes(children, citationMap, onOpenPreview)}
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
