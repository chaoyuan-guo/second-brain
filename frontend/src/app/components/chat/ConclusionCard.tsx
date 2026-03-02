'use client';

interface ConclusionCardProps {
  conclusion: string;
}

/**
 * ConclusionCard - 一句话结论卡片
 * 展示核心结论，字号与对比度高于正文一级
 */
export function ConclusionCard({ conclusion }: ConclusionCardProps) {
  if (!conclusion?.trim()) {
    return null;
  }

  return (
    <div className="conclusion-card">
      <div className="conclusion-icon" aria-hidden>
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
          <polyline points="22 4 12 14.01 9 11.01" />
        </svg>
      </div>
      <p className="conclusion-text">{conclusion}</p>
    </div>
  );
}