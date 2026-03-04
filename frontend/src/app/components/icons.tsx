import { ReactNode } from 'react';

const IconSvg = ({ children }: { children: ReactNode }) => (
  <svg
    width="18"
    height="18"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.8"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    {children}
  </svg>
);

export const PencilIcon = () => (
  <IconSvg>
    <path d="M12 20h9" />
    <path d="M16.5 3.5l4 4L7 21H3v-4L16.5 3.5z" />
  </IconSvg>
);

export const TrashIcon = () => (
  <IconSvg>
    <path d="M3 6h18" />
    <path d="M8 6V4h8v2" />
    <path d="M7 6l1 14h8l1-14" />
    <path d="M10 10v6" />
    <path d="M14 10v6" />
  </IconSvg>
);

export const CheckIcon = () => (
  <IconSvg>
    <path d="M5 13l4 4L19 7" />
  </IconSvg>
);

export const CloseIcon = () => (
  <IconSvg>
    <path d="M6 6l12 12" />
    <path d="M6 18L18 6" />
  </IconSvg>
);

export const SparklesIcon = () => (
  <IconSvg>
    <path d="M12 3l1.2 3.8 3.8 1.2-3.8 1.2-1.2 3.8-1.2-3.8L7 8l3.8-1.2z" />
    <path d="M5 16l.7 2 .7-2 2-.7-2-.6-.7-2-.7 2-2 .6z" />
    <path d="M18 15l.6 1.4.6-1.4 1.4-.6-1.4-.6-.6-1.4-.6 1.4-1.4.6z" />
  </IconSvg>
);

export const ChevronLeftIcon = () => (
  <IconSvg>
    <path d="M15 6l-6 6 6 6" />
  </IconSvg>
);

export const ChevronRightIcon = () => (
  <IconSvg>
    <path d="M9 6l6 6-6 6" />
  </IconSvg>
);

export const UserIcon = () => (
  <IconSvg>
    <circle cx="12" cy="9" r="3" />
    <path d="M6 19c0-2.5 3-4 6-4s6 1.5 6 4" />
  </IconSvg>
);

export const BotIcon = () => (
  <IconSvg>
    <rect x="5" y="7" width="14" height="10" rx="3" />
    <circle cx="9" cy="12" r="1" />
    <circle cx="15" cy="12" r="1" />
    <path d="M12 4v3" />
    <path d="M8 21h8" />
  </IconSvg>
);

export const CopyIcon = () => (
  <IconSvg>
    <rect x="9" y="9" width="12" height="12" rx="2" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </IconSvg>
);

export const SendIcon = () => (
  <IconSvg>
    <path d="M22 2 11 13" />
    <path d="M22 2 15 22 11 13 2 9z" />
  </IconSvg>
);

export const StopIcon = () => (
  <IconSvg>
    <rect x="6" y="6" width="12" height="12" rx="2" />
  </IconSvg>
);

export const UploadIcon = () => (
  <IconSvg>
    <path d="M12 16V4" />
    <path d="M8 8l4-4 4 4" />
    <path d="M4 20h16" />
  </IconSvg>
);

// 时间线相关图标
export const LoaderIcon = () => (
  <IconSvg>
    <path d="M12 2v4" />
    <path d="M12 18v4" />
    <path d="M4.93 4.93l2.83 2.83" />
    <path d="M16.24 16.24l2.83 2.83" />
    <path d="M2 12h4" />
    <path d="M18 12h4" />
    <path d="M4.93 19.07l2.83-2.83" />
    <path d="M16.24 7.76l2.83-2.83" />
  </IconSvg>
);

export const ErrorIcon = () => (
  <IconSvg>
    <circle cx="12" cy="12" r="10" />
    <path d="M15 9l-6 6" />
    <path d="M9 9l6 6" />
  </IconSvg>
);

export const ChevronDownIcon = () => (
  <IconSvg>
    <path d="M6 9l6 6 6-6" />
  </IconSvg>
);
