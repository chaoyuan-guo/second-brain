import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Second Brain Chat',
  description: '类 ChatGPT 的聊天界面，连接本地后端',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
