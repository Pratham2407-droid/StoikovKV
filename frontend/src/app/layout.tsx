import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'StoikovKV Dashboard',
  description: 'Hybrid KV Cache Compression Dashboard',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-sans antialiased bg-bg text-slate-100">
        {children}
      </body>
    </html>
  );
}
