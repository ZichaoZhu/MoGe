import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "MoGe-3 多实验点云对比器",
  description: "交互比较 Exp9 与 Exp12 的训练阶段及不同 SSR 精修次数",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
