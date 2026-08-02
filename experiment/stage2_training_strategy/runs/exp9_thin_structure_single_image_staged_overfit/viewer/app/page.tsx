"use client";

import dynamic from "next/dynamic";

const PointCloudComparison = dynamic(
  () =>
    import("@/components/PointCloudComparison").then(
      (module) => module.PointCloudComparison,
    ),
  {
    ssr: false,
    loading: () => (
      <main className="boot-screen">
        <div className="boot-mark">M3</div>
        <p>正在载入 MoGe-3 多实验三维查看器…</p>
      </main>
    ),
  },
);

export default function Home() {
  return <PointCloudComparison />;
}
