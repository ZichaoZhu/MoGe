import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  allowedDevOrigins: ["10.162.196.95", "127.0.0.1"],
  reactStrictMode: true,
  output: "standalone",
  experimental: {
    optimizePackageImports: ["@react-three/drei", "@react-three/fiber"],
  },
  async headers() {
    return [
      {
        source: "/data/:path*",
        headers: [
          {
            key: "Access-Control-Allow-Origin",
            value: "*",
          },
          {
            key: "Access-Control-Allow-Methods",
            value: "GET, HEAD, OPTIONS",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
