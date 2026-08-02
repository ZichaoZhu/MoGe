import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  allowedDevOrigins: ["10.162.196.95", "127.0.0.1"],
  reactStrictMode: true,
  output: "standalone",
  experimental: {
    optimizePackageImports: ["@react-three/drei", "@react-three/fiber"],
  },
};

export default nextConfig;
