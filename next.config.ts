/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  basePath: '/ai-quizs',
  assetPrefix: '/ai-quizs/',
  images: {
    unoptimized: true
  }
};

export default nextConfig;
