/** @type {import('next').NextConfig} */
const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const nextConfig = {
  output: 'export',
  async rewrites() {
    return [
      {
        source: '/api/predict/:path*',
        destination: `${backendUrl}/api/predict/:path*`,
      },
      {
        source: '/api/compare/:path*',
        destination: `${backendUrl}/api/compare/:path*`,
      },
      {
        source: '/api/ensemble/:path*',
        destination: `${backendUrl}/api/ensemble/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;