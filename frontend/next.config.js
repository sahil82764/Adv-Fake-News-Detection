/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/predict/:path*',
        destination: 'http://localhost:8000/api/predict/:path*', // Proxy to FastAPI backend
      },
      {
        source: '/api/compare/:path*',
        destination: 'http://localhost:8000/api/compare/:path*', // Proxy to FastAPI backend
      },
      {
        source: '/api/ensemble/:path*',
        destination: 'http://localhost:8000/api/ensemble/:path*', // Proxy to FastAPI backend
      },
    ];
  },
};

module.exports = nextConfig;