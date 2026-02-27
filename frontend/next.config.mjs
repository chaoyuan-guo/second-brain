/** @type {import('next').NextConfig} */
const outputMode = process.env.NEXT_OUTPUT_MODE?.trim().toLowerCase();

const nextConfig = {
  reactStrictMode: true,
  trailingSlash: true,
};

if (outputMode === 'export') {
  nextConfig.output = 'export';
}

export default nextConfig;
