import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, __dirname, '');

  const l1ProxyTarget = env.VITE_PROXY_L1_TARGET || 'http://127.0.0.1:8081';
  const l4ProxyTarget = env.VITE_PROXY_L4_TARGET || 'http://127.0.0.1:8082';
  const policyProxyTarget = env.VITE_PROXY_POLICY_TARGET || 'http://127.0.0.1:8083';

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (!id.includes('node_modules')) {
              return;
            }

            if (
              id.includes('/react/') ||
              id.includes('/react-dom/') ||
              id.includes('/react-router-dom/') ||
              id.includes('/@tanstack/')
            ) {
              return 'vendor-react';
            }

            if (id.includes('/lucide-react/')) {
              return 'vendor-icons';
            }

            if (
              id.includes('/axios/') ||
              id.includes('/clsx/') ||
              id.includes('/tailwind-merge/')
            ) {
              return 'vendor-utils';
            }
          },
        },
      },
    },
    server: {
      port: 3000,
      host: '0.0.0.0',
      proxy: {
        '/api/l1': {
          target: l1ProxyTarget,
          changeOrigin: true,
          rewrite: (requestPath) => requestPath.replace(/^\/api\/l1/, ''),
        },
        '/api/l4': {
          target: l4ProxyTarget,
          changeOrigin: true,
          rewrite: (requestPath) => requestPath.replace(/^\/api\/l4/, ''),
        },
        '/api/policy': {
          target: policyProxyTarget,
          changeOrigin: true,
          rewrite: (requestPath) => requestPath.replace(/^\/api\/policy/, '/api/v1/policy'),
        },
      },
    },
  };
});
