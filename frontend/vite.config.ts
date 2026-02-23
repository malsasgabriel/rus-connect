import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Load env variables for the current mode
  const env = loadEnv(mode, process.cwd(), '');
  
  // Use the API URL from environment variable or default to localhost
  const apiUrl = env.VITE_API_URL || 'http://localhost:8080';
  
  return {
    plugins: [react()],
    server: {
      host: '0.0.0.0', // Listen on all network interfaces
      port: 3000,
      watch: {
        usePolling: true, // Needed for Docker on Windows/WSL2
      },
      proxy: {
        '/api': {
          target: apiUrl,
          changeOrigin: true,
          secure: false,
        },
        '/api/v1/trader-mind': {
          target: apiUrl,
          changeOrigin: true,
          secure: false,
        },
        // Add proxy for WebSocket connections
        '/ws': {
          target: apiUrl,
          changeOrigin: true,
          secure: false,
          ws: true, // Enable WebSocket proxying
        },
      },
    },
  };
});