
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // Listen on all network interfaces
    port: 3000,
    watch: {
      usePolling: true, // Needed for Docker on Windows/WSL2
    },
    proxy: {
      '/api': {
        target: 'http://api-gateway:8080',
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
