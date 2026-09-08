import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: '/rsu-hold-vs-sell-simulator/',
  server: {
    port: 3003,
    host: '0.0.0.0',
  },
});
