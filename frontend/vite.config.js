import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  root: '.', // Set project root to frontend/
  publicDir: 'public', // Public assets directory
  build: {
    outDir: 'dist', // Output directory for build
    sourcemap: true, // Generate sourcemaps for debugging
  },
  server: {
    port: 3000, // Development server port
  },
});