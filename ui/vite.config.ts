import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  // Output to ../dist so tauri.conf.json's frontendDist: "../ui/dist" resolves correctly.
  build: {
    outDir: '../dist',
    emptyOutDir: true,
  },
  // Prevents Vite from obscuring Tauri's backend errors during dev.
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
  },
})
