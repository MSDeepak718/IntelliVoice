import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/ws': {
        target: 'http://localhost:8000',
        ws: true,
      },
      '/health': {
        target: 'http://localhost:8000',
      },
      '/chat': {
        target: 'http://localhost:8000',
      },
    },
  },
})

