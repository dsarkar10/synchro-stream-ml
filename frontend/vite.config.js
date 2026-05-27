import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/profile': 'http://localhost:8000',
      '/simulate': 'http://localhost:8000',
    },
  },
})
