import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'; 

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  alias: {
    '@': path.resolve(__dirname, './src'), // Set the alias '@' to point to 'src' folder
  }
})
