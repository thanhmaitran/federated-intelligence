import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = process.env.FI_BACKEND ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  build: { chunkSizeWarningLimit: 1200 },
  server: { port: 5173, proxy: { "/api": backend } },
  preview: { port: 4173, proxy: { "/api": backend } },
});
