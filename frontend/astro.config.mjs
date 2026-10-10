import { defineConfig } from "astro/config";
import tailwindcss from "@tailwindcss/vite";

// WriteTrace frontend build.
//
// * No remote assets, fonts, or scripts are referenced at runtime: typography
//   uses system font stacks and the paper grain is an inline SVG filter.
// * The backend base URL is read from PUBLIC_WRITETRACE_API_BASE (baked into
//   the client bundle at build time) and defaults to the local FastAPI server.
// Dev/preview convenience: forwards the same-origin `/api` path to the local
// FastAPI backend so the UI works without CORS. This is a local proxy only —
// in production the frontend calls PUBLIC_WRITETRACE_API_BASE directly.
const apiProxy = {
  "/api": {
    target: process.env.WRITETRACE_BACKEND_ORIGIN ?? "http://127.0.0.1:8000",
    changeOrigin: true,
  },
};

export default defineConfig({
  site: "http://localhost:4321",
  output: "static",
  trailingSlash: "ignore",
  vite: {
    plugins: [tailwindcss()],
    server: { proxy: apiProxy },
    preview: { proxy: apiProxy },
  },
});