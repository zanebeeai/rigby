import { defineConfig } from "vite";

export default defineConfig({
  // Keep the compiled UI bundles separate from the authored rig assets,
  // which intentionally live at /assets/models and /assets/manifest.json.
  base: "/static/",
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8011",
      "/assets": "http://127.0.0.1:8011",
      "/config": "http://127.0.0.1:8011",
    },
  },
  build: {
    sourcemap: true,
    rollupOptions: {
      input: {
        main: "index.html",
        review: "review.html",
        capture: "capture.html",
        comparison: "comparison.html",
      },
    },
  },
});
