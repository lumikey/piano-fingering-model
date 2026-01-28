import { defineConfig } from "vite";

export default defineConfig({
  base: "/piano-fingering-model/",
  build: {
    outDir: "dist",
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
});
