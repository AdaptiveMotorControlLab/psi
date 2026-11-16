// Static output for GitHub Pages. SITE_URL and SITE_BASE are set by the deploy
// workflow (.github/workflows/site.yml); locally the page is served from "/".
import { defineConfig } from "astro/config";
import { page } from "./src/config.mjs";

export default defineConfig({
  output: "static",
  site: process.env.SITE_URL || new URL(page).origin,
  base: process.env.SITE_BASE || "/",
  trailingSlash: "ignore",
  devToolbar: { enabled: false },
  build: { assets: "assets" },
});
