// Launch the Playwright Chromium used by `npm run screenshots` and `npm run record`.
//
// Install it once with `npx playwright install chromium-headless-shell` (add
// `--with-deps` to also install its system libraries, which needs root).
// Without root, the missing libraries can be unpacked from their distribution
// packages into node_modules/.playwright/apt/root; this helper then points the
// browser at them.
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const local = path.resolve(here, "../node_modules/.playwright");
if (!process.env.PLAYWRIGHT_BROWSERS_PATH && existsSync(local)) process.env.PLAYWRIGHT_BROWSERS_PATH = local;

export async function launch() {
  const { chromium } = await import("playwright");
  const libs = path.join(local, "apt/root/usr/lib/x86_64-linux-gnu");
  const env = { ...process.env };
  if (existsSync(libs)) env.LD_LIBRARY_PATH = [libs, env.LD_LIBRARY_PATH].filter(Boolean).join(":");
  return chromium.launch({ env });
}
