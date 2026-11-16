// Record the swap animation for slides, frame by frame, as MP4 (H.264).
//
//   npm run build && npm run record                     # both themes, both pairs, 1920x1080, 30 fps
//   npm run record -- --theme dark --pair 1 --seconds 9 --width 1280 --height 720
//
// Each frame is drawn by calling the animation's render(t) directly, so the
// video has no dropped frames whatever the machine's speed. Frames are piped
// into ffmpeg; nothing is written but recordings/swap-<theme>[-pair<k>].mp4.
import { spawn } from "node:child_process";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import path from "node:path";
import { launch } from "./browser.mjs";
import { serve } from "./serve.mjs";

const { values: opt } = parseArgs({ options: {
  theme: { type: "string", default: "light,dark" },
  pair: { type: "string" },
  seconds: { type: "string" },
  fps: { type: "string", default: "30" },
  width: { type: "string", default: "1920" },
  height: { type: "string", default: "1080" },
} });
const here = path.dirname(fileURLToPath(import.meta.url));
const out = path.resolve(here, "../recordings");
await mkdir(out, { recursive: true });
const server = await serve();
const browser = await launch();
const fps = Number(opt.fps);

for (const theme of opt.theme.split(",")) {
  const page = await browser.newPage({ viewport: { width: Number(opt.width), height: Number(opt.height) } });
  const pair = opt.pair !== undefined ? `&pair=${opt.pair}` : "";
  await page.goto(`${server.url}/embed/swap/?theme=${theme}&t=0${pair}`, { waitUntil: "networkidle" });
  // CSS transitions run on the wall clock; frames are stepped, so switch them off.
  await page.addStyleTag({ content: "*, *::before, *::after { transition: none !important; }" });
  const { period, pairs } = await page.evaluate(() => ({ period: window.__psiSwap.period, pairs: window.__psiSwap.pairs }));
  const seconds = Number(opt.seconds ?? (opt.pair !== undefined ? period : period * pairs));
  const name = `swap-${theme}${opt.pair !== undefined ? `-pair${opt.pair}` : ""}.mp4`;
  const ffmpeg = spawn("ffmpeg", ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(fps), "-i", "-",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", path.join(out, name)],
    { stdio: ["pipe", "inherit", "inherit"] });
  const done = new Promise((resolve, reject) => ffmpeg.on("close", (c) => (c ? reject(new Error(`ffmpeg ${c}`)) : resolve())));
  const frames = Math.round(seconds * fps);
  for (let i = 0; i < frames; i++) {
    await page.evaluate((t) => window.__psiSwap.render(t), i / fps);
    const png = await page.screenshot({ type: "png" });
    if (!ffmpeg.stdin.write(png)) await new Promise((r) => ffmpeg.stdin.once("drain", r));
  }
  ffmpeg.stdin.end();
  await done;
  console.log(`wrote recordings/${name} (${frames} frames, ${seconds}s)`);
  await page.close();
}
await browser.close();
server.close();
