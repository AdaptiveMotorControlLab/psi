// Screenshots of the built page in both themes: hero, widget, full page, phone width.
//   npm run build && npm run screenshots   ->  screenshots/*.png
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { launch } from "./browser.mjs";
import { serve } from "./serve.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
const out = path.resolve(here, "../screenshots");
await mkdir(out, { recursive: true });
const server = await serve();
const browser = await launch();
const errors = [];

// t freezes the hero: 6.8 s is after the swap, with the donor's outline showing.
const shots = [];
for (const theme of ["light", "dark"]) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 2 });
  page.on("pageerror", (e) => errors.push(`${theme}: ${e.message}`));
  page.on("console", (m) => m.type() === "error" && errors.push(`${theme}: ${m.text()}`));
  // The sticky navigation would sit on top of element screenshots; pin it to the page instead.
  const unpin = () => page.addStyleTag({ content: ".topbar { position: static !important; }" });
  await page.goto(`${server.url}/?theme=${theme}&t=6.8`, { waitUntil: "networkidle" });
  await unpin();
  await page.locator(".hero").screenshot({ path: `${out}/hero-${theme}.png` });
  await page.goto(`${server.url}/?theme=${theme}&t=6.8&pair=1`, { waitUntil: "networkidle" });
  await unpin();
  await page.locator(".hero").screenshot({ path: `${out}/hero-pair2-${theme}.png` });
  await page.goto(`${server.url}/?theme=${theme}&t=2.0`, { waitUntil: "networkidle" });
  await unpin();
  await page.locator(".hero").screenshot({ path: `${out}/hero-project-${theme}.png` });
  await page.locator("#transfer").screenshot({ path: `${out}/widget-${theme}.png` });
  await page.locator("#results").screenshot({ path: `${out}/results-${theme}.png` });
  await page.screenshot({ path: `${out}/page-${theme}.png`, fullPage: true });
  await page.close();

  const phone = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  phone.on("pageerror", (e) => errors.push(`phone ${theme}: ${e.message}`));
  await phone.goto(`${server.url}/?theme=${theme}&t=6.8`, { waitUntil: "networkidle" });
  const overflow = await phone.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  if (overflow > 0) errors.push(`phone ${theme}: page scrolls sideways by ${overflow}px`);
  await phone.screenshot({ path: `${out}/phone-${theme}.png`, fullPage: true });
  await phone.close();
  shots.push(theme);
}
await browser.close();
server.close();
console.log(`wrote screenshots for ${shots.join(", ")} to ${path.relative(process.cwd(), out)}/`);
if (errors.length) { console.error(errors.join("\n")); process.exitCode = 1; }
