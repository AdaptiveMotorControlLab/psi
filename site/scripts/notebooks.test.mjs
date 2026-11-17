import assert from "node:assert/strict";
import test from "node:test";
import katex from "katex";
import { cleanHtml } from "../src/lib/notebooks.mjs";

test("tables retain data while active HTML is removed", () => {
  const html = cleanHtml(
    '<table><tbody><tr><td onclick="alert(1)">measured</td></tr></tbody></table><script>alert(1)</script><iframe src="https://example.org"></iframe>',
  );
  assert.match(html, /<td>measured<\/td>/);
  assert.doesNotMatch(html, /script|iframe|onclick|alert/);
});

test("links reject active schemes and images lose event handlers", () => {
  const html = cleanHtml(
    '<a href="javascript:alert(1)">link</a><img src="https://example.org/plot.png" onerror="alert(1)">',
  );
  assert.doesNotMatch(html, /javascript|onerror|alert/);
  assert.match(html, /https:\/\/example.org\/plot.png/);
});

test("math markup and notebook headings retain their structure", () => {
  const html = cleanHtml(
    '<h1>Reading</h1><span class="katex"><math><semantics><mi>q</mi></semantics></math></span>',
  );
  assert.match(html, /<mi>q<\/mi>/);
  assert.match(html, /class="katex"/);
});

test("KaTeX radical geometry survives sanitizing", () => {
  const html = cleanHtml(
    '<span class="katex"><span style="min-width:0.853em"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400000 1000"><path d="M0 0 L10 10" /></svg></span></span>',
  );
  assert.match(html, /<svg/);
  assert.match(html, /viewBox="0 0 400000 1000"/);
  assert.match(html, /<path d="M0 0 L10 10"/);
  assert.match(html, /min-width:0.853em/);
});

test("real KaTeX roots retain both geometry and radicand spacing", () => {
  const rendered = katex.renderToString(String.raw`\sqrt{x}`);
  const clean = cleanHtml(rendered);
  assert.match(clean, /<svg/);
  assert.match(clean, /<path/);
  assert.match(clean, /padding-left:[0-9.]+em/);
});
