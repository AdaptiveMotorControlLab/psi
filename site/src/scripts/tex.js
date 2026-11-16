// LaTeX to HTML at build time. The page ships KaTeX's CSS and fonts, no KaTeX JavaScript.
import katex from "katex";

export const macros = {
  "\\bh": "\\boldsymbol{h}",
  "\\btau": "\\boldsymbol{\\tau}",
  "\\bpi": "\\boldsymbol{\\pi}",
};

export function tex(source, display = false) {
  return katex.renderToString(source, { displayMode: display, throwOnError: true, output: "html", macros });
}
