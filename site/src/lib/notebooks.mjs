// Read committed notebook JSON at build time. Runtime labels come from its run record.
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import sanitizeHtml from "sanitize-html";
import { createMarkdownProcessor } from "@astrojs/markdown-remark";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const text = (value) =>
  Array.isArray(value) ? value.join("") : String(value ?? "");
const mathTags = [
  "math",
  "semantics",
  "annotation",
  "mrow",
  "mi",
  "mo",
  "mn",
  "ms",
  "mtext",
  "mspace",
  "msup",
  "msub",
  "msubsup",
  "mfrac",
  "mroot",
  "msqrt",
  "mtable",
  "mtr",
  "mtd",
  "mover",
  "munder",
  "munderover",
  "mpadded",
  "menclose",
  "mstyle",
];
export function cleanHtml(html) {
  return sanitizeHtml(html, {
    allowedTags: [
      ...sanitizeHtml.defaults.allowedTags,
      "img",
      ...mathTags,
      "svg",
      "path",
      "line",
      "rect",
      "polyline",
      "polygon",
    ],
    allowedAttributes: {
      "*": ["class", "aria-hidden"],
      a: ["href", "title"],
      img: ["src", "alt", "width", "height"],
      th: ["colspan", "rowspan"],
      td: ["colspan", "rowspan"],
      math: ["xmlns", "display"],
      annotation: ["encoding"],
      svg: ["xmlns", "width", "height", "viewBox", "preserveAspectRatio"],
      path: ["d", "fill", "stroke"],
      line: ["x1", "x2", "y1", "y2", "stroke"],
      rect: ["x", "y", "width", "height", "fill"],
      polyline: ["points", "fill", "stroke"],
      polygon: ["points", "fill", "stroke"],
      span: ["class", "style", "aria-hidden"],
    },
    allowedStyles: {
      "*": {
        height: [/^-?[\d.]+em$/],
        width: [/^-?[\d.]+em$/],
        "min-width": [/^[\d.]+em$/],
        "padding-left": [/^[\d.]+em$/],
        top: [/^-?[\d.]+em$/],
        left: [/^-?[\d.]+em$/],
        "vertical-align": [/^-?[\d.]+em$/],
        "margin-left": [/^-?[\d.]+em$/],
        "margin-right": [/^-?[\d.]+em$/],
        "border-bottom-width": [/^[\d.]+em$/],
        position: [/^relative$/],
      },
    },
    parser: { lowerCaseAttributeNames: false },
    allowedSchemes: ["https", "http", "mailto"],
    allowedSchemesByTag: { img: ["https", "http", "data"] },
    disallowedTagsMode: "discard",
  });
}

let pending;
export function getNotebooks() {
  pending ??= load();
  return pending;
}

async function load() {
  const directory = path.resolve(
    process.cwd(),
    process.env.NOTEBOOKS_DIR || "../notebooks",
  );
  const names = (await readdir(directory))
    .filter((name) => name.endsWith(".ipynb"))
    .sort();
  const markdown = await createMarkdownProcessor({
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  });
  return Promise.all(
    names.map(async (filename) => {
      const notebook = JSON.parse(
        await readFile(path.join(directory, filename), "utf8"),
      );
      const slug = filename.slice(0, -6);
      if (!/^[a-zA-Z0-9-]+$/.test(slug))
        throw new Error(`Invalid notebook filename: ${filename}`);
      const meta = notebook.metadata?.psi || {};
      const run = meta.run || {};
      let lastImage, markedImage;
      const cells = [];
      const figures = [];
      for (const cell of notebook.cells || []) {
        if (cell.cell_type === "markdown") {
          const rendered = await markdown.render(text(cell.source));
          cells.push({ type: "markdown", html: cleanHtml(rendered.code) });
        } else if (cell.cell_type === "code") {
          const outputs = [];
          for (const output of cell.outputs || []) {
            const data = output.data || {};
            if (data["image/png"]) {
              const encoded = text(data["image/png"]).replace(/\s/g, "");
              if (!/^[A-Za-z0-9+/]*={0,2}$/.test(encoded))
                throw new Error(`Invalid PNG output in ${filename}`);
              const name = `figure-${figures.length + 1}`;
              const src = `/demos/${slug}/${name}.png`;
              figures.push({ name, encoded });
              outputs.push({ type: "image", src });
              lastImage = src;
              if (cell.metadata?.psi?.card) markedImage = src;
            } else if (data["text/html"]) {
              outputs.push({
                type: "html",
                html: cleanHtml(text(data["text/html"])),
              });
            } else if (data["text/plain"]) {
              outputs.push({ type: "text", text: text(data["text/plain"]) });
            } else if (output.output_type === "stream") {
              outputs.push({ type: "text", text: text(output.text) });
            } else if (data["application/json"]) {
              outputs.push({
                type: "text",
                text: JSON.stringify(data["application/json"], null, 2),
              });
            }
            if (output.output_type === "error")
              throw new Error(
                `Notebook ${filename} contains a failed execution.`,
              );
          }
          cells.push({ type: "code", source: text(cell.source), outputs });
        }
      }
      if (!markedImage && !lastImage)
        throw new Error(
          `Notebook ${filename} has no PNG figure output. Execute it with an inline plotting backend.`,
        );
      if (!run.model || !run.gpu || !Number.isFinite(run.runtime_seconds))
        throw new Error(`Notebook ${filename} has no computed run record.`);
      const card = run.card ?? null;
      if (
        card &&
        !(
          Array.isArray(card.rows) &&
          card.rows.length > 0 &&
          card.rows.every(
            (row) =>
              typeof row.label === "string" &&
              [row.edit, row.baseline].every((v) => Number.isFinite(v) && v >= 0 && v <= 1),
          )
        )
      )
        throw new Error(`Notebook ${filename} has card scores outside [0, 1] or without labels.`);
      return {
        slug,
        filename,
        title: meta.title || slug,
        description: meta.description || "",
        cells,
        figures,
        image: markedImage || lastImage,
        card,
        model: run.model || "Run record pending",
        gpu: run.gpu || "Hardware pending",
        runtime:
          typeof run.runtime_seconds === "number"
            ? `${(run.runtime_seconds / 60).toFixed(1)} min`
            : "Runtime pending",
        smoke: run.smoke,
        colab: `https://colab.research.google.com/github/AdaptiveMotorControlLab/psi/blob/main/notebooks/${filename}`,
        github: `https://github.com/AdaptiveMotorControlLab/psi/blob/main/notebooks/${filename}`,
      };
    }),
  );
}
