import type { APIRoute } from "astro";
import { getNotebooks } from "../../../lib/notebooks.mjs";

export async function getStaticPaths() {
  return (await getNotebooks()).flatMap((notebook) =>
    notebook.figures.map((figure) => ({
      params: { slug: notebook.slug, figure: figure.name },
      props: { encoded: figure.encoded },
    })),
  );
}

export const GET: APIRoute = ({ props }) =>
  new Response(Buffer.from(props.encoded, "base64"), {
    headers: { "content-type": "image/png" },
  });
