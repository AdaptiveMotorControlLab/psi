# Project page

The source of the project page for the paper: [Astro](https://astro.build),
static output, with plain JavaScript and D3 for the interactive parts. The
numbers and example pairs it shows are in `src/data/*.json`, exported from the
experiments reported in the paper.

```bash
npm install
npm run dev          # http://localhost:4321
npm run build        # static site in dist/
```

The swap animation can also be saved for slides, and the page captured in both
themes. Both need a headless Chromium, installed once with
`npx playwright install chromium-headless-shell`; recording also needs `ffmpeg`.

```bash
npm run build && npm run record        # recordings/swap-<theme>.mp4, 1920x1080, 30 fps
npm run build && npm run screenshots   # screenshots/*.png
```

`.github/workflows/site.yml` builds this folder and publishes it to GitHub Pages.
