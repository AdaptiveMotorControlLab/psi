// "The swap": one interchange of the Psi subspace, drawn from a real test pair.
//
// Everything shown comes from src/data/hero.json, exported from the experiments
// reported in the paper:
// the two prompts, the model's reading of each over its four options, the
// recipient's reading after the interchange, and the layer-17 activations of
// all 200 test prompts in three directions, two inside the learned subspace
// and one outside it. In those coordinates the interchange is exact: the
// edited recipient has the donor's in-plane coordinates and keeps its own
// height above the plane.
//
// The picture is a pure function of time, `render(t)`, so a recorder can step
// through it frame by frame (the page exposes it as window.__psiSwap) and the
// page can drive it from requestAnimationFrame.
import { select } from "d3-selection";
import { easeCubicInOut } from "d3-ease";
import { interpolateRgb } from "d3-interpolate";

const SVG = "http://www.w3.org/2000/svg";
export const PERIOD = 9; // seconds per pair
const PHASES = { project: [1.1, 2.5], swap: [2.7, 4.3], read: [4.5, 5.9], outline: [5.9, 6.5], reset: [8.3, 9.0] };
const STEPS = [
  { at: 0, text: "Read both prompts' answers" },
  { at: PHASES.project[0], text: "Project the donor onto Ψ" },
  { at: PHASES.swap[0], text: "Swap in the donor's Ψ part" },
  { at: PHASES.read[0], text: "The recipient answers at the donor's position" },
];
const THETA = -0.62, SWAY = 0.28; // camera azimuth and its sway, radians
const PHI = 0.72;                 // camera elevation above the plane
const Z = 0.8;                    // vertical axis drawn at 0.8 of the in-plane scale
const HALF = 2.2;                 // plane drawn out to 2.2 standard deviations
const W = 480, H = 300, MARGIN = 14;

const clamp01 = (x) => Math.max(0, Math.min(1, x));
const phase = (t, [a, b]) => easeCubicInOut(clamp01((t - a) / (b - a)));
const lerp = (a, b, s) => a + (b - a) * s;
const argmax = (xs) => xs.reduce((best, x, i) => (x > xs[best] ? i : best), 0);
const css = (el, name) => getComputedStyle(el).getPropertyValue(name).trim();

function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function promptBlock(container, prompt) {
  container.replaceChildren();
  const lines = prompt.split("\n");
  let option = 0;
  for (const line of lines) {
    const div = document.createElement("div");
    div.className = "line";
    div.textContent = line;
    if (/^\S{1,2}\. /.test(line)) div.dataset.position = String(option++);
    container.appendChild(div);
  }
}

function bars(svg, side) {
  // viewBox 0 0 220 112: four slots, baseline at y = 86.
  const g = el("g", {}, svg);
  const items = [0, 1, 2, 3].map((i) => {
    const x = 20 + i * 50;
    return {
      outline: el("rect", { x, width: 30, rx: 2, class: "bar-outline" }, g),
      bar: el("rect", { x, width: 30, rx: 2, class: `bar ${side}` }, g),
      value: el("text", { x: x + 15, class: "bar-value", "text-anchor": "middle" }, g),
      label: el("text", { x: x + 15, y: 104, class: "bar-label", "text-anchor": "middle" }, g),
    };
  });
  el("line", { x1: 10, x2: 210, y1: 86, y2: 86, class: "baseline" }, g);
  return items;
}

// Push overlapping labels apart vertically; earlier labels keep their place.
function separate(labels) {
  const boxes = labels.filter((l) => l.visible).map((l) => {
    const w = l.node.textContent.length * l.size * 0.6;
    const x0 = l.anchor === "end" ? l.x - w : l.anchor === "middle" ? l.x - w / 2 : l.x;
    return { ...l, x0, x1: x0 + w };
  });
  for (let i = 1; i < boxes.length; i++) {
    for (let pass = 0; pass < 3; pass++) {
      for (let j = 0; j < i; j++) {
        const a = boxes[j], b = boxes[i];
        const overlapX = Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0);
        const overlapY = Math.min(a.y, b.y) - Math.max(a.y - a.size, b.y - b.size) + 2;
        if (overlapX > 0 && overlapY > 0) b.y += b.y >= a.y ? overlapY : -overlapY;
      }
    }
  }
  for (const b of boxes) { b.node.setAttribute("x", b.x); b.node.setAttribute("y", b.y); b.node.setAttribute("text-anchor", b.anchor); }
}

function setBars(items, reading, labels, { outline = null, outlineOpacity = 0 } = {}) {
  items.forEach((it, i) => {
    const h = 66 * reading[i];
    it.bar.setAttribute("y", 86 - h);
    it.bar.setAttribute("height", Math.max(h, 0.5));
    it.value.setAttribute("y", 86 - h - 5);
    it.value.textContent = `${Math.round(100 * reading[i])}%`;
    it.label.textContent = labels[i];
    if (outline) {
      const ho = 66 * outline[i];
      it.outline.setAttribute("y", 86 - ho);
      it.outline.setAttribute("height", Math.max(ho, 0.5));
    }
    it.outline.style.opacity = outlineOpacity;
  });
}

export function createSwap(root, data, options = {}) {
  const pairs = data.pairs;
  const setup = data.setup;
  const reduced = options.reducedMotion ?? matchMedia("(prefers-reduced-motion: reduce)").matches;
  let fixedPair = options.pair ?? null;
  // Math labels arrive pre-rendered (KaTeX at build time); plain text otherwise.
  const tx = { s: "s", b: "b", qs: "q(s)", qb: "q(b)", qbs: "q(h_b←s)", kl: "KL", ...(options.labels ?? {}) };

  // ---------------------------------------------------------------- layout
  root.classList.add("swap");
  root.innerHTML = `
    <div class="swap-card donor">
      <div class="card-head"><span class="role donor">donor ${tx.s}</span><span class="answer"></span></div>
      <div class="prompt mono"></div>
      <svg class="bars" viewBox="0 0 220 112" role="img" aria-label="Donor output distribution"></svg>
      <div class="bars-caption"><span>${tx.qs}</span></div>
    </div>
    <div class="swap-stage">
      <svg class="space" viewBox="0 0 480 300" role="img"
        aria-label="Layer-${setup.layer} activations of the test prompts, two directions inside the Psi subspace and one outside it"></svg>
      <ol class="steps"></ol>
    </div>
    <div class="swap-card recipient">
      <div class="card-head"><span class="role recipient">recipient ${tx.b}</span><span class="answer"></span></div>
      <div class="prompt mono"></div>
      <svg class="bars" viewBox="0 0 220 112" role="img" aria-label="Recipient output distribution"></svg>
      <div class="bars-caption"><span class="cap-before">${tx.qb}</span><span class="cap-after">${tx.qbs}</span></div>
      <div class="kl"></div>
    </div>`;
  const donorCard = root.querySelector(".swap-card.donor");
  const recipientCard = root.querySelector(".swap-card.recipient");
  const donorBars = bars(donorCard.querySelector("svg"), "donor");
  const recipientBars = bars(recipientCard.querySelector("svg"), "recipient");
  const stepList = root.querySelector(".steps");
  const stepItems = STEPS.map((s, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="num">${i + 1}.</span><span class="text">${options.steps?.[i] ?? s.text}</span>`;
    stepList.appendChild(li);
    return li;
  });

  // ---------------------------------------------------------------- 3D view
  const space = root.querySelector("svg.space");
  const pts = data.points; // [x, y, z, answer position], each axis in standard deviations
  const box = { x0: -HALF, x1: HALF, y0: -HALF, y1: HALF };
  const camera = { theta: THETA, phi: PHI };
  function raw([x, y, z], theta) {
    const c = Math.cos(theta), s = Math.sin(theta);
    const xr = x * c - y * s, yr = x * s + y * c;
    return [xr, -(Z * z * Math.cos(camera.phi) + yr * Math.sin(camera.phi)), yr];
  }
  // Fit everything the camera will ever show into the view box, once.
  const view = (() => {
    const sample = [...pts.map((p) => p.slice(0, 3)),
      ...[[box.x0, box.y0], [box.x1, box.y0], [box.x1, box.y1], [box.x0, box.y1]].map(([x, y]) => [x, y, 0]),
      ...pairs.flatMap((p) => [p.recipient.point, p.donor.point, p.edited.point])];
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i <= 20; i++) {
      const theta = THETA - SWAY + (2 * SWAY * i) / 20;
      for (const q of sample) {
        const [a, b] = raw(q, theta);
        x0 = Math.min(x0, a); x1 = Math.max(x1, a); y0 = Math.min(y0, b); y1 = Math.max(y1, b);
      }
    }
    const s = Math.min((W - 2 * MARGIN) / (x1 - x0), (H - 2 * MARGIN) / (y1 - y0));
    return { s, cx: W / 2 - (s * (x0 + x1)) / 2, cy: H / 2 - (s * (y0 + y1)) / 2 };
  })();
  function project(q) {
    const [a, b, d] = raw(q, camera.theta);
    return [view.cx + view.s * a, view.cy + view.s * b, d];
  }
  const gPlane = el("g", {}, space);
  const plane = el("path", { class: "plane" }, gPlane);
  const planeLabel = el("text", { class: "plane-label" }, gPlane);
  planeLabel.textContent = `Ψ subspace (2 of ${setup.rank} directions)`;
  const gBelow = el("g", { class: "cloud below" }, space);
  const gAbove = el("g", { class: "cloud above" }, space);
  const letters = "ABCD";
  const cloud = pts.map((p) => {
    const t = el("text", { "text-anchor": "middle", "dominant-baseline": "central" }, p[2] < 0 ? gBelow : gAbove);
    t.textContent = p[3] >= 0 ? letters[p[3]] : "·";
    return t;
  });
  // Plane is drawn between the two halves of the cloud so it occludes the lower one.
  space.insertBefore(gBelow, gPlane);
  const axis = el("g", { class: "axis" }, space);
  const axisLine = el("line", {}, axis);
  const axisLabel = el("text", {}, axis);
  for (const [i, text] of ["outside Ψ", `(1 of ${setup.activation_dim - setup.rank})`].entries()) {
    const span = el("tspan", { dy: i ? "1.15em" : "0" }, axisLabel);
    span.textContent = text;
  }
  const gMarks = el("g", { class: "marks" }, space);
  const mk = {
    rDrop: el("line", { class: "drop recipient" }, gMarks),
    dDrop: el("line", { class: "drop donor" }, gMarks),
    eDrop: el("line", { class: "drop edited" }, gMarks),
    trail: el("line", { class: "trail" }, gMarks),
    rFoot: el("circle", { r: 3.5, class: "foot recipient" }, gMarks),
    dFoot: el("circle", { r: 4.5, class: "foot donor" }, gMarks),
    ghost: el("circle", { r: 6.5, class: "ghost" }, gMarks),
    donor: el("circle", { r: 7, class: "point donor" }, gMarks),
    recipient: el("circle", { r: 7, class: "point recipient" }, gMarks),
    dLabel: el("text", { class: "point-label donor" }, gMarks),
    rLabel: el("text", { class: "point-label recipient" }, gMarks),
    footLabel: el("text", { class: "foot-label" }, gMarks),
  };
  mk.dLabel.textContent = "donor";
  mk.rLabel.textContent = "recipient";
  mk.footLabel.textContent = "donor's Ψ part";

  // ---------------------------------------------------------------- state
  let shownPair = -1;
  let colours = null;
  function readColours() {
    colours = { recipient: css(root, "--recipient"), psi: css(root, "--psi") };
  }
  function showPair(k) {
    if (k === shownPair) return;
    shownPair = k;
    const p = pairs[k];
    promptBlock(donorCard.querySelector(".prompt"), p.donor.prompt);
    promptBlock(recipientCard.querySelector(".prompt"), p.recipient.prompt);
    recipientCard.querySelector(".kl").innerHTML =
      `<span class="legend-outline"></span><span>${tx.kl} = ${p.kl_display}</span>`;
    root.querySelectorAll(".pair-toggle button").forEach((b) =>
      b.setAttribute("aria-pressed", String(fixedPair === null ? false : Number(b.dataset.pair) === k)));
    root.dispatchEvent(new CustomEvent("swap:pair", { detail: k }));
  }

  function highlight(card, position, kind) {
    card.querySelectorAll(".line[data-position]").forEach((line) => {
      line.classList.toggle(`hl-${kind}`, Number(line.dataset.position) === position);
      for (const other of ["donor", "recipient", "edited"]) if (other !== kind) line.classList.remove(`hl-${other}`);
    });
  }

  function render(time) {
    if (!colours) readColours();
    const k = fixedPair ?? Math.floor(time / PERIOD) % pairs.length;
    const t = ((time % PERIOD) + PERIOD) % PERIOD;
    showPair(k);
    const p = pairs[k];
    const reset = phase(t, PHASES.reset);
    const pProject = phase(t, PHASES.project) * (1 - reset);
    const pSwap = phase(t, PHASES.swap) * (1 - reset);
    const pRead = phase(t, PHASES.read) * (1 - reset);
    const pOutline = phase(t, PHASES.outline) * (1 - reset);
    // When pairs alternate, the outgoing pair fades out and the next one fades in.
    const cycling = fixedPair === null && pairs.length > 1;
    const fadeOut = cycling ? clamp01((t - 8.6) / 0.4) : 0;
    const fadeIn = cycling && time >= PERIOD ? clamp01(t / 0.35) : 1;
    root.style.setProperty("--pair-fade", String(Math.min(1 - fadeOut, fadeIn)));

    // camera: a slow sway, continuous across pairs
    camera.theta = THETA + (reduced ? 0 : SWAY * Math.sin((2 * Math.PI * time) / (2 * PERIOD)));

    const corners = [[box.x0, box.y0], [box.x1, box.y0], [box.x1, box.y1], [box.x0, box.y1]].map(([x, y]) => project([x, y, 0]));
    plane.setAttribute("d", `M${corners.map((c) => `${c[0].toFixed(1)},${c[1].toFixed(1)}`).join("L")}Z`);
    const nearest = corners.reduce((a, c) => (c[1] > a[1] ? c : a));
    planeLabel.setAttribute("x", nearest[0]);
    planeLabel.setAttribute("y", Math.min(nearest[1] + 14, H - 3));
    planeLabel.setAttribute("text-anchor", nearest[0] > 0.7 * W ? "end" : nearest[0] < 0.3 * W ? "start" : "middle");
    // The out-of-plane axis stands on the leftmost corner of the plane.
    const far = corners.reduce((a, c) => (c[0] < a[0] ? c : a));
    const rise = view.s * 1.4 * Z * Math.cos(camera.phi);
    axisLine.setAttribute("x1", far[0]); axisLine.setAttribute("y1", far[1]);
    axisLine.setAttribute("x2", far[0]); axisLine.setAttribute("y2", far[1] - rise);
    axisLabel.setAttribute("y", far[1] - rise - 26);
    axisLabel.setAttribute("text-anchor", "middle");
    for (const span of axisLabel.children) span.setAttribute("x", far[0]);

    cloud.forEach((node, i) => {
      const [x, y] = project(pts[i]);
      node.setAttribute("x", x.toFixed(1));
      node.setAttribute("y", y.toFixed(1));
    });

    const R = p.recipient.point, D = p.donor.point;
    const E = [lerp(R[0], D[0], pSwap), lerp(R[1], D[1], pSwap), R[2]];
    const r = project(R), d = project(D), e = project(E);
    const rf = project([R[0], R[1], 0]), df = project([D[0], D[1], 0]), ef = project([E[0], E[1], 0]);
    const line = (n, a, b) => { n.setAttribute("x1", a[0]); n.setAttribute("y1", a[1]); n.setAttribute("x2", b[0]); n.setAttribute("y2", b[1]); };
    const at = (n, a) => { n.setAttribute("cx", a[0]); n.setAttribute("cy", a[1]); };
    line(mk.rDrop, r, rf);
    mk.rDrop.style.opacity = 0.55 * (1 - pSwap);
    // donor's drop line grows from the point to the plane during "project"
    line(mk.dDrop, d, [lerp(d[0], df[0], pProject), lerp(d[1], df[1], pProject)]);
    mk.dDrop.style.opacity = pProject > 0 ? 0.9 : 0;
    at(mk.dFoot, df); mk.dFoot.style.opacity = clamp01(pProject * 2 - 1);
    const footAlpha = clamp01(pProject * 2 - 1) * (1 - pRead);
    mk.footLabel.style.opacity = footAlpha;
    at(mk.rFoot, rf); mk.rFoot.style.opacity = 0.7 * (1 - pSwap);
    line(mk.trail, rf, ef); mk.trail.style.opacity = pSwap > 0 ? 0.9 * (1 - pOutline * 0.6) : 0;
    line(mk.eDrop, e, ef); mk.eDrop.style.opacity = pSwap > 0 ? 0.8 : 0;
    at(mk.ghost, r); mk.ghost.style.opacity = pSwap > 0 ? 0.6 * Math.min(1, pSwap * 3) : 0;
    at(mk.donor, d);
    at(mk.recipient, e);
    mk.recipient.style.fill = interpolateRgb(colours.recipient, colours.psi)(pSwap);
    mk.rLabel.textContent = pSwap > 0.5 ? "edited recipient" : "recipient";
    mk.rLabel.classList.toggle("edited", pSwap > 0.5);
    // donor labelled on its right, recipient on its left, the projected point below it
    separate([
      { node: mk.dLabel, x: d[0] + 11, y: d[1] + 4, anchor: "start", size: 12, visible: true },
      { node: mk.rLabel, x: e[0] - 11, y: e[1] + 4, anchor: "end", size: 12, visible: true },
      { node: mk.footLabel, x: df[0], y: df[1] + 17, anchor: "middle", size: 11, visible: footAlpha > 0 },
    ]);

    // bars
    const shown = p.recipient.reading.map((v, i) => lerp(v, p.edited.reading[i], pRead));
    setBars(donorBars, p.donor.reading, p.donor.labels);
    setBars(recipientBars, shown, p.recipient.labels, { outline: p.donor.reading, outlineOpacity: pOutline });
    const mix = interpolateRgb(colours.recipient, colours.psi)(pRead);
    recipientBars.forEach((b) => (b.bar.style.fill = mix));
    const dTop = argmax(p.donor.reading), rTop = argmax(shown);
    donorCard.querySelector(".answer").innerHTML = `answers <b>${p.donor.labels[dTop]}</b>`;
    recipientCard.querySelector(".answer").innerHTML =
      pRead > 0.5 ? `now answers <b class="edited">${p.recipient.labels[rTop]}</b>` : `answers <b>${p.recipient.labels[rTop]}</b>`;
    highlight(donorCard, dTop, "donor");
    highlight(recipientCard, rTop, pRead > 0.5 ? "edited" : "recipient");
    recipientCard.querySelector(".kl").style.opacity = pOutline;
    recipientCard.classList.toggle("read", pRead > 0.5);

    let active = 0;
    STEPS.forEach((s, i) => { if (t >= s.at && t < PHASES.reset[0]) active = i; });
    if (t >= PHASES.reset[0]) active = 3;
    stepItems.forEach((li, i) => { li.classList.toggle("active", i === active); li.classList.toggle("done", i < active); });
  }

  // ---------------------------------------------------------------- playback
  let playing = false, start = 0, offset = options.time ?? (reduced ? 7 : 0), frame = 0;
  const now = () => performance.now() / 1000;
  function tick() {
    render(now() - start);
    frame = requestAnimationFrame(tick);
  }
  function play() {
    if (playing) return;
    playing = true; start = now() - offset;
    frame = requestAnimationFrame(tick);
    root.dispatchEvent(new CustomEvent("swap:state", { detail: true }));
  }
  function pause() {
    if (!playing) return;
    playing = false; offset = now() - start;
    cancelAnimationFrame(frame);
    root.dispatchEvent(new CustomEvent("swap:state", { detail: false }));
  }
  function setPair(k) {
    fixedPair = k;
    // restart the chosen pair from its beginning
    offset = (k ?? 0) * PERIOD; start = now() - offset;
    shownPair = -1;
    render(offset);
  }
  // theme switches change the colours the interpolations start from
  new MutationObserver(() => { colours = null; render(playing ? now() - start : offset); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { colours = null; });

  render(offset);
  return { render, play, pause, setPair, get playing() { return playing; }, period: PERIOD, pairs: pairs.length };
}
