(function () {
  "use strict";

  const mainCanvas = document.getElementById("fm-main-canvas");
  const sampleCanvas = document.getElementById("fm-sample-canvas");
  if (!mainCanvas || !sampleCanvas) return;

  const $ = (id) => document.getElementById(id);
  const ui = {
    time: $("fm-time"), timeOutput: $("fm-time-output"), timeBadge: $("fm-time-badge"), play: $("fm-play"),
    timeStart: $("fm-time-start"), timeEnd: $("fm-time-end"), timeSymbol: $("fm-time-symbol"),
    paths: $("fm-show-paths"), targets: $("fm-show-targets"), field: $("fm-show-field"),
    pathsLabel: $("fm-paths-label"), targetsLabel: $("fm-targets-label"), fieldLabel: $("fm-field-label"),
    bandwidth: $("fm-bandwidth"), bandwidthOutput: $("fm-bandwidth-output"), seed: $("fm-seed"), reseed: $("fm-reseed"),
    bandwidthHelp: $("fm-bandwidth-help"),
    pairControl: $("fm-pair-control"), tracePair: $("fm-trace-pair"), pairNumber: $("fm-pair-number"),
    stageKicker: $("fm-stage-kicker"), stageTitle: $("fm-stage-title"), stageDescription: $("fm-stage-description"),
    hint: $("fm-canvas-hint"), pairCount: $("fm-pair-count"), statALabel: $("fm-stat-a-label"), statA: $("fm-stat-a"),
    statBLabel: $("fm-stat-b-label"), statB: $("fm-stat-b"), reading: $("fm-reading"),
    steps: $("fm-steps"), stepsOutput: $("fm-steps-output"), generate: $("fm-generate"), resetSamples: $("fm-reset-samples"),
    sampleStatus: $("fm-sample-status")
  };

  const palette = { ink: "#17211d", grid: "#dfe6e0", source: "#7857b2", target: "#e36d3d", flow: "#167d69", flowDark: "#095c4d", paper: "#fbfcf8" };
  const state = {
    seed: 20260719, shape: "eight", stage: "diffuse", t: 0.55, bandwidth: 0.34,
    diffusionT: 0.55, flowT: 0.35,
    pairs: [], targetReference: [], selected: { x: 0, y: 0 }, highlightedPair: 0, playing: false, playFrame: 0,
    sampleOrigin: [], samples: [], sampleTime: 0, sampleStep: 0, sampleSteps: 16, sampleFrame: 0, sampling: false
  };

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussian(rng) {
    const u = Math.max(1e-9, rng());
    const v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function sourcePoint(rng) {
    return { x: gaussian(rng) * 0.95, y: gaussian(rng) * 0.95 };
  }

  function targetPoint(rng, shape) {
    if (shape === "moons") {
      const upper = rng() < 0.5;
      const angle = rng() * Math.PI;
      const noise = 0.07;
      if (upper) return { x: 1.55 * Math.cos(angle) - 0.45 + gaussian(rng) * noise, y: 1.35 * Math.sin(angle) - 0.35 + gaussian(rng) * noise };
      return { x: 1.55 * (1 - Math.cos(angle)) - 1.05 + gaussian(rng) * noise, y: -1.35 * Math.sin(angle) + 0.48 + gaussian(rng) * noise };
    }
    if (shape === "spiral") {
      const arm = rng() < 0.5 ? 0 : Math.PI;
      const s = 0.12 + 0.88 * Math.sqrt(rng());
      const angle = arm + s * Math.PI * 2.25;
      const radius = 0.35 + 2.35 * s;
      return { x: radius * Math.cos(angle) + gaussian(rng) * 0.07, y: radius * Math.sin(angle) + gaussian(rng) * 0.07 };
    }
    const mode = Math.floor(rng() * 8);
    const angle = mode * Math.PI / 4;
    return { x: 2.15 * Math.cos(angle) + gaussian(rng) * 0.16, y: 2.15 * Math.sin(angle) + gaussian(rng) * 0.16 };
  }

  function rebuild() {
    clearTimeout(state.sampleFrame);
    state.sampling = false;
    const rng = mulberry32((state.seed ^ ({ eight: 11, moons: 29, spiral: 47 }[state.shape])) >>> 0);
    state.pairs = [];
    state.targetReference = [];
    for (let i = 0; i < 420; i += 1) {
      const a = sourcePoint(rng);
      const b = targetPoint(rng, state.shape);
      state.pairs.push({ a, b, vx: b.x - a.x, vy: b.y - a.y });
      state.targetReference.push(b);
    }
    state.highlightedPair = state.seed % state.pairs.length;
    ui.pairNumber.value = state.highlightedPair + 1;
    resetSamples();
    updateStats();
    drawAll();
  }

  function resizeCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width; canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: rect.width, h: rect.height };
  }

  function viewport(w, h) {
    const pad = Math.max(30, Math.min(48, Math.min(w, h) * 0.08));
    const size = Math.min(w - pad * 2, h - pad * 2);
    return { left: (w - size) / 2, top: (h - size) / 2, size, min: -3.35, max: 3.35 };
  }

  function screen(p, vp) {
    const scale = vp.size / (vp.max - vp.min);
    return { x: vp.left + (p.x - vp.min) * scale, y: vp.top + (vp.max - p.y) * scale };
  }

  function world(x, y, vp) {
    const scale = (vp.max - vp.min) / vp.size;
    return { x: vp.min + (x - vp.left) * scale, y: vp.max - (y - vp.top) * scale };
  }

  function drawBackground(ctx, w, h, vp, label) {
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#f7f9f5";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = palette.grid;
    ctx.lineWidth = 1;
    for (let n = -3; n <= 3; n += 1) {
      const a = screen({ x: n, y: vp.min }, vp);
      const b = screen({ x: n, y: vp.max }, vp);
      const c = screen({ x: vp.min, y: n }, vp);
      const d = screen({ x: vp.max, y: n }, vp);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.moveTo(c.x, c.y); ctx.lineTo(d.x, d.y); ctx.stroke();
    }
    ctx.strokeStyle = "#adbbb3";
    ctx.strokeRect(vp.left, vp.top, vp.size, vp.size);
    if (label) {
      ctx.fillStyle = "#637068";
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillText(label, vp.left + 8, vp.top + 17);
    }
  }

  function lerpPair(pair, t) {
    return { x: pair.a.x * (1 - t) + pair.b.x * t, y: pair.a.y * (1 - t) + pair.b.y * t };
  }

  function diffusePair(pair, t) {
    const dataScale = Math.sqrt(Math.max(0, 1 - t));
    const noiseScale = Math.sqrt(Math.max(0, t));
    return { x: pair.b.x * dataScale + pair.a.x * noiseScale, y: pair.b.y * dataScale + pair.a.y * noiseScale };
  }

  function arrow(ctx, from, vx, vy, vp, color, alpha, scale, width) {
    const start = screen(from, vp);
    const end = screen({ x: from.x + vx * scale, y: from.y + vy * scale }, vp);
    const dx = end.x - start.x, dy = end.y - start.y;
    const length = Math.hypot(dx, dy);
    if (length < 1) return;
    const head = Math.min(7, Math.max(3, length * 0.3));
    const angle = Math.atan2(dy, dx);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(start.x, start.y); ctx.lineTo(end.x, end.y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(end.x, end.y); ctx.lineTo(end.x - head * Math.cos(angle - 0.55), end.y - head * Math.sin(angle - 0.55));
    ctx.lineTo(end.x - head * Math.cos(angle + 0.55), end.y - head * Math.sin(angle + 0.55)); ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  function fieldAt(x, y, t, bandwidth) {
    const h2 = bandwidth * bandwidth;
    let sum = 0, vx = 0, vy = 0, spread = 0;
    const nearby = [];
    for (let i = 0; i < state.pairs.length; i += 1) {
      const pair = state.pairs[i];
      const p = lerpPair(pair, t);
      const dx = p.x - x, dy = p.y - y;
      const d2 = dx * dx + dy * dy;
      if (d2 > h2 * 12) continue;
      const weight = Math.exp(-d2 / (2 * h2));
      sum += weight; vx += weight * pair.vx; vy += weight * pair.vy;
      nearby.push({ pair, p, weight, d2 });
    }
    if (sum < 1e-7) return { vx: 0, vy: 0, confidence: 0, spread: 0, nearby: [] };
    vx /= sum; vy /= sum;
    for (let i = 0; i < nearby.length; i += 1) {
      const dx = nearby[i].pair.vx - vx, dy = nearby[i].pair.vy - vy;
      spread += nearby[i].weight * (dx * dx + dy * dy);
    }
    return { vx, vy, confidence: Math.min(1, sum / 7), spread: Math.sqrt(spread / sum), nearby };
  }

  function denoiserAt(x, y, t, bandwidth) {
    const h2 = bandwidth * bandwidth;
    let sum = 0, vx = 0, vy = 0, spread = 0;
    const nearby = [];
    for (let i = 0; i < state.pairs.length; i += 1) {
      const pair = state.pairs[i];
      const p = diffusePair(pair, t);
      const dx = p.x - x, dy = p.y - y;
      const d2 = dx * dx + dy * dy;
      if (d2 > h2 * 12) continue;
      const weight = Math.exp(-d2 / (2 * h2));
      const targetX = pair.b.x - p.x, targetY = pair.b.y - p.y;
      sum += weight; vx += weight * targetX; vy += weight * targetY;
      nearby.push({ pair, p, weight, d2, targetX, targetY });
    }
    if (sum < 1e-7) return { vx: 0, vy: 0, confidence: 0, spread: 0, nearby: [] };
    vx /= sum; vy /= sum;
    for (let i = 0; i < nearby.length; i += 1) {
      const dx = nearby[i].targetX - vx, dy = nearby[i].targetY - vy;
      spread += nearby[i].weight * (dx * dx + dy * dy);
    }
    return { vx, vy, confidence: Math.min(1, sum / 7), spread: Math.sqrt(spread / sum), nearby };
  }

  function drawField(ctx, vp, t) {
    const count = vp.size < 440 ? 10 : 13;
    const gap = (vp.max - vp.min) / count;
    for (let iy = 0; iy <= count; iy += 1) {
      for (let ix = 0; ix <= count; ix += 1) {
        const x = vp.min + gap * (ix + 0.5), y = vp.min + gap * (iy + 0.5);
        if (x > vp.max || y > vp.max) continue;
        const f = state.stage === "diffuse" ? denoiserAt(x, y, t, state.bandwidth) : fieldAt(x, y, t, state.bandwidth);
        const mag = Math.hypot(f.vx, f.vy);
        if (f.confidence < 0.05 || mag < 0.03) continue;
        const cap = Math.min(0.33, 0.25 / Math.max(0.01, mag));
        arrow(ctx, { x, y }, f.vx, f.vy, vp, palette.flowDark, 0.18 + f.confidence * 0.62, cap, 1.3);
      }
    }
  }

  function drawDiffusion(ctx, vp, t) {
    if (ui.paths.checked) {
      ctx.strokeStyle = "rgba(92, 105, 98, 0.16)";
      ctx.lineWidth = 0.65;
      for (let i = 0; i < state.pairs.length; i += 4) {
        ctx.beginPath();
        for (let step = 0; step <= 12; step += 1) {
          const p = screen(diffusePair(state.pairs[i], step / 12), vp);
          if (step === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
      }
    }

    ctx.fillStyle = palette.target;
    ctx.globalAlpha = 0.17;
    state.targetReference.forEach((point) => {
      const p = screen(point, vp); ctx.beginPath(); ctx.arc(p.x, p.y, 2.3, 0, Math.PI * 2); ctx.fill();
    });
    ctx.globalAlpha = 1;

    if (ui.field.checked) drawField(ctx, vp, t);
    if (ui.targets.checked && t > 0.005) {
      for (let i = 0; i < state.pairs.length; i += 8) {
        const pair = state.pairs[i], p = diffusePair(pair, t);
        arrow(ctx, p, pair.b.x - p.x, pair.b.y - p.y, vp, palette.target, 0.52, 0.12, 1);
      }
    }

    const pointColor = t < 0.5 ? palette.target : palette.source;
    ctx.fillStyle = pointColor;
    ctx.globalAlpha = 0.8;
    state.pairs.forEach((pair) => {
      const p = screen(diffusePair(pair, t), vp); ctx.beginPath(); ctx.arc(p.x, p.y, 2.15, 0, Math.PI * 2); ctx.fill();
    });
    ctx.globalAlpha = 1;
  }

  function drawHighlightedPair(ctx, vp, t) {
    const pair = state.pairs[state.highlightedPair % state.pairs.length];
    if (!pair) return;
    const a = screen(pair.a, vp), b = screen(pair.b, vp), current = lerpPair(pair, t), c = screen(current, vp);
    ctx.save();
    ctx.strokeStyle = palette.ink; ctx.lineWidth = 2.2;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    ctx.fillStyle = palette.source; ctx.strokeStyle = palette.paper; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(a.x, a.y, 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = palette.target;
    ctx.beginPath(); ctx.arc(b.x, b.y, 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = palette.flow; ctx.strokeStyle = palette.paper;
    ctx.beginPath(); ctx.arc(c.x, c.y, 6.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    arrow(ctx, current, pair.vx, pair.vy, vp, palette.flowDark, 1, 0.16, 2.6);
    ctx.font = "bold 11px system-ui, sans-serif";
    ctx.fillStyle = palette.source; ctx.fillText("random noise", a.x + 9, a.y - 8);
    ctx.fillStyle = palette.target; ctx.fillText("random data example", b.x + 9, b.y - 8);
    ctx.restore();
  }

  function drawMain() {
    const { ctx, w, h } = resizeCanvas(mainCanvas);
    const vp = viewport(w, h);
    const t = state.t;
    const label = state.stage === "diffuse" ? "forward diffusion · generation runs in reverse" : state.stage === "regress" ? "click to inspect the local regression target" : "x-space";
    drawBackground(ctx, w, h, vp, label);

    if (state.stage === "diffuse") {
      drawDiffusion(ctx, vp, t);
      return;
    }

    if (ui.paths.checked) {
      ctx.lineWidth = 0.65;
      for (let i = 0; i < state.pairs.length; i += 3) {
        const pair = state.pairs[i], a = screen(pair.a, vp), b = screen(pair.b, vp);
        ctx.strokeStyle = "rgba(92, 105, 98, 0.18)";
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
    }

    if (ui.field.checked || state.stage === "regress") drawField(ctx, vp, t);

    if (ui.targets.checked) {
      const stride = state.stage === "regress" ? 5 : 8;
      for (let i = 0; i < state.pairs.length; i += stride) {
        const pair = state.pairs[i], p = lerpPair(pair, t);
        arrow(ctx, p, pair.vx, pair.vy, vp, palette.target, state.stage === "regress" ? 0.23 : 0.54, 0.105, 1);
      }
    }

    const pointColor = t < 0.5 ? palette.source : palette.target;
    ctx.fillStyle = pointColor;
    ctx.globalAlpha = 0.78;
    for (let i = 0; i < state.pairs.length; i += 1) {
      const p = screen(lerpPair(state.pairs[i], t), vp);
      ctx.beginPath(); ctx.arc(p.x, p.y, 2.15, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;

    if (state.stage === "bridge") drawHighlightedPair(ctx, vp, t);
    if (state.stage === "regress") drawInspector(ctx, vp);
  }

  function drawInspector(ctx, vp) {
    const selected = state.selected;
    const f = fieldAt(selected.x, selected.y, state.t, state.bandwidth);
    const ordered = f.nearby.slice().sort((a, b) => a.d2 - b.d2).slice(0, 14);
    ctx.save();
    ctx.strokeStyle = "rgba(22, 125, 105, 0.24)";
    ctx.fillStyle = "rgba(22, 125, 105, 0.06)";
    const s = screen(selected, vp);
    const radius = state.bandwidth * vp.size / (vp.max - vp.min) * 1.8;
    ctx.beginPath(); ctx.arc(s.x, s.y, radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ordered.forEach((item) => arrow(ctx, selected, item.pair.vx, item.pair.vy, vp, palette.target, 0.32, 0.13, 1));
    arrow(ctx, selected, f.vx, f.vy, vp, palette.flowDark, 1, 0.2, 3.2);
    ctx.fillStyle = palette.paper; ctx.strokeStyle = palette.flowDark; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(s.x, s.y, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.restore();
  }

  function drawSample() {
    const { ctx, w, h } = resizeCanvas(sampleCanvas);
    const vp = viewport(w, h);
    drawBackground(ctx, w, h, vp, "fresh samples · t = " + state.sampleTime.toFixed(2));
    ctx.fillStyle = palette.target;
    ctx.globalAlpha = 0.18;
    state.targetReference.forEach((p) => {
      const s = screen(p, vp); ctx.beginPath(); ctx.arc(s.x, s.y, 2.4, 0, Math.PI * 2); ctx.fill();
    });
    ctx.globalAlpha = 0.82;
    ctx.fillStyle = palette.flow;
    state.samples.forEach((p) => {
      const s = screen(p, vp); ctx.beginPath(); ctx.arc(s.x, s.y, 2.8, 0, Math.PI * 2); ctx.fill();
    });
    ctx.globalAlpha = 1;
  }

  function drawAll() { drawMain(); drawSample(); }

  function updateStats() {
    ui.timeOutput.value = state.t.toFixed(2);
    ui.timeBadge.textContent = state.t.toFixed(2);
    ui.bandwidthOutput.value = state.bandwidth.toFixed(2);
    const meanLength = state.pairs.reduce((sum, p) => sum + Math.hypot(p.vx, p.vy), 0) / state.pairs.length;
    const local = fieldAt(state.selected.x, state.selected.y, state.t, state.bandwidth);
    if (state.stage === "diffuse") {
      ui.statALabel.textContent = "corruption level"; ui.statA.textContent = Math.round(state.t * 100) + "%";
      ui.statBLabel.textContent = "direction at generation"; ui.statB.textContent = "← reverse";
    } else if (state.stage === "regress") {
      ui.statALabel.textContent = "local target spread"; ui.statA.textContent = local.spread.toFixed(2);
      ui.statBLabel.textContent = "averaged speed"; ui.statB.textContent = Math.hypot(local.vx, local.vy).toFixed(2);
    } else {
      ui.statALabel.textContent = "mean path length"; ui.statA.textContent = meanLength.toFixed(2);
      ui.statBLabel.textContent = "mean speed"; ui.statB.textContent = meanLength.toFixed(2);
    }
  }

  const stageCopy = {
    diffuse: {
      kicker: "Baseline · Forward diffusion", title: "Destroy the data in a way we can learn to undo",
      description: "Each orange data point is mixed with a known Gaussian-noise sample. Scrub corruption time from clean data to pure noise.",
      hint: "Generation runs in the opposite direction: noise → data.",
      reading: "<strong>What diffusion learns:</strong> from a noisy point and its noise level, predict the noise component. Removing that estimate a little at a time reveals data."
    },
    bridge: {
      kicker: "Flow matching · Conditional paths", title: "Replace corruption with a route",
      description: "Yes, each pairing is arbitrary: independently draw noise and data, then connect them only to manufacture a velocity label.",
      hint: "Random connections are training scaffolding. They are discarded before generation.",
      reading: "<strong>No point chooses a mode.</strong> The model sees only current position and time—not the highlighted endpoint—and learns the average velocity across many random pairings."
    },
    regress: {
      kicker: "Flow matching · Marginal field", title: "Average conflicting instructions",
      description: "Click anywhere in the plot. Orange arrows show nearby conditional targets; the thick green arrow is their kernel-weighted regression target.",
      hint: "Click or drag to move the regression microscope.",
      reading: "<strong>Read this view:</strong> the fitted field is smoother than the paired arrows because squared-error regression averages targets that meet at the same place and time."
    },
    sample: {
      kicker: "Flow matching · Probability-flow ODE", title: "Release particles into the field",
      description: "The paths have done their job: they supplied training labels. New noise can now move without knowing any destination.",
      hint: "Use the generator below to integrate fresh noise.",
      reading: "<strong>Read this view:</strong> at generation time there are no pairs and no target examples—only the learned velocity field."
    }
  };

  function setStage(stage) {
    const wasDiffusion = state.stage === "diffuse";
    if (wasDiffusion) state.diffusionT = state.t; else state.flowT = state.t;
    state.stage = stage;
    state.t = stage === "diffuse" ? state.diffusionT : state.flowT;
    ui.time.value = state.t;
    document.querySelectorAll(".fm-tabs button").forEach((button) => {
      const active = button.dataset.stage === stage;
      button.classList.toggle("is-active", active); button.setAttribute("aria-pressed", String(active));
    });
    const copy = stageCopy[stage];
    ui.stageKicker.textContent = copy.kicker; ui.stageTitle.textContent = copy.title;
    ui.stageDescription.textContent = copy.description; ui.hint.textContent = copy.hint; ui.reading.innerHTML = copy.reading;
    const diffusion = stage === "diffuse";
    ui.timeStart.textContent = diffusion ? "data" : "noise";
    ui.timeEnd.textContent = diffusion ? "noise" : "data";
    ui.timeSymbol.textContent = diffusion ? "τ" : "t";
    ui.pathsLabel.textContent = diffusion ? "corruption paths" : "paired paths";
    ui.targetsLabel.textContent = diffusion ? "derived denoising directions" : "training arrows";
    ui.fieldLabel.textContent = diffusion ? "averaged denoiser" : "fitted velocity field";
    ui.bandwidthHelp.textContent = diffusion ? "How locally the demo averages denoising targets. It plays the role of model smoothness." : "How locally the demo averages velocity targets. It plays the role of model smoothness.";
    ui.pairControl.hidden = stage !== "bridge";
    ui.play.textContent = state.t >= 0.999 ? "Replay" : "Play";
    if (stage === "regress") { ui.field.checked = true; ui.targets.checked = true; }
    if (stage === "sample") document.querySelector(".fm-sampler").scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
    updateStats(); drawMain();
  }

  function stopPlay(completed) {
    state.playing = false; ui.play.textContent = completed || state.t >= 0.999 ? "Replay" : "Play"; clearTimeout(state.playFrame);
  }

  function togglePlay() {
    if (state.playing) { stopPlay(); return; }
    if (state.t >= 0.999) {
      state.t = 0; ui.time.value = 0; updateStats(); drawMain();
    }
    state.playing = true; ui.play.textContent = "Pause";
    let previous = performance.now();
    function tick() {
      if (!state.playing) return;
      const now = performance.now();
      const delta = Math.min(0.04, (now - previous) / 1000); previous = now;
      state.t += delta * 0.28;
      if (state.t >= 1) state.t = 1;
      ui.time.value = state.t; updateStats(); drawMain();
      if (state.t >= 1) { stopPlay(true); return; }
      state.playFrame = window.setTimeout(tick, 16);
    }
    state.playFrame = window.setTimeout(tick, 16);
  }

  function resetSamples() {
    clearTimeout(state.sampleFrame);
    state.sampling = false; state.sampleTime = 0; state.sampleStep = 0;
    const rng = mulberry32((state.seed ^ 0x9E3779B9) >>> 0);
    state.sampleOrigin = []; state.samples = [];
    for (let i = 0; i < 180; i += 1) {
      const p = sourcePoint(rng); state.sampleOrigin.push({ x: p.x, y: p.y }); state.samples.push({ x: p.x, y: p.y });
    }
    ui.generate.disabled = false; ui.sampleStatus.textContent = "Ready · 180 fresh noise points";
    drawSample();
  }

  function generate() {
    if (state.sampling) return;
    state.samples = state.sampleOrigin.map((p) => ({ x: p.x, y: p.y }));
    state.sampleStep = 0; state.sampleTime = 0; state.sampling = true; ui.generate.disabled = true;
    const steps = state.sampleSteps, dt = 1 / steps;
    function animate() {
      if (!state.sampling) return;
      const t = Math.min(1, state.sampleStep * dt);
      state.samples = state.samples.map((p) => {
        const f = fieldAt(p.x, p.y, t, state.bandwidth);
        return { x: p.x + dt * f.vx, y: p.y + dt * f.vy };
      });
      state.sampleStep += 1; state.sampleTime = Math.min(1, state.sampleStep * dt);
      ui.sampleStatus.textContent = "Integrating · step " + state.sampleStep + " of " + steps + " · t = " + state.sampleTime.toFixed(2);
      drawSample();
      if (state.sampleStep < steps) state.sampleFrame = window.setTimeout(animate, 70);
      else {
        state.sampling = false; ui.generate.disabled = false;
        ui.sampleStatus.textContent = "Complete · " + steps + " Euler steps · no paired destinations used";
      }
    }
    state.sampleFrame = window.setTimeout(animate, 0);
  }

  document.querySelectorAll(".fm-tabs button").forEach((button) => button.addEventListener("click", () => setStage(button.dataset.stage)));
  document.querySelectorAll("#fm-shape-choices button").forEach((button) => button.addEventListener("click", () => {
    state.shape = button.dataset.shape;
    document.querySelectorAll("#fm-shape-choices button").forEach((b) => b.classList.toggle("is-active", b === button));
    rebuild();
  }));
  ui.time.addEventListener("input", () => {
    stopPlay(); state.t = Number(ui.time.value); ui.play.textContent = state.t >= 0.999 ? "Replay" : "Play"; updateStats(); drawMain();
  });
  ui.play.addEventListener("click", togglePlay);
  [ui.paths, ui.targets, ui.field].forEach((input) => input.addEventListener("change", drawMain));
  ui.bandwidth.addEventListener("input", () => { state.bandwidth = Number(ui.bandwidth.value); updateStats(); drawAll(); });
  ui.reseed.addEventListener("click", () => {
    let seed = Number(ui.seed.value);
    if (!Number.isFinite(seed)) seed = 20260719;
    if ((seed >>> 0) === state.seed) seed = (state.seed + 1) >>> 0;
    state.seed = seed >>> 0; ui.seed.value = state.seed; rebuild();
  });
  ui.tracePair.addEventListener("click", () => {
    state.highlightedPair = (state.highlightedPair + 37) % state.pairs.length;
    ui.pairNumber.value = state.highlightedPair + 1;
    drawMain();
  });
  ui.steps.addEventListener("input", () => {
    state.sampleSteps = 2 ** Number(ui.steps.value); ui.stepsOutput.value = state.sampleSteps; resetSamples();
  });
  ui.generate.addEventListener("click", generate);
  ui.resetSamples.addEventListener("click", resetSamples);

  function moveInspector(event) {
    if (state.stage !== "regress") return;
    const rect = mainCanvas.getBoundingClientRect();
    const vp = viewport(rect.width, rect.height);
    const p = world(event.clientX - rect.left, event.clientY - rect.top, vp);
    state.selected.x = Math.max(vp.min, Math.min(vp.max, p.x));
    state.selected.y = Math.max(vp.min, Math.min(vp.max, p.y));
    updateStats(); drawMain();
  }
  mainCanvas.addEventListener("pointerdown", (event) => { mainCanvas.setPointerCapture(event.pointerId); moveInspector(event); });
  mainCanvas.addEventListener("pointermove", (event) => { if (mainCanvas.hasPointerCapture(event.pointerId)) moveInspector(event); });

  let resizeFrame = 0;
  window.addEventListener("resize", () => { cancelAnimationFrame(resizeFrame); resizeFrame = requestAnimationFrame(drawAll); });
  if (window.ResizeObserver) {
    const observer = new ResizeObserver(() => { cancelAnimationFrame(resizeFrame); resizeFrame = requestAnimationFrame(drawAll); });
    observer.observe(mainCanvas.parentElement); observer.observe(sampleCanvas.parentElement);
  }

  state.bandwidth = Number(ui.bandwidth.value);
  state.sampleSteps = 2 ** Number(ui.steps.value);
  rebuild();
}());
