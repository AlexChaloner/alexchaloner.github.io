(function () {
  "use strict";

  const mainCanvas = document.getElementById("fm-main-canvas");
  const sampleCanvas = document.getElementById("fm-sample-canvas");
  if (!mainCanvas || !sampleCanvas) return;

  const $ = (id) => document.getElementById(id);
  const ui = {
    time: $("fm-time"), timeOutput: $("fm-time-output"), timeBadge: $("fm-time-badge"), play: $("fm-play"),
    pathControls: $("fm-path-controls"),
    diffusionTimeBadge: $("fm-diffusion-time-badge"), diffusionHint: $("fm-diffusion-hint"),
    paths: $("fm-show-paths"), targets: $("fm-show-targets"), field: $("fm-show-field"),
    bandwidth: $("fm-bandwidth"), bandwidthOutput: $("fm-bandwidth-output"), seed: $("fm-seed"), reseed: $("fm-reseed"),
    bandwidthHelp: $("fm-bandwidth-help"),
    pairControl: $("fm-pair-control"), tracePair: $("fm-trace-pair"), pairNumber: $("fm-pair-number"),
    trainingCount: $("fm-training-count"), trainingCountOutput: $("fm-training-count-output"), train: $("fm-train"),
    generationControls: $("fm-generation-controls"),
    stageKicker: $("fm-stage-kicker"), stageTitle: $("fm-stage-title"), stageDescription: $("fm-stage-description"),
    hint: $("fm-canvas-hint"), pairCount: $("fm-pair-count"), statALabel: $("fm-stat-a-label"), statA: $("fm-stat-a"),
    statBLabel: $("fm-stat-b-label"), statB: $("fm-stat-b"), reading: $("fm-reading"),
    steps: $("fm-steps"), stepsOutput: $("fm-steps-output"), generate: $("fm-generate"), resetSamples: $("fm-reset-samples"),
    sampleStatus: $("fm-sample-status")
  };

  const palette = { ink: "#17211d", grid: "#dfe6e0", source: "#7857b2", sourceDark: "#58388d", target: "#e36d3d", flow: "#167d69", flowDark: "#095c4d", paper: "#fbfcf8" };
  const state = {
    seed: 20260719, shape: "eight", stage: "targets", t: 0.55, bandwidth: 0.34, trainingCount: 420,
    pairs: [], targetReference: [], selected: { x: 0, y: 0 }, highlightedPair: 0, playing: false, playFrame: 0,
    training: false, trainingFrame: 0,
    sampleOrigin: [], samples: [], diffusionSamples: [], sampleTime: 0, diffusionTime: 0.995,
    sampleStep: 0, sampleSteps: 16, sampleFrame: 0, sampling: false
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
    state.highlightedPair = state.seed % state.trainingCount;
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
    const limit = Math.min(state.trainingCount, state.pairs.length);
    for (let i = 0; i < limit; i += 1) {
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
    const limit = Math.min(state.trainingCount, state.pairs.length);
    for (let i = 0; i < limit; i += 1) {
      const pair = state.pairs[i];
      const p = diffusePair(pair, t);
      const dx = p.x - x, dy = p.y - y;
      const d2 = dx * dx + dy * dy;
      if (d2 > h2 * 12) continue;
      const weight = Math.exp(-d2 / (2 * h2));
      const targetX = pair.a.x, targetY = pair.a.y;
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

  function drawField(ctx, vp, t, method) {
    const count = vp.size < 440 ? 10 : 13;
    const gap = (vp.max - vp.min) / count;
    for (let iy = 0; iy <= count; iy += 1) {
      for (let ix = 0; ix <= count; ix += 1) {
        const x = vp.min + gap * (ix + 0.5), y = vp.min + gap * (iy + 0.5);
        if (x > vp.max || y > vp.max) continue;
        const f = method === "diffusion" ? denoiserAt(x, y, t, state.bandwidth) : fieldAt(x, y, t, state.bandwidth);
        const mag = Math.hypot(f.vx, f.vy);
        if (f.confidence < 0.05 || mag < 0.03) continue;
        const cap = Math.min(0.33, 0.25 / Math.max(0.01, mag));
        const color = method === "diffusion" ? palette.sourceDark : palette.flowDark;
        arrow(ctx, { x, y }, f.vx, f.vy, vp, color, 0.18 + f.confidence * 0.62, cap, 1.3);
      }
    }
  }

  function drawDiffusion(ctx, vp, t) {
    if (ui.paths.checked) {
      ctx.strokeStyle = "rgba(92, 105, 98, 0.16)";
      ctx.lineWidth = 0.65;
      for (let i = 0; i < state.trainingCount; i += 4) {
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

    if (ui.field.checked || state.stage === "fields") drawField(ctx, vp, t, "diffusion");
    if (ui.targets.checked && t > 0.005) {
      for (let i = 0; i < state.trainingCount; i += 8) {
        const pair = state.pairs[i], p = diffusePair(pair, t);
        arrow(ctx, p, pair.a.x, pair.a.y, vp, palette.source, state.stage === "fields" ? 0.2 : 0.55, 0.12, 1);
      }
    }

    const pointColor = t < 0.5 ? palette.target : palette.source;
    ctx.fillStyle = pointColor;
    ctx.globalAlpha = 0.8;
    state.pairs.slice(0, state.trainingCount).forEach((pair) => {
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
    if (state.stage === "sample") { drawGenerated(ctx, w, h, vp, state.samples, "flow ODE · t = " + state.sampleTime.toFixed(2), palette.flow); return; }
    drawBackground(ctx, w, h, vp, state.stage === "fields" ? "learned velocity v(x, t)" : "conditional velocity targets");

    if (ui.paths.checked) {
      ctx.lineWidth = 0.65;
      for (let i = 0; i < state.trainingCount; i += 3) {
        const pair = state.pairs[i], a = screen(pair.a, vp), b = screen(pair.b, vp);
        ctx.strokeStyle = "rgba(92, 105, 98, 0.18)";
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
    }

    if (ui.field.checked || state.stage === "fields") drawField(ctx, vp, t, "flow");

    if (ui.targets.checked) {
      const stride = state.stage === "fields" ? 5 : 8;
      for (let i = 0; i < state.trainingCount; i += stride) {
        const pair = state.pairs[i], p = lerpPair(pair, t);
        arrow(ctx, p, pair.vx, pair.vy, vp, palette.target, state.stage === "fields" ? 0.2 : 0.54, 0.105, 1);
      }
    }

    const pointColor = t < 0.5 ? palette.source : palette.target;
    ctx.fillStyle = pointColor;
    ctx.globalAlpha = 0.78;
    for (let i = 0; i < state.trainingCount; i += 1) {
      const p = screen(lerpPair(state.pairs[i], t), vp);
      ctx.beginPath(); ctx.arc(p.x, p.y, 2.15, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;

    if (state.stage === "targets") drawHighlightedPair(ctx, vp, t);
  }

  function drawSample() {
    const { ctx, w, h } = resizeCanvas(sampleCanvas);
    const vp = viewport(w, h);
    if (state.stage === "sample") {
      drawGenerated(ctx, w, h, vp, state.diffusionSamples, "reverse diffusion · τ = " + state.diffusionTime.toFixed(2), palette.sourceDark);
      return;
    }
    drawBackground(ctx, w, h, vp, state.stage === "fields" ? "predicted noise ε̂(x, τ)" : "exact added-noise ε targets");
    drawDiffusion(ctx, vp, state.t);
  }

  function drawGenerated(ctx, w, h, vp, samples, label, color) {
    drawBackground(ctx, w, h, vp, label);
    ctx.fillStyle = palette.target;
    ctx.globalAlpha = 0.18;
    state.targetReference.forEach((p) => {
      const s = screen(p, vp); ctx.beginPath(); ctx.arc(s.x, s.y, 2.4, 0, Math.PI * 2); ctx.fill();
    });
    ctx.globalAlpha = 0.82;
    ctx.fillStyle = color;
    samples.forEach((p) => {
      const s = screen(p, vp); ctx.beginPath(); ctx.arc(s.x, s.y, 2.8, 0, Math.PI * 2); ctx.fill();
    });
    ctx.globalAlpha = 1;
  }

  function drawAll() { drawMain(); drawSample(); }

  function updateStats() {
    ui.timeOutput.value = state.t.toFixed(2);
    ui.timeBadge.textContent = state.t.toFixed(2);
    ui.diffusionTimeBadge.textContent = state.t.toFixed(2);
    ui.bandwidthOutput.value = state.bandwidth.toFixed(2);
    ui.trainingCountOutput.value = state.trainingCount + " / 420";
    ui.pairCount.textContent = state.trainingCount;
    if (state.stage === "fields") {
      ui.statALabel.textContent = "diffusion estimate"; ui.statA.textContent = "predicted ε̂";
      ui.statBLabel.textContent = "flow estimate"; ui.statB.textContent = "predicted v̂";
    } else if (state.stage === "sample") {
      ui.statALabel.textContent = "diffusion sampling"; ui.statA.textContent = "reverse noise";
      ui.statBLabel.textContent = "flow sampling"; ui.statB.textContent = "follow velocity";
    } else {
      ui.statALabel.textContent = "diffusion target"; ui.statA.textContent = "noise ε";
      ui.statBLabel.textContent = "flow target"; ui.statB.textContent = "velocity v";
    }
  }

  const stageCopy = {
    targets: {
      kicker: "Step 1 · Supervised learning targets", title: "The crucial difference is what appears after the arrow",
      description: "Diffusion predicts the exact Gaussian noise mixed into a data example. Flow matching predicts the velocity of a temporary noise–data path.",
      diffusionHint: "Purple arrows are the exact added-noise ε targets.",
      flowHint: "The highlighted random pair exists only to make a velocity label.",
      reading: "<strong>Same input signature, different answer:</strong> both models receive a point and a time. Diffusion is graded against added noise; flow matching is graded against motion."
    },
    fields: {
      kicker: "Step 2 · Evidence accumulates", title: "Many incompatible labels become one usable field",
      description: "Replay learning or scrub the example count. Diffusion averages noise targets into ε̂(x, τ); flow matching averages velocity targets into v̂(x, t).",
      diffusionHint: "Purple arrows show the learned noise predictor ε̂(x, τ).",
      flowHint: "Green arrows show the learned velocity field v̂(x, t).",
      reading: "<strong>What has been learned:</strong> diffusion can turn ε̂ into a denoising update. Flow matching can use v̂ directly as the derivative of its sampling ODE."
    },
    sample: {
      kicker: "Step 3 · Generation", title: "Release the same fresh noise into both learners",
      description: "Diffusion repeatedly removes predicted noise. Flow matching repeatedly follows predicted velocity. Neither sampler receives a paired destination.",
      diffusionHint: "Reverse diffusion repeatedly uses ε̂ to step toward clean data.",
      flowHint: "The flow solver repeatedly follows v̂ from noise to data.",
      reading: "<strong>Generation uses the predictions differently:</strong> diffusion converts noise estimates into reverse denoising steps; flow matching integrates velocity as an ODE."
    }
  };

  function setStage(stage) {
    state.stage = stage;
    document.querySelectorAll(".fm-tabs button").forEach((button) => {
      const active = button.dataset.stage === stage;
      button.classList.toggle("is-active", active); button.setAttribute("aria-pressed", String(active));
    });
    const copy = stageCopy[stage];
    ui.stageKicker.textContent = copy.kicker; ui.stageTitle.textContent = copy.title;
    ui.stageDescription.textContent = copy.description; ui.diffusionHint.innerHTML = copy.diffusionHint;
    ui.hint.textContent = copy.flowHint; ui.reading.innerHTML = copy.reading;
    ui.pathControls.hidden = stage === "sample";
    ui.timeBadge.parentElement.hidden = stage === "sample";
    ui.diffusionTimeBadge.parentElement.hidden = stage === "sample";
    ui.generationControls.hidden = stage !== "sample";
    ui.pairControl.hidden = stage !== "targets";
    if (stage === "fields") { ui.field.checked = true; ui.targets.checked = true; }
    if (stage === "sample") resetSamples();
    updateStats(); drawAll();
  }

  function stopPlay(completed) {
    state.playing = false; ui.play.textContent = completed || state.t >= 0.994 ? "Replay" : "Play once"; clearTimeout(state.playFrame);
  }

  function togglePlay() {
    if (state.playing) { stopPlay(); return; }
    if (state.t >= 0.994) {
      state.t = 0.005; ui.time.value = state.t; updateStats(); drawAll();
    }
    state.playing = true; ui.play.textContent = "Pause";
    let previous = performance.now();
    function tick() {
      if (!state.playing) return;
      const now = performance.now();
      const delta = Math.min(0.04, (now - previous) / 1000); previous = now;
      state.t += delta * 0.28;
      if (state.t >= 0.995) state.t = 0.995;
      ui.time.value = state.t; updateStats(); drawAll();
      if (state.t >= 0.995) { stopPlay(true); return; }
      state.playFrame = window.setTimeout(tick, 16);
    }
    state.playFrame = window.setTimeout(tick, 16);
  }

  function resetSamples() {
    clearTimeout(state.sampleFrame);
    state.sampling = false; state.sampleTime = 0; state.diffusionTime = 0.995; state.sampleStep = 0;
    const rng = mulberry32((state.seed ^ 0x9E3779B9) >>> 0);
    state.sampleOrigin = []; state.samples = []; state.diffusionSamples = [];
    for (let i = 0; i < 180; i += 1) {
      const p = sourcePoint(rng);
      state.sampleOrigin.push({ x: p.x, y: p.y });
      state.samples.push({ x: p.x, y: p.y }); state.diffusionSamples.push({ x: p.x, y: p.y });
    }
    ui.generate.disabled = false; ui.sampleStatus.textContent = "Ready · 180 shared noise points";
    if (state.stage === "sample") drawAll();
  }

  function generate() {
    if (state.sampling) return;
    state.samples = state.sampleOrigin.map((p) => ({ x: p.x, y: p.y }));
    state.diffusionSamples = state.sampleOrigin.map((p) => ({ x: p.x, y: p.y }));
    state.sampleStep = 0; state.sampleTime = 0; state.diffusionTime = 0.995; state.sampling = true; ui.generate.disabled = true;
    const steps = state.sampleSteps, dt = 1 / steps, diffusionDt = 0.99 / steps;
    function animate() {
      if (!state.sampling) return;
      const t = Math.min(1, state.sampleStep * dt);
      state.samples = state.samples.map((p) => {
        const f = fieldAt(p.x, p.y, t, state.bandwidth);
        return { x: p.x + dt * f.vx, y: p.y + dt * f.vy };
      });
      const tau = Math.max(0.005, 0.995 - state.sampleStep * diffusionDt);
      const nextTau = Math.max(0.005, tau - diffusionDt);
      state.diffusionSamples = state.diffusionSamples.map((p) => {
        const eps = denoiserAt(p.x, p.y, tau, state.bandwidth);
        const dataScale = Math.sqrt(Math.max(0.005, 1 - tau));
        const noiseScale = Math.sqrt(tau);
        const cleanX = Math.max(-4, Math.min(4, (p.x - noiseScale * eps.vx) / dataScale));
        const cleanY = Math.max(-4, Math.min(4, (p.y - noiseScale * eps.vy) / dataScale));
        return {
          x: Math.sqrt(1 - nextTau) * cleanX + Math.sqrt(nextTau) * eps.vx,
          y: Math.sqrt(1 - nextTau) * cleanY + Math.sqrt(nextTau) * eps.vy
        };
      });
      state.sampleStep += 1; state.sampleTime = Math.min(1, state.sampleStep * dt);
      state.diffusionTime = nextTau;
      ui.sampleStatus.textContent = "Generating both · step " + state.sampleStep + " of " + steps;
      drawAll();
      if (state.sampleStep < steps) state.sampleFrame = window.setTimeout(animate, 70);
      else {
        state.sampling = false; ui.generate.disabled = false;
        ui.sampleStatus.textContent = "Complete · " + steps + " steps each · same starting noise";
      }
    }
    state.sampleFrame = window.setTimeout(animate, 0);
  }

  function stopTraining(completed) {
    state.training = false; clearTimeout(state.trainingFrame);
    ui.train.textContent = completed || state.trainingCount >= 420 ? "Replay learning" : "Continue learning";
  }

  function replayTraining() {
    if (state.training) { stopTraining(false); return; }
    if (state.trainingCount >= 420) state.trainingCount = 20;
    state.training = true; ui.train.textContent = "Pause learning";
    function tick() {
      if (!state.training) return;
      state.trainingCount = Math.min(420, state.trainingCount + 4);
      ui.trainingCount.value = state.trainingCount; updateStats(); drawAll();
      if (state.trainingCount >= 420) { stopTraining(true); return; }
      state.trainingFrame = window.setTimeout(tick, 24);
    }
    updateStats(); drawAll(); state.trainingFrame = window.setTimeout(tick, 24);
  }

  document.querySelectorAll(".fm-tabs button").forEach((button) => button.addEventListener("click", () => setStage(button.dataset.stage)));
  document.querySelectorAll("#fm-shape-choices button").forEach((button) => button.addEventListener("click", () => {
    state.shape = button.dataset.shape;
    document.querySelectorAll("#fm-shape-choices button").forEach((b) => b.classList.toggle("is-active", b === button));
    rebuild();
  }));
  ui.time.addEventListener("input", () => {
    stopPlay(); state.t = Number(ui.time.value); ui.play.textContent = state.t >= 0.994 ? "Replay" : "Play once"; updateStats(); drawAll();
  });
  ui.play.addEventListener("click", togglePlay);
  [ui.paths, ui.targets, ui.field].forEach((input) => input.addEventListener("change", drawAll));
  ui.bandwidth.addEventListener("input", () => { state.bandwidth = Number(ui.bandwidth.value); updateStats(); drawAll(); });
  ui.trainingCount.addEventListener("input", () => {
    stopTraining(false); state.trainingCount = Number(ui.trainingCount.value);
    state.highlightedPair %= state.trainingCount; ui.pairNumber.value = state.highlightedPair + 1;
    updateStats(); drawAll();
  });
  ui.train.addEventListener("click", replayTraining);
  ui.reseed.addEventListener("click", () => {
    let seed = Number(ui.seed.value);
    if (!Number.isFinite(seed)) seed = 20260719;
    if ((seed >>> 0) === state.seed) seed = (state.seed + 1) >>> 0;
    state.seed = seed >>> 0; ui.seed.value = state.seed; rebuild();
  });
  ui.tracePair.addEventListener("click", () => {
    state.highlightedPair = (state.highlightedPair + 37) % state.trainingCount;
    ui.pairNumber.value = state.highlightedPair + 1;
    drawMain();
  });
  ui.steps.addEventListener("input", () => {
    state.sampleSteps = 2 ** Number(ui.steps.value); ui.stepsOutput.value = state.sampleSteps; resetSamples();
  });
  ui.generate.addEventListener("click", generate);
  ui.resetSamples.addEventListener("click", resetSamples);

  let resizeFrame = 0;
  window.addEventListener("resize", () => { cancelAnimationFrame(resizeFrame); resizeFrame = requestAnimationFrame(drawAll); });
  if (window.ResizeObserver) {
    const observer = new ResizeObserver(() => { cancelAnimationFrame(resizeFrame); resizeFrame = requestAnimationFrame(drawAll); });
    observer.observe(mainCanvas.parentElement); observer.observe(sampleCanvas.parentElement);
  }

  state.bandwidth = Number(ui.bandwidth.value);
  state.sampleSteps = 2 ** Number(ui.steps.value);
  state.trainingCount = Number(ui.trainingCount.value);
  rebuild();
}());
