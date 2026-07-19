(function () {
  "use strict";

  const data = window.MNISTLatentData;
  const mount = document.getElementById("mnist-lab");
  if (!data || !mount) return;

  const $ = (id) => document.getElementById(id);
  const ui = {
    progress: $("mnist-progress"), progressLabel: $("mnist-progress-label"),
    train: $("mnist-train"), pause: $("mnist-pause"), reset: $("mnist-reset"),
    diffusionLoss: $("mnist-diffusion-loss"), flowLoss: $("mnist-flow-loss"),
    diffusionSamples: $("mnist-diffusion-samples"), flowSamples: $("mnist-flow-samples"),
    diffusionLossChart: $("mnist-diffusion-loss-chart"), flowLossChart: $("mnist-flow-loss-chart"),
    digit: $("mnist-digit"), updates: $("mnist-updates"), updatesOutput: $("mnist-updates-output"),
    learningRate: $("mnist-learning-rate"), learningRateOutput: $("mnist-learning-rate-output"),
    speed: $("mnist-speed"), speedOutput: $("mnist-speed-output"),
    solverSteps: $("mnist-solver-steps"), solverStepsOutput: $("mnist-solver-steps-output"),
    temperature: $("mnist-temperature"), temperatureOutput: $("mnist-temperature-output"),
    seed: $("mnist-seed"), regenerate: $("mnist-regenerate"), status: $("mnist-status"),
    journey: $("mnist-journey"), journeyLabel: $("mnist-journey-label"),
    diffusionCurrent: $("mnist-diffusion-current"), flowCurrent: $("mnist-flow-current"),
    diffusionFilm: $("mnist-diffusion-film"), flowFilm: $("mnist-flow-film"),
    diffusionAction: $("mnist-diffusion-action"), flowAction: $("mnist-flow-action"),
    pathChart: $("mnist-path-chart")
  };

  const D = data.latentDim;
  const H = 32;
  const INPUT = D + 1;
  const BATCH = 24;
  const SAMPLE_COUNT = 12;

  function decodeBase64(value, signed) {
    const raw = atob(value);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i += 1) bytes[i] = raw.charCodeAt(i);
    return signed ? new Int8Array(bytes.buffer) : bytes;
  }

  const latentBytes = decodeBase64(data.latents, true);
  const labels = decodeBase64(data.labels, false);
  const decoderW1 = decodeBase64(data.decoder.w1, true);
  const decoderW2 = decodeBase64(data.decoder.w2, true);
  const decoderB1 = Float32Array.from(data.decoder.b1);
  const decoderB2 = Float32Array.from(data.decoder.b2);

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function normalSource(rng) {
    let spare = null;
    return function () {
      if (spare !== null) { const value = spare; spare = null; return value; }
      const radius = Math.sqrt(-2 * Math.log(Math.max(1e-9, rng())));
      const angle = Math.PI * 2 * rng();
      spare = radius * Math.sin(angle);
      return radius * Math.cos(angle);
    };
  }

  function latentAt(index, out, offset) {
    const base = index * D;
    for (let j = 0; j < D; j += 1) out[offset + j] = latentBytes[base + j] * data.latentScale;
  }

  function makeNetwork(rng) {
    const w1 = new Float32Array(H * INPUT);
    const b1 = new Float32Array(H);
    const w2 = new Float32Array(D * H);
    const b2 = new Float32Array(D);
    const limit1 = Math.sqrt(6 / (INPUT + H));
    const limit2 = Math.sqrt(6 / (H + D));
    for (let i = 0; i < w1.length; i += 1) w1[i] = (rng() * 2 - 1) * limit1;
    for (let i = 0; i < w2.length; i += 1) w2[i] = (rng() * 2 - 1) * limit2;
    const params = [w1, b1, w2, b2];
    return {
      w1, b1, w2, b2, step: 0,
      m: params.map((p) => new Float32Array(p.length)),
      v: params.map((p) => new Float32Array(p.length)),
      g: params.map((p) => new Float32Array(p.length))
    };
  }

  function predict(net, latent, time, hidden, output) {
    for (let k = 0; k < H; k += 1) {
      let sum = net.b1[k] + net.w1[k * INPUT + D] * time;
      const row = k * INPUT;
      for (let j = 0; j < D; j += 1) sum += net.w1[row + j] * latent[j];
      hidden[k] = Math.tanh(sum);
    }
    for (let j = 0; j < D; j += 1) {
      let sum = net.b2[j];
      const row = j * H;
      for (let k = 0; k < H; k += 1) sum += net.w2[row + k] * hidden[k];
      output[j] = sum;
    }
    return output;
  }

  function adam(net, learningRate) {
    net.step += 1;
    const correction1 = 1 - Math.pow(0.9, net.step);
    const correction2 = 1 - Math.pow(0.999, net.step);
    const params = [net.w1, net.b1, net.w2, net.b2];
    for (let group = 0; group < params.length; group += 1) {
      const p = params[group], g = net.g[group], m = net.m[group], v = net.v[group];
      for (let i = 0; i < p.length; i += 1) {
        const gradient = Math.max(-5, Math.min(5, g[i]));
        m[i] = 0.9 * m[i] + 0.1 * gradient;
        v[i] = 0.999 * v[i] + 0.001 * gradient * gradient;
        p[i] -= learningRate * (m[i] / correction1) / (Math.sqrt(v[i] / correction2) + 1e-8);
      }
    }
  }

  function makeBatch() {
    const clean = new Float32Array(BATCH * D);
    const noise = new Float32Array(BATCH * D);
    const times = new Float32Array(BATCH);
    for (let item = 0; item < BATCH; item += 1) {
      const index = state.indices[Math.floor(state.trainingRng() * state.indices.length)];
      latentAt(index, clean, item * D);
      times[item] = 0.02 + state.trainingRng() * 0.96;
      for (let j = 0; j < D; j += 1) noise[item * D + j] = state.trainingNormal();
    }
    return { clean, noise, times };
  }

  function trainNetwork(net, batch, method, learningRate) {
    net.g.forEach((gradient) => gradient.fill(0));
    const input = new Float32Array(D);
    const target = new Float32Array(D);
    const hidden = new Float32Array(H);
    const output = new Float32Array(D);
    const hiddenGradient = new Float32Array(H);
    let loss = 0;
    for (let item = 0; item < BATCH; item += 1) {
      const offset = item * D;
      const t = batch.times[item];
      const cleanScale = method === "diffusion" ? Math.sqrt(1 - t) : t;
      const noiseScale = method === "diffusion" ? Math.sqrt(t) : 1 - t;
      for (let j = 0; j < D; j += 1) {
        const clean = batch.clean[offset + j], noise = batch.noise[offset + j];
        input[j] = cleanScale * clean + noiseScale * noise;
        target[j] = method === "diffusion" ? noise : clean - noise;
      }
      predict(net, input, t, hidden, output);
      hiddenGradient.fill(0);
      for (let j = 0; j < D; j += 1) {
        const error = output[j] - target[j];
        loss += error * error;
        const derivative = 2 * error / (BATCH * D);
        net.g[3][j] += derivative;
        const row = j * H;
        for (let k = 0; k < H; k += 1) {
          net.g[2][row + k] += derivative * hidden[k];
          hiddenGradient[k] += net.w2[row + k] * derivative;
        }
      }
      for (let k = 0; k < H; k += 1) {
        const derivative = hiddenGradient[k] * (1 - hidden[k] * hidden[k]);
        net.g[1][k] += derivative;
        const row = k * INPUT;
        for (let j = 0; j < D; j += 1) net.g[0][row + j] += derivative * input[j];
        net.g[0][row + D] += derivative * t;
      }
    }
    adam(net, learningRate);
    return loss / (BATCH * D);
  }

  function decoder(latent) {
    const hidden = new Float32Array(data.decoder.hiddenDim);
    for (let k = 0; k < hidden.length; k += 1) {
      let sum = decoderB1[k];
      const row = k * D;
      for (let j = 0; j < D; j += 1) sum += decoderW1[row + j] * data.decoder.w1Scale * latent[j];
      hidden[k] = Math.max(0, sum);
    }
    const pixels = new Uint8ClampedArray(784);
    for (let p = 0; p < 784; p += 1) {
      let sum = decoderB2[p];
      const row = p * hidden.length;
      for (let k = 0; k < hidden.length; k += 1) sum += decoderW2[row + k] * data.decoder.w2Scale * hidden[k];
      pixels[p] = Math.round(255 / (1 + Math.exp(-Math.max(-12, Math.min(12, sum)))));
    }
    return pixels;
  }

  function drawDigits(canvas, latents, tint) {
    const columns = 4, rows = 3, cell = 28;
    canvas.width = columns * cell; canvas.height = rows * cell;
    const ctx = canvas.getContext("2d");
    const image = ctx.createImageData(canvas.width, canvas.height);
    for (let n = 0; n < SAMPLE_COUNT; n += 1) {
      const latent = latents.subarray(n * D, (n + 1) * D);
      const pixels = decoder(latent);
      const cellX = (n % columns) * cell, cellY = Math.floor(n / columns) * cell;
      for (let y = 0; y < 28; y += 1) {
        for (let x = 0; x < 28; x += 1) {
          const value = pixels[y * 28 + x] / 255;
          const target = ((cellY + y) * canvas.width + cellX + x) * 4;
          image.data[target] = Math.round(tint[0] * value);
          image.data[target + 1] = Math.round(tint[1] * value);
          image.data[target + 2] = Math.round(tint[2] * value);
          image.data[target + 3] = 255;
        }
      }
    }
    ctx.putImageData(image, 0, 0);
  }

  function paintDigit(image, imageWidth, pixels, cellX, cellY, tint) {
    for (let y = 0; y < 28; y += 1) {
      for (let x = 0; x < 28; x += 1) {
        const value = pixels[y * 28 + x] / 255;
        const target = ((cellY + y) * imageWidth + cellX + x) * 4;
        image.data[target] = Math.round(tint[0] * value);
        image.data[target + 1] = Math.round(tint[1] * value);
        image.data[target + 2] = Math.round(tint[2] * value);
        image.data[target + 3] = 255;
      }
    }
  }

  function drawJourneyDigit(canvas, latent, tint) {
    canvas.width = 28; canvas.height = 28;
    const ctx = canvas.getContext("2d");
    const image = ctx.createImageData(28, 28);
    paintDigit(image, 28, decoder(latent), 0, 0, tint);
    ctx.putImageData(image, 0, 0);
  }

  function drawFilmstrip(canvas, journey, tint, activeCheckpoint, highlight) {
    const count = 6;
    canvas.width = count * 28; canvas.height = 28;
    const ctx = canvas.getContext("2d");
    const image = ctx.createImageData(canvas.width, canvas.height);
    for (let i = 0; i < count; i += 1) {
      const index = Math.round((journey.length - 1) * i / (count - 1));
      paintDigit(image, canvas.width, decoder(journey[index]), i * 28, 0, tint);
    }
    ctx.putImageData(image, 0, 0);
    ctx.strokeStyle = highlight; ctx.lineWidth = 2;
    ctx.strokeRect(activeCheckpoint * 28 + 1, 1, 26, 26);
  }

  function drawPathChart(canvas, diffusionJourney, flowJourney, activeIndex) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = rect.width, h = rect.height, pad = 22;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, w, h);

    const project = (latent) => {
      let x = 0, y = 0;
      for (let j = 0; j < D; j += 1) {
        x += latent[j] * Math.sin((j + 1) * 1.71);
        y += latent[j] * Math.cos((j + 1) * 2.17);
      }
      return [x, y];
    };
    const diffusion = diffusionJourney.map(project), flow = flowJourney.map(project);
    const points = diffusion.concat(flow);
    let minX = Math.min.apply(null, points.map((p) => p[0]));
    let maxX = Math.max.apply(null, points.map((p) => p[0]));
    let minY = Math.min.apply(null, points.map((p) => p[1]));
    let maxY = Math.max.apply(null, points.map((p) => p[1]));
    if (maxX - minX < 0.2) { minX -= 0.1; maxX += 0.1; }
    if (maxY - minY < 0.2) { minY -= 0.1; maxY += 0.1; }
    const marginX = (maxX - minX) * 0.12, marginY = (maxY - minY) * 0.12;
    minX -= marginX; maxX += marginX; minY -= marginY; maxY += marginY;
    const screen = (p) => [pad + (w - pad * 2) * (p[0] - minX) / (maxX - minX), h - pad - (h - pad * 2) * (p[1] - minY) / (maxY - minY)];

    ctx.strokeStyle = "#e2e7e3"; ctx.lineWidth = 1;
    for (let i = 1; i < 5; i += 1) {
      const x = pad + (w - pad * 2) * i / 5, y = pad + (h - pad * 2) * i / 5;
      ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, h - pad); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
    }

    function path(pointsToDraw, color) {
      ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.lineJoin = "round"; ctx.lineCap = "round";
      ctx.beginPath();
      pointsToDraw.forEach((point, index) => {
        const p = screen(point);
        if (index === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
      });
      ctx.stroke();
    }
    path(diffusion, "#7857b2"); path(flow, "#167d69");

    const start = screen(diffusion[0]);
    ctx.fillStyle = "#17211d"; ctx.beginPath(); ctx.arc(start[0], start[1], 4, 0, Math.PI * 2); ctx.fill();
    ctx.font = "10px system-ui, sans-serif"; ctx.fillText("same noise", start[0] + 7, start[1] - 7);

    [[diffusion, "#7857b2"], [flow, "#167d69"]].forEach(([journey, color]) => {
      const selected = screen(journey[activeIndex]);
      ctx.fillStyle = "#fff"; ctx.strokeStyle = color; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(selected[0], selected[1], 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      const end = screen(journey[journey.length - 1]);
      ctx.fillStyle = color; ctx.beginPath(); ctx.arc(end[0], end[1], 3, 0, Math.PI * 2); ctx.fill();
    });
  }

  function renderMicroscope() {
    if (!state.diffusionJourney || !state.flowJourney) return;
    const steps = state.diffusionJourney.length - 1;
    const progress = Number(ui.journey.value) / 100;
    const index = Math.round(progress * steps);
    const checkpoint = Math.round(progress * 5);
    drawJourneyDigit(ui.diffusionCurrent, state.diffusionJourney[index], [205, 184, 255]);
    drawJourneyDigit(ui.flowCurrent, state.flowJourney[index], [163, 245, 215]);
    drawFilmstrip(ui.diffusionFilm, state.diffusionJourney, [205, 184, 255], checkpoint, "#b99ae9");
    drawFilmstrip(ui.flowFilm, state.flowJourney, [163, 245, 215], checkpoint, "#55b79e");
    drawPathChart(ui.pathChart, state.diffusionJourney, state.flowJourney, index);

    if (index === 0) {
      ui.journeyLabel.textContent = "step 0 / " + steps + " · identical latent noise";
      ui.diffusionAction.innerHTML = "<strong>Start:</strong> the decoder is looking at raw Gaussian latent noise.";
      ui.flowAction.innerHTML = "<strong>Start:</strong> exactly the same Gaussian latent noise as diffusion.";
    } else if (index === steps) {
      ui.journeyLabel.textContent = "step " + steps + " / " + steps + " · final decoded states";
      ui.diffusionAction.innerHTML = "<strong>Finish:</strong> repeated noise estimates have been converted into a low-noise digit latent.";
      ui.flowAction.innerHTML = "<strong>Finish:</strong> integrated velocity updates have transported the noise to a digit latent.";
    } else {
      const tau = Math.max(0.02, 0.98 - index * 0.96 / steps);
      const t = index / steps;
      ui.journeyLabel.textContent = "step " + index + " / " + steps + " · synchronized model evaluations";
      ui.diffusionAction.innerHTML = "<strong>Noise level τ = " + tau.toFixed(2) + ":</strong> estimate <i>ε̂</i>; the sampler algebraically turns it into the next cleaner latent.";
      ui.flowAction.innerHTML = "<strong>Path time t = " + t.toFixed(2) + ":</strong> predict <i>v̂</i>; the solver moves the latent directly by <i>v̂ Δt</i>.";
    }
  }

  function drawLoss(canvas, history, color) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = rect.width, h = rect.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#f7f9f5"; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#d8e0da"; ctx.lineWidth = 1;
    for (let y = 1; y < 4; y += 1) { ctx.beginPath(); ctx.moveTo(0, y * h / 4); ctx.lineTo(w, y * h / 4); ctx.stroke(); }
    if (history.length < 2) {
      ctx.fillStyle = "#67736c"; ctx.font = "11px system-ui, sans-serif"; ctx.fillText("Loss appears during training", 10, 18); return;
    }
    const values = history.map((item) => Math.log(Math.max(1e-4, item.loss)));
    let min = Math.min.apply(null, values), max = Math.max.apply(null, values);
    if (max - min < 0.1) { min -= 0.05; max += 0.05; }
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    for (let i = 0; i < values.length; i += 1) {
      const x = 6 + (w - 12) * i / (values.length - 1);
      const y = 6 + (h - 12) * (1 - (values[i] - min) / (max - min));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function sampleModels() {
    const seed = (Number(ui.seed.value) || 20260719) >>> 0;
    const rng = mulberry32((seed ^ 0xA511E9B3) >>> 0);
    const normal = normalSource(rng);
    const temperature = Number(ui.temperature.value);
    const flow = new Float32Array(SAMPLE_COUNT * D);
    const diffusion = new Float32Array(SAMPLE_COUNT * D);
    for (let i = 0; i < flow.length; i += 1) flow[i] = diffusion[i] = normal() * temperature;
    const hidden = new Float32Array(H), output = new Float32Array(D), point = new Float32Array(D);
    const steps = state.solverSteps;
    const flowJourney = [new Float32Array(flow.subarray(0, D))];
    const diffusionJourney = [new Float32Array(diffusion.subarray(0, D))];
    for (let step = 0; step < steps; step += 1) {
      const t = (step + 0.5) / steps;
      for (let n = 0; n < SAMPLE_COUNT; n += 1) {
        const offset = n * D;
        point.set(flow.subarray(offset, offset + D));
        predict(state.flowNet, point, t, hidden, output);
        for (let j = 0; j < D; j += 1) flow[offset + j] += output[j] / steps;
      }
      flowJourney.push(new Float32Array(flow.subarray(0, D)));
    }
    for (let step = 0; step < steps; step += 1) {
      const tau = 0.98 - step * 0.96 / steps;
      const nextTau = Math.max(0.02, tau - 0.96 / steps);
      const cleanScale = Math.sqrt(Math.max(0.02, 1 - tau));
      const noiseScale = Math.sqrt(tau);
      for (let n = 0; n < SAMPLE_COUNT; n += 1) {
        const offset = n * D;
        point.set(diffusion.subarray(offset, offset + D));
        predict(state.diffusionNet, point, tau, hidden, output);
        for (let j = 0; j < D; j += 1) {
          const clean = Math.max(-5, Math.min(5, (diffusion[offset + j] - noiseScale * output[j]) / cleanScale));
          diffusion[offset + j] = Math.sqrt(1 - nextTau) * clean + Math.sqrt(nextTau) * output[j];
        }
      }
      diffusionJourney.push(new Float32Array(diffusion.subarray(0, D)));
    }
    drawDigits(ui.diffusionSamples, diffusion, [205, 184, 255]);
    drawDigits(ui.flowSamples, flow, [163, 245, 215]);
    state.diffusionJourney = diffusionJourney; state.flowJourney = flowJourney;
    renderMicroscope();
  }

  const state = {
    diffusionNet: null, flowNet: null, indices: [], update: 0, budget: 1200, solverSteps: 24,
    running: false, frame: 0, trainingRng: null, trainingNormal: null,
    diffusionEma: null, flowEma: null, diffusionHistory: [], flowHistory: [],
    diffusionJourney: null, flowJourney: null
  };

  function setIndices() {
    const choice = ui.digit.value;
    state.indices = [];
    for (let i = 0; i < labels.length; i += 1) if (choice === "all" || labels[i] === Number(choice)) state.indices.push(i);
  }

  function updateUi() {
    ui.progress.max = state.budget; ui.progress.value = state.update;
    ui.progressLabel.textContent = state.update.toLocaleString() + " / " + state.budget.toLocaleString() + " updates";
    ui.diffusionLoss.textContent = state.diffusionEma === null ? "untrained" : state.diffusionEma.toFixed(4);
    ui.flowLoss.textContent = state.flowEma === null ? "untrained" : state.flowEma.toFixed(4);
    ui.train.disabled = state.running || state.update >= state.budget;
    ui.pause.disabled = !state.running;
    ui.train.textContent = state.update >= state.budget ? "Training complete" : state.update > 0 ? "Continue both" : "Train both";
  }

  function resetLearners() {
    clearTimeout(state.frame);
    state.running = false; state.update = 0; state.diffusionEma = null; state.flowEma = null;
    state.diffusionHistory = []; state.flowHistory = [];
    const seed = ((Number(ui.seed.value) || 20260719) ^ 0x71C3A95D) >>> 0;
    state.diffusionNet = makeNetwork(mulberry32(seed)); state.flowNet = makeNetwork(mulberry32(seed));
    state.trainingRng = mulberry32((seed ^ 0xB5297A4D) >>> 0); state.trainingNormal = normalSource(state.trainingRng);
    setIndices(); updateUi();
    drawLoss(ui.diffusionLossChart, state.diffusionHistory, "#7857b2");
    drawLoss(ui.flowLossChart, state.flowHistory, "#167d69");
    sampleModels();
    ui.status.textContent = "Ready · models are untrained · " + state.indices.length.toLocaleString() + " examples available";
  }

  function trainFrame() {
    if (!state.running) return;
    const count = Math.min(Number(ui.speed.value), state.budget - state.update);
    const learningRate = Math.pow(10, Number(ui.learningRate.value));
    for (let i = 0; i < count; i += 1) {
      const batch = makeBatch();
      const diffusionLoss = trainNetwork(state.diffusionNet, batch, "diffusion", learningRate);
      const flowLoss = trainNetwork(state.flowNet, batch, "flow", learningRate);
      state.diffusionEma = state.diffusionEma === null ? diffusionLoss : 0.96 * state.diffusionEma + 0.04 * diffusionLoss;
      state.flowEma = state.flowEma === null ? flowLoss : 0.96 * state.flowEma + 0.04 * flowLoss;
      state.update += 1;
      if (state.update % 10 === 0) {
        state.diffusionHistory.push({ step: state.update, loss: state.diffusionEma });
        state.flowHistory.push({ step: state.update, loss: state.flowEma });
      }
    }
    updateUi();
    if (state.update % 40 < count || state.update >= state.budget) {
      drawLoss(ui.diffusionLossChart, state.diffusionHistory, "#7857b2");
      drawLoss(ui.flowLossChart, state.flowHistory, "#167d69");
      sampleModels();
    }
    ui.status.textContent = "Training live · update " + state.update.toLocaleString() + " · " + state.indices.length.toLocaleString() + " digit latents";
    if (state.update >= state.budget) {
      state.running = false; updateUi(); ui.status.textContent = "Complete · adjust a knob, regenerate, or reset to compare again"; return;
    }
    state.frame = window.setTimeout(trainFrame, 0);
  }

  ui.train.addEventListener("click", () => { if (state.update >= state.budget) return; state.running = true; updateUi(); state.frame = window.setTimeout(trainFrame, 0); });
  ui.pause.addEventListener("click", () => { state.running = false; clearTimeout(state.frame); updateUi(); ui.status.textContent = "Paused at update " + state.update.toLocaleString(); });
  ui.reset.addEventListener("click", resetLearners);
  ui.digit.addEventListener("change", resetLearners);
  ui.updates.addEventListener("input", () => { state.budget = Number(ui.updates.value); ui.updatesOutput.value = state.budget.toLocaleString(); updateUi(); });
  ui.learningRate.addEventListener("input", () => { ui.learningRateOutput.value = Math.pow(10, Number(ui.learningRate.value)).toFixed(4); });
  ui.speed.addEventListener("input", () => { ui.speedOutput.value = ui.speed.value + " updates / frame"; });
  ui.solverSteps.addEventListener("input", () => { state.solverSteps = 2 ** Number(ui.solverSteps.value) + 8; ui.solverStepsOutput.value = state.solverSteps; sampleModels(); });
  ui.temperature.addEventListener("input", () => { ui.temperatureOutput.value = Number(ui.temperature.value).toFixed(2); sampleModels(); });
  ui.regenerate.addEventListener("click", () => { let seed = Number(ui.seed.value); if (!Number.isFinite(seed)) seed = 20260719; ui.seed.value = (seed + 1) >>> 0; sampleModels(); ui.status.textContent = "Regenerated from a new shared noise draw"; });
  ui.journey.addEventListener("input", renderMicroscope);
  window.addEventListener("resize", () => { drawLoss(ui.diffusionLossChart, state.diffusionHistory, "#7857b2"); drawLoss(ui.flowLossChart, state.flowHistory, "#167d69"); renderMicroscope(); });

  state.budget = Number(ui.updates.value);
  state.solverSteps = 2 ** Number(ui.solverSteps.value) + 8;
  ui.solverStepsOutput.value = state.solverSteps;
  resetLearners();
}());
