(function () {
  "use strict";

  const data = window.MNISTPixelData;
  const mount = document.getElementById("digit-flow-lab");
  if (!data || !mount) return;

  const $ = (id) => document.getElementById(id);
  const ui = {
    title: $("digit-flow-title"), summary: $("digit-flow-summary"), scopeCopy: $("digit-flow-scope-copy"),
    motionTitle: $("digit-flow-motion-title"), motionCopy: $("digit-flow-motion-copy"),
    source: $("digit-flow-source"), target: $("digit-flow-target"), pairing: $("digit-flow-pairing"),
    progress: $("digit-flow-progress"), progressLabel: $("digit-flow-progress-label"),
    diffusionLoss: $("digit-diffusion-loss"), flowLoss: $("digit-flow-loss"),
    train: $("digit-flow-train"), trainPause: $("digit-flow-train-pause"), reset: $("digit-flow-reset"),
    budget: $("digit-flow-budget"), budgetOutput: $("digit-flow-budget-output"),
    speed: $("digit-flow-speed"), speedOutput: $("digit-flow-speed-output"),
    steps: $("digit-flow-steps"), stepsOutput: $("digit-flow-steps-output"), status: $("digit-flow-status"),
    diffusionStage: $("digit-diffusion-stage"), flowStage: $("digit-flow-stage"),
    time: $("digit-flow-time"), timeLabel: $("digit-flow-time-label"),
    axisSource: $("digit-flow-axis-source"), axisTarget: $("digit-flow-axis-target"),
    play: $("digit-flow-play"), playPause: $("digit-flow-play-pause"),
    examples: $("digit-flow-examples"), selection: $("digit-flow-selection"),
    couplingCopy: $("digit-flow-coupling-copy"),
    diffusion: {
      inspectSource: $("digit-diffusion-inspect-source"), inspectCurrent: $("digit-diffusion-inspect-current"),
      inspectPrediction: $("digit-diffusion-inspect-prediction"), inspectNext: $("digit-diffusion-inspect-next"),
      inspectTarget: $("digit-diffusion-inspect-target"), sourceCaption: $("digit-diffusion-source-caption"),
      currentCaption: $("digit-diffusion-current-caption"), nextCaption: $("digit-diffusion-next-caption"),
      targetCaption: $("digit-diffusion-target-caption"), film: $("digit-diffusion-film"),
      filmCopy: $("digit-diffusion-film-copy")
    },
    flow: {
      inspectSource: $("digit-flow-inspect-source"), inspectCurrent: $("digit-flow-inspect-current"),
      inspectPrediction: $("digit-flow-inspect-prediction"), inspectNext: $("digit-flow-inspect-next"),
      inspectTarget: $("digit-flow-inspect-target"), sourceCaption: $("digit-flow-source-caption"),
      currentCaption: $("digit-flow-current-caption"), nextCaption: $("digit-flow-next-caption"),
      targetCaption: $("digit-flow-target-caption"), film: $("digit-flow-film")
    }
  };

  const D = data.pixelDim;
  const IMAGE_SIDE = data.imageSide;
  const H = 96;
  const INPUT = D * 2 + 1;
  const BATCH = 24;
  const LANE_COUNT = 6;
  const DIGIT_PLURALS = ["zeros", "ones", "twos", "threes", "fours", "fives", "sixes", "sevens", "eights", "nines"];
  const PARAMETER_COUNT = H * INPUT + H + D * H + D + 3 * D;
  const DIFFUSION_TINT = [205, 184, 255];
  const FLOW_TINT = [134, 240, 210];
  const SOURCE_TINT = [198, 210, 203];
  const TARGET_TINT = [241, 215, 142];

  function decodeBase64(value) {
    const raw = atob(value);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i += 1) bytes[i] = raw.charCodeAt(i);
    return bytes;
  }

  const pixelBytes = decodeBase64(data.pixels);
  const labels = decodeBase64(data.labels);
  const indicesByDigit = Array.from({ length: 10 }, () => []);
  for (let i = 0; i < labels.length; i += 1) indicesByDigit[labels[i]].push(i);

  const digitMeans = Array.from({ length: 10 }, () => new Float32Array(D));
  for (let digit = 0; digit < 10; digit += 1) {
    for (let i = 0; i < indicesByDigit[digit].length; i += 1) {
      const base = indicesByDigit[digit][i] * D;
      for (let j = 0; j < D; j += 1) digitMeans[digit][j] += pixelBytes[base + j] / 127.5 - 1;
    }
    for (let j = 0; j < D; j += 1) digitMeans[digit][j] /= indicesByDigit[digit].length;
  }

  const canonicalRows = {
    0: ["..####..", ".##..##.", ".##..##.", ".##..##.", ".##..##.", ".##..##.", ".##..##.", "..####.."],
    5: [".######.", ".##.....", ".##.....", ".#####..", "....##..", "....##..", ".#..##..", "..###..."]
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

  function normalSource(rng) {
    let spare = null;
    return function () {
      if (spare !== null) {
        const value = spare; spare = null; return value;
      }
      const radius = Math.sqrt(-2 * Math.log(Math.max(1e-9, rng())));
      const angle = Math.PI * 2 * rng();
      spare = radius * Math.sin(angle);
      return radius * Math.cos(angle);
    };
  }

  function shuffled(values, rng) {
    const result = values.slice();
    for (let i = result.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rng() * (i + 1));
      const temporary = result[i]; result[i] = result[j]; result[j] = temporary;
    }
    return result;
  }

  function imageAt(index, out, offset) {
    const base = index * D;
    for (let j = 0; j < D; j += 1) out[offset + j] = pixelBytes[base + j] / 127.5 - 1;
  }

  function pixelDistance(firstIndex, secondIndex) {
    const first = firstIndex * D, second = secondIndex * D;
    let total = 0;
    for (let j = 0; j < D; j += 1) {
      const difference = pixelBytes[first + j] - pixelBytes[second + j];
      total += difference * difference;
    }
    return total;
  }

  function representativeDistance(index, digit) {
    const base = index * D, mean = digitMeans[digit];
    let total = 0;
    for (let j = 0; j < D; j += 1) {
      const difference = pixelBytes[base + j] / 127.5 - 1 - mean[j];
      total += difference * difference;
    }
    return total;
  }

  function displayDistance(index, digit) {
    const rows = canonicalRows[digit];
    if (!rows) return representativeDistance(index, digit);
    const base = index * D;
    let templateDistance = 0;
    for (let y = 0; y < IMAGE_SIDE; y += 1) {
      for (let x = 0; x < IMAGE_SIDE; x += 1) {
        const target = rows[y][x] === "#" ? 1 : -1;
        const difference = pixelBytes[base + y * IMAGE_SIDE + x] / 127.5 - 1 - target;
        templateDistance += difference * difference;
      }
    }
    return 0.35 * representativeDistance(index, digit) + 0.65 * templateDistance;
  }

  function makePairs(sourceDigit, targetDigit, strategy) {
    const seed = (0xA341316C ^ Math.imul(sourceDigit, 0x9E3779B1) ^ Math.imul(targetDigit, 0x85EBCA77)) >>> 0;
    const rng = mulberry32(seed);
    const sources = shuffled(indicesByDigit[sourceDigit], rng);
    const targets = shuffled(indicesByDigit[targetDigit], rng);
    const count = Math.min(sources.length, targets.length);
    if (strategy === "random") {
      const result = new Array(count);
      for (let i = 0; i < count; i += 1) result[i] = [sources[i], targets[i]];
      return result;
    }

    const available = new Uint8Array(targets.length); available.fill(1);
    const result = new Array(count);
    for (let i = 0; i < count; i += 1) {
      let best = -1, bestDistance = Infinity;
      for (let k = 0; k < targets.length; k += 1) {
        if (!available[k]) continue;
        const distance = pixelDistance(sources[i], targets[k]);
        if (distance < bestDistance) { bestDistance = distance; best = k; }
      }
      available[best] = 0;
      result[i] = [sources[i], targets[best]];
    }
    return result;
  }

  function makeNetwork(rng) {
    const w1 = new Float32Array(H * INPUT), b1 = new Float32Array(H);
    const w2 = new Float32Array(D * H), b2 = new Float32Array(D);
    const skip0 = new Float32Array(D), skip1 = new Float32Array(D), skip2 = new Float32Array(D);
    const limit1 = Math.sqrt(6 / (INPUT + H)), limit2 = Math.sqrt(6 / (H + D));
    for (let i = 0; i < w1.length; i += 1) w1[i] = (rng() * 2 - 1) * limit1;
    for (let i = 0; i < w2.length; i += 1) w2[i] = (rng() * 2 - 1) * limit2;
    const parameters = [w1, b1, w2, b2, skip0, skip1, skip2];
    return {
      w1, b1, w2, b2, skip0, skip1, skip2, step: 0,
      m: parameters.map((parameter) => new Float32Array(parameter.length)),
      v: parameters.map((parameter) => new Float32Array(parameter.length)),
      g: parameters.map((parameter) => new Float32Array(parameter.length))
    };
  }

  function predict(net, point, sourceCondition, time, hidden, output) {
    for (let k = 0; k < H; k += 1) {
      const row = k * INPUT;
      let sum = net.b1[k] + net.w1[row + D * 2] * time;
      for (let j = 0; j < D; j += 1) sum += net.w1[row + j] * point[j];
      for (let j = 0; j < D; j += 1) sum += net.w1[row + D + j] * sourceCondition[j];
      hidden[k] = Math.tanh(sum);
    }
    for (let j = 0; j < D; j += 1) {
      const skip = net.skip0[j] + time * net.skip1[j] + time * time * net.skip2[j];
      let sum = net.b2[j] + skip * point[j];
      const row = j * H;
      for (let k = 0; k < H; k += 1) sum += net.w2[row + k] * hidden[k];
      output[j] = sum;
    }
  }

  function adam(net, learningRate) {
    net.step += 1;
    const correction1 = 1 - Math.pow(0.9, net.step);
    const correction2 = 1 - Math.pow(0.999, net.step);
    const parameters = [net.w1, net.b1, net.w2, net.b2, net.skip0, net.skip1, net.skip2];
    for (let group = 0; group < parameters.length; group += 1) {
      const parameter = parameters[group], gradient = net.g[group], mean = net.m[group], variance = net.v[group];
      for (let i = 0; i < parameter.length; i += 1) {
        const clipped = Math.max(-5, Math.min(5, gradient[i]));
        mean[i] = 0.9 * mean[i] + 0.1 * clipped;
        variance[i] = 0.999 * variance[i] + 0.001 * clipped * clipped;
        parameter[i] -= learningRate * (mean[i] / correction1) / (Math.sqrt(variance[i] / correction2) + 1e-8);
      }
    }
  }

  function makeBatch() {
    const source = new Float32Array(BATCH * D), target = new Float32Array(BATCH * D);
    const noise = new Float32Array(BATCH * D), times = new Float32Array(BATCH);
    for (let item = 0; item < BATCH; item += 1) {
      const pair = state.pairs[Math.floor(state.trainingRng() * state.pairs.length)];
      const offset = item * D;
      imageAt(pair[0], source, offset); imageAt(pair[1], target, offset);
      times[item] = 0.02 + 0.96 * state.trainingRng();
      for (let j = 0; j < D; j += 1) noise[offset + j] = state.trainingNormal();
    }
    return { source, target, noise, times };
  }

  function trainNetwork(net, batch, method, learningRate) {
    net.g.forEach((gradient) => gradient.fill(0));
    const source = new Float32Array(D), input = new Float32Array(D), desired = new Float32Array(D);
    const hidden = new Float32Array(H), output = new Float32Array(D), hiddenGradient = new Float32Array(H);
    let loss = 0;
    for (let item = 0; item < BATCH; item += 1) {
      const offset = item * D, time = batch.times[item];
      const cleanScale = Math.sqrt(1 - time), noiseScale = Math.sqrt(time);
      for (let j = 0; j < D; j += 1) {
        source[j] = batch.source[offset + j];
        if (method === "diffusion") {
          input[j] = cleanScale * batch.target[offset + j] + noiseScale * batch.noise[offset + j];
          desired[j] = batch.noise[offset + j];
        } else {
          input[j] = (1 - time) * batch.source[offset + j] + time * batch.target[offset + j];
          desired[j] = batch.target[offset + j] - batch.source[offset + j];
        }
      }
      predict(net, input, source, time, hidden, output);
      hiddenGradient.fill(0);
      for (let j = 0; j < D; j += 1) {
        const error = output[j] - desired[j];
        loss += error * error;
        const derivative = 2 * error / (BATCH * D);
        net.g[3][j] += derivative;
        net.g[4][j] += derivative * input[j];
        net.g[5][j] += derivative * input[j] * time;
        net.g[6][j] += derivative * input[j] * time * time;
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
        for (let j = 0; j < D; j += 1) net.g[0][row + D + j] += derivative * source[j];
        net.g[0][row + D * 2] += derivative * time;
      }
    }
    adam(net, learningRate);
    return loss / (BATCH * D);
  }

  function visibleValue(value) {
    return Math.max(0, Math.min(1, (value + 1) / 2));
  }

  function paintPixelImage(image, imageWidth, point, cellX, cellY, tint) {
    for (let y = 0; y < IMAGE_SIDE; y += 1) {
      for (let x = 0; x < IMAGE_SIDE; x += 1) {
        const value = visibleValue(point[y * IMAGE_SIDE + x]);
        const offset = ((cellY + y) * imageWidth + cellX + x) * 4;
        image.data[offset] = Math.round(tint[0] * value);
        image.data[offset + 1] = Math.round(tint[1] * value);
        image.data[offset + 2] = Math.round(tint[2] * value);
        image.data[offset + 3] = 255;
      }
    }
  }

  function drawPixelCanvas(canvas, point, tint) {
    canvas.width = IMAGE_SIDE; canvas.height = IMAGE_SIDE;
    const context = canvas.getContext("2d");
    const image = context.createImageData(IMAGE_SIDE, IMAGE_SIDE);
    paintPixelImage(image, IMAGE_SIDE, point, 0, 0, tint);
    context.putImageData(image, 0, 0);
  }

  function signalScale(point) {
    let scale = 1;
    for (let j = 0; j < D; j += 1) scale = Math.max(scale, Math.abs(point[j]));
    return scale;
  }

  function drawSignedCanvas(canvas, point) {
    canvas.width = IMAGE_SIDE; canvas.height = IMAGE_SIDE;
    const context = canvas.getContext("2d");
    const image = context.createImageData(IMAGE_SIDE, IMAGE_SIDE);
    const scale = signalScale(point), neutral = [15, 21, 18], positive = [86, 190, 235], negative = [235, 132, 69];
    for (let j = 0; j < D; j += 1) {
      const strength = Math.min(1, Math.abs(point[j]) / scale);
      const color = point[j] >= 0 ? positive : negative;
      const offset = j * 4;
      image.data[offset] = Math.round(neutral[0] + strength * (color[0] - neutral[0]));
      image.data[offset + 1] = Math.round(neutral[1] + strength * (color[1] - neutral[1]));
      image.data[offset + 2] = Math.round(neutral[2] + strength * (color[2] - neutral[2]));
      image.data[offset + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }

  function drawTile(context, point, centerX, centerY, size, tint, alpha, border) {
    const pixelSize = size / IMAGE_SIDE;
    const left = centerX - size / 2, top = centerY - size / 2;
    context.save(); context.globalAlpha = alpha;
    context.fillStyle = "#0c120f"; context.fillRect(left - 2, top - 2, size + 4, size + 4);
    for (let y = 0; y < IMAGE_SIDE; y += 1) {
      for (let x = 0; x < IMAGE_SIDE; x += 1) {
        const value = visibleValue(point[y * IMAGE_SIDE + x]);
        context.fillStyle = "rgb(" + Math.round(tint[0] * value) + "," + Math.round(tint[1] * value) + "," + Math.round(tint[2] * value) + ")";
        context.fillRect(left + x * pixelSize, top + y * pixelSize, Math.ceil(pixelSize), Math.ceil(pixelSize));
      }
    }
    if (border) {
      context.strokeStyle = border; context.lineWidth = 2;
      context.strokeRect(left - 3, top - 3, size + 6, size + 6);
    }
    context.restore();
  }

  function chooseExamples() {
    const representative = state.pairs.slice().sort((first, second) => {
      const firstScore = displayDistance(first[0], state.sourceDigit) + 1.5 * displayDistance(first[1], state.targetDigit);
      const secondScore = displayDistance(second[0], state.sourceDigit) + 1.5 * displayDistance(second[1], state.targetDigit);
      return firstScore - secondScore;
    });
    const pool = representative.slice(0, Math.max(LANE_COUNT, Math.floor(representative.length * 0.1)));
    const offset = state.exampleSeed % Math.max(1, pool.length - LANE_COUNT + 1);
    state.examples = pool.slice(offset, offset + LANE_COUNT);
  }

  function laneNoise(pair, lane) {
    const seed = (0xD1B54A35 ^ Math.imul(pair[0] + 1, 0x9E3779B1) ^ Math.imul(pair[1] + 1, 0x85EBCA77) ^ Math.imul(state.exampleSeed + lane + 1, 0xC2B2AE3D)) >>> 0;
    const normal = normalSource(mulberry32(seed));
    const point = new Float32Array(D);
    for (let j = 0; j < D; j += 1) point[j] = normal();
    return point;
  }

  function computeJourneys() {
    if (!state.examples.length || !state.diffusionNet || !state.flowNet) return;
    state.diffusionJourneys = []; state.flowJourneys = [];
    const hidden = new Float32Array(H), output = new Float32Array(D);
    for (let lane = 0; lane < LANE_COUNT; lane += 1) {
      const pair = state.examples[lane], source = new Float32Array(D);
      imageAt(pair[0], source, 0);

      const flowPoint = new Float32Array(source);
      const flowJourney = [new Float32Array(flowPoint)];
      for (let step = 0; step < state.solverSteps; step += 1) {
        predict(state.flowNet, flowPoint, source, (step + 0.5) / state.solverSteps, hidden, output);
        for (let j = 0; j < D; j += 1) flowPoint[j] += output[j] / state.solverSteps;
        flowJourney.push(new Float32Array(flowPoint));
      }
      state.flowJourneys.push(flowJourney);

      const diffusionPoint = laneNoise(pair, lane);
      const diffusionJourney = [new Float32Array(diffusionPoint)];
      for (let step = 0; step < state.solverSteps; step += 1) {
        const tau = 0.98 * (1 - step / state.solverSteps);
        const nextTau = 0.98 * (1 - (step + 1) / state.solverSteps);
        predict(state.diffusionNet, diffusionPoint, source, tau, hidden, output);
        const cleanScale = Math.sqrt(1 - tau), noiseScale = Math.sqrt(tau);
        const nextCleanScale = Math.sqrt(1 - nextTau), nextNoiseScale = Math.sqrt(nextTau);
        for (let j = 0; j < D; j += 1) {
          const clean = Math.max(-1.5, Math.min(1.5, (diffusionPoint[j] - noiseScale * output[j]) / cleanScale));
          diffusionPoint[j] = nextCleanScale * clean + nextNoiseScale * output[j];
        }
        diffusionJourney.push(new Float32Array(diffusionPoint));
      }
      state.diffusionJourneys.push(diffusionJourney);
    }
    renderMotion();
  }

  function stateAt(journey, progress, out) {
    const position = progress * (journey.length - 1);
    const first = Math.min(journey.length - 1, Math.floor(position));
    const second = Math.min(journey.length - 1, first + 1);
    const fraction = position - first;
    for (let j = 0; j < D; j += 1) out[j] = journey[first][j] * (1 - fraction) + journey[second][j] * fraction;
    return out;
  }

  function stageLayout(width, height) {
    const compact = width < 560;
    const tile = compact ? 31 : 46;
    const top = compact ? 44 : 52;
    const bottom = 14;
    const rowHeight = (height - top - bottom) / LANE_COUNT;
    const sourceX = compact ? 34 : 42;
    const targetX = width - sourceX;
    const trackStart = sourceX + tile + (compact ? 12 : 25);
    const trackEnd = targetX - tile - (compact ? 12 : 25);
    return { tile, top, rowHeight, sourceX, targetX, trackStart, trackEnd };
  }

  function drawStage(canvas, journeys, method) {
    if (!journeys.length) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    const context = canvas.getContext("2d");
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    const width = rect.width, height = rect.height, layout = stageLayout(width, height);
    const diffusion = method === "diffusion";
    const tint = diffusion ? DIFFUSION_TINT : FLOW_TINT;
    const selectedBorder = diffusion ? "#eadfff" : "#eafbf5";
    const dimBorder = diffusion ? "#8066a6" : "#4e9d87";
    context.fillStyle = "#111814"; context.fillRect(0, 0, width, height);
    context.font = "700 9px system-ui, sans-serif"; context.textAlign = "center";
    context.fillStyle = "#aebbb4";
    context.fillText((diffusion ? "CONDITION " : "START ") + state.sourceDigit, layout.sourceX, 21);
    context.fillText(diffusion ? "DENOISING STATE" : "FLOWING STATE", (layout.trackStart + layout.trackEnd) / 2, 21);
    context.fillText("PAIRED " + state.targetDigit, layout.targetX, 21);

    const current = new Float32Array(D), ghost = new Float32Array(D), source = new Float32Array(D), target = new Float32Array(D);
    for (let lane = 0; lane < LANE_COUNT; lane += 1) {
      const centerY = layout.top + (lane + 0.5) * layout.rowHeight;
      if (lane === state.selectedLane) {
        context.fillStyle = diffusion ? "rgba(166,130,214,0.10)" : "rgba(72,190,154,0.09)";
        context.fillRect(0, centerY - layout.rowHeight / 2, width, layout.rowHeight);
      }
      context.strokeStyle = lane === state.selectedLane
        ? (diffusion ? "rgba(210,187,245,0.58)" : "rgba(134,220,196,0.55)")
        : "rgba(174,187,180,0.22)";
      context.lineWidth = lane === state.selectedLane ? 2 : 1;
      context.setLineDash([4, 5]);
      context.beginPath(); context.moveTo(layout.trackStart, centerY); context.lineTo(layout.trackEnd, centerY); context.stroke();
      context.setLineDash([]);

      imageAt(state.examples[lane][0], source, 0); imageAt(state.examples[lane][1], target, 0);
      drawTile(context, source, layout.sourceX, centerY, layout.tile, SOURCE_TINT, 1, lane === state.selectedLane ? selectedBorder : null);
      drawTile(context, target, layout.targetX, centerY, layout.tile, TARGET_TINT, 1, null);
      [0.25, 0.5, 0.75].forEach((checkpoint) => {
        stateAt(journeys[lane], checkpoint, ghost);
        const ghostX = layout.trackStart + checkpoint * (layout.trackEnd - layout.trackStart);
        drawTile(context, ghost, ghostX, centerY, layout.tile * 0.78, tint, 0.15, null);
      });
      stateAt(journeys[lane], state.motionProgress, current);
      const movingX = layout.trackStart + state.motionProgress * (layout.trackEnd - layout.trackStart);
      drawTile(context, current, movingX, centerY, layout.tile, tint, 1, lane === state.selectedLane ? selectedBorder : dimBorder);
      context.fillStyle = lane === state.selectedLane ? "#f7f5fb" : "#75837b";
      context.font = "10px monospace"; context.textAlign = "left";
      context.fillText(String(lane + 1), 4, centerY + 3);
    }
  }

  function drawFilm(canvas, journey, tint, border) {
    const count = 7;
    canvas.width = count * IMAGE_SIDE; canvas.height = IMAGE_SIDE;
    const context = canvas.getContext("2d"), image = context.createImageData(canvas.width, IMAGE_SIDE);
    for (let checkpoint = 0; checkpoint < count; checkpoint += 1) {
      const index = Math.round((journey.length - 1) * checkpoint / (count - 1));
      paintPixelImage(image, canvas.width, journey[index], checkpoint * IMAGE_SIDE, 0, tint);
    }
    context.putImageData(image, 0, 0);
    const active = Math.round(state.motionProgress * (count - 1));
    context.strokeStyle = border; context.lineWidth = 1;
    context.strokeRect(active * IMAGE_SIDE + 0.5, 0.5, IMAGE_SIDE - 1, IMAGE_SIDE - 1);
  }

  function renderInspection(method) {
    const journeys = method === "diffusion" ? state.diffusionJourneys : state.flowJourneys;
    if (!journeys.length) return;
    const refs = ui[method], pair = state.examples[state.selectedLane], journey = journeys[state.selectedLane];
    const source = new Float32Array(D), target = new Float32Array(D), current = new Float32Array(D);
    const prediction = new Float32Array(D), next = new Float32Array(D), hidden = new Float32Array(H);
    imageAt(pair[0], source, 0); imageAt(pair[1], target, 0); stateAt(journey, state.motionProgress, current);

    if (method === "diffusion") {
      const tau = 0.98 * (1 - state.motionProgress);
      predict(state.diffusionNet, current, source, tau, hidden, prediction);
      if (state.motionProgress >= 1) {
        next.set(current);
      } else {
        const nextProgress = Math.min(1, state.motionProgress + 1 / state.solverSteps);
        const nextTau = 0.98 * (1 - nextProgress);
        const cleanScale = Math.sqrt(1 - tau), noiseScale = Math.sqrt(tau);
        for (let j = 0; j < D; j += 1) {
          const clean = Math.max(-1.5, Math.min(1.5, (current[j] - noiseScale * prediction[j]) / cleanScale));
          next[j] = Math.sqrt(1 - nextTau) * clean + Math.sqrt(nextTau) * prediction[j];
        }
      }
      refs.currentCaption.textContent = state.motionProgress >= 1 ? "clean endpoint · τ = 0.00" : "noise level τ = " + tau.toFixed(2);
      refs.nextCaption.textContent = state.motionProgress >= 1 ? "journey complete" : "next τ = " + (0.98 * (1 - Math.min(1, state.motionProgress + 1 / state.solverSteps))).toFixed(2);
    } else {
      const time = Math.min(0.999, state.motionProgress);
      predict(state.flowNet, current, source, time, hidden, prediction);
      const delta = state.motionProgress >= 1 ? 0 : Math.min(1 / state.solverSteps, 1 - state.motionProgress);
      for (let j = 0; j < D; j += 1) next[j] = current[j] + delta * prediction[j];
      refs.currentCaption.textContent = "x at t = " + state.motionProgress.toFixed(2);
      refs.nextCaption.textContent = state.motionProgress >= 1 ? "journey complete" : "Δt = " + delta.toFixed(3);
    }

    const tint = method === "diffusion" ? DIFFUSION_TINT : FLOW_TINT;
    drawPixelCanvas(refs.inspectSource, source, SOURCE_TINT);
    drawPixelCanvas(refs.inspectCurrent, current, tint);
    drawSignedCanvas(refs.inspectPrediction, prediction);
    drawPixelCanvas(refs.inspectNext, next, tint);
    drawPixelCanvas(refs.inspectTarget, target, TARGET_TINT);
    refs.sourceCaption.textContent = "source " + state.sourceDigit;
    refs.targetCaption.textContent = "paired " + state.targetDigit;
    drawFilm(refs.film, journey, tint, method === "diffusion" ? "#eadfff" : "#eafbf5");
  }

  function renderMotion() {
    const progress = state.motionProgress;
    ui.time.value = Math.round(progress * 1000);
    if (progress <= 0) ui.timeLabel.textContent = "0% · diffusion has noise; flow has source " + state.sourceDigit;
    else if (progress >= 1) ui.timeLabel.textContent = "100% · both aim for target " + state.targetDigit;
    else ui.timeLabel.textContent = Math.round(progress * 100) + "% · diffusion τ = " + (0.98 * (1 - progress)).toFixed(2) + " · flow t = " + progress.toFixed(2);
    ui.selection.textContent = "Inspecting row " + (state.selectedLane + 1) + " of " + LANE_COUNT;
    drawStage(ui.diffusionStage, state.diffusionJourneys, "diffusion");
    drawStage(ui.flowStage, state.flowJourneys, "flow");
    renderInspection("diffusion"); renderInspection("flow");
  }

  function updateLabels() {
    ui.title.textContent = "Two ways to learn " + state.sourceDigit + " → " + state.targetDigit;
    ui.summary.textContent = "Both learners see the same paired " + DIGIT_PLURALS[state.sourceDigit] + " and " + DIGIT_PLURALS[state.targetDigit] + ". Conditional diffusion learns to remove noise from a " + state.targetDigit + " while reading the " + state.sourceDigit + " as context; flow matching learns the pixel velocity that moves the " + state.sourceDigit + " itself toward a " + state.targetDigit + ".";
    ui.scopeCopy.textContent = "Diffusion begins from an independent random canvas; its source " + state.sourceDigit + " stays beside the journey as a fixed condition. Flow matching begins from the " + state.sourceDigit + " itself. Both use identical " + PARAMETER_COUNT.toLocaleString() + "-parameter networks, paired batches, update budgets, and solver-call counts. They aim for the same target even though the routes and training labels differ.";
    ui.motionTitle.textContent = "The same six " + DIGIT_PLURALS[state.sourceDigit] + ", transformed two different ways";
    ui.motionCopy.textContent = "Each row uses the same source " + state.sourceDigit + " and paired target " + state.targetDigit + " in both columns. Horizontal travel is only a shared clock; the changing 8×8 pixels are the model states. Click either copy of a row to inspect it below.";
    ui.diffusion.filmCopy.textContent = "source " + state.sourceDigit + " remains fixed context";
    ui.axisSource.textContent = "diffusion: noise · flow: source " + state.sourceDigit;
    ui.axisTarget.textContent = "both aim for target " + state.targetDigit;
    const pairing = state.pairing === "matched"
      ? "Closest-looking pairing gives both learners a relatively unambiguous relationship to learn."
      : "Random pairing asks both learners to reconcile many less-consistent source-to-target relationships.";
    ui.couplingCopy.textContent = "The endpoints are deliberately not the punchline: both models aim for plausible " + DIGIT_PLURALS[state.targetDigit] + ". Compare the beginning and middle. Diffusion reveals structure by repeatedly removing noise from a random canvas while consulting the " + state.sourceDigit + "; flow preserves and reshapes structure already present in it. At this tiny live-training budget diffusion may remain visibly rougher; that is an optimization result, not a different goal. " + pairing;
  }

  function updateTrainingUi() {
    ui.progress.max = state.budget; ui.progress.value = state.update;
    ui.progressLabel.textContent = state.update.toLocaleString() + " / " + state.budget.toLocaleString() + " updates";
    ui.diffusionLoss.value = state.diffusionEma === null ? "untrained" : state.diffusionEma.toFixed(4);
    ui.flowLoss.value = state.flowEma === null ? "untrained" : state.flowEma.toFixed(4);
    ui.train.disabled = state.training || state.update >= state.budget;
    ui.trainPause.disabled = !state.training;
    ui.train.textContent = state.update >= state.budget ? "Training complete" : state.update > 0 ? "Continue both" : "Train both";
  }

  function scheduledLearningRate() {
    const progress = Math.min(1, (state.update + 1) / Math.max(1, state.budget));
    return 0.02 * (0.08 + 0.92 * 0.5 * (1 + Math.cos(Math.PI * progress)));
  }

  function stopTraining() {
    state.training = false; clearTimeout(state.trainingTimer); updateTrainingUi();
  }

  function trainFrame() {
    if (!state.training) return;
    const count = Math.min(Number(ui.speed.value), state.budget - state.update);
    let learningRate = scheduledLearningRate();
    for (let i = 0; i < count; i += 1) {
      learningRate = scheduledLearningRate();
      const batch = makeBatch();
      const diffusionLoss = trainNetwork(state.diffusionNet, batch, "diffusion", learningRate);
      const flowLoss = trainNetwork(state.flowNet, batch, "flow", learningRate);
      state.diffusionEma = state.diffusionEma === null ? diffusionLoss : 0.96 * state.diffusionEma + 0.04 * diffusionLoss;
      state.flowEma = state.flowEma === null ? flowLoss : 0.96 * state.flowEma + 0.04 * flowLoss;
      state.update += 1;
    }
    updateTrainingUi();
    if (state.update % 80 < count || state.update >= state.budget) computeJourneys();
    ui.status.textContent = "Training both · update " + state.update.toLocaleString() + " · diffusion noise MSE " + state.diffusionEma.toFixed(4) + " · flow velocity MSE " + state.flowEma.toFixed(4) + " · lr " + learningRate.toFixed(4);
    if (state.update >= state.budget) {
      stopTraining(); computeJourneys();
      ui.status.textContent = "Training complete · press Play once to compare both " + state.sourceDigit + " → " + state.targetDigit + " journeys";
      return;
    }
    state.trainingTimer = window.setTimeout(trainFrame, 0);
  }

  function stopMotion(paused) {
    state.playing = false;
    if (state.animationFrame) window.cancelAnimationFrame(state.animationFrame);
    state.animationFrame = 0;
    ui.playPause.disabled = true;
    ui.play.textContent = state.motionProgress >= 1 ? "Replay both" : "Play once";
    if (paused) ui.status.textContent = "Motion paused at " + Math.round(state.motionProgress * 100) + "%";
  }

  function motionFrame(now) {
    if (!state.playing) return;
    const elapsed = now - state.motionStartedAt;
    const progress = Math.min(1, state.motionStartProgress + elapsed / state.motionDuration);
    state.motionProgress = progress; renderMotion();
    if (progress >= 1) {
      stopMotion(false);
      ui.status.textContent = "Journeys complete · diffusion travelled noise → " + state.targetDigit + "; flow travelled " + state.sourceDigit + " → " + state.targetDigit;
      return;
    }
    state.animationFrame = window.requestAnimationFrame(motionFrame);
  }

  function playMotion() {
    if (state.playing) return;
    if (state.motionProgress >= 1) state.motionProgress = 0;
    state.playing = true;
    state.motionStartProgress = state.motionProgress;
    state.motionStartedAt = performance.now();
    state.motionDuration = Math.max(500, 5200 * (1 - state.motionStartProgress));
    ui.play.textContent = "Playing once…"; ui.playPause.disabled = false;
    state.animationFrame = window.requestAnimationFrame(motionFrame);
  }

  function resetLearners() {
    stopTraining(); stopMotion(false);
    state.update = 0; state.diffusionEma = null; state.flowEma = null; state.motionProgress = 0;
    const seed = (0x71C3A95D ^ Math.imul(state.sourceDigit, 0x9E3779B1) ^ Math.imul(state.targetDigit, 0x85EBCA77) ^ (state.pairing === "random" ? 0x27D4EB2F : 0)) >>> 0;
    state.diffusionNet = makeNetwork(mulberry32(seed));
    state.flowNet = makeNetwork(mulberry32(seed));
    state.trainingRng = mulberry32((seed ^ 0xB5297A4D) >>> 0);
    state.trainingNormal = normalSource(mulberry32((seed ^ 0x68E31DA4) >>> 0));
    chooseExamples(); updateLabels(); updateTrainingUi(); computeJourneys();
    ui.status.textContent = "Ready · " + state.pairs.length.toLocaleString() + " shared " + (state.pairing === "matched" ? "closest-looking" : "random") + " pairs · " + PARAMETER_COUNT.toLocaleString() + " parameters each · both models untrained";
  }

  function configureExperiment(changedControl) {
    stopTraining(); stopMotion(false);
    let source = Number(ui.source.value), target = Number(ui.target.value);
    if (source === target) {
      if (changedControl === "source") { target = (source + 1) % 10; ui.target.value = String(target); }
      else { source = (target + 9) % 10; ui.source.value = String(source); }
    }
    state.sourceDigit = source; state.targetDigit = target; state.pairing = ui.pairing.value;
    ui.status.textContent = "Building shared " + (state.pairing === "matched" ? "closest-looking" : "random") + " " + source + " → " + target + " pairs…";
    state.pairs = makePairs(source, target, state.pairing);
    resetLearners();
  }

  function selectLane(event, canvas) {
    const rect = canvas.getBoundingClientRect(), layout = stageLayout(rect.width, rect.height);
    const lane = Math.floor((event.clientY - rect.top - layout.top) / layout.rowHeight);
    if (lane >= 0 && lane < LANE_COUNT) { state.selectedLane = lane; renderMotion(); }
  }

  const state = {
    sourceDigit: 0, targetDigit: 5, pairing: "matched", pairs: [], examples: [],
    diffusionJourneys: [], flowJourneys: [], diffusionNet: null, flowNet: null,
    trainingRng: null, trainingNormal: null, update: 0, budget: Number(ui.budget.value),
    diffusionEma: null, flowEma: null, solverSteps: Number(ui.steps.value),
    selectedLane: 0, exampleSeed: 0, training: false, trainingTimer: 0,
    playing: false, animationFrame: 0, motionProgress: 0, motionStartProgress: 0,
    motionStartedAt: 0, motionDuration: 5200
  };

  ui.source.addEventListener("change", () => configureExperiment("source"));
  ui.target.addEventListener("change", () => configureExperiment("target"));
  ui.pairing.addEventListener("change", () => configureExperiment("pairing"));
  ui.train.addEventListener("click", () => {
    if (state.update >= state.budget) return;
    state.training = true; updateTrainingUi(); state.trainingTimer = window.setTimeout(trainFrame, 0);
  });
  ui.trainPause.addEventListener("click", () => {
    stopTraining(); computeJourneys(); ui.status.textContent = "Both learners paused at update " + state.update.toLocaleString();
  });
  ui.reset.addEventListener("click", resetLearners);
  ui.budget.addEventListener("input", () => {
    state.budget = Number(ui.budget.value); ui.budgetOutput.value = state.budget.toLocaleString();
    if (state.training && state.update >= state.budget) stopTraining();
    updateTrainingUi();
  });
  ui.speed.addEventListener("input", () => { ui.speedOutput.value = ui.speed.value + " / frame"; });
  ui.steps.addEventListener("input", () => {
    state.solverSteps = Number(ui.steps.value); ui.stepsOutput.value = state.solverSteps; computeJourneys();
  });
  ui.time.addEventListener("input", () => {
    stopMotion(false); state.motionProgress = Number(ui.time.value) / 1000; renderMotion();
  });
  ui.play.addEventListener("click", playMotion);
  ui.playPause.addEventListener("click", () => stopMotion(true));
  ui.examples.addEventListener("click", () => {
    stopMotion(false); state.exampleSeed = (state.exampleSeed + LANE_COUNT) >>> 0; state.motionProgress = 0;
    chooseExamples(); computeJourneys(); ui.status.textContent = "Loaded six new shared source/target rows; both trained models are unchanged";
  });
  ui.diffusionStage.addEventListener("click", (event) => selectLane(event, ui.diffusionStage));
  ui.flowStage.addEventListener("click", (event) => selectLane(event, ui.flowStage));
  window.addEventListener("resize", () => {
    drawStage(ui.diffusionStage, state.diffusionJourneys, "diffusion");
    drawStage(ui.flowStage, state.flowJourneys, "flow");
  });

  configureExperiment("pairing");
}());
