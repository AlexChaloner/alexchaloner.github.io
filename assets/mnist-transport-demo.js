(function () {
  "use strict";

  const data = window.MNISTPixelData;
  const mount = document.getElementById("digit-flow-lab");
  if (!data || !mount) return;

  const $ = (id) => document.getElementById(id);
  const ui = {
    title: $("digit-flow-title"), summary: $("digit-flow-summary"), motionTitle: $("digit-flow-motion-title"),
    source: $("digit-flow-source"), target: $("digit-flow-target"), pairing: $("digit-flow-pairing"),
    progress: $("digit-flow-progress"), progressLabel: $("digit-flow-progress-label"),
    train: $("digit-flow-train"), trainPause: $("digit-flow-train-pause"), reset: $("digit-flow-reset"),
    budget: $("digit-flow-budget"), budgetOutput: $("digit-flow-budget-output"),
    speed: $("digit-flow-speed"), speedOutput: $("digit-flow-speed-output"),
    steps: $("digit-flow-steps"), stepsOutput: $("digit-flow-steps-output"), status: $("digit-flow-status"),
    stage: $("digit-flow-stage"), time: $("digit-flow-time"), timeLabel: $("digit-flow-time-label"),
    axisSource: $("digit-flow-axis-source"), axisTarget: $("digit-flow-axis-target"),
    play: $("digit-flow-play"), playPause: $("digit-flow-play-pause"), examples: $("digit-flow-examples"), selection: $("digit-flow-selection"),
    inspectSource: $("digit-flow-inspect-source"), inspectCurrent: $("digit-flow-inspect-current"),
    inspectVelocity: $("digit-flow-inspect-velocity"), inspectNext: $("digit-flow-inspect-next"), inspectTarget: $("digit-flow-inspect-target"),
    sourceCaption: $("digit-flow-source-caption"), currentCaption: $("digit-flow-current-caption"),
    nextCaption: $("digit-flow-next-caption"), targetCaption: $("digit-flow-target-caption"), film: $("digit-flow-film"),
    couplingCopy: $("digit-flow-coupling-copy")
  };

  const D = data.pixelDim;
  const IMAGE_SIDE = data.imageSide;
  const H = 96;
  const INPUT = D * 2 + 1;
  const BATCH = 24;
  const LANE_COUNT = 6;
  const DIGIT_PLURALS = ["zeros", "ones", "twos", "threes", "fours", "fives", "sixes", "sevens", "eights", "nines"];
  const PARAMETER_COUNT = H * INPUT + H + D * H + D + 3 * D;

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
    const seed = (0xA341316C ^ (sourceDigit * 0x9E3779B1) ^ (targetDigit * 0x85EBCA77)) >>> 0;
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

  function trainNetwork(learningRate) {
    const net = state.net;
    net.g.forEach((gradient) => gradient.fill(0));
    const source = new Float32Array(D), target = new Float32Array(D), input = new Float32Array(D);
    const hidden = new Float32Array(H), output = new Float32Array(D), hiddenGradient = new Float32Array(H);
    let loss = 0;
    for (let item = 0; item < BATCH; item += 1) {
      const pair = state.pairs[Math.floor(state.trainingRng() * state.pairs.length)];
      const time = 0.02 + 0.96 * state.trainingRng();
      imageAt(pair[0], source, 0); imageAt(pair[1], target, 0);
      for (let j = 0; j < D; j += 1) input[j] = (1 - time) * source[j] + time * target[j];
      predict(net, input, source, time, hidden, output);
      hiddenGradient.fill(0);
      for (let j = 0; j < D; j += 1) {
        const error = output[j] - (target[j] - source[j]);
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

  function computeJourneys() {
    state.journeys = [];
    const hidden = new Float32Array(H), velocity = new Float32Array(D);
    for (let lane = 0; lane < LANE_COUNT; lane += 1) {
      const point = new Float32Array(D), source = new Float32Array(D);
      imageAt(state.examples[lane][0], point, 0);
      source.set(point);
      const journey = [new Float32Array(point)];
      for (let step = 0; step < state.solverSteps; step += 1) {
        predict(state.net, point, source, (step + 0.5) / state.solverSteps, hidden, velocity);
        for (let j = 0; j < D; j += 1) point[j] += velocity[j] / state.solverSteps;
        journey.push(new Float32Array(point));
      }
      state.journeys.push(journey);
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
    const tile = compact ? 32 : 48;
    const top = compact ? 45 : 52;
    const bottom = 16;
    const rowHeight = (height - top - bottom) / LANE_COUNT;
    const sourceX = compact ? 27 : 42;
    const targetX = width - sourceX;
    const trackStart = sourceX + tile + (compact ? 14 : 28);
    const trackEnd = targetX - tile - (compact ? 14 : 28);
    return { tile, top, rowHeight, sourceX, targetX, trackStart, trackEnd };
  }

  function drawStage() {
    if (!state.journeys.length) return;
    const canvas = ui.stage, rect = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    const context = canvas.getContext("2d");
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    const width = rect.width, height = rect.height, layout = stageLayout(width, height);
    context.fillStyle = "#111814"; context.fillRect(0, 0, width, height);
    context.font = "700 10px system-ui, sans-serif"; context.textAlign = "center";
    context.fillStyle = "#aebbb4";
    context.fillText("SOURCE " + state.sourceDigit, layout.sourceX, 22);
    context.fillText("LEARNED STATE  xₜ", (layout.trackStart + layout.trackEnd) / 2, 22);
    context.fillText("PAIRED " + state.targetDigit, layout.targetX, 22);

    const current = new Float32Array(D), ghost = new Float32Array(D), source = new Float32Array(D), target = new Float32Array(D);
    for (let lane = 0; lane < LANE_COUNT; lane += 1) {
      const centerY = layout.top + (lane + 0.5) * layout.rowHeight;
      if (lane === state.selectedLane) {
        context.fillStyle = "rgba(72, 190, 154, 0.09)";
        context.fillRect(0, centerY - layout.rowHeight / 2, width, layout.rowHeight);
      }
      context.strokeStyle = lane === state.selectedLane ? "rgba(134,220,196,0.55)" : "rgba(174,187,180,0.22)";
      context.lineWidth = lane === state.selectedLane ? 2 : 1;
      context.setLineDash([4, 5]);
      context.beginPath(); context.moveTo(layout.trackStart, centerY); context.lineTo(layout.trackEnd, centerY); context.stroke();
      context.setLineDash([]);

      imageAt(state.examples[lane][0], source, 0); imageAt(state.examples[lane][1], target, 0);
      drawTile(context, source, layout.sourceX, centerY, layout.tile, [198, 210, 203], 1, lane === state.selectedLane ? "#86dcc4" : null);
      drawTile(context, target, layout.targetX, centerY, layout.tile, [241, 215, 142], 1, null);
      [0.25, 0.5, 0.75].forEach((checkpoint) => {
        stateAt(state.journeys[lane], checkpoint, ghost);
        const ghostX = layout.trackStart + checkpoint * (layout.trackEnd - layout.trackStart);
        drawTile(context, ghost, ghostX, centerY, layout.tile * 0.82, [134, 240, 210], 0.15, null);
      });
      stateAt(state.journeys[lane], state.motionProgress, current);
      const movingX = layout.trackStart + state.motionProgress * (layout.trackEnd - layout.trackStart);
      drawTile(context, current, movingX, centerY, layout.tile, [134, 240, 210], 1, lane === state.selectedLane ? "#eafbf5" : "#4e9d87");
      context.fillStyle = lane === state.selectedLane ? "#eafbf5" : "#75837b";
      context.font = "10px monospace"; context.textAlign = "left";
      context.fillText(String(lane + 1), 4, centerY + 3);
    }
  }

  function drawFilm() {
    const journey = state.journeys[state.selectedLane];
    const count = 7;
    ui.film.width = count * IMAGE_SIDE; ui.film.height = IMAGE_SIDE;
    const context = ui.film.getContext("2d"), image = context.createImageData(ui.film.width, IMAGE_SIDE);
    for (let checkpoint = 0; checkpoint < count; checkpoint += 1) {
      const index = Math.round((journey.length - 1) * checkpoint / (count - 1));
      paintPixelImage(image, ui.film.width, journey[index], checkpoint * IMAGE_SIDE, 0, [134, 240, 210]);
    }
    context.putImageData(image, 0, 0);
    const active = Math.round(state.motionProgress * (count - 1));
    context.strokeStyle = "#eafbf5"; context.lineWidth = 1;
    context.strokeRect(active * IMAGE_SIDE + 0.5, 0.5, IMAGE_SIDE - 1, IMAGE_SIDE - 1);
  }

  function renderInspection() {
    if (!state.journeys.length) return;
    const pair = state.examples[state.selectedLane], journey = state.journeys[state.selectedLane];
    const source = new Float32Array(D), target = new Float32Array(D), current = new Float32Array(D);
    const velocity = new Float32Array(D), next = new Float32Array(D), hidden = new Float32Array(H);
    imageAt(pair[0], source, 0); imageAt(pair[1], target, 0); stateAt(journey, state.motionProgress, current);
    const modelTime = Math.min(0.999, state.motionProgress);
    predict(state.net, current, source, modelTime, hidden, velocity);
    const delta = state.motionProgress >= 1 ? 0 : Math.min(1 / state.solverSteps, 1 - state.motionProgress);
    for (let j = 0; j < D; j += 1) next[j] = current[j] + delta * velocity[j];
    drawPixelCanvas(ui.inspectSource, source, [198, 210, 203]);
    drawPixelCanvas(ui.inspectCurrent, current, [134, 240, 210]);
    drawSignedCanvas(ui.inspectVelocity, velocity);
    drawPixelCanvas(ui.inspectNext, next, [134, 240, 210]);
    drawPixelCanvas(ui.inspectTarget, target, [241, 215, 142]);
    ui.sourceCaption.textContent = "source " + state.sourceDigit;
    ui.currentCaption.textContent = "x at t = " + state.motionProgress.toFixed(2);
    ui.nextCaption.textContent = state.motionProgress >= 1 ? "journey complete" : "Δt = " + delta.toFixed(3);
    ui.targetCaption.textContent = "paired " + state.targetDigit;
    drawFilm();
  }

  function renderMotion() {
    const percent = Math.round(state.motionProgress * 100);
    ui.time.value = Math.round(state.motionProgress * 1000);
    if (state.motionProgress <= 0) ui.timeLabel.textContent = "t = 0.00 · source distribution";
    else if (state.motionProgress >= 1) ui.timeLabel.textContent = "t = 1.00 · generated target distribution";
    else ui.timeLabel.textContent = "t = " + state.motionProgress.toFixed(2) + " · integrating velocity · " + percent + "%";
    ui.selection.textContent = "Inspecting lane " + (state.selectedLane + 1) + " of " + LANE_COUNT;
    drawStage(); renderInspection();
  }

  function updateLabels() {
    ui.title.textContent = "Watch a " + state.sourceDigit + " learn to become a " + state.targetDigit;
    ui.summary.textContent = "Here the easy source is not Gaussian noise—it is a distribution of handwritten " + DIGIT_PLURALS[state.sourceDigit] + ". Flow matching learns the changing pixel velocity that carries those " + DIGIT_PLURALS[state.sourceDigit] + " into the distribution of " + DIGIT_PLURALS[state.targetDigit] + ".";
    ui.motionTitle.textContent = "Six " + DIGIT_PLURALS[state.sourceDigit] + " moving through learned image space";
    ui.axisSource.textContent = "source " + state.sourceDigit;
    ui.axisTarget.textContent = "target " + state.targetDigit;
    ui.couplingCopy.textContent = "MNIST does not contain naturally paired " + DIGIT_PLURALS[state.sourceDigit] + " and " + DIGIT_PLURALS[state.targetDigit] + ", so training must temporarily couple examples. Closest-looking pairs encourage shorter, more coherent paths. Random pairs create more conflicting velocities; the model still aims for the same target distribution, but individual journeys can become less direct.";
  }

  function updateTrainingUi() {
    ui.progress.max = state.budget; ui.progress.value = state.update;
    ui.progressLabel.textContent = state.update.toLocaleString() + " / " + state.budget.toLocaleString() + " updates";
    ui.train.disabled = state.training || state.update >= state.budget;
    ui.trainPause.disabled = !state.training;
    ui.train.textContent = state.update >= state.budget ? "Training complete" : state.update > 0 ? "Continue training" : "Train the flow";
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
      const loss = trainNetwork(learningRate);
      state.lossEma = state.lossEma === null ? loss : 0.96 * state.lossEma + 0.04 * loss;
      state.update += 1;
    }
    updateTrainingUi();
    if (state.update % 80 < count || state.update >= state.budget) computeJourneys();
    ui.status.textContent = "Training · update " + state.update.toLocaleString() + " · velocity MSE " + state.lossEma.toFixed(4) + " · lr " + learningRate.toFixed(4);
    if (state.update >= state.budget) {
      stopTraining(); computeJourneys();
      ui.status.textContent = "Training complete · press Play once to watch " + state.sourceDigit + " → " + state.targetDigit;
      return;
    }
    state.trainingTimer = window.setTimeout(trainFrame, 0);
  }

  function stopMotion(paused) {
    state.playing = false;
    if (state.animationFrame) window.cancelAnimationFrame(state.animationFrame);
    state.animationFrame = 0;
    ui.playPause.disabled = true;
    ui.play.textContent = state.motionProgress >= 1 ? "Replay motion" : "Play once";
    if (paused) ui.status.textContent = "Motion paused at t = " + state.motionProgress.toFixed(2);
  }

  function motionFrame(now) {
    if (!state.playing) return;
    const elapsed = now - state.motionStartedAt;
    const progress = Math.min(1, state.motionStartProgress + elapsed / state.motionDuration);
    state.motionProgress = progress; renderMotion();
    if (progress >= 1) {
      stopMotion(false);
      ui.status.textContent = "Journey complete · the left-to-right position showed time; the changing pixels were the learned flow";
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

  function resetLearner() {
    stopTraining(); stopMotion(false);
    state.update = 0; state.lossEma = null; state.motionProgress = 0;
    const seed = (0x71C3A95D ^ (state.sourceDigit * 0x9E3779B1) ^ (state.targetDigit * 0x85EBCA77) ^ (state.pairing === "random" ? 0x27D4EB2F : 0)) >>> 0;
    state.net = makeNetwork(mulberry32(seed));
    state.trainingRng = mulberry32((seed ^ 0xB5297A4D) >>> 0);
    chooseExamples(); computeJourneys(); updateLabels(); updateTrainingUi();
    ui.status.textContent = "Ready · " + state.pairs.length.toLocaleString() + " " + (state.pairing === "matched" ? "closest-looking" : "random") + " pairs · " + PARAMETER_COUNT.toLocaleString() + " parameters · model untrained";
  }

  function configureExperiment(changedControl) {
    stopTraining(); stopMotion(false);
    let source = Number(ui.source.value), target = Number(ui.target.value);
    if (source === target) {
      if (changedControl === "source") { target = (source + 1) % 10; ui.target.value = String(target); }
      else { source = (target + 9) % 10; ui.source.value = String(source); }
    }
    state.sourceDigit = source; state.targetDigit = target; state.pairing = ui.pairing.value;
    ui.status.textContent = "Building " + (state.pairing === "matched" ? "closest-looking" : "random") + " " + source + " → " + target + " pairs…";
    state.pairs = makePairs(source, target, state.pairing);
    resetLearner();
  }

  const state = {
    sourceDigit: 0, targetDigit: 5, pairing: "matched", pairs: [], examples: [], journeys: [],
    net: null, trainingRng: null, update: 0, budget: Number(ui.budget.value), lossEma: null,
    solverSteps: Number(ui.steps.value), selectedLane: 0, exampleSeed: 0,
    training: false, trainingTimer: 0, playing: false, animationFrame: 0,
    motionProgress: 0, motionStartProgress: 0, motionStartedAt: 0, motionDuration: 5200
  };

  ui.source.addEventListener("change", () => configureExperiment("source"));
  ui.target.addEventListener("change", () => configureExperiment("target"));
  ui.pairing.addEventListener("change", () => configureExperiment("pairing"));
  ui.train.addEventListener("click", () => {
    if (state.update >= state.budget) return;
    state.training = true; updateTrainingUi(); state.trainingTimer = window.setTimeout(trainFrame, 0);
  });
  ui.trainPause.addEventListener("click", () => {
    stopTraining(); computeJourneys(); ui.status.textContent = "Training paused at update " + state.update.toLocaleString();
  });
  ui.reset.addEventListener("click", resetLearner);
  ui.budget.addEventListener("input", () => {
    state.budget = Number(ui.budget.value); ui.budgetOutput.value = state.budget.toLocaleString(); updateTrainingUi();
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
    chooseExamples(); computeJourneys(); ui.status.textContent = "Loaded six new source examples; the trained field is unchanged";
  });
  ui.stage.addEventListener("click", (event) => {
    const rect = ui.stage.getBoundingClientRect(), layout = stageLayout(rect.width, rect.height);
    const lane = Math.floor((event.clientY - rect.top - layout.top) / layout.rowHeight);
    if (lane >= 0 && lane < LANE_COUNT) { state.selectedLane = lane; renderMotion(); }
  });
  window.addEventListener("resize", drawStage);

  configureExperiment("pairing");
}());
