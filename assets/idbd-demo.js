(function () {
  "use strict";

  const FEATURE_COUNT = 64;
  const FEATURE_PROBABILITY = 0.01;
  const GAUSSIAN_VARIANCE = 5;
  const GAUSSIAN_STANDARD_DEVIATION = Math.sqrt(GAUSSIAN_VARIANCE);
  const SPARSE_NOISE_PROBABILITY = 0.01;
  const WEIGHT_LINEAR_THRESHOLD = 0.001;
  const PREDICTION_LOSS_BOUNDS = Object.freeze({ minimum: 0.1, maximum: 100 });
  const SIGNAL_LOSS_BOUNDS = Object.freeze({ minimum: 0.0001, maximum: 1 });
  const MODEL_WEIGHT_EXTENT = 2;
  const LOG_ONE_MINUS_FEATURE_PROBABILITY = Math.log(1 - FEATURE_PROBABILITY);
  const STEP_OPTIONS = [
    256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    65536, 100000, 250000, 500000, 1000000, 2000000, 5000000
  ];
  const COLORS = {
    ink: "#222222",
    muted: "#555555",
    grid: "#d6d6d6",
    sgd: "#494949",
    sgdFill: "rgba(73, 73, 73, 0.12)",
    idbd: "#1b5eaa",
    idbdFill: "rgba(27, 94, 170, 0.11)",
    signal: "#287a45",
    noise: "#999999",
    alert: "#b42318"
  };

  const controls = {
    sgdRate: document.getElementById("sgd-rate"),
    idbdRate: document.getElementById("idbd-rate"),
    theta: document.getElementById("theta"),
    batch: document.getElementById("batch-size"),
    steps: document.getElementById("training-steps"),
    lockRates: document.getElementById("lock-rates")
  };
  const outputs = {
    sgdRate: document.getElementById("sgd-rate-value"),
    idbdRate: document.getElementById("idbd-rate-value"),
    theta: document.getElementById("theta-value"),
    batch: document.getElementById("batch-size-value"),
    steps: document.getElementById("training-steps-value"),
    status: document.getElementById("run-status")
  };

  if (!controls.sgdRate || !controls.idbdRate || !controls.theta || !controls.batch || !controls.steps) return;

  let seed = 20260713;
  let scheduled = false;
  let runVersion = 0;
  let currentState = null;
  let resizeScheduled = false;

  function mulberry32(initialSeed) {
    let state = initialSeed >>> 0;
    return function () {
      state += 0x6D2B79F5;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function makeGaussian(random) {
    let spare = null;
    return function () {
      if (spare !== null) {
        const value = spare;
        spare = null;
        return value;
      }
      const radius = Math.sqrt(-2 * Math.log(Math.max(random(), 1e-12)));
      const angle = 2 * Math.PI * random();
      spare = radius * Math.sin(angle);
      return radius * Math.cos(angle);
    };
  }

  function fillActiveFeatures(random, active) {
    let count = 0;
    let feature = Math.floor(
      Math.log(1 - random()) / LOG_ONE_MINUS_FEATURE_PROBABILITY
    );
    while (feature < FEATURE_COUNT) {
      active[count] = feature;
      count += 1;
      feature += 1 + Math.floor(
        Math.log(1 - random()) / LOG_ONE_MINUS_FEATURE_PROBABILITY
      );
    }
    return count;
  }

  function prediction(weights, active, activeCount) {
    let total = 0;
    for (let index = 0; index < activeCount; index += 1) {
      total += weights[active[index]];
    }
    return total;
  }

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function symmetricLog(value) {
    return Math.sign(value) * Math.log10(1 + Math.abs(value) / WEIGHT_LINEAR_THRESHOLD);
  }

  function noiselessSignalMse(weights) {
    let coefficientSum = 0;
    let coefficientSquares = 0;
    for (let index = 0; index < FEATURE_COUNT; index += 1) {
      const coefficient = weights[index] - (index === 0 ? 1 : 0);
      coefficientSum += coefficient;
      coefficientSquares += coefficient * coefficient;
    }
    const probability = FEATURE_PROBABILITY;
    return probability * (1 - probability) * coefficientSquares
      + probability * probability * coefficientSum * coefficientSum;
  }

  function createTrainingState(streamSeed, sgdRate, idbdInitialRate, theta, batchSize, exampleCount) {
    const random = mulberry32(streamSeed);
    const beta = new Float64Array(FEATURE_COUNT);
    beta.fill(Math.log(idbdInitialRate));
    const totalBatches = Math.ceil(exampleCount / batchSize);
    return {
      streamSeed,
      sgdRate,
      theta,
      batchSize,
      exampleCount,
      totalBatches,
      maxBatchesPerFrame: Math.max(1, Math.ceil(totalBatches / 30)),
      recordEvery: Math.max(1, Math.floor(totalBatches / 150)),
      random,
      gaussian: makeGaussian(random),
      sgdWeights: new Float64Array(FEATURE_COUNT),
      idbdWeights: new Float64Array(FEATURE_COUNT),
      beta,
      trace: new Float64Array(FEATURE_COUNT),
      directionSgd: new Float64Array(FEATURE_COUNT),
      directionIdbd: new Float64Array(FEATURE_COUNT),
      activeCounts: new Float64Array(FEATURE_COUNT),
      active: new Uint8Array(FEATURE_COUNT),
      touched: new Uint8Array(FEATURE_COUNT),
      curves: {
        steps: [0],
        sgdLoss: [null],
        idbdLoss: [null],
        sgdSignalLoss: [FEATURE_PROBABILITY],
        idbdSignalLoss: [FEATURE_PROBABILITY]
      },
      sgdLossEma: null,
      idbdLossEma: null,
      examplesSeen: 0,
      batchIndex: 0,
      started: performance.now(),
      lastStatusUpdate: 0
    };
  }

  function processBatch(state) {
    const start = state.examplesSeen;
    const end = Math.min(state.exampleCount, start + state.batchSize);
    const actualBatchSize = end - start;
    let touchedCount = 0;
    let batchSgdSquaredError = 0;
    let batchIdbdSquaredError = 0;

    for (let row = start; row < end; row += 1) {
      const activeCount = fillActiveFeatures(state.random, state.active);
      const predictableTarget = activeCount > 0 && state.active[0] === 0 ? 1 : 0;
      const sparseNoise = state.random() < SPARSE_NOISE_PROBABILITY
        ? (state.random() < 0.5 ? -1 : 1)
        : 0;
      const target = predictableTarget + sparseNoise + GAUSSIAN_STANDARD_DEVIATION * state.gaussian();
      const sgdError = target - prediction(state.sgdWeights, state.active, activeCount);
      const idbdError = target - prediction(state.idbdWeights, state.active, activeCount);
      batchSgdSquaredError += sgdError * sgdError;
      batchIdbdSquaredError += idbdError * idbdError;

      for (let index = 0; index < activeCount; index += 1) {
        const feature = state.active[index];
        if (state.activeCounts[feature] === 0) {
          state.touched[touchedCount] = feature;
          touchedCount += 1;
        }
        state.directionSgd[feature] += sgdError;
        state.directionIdbd[feature] += idbdError;
        state.activeCounts[feature] += 1;
      }
    }

    for (let index = 0; index < touchedCount; index += 1) {
      const feature = state.touched[index];
      const sgdDirection = state.directionSgd[feature] / actualBatchSize;
      const idbdDirection = state.directionIdbd[feature] / actualBatchSize;
      const curvature = state.activeCounts[feature] / actualBatchSize;
      state.sgdWeights[feature] += state.sgdRate * sgdDirection;

      const betaChange = clamp(state.theta * idbdDirection * state.trace[feature], -2, 2);
      state.beta[feature] = clamp(state.beta[feature] + betaChange, -10, Math.log(0.5));
      const featureRate = Math.exp(state.beta[feature]);
      const weightChange = featureRate * idbdDirection;
      state.idbdWeights[feature] += weightChange;
      state.trace[feature] = state.trace[feature] * Math.max(0, 1 - featureRate * curvature) + weightChange;
      state.directionSgd[feature] = 0;
      state.directionIdbd[feature] = 0;
      state.activeCounts[feature] = 0;
    }

    const sgdBatchLoss = batchSgdSquaredError / actualBatchSize;
    const idbdBatchLoss = batchIdbdSquaredError / actualBatchSize;
    state.sgdLossEma = state.sgdLossEma === null ? sgdBatchLoss : 0.92 * state.sgdLossEma + 0.08 * sgdBatchLoss;
    state.idbdLossEma = state.idbdLossEma === null ? idbdBatchLoss : 0.92 * state.idbdLossEma + 0.08 * idbdBatchLoss;
    state.examplesSeen = end;

    if (state.batchIndex % state.recordEvery === 0 || end === state.exampleCount) {
      state.curves.steps.push(end);
      state.curves.sgdLoss.push(state.sgdLossEma);
      state.curves.idbdLoss.push(state.idbdLossEma);
      state.curves.sgdSignalLoss.push(noiselessSignalMse(state.sgdWeights));
      state.curves.idbdSignalLoss.push(noiselessSignalMse(state.idbdWeights));
    }
    state.batchIndex += 1;
  }

  function processTrainingChunk(state) {
    const chunkStarted = performance.now();
    let batchesProcessed = 0;
    while (
      state.examplesSeen < state.exampleCount &&
      batchesProcessed < state.maxBatchesPerFrame
    ) {
      processBatch(state);
      batchesProcessed += 1;
      if (
        (batchesProcessed & 63) === 0 &&
        performance.now() - chunkStarted >= 8
      ) break;
    }
  }

  function formatRate(value) {
    if (value < 0.001) return value.toExponential(1);
    if (value < 0.1) return value.toFixed(4);
    return value.toFixed(3);
  }

  function formatScore(value) {
    if (!Number.isFinite(value)) return "unstable";
    if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(1);
    if (value >= 100) return value.toExponential(1);
    return value.toFixed(3);
  }

  function formatInteger(value) {
    return Math.round(value).toLocaleString("en-GB");
  }

  function setupCanvas(canvas) {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(260, canvas.clientWidth);
    const height = Math.max(140, canvas.clientHeight);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    return { context, width, height };
  }

  function drawEdgeWarning(context, right, y, text) {
    context.save();
    context.font = "bold 11px sans-serif";
    context.textAlign = "right";
    context.textBaseline = "middle";
    const textWidth = context.measureText(text).width;
    context.fillStyle = "rgba(252, 252, 252, 0.92)";
    context.fillRect(right - textWidth - 5, y - 8, textWidth + 7, 16);
    context.fillStyle = COLORS.alert;
    context.fillText(text, right - 2, y);
    context.restore();
  }

  function drawLineChart(canvasId, steps, values, color, fill, bounds, exampleCount) {
    const canvas = document.getElementById(canvasId);
    const surface = setupCanvas(canvas);
    const context = surface.context;
    const width = surface.width;
    const height = surface.height;
    const margins = { top: 10, right: 10, bottom: 27, left: 42 };
    const plotWidth = width - margins.left - margins.right;
    const plotHeight = height - margins.top - margins.bottom;
    const minimum = bounds.minimum;
    const maximum = bounds.maximum;
    const logMinimum = Math.log10(minimum);
    const logRange = Math.log10(maximum) - logMinimum;
    let aboveScale = false;
    let belowScale = false;

    context.fillStyle = "#fcfcfc";
    context.fillRect(margins.left, margins.top, plotWidth, plotHeight);
    context.font = "11px serif";
    context.lineWidth = 1;
    context.textBaseline = "middle";
    for (let tick = 0; tick <= 2; tick += 1) {
      const fraction = tick / 2;
      const y = margins.top + plotHeight * (1 - fraction);
      const tickValue = Math.pow(10, logMinimum + logRange * fraction);
      context.strokeStyle = COLORS.grid;
      context.beginPath();
      context.moveTo(margins.left, y);
      context.lineTo(width - margins.right, y);
      context.stroke();
      context.fillStyle = COLORS.muted;
      context.textAlign = "right";
      context.fillText(formatScore(tickValue), margins.left - 7, y);
    }

    context.strokeStyle = "#aaa";
    context.beginPath();
    context.moveTo(margins.left, margins.top);
    context.lineTo(margins.left, margins.top + plotHeight);
    context.lineTo(width - margins.right, margins.top + plotHeight);
    context.stroke();

    context.textBaseline = "alphabetic";
    context.fillStyle = COLORS.muted;
    context.textAlign = "left";
    context.fillText("0", margins.left, height - 5);
    context.textAlign = "right";
    context.fillText(formatInteger(exampleCount) + " examples", width - margins.right, height - 5);

    const coordinates = [];
    for (let index = 0; index < values.length; index += 1) {
      if (!Number.isFinite(values[index])) {
        if (values[index] !== null) aboveScale = true;
        continue;
      }
      if (values[index] > maximum) aboveScale = true;
      if (values[index] < minimum) belowScale = true;
      if (values[index] <= 0) continue;
      const logValue = Math.log10(Math.max(values[index], minimum));
      coordinates.push({
        x: margins.left + (steps[index] / exampleCount) * plotWidth,
        y: margins.top + (1 - clamp((logValue - logMinimum) / logRange, 0, 1)) * plotHeight
      });
    }
    if (coordinates.length) {
      context.beginPath();
      context.moveTo(coordinates[0].x, margins.top + plotHeight);
      for (let index = 0; index < coordinates.length; index += 1) {
        context.lineTo(coordinates[index].x, coordinates[index].y);
      }
      context.lineTo(coordinates[coordinates.length - 1].x, margins.top + plotHeight);
      context.closePath();
      context.fillStyle = fill;
      context.fill();

      context.beginPath();
      context.moveTo(coordinates[0].x, coordinates[0].y);
      for (let index = 1; index < coordinates.length; index += 1) {
        context.lineTo(coordinates[index].x, coordinates[index].y);
      }
      context.strokeStyle = color;
      context.lineWidth = 2.5;
      context.lineJoin = "round";
      context.stroke();

      const last = coordinates[coordinates.length - 1];
      context.beginPath();
      context.arc(last.x, last.y, 3.5, 0, Math.PI * 2);
      context.fillStyle = color;
      context.fill();
    }

    if (aboveScale) {
      drawEdgeWarning(context, width - margins.right, margins.top + 10, "↑ off scale (above " + String(maximum) + ")");
    }
    if (belowScale) {
      drawEdgeWarning(context, width - margins.right, margins.top + plotHeight - 10, "↓ off scale (below " + String(minimum) + ")");
    }
  }

  function drawWeights(canvasId, weights, color) {
    const canvas = document.getElementById(canvasId);
    const surface = setupCanvas(canvas);
    const context = surface.context;
    const width = surface.width;
    const height = surface.height;
    const margins = { top: 11, right: 9, bottom: 34, left: 42 };
    const plotWidth = width - margins.left - margins.right;
    const plotHeight = height - margins.top - margins.bottom;
    const extent = MODEL_WEIGHT_EXTENT;
    const transformedExtent = symmetricLog(extent);
    const zeroY = margins.top + plotHeight / 2;
    const barStep = plotWidth / FEATURE_COUNT;
    let aboveScale = false;
    let belowScale = false;

    function weightY(value) {
      return zeroY - symmetricLog(clamp(value, -extent, extent)) /
        transformedExtent * (plotHeight / 2);
    }

    const ticks = [0];
    const largestExponent = Math.ceil(Math.log10(extent));
    for (let exponent = -2; exponent <= largestExponent; exponent += 1) {
      const magnitude = Math.pow(10, exponent);
      if (magnitude <= extent * 1.000001) {
        ticks.push(-magnitude, magnitude);
      }
    }
    ticks.sort(function (first, second) { return first - second; });

    context.font = "11px serif";
    ticks.forEach(function (tick) {
      const y = weightY(tick);
      context.strokeStyle = tick === 0 ? COLORS.noise : COLORS.grid;
      context.beginPath();
      context.moveTo(margins.left, y);
      context.lineTo(width - margins.right, y);
      context.stroke();
      context.fillStyle = COLORS.muted;
      context.textAlign = "right";
      context.textBaseline = "middle";
      const label = tick < 0 ? "−" + String(Math.abs(tick)) : String(tick);
      context.fillText(label, margins.left - 7, y);
    });

    if (extent >= 1) {
      const referenceY = weightY(1);
      context.save();
      context.setLineDash([4, 3]);
      context.strokeStyle = COLORS.signal;
      context.globalAlpha = 0.75;
      context.beginPath();
      context.moveTo(margins.left, referenceY);
      context.lineTo(width - margins.right, referenceY);
      context.stroke();
      context.restore();
      context.fillStyle = COLORS.signal;
      context.textAlign = "right";
      context.fillText("target = 1", width - margins.right, referenceY - 8);
    }

    for (let index = 0; index < weights.length; index += 1) {
      if (!Number.isFinite(weights[index]) || weights[index] > extent) aboveScale = true;
      if (Number.isFinite(weights[index]) && weights[index] < -extent) belowScale = true;
      const value = clamp(weights[index], -extent, extent);
      const valueY = weightY(value);
      const valueHeight = Math.abs(valueY - zeroY);
      const x = margins.left + index * barStep + Math.max(0.5, barStep * 0.12);
      const y = Math.min(valueY, zeroY);
      context.fillStyle = index === 0 ? COLORS.signal : color;
      context.globalAlpha = index === 0 ? 1 : 0.62;
      context.fillRect(x, y, Math.max(1, barStep * 0.72), valueHeight);
    }
    context.globalAlpha = 1;
    context.fillStyle = COLORS.signal;
    context.textAlign = "left";
    context.textBaseline = "alphabetic";
    context.fillText("signal", margins.left, height - 7);
    context.fillStyle = COLORS.muted;
    context.textAlign = "right";
    context.fillText("63 irrelevant features", width - margins.right, height - 7);
    if (aboveScale) {
      drawEdgeWarning(context, width - margins.right, margins.top + 10, "↑ off scale (above +" + String(extent) + ")");
    }
    if (belowScale) {
      drawEdgeWarning(context, width - margins.right, margins.top + plotHeight - 10, "↓ off scale (below −" + String(extent) + ")");
    }
  }

  function updateBatchNote(batchSize) {
    const note = document.getElementById("batch-note");
    if (batchSize === 1) {
      note.innerHTML = "<strong>At batch size 1</strong>, this is classic online IDBD. Move the batch slider to explore a mean-gradient diagonal approximation.";
    } else {
      note.innerHTML = "<strong>At batch size " + batchSize + "</strong>, IDBD uses a mean-gradient diagonal approximation. Classic 1992 IDBD is the batch-size-one setting.";
    }
  }

  function selectedSettings() {
    const sgdRate = Math.pow(10, Number(controls.sgdRate.value));
    const idbdInitialRate = Math.pow(10, Number(controls.idbdRate.value));
    const theta = Math.pow(10, Number(controls.theta.value));
    const batchSize = Math.pow(2, Number(controls.batch.value));
    const exampleCount = STEP_OPTIONS[Number(controls.steps.value)];
    return { sgdRate, idbdInitialRate, theta, batchSize, exampleCount };
  }

  function updateControlOutputs(settings) {
    const values = settings || selectedSettings();
    outputs.sgdRate.value = formatRate(values.sgdRate);
    outputs.idbdRate.value = formatRate(values.idbdInitialRate);
    outputs.theta.value = formatRate(values.theta);
    outputs.batch.value = String(values.batchSize);
    outputs.steps.value = formatInteger(values.exampleCount);
    document.getElementById("example-count-label").textContent = formatInteger(values.exampleCount) + " examples";
  }

  function idbdRateRatio(beta) {
    const rates = Array.from(beta, Math.exp);
    const irrelevant = rates.slice(1).sort(function (first, second) {
      return first - second;
    });
    return rates[0] / Math.max(irrelevant[Math.floor(irrelevant.length / 2)], 1e-12);
  }

  function drawTrainingState(state) {
    document.getElementById("sgd-loss-score").textContent = state.sgdLossEma === null ? "—" : formatScore(state.sgdLossEma);
    document.getElementById("idbd-loss-score").textContent = state.idbdLossEma === null ? "—" : formatScore(state.idbdLossEma);
    document.getElementById("sgd-signal-weight").textContent = formatScore(state.sgdWeights[0]);
    document.getElementById("idbd-rate-ratio").textContent = formatScore(idbdRateRatio(state.beta)) + "×";
    document.getElementById("sgd-signal-loss-score").textContent = formatScore(noiselessSignalMse(state.sgdWeights));
    document.getElementById("idbd-signal-loss-score").textContent = formatScore(noiselessSignalMse(state.idbdWeights));

    drawLineChart("sgd-loss-chart", state.curves.steps, state.curves.sgdLoss, COLORS.sgd, COLORS.sgdFill, PREDICTION_LOSS_BOUNDS, state.exampleCount);
    drawLineChart("idbd-loss-chart", state.curves.steps, state.curves.idbdLoss, COLORS.idbd, COLORS.idbdFill, PREDICTION_LOSS_BOUNDS, state.exampleCount);
    drawLineChart("sgd-signal-loss-chart", state.curves.steps, state.curves.sgdSignalLoss, COLORS.sgd, COLORS.sgdFill, SIGNAL_LOSS_BOUNDS, state.exampleCount);
    drawLineChart("idbd-signal-loss-chart", state.curves.steps, state.curves.idbdSignalLoss, COLORS.idbd, COLORS.idbdFill, SIGNAL_LOSS_BOUNDS, state.exampleCount);
    drawWeights("sgd-weights-chart", state.sgdWeights, COLORS.sgd);
    drawWeights("idbd-weights-chart", state.idbdWeights, COLORS.idbd);
  }

  function elapsedLabel(milliseconds) {
    return milliseconds < 1000
      ? Math.round(milliseconds) + " ms"
      : (milliseconds / 1000).toFixed(1) + " s";
  }

  function updateProgressStatus(state, complete) {
    const now = performance.now();
    if (!complete && now - state.lastStatusUpdate < 80) return;
    state.lastStatusUpdate = now;
    if (complete) {
      outputs.status.textContent = "Complete · " + formatInteger(state.totalBatches) + " updates · " + elapsedLabel(now - state.started) + " · stream " + String(state.streamSeed).slice(-4);
      return;
    }
    const percent = 100 * state.examplesSeen / state.exampleCount;
    outputs.status.textContent = "Training · " + formatInteger(state.examplesSeen) + " / " + formatInteger(state.exampleCount) + " · " + percent.toFixed(0) + "%";
  }

  function continueTraining(version, state) {
    if (version !== runVersion) return;
    processTrainingChunk(state);
    if (version !== runVersion) return;
    drawTrainingState(state);
    const complete = state.examplesSeen >= state.exampleCount;
    updateProgressStatus(state, complete);
    if (!complete) {
      window.requestAnimationFrame(function () {
        continueTraining(version, state);
      });
    }
  }

  function startTraining(version) {
    if (version !== runVersion) return;
    scheduled = false;
    const settings = selectedSettings();
    updateControlOutputs(settings);
    const state = createTrainingState(
      seed,
      settings.sgdRate,
      settings.idbdInitialRate,
      settings.theta,
      settings.batchSize,
      settings.exampleCount
    );
    currentState = state;
    updateBatchNote(settings.batchSize);
    drawTrainingState(state);
    outputs.status.textContent = "Training · 0 / " + formatInteger(settings.exampleCount) + " · 0%";
    window.requestAnimationFrame(function () {
      continueTraining(version, state);
    });
  }

  function scheduleTraining() {
    runVersion += 1;
    updateControlOutputs();
    outputs.status.textContent = currentState ? "Restarting…" : "Preparing…";
    if (scheduled) return;
    scheduled = true;
    window.requestAnimationFrame(function () {
      startTraining(runVersion);
    });
  }

  function handleRateInput(source, target) {
    if (controls.lockRates.checked) target.value = source.value;
    scheduleTraining();
  }

  controls.sgdRate.addEventListener("input", function () {
    handleRateInput(controls.sgdRate, controls.idbdRate);
  });
  controls.idbdRate.addEventListener("input", function () {
    handleRateInput(controls.idbdRate, controls.sgdRate);
  });
  controls.theta.addEventListener("input", scheduleTraining);
  controls.batch.addEventListener("input", scheduleTraining);
  controls.steps.addEventListener("input", scheduleTraining);
  controls.lockRates.addEventListener("change", function () {
    if (controls.lockRates.checked) controls.idbdRate.value = controls.sgdRate.value;
    scheduleTraining();
  });
  document.getElementById("new-stream").addEventListener("click", function () {
    seed += 1;
    scheduleTraining();
  });
  window.addEventListener("resize", function () {
    if (resizeScheduled) return;
    resizeScheduled = true;
    window.requestAnimationFrame(function () {
      resizeScheduled = false;
      if (currentState) drawTrainingState(currentState);
    });
  });
  scheduleTraining();
}());
