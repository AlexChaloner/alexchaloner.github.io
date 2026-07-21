(function () {
  "use strict";

  const FEATURE_COUNT = 64;
  const FEATURE_PROBABILITY = 0.01;
  const GAUSSIAN_VARIANCE = 5;
  const GAUSSIAN_STANDARD_DEVIATION = Math.sqrt(GAUSSIAN_VARIANCE);
  const SPARSE_NOISE_PROBABILITY = 0.01;
  const WEIGHT_LINEAR_THRESHOLD = 0.001;
  const PREDICTION_LOSS_BOUNDS = Object.freeze({ minimum: 0.1, maximum: 100 });
  const SIGNAL_LOSS_BOUNDS = Object.freeze({ minimum: 0.00001, maximum: 100 });
  const LEARNING_RATE_BOUNDS = Object.freeze({ minimum: 0.00001, maximum: 10 });
  const MODEL_WEIGHT_EXTENT = 2;
  const TRACE_EXTENT = 2;
  const PREVIEW_LENGTH = 500;
  const LOG_ONE_MINUS_FEATURE_PROBABILITY = Math.log(1 - FEATURE_PROBABILITY);
  const STEP_OPTIONS = [
    256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    65536, 100000, 250000, 500000, 1000000, 2000000, 5000000,
    10000000, 25000000, 50000000, 100000000
  ];
  const WEIGHT_DECAY_OPTIONS = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1];
  const COLORS = {
    ink: "#222222",
    muted: "#555555",
    grid: "#d6d6d6",
    sgd: "#494949",
    sgdFill: "rgba(73, 73, 73, 0.12)",
    idbd: "#1b5eaa",
    idbdFill: "rgba(27, 94, 170, 0.11)",
    trace: "#71409a",
    signal: "#287a45",
    noise: "#999999",
    alert: "#b42318"
  };

  function createExperiment(prefix, extensionsEnabled) {
    function byId(name) {
      return document.getElementById(prefix ? prefix + "-" + name : name);
    }

  const controls = {
    sgdRate: byId("sgd-rate"),
    idbdRate: byId("idbd-rate"),
    theta: byId("theta"),
    batch: byId("batch-size"),
    steps: byId("training-steps"),
    lockRates: byId("lock-rates"),
    seed: byId("stream-seed"),
    reseed: byId("reseed-stream"),
    newStream: byId("new-stream"),
    pause: byId("pause-training"),
    play: byId("play-training"),
    stop: byId("stop-training"),
    momentum: byId("momentum"),
    momentumMode: byId("momentum-mode"),
    weightDecay: byId("weight-decay"),
    weightDecayMode: byId("weight-decay-mode")
  };
  const outputs = {
    sgdRate: byId("sgd-rate-value"),
    idbdRate: byId("idbd-rate-value"),
    theta: byId("theta-value"),
    batch: byId("batch-size-value"),
    steps: byId("training-steps-value"),
    status: byId("run-status"),
    momentum: byId("momentum-value"),
    weightDecay: byId("weight-decay-value")
  };

  if (!controls.sgdRate || !controls.idbdRate || !controls.theta || !controls.batch || !controls.steps) return;

  let seed = 20260713;
  let scheduled = false;
  let scheduledFrame = null;
  let runVersion = 0;
  let currentState = null;
  let resizeScheduled = false;
  let paused = false;
  let stopped = false;
  let pausedAt = 0;

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

  function symmetricLog(value, linearThreshold) {
    const threshold = linearThreshold || WEIGHT_LINEAR_THRESHOLD;
    return Math.sign(value) * Math.log10(1 + Math.abs(value) / threshold);
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

  function createStreamPreview(streamSeed) {
    const random = mulberry32((streamSeed ^ 0x9E3779B9) >>> 0);
    const gaussian = makeGaussian(random);
    const active = new Uint8Array(FEATURE_COUNT);
    const features = new Uint8Array(PREVIEW_LENGTH * FEATURE_COUNT);
    const cleanTarget = new Float64Array(PREVIEW_LENGTH);
    const noisyTarget = new Float64Array(PREVIEW_LENGTH);
    for (let step = 0; step < PREVIEW_LENGTH; step += 1) {
      const activeCount = fillActiveFeatures(random, active);
      for (let index = 0; index < activeCount; index += 1) {
        features[step * FEATURE_COUNT + active[index]] = 1;
      }
      const predictableTarget = features[step * FEATURE_COUNT] ? 1 : 0;
      const sparseNoise = random() < SPARSE_NOISE_PROBABILITY
        ? (random() < 0.5 ? -1 : 1)
        : 0;
      cleanTarget[step] = predictableTarget;
      noisyTarget[step] = predictableTarget + sparseNoise + GAUSSIAN_STANDARD_DEVIATION * gaussian();
    }
    return {
      features,
      cleanTarget,
      noisyTarget,
      sgdPrediction: new Float64Array(PREVIEW_LENGTH),
      idbdPrediction: new Float64Array(PREVIEW_LENGTH)
    };
  }

  function createTrainingState(streamSeed, sgdRate, idbdInitialRate, theta, batchSize, exampleCount, optimizerOptions) {
    const random = mulberry32(streamSeed);
    const beta = new Float64Array(FEATURE_COUNT);
    beta.fill(Math.log(idbdInitialRate));
    const totalBatches = Math.ceil(exampleCount / batchSize);
    return {
      streamSeed,
      sgdRate,
      idbdInitialRate,
      theta,
      momentum: optimizerOptions.momentum,
      momentumMode: optimizerOptions.momentumMode,
      weightDecay: optimizerOptions.weightDecay,
      weightDecayMode: optimizerOptions.weightDecayMode,
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
      sgdMomentum: new Float64Array(FEATURE_COUNT),
      idbdMomentum: new Float64Array(FEATURE_COUNT),
      momentumTrace: new Float64Array(FEATURE_COUNT),
      lastUpdatedBatch: new Uint32Array(FEATURE_COUNT),
      inactiveSeriesScratch: new Float64Array(4),
      directionSgd: new Float64Array(FEATURE_COUNT),
      directionIdbd: new Float64Array(FEATURE_COUNT),
      activeCounts: new Float64Array(FEATURE_COUNT),
      active: new Uint8Array(FEATURE_COUNT),
      touched: new Uint8Array(FEATURE_COUNT),
      preview: createStreamPreview(streamSeed),
      curves: {
        steps: [0],
        sgdLoss: [null],
        idbdLoss: [null],
        sgdSignalLoss: [FEATURE_PROBABILITY],
        idbdSignalLoss: [FEATURE_PROBABILITY],
        idbdSignalRate: [idbdInitialRate],
        idbdMedianNoiseRate: [idbdInitialRate]
      },
      sgdLossEma: null,
      idbdLossEma: null,
      examplesSeen: 0,
      batchIndex: 0,
      started: performance.now(),
      lastStatusUpdate: 0
    };
  }

  // Sum n zero-gradient updates in closed form. Writing into a shared scratch
  // array avoids allocating short-lived objects in the per-feature hot path.
  function fillInactiveSeries(decay, momentum, count, result) {
    const decayPower = Math.pow(decay, count);
    const momentumPower = Math.pow(momentum, count);
    result[0] = decayPower;
    result[1] = momentumPower;
    if (momentum === 0) {
      result[2] = 0;
      result[3] = 0;
      return;
    }

    const difference = decay - momentum;
    if (Math.abs(difference) <= 1e-8 * Math.max(1, Math.abs(decay), Math.abs(momentum))) {
      result[2] = count * momentumPower;
      result[3] = count > 1
        ? count * (count - 1) * 0.5 * Math.pow(momentum, count - 1)
        : 0;
      return;
    }

    const decayDerivative = count * Math.pow(decay, count - 1);
    result[2] = momentum * (decayPower - momentumPower) / difference;
    result[3] = momentum * (
      decayDerivative * difference - (decayPower - momentumPower)
    ) / (difference * difference);
  }

  // Sparse features can sit idle for hundreds of batches. Their momentum,
  // decay, and IDBD trace recurrences are linear while the gradient is zero,
  // so fast-forward them only when a feature is next read or plotted.
  function materializeFeature(state, feature, completedBatchCount) {
    const inactiveCount = completedBatchCount - state.lastUpdatedBatch[feature];
    if (inactiveCount <= 0) return;

    const momentum = state.momentum;
    const sgdDecay = state.weightDecay > 0
      ? 1 - state.sgdRate * state.weightDecay
      : 1;
    const series = state.inactiveSeriesScratch;
    fillInactiveSeries(sgdDecay, momentum, inactiveCount, series);
    const oldSgdMomentum = state.sgdMomentum[feature];
    state.sgdWeights[feature] = series[0] * state.sgdWeights[feature]
      + state.sgdRate * oldSgdMomentum * series[2];
    state.sgdMomentum[feature] = series[1] * oldSgdMomentum;

    const featureRate = Math.exp(state.beta[feature]);
    let idbdDecay = 1;
    if (state.weightDecay > 0) {
      idbdDecay = state.weightDecayMode === "fixed"
        ? 1 - state.idbdInitialRate * state.weightDecay
        : 1 - featureRate * state.weightDecay;
    }
    fillInactiveSeries(idbdDecay, momentum, inactiveCount, series);
    const oldWeight = state.idbdWeights[feature];
    const oldMomentum = state.idbdMomentum[feature];
    const oldMomentumTrace = state.momentumTrace[feature];
    state.idbdWeights[feature] = series[0] * oldWeight
      + featureRate * oldMomentum * series[2];

    const derivedMomentum = momentum > 0 && state.momentumMode === "derived";
    if (state.weightDecay > 0 && state.weightDecayMode === "traced") {
      const decayDerivative = idbdDecay - 1;
      state.trace[feature] = series[0] * state.trace[feature]
        + inactiveCount * Math.pow(idbdDecay, inactiveCount - 1) * decayDerivative * oldWeight
        + featureRate * (oldMomentum + (derivedMomentum ? oldMomentumTrace : 0)) * series[2]
        + featureRate * oldMomentum * series[3] * decayDerivative;
    } else if (momentum > 0) {
      const momentumSum = momentum === 1
        ? inactiveCount
        : momentum * (1 - series[1]) / (1 - momentum);
      state.trace[feature] += featureRate
        * (oldMomentum + (derivedMomentum ? oldMomentumTrace : 0))
        * momentumSum;
    }

    state.idbdMomentum[feature] = series[1] * oldMomentum;
    if (derivedMomentum) {
      state.momentumTrace[feature] = series[1] * oldMomentumTrace;
    }
    state.lastUpdatedBatch[feature] = completedBatchCount;
  }

  function materializeAllFeatures(state, completedBatchCount) {
    if (state.momentum === 0 && state.weightDecay === 0) return;
    for (let feature = 0; feature < FEATURE_COUNT; feature += 1) {
      materializeFeature(state, feature, completedBatchCount);
    }
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
      if (state.momentum > 0 || state.weightDecay > 0) {
        for (let index = 0; index < activeCount; index += 1) {
          materializeFeature(state, state.active[index], state.batchIndex);
        }
      }
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
      let sgdUpdateDirection = sgdDirection;
      if (state.momentum > 0) {
        state.sgdMomentum[feature] = state.momentum * state.sgdMomentum[feature]
          + (1 - state.momentum) * sgdDirection;
        sgdUpdateDirection = state.sgdMomentum[feature];
      }
      if (state.weightDecay > 0) {
        state.sgdWeights[feature] *= 1 - state.sgdRate * state.weightDecay;
      }
      state.sgdWeights[feature] += state.sgdRate * sgdUpdateDirection;

      const betaChange = clamp(state.theta * idbdDirection * state.trace[feature], -2, 2);
      state.beta[feature] = clamp(state.beta[feature] + betaChange, -10, Math.log(10));
      const featureRate = Math.exp(state.beta[feature]);
      const oldTrace = state.trace[feature];
      let updateDirection = idbdDirection;
      let derivedMomentumTrace = null;
      if (state.momentum > 0) {
        state.idbdMomentum[feature] = state.momentum * state.idbdMomentum[feature]
          + (1 - state.momentum) * idbdDirection;
        updateDirection = state.idbdMomentum[feature];
        if (state.momentumMode === "derived") {
          state.momentumTrace[feature] = state.momentum * state.momentumTrace[feature]
            - (1 - state.momentum) * curvature * oldTrace;
          derivedMomentumTrace = state.momentumTrace[feature];
        }
      }

      const oldWeight = state.idbdWeights[feature];
      let baseTrace = oldTrace;
      if (state.weightDecay > 0) {
        if (state.weightDecayMode === "fixed") {
          state.idbdWeights[feature] *= 1 - state.idbdInitialRate * state.weightDecay;
        } else {
          state.idbdWeights[feature] *= 1 - featureRate * state.weightDecay;
          if (state.weightDecayMode === "traced") {
            baseTrace = (1 - featureRate * state.weightDecay) * oldTrace
              - featureRate * state.weightDecay * oldWeight;
          }
        }
      }

      const weightChange = featureRate * updateDirection;
      state.idbdWeights[feature] += weightChange;
      if (derivedMomentumTrace !== null) {
        state.trace[feature] = baseTrace + weightChange + featureRate * derivedMomentumTrace;
      } else {
        state.trace[feature] = baseTrace * Math.max(0, 1 - featureRate * curvature) + weightChange;
      }
      state.directionSgd[feature] = 0;
      state.directionIdbd[feature] = 0;
      state.activeCounts[feature] = 0;
      state.lastUpdatedBatch[feature] = state.batchIndex + 1;
    }

    const sgdBatchLoss = batchSgdSquaredError / actualBatchSize;
    const idbdBatchLoss = batchIdbdSquaredError / actualBatchSize;
    state.sgdLossEma = state.sgdLossEma === null ? sgdBatchLoss : 0.92 * state.sgdLossEma + 0.08 * sgdBatchLoss;
    state.idbdLossEma = state.idbdLossEma === null ? idbdBatchLoss : 0.92 * state.idbdLossEma + 0.08 * idbdBatchLoss;
    state.examplesSeen = end;

    if (state.batchIndex % state.recordEvery === 0 || end === state.exampleCount) {
      materializeAllFeatures(state, state.batchIndex + 1);
      state.curves.steps.push(end);
      state.curves.sgdLoss.push(state.sgdLossEma);
      state.curves.idbdLoss.push(state.idbdLossEma);
      state.curves.sgdSignalLoss.push(noiselessSignalMse(state.sgdWeights));
      state.curves.idbdSignalLoss.push(noiselessSignalMse(state.idbdWeights));
      const rateSummary = idbdRateSummary(state.beta);
      state.curves.idbdSignalRate.push(rateSummary.signal);
      state.curves.idbdMedianNoiseRate.push(rateSummary.medianNoise);
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
    if (value >= 10) return value.toFixed(value >= 100 ? 0 : 1);
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

  function drawTraceChart(canvasId, series, bounds, ticks) {
    const canvas = byId(canvasId);
    const surface = setupCanvas(canvas);
    const context = surface.context;
    const width = surface.width;
    const height = surface.height;
    const margins = { top: 9, right: 9, bottom: 25, left: 42 };
    const plotWidth = width - margins.left - margins.right;
    const plotHeight = height - margins.top - margins.bottom;
    const length = series[0].values.length;
    let observedMaximum = -Infinity;
    let observedMinimum = Infinity;
    let unstable = false;

    function valueY(value) {
      return margins.top + (1 - (clamp(value, bounds.minimum, bounds.maximum) - bounds.minimum) /
        (bounds.maximum - bounds.minimum)) * plotHeight;
    }

    context.fillStyle = "#fcfcfc";
    context.fillRect(margins.left, margins.top, plotWidth, plotHeight);
    context.font = "11px serif";
    context.lineWidth = 1;
    context.textBaseline = "middle";
    ticks.forEach(function (tick) {
      const y = valueY(tick);
      context.strokeStyle = tick === 0 ? COLORS.noise : COLORS.grid;
      context.beginPath();
      context.moveTo(margins.left, y);
      context.lineTo(width - margins.right, y);
      context.stroke();
      context.fillStyle = COLORS.muted;
      context.textAlign = "right";
      context.fillText(String(tick).replace("-", "−"), margins.left - 7, y);
    });

    context.strokeStyle = "#aaa";
    context.beginPath();
    context.moveTo(margins.left, margins.top);
    context.lineTo(margins.left, margins.top + plotHeight);
    context.lineTo(width - margins.right, margins.top + plotHeight);
    context.stroke();

    context.fillStyle = COLORS.muted;
    context.textBaseline = "alphabetic";
    context.textAlign = "left";
    context.fillText("0", margins.left, height - 5);
    context.textAlign = "right";
    context.fillText(String(length) + " steps", width - margins.right, height - 5);

    context.save();
    context.beginPath();
    context.rect(margins.left, margins.top, plotWidth, plotHeight);
    context.clip();
    series.forEach(function (item) {
      context.beginPath();
      let drawing = false;
      for (let index = 0; index < item.values.length; index += 1) {
        const value = item.values[index];
        if (!Number.isFinite(value)) {
          unstable = true;
          drawing = false;
          continue;
        }
        observedMaximum = Math.max(observedMaximum, value);
        observedMinimum = Math.min(observedMinimum, value);
        const x = margins.left + index / Math.max(1, length - 1) * plotWidth;
        const y = valueY(value);
        if (!drawing) {
          context.moveTo(x, y);
          drawing = true;
        } else {
          context.lineTo(x, y);
        }
      }
      context.strokeStyle = item.color;
      context.lineWidth = item.width || 1.5;
      context.lineJoin = "round";
      context.setLineDash(item.dash || []);
      context.stroke();
    });
    context.restore();

    if (unstable || observedMaximum > bounds.maximum) {
      drawEdgeWarning(context, width - margins.right, margins.top + 10, "↑ max " + formatScore(unstable ? Infinity : observedMaximum) + " (limit " + String(bounds.maximum) + ")");
    }
    if (observedMinimum < bounds.minimum) {
      drawEdgeWarning(context, width - margins.right, margins.top + plotHeight - 10, "↓ min " + formatScore(observedMinimum) + " (limit " + String(bounds.minimum) + ")");
    }
  }

  function updatePreviewPredictions(preview, sgdWeights, idbdWeights) {
    for (let step = 0; step < PREVIEW_LENGTH; step += 1) {
      const offset = step * FEATURE_COUNT;
      let sgdPrediction = 0;
      let idbdPrediction = 0;
      for (let feature = 0; feature < FEATURE_COUNT; feature += 1) {
        if (!preview.features[offset + feature]) continue;
        sgdPrediction += sgdWeights[feature];
        idbdPrediction += idbdWeights[feature];
      }
      preview.sgdPrediction[step] = sgdPrediction;
      preview.idbdPrediction[step] = idbdPrediction;
    }
  }

  function drawLineChart(canvasId, steps, values, color, fill, bounds, exampleCount) {
    const canvas = byId(canvasId);
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
    let observedMaximum = -Infinity;
    let observedMinimum = Infinity;

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
        if (values[index] !== null) {
          aboveScale = true;
          observedMaximum = Infinity;
        }
        continue;
      }
      observedMaximum = Math.max(observedMaximum, values[index]);
      observedMinimum = Math.min(observedMinimum, values[index]);
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
      drawEdgeWarning(context, width - margins.right, margins.top + 10, "↑ max " + formatScore(observedMaximum) + " (limit " + String(maximum) + ")");
    }
    if (belowScale) {
      drawEdgeWarning(context, width - margins.right, margins.top + plotHeight - 10, "↓ min " + formatScore(observedMinimum) + " (limit " + String(minimum) + ")");
    }
  }

  function drawLogSeriesChart(canvasId, steps, series, bounds, exampleCount) {
    const canvas = byId(canvasId);
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
    const logMaximum = Math.log10(maximum);
    const logRange = logMaximum - logMinimum;
    let observedMaximum = -Infinity;
    let observedMinimum = Infinity;
    let unstable = false;

    function valueY(value) {
      const logValue = Math.log10(clamp(value, minimum, maximum));
      return margins.top + (1 - (logValue - logMinimum) / logRange) * plotHeight;
    }

    context.fillStyle = "#fcfcfc";
    context.fillRect(margins.left, margins.top, plotWidth, plotHeight);
    context.font = "11px serif";
    context.lineWidth = 1;
    context.textBaseline = "middle";
    for (let exponent = Math.ceil(logMinimum); exponent <= Math.floor(logMaximum); exponent += 1) {
      const value = Math.pow(10, exponent);
      const y = valueY(value);
      context.strokeStyle = COLORS.grid;
      context.beginPath();
      context.moveTo(margins.left, y);
      context.lineTo(width - margins.right, y);
      context.stroke();
      context.fillStyle = COLORS.muted;
      context.textAlign = "right";
      const label = exponent === 0 ? "1" : exponent > 0 ? "10" : "1e−" + String(Math.abs(exponent));
      context.fillText(label, margins.left - 7, y);
    }

    context.strokeStyle = "#aaa";
    context.beginPath();
    context.moveTo(margins.left, margins.top);
    context.lineTo(margins.left, margins.top + plotHeight);
    context.lineTo(width - margins.right, margins.top + plotHeight);
    context.stroke();

    context.fillStyle = COLORS.muted;
    context.textBaseline = "alphabetic";
    context.textAlign = "left";
    context.fillText("0", margins.left, height - 5);
    context.textAlign = "right";
    context.fillText(formatInteger(exampleCount) + " examples", width - margins.right, height - 5);

    context.save();
    context.beginPath();
    context.rect(margins.left, margins.top, plotWidth, plotHeight);
    context.clip();
    series.forEach(function (item) {
      context.beginPath();
      let drawing = false;
      let lastPoint = null;
      for (let index = 0; index < item.values.length; index += 1) {
        const value = item.values[index];
        if (!Number.isFinite(value) || value <= 0) {
          if (value !== null) unstable = true;
          drawing = false;
          continue;
        }
        observedMaximum = Math.max(observedMaximum, value);
        observedMinimum = Math.min(observedMinimum, value);
        const point = {
          x: margins.left + steps[index] / exampleCount * plotWidth,
          y: valueY(value)
        };
        if (drawing) context.lineTo(point.x, point.y);
        else context.moveTo(point.x, point.y);
        drawing = true;
        lastPoint = point;
      }
      context.strokeStyle = item.color;
      context.lineWidth = item.width || 2;
      context.lineJoin = "round";
      context.setLineDash(item.dash || []);
      context.stroke();
      if (lastPoint) {
        context.setLineDash([]);
        context.beginPath();
        context.arc(lastPoint.x, lastPoint.y, 3, 0, Math.PI * 2);
        context.fillStyle = item.color;
        context.fill();
      }
    });
    context.restore();

    if (unstable || observedMaximum > maximum) {
      drawEdgeWarning(context, width - margins.right, margins.top + 10, "↑ max " + formatScore(unstable ? Infinity : observedMaximum) + " (limit " + String(maximum) + ")");
    }
    if (observedMinimum < minimum) {
      drawEdgeWarning(context, width - margins.right, margins.top + plotHeight - 10, "↓ min " + formatScore(observedMinimum) + " (limit " + String(minimum) + ")");
    }
  }

  function drawLearningRates(canvasId, color, rateAt) {
    const canvas = byId(canvasId);
    const surface = setupCanvas(canvas);
    const context = surface.context;
    const width = surface.width;
    const height = surface.height;
    const margins = { top: 10, right: 9, bottom: 34, left: 42 };
    const plotWidth = width - margins.left - margins.right;
    const plotHeight = height - margins.top - margins.bottom;
    const minimum = LEARNING_RATE_BOUNDS.minimum;
    const maximum = LEARNING_RATE_BOUNDS.maximum;
    const logMinimum = Math.log10(minimum);
    const logRange = Math.log10(maximum) - logMinimum;
    const barStep = plotWidth / FEATURE_COUNT;
    let aboveScale = false;
    let belowScale = false;
    let observedMaximum = -Infinity;
    let observedMinimum = Infinity;

    function rateY(rate) {
      const logRate = Math.log10(clamp(rate, minimum, maximum));
      return margins.top + (1 - (logRate - logMinimum) / logRange) * plotHeight;
    }

    context.fillStyle = "#fcfcfc";
    context.fillRect(margins.left, margins.top, plotWidth, plotHeight);
    context.font = "11px serif";
    context.lineWidth = 1;
    context.textBaseline = "middle";
    for (let exponent = -5; exponent <= Math.log10(maximum); exponent += 1) {
      const rate = Math.pow(10, exponent);
      const y = rateY(rate);
      context.strokeStyle = COLORS.grid;
      context.beginPath();
      context.moveTo(margins.left, y);
      context.lineTo(width - margins.right, y);
      context.stroke();
      context.fillStyle = COLORS.muted;
      context.textAlign = "right";
      context.fillText(exponent === 0 ? "1" : exponent > 0 ? "10" : "1e−" + String(Math.abs(exponent)), margins.left - 7, y);
    }

    context.strokeStyle = "#aaa";
    context.beginPath();
    context.moveTo(margins.left, margins.top);
    context.lineTo(margins.left, margins.top + plotHeight);
    context.lineTo(width - margins.right, margins.top + plotHeight);
    context.stroke();

    for (let index = 0; index < FEATURE_COUNT; index += 1) {
      const rate = rateAt(index);
      if (!Number.isFinite(rate)) {
        aboveScale = true;
        observedMaximum = Infinity;
        continue;
      }
      observedMaximum = Math.max(observedMaximum, rate);
      observedMinimum = Math.min(observedMinimum, rate);
      if (rate > maximum) aboveScale = true;
      if (rate < minimum) belowScale = true;
      if (rate <= 0) continue;
      const y = rateY(rate);
      const x = margins.left + index * barStep + Math.max(0.5, barStep * 0.12);
      context.fillStyle = index === 0 ? COLORS.signal : color;
      context.globalAlpha = index === 0 ? 1 : 0.62;
      context.fillRect(x, y, Math.max(1, barStep * 0.72), margins.top + plotHeight - y);
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
      drawEdgeWarning(context, width - margins.right, margins.top + 10, "↑ max " + formatScore(observedMaximum) + " (limit " + String(maximum) + ")");
    }
    if (belowScale) {
      drawEdgeWarning(context, width - margins.right, margins.top + plotHeight - 10, "↓ min " + formatScore(observedMinimum) + " (limit 0.00001)");
    }
  }

  function drawSignedParameterValues(canvasId, values, color, options) {
    const canvas = byId(canvasId);
    const surface = setupCanvas(canvas);
    const context = surface.context;
    const width = surface.width;
    const height = surface.height;
    const margins = { top: 11, right: 9, bottom: 34, left: 42 };
    const plotWidth = width - margins.left - margins.right;
    const plotHeight = height - margins.top - margins.bottom;
    const extent = options.extent;
    const linearThreshold = options.linearThreshold || WEIGHT_LINEAR_THRESHOLD;
    const transformedExtent = symmetricLog(extent, linearThreshold);
    const zeroY = margins.top + plotHeight / 2;
    const barStep = plotWidth / FEATURE_COUNT;
    let aboveScale = false;
    let belowScale = false;
    let observedMaximum = -Infinity;
    let observedMinimum = Infinity;

    function valueY(value) {
      return zeroY - symmetricLog(clamp(value, -extent, extent), linearThreshold) /
        transformedExtent * (plotHeight / 2);
    }

    const ticks = [0];
    const largestExponent = Math.ceil(Math.log10(extent));
    for (let exponent = options.minimumTickExponent; exponent <= largestExponent; exponent += 1) {
      const magnitude = Math.pow(10, exponent);
      if (magnitude <= extent * 1.000001) {
        ticks.push(-magnitude, magnitude);
      }
    }
    ticks.sort(function (first, second) { return first - second; });

    context.font = "11px serif";
    ticks.forEach(function (tick) {
      const y = valueY(tick);
      context.strokeStyle = tick === 0 ? COLORS.noise : COLORS.grid;
      context.beginPath();
      context.moveTo(margins.left, y);
      context.lineTo(width - margins.right, y);
      context.stroke();
      context.fillStyle = COLORS.muted;
      context.textAlign = "right";
      context.textBaseline = "middle";
      const magnitude = Math.abs(tick);
      const magnitudeLabel = magnitude !== 0 && magnitude < 0.001
        ? magnitude.toExponential(0).replace("e-", "e−")
        : String(magnitude);
      const label = tick < 0 ? "−" + magnitudeLabel : magnitudeLabel;
      context.fillText(label, margins.left - 7, y);
    });

    if (options.referenceValue !== undefined) {
      const referenceY = valueY(options.referenceValue);
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
      context.fillText(options.referenceLabel, width - margins.right, referenceY - 8);
    }

    for (let index = 0; index < values.length; index += 1) {
      if (!Number.isFinite(values[index])) {
        aboveScale = true;
        observedMaximum = Infinity;
      } else {
        observedMaximum = Math.max(observedMaximum, values[index]);
        observedMinimum = Math.min(observedMinimum, values[index]);
        if (values[index] > extent) aboveScale = true;
        if (values[index] < -extent) belowScale = true;
      }
      const value = clamp(values[index], -extent, extent);
      const yValue = valueY(value);
      const valueHeight = Math.abs(yValue - zeroY);
      const x = margins.left + index * barStep + Math.max(0.5, barStep * 0.12);
      const y = Math.min(yValue, zeroY);
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
      const maximumLabel = Number.isFinite(observedMaximum) ? "+" + formatScore(observedMaximum) : formatScore(observedMaximum);
      drawEdgeWarning(context, width - margins.right, margins.top + 10, "↑ max " + maximumLabel + " (limit +" + String(extent) + ")");
    }
    if (belowScale) {
      drawEdgeWarning(context, width - margins.right, margins.top + plotHeight - 10, "↓ min −" + formatScore(Math.abs(observedMinimum)) + " (limit −" + String(extent) + ")");
    }
  }

  function drawWeights(canvasId, weights, color) {
    drawSignedParameterValues(canvasId, weights, color, {
      extent: MODEL_WEIGHT_EXTENT,
      minimumTickExponent: -2,
      referenceValue: 1,
      referenceLabel: "target = 1"
    });
  }

  function drawIdbdTrace(canvasId, trace) {
    drawSignedParameterValues(canvasId, trace, COLORS.trace, {
      extent: TRACE_EXTENT,
      minimumTickExponent: -3,
      linearThreshold: 0.0001
    });
  }

  function updateBatchNote(batchSize) {
    const note = byId("batch-note");
    if (extensionsEnabled) {
      if (batchSize === 1) {
        note.innerHTML = "<strong>At batch size 1</strong>, the gradient stream is online. With both shared additions off this reduces to plain SGD and classic IDBD; momentum or decay changes both updates and IDBD’s trace.";
      } else {
        note.innerHTML = "<strong>At batch size " + batchSize + "</strong>, both learners use the shared optimizer settings on the same mean-gradient batch; IDBD retains its diagonal trace approximation.";
      }
      return;
    }
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
    const momentum = extensionsEnabled ? Number(controls.momentum.value) : 0;
    const momentumMode = extensionsEnabled ? controls.momentumMode.value : "derived";
    const weightDecay = extensionsEnabled
      ? WEIGHT_DECAY_OPTIONS[Number(controls.weightDecay.value)]
      : 0;
    const weightDecayMode = extensionsEnabled ? controls.weightDecayMode.value : "traced";
    return {
      sgdRate, idbdInitialRate, theta, batchSize, exampleCount,
      momentum, momentumMode, weightDecay, weightDecayMode
    };
  }

  function updateControlOutputs(settings) {
    const values = settings || selectedSettings();
    outputs.sgdRate.value = formatRate(values.sgdRate);
    outputs.idbdRate.value = formatRate(values.idbdInitialRate);
    outputs.theta.value = formatRate(values.theta);
    outputs.batch.value = String(values.batchSize);
    outputs.steps.value = formatInteger(values.exampleCount);
    if (extensionsEnabled) {
      outputs.momentum.value = values.momentum.toFixed(2);
      outputs.weightDecay.value = values.weightDecay === 0 ? "off" : formatRate(values.weightDecay);
    }
    byId("example-count-label").textContent = formatInteger(values.exampleCount) + " examples";
  }

  function idbdRateSummary(beta) {
    const rates = Array.from(beta, Math.exp);
    const irrelevant = rates.slice(1).sort(function (first, second) {
      return first - second;
    });
    return {
      signal: rates[0],
      medianNoise: irrelevant[Math.floor(irrelevant.length / 2)]
    };
  }

  function idbdRateRatio(beta) {
    const summary = idbdRateSummary(beta);
    return summary.signal / Math.max(summary.medianNoise, 1e-12);
  }

  function drawTrainingState(state) {
    materializeAllFeatures(state, state.batchIndex);
    byId("sgd-loss-score").textContent = state.sgdLossEma === null ? "—" : formatScore(state.sgdLossEma);
    byId("idbd-loss-score").textContent = state.idbdLossEma === null ? "—" : formatScore(state.idbdLossEma);
    byId("sgd-signal-weight").textContent = formatScore(state.sgdWeights[0]);
    byId("idbd-rate-ratio").textContent = formatScore(idbdRateRatio(state.beta)) + "×";
    byId("sgd-signal-loss-score").textContent = formatScore(noiselessSignalMse(state.sgdWeights));
    byId("idbd-signal-loss-score").textContent = formatScore(noiselessSignalMse(state.idbdWeights));

    updatePreviewPredictions(state.preview, state.sgdWeights, state.idbdWeights);
    if (byId("walkthrough-sgd-chart")) {
      drawTraceChart("walkthrough-sgd-chart", [
        { values: state.preview.noisyTarget, color: "rgba(73, 73, 73, 0.28)", width: 0.8 },
        { values: state.preview.cleanTarget, color: COLORS.signal, width: 1.2, dash: [4, 3] },
        { values: state.preview.sgdPrediction, color: COLORS.sgd, width: 2 }
      ], { minimum: -10, maximum: 10 }, [-10, 0, 1, 10]);
    }
    if (byId("walkthrough-idbd-chart")) {
      drawTraceChart("walkthrough-idbd-chart", [
        { values: state.preview.noisyTarget, color: "rgba(73, 73, 73, 0.28)", width: 0.8 },
        { values: state.preview.cleanTarget, color: COLORS.signal, width: 1.2, dash: [4, 3] },
        { values: state.preview.idbdPrediction, color: COLORS.idbd, width: 2 }
      ], { minimum: -10, maximum: 10 }, [-10, 0, 1, 10]);
    }
    drawTraceChart("feature-stream-chart", [
      { values: state.preview.cleanTarget, color: COLORS.signal, width: 1.6 }
    ], { minimum: -0.2, maximum: 1.2 }, [0, 1]);
    drawTraceChart("target-stream-chart", [
      { values: state.preview.noisyTarget, color: COLORS.ink, width: 1.25 }
    ], { minimum: -8, maximum: 8 }, [-8, 0, 8]);
    drawTraceChart("sgd-prediction-chart", [
      { values: state.preview.cleanTarget, color: COLORS.signal, width: 1.1, dash: [4, 3] },
      { values: state.preview.sgdPrediction, color: COLORS.sgd, width: 1.7 }
    ], { minimum: -2, maximum: 2 }, [-2, 0, 1, 2]);
    drawTraceChart("idbd-prediction-chart", [
      { values: state.preview.cleanTarget, color: COLORS.signal, width: 1.1, dash: [4, 3] },
      { values: state.preview.idbdPrediction, color: COLORS.idbd, width: 1.7 }
    ], { minimum: -2, maximum: 2 }, [-2, 0, 1, 2]);

    drawLineChart("sgd-loss-chart", state.curves.steps, state.curves.sgdLoss, COLORS.sgd, COLORS.sgdFill, PREDICTION_LOSS_BOUNDS, state.exampleCount);
    drawLineChart("idbd-loss-chart", state.curves.steps, state.curves.idbdLoss, COLORS.idbd, COLORS.idbdFill, PREDICTION_LOSS_BOUNDS, state.exampleCount);
    drawLineChart("sgd-signal-loss-chart", state.curves.steps, state.curves.sgdSignalLoss, COLORS.sgd, COLORS.sgdFill, SIGNAL_LOSS_BOUNDS, state.exampleCount);
    drawLineChart("idbd-signal-loss-chart", state.curves.steps, state.curves.idbdSignalLoss, COLORS.idbd, COLORS.idbdFill, SIGNAL_LOSS_BOUNDS, state.exampleCount);
    if (byId("sgd-rate-history-chart")) {
      drawLogSeriesChart("sgd-rate-history-chart", state.curves.steps, [
        { values: state.curves.steps.map(function () { return state.sgdRate; }), color: COLORS.sgd, width: 2.2 }
      ], LEARNING_RATE_BOUNDS, state.exampleCount);
    }
    drawLogSeriesChart("idbd-rate-history-chart", state.curves.steps, [
      { values: state.curves.idbdSignalRate, color: COLORS.signal, width: 2.4 },
      { values: state.curves.idbdMedianNoiseRate, color: COLORS.noise, width: 2, dash: [5, 3] }
    ], LEARNING_RATE_BOUNDS, state.exampleCount);
    drawIdbdTrace("idbd-trace-chart", state.trace);
    drawLearningRates("sgd-rates-chart", COLORS.sgd, function () { return state.sgdRate; });
    drawLearningRates("idbd-rates-chart", COLORS.idbd, function (index) { return Math.exp(state.beta[index]); });
    drawWeights("sgd-weights-chart", state.sgdWeights, COLORS.sgd);
    drawWeights("idbd-weights-chart", state.idbdWeights, COLORS.idbd);
  }

  function elapsedLabel(milliseconds) {
    return milliseconds < 1000
      ? Math.round(milliseconds) + " ms"
      : (milliseconds / 1000).toFixed(1) + " s";
  }

  function progressLabel(label, state) {
    const percent = 100 * state.examplesSeen / state.exampleCount;
    return label + " · " + formatInteger(state.examplesSeen) + " / " +
      formatInteger(state.exampleCount) + " · " + percent.toFixed(0) + "%";
  }

  function updateTransportControls() {
    const complete = currentState && currentState.examplesSeen >= currentState.exampleCount;
    const active = currentState && !complete && !paused && !stopped && !scheduled;
    controls.pause.disabled = !active;
    controls.play.disabled = !currentState || active || scheduled;
    controls.stop.disabled = !currentState || complete || stopped || scheduled;
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
    if (version !== runVersion || paused || stopped) return;
    processTrainingChunk(state);
    if (version !== runVersion || paused || stopped) return;
    drawTrainingState(state);
    const complete = state.examplesSeen >= state.exampleCount;
    updateProgressStatus(state, complete);
    updateTransportControls();
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
      settings.exampleCount,
      settings
    );
    currentState = state;
    paused = false;
    stopped = false;
    pausedAt = 0;
    updateBatchNote(settings.batchSize);
    drawTrainingState(state);
    outputs.status.textContent = "Training · 0 / " + formatInteger(settings.exampleCount) + " · 0%";
    updateTransportControls();
    window.requestAnimationFrame(function () {
      continueTraining(version, state);
    });
  }

  function scheduleTraining() {
    runVersion += 1;
    paused = false;
    stopped = false;
    pausedAt = 0;
    updateControlOutputs();
    outputs.status.textContent = currentState ? "Restarting…" : "Preparing…";
    updateTransportControls();
    if (scheduled) return;
    scheduled = true;
    scheduledFrame = window.requestAnimationFrame(function () {
      scheduled = false;
      scheduledFrame = null;
      if (stopped) return;
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
  if (extensionsEnabled) {
    controls.momentum.addEventListener("input", scheduleTraining);
    controls.momentumMode.addEventListener("change", scheduleTraining);
    controls.weightDecay.addEventListener("input", scheduleTraining);
    controls.weightDecayMode.addEventListener("change", scheduleTraining);
  }
  controls.reseed.addEventListener("click", function () {
    const requestedSeed = Number(controls.seed.value);
    if (!Number.isFinite(requestedSeed)) {
      controls.seed.value = String(seed);
      return;
    }
    seed = Math.min(4294967295, Math.max(0, Math.trunc(requestedSeed)));
    controls.seed.value = String(seed);
    scheduleTraining();
  });
  controls.seed.addEventListener("keydown", function (event) {
    if (event.key === "Enter") controls.reseed.click();
  });
  controls.newStream.addEventListener("click", function () {
    seed = (seed + 1) >>> 0;
    controls.seed.value = String(seed);
    scheduleTraining();
  });
  controls.pause.addEventListener("click", function () {
    if (!currentState || paused || stopped || currentState.examplesSeen >= currentState.exampleCount) return;
    paused = true;
    pausedAt = performance.now();
    outputs.status.textContent = progressLabel("Paused", currentState);
    updateTransportControls();
  });
  controls.play.addEventListener("click", function () {
    if (!currentState) return;
    if (paused) {
      currentState.started += performance.now() - pausedAt;
      paused = false;
      pausedAt = 0;
      outputs.status.textContent = progressLabel("Training", currentState);
      updateTransportControls();
      const version = runVersion;
      const state = currentState;
      window.requestAnimationFrame(function () {
        continueTraining(version, state);
      });
      return;
    }
    if (stopped || currentState.examplesSeen >= currentState.exampleCount) scheduleTraining();
  });
  controls.stop.addEventListener("click", function () {
    if (!currentState || stopped || currentState.examplesSeen >= currentState.exampleCount) return;
    runVersion += 1;
    if (scheduledFrame !== null) window.cancelAnimationFrame(scheduledFrame);
    scheduledFrame = null;
    scheduled = false;
    paused = false;
    pausedAt = 0;
    stopped = true;
    outputs.status.textContent = progressLabel("Stopped", currentState);
    updateTransportControls();
  });
  window.addEventListener("resize", function () {
    if (resizeScheduled) return;
    resizeScheduled = true;
    window.requestAnimationFrame(function () {
      resizeScheduled = false;
      if (currentState) drawTrainingState(currentState);
    });
  });
  updateTransportControls();
  scheduleTraining();
  }

  function prefixCloneIds(clone, prefix) {
    const idMap = new Map();
    clone.querySelectorAll("[id]").forEach(function (element) {
      const oldId = element.id;
      const newId = prefix + "-" + oldId;
      idMap.set(oldId, newId);
      element.id = newId;
    });
    ["for", "aria-labelledby", "aria-describedby"].forEach(function (attribute) {
      const elements = Array.from(clone.querySelectorAll("[" + attribute + "]"));
      if (clone.hasAttribute(attribute)) elements.unshift(clone);
      elements.forEach(function (element) {
        const rewritten = element.getAttribute(attribute).split(/\s+/).map(function (value) {
          return idMap.get(value) || value;
        }).join(" ");
        element.setAttribute(attribute, rewritten);
      });
    });
  }

  function mountExtendedExperiment() {
    const source = document.getElementById("idbd-playground");
    const mount = document.getElementById("extended-experiment-mount");
    if (!source || !mount) return;
    const clone = source.cloneNode(true);
    clone.removeAttribute("hidden");
    clone.id = "extended-idbd-playground";
    clone.classList.add("extended-experiment");
    prefixCloneIds(clone, "extended");
    clone.querySelector(".experiment-heading .section-label").textContent = "Interactive experiment · optimizer grafts";
    clone.querySelector(".experiment-heading h2").textContent = "What changes when both learners inherit momentum and decay?";
    clone.querySelector(".experiment-lede").textContent = "The same stream, now with shared momentum and weight decay for SGD and IDBD, plus IDBD-specific choices for how its meta-gradient trace follows those updates.";
    clone.querySelector(".sgd-card .method-pill").textContent = "fixed rate + shared grafts";
    clone.querySelector(".idbd-card .method-pill").textContent = "one rate per feature + grafts";
    clone.querySelector(".simulation-scope").innerHTML = "<strong>Computed live in this browser.</strong> Momentum and weight decay are shared by both learners; the trace mechanism choices apply only to IDBD. Training yields between animation frames, and changing a setting cancels and restarts the run.";

    const sharedGraftControls = document.createElement("div");
    sharedGraftControls.className = "optimizer-graft-controls shared-optimizer-controls";
    sharedGraftControls.innerHTML = [
      '<p class="graft-controls-title">Shared optimizer additions · SGD + IDBD</p>',
      '<div class="control">',
      '  <div class="control-heading"><label for="extended-momentum">Momentum <span class="math">μ</span></label><output id="extended-momentum-value" for="extended-momentum">0.00</output></div>',
      '  <input id="extended-momentum" type="range" min="0" max="0.99" step="0.01" value="0" aria-describedby="extended-momentum-help">',
      '  <div class="range-labels" aria-hidden="true"><span>off</span><span>0.99</span></div>',
      '  <p id="extended-momentum-help">The same EMA momentum coefficient and update direction for both learners.</p>',
      '</div>',
      '<div class="control">',
      '  <div class="control-heading"><label for="extended-weight-decay">Weight decay <span class="math">λ</span></label><output id="extended-weight-decay-value" for="extended-weight-decay">off</output></div>',
      '  <input id="extended-weight-decay" type="range" min="0" max="7" step="1" value="0" aria-describedby="extended-weight-decay-help">',
      '  <div class="range-labels" aria-hidden="true"><span>off</span><span>1.0</span></div>',
      '  <p id="extended-weight-decay-help">The same decay strength for both; each learner scales it by its own learning rate.</p>',
      '</div>'
    ].join("");
    clone.querySelector(".shared-controls").insertBefore(
      sharedGraftControls,
      clone.querySelector(".shared-control-grid")
    );

    const idbdTraceControls = document.createElement("div");
    idbdTraceControls.className = "idbd-trace-controls";
    idbdTraceControls.innerHTML = [
      '<p class="graft-controls-title">IDBD trace treatment</p>',
      '<div class="control">',
      '  <label class="mode-label" for="extended-momentum-mode">With momentum</label>',
      '  <select id="extended-momentum-mode"><option value="derived" selected>Derived trace</option><option value="naive">Naïve trace</option></select>',
      '  <p>Derived mode carries p = dm/dβ.</p>',
      '</div>',
      '<div class="control">',
      '  <label class="mode-label" for="extended-weight-decay-mode">Decay mechanism</label>',
      '  <select id="extended-weight-decay-mode"><option value="traced" selected>Traced α-coupled</option><option value="alpha_coupled">α-coupled</option><option value="fixed">Fixed-rate</option></select>',
      '  <p>Traced mode includes decay’s β-derivative in h.</p>',
      '</div>'
    ].join("");
    clone.querySelector(".idbd-method-controls").appendChild(idbdTraceControls);
    mount.replaceWith(clone);
    source.remove();
  }

  mountExtendedExperiment();
  createExperiment("extended", true);
}());
