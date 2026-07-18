(function () {
  "use strict";

  const FEATURE_COUNT = 64;
  const EXAMPLE_COUNT = 4096;
  const FEATURE_PROBABILITY = 0.08;
  const GAUSSIAN_VARIANCE = 2;
  const SPARSE_NOISE_PROBABILITY = 0.02;
  const COLORS = {
    ink: "#15201d",
    muted: "#707a75",
    grid: "#e9ebe6",
    sgd: "#5d6973",
    sgdFill: "rgba(93, 105, 115, 0.10)",
    idbd: "#df6844",
    idbdFill: "rgba(223, 104, 68, 0.11)",
    signal: "#16805b",
    noise: "#aab1ad"
  };

  const controls = {
    rate: document.getElementById("initial-rate"),
    theta: document.getElementById("theta"),
    batch: document.getElementById("batch-size")
  };
  const outputs = {
    rate: document.getElementById("initial-rate-value"),
    theta: document.getElementById("theta-value"),
    batch: document.getElementById("batch-size-value"),
    status: document.getElementById("run-status")
  };

  if (!controls.rate || !controls.theta || !controls.batch) return;

  let seed = 20260713;
  let scheduled = false;

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

  function gaussian(random) {
    const first = Math.max(random(), 1e-12);
    const second = random();
    return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
  }

  function makeStream(streamSeed) {
    const random = mulberry32(streamSeed);
    const features = new Array(EXAMPLE_COUNT);
    const targets = new Float64Array(EXAMPLE_COUNT);

    for (let row = 0; row < EXAMPLE_COUNT; row += 1) {
      const active = [];
      for (let feature = 0; feature < FEATURE_COUNT; feature += 1) {
        if (random() < FEATURE_PROBABILITY) active.push(feature);
      }
      const cleanTarget = active.length && active[0] === 0 ? 1 : 0;
      const sparseNoise = random() < SPARSE_NOISE_PROBABILITY
        ? (random() < 0.5 ? -1 : 1)
        : 0;
      features[row] = active;
      targets[row] = cleanTarget + sparseNoise + Math.sqrt(GAUSSIAN_VARIANCE) * gaussian(random);
    }
    return { features, targets };
  }

  function prediction(weights, active) {
    let total = 0;
    for (let index = 0; index < active.length; index += 1) {
      total += weights[active[index]];
    }
    return total;
  }

  function cleanRms(weights) {
    let coefficientSum = 0;
    let coefficientSquares = 0;
    for (let index = 0; index < weights.length; index += 1) {
      const coefficient = weights[index] - (index === 0 ? 1 : 0);
      coefficientSum += coefficient;
      coefficientSquares += coefficient * coefficient;
    }
    const p = FEATURE_PROBABILITY;
    return Math.sqrt(Math.max(0, p * (1 - p) * coefficientSquares + p * p * coefficientSum * coefficientSum));
  }

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function runExperiment(stream, initialRate, theta, batchSize) {
    const sgdWeights = new Float64Array(FEATURE_COUNT);
    const idbdWeights = new Float64Array(FEATURE_COUNT);
    const beta = new Float64Array(FEATURE_COUNT);
    const trace = new Float64Array(FEATURE_COUNT);
    const directionSgd = new Float64Array(FEATURE_COUNT);
    const directionIdbd = new Float64Array(FEATURE_COUNT);
    const activeCounts = new Float64Array(FEATURE_COUNT);
    beta.fill(Math.log(initialRate));

    const curves = {
      steps: [0],
      sgdLoss: [null],
      idbdLoss: [null],
      sgdClean: [cleanRms(sgdWeights)],
      idbdClean: [cleanRms(idbdWeights)]
    };
    let sgdLossEma = null;
    let idbdLossEma = null;

    for (let start = 0; start < EXAMPLE_COUNT; start += batchSize) {
      const end = Math.min(EXAMPLE_COUNT, start + batchSize);
      const actualBatchSize = end - start;
      directionSgd.fill(0);
      directionIdbd.fill(0);
      activeCounts.fill(0);
      let batchSgdSquaredError = 0;
      let batchIdbdSquaredError = 0;

      for (let row = start; row < end; row += 1) {
        const active = stream.features[row];
        const target = stream.targets[row];
        const sgdError = target - prediction(sgdWeights, active);
        const idbdError = target - prediction(idbdWeights, active);
        batchSgdSquaredError += sgdError * sgdError;
        batchIdbdSquaredError += idbdError * idbdError;

        for (let index = 0; index < active.length; index += 1) {
          const feature = active[index];
          directionSgd[feature] += sgdError;
          directionIdbd[feature] += idbdError;
          activeCounts[feature] += 1;
        }
      }

      for (let feature = 0; feature < FEATURE_COUNT; feature += 1) {
        const sgdDirection = directionSgd[feature] / actualBatchSize;
        const idbdDirection = directionIdbd[feature] / actualBatchSize;
        const curvature = activeCounts[feature] / actualBatchSize;
        sgdWeights[feature] += initialRate * sgdDirection;

        const betaChange = clamp(theta * idbdDirection * trace[feature], -2, 2);
        beta[feature] = clamp(beta[feature] + betaChange, -10, Math.log(0.5));
        const featureRate = Math.exp(beta[feature]);
        const weightChange = featureRate * idbdDirection;
        idbdWeights[feature] += weightChange;
        trace[feature] = trace[feature] * Math.max(0, 1 - featureRate * curvature) + weightChange;
      }

      const sgdBatchLoss = batchSgdSquaredError / actualBatchSize;
      const idbdBatchLoss = batchIdbdSquaredError / actualBatchSize;
      sgdLossEma = sgdLossEma === null ? sgdBatchLoss : 0.92 * sgdLossEma + 0.08 * sgdBatchLoss;
      idbdLossEma = idbdLossEma === null ? idbdBatchLoss : 0.92 * idbdLossEma + 0.08 * idbdBatchLoss;

      const batchIndex = Math.floor(start / batchSize);
      const totalBatches = Math.ceil(EXAMPLE_COUNT / batchSize);
      const recordEvery = Math.max(1, Math.floor(totalBatches / 150));
      if (batchIndex % recordEvery === 0 || end === EXAMPLE_COUNT) {
        curves.steps.push(end);
        curves.sgdLoss.push(sgdLossEma);
        curves.idbdLoss.push(idbdLossEma);
        curves.sgdClean.push(cleanRms(sgdWeights));
        curves.idbdClean.push(cleanRms(idbdWeights));
      }
    }

    const idbdRates = Array.from(beta, Math.exp);
    const irrelevantRates = idbdRates.slice(1).sort(function (a, b) { return a - b; });
    const medianNoiseRate = irrelevantRates[Math.floor(irrelevantRates.length / 2)];
    return {
      curves,
      sgd: {
        weights: Array.from(sgdWeights),
        clean: cleanRms(sgdWeights),
        loss: sgdLossEma,
        signal: sgdWeights[0]
      },
      idbd: {
        weights: Array.from(idbdWeights),
        clean: cleanRms(idbdWeights),
        loss: idbdLossEma,
        rateRatio: idbdRates[0] / Math.max(medianNoiseRate, 1e-12)
      }
    };
  }

  function formatRate(value) {
    if (value < 0.001) return value.toExponential(1);
    if (value < 0.1) return value.toFixed(4);
    return value.toFixed(3);
  }

  function formatScore(value) {
    if (!Number.isFinite(value)) return "unstable";
    if (value >= 100) return value.toExponential(1);
    return value.toFixed(3);
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

  function niceMaximum(value) {
    if (!Number.isFinite(value) || value <= 0) return 1;
    const power = Math.pow(10, Math.floor(Math.log10(value)));
    const scaled = value / power;
    const nice = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
    return nice * power;
  }

  function drawLineChart(canvasId, steps, values, color, fill, sharedMax) {
    const canvas = document.getElementById(canvasId);
    const surface = setupCanvas(canvas);
    const context = surface.context;
    const width = surface.width;
    const height = surface.height;
    const margins = { top: 10, right: 10, bottom: 27, left: 42 };
    const plotWidth = width - margins.left - margins.right;
    const plotHeight = height - margins.top - margins.bottom;
    const finiteValues = values.filter(Number.isFinite);
    const maximum = sharedMax || niceMaximum(Math.max.apply(null, finiteValues));

    context.font = "10px Inter, sans-serif";
    context.lineWidth = 1;
    context.textBaseline = "middle";
    for (let tick = 0; tick <= 2; tick += 1) {
      const fraction = tick / 2;
      const y = margins.top + plotHeight * (1 - fraction);
      context.strokeStyle = COLORS.grid;
      context.beginPath();
      context.moveTo(margins.left, y);
      context.lineTo(width - margins.right, y);
      context.stroke();
      context.fillStyle = COLORS.muted;
      context.textAlign = "right";
      context.fillText(formatScore(maximum * fraction), margins.left - 7, y);
    }

    context.textBaseline = "alphabetic";
    context.fillStyle = COLORS.muted;
    context.textAlign = "left";
    context.fillText("0", margins.left, height - 5);
    context.textAlign = "right";
    context.fillText("4,096 examples", width - margins.right, height - 5);

    const coordinates = [];
    for (let index = 0; index < values.length; index += 1) {
      if (!Number.isFinite(values[index])) continue;
      coordinates.push({
        x: margins.left + (steps[index] / EXAMPLE_COUNT) * plotWidth,
        y: margins.top + (1 - clamp(values[index] / maximum, 0, 1)) * plotHeight
      });
    }
    if (!coordinates.length) return;

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
    context.lineWidth = 2;
    context.lineJoin = "round";
    context.stroke();

    const last = coordinates[coordinates.length - 1];
    context.beginPath();
    context.arc(last.x, last.y, 3, 0, Math.PI * 2);
    context.fillStyle = color;
    context.fill();
  }

  function drawWeights(canvasId, weights, color, sharedExtent) {
    const canvas = document.getElementById(canvasId);
    const surface = setupCanvas(canvas);
    const context = surface.context;
    const width = surface.width;
    const height = surface.height;
    const margins = { top: 11, right: 9, bottom: 34, left: 42 };
    const plotWidth = width - margins.left - margins.right;
    const plotHeight = height - margins.top - margins.bottom;
    const extent = Math.max(0.1, sharedExtent);
    const zeroY = margins.top + plotHeight / 2;
    const barStep = plotWidth / FEATURE_COUNT;

    context.font = "10px Inter, sans-serif";
    context.strokeStyle = COLORS.grid;
    context.beginPath();
    context.moveTo(margins.left, zeroY);
    context.lineTo(width - margins.right, zeroY);
    context.stroke();
    context.fillStyle = COLORS.muted;
    context.textAlign = "right";
    context.textBaseline = "middle";
    context.fillText("+" + formatScore(extent), margins.left - 7, margins.top);
    context.fillText("0", margins.left - 7, zeroY);
    context.fillText("−" + formatScore(extent), margins.left - 7, margins.top + plotHeight);

    for (let index = 0; index < weights.length; index += 1) {
      const value = clamp(weights[index], -extent, extent);
      const valueHeight = Math.abs(value) / extent * (plotHeight / 2);
      const x = margins.left + index * barStep + Math.max(0.5, barStep * 0.12);
      const y = value >= 0 ? zeroY - valueHeight : zeroY;
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
  }

  function maximumAcross(first, second, minimum) {
    const values = first.concat(second).filter(Number.isFinite);
    return niceMaximum(Math.max(minimum || 0, Math.max.apply(null, values)) * 1.05);
  }

  function updateBatchNote(batchSize) {
    const note = document.getElementById("batch-note");
    if (batchSize === 1) {
      note.innerHTML = "<strong>At batch size 1</strong>, this is classic online IDBD. Move the batch slider to explore a mean-gradient diagonal approximation.";
    } else {
      note.innerHTML = "<strong>At batch size " + batchSize + "</strong>, IDBD uses a mean-gradient diagonal approximation. Classic 1992 IDBD is the batch-size-one setting.";
    }
  }

  function render() {
    scheduled = false;
    const initialRate = Math.pow(10, Number(controls.rate.value));
    const theta = Math.pow(10, Number(controls.theta.value));
    const batchSize = Math.pow(2, Number(controls.batch.value));
    outputs.rate.value = formatRate(initialRate);
    outputs.theta.value = formatRate(theta);
    outputs.batch.value = String(batchSize);
    outputs.status.textContent = "Running identical streams…";

    window.requestAnimationFrame(function () {
      const stream = makeStream(seed);
      const result = runExperiment(stream, initialRate, theta, batchSize);
      const curves = result.curves;

      document.getElementById("sgd-clean-score").textContent = formatScore(result.sgd.clean);
      document.getElementById("idbd-clean-score").textContent = formatScore(result.idbd.clean);
      document.getElementById("sgd-loss-score").textContent = formatScore(result.sgd.loss);
      document.getElementById("idbd-loss-score").textContent = formatScore(result.idbd.loss);
      document.getElementById("sgd-signal-weight").textContent = formatScore(result.sgd.signal);
      document.getElementById("idbd-rate-ratio").textContent = formatScore(result.idbd.rateRatio) + "×";

      const lossMax = maximumAcross(curves.sgdLoss, curves.idbdLoss, GAUSSIAN_VARIANCE + 1);
      const cleanMax = maximumAcross(curves.sgdClean, curves.idbdClean, 1);
      const weightMax = niceMaximum(Math.max(
        1,
        Math.max.apply(null, result.sgd.weights.map(Math.abs)),
        Math.max.apply(null, result.idbd.weights.map(Math.abs))
      ) * 1.05);

      drawLineChart("sgd-loss-chart", curves.steps, curves.sgdLoss, COLORS.sgd, COLORS.sgdFill, lossMax);
      drawLineChart("idbd-loss-chart", curves.steps, curves.idbdLoss, COLORS.idbd, COLORS.idbdFill, lossMax);
      drawLineChart("sgd-clean-chart", curves.steps, curves.sgdClean, COLORS.sgd, COLORS.sgdFill, cleanMax);
      drawLineChart("idbd-clean-chart", curves.steps, curves.idbdClean, COLORS.idbd, COLORS.idbdFill, cleanMax);
      drawWeights("sgd-weights-chart", result.sgd.weights, COLORS.sgd, weightMax);
      drawWeights("idbd-weights-chart", result.idbd.weights, COLORS.idbd, weightMax);
      updateBatchNote(batchSize);
      outputs.status.textContent = "Complete · stream " + String(seed).slice(-4);
    });
  }

  function scheduleRender() {
    const initialRate = Math.pow(10, Number(controls.rate.value));
    const theta = Math.pow(10, Number(controls.theta.value));
    outputs.rate.value = formatRate(initialRate);
    outputs.theta.value = formatRate(theta);
    outputs.batch.value = String(Math.pow(2, Number(controls.batch.value)));
    if (scheduled) return;
    scheduled = true;
    window.requestAnimationFrame(render);
  }

  controls.rate.addEventListener("input", scheduleRender);
  controls.theta.addEventListener("input", scheduleRender);
  controls.batch.addEventListener("input", scheduleRender);
  document.getElementById("new-stream").addEventListener("click", function () {
    seed += 1;
    scheduleRender();
  });
  window.addEventListener("resize", scheduleRender);
  render();
}());
