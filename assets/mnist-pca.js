(async function () {
  "use strict";
  const mount = document.getElementById("mnist-pca");
  if (!mount) return;
  const $ = (id) => document.getElementById(`mnist-pca-${id}`);
  const canvas = $("chart"), ctx = canvas.getContext("2d");
  const tooltip = $("tooltip"), digit = $("digit");
  const colors = ["#2679b2", "#dc7436", "#31885a", "#cc5054", "#8c62b1", "#956444", "#c75b9d", "#394a73", "#9b8a22", "#23989f"];
  const noiseColor = "#888888";
  const pointColor = (label) => label === null ? noiseColor : colors[label];
  const pointName = (label) => label === null ? "Noise" : `Digit ${label}`;
  try {
    const atlas = new Image();
    atlas.src = mount.dataset.atlasUrl;
    const [data] = await Promise.all([
      fetch(mount.dataset.pointsUrl).then((response) => {
        if (!response.ok) throw new Error("Could not load MNIST points");
        return response.json();
      }), atlas.decode()
    ]);
    [...colors, noiseColor].forEach((color, label) => {
      const entry = document.createElement("span");
      const swatch = document.createElement("i");
      swatch.style.setProperty("--digit-color", color);
      swatch.setAttribute("aria-hidden", "true");
      entry.append(swatch, document.createTextNode(label === 10 ? "Noise · starting points" : label));
      $("legend").append(entry);
    });
    const percentages = data.explainedVariance.map((value) => (100 * value).toFixed(1));
    const total = (100 * data.explainedVariance.reduce((a, b) => a + b, 0)).toFixed(1);
    $("caption").textContent = data.representation === "model-hidden-features"
      ? `${data.digitCount} digits + ${data.noiseCount} noise starting points · ${data.featureDimension.toLocaleString("en-US")} learned features projected onto two PCs (${total}% of their joint variation). Images use the same feature extraction settings. Noise previews are clipped for display.`
      : `${data.digitCount / 10} images per digit + ${data.noiseCount} Gaussian noise starting points · Same PCA axes, fitted to ${data.pcaFitCount.toLocaleString("en-US")} digits (${total}% of pixel variation). Noise is sampled in the full image space and projected here; its preview is clipped for display.`;
    const imageSide = data.imageSide;
    digit.width = digit.height = imageSide;
    let width, height, positions = [], selected = -1;
    const xs = data.points.map((p) => p[0]), ys = data.points.map((p) => p[1]);
    const bounds = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];

    function showPreview() {
      tooltip.hidden = selected < 0;
      if (selected < 0) return;
      const [x, y] = positions[selected];
      const label = data.labels[selected];
      $("label").textContent = pointName(label);
      digit.setAttribute("aria-label", label === null ? "Gaussian noise starting image, clipped for display" : `Original MNIST image of digit ${label}`);
      digit.getContext("2d").drawImage(atlas,
        (selected % data.atlasColumns) * imageSide, Math.floor(selected / data.atlasColumns) * imageSide,
        imageSide, imageSide, 0, 0, imageSide, imageSide);
      $("coordinates").textContent = `PC1 ${data.points[selected][0].toFixed(2)} · PC2 ${data.points[selected][1].toFixed(2)}`;
      const left = x + 18 + 126 < width ? x + 18 : x - 144;
      tooltip.style.left = `${Math.max(4, Math.min(width - 130, left))}px`;
      tooltip.style.top = `${Math.max(4, Math.min(height - tooltip.offsetHeight - 4, y - 65))}px`;
    }

    function draw() {
      ctx.clearRect(0, 0, width, height);
      const left = 58, right = width - 22, top = 22, bottom = height - 48;
      // Equal units on both axes preserve the geometry of the PCA projection.
      const scale = Math.min((right - left) / ((bounds[1] - bounds[0]) * 1.12),
        (bottom - top) / ((bounds[3] - bounds[2]) * 1.12));
      const midX = (bounds[0] + bounds[1]) / 2, midY = (bounds[2] + bounds[3]) / 2;
      const px = (x) => (left + right) / 2 + (x - midX) * scale;
      const py = (y) => (top + bottom) / 2 - (y - midY) * scale;
      ctx.font = "11px system-ui, sans-serif";
      const rawStep = Math.max(bounds[1] - bounds[0], bounds[3] - bounds[2]) / 6;
      const magnitude = 10 ** Math.floor(Math.log10(rawStep));
      const tickStep = [1, 2, 5, 10].find((v) => v * magnitude >= rawStep) * magnitude;
      const visibleMin = Math.min(midX - (right - left) / (2 * scale), midY - (bottom - top) / (2 * scale));
      const visibleMax = Math.max(midX + (right - left) / (2 * scale), midY + (bottom - top) / (2 * scale));
      for (let tick = Math.ceil(visibleMin / tickStep); tick <= Math.floor(visibleMax / tickStep); tick += 1) {
        const value = Number((tick * tickStep).toPrecision(6));
        const x = px(value), y = py(value);
        ctx.strokeStyle = value === 0 ? "#b9c5be" : "#e4e9e2";
        ctx.fillStyle = "#5f6d66";
        if (x >= left && x <= right) {
          ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
          ctx.textAlign = "center"; ctx.fillText(value, x, bottom + 17);
        }
        if (y >= top && y <= bottom) {
          ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
          ctx.textAlign = "right"; ctx.fillText(value, left - 9, y + 4);
        }
      }
      ctx.fillStyle = "#5f6d66"; ctx.textAlign = "center";
      ctx.fillText(`PC1 · ${percentages[0]}% of variation`, (left + right) / 2, height - 10);
      ctx.save(); ctx.translate(15, (top + bottom) / 2); ctx.rotate(-Math.PI / 2);
      ctx.fillText(`PC2 · ${percentages[1]}%`, 0, 0); ctx.restore();
      positions = data.points.map(([x, y]) => [px(x), py(y)]);
      ctx.globalAlpha = 0.65;
      positions.forEach(([x, y], index) => {
        ctx.fillStyle = pointColor(data.labels[index]);
        ctx.beginPath(); ctx.arc(x, y, width < 500 ? 3 : 3.8, 0, 2 * Math.PI); ctx.fill();
      });
      ctx.globalAlpha = 1;
      if (selected >= 0) {
        const [x, y] = positions[selected];
        ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI);
        ctx.fillStyle = pointColor(data.labels[selected]); ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = "#17211d"; ctx.stroke(); ctx.lineWidth = 1;
      }
      showPreview();
    }

    function select(index, announce = false) {
      if (index === selected) return;
      selected = index;
      canvas.style.cursor = index >= 0 ? "pointer" : "crosshair";
      if (announce && index >= 0) $("announcement").textContent = `${pointName(data.labels[index])}, sample ${index + 1} of ${data.count}.`;
      draw();
    }
    function inspect(event) {
      const box = canvas.getBoundingClientRect();
      const x = event.clientX - box.left, y = event.clientY - box.top;
      let nearest = -1, distance = event.pointerType === "touch" ? 625 : 144;
      positions.forEach(([px, py], index) => {
        const d = (px - x) ** 2 + (py - y) ** 2;
        if (d < distance) { nearest = index; distance = d; }
      });
      select(nearest, event.type === "pointerdown");
    }
    canvas.addEventListener("pointermove", (event) => { if (event.pointerType !== "touch") inspect(event); });
    canvas.addEventListener("pointerdown", inspect);
    canvas.addEventListener("pointerleave", (event) => { if (event.pointerType !== "touch") select(-1); });
    canvas.addEventListener("focus", () => { if (selected < 0) select(0, true); });
    canvas.addEventListener("blur", () => select(-1));
    canvas.addEventListener("keydown", (event) => {
      if (event.key === "Escape") { select(-1); return; }
      if (!["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp"].includes(event.key)) return;
      event.preventDefault();
      const delta = ["ArrowRight", "ArrowDown"].includes(event.key) ? 1 : -1;
      select((Math.max(0, selected) + delta + data.count) % data.count, true);
    });
    new ResizeObserver(() => {
      const box = canvas.getBoundingClientRect();
      width = box.width; height = box.height;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }).observe(canvas);
  } catch (error) {
    $("caption").textContent = "The digit map couldn’t load. Refresh the page to try again.";
    console.error(error);
  }
})();
