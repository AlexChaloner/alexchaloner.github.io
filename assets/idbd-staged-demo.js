(function () {
  "use strict";

  const api = window.IDBDDemo;
  const mount = document.getElementById("staged-experiment-mount");
  const source = document.getElementById("idbd-playground");
  if (!api || !mount || !source) return;

  const stages = [
    {
      title: "Watch SGD learn",
      description: "One learner follows a recurring signal through a noisy stream. For now, its learning rate and horizon are fixed.",
      next: "Control the learner"
    },
    {
      title: "Control SGD",
      description: "Change how quickly SGD updates and how much experience it receives. Every change restarts the same deterministic stream.",
      next: "Introduce IDBD"
    },
    {
      title: "Meet IDBD",
      description: "The same stream now trains two learners. SGD keeps one rate; IDBD adapts a separate rate for every feature.",
      next: "Look inside IDBD"
    },
    {
      title: "Look inside IDBD",
      description: "The rates, sensitivity trace, and coefficients reveal how IDBD decides what to retain and what to ignore.",
      next: "Extend the optimizers"
    },
    {
      title: "Extend the optimizer",
      description: "Momentum and weight decay now apply to both learners, while IDBD exposes how its meta-gradient trace follows them.",
      next: "Start over"
    }
  ];

  let initialized = false;

  function element(tagName, className, html) {
    const node = document.createElement(tagName);
    if (className) node.className = className;
    if (html !== undefined) node.innerHTML = html;
    return node;
  }

  function addOptimizerControls(clone) {
    const sharedGrafts = element("div", "optimizer-graft-controls shared-optimizer-controls");
    sharedGrafts.innerHTML = [
      '<p class="graft-controls-title">Shared optimizer additions · SGD + IDBD</p>',
      '<div class="control">',
      '  <div class="control-heading"><label for="staged-momentum">Momentum <span class="math">μ</span></label><output id="staged-momentum-value" for="staged-momentum">0.00</output></div>',
      '  <input id="staged-momentum" type="range" min="0" max="0.99" step="0.01" value="0" aria-describedby="staged-momentum-help">',
      '  <div class="range-labels" aria-hidden="true"><span>off</span><span>0.99</span></div>',
      '  <p id="staged-momentum-help">The same EMA momentum coefficient and update direction for both learners.</p>',
      '</div>',
      '<div class="control">',
      '  <div class="control-heading"><label for="staged-weight-decay">Weight decay <span class="math">λ</span></label><output id="staged-weight-decay-value" for="staged-weight-decay">off</output></div>',
      '  <input id="staged-weight-decay" type="range" min="0" max="7" step="1" value="0" aria-describedby="staged-weight-decay-help">',
      '  <div class="range-labels" aria-hidden="true"><span>off</span><span>1.0</span></div>',
      '  <p id="staged-weight-decay-help">The same decay strength for both; each learner scales it by its own learning rate.</p>',
      '</div>'
    ].join("");
    const sharedControls = clone.querySelector(".shared-controls");
    sharedControls.insertBefore(sharedGrafts, sharedControls.querySelector(".shared-control-grid"));

    const traceControls = element("div", "idbd-trace-controls");
    traceControls.innerHTML = [
      '<p class="graft-controls-title">IDBD trace treatment</p>',
      '<div class="control">',
      '  <label class="mode-label" for="staged-momentum-mode">With momentum</label>',
      '  <select id="staged-momentum-mode"><option value="derived" selected>Derived trace</option><option value="naive">Naïve trace</option></select>',
      '  <p>Derived mode carries p = dm/dβ.</p>',
      '</div>',
      '<div class="control">',
      '  <label class="mode-label" for="staged-weight-decay-mode">Decay mechanism</label>',
      '  <select id="staged-weight-decay-mode"><option value="traced" selected>Traced α-coupled</option><option value="alpha_coupled">α-coupled</option><option value="fixed">Fixed-rate</option></select>',
      '  <p>Traced mode includes decay’s β-derivative in h.</p>',
      '</div>'
    ].join("");
    clone.querySelector(".idbd-method-controls").appendChild(traceControls);
  }

  function addWalkthroughPanel(clone) {
    const panel = element("section", "staged-walkthrough-panel");
    panel.setAttribute("aria-labelledby", "staged-walkthrough-title");
    panel.innerHTML = [
      '<div class="staged-stage-heading">',
      '  <p id="staged-stage-count" class="section-label">Stage 1 of 5</p>',
      '  <h3 id="staged-walkthrough-title">Watch SGD learn</h3>',
      '  <p id="staged-stage-description"></p>',
      '</div>',
      '<div class="staged-live-row">',
      '  <figure class="staged-stream-chart">',
      '    <figcaption><span>SGD on the stream</span><small>faint noisy target · dashed clean signal · solid prediction</small></figcaption>',
      '    <canvas id="staged-walkthrough-chart" role="img" aria-label="SGD prediction, clean signal, and noisy target on a shared sample stream"></canvas>',
      '  </figure>',
      '  <aside class="staged-live-status">',
      '    <span>Recent prediction MSE</span>',
      '    <strong id="staged-walkthrough-loss">—</strong>',
      '    <p id="staged-walkthrough-status" aria-live="polite">Preparing…</p>',
      '    <div>',
      '      <button id="staged-walkthrough-pause" type="button">Pause</button>',
      '      <button id="staged-walkthrough-play" type="button">Play</button>',
      '    </div>',
      '  </aside>',
      '</div>',
      '<div class="staged-stage-two">',
      '  <div class="staged-proxy-controls">',
      '    <div class="control">',
      '      <div class="control-heading"><label for="staged-proxy-rate">Learning rate</label><output id="staged-proxy-rate-value" for="staged-proxy-rate">0.0100</output></div>',
      '      <input id="staged-proxy-rate" type="range" min="-3" max="1" step="0.05" value="-2">',
      '      <div class="range-labels" aria-hidden="true"><span>0.001</span><span>10.0</span></div>',
      '      <p>How far SGD moves after each gradient estimate.</p>',
      '    </div>',
      '    <div class="control">',
      '      <div class="control-heading"><label for="staged-proxy-steps">Total steps</label><output id="staged-proxy-steps-value" for="staged-proxy-steps">100,000</output></div>',
      '      <input id="staged-proxy-steps" type="range" min="0" max="18" step="1" value="9">',
      '      <div class="range-labels" aria-hidden="true"><span>256</span><span>100,000,000</span></div>',
      '      <p>How much of the stream the learner experiences.</p>',
      '    </div>',
      '  </div>',
      '  <figure class="chart-block staged-walkthrough-loss">',
      '    <figcaption><span>SGD prediction MSE</span><small>fixed 0.1–100 · logarithmic scale</small></figcaption>',
      '    <canvas id="staged-walkthrough-loss-chart" role="img" aria-label="SGD prediction mean squared error over training examples"></canvas>',
      '  </figure>',
      '</div>'
    ].join("");
    clone.querySelector(".experiment-heading").after(panel);
  }

  function addNavigation(clone) {
    const navigation = element("nav", "staged-navigation");
    navigation.setAttribute("aria-label", "Walkthrough stages");
    navigation.innerHTML = [
      '<button id="staged-back" type="button">Back</button>',
      '<div><span id="staged-nav-stage">Stage 1 of 5</span><strong id="staged-nav-next">Next: control the learner</strong></div>',
      '<button id="staged-continue" class="staged-continue" type="button" disabled>Continue</button>',
      '<button id="staged-show-all" class="staged-show-all" type="button">Show full experiment</button>'
    ].join("");
    clone.appendChild(navigation);
  }

  function annotateFigures(clone) {
    [
      "sgd-loss-chart", "idbd-loss-chart",
      "sgd-signal-loss-chart", "idbd-signal-loss-chart"
    ].forEach(function (id) {
      clone.querySelector("#staged-" + id).closest("figure").classList.add("staged-stage-three-chart");
    });
    [
      "sgd-prediction-chart", "idbd-prediction-chart",
      "sgd-rates-chart", "idbd-rates-chart",
      "sgd-weights-chart", "idbd-weights-chart"
    ].forEach(function (id) {
      clone.querySelector("#staged-" + id).closest("figure").classList.add("staged-stage-four-chart");
    });
  }

  function resetInputs(clone) {
    const values = {
      "staged-sgd-rate": "-2",
      "staged-idbd-rate": "-2",
      "staged-theta": "-1",
      "staged-batch-size": "0",
      "staged-training-steps": "9",
      "staged-stream-seed": "20260713"
    };
    Object.keys(values).forEach(function (id) {
      clone.querySelector("#" + id).value = values[id];
    });
    clone.querySelector("#staged-lock-rates").checked = true;
  }

  function initialize() {
    if (initialized) return;
    initialized = true;

    const clone = source.cloneNode(true);
    clone.id = "staged-idbd-playground";
    clone.classList.add("staged-experiment", "stage-1");
    api.prefixCloneIds(clone, "staged");
    resetInputs(clone);
    addWalkthroughPanel(clone);
    addOptimizerControls(clone);
    annotateFigures(clone);
    addNavigation(clone);
    mount.replaceChildren(clone);

    const actual = {
      rate: clone.querySelector("#staged-sgd-rate"),
      rateOutput: clone.querySelector("#staged-sgd-rate-value"),
      steps: clone.querySelector("#staged-training-steps"),
      stepsOutput: clone.querySelector("#staged-training-steps-value"),
      status: clone.querySelector("#staged-run-status"),
      pause: clone.querySelector("#staged-pause-training"),
      play: clone.querySelector("#staged-play-training"),
      momentum: clone.querySelector("#staged-momentum"),
      decay: clone.querySelector("#staged-weight-decay")
    };
    const proxy = {
      rate: clone.querySelector("#staged-proxy-rate"),
      rateOutput: clone.querySelector("#staged-proxy-rate-value"),
      steps: clone.querySelector("#staged-proxy-steps"),
      stepsOutput: clone.querySelector("#staged-proxy-steps-value"),
      status: clone.querySelector("#staged-walkthrough-status"),
      pause: clone.querySelector("#staged-walkthrough-pause"),
      play: clone.querySelector("#staged-walkthrough-play")
    };
    const stageCount = clone.querySelector("#staged-stage-count");
    const title = clone.querySelector("#staged-walkthrough-title");
    const description = clone.querySelector("#staged-stage-description");
    const back = clone.querySelector("#staged-back");
    const continueButton = clone.querySelector("#staged-continue");
    const showAll = clone.querySelector("#staged-show-all");
    const navStage = clone.querySelector("#staged-nav-stage");
    const navNext = clone.querySelector("#staged-nav-next");
    const sgdPill = clone.querySelector(".sgd-card .method-pill");
    const idbdPill = clone.querySelector(".idbd-card .method-pill");
    const scope = clone.querySelector(".simulation-scope");
    const walkthroughChart = clone.querySelector("#staged-walkthrough-chart");
    const walkthroughCaption = clone.querySelector(".staged-stream-chart figcaption small");
    const walkthroughScoreLabel = clone.querySelector(".staged-live-status > span");
    const batchControl = clone.querySelector("#staged-batch-size");
    const batchNote = clone.querySelector("#staged-batch-note");
    let currentStage = 1;
    let readinessTimer = null;

    function syncProxyOutputs() {
      proxy.rate.value = actual.rate.value;
      proxy.steps.value = actual.steps.value;
      proxy.rateOutput.value = actual.rateOutput.value;
      proxy.stepsOutput.value = actual.stepsOutput.value;
    }

    function markReadySoon() {
      window.clearTimeout(readinessTimer);
      continueButton.disabled = true;
      readinessTimer = window.setTimeout(function () {
        continueButton.disabled = false;
      }, 700);
    }

    function syncStatus() {
      const text = actual.status.textContent;
      proxy.status.textContent = text;
      proxy.pause.disabled = actual.pause.disabled;
      proxy.play.disabled = actual.play.disabled;
      if (text.startsWith("Complete") || (text.startsWith("Training") && !text.includes("0 /"))) {
        continueButton.disabled = false;
      }
    }

    function restartSameStream() {
      actual.steps.dispatchEvent(new Event("input", { bubbles: true }));
    }

    function clearHiddenExtensions() {
      let changed = false;
      if (actual.momentum.value !== "0") {
        actual.momentum.value = "0";
        changed = true;
      }
      if (actual.decay.value !== "0") {
        actual.decay.value = "0";
        changed = true;
      }
      if (changed) actual.momentum.dispatchEvent(new Event("input", { bubbles: true }));
    }

    function syncBatchNote() {
      const batchSize = Math.pow(2, Number(batchControl.value));
      if (currentStage < 5) {
        batchNote.innerHTML = batchSize === 1
          ? "<strong>At batch size 1</strong>, this is classic online IDBD. Larger batches use a mean-gradient diagonal approximation."
          : "<strong>At batch size " + String(batchSize) + "</strong>, IDBD uses a mean-gradient diagonal approximation; the batch-size-one setting is the original online rule.";
        return;
      }
      batchNote.innerHTML = batchSize === 1
        ? "<strong>At batch size 1</strong>, the gradient stream is online. Momentum or decay changes both learners’ updates and IDBD’s trace."
        : "<strong>At batch size " + String(batchSize) + "</strong>, both learners use the shared optimizer settings on the same mean-gradient batch.";
    }

    function setStage(stage, options) {
      const previousStage = currentStage;
      currentStage = Math.max(1, Math.min(stages.length, stage));
      const details = stages[currentStage - 1];
      clone.classList.remove("stage-1", "stage-2", "stage-3", "stage-4", "stage-5");
      clone.classList.add("stage-" + String(currentStage));
      stageCount.textContent = "Stage " + String(currentStage) + " of " + String(stages.length);
      title.textContent = details.title;
      description.textContent = details.description;
      navStage.textContent = stageCount.textContent;
      navNext.textContent = currentStage === stages.length ? "The complete staged experiment" : "Next: " + details.next.toLowerCase();
      continueButton.textContent = currentStage === stages.length ? "Start over" : "Continue";
      back.disabled = currentStage === 1;
      showAll.hidden = currentStage === stages.length;

      const extended = currentStage === stages.length;
      walkthroughChart.dataset.showIdbd = currentStage >= 3 ? "true" : "false";
      walkthroughCaption.textContent = currentStage >= 3
        ? "faint target · dashed signal · solid grey SGD · solid blue IDBD"
        : "faint noisy target · dashed clean signal · solid prediction";
      walkthroughScoreLabel.textContent = currentStage >= 3 ? "SGD recent prediction MSE" : "Recent prediction MSE";
      sgdPill.textContent = extended ? "fixed rate + shared grafts" : "one fixed rate";
      idbdPill.textContent = extended ? "one rate per feature + grafts" : "one rate per feature";
      scope.innerHTML = extended
        ? "<strong>Independent live simulation.</strong> Momentum and weight decay are shared by both learners; the trace mechanism choices apply only to IDBD."
        : "<strong>Computed live in this browser.</strong> Both learners receive the same deterministic stream; changing a setting cancels and restarts the current run.";

      if (currentStage < 5) clearHiddenExtensions();
      syncBatchNote();
      if (currentStage === 3 && previousStage < 3 && !(options && options.noRestart)) {
        restartSameStream();
      }
      syncProxyOutputs();
      markReadySoon();
      window.requestAnimationFrame(function () {
        window.dispatchEvent(new Event("resize"));
        const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        clone.querySelector(".staged-walkthrough-panel").scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
      });
    }

    proxy.rate.addEventListener("input", function () {
      actual.rate.value = proxy.rate.value;
      actual.rate.dispatchEvent(new Event("input", { bubbles: true }));
      syncProxyOutputs();
      markReadySoon();
    });
    proxy.steps.addEventListener("input", function () {
      actual.steps.value = proxy.steps.value;
      actual.steps.dispatchEvent(new Event("input", { bubbles: true }));
      syncProxyOutputs();
      markReadySoon();
    });
    proxy.pause.addEventListener("click", function () { actual.pause.click(); });
    proxy.play.addEventListener("click", function () { actual.play.click(); });
    continueButton.addEventListener("click", function () {
      setStage(currentStage === stages.length ? 1 : currentStage + 1);
    });
    back.addEventListener("click", function () { setStage(currentStage - 1); });
    showAll.addEventListener("click", function () { setStage(stages.length); });

    new MutationObserver(syncStatus).observe(actual.status, { childList: true, subtree: true, characterData: true });
    actual.rate.addEventListener("input", syncProxyOutputs);
    actual.steps.addEventListener("input", syncProxyOutputs);
    batchControl.addEventListener("input", syncBatchNote);

    api.createExperiment("staged", true);
    syncProxyOutputs();
    syncStatus();
    setStage(1, { noRestart: true });
  }

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(function (entries) {
      if (!entries.some(function (entry) { return entry.isIntersecting; })) return;
      observer.disconnect();
      initialize();
    }, { rootMargin: "500px 0px" });
    observer.observe(mount);
  } else {
    initialize();
  }
}());
