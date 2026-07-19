(function () {
  "use strict";

  const api = window.IDBDDemo;
  const mount = document.getElementById("staged-experiment-mount");
  const source = document.getElementById("idbd-playground");
  if (!api || !mount || !source) return;

  const stages = [
    ["Watch SGD learn", "One learner follows a recurring signal through noise.", "Set its learning rate", "stream"],
    ["Set the learning rate", "Choose how far SGD moves after each gradient estimate.", "See the loss curve", "stream"],
    ["See the loss curve", "Switch from the latest prediction to SGD’s progress over the full run.", "Add IDBD", "loss"],
    ["Add IDBD", "A second learner enters the same fixed spatial lane and replays the identical stream.", "Set IDBD’s initial rate", "stream"],
    ["Set a fair starting point", "Give IDBD an initial rate, or lock both learners to the same starting value.", "Enable adaptation", "stream"],
    ["Enable adaptation", "Theta controls how quickly IDBD changes each feature’s learning rate.", "Judge the clean signal", "loss"],
    ["Judge the clean signal", "Remove observation noise from the score and compare what each learner retained.", "Inspect every rate", "clean"],
    ["Inspect every rate", "The predictive feature is green; the 63 irrelevant features occupy the same positions in both lanes.", "Watch the rates separate", "rates"],
    ["Watch the rates separate", "Compare the predictive-feature rate with the typical irrelevant-feature rate over time.", "Inspect the trace", "history"],
    ["Inspect the sensitivity trace", "IDBD’s signed h trace is the information SGD does not maintain.", "Inspect the model", "trace"],
    ["Inspect the model", "The predictive coefficient should approach one while irrelevant coefficients stay near zero.", "Open the optimizer lab", "model"],
    ["Open the optimizer lab", "Use the same fixed dock for run settings, momentum, decay, and IDBD trace variants.", "Start over", "model"]
  ];
  const viewUnlocks = { stream: 1, loss: 3, clean: 7, rates: 8, history: 9, trace: 10, model: 11 };
  const tabUnlocks = { stream: 3, loss: 3, clean: 7, rates: 8, history: 9, trace: 10, model: 11 };
  let initialized = false;

  function make(tagName, className, html) {
    const node = document.createElement(tagName);
    if (className) node.className = className;
    if (html !== undefined) node.innerHTML = html;
    return node;
  }

  function addOptimizerControls(clone) {
    const shared = make("div", "optimizer-graft-controls shared-optimizer-controls");
    shared.innerHTML = [
      '<p class="graft-controls-title">Shared optimizer additions · SGD + IDBD</p>',
      '<div class="control">',
      '  <div class="control-heading"><label for="staged-momentum">Momentum <span class="math">μ</span></label><output id="staged-momentum-value" for="staged-momentum">0.00</output></div>',
      '  <input id="staged-momentum" type="range" min="0" max="0.99" step="0.01" value="0" aria-describedby="staged-momentum-help">',
      '  <div class="range-labels" aria-hidden="true"><span>off</span><span>0.99</span></div>',
      '  <p id="staged-momentum-help">The same EMA momentum coefficient for both learners.</p>',
      '</div>',
      '<div class="control">',
      '  <div class="control-heading"><label for="staged-weight-decay">Weight decay <span class="math">λ</span></label><output id="staged-weight-decay-value" for="staged-weight-decay">off</output></div>',
      '  <input id="staged-weight-decay" type="range" min="0" max="7" step="1" value="0" aria-describedby="staged-weight-decay-help">',
      '  <div class="range-labels" aria-hidden="true"><span>off</span><span>1.0</span></div>',
      '  <p id="staged-weight-decay-help">The same decay strength for both learners.</p>',
      '</div>'
    ].join("");
    const sharedControls = clone.querySelector(".shared-controls");
    sharedControls.insertBefore(shared, sharedControls.querySelector(".shared-control-grid"));

    const trace = make("div", "idbd-trace-controls");
    trace.innerHTML = [
      '<p class="graft-controls-title">IDBD trace treatment</p>',
      '<div class="control">',
      '  <label class="mode-label" for="staged-momentum-mode">With momentum</label>',
      '  <select id="staged-momentum-mode"><option value="derived" selected>Derived trace</option><option value="naive">Naïve trace</option></select>',
      '  <p>Derived mode carries p = dm/dβ.</p>',
      '</div>',
      '<div class="control">',
      '  <label class="mode-label" for="staged-weight-decay-mode">With decay</label>',
      '  <select id="staged-weight-decay-mode"><option value="traced" selected>Traced α-coupled</option><option value="alpha_coupled">α-coupled</option><option value="fixed">Fixed-rate</option></select>',
      '  <p>Traced mode includes decay’s β-derivative.</p>',
      '</div>'
    ].join("");
    clone.querySelector(".idbd-method-controls").appendChild(trace);
  }

  function streamFigure(id, learner) {
    const figure = make("figure", "chart-block staged-panel");
    figure.dataset.view = "stream";
    figure.innerHTML = [
      '<figcaption><span>' + learner + ' on the stream</span><small>faint target · dashed clean signal · solid prediction</small></figcaption>',
      '<canvas id="' + id + '" role="img" aria-label="' + learner + ' prediction, clean signal, and noisy target on a shared sample stream"></canvas>'
    ].join("");
    return figure;
  }

  function historyFigure() {
    const figure = make("figure", "chart-block staged-panel");
    figure.dataset.view = "history";
    figure.innerHTML = [
      '<figcaption><span>Fixed learning rate</span><small>one rate for every feature · 10⁻⁵–10 log scale</small></figcaption>',
      '<canvas id="staged-sgd-rate-history-chart" role="img" aria-label="SGD fixed learning rate over training examples"></canvas>'
    ].join("");
    return figure;
  }

  function emptyTracePanel() {
    const panel = make("div", "staged-panel staged-empty-panel");
    panel.dataset.view = "trace";
    panel.innerHTML = '<div><span>SGD</span><strong>No meta-gradient trace</strong><p>A fixed learning rate does not require <span class="math">h = ∂w/∂β</span>.</p></div>';
    return panel;
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
    Object.keys(values).forEach(function (id) { clone.querySelector("#" + id).value = values[id]; });
    clone.querySelector("#staged-lock-rates").checked = true;
  }

  function buildWorkbench(clone) {
    const workbench = make("div", "staged-workbench");
    workbench.innerHTML = [
      '<header class="staged-workbench-heading">',
      '  <div><p id="staged-stage-count" class="section-label">Stage 1 of 12</p><h3 id="staged-stage-title">Watch SGD learn</h3><p id="staged-stage-description"></p></div>',
      '  <div class="staged-transport"><span id="staged-workbench-status" aria-live="polite">Preparing…</span><button id="staged-workbench-pause" type="button">Pause</button><button id="staged-workbench-play" type="button">Play</button></div>',
      '</header>',
      '<nav class="staged-view-tabs" aria-label="Visible measurement">',
      '  <button type="button" data-view="stream">Stream</button><button type="button" data-view="loss">Loss</button><button type="button" data-view="clean">Clean loss</button><button type="button" data-view="rates">Rates</button><button type="button" data-view="history">Rate history</button><button type="button" data-view="trace">h trace</button><button type="button" data-view="model">Model</button>',
      '</nav>',
      '<div class="staged-learner-grid">',
      '  <section class="staged-learner staged-sgd-lane" aria-label="SGD lane"><div class="staged-lane-heading"></div><div class="staged-lane-score"></div><div class="staged-viewport"></div></section>',
      '  <section class="staged-learner staged-idbd-lane" aria-label="IDBD lane"><div class="staged-lane-heading"></div><div class="staged-lane-score"></div><div class="staged-viewport"></div><div class="staged-lane-curtain" aria-hidden="true"></div></section>',
      '</div>',
      '<section class="staged-control-dock" aria-label="Experiment controls">',
      '  <nav class="staged-control-tabs" aria-label="Control groups"><button type="button" data-page="basic">Learning</button><button type="button" data-page="run">Run settings</button><button type="button" data-page="extensions">Extensions</button></nav>',
      '  <div class="staged-control-page staged-basic-page" data-page="basic"><div class="staged-control-slot" data-slot="sgd-rate" data-unlock="2"></div><div class="staged-control-slot" data-slot="idbd-rate" data-unlock="5"></div><div class="staged-control-slot" data-slot="theta" data-unlock="6"></div><div class="staged-control-slot" data-slot="steps" data-unlock="1"></div><div class="staged-lock-slot" data-unlock="5"></div></div>',
      '  <div class="staged-control-page staged-run-page" data-page="run"><div data-slot="batch"></div><div data-slot="stream"></div></div>',
      '  <div class="staged-control-page staged-extension-page" data-page="extensions"><div data-slot="momentum"></div><div data-slot="decay"></div><div data-slot="momentum-mode"></div><div data-slot="decay-mode"></div></div>',
      '</section>',
      '<nav class="staged-navigation" aria-label="Walkthrough stages"><button id="staged-back" type="button">Back</button><div><span id="staged-nav-stage">Stage 1 of 12</span><strong id="staged-nav-next">Next: set its learning rate</strong></div><button id="staged-continue" class="staged-continue" type="button" disabled>Continue</button><button id="staged-show-all" class="staged-show-all" type="button">Open full lab</button></nav>'
    ].join("");
    clone.querySelector(".experiment-heading").after(workbench);

    const sgdLane = workbench.querySelector(".staged-sgd-lane");
    const idbdLane = workbench.querySelector(".staged-idbd-lane");
    sgdLane.querySelector(".staged-lane-heading").appendChild(clone.querySelector(".sgd-card .method-heading"));
    sgdLane.querySelector(".staged-lane-score").appendChild(clone.querySelector(".sgd-card .score-row"));
    idbdLane.querySelector(".staged-lane-heading").appendChild(clone.querySelector(".idbd-card .method-heading"));
    idbdLane.querySelector(".staged-lane-score").appendChild(clone.querySelector(".idbd-card .score-row"));

    const sgdViewport = sgdLane.querySelector(".staged-viewport");
    const idbdViewport = idbdLane.querySelector(".staged-viewport");
    sgdViewport.appendChild(streamFigure("staged-walkthrough-sgd-chart", "SGD"));
    idbdViewport.appendChild(streamFigure("staged-walkthrough-idbd-chart", "IDBD"));

    function moveFigure(canvasId, view, viewport) {
      const figure = clone.querySelector("#staged-" + canvasId).closest("figure");
      figure.classList.add("staged-panel");
      figure.dataset.view = view;
      viewport.appendChild(figure);
    }
    moveFigure("sgd-loss-chart", "loss", sgdViewport);
    moveFigure("idbd-loss-chart", "loss", idbdViewport);
    moveFigure("sgd-signal-loss-chart", "clean", sgdViewport);
    moveFigure("idbd-signal-loss-chart", "clean", idbdViewport);
    moveFigure("sgd-rates-chart", "rates", sgdViewport);
    moveFigure("idbd-rates-chart", "rates", idbdViewport);
    sgdViewport.appendChild(historyFigure());
    moveFigure("idbd-rate-history-chart", "history", idbdViewport);
    sgdViewport.appendChild(emptyTracePanel());
    moveFigure("idbd-trace-chart", "trace", idbdViewport);
    moveFigure("sgd-weights-chart", "model", sgdViewport);
    moveFigure("idbd-weights-chart", "model", idbdViewport);

    const basic = workbench.querySelector(".staged-basic-page");
    basic.querySelector('[data-slot="sgd-rate"]').appendChild(clone.querySelector("#staged-sgd-rate").closest(".control"));
    basic.querySelector('[data-slot="idbd-rate"]').appendChild(clone.querySelector("#staged-idbd-rate").closest(".control"));
    basic.querySelector('[data-slot="theta"]').appendChild(clone.querySelector("#staged-theta").closest(".control"));
    basic.querySelector('[data-slot="steps"]').appendChild(clone.querySelector("#staged-training-steps").closest(".control"));
    basic.querySelector(".staged-lock-slot").appendChild(clone.querySelector(".rate-lock-control"));

    const run = workbench.querySelector(".staged-run-page");
    run.querySelector('[data-slot="batch"]').appendChild(clone.querySelector("#staged-batch-size").closest(".control"));
    run.querySelector('[data-slot="stream"]').appendChild(clone.querySelector(".stream-control"));

    const extension = workbench.querySelector(".staged-extension-page");
    extension.querySelector('[data-slot="momentum"]').appendChild(clone.querySelector("#staged-momentum").closest(".control"));
    extension.querySelector('[data-slot="decay"]').appendChild(clone.querySelector("#staged-weight-decay").closest(".control"));
    extension.querySelector('[data-slot="momentum-mode"]').appendChild(clone.querySelector("#staged-momentum-mode").closest(".control"));
    extension.querySelector('[data-slot="decay-mode"]').appendChild(clone.querySelector("#staged-weight-decay-mode").closest(".control"));
    return workbench;
  }

  function initialize() {
    if (initialized) return;
    initialized = true;
    const clone = source.cloneNode(true);
    clone.id = "staged-idbd-playground";
    clone.classList.add("staged-experiment");
    api.prefixCloneIds(clone, "staged");
    resetInputs(clone);
    addOptimizerControls(clone);
    const workbench = buildWorkbench(clone);
    mount.replaceChildren(clone);

    const actual = {
      status: clone.querySelector("#staged-run-status"), pause: clone.querySelector("#staged-pause-training"), play: clone.querySelector("#staged-play-training"),
      steps: clone.querySelector("#staged-training-steps"), momentum: clone.querySelector("#staged-momentum"), decay: clone.querySelector("#staged-weight-decay")
    };
    const status = workbench.querySelector("#staged-workbench-status");
    const pause = workbench.querySelector("#staged-workbench-pause");
    const play = workbench.querySelector("#staged-workbench-play");
    const back = workbench.querySelector("#staged-back");
    const next = workbench.querySelector("#staged-continue");
    const showAll = workbench.querySelector("#staged-show-all");
    const stageCount = workbench.querySelector("#staged-stage-count");
    const title = workbench.querySelector("#staged-stage-title");
    const description = workbench.querySelector("#staged-stage-description");
    const navStage = workbench.querySelector("#staged-nav-stage");
    const navNext = workbench.querySelector("#staged-nav-next");
    const viewButtons = Array.from(workbench.querySelectorAll(".staged-view-tabs button"));
    const controlButtons = Array.from(workbench.querySelectorAll(".staged-control-tabs button"));
    const panels = Array.from(workbench.querySelectorAll(".staged-panel"));
    const controlPages = Array.from(workbench.querySelectorAll(".staged-control-page"));
    let currentStage = 1;
    let currentView = "stream";
    let readyTimer = null;

    function setView(view) {
      if (viewUnlocks[view] > currentStage) return;
      currentView = view;
      viewButtons.forEach(function (button) { button.classList.toggle("is-active", button.dataset.view === view); });
      panels.forEach(function (panel) { panel.classList.toggle("is-active", panel.dataset.view === view); });
      window.dispatchEvent(new Event("resize"));
    }

    function setControlPage(page) {
      controlButtons.forEach(function (button) { button.classList.toggle("is-active", button.dataset.page === page); });
      controlPages.forEach(function (panel) { panel.classList.toggle("is-active", panel.dataset.page === page); });
      window.dispatchEvent(new Event("resize"));
    }

    function syncStatus() {
      status.textContent = actual.status.textContent;
      pause.disabled = actual.pause.disabled;
      play.disabled = actual.play.disabled;
      const text = actual.status.textContent;
      if (text.startsWith("Complete") || (text.startsWith("Training") && !text.includes("0 /"))) next.disabled = false;
    }

    function readySoon() {
      window.clearTimeout(readyTimer);
      next.disabled = true;
      readyTimer = window.setTimeout(function () { next.disabled = false; }, 550);
    }

    function clearExtensions() {
      let changed = false;
      if (actual.momentum.value !== "0") { actual.momentum.value = "0"; changed = true; }
      if (actual.decay.value !== "0") { actual.decay.value = "0"; changed = true; }
      if (changed) actual.momentum.dispatchEvent(new Event("input", { bubbles: true }));
    }

    function setStage(stage, skipReplay) {
      const previous = currentStage;
      currentStage = Math.max(1, Math.min(stages.length, stage));
      const details = stages[currentStage - 1];
      workbench.dataset.stage = String(currentStage);
      stageCount.textContent = "Stage " + String(currentStage) + " of " + String(stages.length);
      navStage.textContent = stageCount.textContent;
      title.textContent = details[0];
      description.textContent = details[1];
      navNext.textContent = currentStage === stages.length ? "The complete optimizer lab" : "Next: " + details[2].toLowerCase();
      next.textContent = currentStage === stages.length ? "Start over" : "Continue";
      back.disabled = currentStage === 1;
      showAll.hidden = currentStage === stages.length;
      workbench.querySelector(".staged-idbd-lane").classList.toggle("is-revealed", currentStage >= 4);
      workbench.querySelectorAll("[data-unlock]").forEach(function (node) {
        node.classList.toggle("is-unlocked", currentStage >= Number(node.dataset.unlock));
      });
      viewButtons.forEach(function (button) {
        const available = currentStage >= viewUnlocks[button.dataset.view];
        const visible = currentStage >= tabUnlocks[button.dataset.view];
        button.disabled = !available;
        button.classList.toggle("is-unlocked", visible);
      });
      controlButtons.forEach(function (button) {
        const available = button.dataset.page !== "extensions" || currentStage >= 12;
        button.disabled = !available;
        button.classList.add("is-unlocked");
      });
      clone.querySelectorAll(".staged-lane-score dl").forEach(function (detailsList) {
        detailsList.classList.toggle("is-unlocked", currentStage >= 7);
      });
      if (currentStage < 12) clearExtensions();
      setControlPage(currentStage === 12 ? "extensions" : "basic");
      setView(details[3]);
      if (currentStage === 4 && previous < 4 && !skipReplay) actual.steps.dispatchEvent(new Event("input", { bubbles: true }));
      readySoon();
    }

    viewButtons.forEach(function (button) { button.addEventListener("click", function () { setView(button.dataset.view); }); });
    controlButtons.forEach(function (button) { button.addEventListener("click", function () { setControlPage(button.dataset.page); }); });
    pause.addEventListener("click", function () { actual.pause.click(); });
    play.addEventListener("click", function () { actual.play.click(); });
    back.addEventListener("click", function () { setStage(currentStage - 1); });
    next.addEventListener("click", function () { setStage(currentStage === stages.length ? 1 : currentStage + 1); });
    showAll.addEventListener("click", function () { setStage(stages.length); });
    new MutationObserver(syncStatus).observe(actual.status, { childList: true, subtree: true, characterData: true });

    api.createExperiment("staged", true);
    syncStatus();
    setStage(1, true);
  }

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(function (entries) {
      if (!entries.some(function (entry) { return entry.isIntersecting; })) return;
      observer.disconnect();
      initialize();
    }, { rootMargin: "400px 0px" });
    observer.observe(mount);
  } else initialize();
}());
