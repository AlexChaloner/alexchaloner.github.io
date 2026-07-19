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
    ["Inspect every rate", "The predictive feature is green; the 63 irrelevant features occupy the same positions in both lanes.", "Inspect the trace", "rates"],
    ["Inspect the sensitivity trace", "IDBD’s signed h trace is the information SGD does not maintain.", "Inspect the model", "trace"],
    ["Inspect the model", "The predictive coefficient should approach one while irrelevant coefficients stay near zero.", "Set the batch size", "model"],
    ["Set the batch size", "Choose how many examples contribute to each parameter update.", "Change the stream", "model"],
    ["Change the stream", "Reseed the noise while keeping both learners on the same experience.", "Add momentum", "stream"],
    ["Add momentum", "Give both learners the same moving average of recent gradients.", "Add weight decay", "loss"],
    ["Add weight decay", "Apply the same shrinkage to both learners and compare what remains.", "Choose the momentum trace", "clean"],
    ["Choose the momentum trace", "Decide whether IDBD differentiates through momentum or uses the naïve trace.", "Choose the decay trace", "trace"],
    ["Choose the decay trace", "Decide how weight decay participates in IDBD’s meta-gradient.", "Start over", "trace"]
  ];
  const viewUnlocks = { stream: 1, loss: 3, clean: 7, rates: 8, trace: 9, model: 10 };
  const tabUnlocks = { stream: 3, loss: 3, clean: 7, rates: 8, trace: 9, model: 10 };
  const unlockTargets = {
    2: ["control", "sgd-rate", "Add learning rate"],
    3: ["view", "loss", "Add loss view"],
    4: ["lane", "idbd", "Add IDBD"],
    5: ["control", "idbd-rate", "Add IDBD learning rate"],
    6: ["control", "theta", "Add adaptation"],
    7: ["view", "clean", "Add clean loss"],
    8: ["view", "rates", "Add rate distribution"],
    9: ["view", "trace", "Add h trace"],
    10: ["view", "model", "Add model view"],
    11: ["control", "batch", "Add batch size"],
    12: ["control", "stream", "Add stream controls"],
    13: ["control", "momentum", "Add momentum"],
    14: ["control", "decay", "Add weight decay"],
    15: ["control", "momentum-mode", "Add momentum trace"],
    16: ["control", "decay-mode", "Add decay trace"]
  };
  const gateHints = {
    1: "Adjust training steps",
    2: "Adjust the SGD learning rate",
    3: "Open the new Loss view",
    4: "Watch IDBD begin on the shared stream",
    5: "Adjust the IDBD starting rate",
    6: "Adjust the meta learning rate",
    7: "Open the new Clean loss view",
    8: "Open the new Rates view",
    9: "Open the new h trace view",
    10: "Open the new Model view",
    11: "Adjust the batch size",
    12: "Generate a new noise stream",
    13: "Adjust momentum",
    14: "Adjust weight decay",
    15: "Change the momentum trace",
    16: "Change the decay trace"
  };
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
      '  <div><p id="staged-stage-count" class="section-label">Stage 1 of 16</p><h3 id="staged-stage-title">Watch SGD learn</h3><p id="staged-stage-description"></p></div>',
      '  <div class="staged-transport"><span id="staged-workbench-status" aria-live="polite">Preparing…</span><button id="staged-workbench-pause" type="button">Pause</button><button id="staged-workbench-play" type="button">Play</button></div>',
      '</header>',
      '<nav class="staged-view-tabs" aria-label="Visible measurement">',
      '  <button type="button" data-view="stream">Stream</button><button type="button" data-view="loss">Loss</button><button type="button" data-view="clean">Clean loss</button><button type="button" data-view="rates">Rates</button><button type="button" data-view="trace">h trace</button><button type="button" data-view="model">Model</button>',
      '</nav>',
      '<div class="staged-learner-grid">',
      '  <section class="staged-learner staged-sgd-lane" aria-label="SGD lane"><div class="staged-lane-heading"></div><div class="staged-lane-score"></div><div class="staged-viewport"></div></section>',
      '  <section class="staged-learner staged-idbd-lane" aria-label="IDBD lane"><div class="staged-lane-heading"></div><div class="staged-lane-score"></div><div class="staged-viewport"></div><div class="staged-lane-curtain"><button class="staged-unlock" type="button" data-next-stage="4">+ Add IDBD</button></div></section>',
      '</div>',
      '<section class="staged-control-dock" aria-label="Experiment controls">',
      '  <div class="staged-control-space">',
      '    <div class="staged-control-slot" data-slot="steps" data-owner="shared" data-unlock="1"></div>',
      '    <div class="staged-control-slot" data-slot="sgd-rate" data-owner="sgd" data-unlock="2"></div>',
      '    <div class="staged-control-slot" data-slot="idbd-rate" data-owner="idbd" data-unlock="5"></div>',
      '    <div class="staged-control-slot" data-slot="theta" data-owner="idbd" data-unlock="6"></div>',
      '    <div class="staged-control-slot" data-slot="batch" data-owner="shared" data-unlock="11"></div>',
      '    <div class="staged-control-slot staged-stream-slot" data-slot="stream" data-owner="shared" data-unlock="12"></div>',
      '    <div class="staged-control-slot" data-slot="momentum" data-owner="shared" data-unlock="13"></div>',
      '    <div class="staged-control-slot" data-slot="decay" data-owner="shared" data-unlock="14"></div>',
      '    <div class="staged-control-slot" data-slot="momentum-mode" data-owner="idbd" data-unlock="15"></div>',
      '    <div class="staged-control-slot" data-slot="decay-mode" data-owner="idbd" data-unlock="16"></div>',
      '    <div class="staged-lock-slot" data-owner="bridge" data-unlock="5"></div>',
      '  </div>',
      '</section>',
      '<nav class="staged-navigation" aria-label="Walkthrough stages"><button id="staged-back" type="button">Back</button><div><span id="staged-nav-stage">Stage 1 of 16</span><strong id="staged-nav-next">Adjust training steps (0/5)</strong></div><button id="staged-show-all" class="staged-show-all" type="button">Open full lab</button></nav>'
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
    sgdViewport.appendChild(emptyTracePanel());
    moveFigure("idbd-trace-chart", "trace", idbdViewport);
    moveFigure("sgd-weights-chart", "model", sgdViewport);
    moveFigure("idbd-weights-chart", "model", idbdViewport);

    const controls = workbench.querySelector(".staged-control-space");
    controls.querySelector('[data-slot="sgd-rate"]').appendChild(clone.querySelector("#staged-sgd-rate").closest(".control"));
    controls.querySelector('[data-slot="idbd-rate"]').appendChild(clone.querySelector("#staged-idbd-rate").closest(".control"));
    controls.querySelector('[data-slot="theta"]').appendChild(clone.querySelector("#staged-theta").closest(".control"));
    controls.querySelector('[data-slot="steps"]').appendChild(clone.querySelector("#staged-training-steps").closest(".control"));
    controls.querySelector('[data-slot="batch"]').appendChild(clone.querySelector("#staged-batch-size").closest(".control"));
    controls.querySelector('[data-slot="stream"]').appendChild(clone.querySelector(".stream-control"));
    controls.querySelector('[data-slot="momentum"]').appendChild(clone.querySelector("#staged-momentum").closest(".control"));
    controls.querySelector('[data-slot="decay"]').appendChild(clone.querySelector("#staged-weight-decay").closest(".control"));
    controls.querySelector('[data-slot="momentum-mode"]').appendChild(clone.querySelector("#staged-momentum-mode").closest(".control"));
    controls.querySelector('[data-slot="decay-mode"]').appendChild(clone.querySelector("#staged-weight-decay-mode").closest(".control"));
    controls.querySelector(".staged-lock-slot").appendChild(clone.querySelector(".rate-lock-control"));
    Object.keys(unlockTargets).forEach(function (stage) {
      const target = unlockTargets[stage];
      if (target[0] !== "control") return;
      const button = make("button", "staged-unlock", "+ " + target[2]);
      button.type = "button";
      button.dataset.nextStage = stage;
      controls.querySelector('[data-slot="' + target[1] + '"]').appendChild(button);
    });
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
      steps: clone.querySelector("#staged-training-steps"), sgdRate: clone.querySelector("#staged-sgd-rate"), idbdRate: clone.querySelector("#staged-idbd-rate"),
      theta: clone.querySelector("#staged-theta"), batch: clone.querySelector("#staged-batch-size"), newStream: clone.querySelector("#staged-new-stream"),
      momentum: clone.querySelector("#staged-momentum"), decay: clone.querySelector("#staged-weight-decay"),
      momentumMode: clone.querySelector("#staged-momentum-mode"), decayMode: clone.querySelector("#staged-weight-decay-mode")
    };
    const status = workbench.querySelector("#staged-workbench-status");
    const pause = workbench.querySelector("#staged-workbench-pause");
    const play = workbench.querySelector("#staged-workbench-play");
    const back = workbench.querySelector("#staged-back");
    const showAll = workbench.querySelector("#staged-show-all");
    const stageCount = workbench.querySelector("#staged-stage-count");
    const title = workbench.querySelector("#staged-stage-title");
    const description = workbench.querySelector("#staged-stage-description");
    const navStage = workbench.querySelector("#staged-nav-stage");
    const navNext = workbench.querySelector("#staged-nav-next");
    const viewButtons = Array.from(workbench.querySelectorAll(".staged-view-tabs button"));
    viewButtons.forEach(function (button) { button.dataset.label = button.textContent; });
    Object.keys(unlockTargets).forEach(function (stage) {
      const target = unlockTargets[stage];
      if (target[0] !== "view") return;
      workbench.querySelector('.staged-view-tabs [data-view="' + target[1] + '"]').dataset.nextStage = stage;
    });
    const unlockButtons = Array.from(workbench.querySelectorAll(".staged-unlock"));
    const panels = Array.from(workbench.querySelectorAll(".staged-panel"));
    let currentStage = 1;
    let currentView = "stream";
    let suppressProgress = false;
    const progress = {};

    function setView(view, userInitiated) {
      if (viewUnlocks[view] > currentStage) return;
      currentView = view;
      viewButtons.forEach(function (button) { button.classList.toggle("is-active", button.dataset.view === view); });
      panels.forEach(function (panel) { panel.classList.toggle("is-active", panel.dataset.view === view); });
      window.dispatchEvent(new Event("resize"));
      if (userInitiated) {
        const viewStage = { loss: 3, clean: 7, rates: 8, trace: 9, model: 10 }[view];
        if (viewStage === currentStage) markProgress(currentStage, 1);
      }
    }

    function syncStatus() {
      status.textContent = actual.status.textContent;
      pause.disabled = actual.pause.disabled;
      play.disabled = actual.play.disabled;
      const text = actual.status.textContent;
      if (currentStage === 4 && (text.startsWith("Complete") || (text.startsWith("Training") && !text.includes("0 /")))) {
        markProgress(4, 1);
      }
    }

    function requiredProgress(stage) {
      return [1, 2, 5, 6, 11, 13, 14].includes(stage) ? 5 : 1;
    }

    function markProgress(stage, amount) {
      if (currentStage !== stage) return;
      progress[stage] = Math.min(requiredProgress(stage), (progress[stage] || 0) + amount);
      updateOffer();
    }

    function clearOffers() {
      workbench.querySelectorAll(".is-offering").forEach(function (node) { node.classList.remove("is-offering"); });
      viewButtons.forEach(function (button) { button.textContent = button.dataset.label; });
    }

    function updateOffer() {
      clearOffers();
      if (currentStage === stages.length) {
        navNext.textContent = (progress[currentStage] || 0) >= requiredProgress(currentStage) ? "Walkthrough complete" : gateHints[currentStage];
        return;
      }
      if ((progress[currentStage] || 0) < requiredProgress(currentStage)) {
        const required = requiredProgress(currentStage);
        navNext.textContent = gateHints[currentStage] + (required > 1 ? " (" + String(progress[currentStage] || 0) + "/" + String(required) + ")" : "");
        return;
      }
      const nextStage = currentStage + 1;
      const target = unlockTargets[nextStage];
      navNext.textContent = "Ready: " + target[2].toLowerCase();
      if (target[0] === "control") {
        workbench.querySelector('[data-slot="' + target[1] + '"]').classList.add("is-offering");
      } else if (target[0] === "view") {
        const button = workbench.querySelector('.staged-view-tabs [data-view="' + target[1] + '"]');
        button.disabled = false;
        button.textContent = "+ " + target[2];
        button.classList.add("is-offering");
      } else {
        workbench.querySelector(".staged-lane-curtain").classList.add("is-offering");
      }
    }

    function clearHiddenExtensions() {
      let changed = false;
      if (currentStage < 13 && actual.momentum.value !== "0") { actual.momentum.value = "0"; changed = true; }
      if (currentStage < 14 && actual.decay.value !== "0") { actual.decay.value = "0"; changed = true; }
      if (currentStage < 15 && actual.momentumMode.value !== "derived") { actual.momentumMode.value = "derived"; changed = true; }
      if (currentStage < 16 && actual.decayMode.value !== "traced") { actual.decayMode.value = "traced"; changed = true; }
      if (changed) {
        suppressProgress = true;
        actual.momentum.dispatchEvent(new Event("input", { bubbles: true }));
        suppressProgress = false;
      }
    }

    function setStage(stage, skipReplay, preserveView) {
      const previous = currentStage;
      currentStage = Math.max(1, Math.min(stages.length, stage));
      const details = stages[currentStage - 1];
      clearOffers();
      workbench.dataset.stage = String(currentStage);
      stageCount.textContent = "Stage " + String(currentStage) + " of " + String(stages.length);
      navStage.textContent = stageCount.textContent;
      title.textContent = details[0];
      description.textContent = details[1];
      back.disabled = currentStage === 1;
      showAll.textContent = currentStage === stages.length ? "Start over" : "Open full lab";
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
      clone.querySelectorAll(".staged-lane-score dl").forEach(function (detailsList) {
        detailsList.classList.toggle("is-unlocked", currentStage >= 7);
      });
      clearHiddenExtensions();
      setView(preserveView && viewUnlocks[currentView] <= currentStage ? currentView : details[3], false);
      if (currentStage === 4 && previous < 4 && !skipReplay) actual.steps.dispatchEvent(new Event("input", { bubbles: true }));
      updateOffer();
    }

    viewButtons.forEach(function (button) {
      button.addEventListener("click", function () {
        if (button.classList.contains("is-offering")) {
          setStage(Number(button.dataset.nextStage), false, true);
          return;
        }
        setView(button.dataset.view, true);
      });
    });
    unlockButtons.forEach(function (button) {
      button.addEventListener("click", function () { setStage(Number(button.dataset.nextStage), false, true); });
    });
    [[actual.steps, 1], [actual.sgdRate, 2], [actual.idbdRate, 5], [actual.theta, 6], [actual.batch, 11], [actual.momentum, 13], [actual.decay, 14]].forEach(function (entry) {
      entry[0].addEventListener("input", function () { if (!suppressProgress) markProgress(entry[1], 1); });
    });
    actual.newStream.addEventListener("click", function () { markProgress(12, 1); });
    actual.momentumMode.addEventListener("change", function () { markProgress(15, 1); });
    actual.decayMode.addEventListener("change", function () { markProgress(16, 1); });
    pause.addEventListener("click", function () { actual.pause.click(); });
    play.addEventListener("click", function () { actual.play.click(); });
    back.addEventListener("click", function () { setStage(currentStage - 1); });
    showAll.addEventListener("click", function () {
      if (currentStage === stages.length) {
        Object.keys(progress).forEach(function (stage) { delete progress[stage]; });
        setStage(1, true);
      } else setStage(stages.length, false, true);
    });
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
