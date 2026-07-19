(function () {
  "use strict";

  const api = window.IDBDDemo;
  const mount = document.getElementById("staged-experiment-mount");
  const source = document.getElementById("idbd-playground");
  if (!api || !mount || !source) return;

  const stages = [
    ["Watch SGD learn", "One learner follows a recurring signal through noise.", "Add a learning rate", "stream"],
    ["Tune SGD", "Change how far SGD moves and inspect the loss whenever you want.", "Add IDBD", "loss"],
    ["Compare starting rates", "Move either locked rate, or unlock them to give IDBD a different starting point.", "Enable adaptation", "stream"],
    ["Enable adaptation", "Theta controls how quickly IDBD changes each feature’s learning rate.", "Open the lab", "loss"],
    ["Choose the batch size", "The diagnostic views are available; now choose how many examples contribute to an update.", "Control the stream", "model"],
    ["Control the stream", "Pause, resume, or reseed the shared experience.", "Add momentum", "stream"],
    ["Add momentum", "Give both learners the same moving average of recent gradients.", "Extend the optimizer", "loss"],
    ["Extend the optimizer", "Try weight decay or change how IDBD carries its trace through momentum.", "Add the decay trace", "clean"],
    ["Choose the decay trace", "Decide how weight decay participates in IDBD’s meta-gradient.", "Start over", "trace"]
  ];
  const unlockBundles = {
    2: [
      ["control", "sgd-rate", "Add learning rate"],
      ["view", "loss", "Add loss view"]
    ],
    3: [
      ["lane", "idbd", "Add IDBD"],
      ["control", "idbd-rate", "Add IDBD learning rate"]
    ],
    4: [["control", "theta", "Add adaptation"]],
    5: [
      ["view", "clean", "Add clean loss"],
      ["view", "rates", "Add rate distribution"],
      ["view", "trace", "Add h trace"],
      ["view", "model", "Add model view"],
      ["control", "batch", "Add batch size"]
    ],
    6: [["control", "stream", "Add stream controls"]],
    7: [["control", "momentum", "Add momentum"]],
    8: [
      ["control", "momentum-mode", "Add momentum trace"],
      ["control", "decay", "Add weight decay"]
    ],
    9: [["control", "decay-mode", "Add decay trace"]]
  };
  const gateHints = {
    1: "Adjust training steps",
    2: "Adjust the SGD learning rate",
    3: "Adjust the starting rates",
    4: "Adjust the meta learning rate",
    5: "Adjust the batch size",
    6: "Use a stream control",
    7: "Adjust momentum",
    8: "Try the optimizer additions",
    9: "Change the decay trace"
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
      '  <div><p id="staged-stage-count" class="section-label">Stage 1 of 9</p><h3 id="staged-stage-title">Watch SGD learn</h3><p id="staged-stage-description"></p></div>',
      '  <div class="staged-transport"><span id="staged-workbench-status" aria-live="polite">Preparing…</span><button id="staged-workbench-pause" type="button">Pause</button><button id="staged-workbench-play" type="button">Play</button></div>',
      '</header>',
      '<nav class="staged-view-tabs" aria-label="Visible measurement">',
      '  <button type="button" data-view="stream">Stream</button><button type="button" data-view="loss">Loss</button><button type="button" data-view="clean">Clean loss</button><button type="button" data-view="rates">Rates</button><button type="button" data-view="trace">h trace</button><button type="button" data-view="model">Model</button>',
      '</nav>',
      '<div class="staged-learner-grid">',
      '  <section class="staged-learner staged-sgd-lane" aria-label="SGD lane"><div class="staged-lane-heading"></div><div class="staged-lane-score"></div><div class="staged-viewport"></div></section>',
      '  <section class="staged-learner staged-idbd-lane" aria-label="IDBD lane"><div class="staged-lane-heading"></div><div class="staged-lane-score"></div><div class="staged-viewport"></div><div class="staged-lane-curtain"><button class="staged-unlock" type="button" data-next-stage="3" data-target-kind="lane" data-target-name="idbd">+ Add IDBD</button></div></section>',
      '</div>',
      '<section class="staged-control-dock" aria-label="Experiment controls">',
      '  <div class="staged-control-space">',
      '    <div class="staged-control-slot" data-slot="steps" data-owner="shared" data-unlock="1"></div>',
      '    <div class="staged-control-slot" data-slot="sgd-rate" data-owner="sgd" data-unlock="2"></div>',
      '    <div class="staged-control-slot" data-slot="idbd-rate" data-owner="idbd" data-unlock="3"></div>',
      '    <div class="staged-control-slot" data-slot="theta" data-owner="idbd" data-unlock="4"></div>',
      '    <div class="staged-control-slot" data-slot="batch" data-owner="shared" data-unlock="5"></div>',
      '    <div class="staged-control-slot staged-stream-slot" data-slot="stream" data-owner="shared" data-unlock="6"></div>',
      '    <div class="staged-control-slot" data-slot="momentum" data-owner="shared" data-unlock="7"></div>',
      '    <div class="staged-control-slot" data-slot="decay" data-owner="shared" data-unlock="8"></div>',
      '    <div class="staged-control-slot" data-slot="momentum-mode" data-owner="idbd" data-unlock="8"></div>',
      '    <div class="staged-control-slot" data-slot="decay-mode" data-owner="idbd" data-unlock="9"></div>',
      '    <div class="staged-lock-slot" data-owner="bridge" data-unlock="3"></div>',
      '  </div>',
      '</section>',
      '<nav class="staged-navigation" aria-label="Walkthrough stages"><button id="staged-back" type="button">Back</button><div><span id="staged-nav-stage">Stage 1 of 9</span><strong id="staged-nav-next">Adjust training steps (0/5)</strong></div><button id="staged-show-all" class="staged-show-all" type="button">Open full lab</button></nav>'
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
    Object.keys(unlockBundles).forEach(function (stage) {
      unlockBundles[stage].forEach(function (target) {
        if (target[0] !== "control") return;
        const button = make("button", "staged-unlock", "+ " + target[2]);
        button.type = "button";
        button.dataset.nextStage = stage;
        button.dataset.targetKind = target[0];
        button.dataset.targetName = target[1];
        controls.querySelector('[data-slot="' + target[1] + '"]').appendChild(button);
      });
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
      lockRates: clone.querySelector("#staged-lock-rates"), theta: clone.querySelector("#staged-theta"), batch: clone.querySelector("#staged-batch-size"),
      newStream: clone.querySelector("#staged-new-stream"), reseed: clone.querySelector("#staged-reseed-stream"),
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
    Object.keys(unlockBundles).forEach(function (stage) {
      unlockBundles[stage].forEach(function (target) {
        if (target[0] !== "view") return;
        const button = workbench.querySelector('.staged-view-tabs [data-view="' + target[1] + '"]');
        button.dataset.nextStage = stage;
        button.dataset.targetKind = target[0];
        button.dataset.targetName = target[1];
      });
    });
    const unlockButtons = Array.from(workbench.querySelectorAll(".staged-unlock"));
    const panels = Array.from(workbench.querySelectorAll(".staged-panel"));
    let currentStage = 1;
    let currentView = "stream";
    let suppressProgress = false;
    const progress = {};
    const addedTargets = new Set(["control:steps", "view:stream"]);
    const targetStages = {};
    const stageEntrances = {
      2: "control:sgd-rate",
      3: "control:idbd-rate",
      4: "control:theta",
      5: "control:batch",
      6: "control:stream",
      7: "control:momentum",
      8: "control:decay",
      9: "control:decay-mode"
    };

    Object.keys(unlockBundles).forEach(function (stage) {
      unlockBundles[stage].forEach(function (target) {
        targetStages[target[0] + ":" + target[1]] = Number(stage);
      });
    });

    function targetKey(target) {
      return target[0] + ":" + target[1];
    }

    function applyAddedTargets() {
      workbench.querySelectorAll(".staged-control-slot").forEach(function (slot) {
        slot.classList.toggle("is-unlocked", addedTargets.has("control:" + slot.dataset.slot));
      });
      workbench.querySelector(".staged-lock-slot").classList.toggle("is-unlocked", addedTargets.has("control:idbd-rate"));
      workbench.querySelector(".staged-idbd-lane").classList.toggle("is-revealed", addedTargets.has("lane:idbd"));

      const hasVisibleTabs = Array.from(addedTargets).some(function (key) { return key.startsWith("view:") && key !== "view:stream"; });
      viewButtons.forEach(function (button) {
        const added = addedTargets.has("view:" + button.dataset.view);
        button.disabled = !added;
        button.classList.toggle("is-unlocked", added && (button.dataset.view !== "stream" || hasVisibleTabs));
      });
      clone.querySelectorAll(".staged-lane-score dl").forEach(function (detailsList) {
        detailsList.classList.toggle("is-unlocked", addedTargets.has("view:clean"));
      });
    }

    function addTarget(target) {
      addedTargets.add(targetKey(target));
      applyAddedTargets();
    }

    function removeTargetsAfter(stage) {
      Array.from(addedTargets).forEach(function (key) {
        if (targetStages[key] > stage) addedTargets.delete(key);
      });
    }

    function setView(view, userInitiated) {
      if (!addedTargets.has("view:" + view)) return;
      currentView = view;
      viewButtons.forEach(function (button) { button.classList.toggle("is-active", button.dataset.view === view); });
      panels.forEach(function (panel) { panel.classList.toggle("is-active", panel.dataset.view === view); });
      window.dispatchEvent(new Event("resize"));
    }

    function syncStatus() {
      status.textContent = actual.status.textContent;
      pause.disabled = actual.pause.disabled;
      play.disabled = actual.play.disabled;
    }

    function requiredProgress(stage) {
      return [1, 2, 3, 4, 5, 7, 8].includes(stage) ? 5 : 1;
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

    function offerTarget(target) {
      if (addedTargets.has(targetKey(target))) return;
      if (target[0] === "control") {
        workbench.querySelector('[data-slot="' + target[1] + '"]').classList.add("is-offering");
      } else if (target[0] === "view") {
        const button = workbench.querySelector('.staged-view-tabs [data-view="' + target[1] + '"]');
        button.disabled = false;
        button.textContent = "+ " + button.dataset.label;
        button.classList.add("is-offering");
      } else {
        workbench.querySelector(".staged-lane-curtain").classList.add("is-offering");
      }
    }

    function updateOffer() {
      clearOffers();
      Object.keys(unlockBundles).forEach(function (stage) {
        if (Number(stage) > currentStage) return;
        unlockBundles[stage].forEach(offerTarget);
      });

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
      const targets = unlockBundles[nextStage];
      const missing = targets.filter(function (target) { return !addedTargets.has(targetKey(target)); });
      navNext.textContent = missing.length === 1
        ? "Ready: " + missing[0][2].toLowerCase()
        : "Ready: " + String(missing.length) + " additions";
      missing.forEach(offerTarget);
    }

    function clearHiddenExtensions() {
      let changed = false;
      if (currentStage < 7 && actual.momentum.value !== "0") { actual.momentum.value = "0"; changed = true; }
      if (currentStage < 8 && actual.decay.value !== "0") { actual.decay.value = "0"; changed = true; }
      if (currentStage < 8 && actual.momentumMode.value !== "derived") { actual.momentumMode.value = "derived"; changed = true; }
      if (currentStage < 9 && actual.decayMode.value !== "traced") { actual.decayMode.value = "traced"; changed = true; }
      if (changed) {
        suppressProgress = true;
        actual.momentum.dispatchEvent(new Event("input", { bubbles: true }));
        suppressProgress = false;
      }
    }

    function setStage(stage, skipReplay, preserveView) {
      const previous = currentStage;
      currentStage = Math.max(1, Math.min(stages.length, stage));
      if (currentStage < previous) removeTargetsAfter(currentStage);
      const details = stages[currentStage - 1];
      clearOffers();
      workbench.dataset.stage = String(currentStage);
      stageCount.textContent = "Stage " + String(currentStage) + " of " + String(stages.length);
      navStage.textContent = stageCount.textContent;
      title.textContent = details[0];
      description.textContent = details[1];
      back.disabled = currentStage === 1;
      showAll.textContent = currentStage === stages.length ? "Start over" : "Open full lab";
      applyAddedTargets();
      clearHiddenExtensions();
      const preferredView = preserveView && addedTargets.has("view:" + currentView)
        ? currentView
        : (addedTargets.has("view:" + details[3]) ? details[3] : "stream");
      setView(preferredView, false);
      if (currentStage === 3 && previous < 3 && !skipReplay) actual.steps.dispatchEvent(new Event("input", { bubbles: true }));
      updateOffer();
    }

    viewButtons.forEach(function (button) {
      button.addEventListener("click", function () {
        if (button.classList.contains("is-offering")) {
          const target = [button.dataset.targetKind, button.dataset.targetName];
          addTarget(target);
          const nextStage = Number(button.dataset.nextStage);
          if (stageEntrances[nextStage] === targetKey(target)) setStage(nextStage, false, true);
          else updateOffer();
          return;
        }
        setView(button.dataset.view, true);
      });
    });
    unlockButtons.forEach(function (button) {
      button.addEventListener("click", function () {
        const target = [button.dataset.targetKind, button.dataset.targetName];
        addTarget(target);
        const nextStage = Number(button.dataset.nextStage);
        if (stageEntrances[nextStage] === targetKey(target)) setStage(nextStage, false, true);
        else updateOffer();
      });
    });
    [[actual.steps, 1], [actual.idbdRate, 3], [actual.theta, 4], [actual.batch, 5], [actual.momentum, 7], [actual.decay, 8]].forEach(function (entry) {
      entry[0].addEventListener("input", function () { if (!suppressProgress) markProgress(entry[1], 1); });
    });
    actual.sgdRate.addEventListener("input", function () {
      if (suppressProgress) return;
      markProgress(2, 1);
      if (actual.lockRates.checked) markProgress(3, 1);
    });
    actual.lockRates.addEventListener("change", function () { markProgress(3, 1); });
    actual.newStream.addEventListener("click", function () { markProgress(6, 1); });
    actual.reseed.addEventListener("click", function () { markProgress(6, 1); });
    actual.momentumMode.addEventListener("change", function () { markProgress(8, 1); markProgress(9, 1); });
    actual.decayMode.addEventListener("change", function () { markProgress(9, 1); });
    pause.addEventListener("click", function () { actual.pause.click(); markProgress(6, 1); });
    play.addEventListener("click", function () { actual.play.click(); markProgress(6, 1); });
    back.addEventListener("click", function () { setStage(currentStage - 1); });
    showAll.addEventListener("click", function () {
      if (currentStage === stages.length) {
        Object.keys(progress).forEach(function (stage) { delete progress[stage]; });
        addedTargets.clear();
        addedTargets.add("control:steps");
        addedTargets.add("view:stream");
        setStage(1, true);
      } else {
        Object.keys(unlockBundles).forEach(function (stage) { unlockBundles[stage].forEach(addTarget); });
        setStage(stages.length, false, true);
      }
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
