---
layout: default
title: About Me
date: 2025-08-25
---

<style>
  @page { size: A4; margin: 0; }
  :root { --navy: #2e3a4e; --rule: #9aa3b0; }

  .cv * { margin: 0; padding: 0; box-sizing: border-box; }
  .cv {
    box-sizing: border-box;
    font-family: 'Lato', sans-serif;
    font-size: 9.5pt;
    color: #3d4451;
    line-height: 1.36;
    padding: 9mm 13mm 7mm 13mm;
    background: #ffffff;
    width: 210mm;
    max-width: 100%;
    margin: 1.5rem auto;
    box-shadow: 0 1px 10px rgba(0, 0, 0, 0.14);
  }

  /* ---------- header ---------- */
  .cv .header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 3.5mm; }
  .cv .name {
    font-family: 'Poppins', sans-serif;
    font-size: 26pt; font-weight: 700; color: var(--navy); color: #121ecc;
    line-height: 1.05; letter-spacing: 0.5px;
  }
  .cv .role-title {
    font-family: 'Poppins', sans-serif;
    font-size: 10.5pt; font-weight: 400; color: #59616e;
    letter-spacing: 4px; margin-top: 1.8mm;
  }
  .cv .contact { font-size: 8.6pt; color: #4a5160; padding-top: 1mm; line-height: 1.65; text-align: right; }
  .cv .contact a { color: #4a5160; }

  /* ---------- summary ---------- */
  .cv .summary {
    margin-bottom: 4.5mm;
    font-size: 9.7pt;
    color: #404856;
    border-left: 2.5px solid var(--navy);
    padding-left: 4mm;
    text-align: justify;
  }

  /* ---------- sections ---------- */
  .cv .section { margin-bottom: 3.8mm; }
  .cv .section-title {
    font-family: 'Poppins', sans-serif;
    font-size: 13.5pt; font-weight: 700; color: var(--navy);
    border-bottom: 1px solid var(--rule);
    padding-bottom: 1.2mm; margin-bottom: 2.6mm;
  }

  /* ---------- timeline ---------- */
  .cv .timeline { position: relative; margin-left: 2mm; }
  .cv .timeline::before {
    content: ""; position: absolute; left: 0; top: 2.2mm; bottom: 2mm;
    width: 1px; background: #6b7484;
  }
  .cv .job { position: relative; padding-left: 6.5mm; margin-bottom: 3mm; }
  .cv .job:last-child { margin-bottom: 0; }
  .cv .job::before {
    content: ""; position: absolute; left: -1.35mm; top: 0.7mm;
    width: 2.5mm; height: 2.5mm;
    border: 1.1px solid #25272b; border-radius: 100%; background: #fff;
  }
  .cv .job-head { display: flex; justify-content: space-between; align-items: baseline; }
  .cv .job-dates { font-weight: 700; color: var(--navy); font-size: 9.6pt; }
  .cv .job-company { font-weight: 550; font-size: 16px; color: #121ecc; margin: 0mm }
  .cv .job-role { font-weight: 700; color: var(--navy); font-size: 10pt; margin: 0.4mm 0 0.4mm 0; }
  .cv .job-scope { font-style: italic; color: #59616e; margin-bottom: 1.2mm; }

  /* ---------- bullets ---------- */
  .cv ul { list-style: none; }
  .cv ul.bullets > li { position: relative; padding-left: 4mm; margin-bottom: 0.8mm; text-align: justify; }
  .cv ul.bullets > li::before {
    content: ""; position: absolute; left: 0.5mm; top: 1.5mm;
    width: 1.3mm; height: 1.3mm; border-radius: 50%; background: #abadb1;
  }
  .cv b { color: #343b48; }

  /* ---------- projects ---------- */
  .cv .project { margin-left: 2mm; padding-left: 6.5mm; margin-bottom: 2.2mm; text-align: justify; }
  .cv .project-name { font-weight: 700; color: var(--navy); }
  .cv .project a { color: #4a5160; font-size: 8.6pt; }

  /* ---------- education & skills ---------- */
  .cv .edu { margin-left: 2mm; padding-left: 6.5mm; }
  .cv .edu-degree { font-weight: 700; color: var(--navy); font-size: 10pt; }
  .cv .edu-school { font-weight: 700; color: #4a5160; }
  .cv .edu-detail { font-size: 9pt; color: #4a5160; margin-top: 0.5mm; text-align: justify; }
  .cv .skills { margin-left: 2mm; padding-left: 6.5mm; }
</style>

<div class="cv">

  <!-- 1. Identity & contact -->
  <div class="header">
    <div>
      <div class="name">Alex Chaloner</div>
      <div class="role-title">MACHINE LEARNING ENGINEER</div>
    </div>
    <div class="contact">
      London, UK<br>
      <a href="https://linkedin.com/in/alexchaloner/">linkedin.com/in/alexchaloner</a><br>
      <a href="https://github.com/AlexChaloner">github.com/AlexChaloner</a><br>
    </div>
  </div>

  <!-- 2. Summary -->
  <div class="summary">
    ML engineer at Meta working on LLMs for compromise detection at scale - finetuning with <b>GRPO reinforcement learning</b>, ML at <b>internet scale</b>.
    Oxford Master's in Mathematics &amp; Computer Science with dissertation in reinforcement learning, and independent projects on LLM pre- and post-training.
  </div>

  <!-- 3. Experience -->
  <div class="section">
    <div class="section-title">Experience</div>
    <div class="timeline">
      <div class="job">
        <div class="job-head">
          <span class="job-dates">July 2024 - Present</span>
        </div>
          <span class="job-company">Meta</span>

        <div class="job-role">Software Engineer (Machine Learning)</div>
        <div class="job-scope">Machine learning for account compromise</div>
        <ul class="bullets">
          <li>Led ML within the team, growing ML-based detection from <b>12% to 70%</b> of abusive-compromise detections.</li>
          <li>Finetuned LLMs with <b>GRPO reinforcement learning</b> for detection and measurement, increasing users detected/measured by LLM systems by <b>10%</b>.</li>
          <li>Deployed unsupervised ML models at <b>internet scale</b>, protecting <b>millions of compromised users per year</b>.</li>
          <li>Onboarded software engineers onto ML model development, adding protection for <b>millions more users per year</b>.</li>
          <li>Authored the org's scaling strategy: demonstrated empirically that <b>scaling laws</b> held in our domain from 5M- to 5B-parameter models and equivalent data range, determined scaling law formulae, and drove adoption of self-supervised methods to exploit data scale.</li>
        </ul>
      </div>

      <div class="job">
        <div class="job-head">
          <span class="job-dates">November 2021 - July 2024</span>
        </div>
        <span class="job-company">Ocado Technology</span>
        <div class="job-role">Software Engineer</div>
        <ul class="bullets">
          <li>Built full-stack software for the automated freezer warehouse in Luton (live since Sept 2023): <b>Java</b>/<b>Spring</b>
            microservices with <b>AWS</b> SQS, S3 and Lambda; <b>Terraform</b> deployments via GitLab CI.</li>
          <li>Joined the internal ML team and built a <b>GPT-4</b>-powered knowledge-sharing tool (via RAG) for engineers.</li>
          <li>Built warehouse analytics dashboards: Java + Scala services on <b>DynamoDB</b>, with a TypeScript+React frontend.</li>
        </ul>
      </div>

      <div class="job">
        <div class="job-head">
          <span class="job-dates">October 2020 - July 2021</span>
        </div>
        <span class="job-company">Agari Data, Inc</span>
        <div class="job-role">Data Scientist</div>
        <ul class="bullets">
          <li>Applied <b>LLMs</b> to detection of malicious business emails; built the internal fraud-analytics frontend in <b>Python</b>.</li>
          <li>Data analysis with heavy <b>SQL</b> usage across large email datasets.</li>
        </ul>
      </div>
    </div>
  </div>

  <!-- 4. Independent research -->
  <div class="section">
    <div class="section-title">Independent Projects</div>
    <div class="project">
      <span class="project-name">Muon vs Adam</span> - <a href="https://github.com/AlexChaloner/muon_vs_adam">github.com/AlexChaloner/muon_vs_adam</a><br>
      Empirical comparison of the <b>Muon</b> optimizer against AdamW on transformer training.
      Muon consistently beats Adam and AdamW in both validation loss and wallclock time, and much less sensitive to hyperparameters.
    </div>
    <div class="project">
      <span class="project-name">Self-Distillation Experiments</span> - <a href="https://github.com/AlexChaloner/simple_self_distillation_experiment">github.com/AlexChaloner/simple_self_distillation_experiment</a><br>
      Minimal reproduction of Apple's <b>simple self-distillation</b> paper. Replicated the changed top-p shape but
      could not replicate success - got consistently worse performance on coding benchmark -
      also designed a closed-form cheap transformation on token distribution to achieve same end.
    </div>
  </div>

  <!-- 5. Education -->
  <div class="section">
    <div class="section-title">Education</div>
    <div class="edu">
      <div class="job-head">
        <span class="job-dates">2016 - 2020</span>
      </div>
      <span class="job-company">University of Oxford</span>
      <div class="edu-degree">Master of Mathematics and Computer Science (MMathCompSci)</div>
      <div class="edu-detail">2:1 (upper second-class honours). Dissertation: <i>Hierarchical Reinforcement Learning</i> - compressing action policies via graph abstraction.</div>
    </div>
  </div>

</div>

<script>
/*
 * Smart last-line justification.
 *
 * `text-align: justify` (set in CSS) stretches every line of a block EXCEPT
 * the last one. That's usually right: a short last line should stay ragged,
 * not be blown apart into ugly gaps. But when the last line already reaches
 * most of the way across the column, leaving it ragged looks like a mistake
 * next to the fully-justified lines above it.
 *
 * So: for each target block we measure how much of the column its last line
 * fills. If that's more than FILL_THRESHOLD, we also justify the last line
 * (text-align-last); otherwise we leave it ragged. Single-line bullets count
 * too — a long one gets stretched, a short one is left alone.
 *
 * Tune FILL_THRESHOLD to taste: lower = more last lines get stretched,
 * higher = only the very-nearly-full ones do.
 */
(function () {
  var FILL_THRESHOLD = 0.80;
  var TARGETS = '.cv ul.bullets > li, .cv .summary';

  function contentWidth(el) {
    var cs = getComputedStyle(el);
    return el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  }

  function lastLineRatio(el) {
    var range = document.createRange();
    range.selectNodeContents(el);
    var rects = range.getClientRects();
    if (!rects.length) return 0;

    // Bottom-most line = largest top coordinate.
    var maxTop = -Infinity;
    for (var i = 0; i < rects.length; i++) {
      if (rects[i].top > maxTop) maxTop = rects[i].top;
    }
    // A single visual line can be split into several rects by inline <b>
    // tags; they share a top, so gather all rects on the bottom line and
    // take the full left-to-right extent.
    var left = Infinity, right = -Infinity;
    for (var j = 0; j < rects.length; j++) {
      if (Math.abs(rects[j].top - maxTop) <= 2) {
        if (rects[j].left < left) left = rects[j].left;
        if (rects[j].right > right) right = rects[j].right;
      }
    }
    var avail = contentWidth(el);
    return avail > 0 ? (right - left) / avail : 0;
  }

  function apply() {
    var els = document.querySelectorAll(TARGETS);
    for (var i = 0; i < els.length; i++) {
      els[i].style.textAlignLast = '';                 // reset before measuring
      if (lastLineRatio(els[i]) > FILL_THRESHOLD) {
        els[i].style.textAlignLast = 'justify';
      }
    }
  }

  // Re-run whenever layout could have changed (fonts loading, resize, print).
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(apply);
  window.addEventListener('load', apply);
  window.addEventListener('resize', apply);
  window.addEventListener('beforeprint', apply);
  apply();
})();
</script>
