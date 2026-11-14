(() => {
  "use strict";

  // Same-origin by default. For local testing of the HTML on its own, set
  // window.API_URL = "http://127.0.0.1:8000/predict" before this script loads.
  const API_URL = window.API_URL || "/predict";
  const REQUEST_TIMEOUT_MS = 60000; // Render free instances can take ~30-50s to wake up
  const MIN_ANALYSIS_MS = 1700;     // keep the scan animation visible even on fast responses
  const CIRC = 2 * Math.PI * 64;    // ring circumference (r = 64)

  const $ = (id) => document.getElementById(id);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- Particle background ---------- */
  const cv = $("bg");
  const cx = cv.getContext("2d");
  let W = 0, H = 0, particles = [];
  let mx = 0.5, my = 0.5, tint = "124,131,255";

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth;
    H = window.innerHeight;
    cv.width = W * dpr;
    cv.height = H * dpr;
    cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const count = Math.round(Math.min(110, Math.max(40, (W * H) / 16000)));
    particles = Array.from({ length: count }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      z: Math.random() * 0.8 + 0.2,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      dx: 0,
      dy: 0,
    }));
  }

  function frame() {
    cx.clearRect(0, 0, W, H);
    for (const p of particles) {
      if (!reduceMotion) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0) p.x = W; else if (p.x > W) p.x = 0;
        if (p.y < 0) p.y = H; else if (p.y > H) p.y = 0;
      }
      p.dx = p.x + (mx - 0.5) * 50 * p.z;
      p.dy = p.y + (my - 0.5) * 50 * p.z;
      cx.fillStyle = `rgba(${tint},${0.25 + p.z * 0.5})`;
      cx.beginPath();
      cx.arc(p.dx, p.dy, p.z * 2, 0, Math.PI * 2);
      cx.fill();
    }
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i], b = particles[j];
        const d = Math.hypot(a.dx - b.dx, a.dy - b.dy);
        if (d < 100) {
          cx.strokeStyle = `rgba(${tint},${0.15 * (1 - d / 100)})`;
          cx.beginPath();
          cx.moveTo(a.dx, a.dy);
          cx.lineTo(b.dx, b.dy);
          cx.stroke();
        }
      }
    }
    if (!reduceMotion) requestAnimationFrame(frame);
  }

  resize();
  frame();
  window.addEventListener("resize", () => { resize(); if (reduceMotion) frame(); });

  /* ---------- 3D tilt + glare following the pointer ---------- */
  const tilt = $("tilt");
  const card = $("card");
  const glare = $("glare");
  const finePointer = window.matchMedia("(pointer: fine)").matches;

  if (finePointer && !reduceMotion) {
    window.addEventListener("mousemove", (e) => {
      mx = e.clientX / window.innerWidth;
      my = e.clientY / window.innerHeight;
      const flipped = card.classList.contains("flipped");
      const ry = (mx - 0.5) * 12;
      const rx = (0.5 - my) * 9;
      // The flip rotation lives on .card; the tilt is applied to the wrapper.
      tilt.style.transform = `rotateY(${ry}deg) rotateX(${rx}deg)`;
      if (!flipped) {
        const r = tilt.getBoundingClientRect();
        const gx = ((e.clientX - r.left) / r.width) * 100;
        const gy = ((e.clientY - r.top) / r.height) * 100;
        glare.style.background =
          `radial-gradient(circle at ${gx}% ${gy}%, rgba(255,255,255,0.12), transparent 55%)`;
      }
    });
    document.addEventListener("mouseleave", () => {
      tilt.style.transform = "";
      mx = my = 0.5;
    });
    tilt.style.transition = "transform 0.15s ease-out";
    tilt.style.transformStyle = "preserve-3d";
  }

  /* ---------- Grade rules (taken from credit_risk_dataset.csv) ---------- */
  // Interest rate range per grade: 1st to 99th percentile of the training data,
  // so a few outlier rows (e.g. 6.0% on grade D) don't widen the range.
  const RATE_RANGES = {
    A: [5.4, 9.6],
    B: [8.9, 12.7],
    C: [10.9, 16.0],
    D: [12.5, 18.3],
    E: [14.1, 19.9],
    F: [15.4, 21.7],
    G: [17.9, 22.8],
  };
  // No applicant with a prior default has grade A or B in the training data.
  const GRADES_BLOCKED_ON_DEFAULT = ["A", "B"];
  const FIRST_GRADE_AFTER_DEFAULT = "C";

  /* ---------- Chip groups (radio behaviour) ---------- */
  function selectChip(group, btn) {
    group.querySelectorAll("button").forEach((b) => {
      const on = b === btn;
      b.classList.toggle("on", on);
      b.setAttribute("aria-checked", String(on));
    });
  }

  document.querySelectorAll(".chips").forEach((group) => {
    group.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn || btn.disabled) return;
      selectChip(group, btn);
      if (group.id === "cb_person_default_on_file") applyDefaultRule();
      if (group.id === "loan_grade") onGradeChange();
    });
  });
  const chipValue = (id) => $(id).querySelector("button.on").dataset.value;

  /* ---------- Error / hint text under a field ---------- */
  const errOf = (el) => el.parentElement.querySelector(".err");

  function setFieldError(el, msg) {
    el.classList.toggle("bad", Boolean(msg));
    const err = errOf(el);
    err.classList.remove("hint");
    err.textContent = msg;
  }

  // Shows the typical interest range for the selected grade, unless the field
  // is currently showing an error.
  function updateRateHint() {
    const rateEl = $("loan_int_rate");
    if (rateEl.classList.contains("bad")) return;
    const g = chipValue("loan_grade");
    const [lo, hi] = RATE_RANGES[g];
    const err = errOf(rateEl);
    err.classList.add("hint");
    err.textContent = `Grade ${g}: ${lo}–${hi}%`;
  }

  function onGradeChange() {
    // A new grade means a new range, so drop any old range error and show the new hint.
    const rateEl = $("loan_int_rate");
    if (rateEl.classList.contains("bad") && rateEl.dataset.rangeError === "1") {
      setFieldError(rateEl, "");
      delete rateEl.dataset.rangeError;
    }
    updateRateHint();
  }

  /* ---------- Rule: prior default => grade C or lower ---------- */
  const gradeGroup = $("loan_grade");
  const gradeNote = $("grade-note");

  function applyDefaultRule() {
    const hasDefault = chipValue("cb_person_default_on_file") === "Y";
    let moved = false;

    gradeGroup.querySelectorAll("button").forEach((b) => {
      const blocked = hasDefault && GRADES_BLOCKED_ON_DEFAULT.includes(b.dataset.value);
      b.disabled = blocked;
      if (blocked && b.classList.contains("on")) moved = true;
    });

    if (moved) {
      selectChip(gradeGroup, gradeGroup.querySelector(`button[data-value="${FIRST_GRADE_AFTER_DEFAULT}"]`));
    }

    gradeNote.textContent = hasDefault
      ? (moved
          ? `Grade moved to ${FIRST_GRADE_AFTER_DEFAULT}: applicants with a prior default are graded C or lower.`
          : "Applicants with a prior default are graded C or lower.")
      : "";

    onGradeChange();
  }

  /* ---------- Validation ---------- */
  const RULES = {
    person_age:                 { label: "Age",            min: 18,  max: 100, int: true },
    person_income:              { label: "Income",         min: 1 },
    person_emp_length:          { label: "Employment",     min: 0,   max: 60 },
    loan_amnt:                  { label: "Loan amount",    min: 1 },
    loan_int_rate:              { label: "Interest rate",  min: 0.1, max: 40 },
    cb_person_cred_hist_length: { label: "Credit history", min: 0,   max: 80,  int: true },
  };

  function validate() {
    const values = {};
    let firstBad = null;
    for (const [id, r] of Object.entries(RULES)) {
      const el = $(id);
      const raw = el.value.trim();
      const n = Number(raw);
      let msg = "";
      if (raw === "" || Number.isNaN(n)) msg = "Enter a number";
      else if (r.int && !Number.isInteger(n)) msg = "Use a whole number";
      else if (r.min !== undefined && n < r.min) msg = `Must be at least ${r.min}`;
      else if (r.max !== undefined && n > r.max) msg = `Must be at most ${r.max}`;
      setFieldError(el, msg);
      if (msg && !firstBad) firstBad = el;
      values[id] = n;
    }

    // The model was trained on rows where employment length <= age.
    const empEl = $("person_emp_length");
    if (!empEl.classList.contains("bad") && values.person_emp_length > values.person_age) {
      setFieldError(empEl, "Can't exceed age");
      if (!firstBad) firstBad = empEl;
    }

    // Interest rate must sit inside the range seen for the selected grade.
    const grade = chipValue("loan_grade");
    const rateEl = $("loan_int_rate");
    delete rateEl.dataset.rangeError;
    if (!rateEl.classList.contains("bad")) {
      const [lo, hi] = RATE_RANGES[grade];
      if (values.loan_int_rate < lo || values.loan_int_rate > hi) {
        setFieldError(rateEl, `Grade ${grade} needs ${lo}–${hi}%`);
        rateEl.dataset.rangeError = "1";
        if (!firstBad) firstBad = rateEl;
      } else {
        updateRateHint();
      }
    }

    // Safety net: the chips are disabled, but never send this combination.
    if (chipValue("cb_person_default_on_file") === "Y" && GRADES_BLOCKED_ON_DEFAULT.includes(grade)) {
      applyDefaultRule();
      if (!firstBad) firstBad = gradeGroup.querySelector("button.on");
    }

    return { values, firstBad };
  }

  document.querySelectorAll(".in").forEach((el) => {
    el.addEventListener("input", () => {
      el.classList.remove("bad");
      delete el.dataset.rangeError;
      const err = el.parentElement.querySelector(".err");
      if (err) err.textContent = "";
      if (el.id === "loan_int_rate") updateRateHint();
    });
  });

  /* ---------- Status line + API call ---------- */
  const statusEl = $("status");
  const goBtn = $("go");
  const beam = $("beam");
  const STEPS = ["Reading the application", "Scoring 11 features", "Applying the risk threshold"];

  function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.remove("error");
    if (isError) {
      void statusEl.offsetWidth; // restart the shake animation
      statusEl.classList.add("error");
    }
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function callApi(payload) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), REQUEST_TIMEOUT_MS);
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: ctrl.signal,
      });
      if (!res.ok) {
        let detail = "";
        try {
          const j = await res.json();
          detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
        } catch (_) { /* ignore */ }
        const e = new Error(detail || `Server returned ${res.status}`);
        e.status = res.status;
        throw e;
      }
      return await res.json();
    } finally {
      clearTimeout(timer);
    }
  }

  /* ---------- Result rendering ---------- */
  const ring = $("ring");
  const tick = $("tick");
  const big = $("big");
  const glow = $("glow");
  const fmtMoney = (n) => "$" + Math.round(n).toLocaleString("en-US");

  function showResult(res, v, grade) {
    const p = Math.min(Math.max(Number(res.default_probability), 0), 1);
    const thr = Number(res.threshold);
    const high = Number(res.default_prediction) === 1;

    document.documentElement.style.setProperty("--risk", high ? "var(--bad)" : "var(--ok)");
    glow.classList.remove("ok", "bad");
    glow.classList.add(high ? "bad" : "ok");
    tint = high ? "248,113,113" : "52,211,153";
    if (reduceMotion) frame();

    $("verdict").textContent = high ? "High risk" : "Low risk";
    $("thr").textContent =
      `Flagged as high risk at ${(thr * 100).toFixed(1)}% or above. Marker on the ring shows the cut-off.`;

    // Reset to empty, then flip and animate in.
    ring.style.strokeDashoffset = CIRC;
    tick.style.transform = "rotate(0deg)";
    big.textContent = "0%";
    ["f1", "f2", "f3", "f4"].forEach((id) => { $(id).style.width = "0"; });

    const lpi = v.loan_amnt / v.person_income;
    const bars = [
      ["f1", "v1", Math.min(lpi / 0.5, 1), `${Math.round(lpi * 100)}%`],
      ["f2", "v2", Math.min(v.loan_int_rate / 25, 1), `${v.loan_int_rate.toFixed(1)}%`],
      ["f3", "v3", (("ABCDEFG".indexOf(grade)) + 1) / 7, grade],
      ["f4", "v4", Math.min(v.cb_person_cred_hist_length / 30, 1), `${v.cb_person_cred_hist_length} yrs`],
    ];
    bars.forEach(([f, t, , text]) => { $(t).textContent = text; });

    card.classList.add("flipped");
    $("back").setAttribute("aria-hidden", "false");

    requestAnimationFrame(() => requestAnimationFrame(() => {
      ring.style.strokeDashoffset = CIRC * (1 - p);
      tick.style.transform = `rotate(${thr * 360}deg)`;
      bars.forEach(([f, , frac]) => { $(f).style.width = `${Math.max(4, frac * 100)}%`; });
    }));

    // Count-up, time-based so it matches the ring regardless of value.
    const target = p * 100;
    const t0 = performance.now() + 550;
    const dur = reduceMotion ? 1 : 1400;
    (function tickUp(now) {
      const k = Math.min(Math.max((now - t0) / dur, 0), 1);
      const eased = 1 - Math.pow(1 - k, 3);
      big.textContent = `${(target * eased).toFixed(target < 10 ? 1 : 0)}%`;
      if (k < 1) requestAnimationFrame(tickUp);
      else big.textContent = `${target.toFixed(target < 10 ? 1 : 0)}%`;
    })(performance.now());

    setTimeout(() => $("edit").focus({ preventScroll: true }), 900);
  }

  /* ---------- Submit ---------- */
  $("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (goBtn.disabled) return;

    const { values: v, firstBad } = validate();
    if (firstBad) {
      firstBad.focus();
      setStatus("Fix the highlighted fields to continue.", true);
      return;
    }

    const grade = chipValue("loan_grade");
    const payload = {
      person_age: v.person_age,
      person_income: v.person_income,
      person_home_ownership: chipValue("person_home_ownership"),
      person_emp_length: v.person_emp_length,
      loan_intent: $("loan_intent").value,
      loan_grade: grade,
      loan_amnt: v.loan_amnt,
      loan_int_rate: v.loan_int_rate,
      loan_percent_income: Math.round((v.loan_amnt / v.person_income) * 100) / 100,
      cb_person_default_on_file: chipValue("cb_person_default_on_file"),
      cb_person_cred_hist_length: v.cb_person_cred_hist_length,
    };

    goBtn.disabled = true;
    goBtn.classList.add("loading");
    goBtn.textContent = "Analyzing…";
    beam.classList.add("go");

    let stepIdx = 0;
    setStatus(STEPS[0]);
    const stepTimer = setInterval(() => {
      stepIdx = Math.min(stepIdx + 1, STEPS.length - 1);
      setStatus(STEPS[stepIdx]);
    }, 600);
    const wakeTimer = setTimeout(() => {
      clearInterval(stepTimer);
      setStatus("Waking up the server. The first request can take up to a minute…");
    }, 4000);

    try {
      const [res] = await Promise.all([callApi(payload), sleep(MIN_ANALYSIS_MS)]);
      showResult(res, v, grade);
      setStatus("");
    } catch (err) {
      let msg;
      if (err.name === "AbortError") msg = "The server took too long to respond. Try again in a moment.";
      else if (err instanceof TypeError) msg = "Can't reach the server. Check your connection and try again.";
      else if (err.status === 422) msg = "The server rejected one of the values. Check the form and try again.";
      else msg = "Something went wrong on the server. Try again.";
      setStatus(msg, true);
      console.error("Prediction failed:", err);
    } finally {
      clearInterval(stepTimer);
      clearTimeout(wakeTimer);
      beam.classList.remove("go");
      goBtn.disabled = false;
      goBtn.classList.remove("loading");
      goBtn.textContent = "Analyze application";
    }
  });

  /* ---------- Back to the form ---------- */
  $("edit").addEventListener("click", () => {
    card.classList.remove("flipped");
    $("back").setAttribute("aria-hidden", "true");
    glow.classList.remove("ok", "bad");
    tint = "124,131,255";
    if (reduceMotion) frame();
    setTimeout(() => $("person_age").focus({ preventScroll: true }), 700);
  });

  $("back").setAttribute("aria-hidden", "true");
  applyDefaultRule(); // sets the initial grade state and the interest hint
})();
