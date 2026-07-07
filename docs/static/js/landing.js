(function () {
  "use strict";

  /* ---------- Nav: background on scroll ---------- */
  var nav = document.getElementById("nav");
  function onScroll() {
    if (window.scrollY > 24) nav.classList.add("scrolled");
    else nav.classList.remove("scrolled");
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---------- Mobile menu ---------- */
  var toggle = document.querySelector(".nav-toggle");
  var menu = document.querySelector(".nav nav");
  if (toggle && menu) {
    toggle.addEventListener("click", function () { menu.classList.toggle("open"); });
    menu.addEventListener("click", function (e) { if (e.target.tagName === "A") menu.classList.remove("open"); });
  }

  /* ---------- Typewriter: headline + cascading sub-lines (loops) ---------- */
  var typed = document.getElementById("typed");
  var leadBox = document.getElementById("lead-lines");
  var headLive = document.querySelector(".th-live");
  var caret = headLive ? headLive.querySelector(".caret") : null;
  if (typed && leadBox && headLive && caret) {
    var HEAD = "An Open Exploration to a Unified Motion Foundation Model";
    typed.textContent = HEAD; // headline is static (no typing)
    var SUBS = [
      { label: "Unify",    text: "Unifying a fragmented landscape of motion settings, studied at modern scale." },
      { label: "Transfer", text: "Learning from a single wrist sensor, transferred across the body, devices, and tasks." },
      { label: "Apply",    text: "From everyday movement to behavior, mobility, and health \u2014 one backbone for it all." }
    ];
    var steps = [];
    SUBS.forEach(function (sub) {
      var line = document.createElement("div");
      line.className = "lead-line";
      var lbl = document.createElement("span");
      lbl.className = "ln-no";
      lbl.textContent = sub.label;
      var content = document.createElement("span");
      content.className = "ln-content";
      // ghost reserves the final width/height so nothing reflows while typing
      var ghost = document.createElement("span");
      ghost.className = "ln-ghost";
      ghost.textContent = sub.text;
      var text = document.createElement("span");
      text.className = "ln-text";
      var typed2 = document.createElement("span");
      text.appendChild(typed2);
      content.appendChild(ghost);
      content.appendChild(text);
      line.appendChild(lbl);
      line.appendChild(content);
      leadBox.appendChild(line);
      steps.push({ target: typed2, host: text, text: sub.text, speed: 20, pause: 850 });
    });

    var si = 0, ci = 0;
    function startStep() {
      if (si >= steps.length) { setTimeout(resetAll, 14000); return; }
      steps[si].host.appendChild(caret);
      ci = 0;
      typeChar();
    }
    function typeChar() {
      var s = steps[si];
      ci++;
      s.target.textContent = s.text.slice(0, ci);
      if (ci < s.text.length) setTimeout(typeChar, s.speed + Math.random() * 30);
      else { si++; setTimeout(startStep, si < steps.length ? steps[si].pause : 0); }
    }
    function resetAll() {
      steps.forEach(function (s) { s.target.textContent = ""; });
      si = 0;
      setTimeout(startStep, 450);
    }
    if (steps[0]) steps[0].host.appendChild(caret); // keep caret with the typed lines, not the static headline
    setTimeout(startStep, 500);
  }

  /* ---------- Fixed-design hero stage: scale the whole hero to fit any screen ---------- */
  (function () {
    var wrap = document.querySelector(".hero-scalewrap");
    var stage = document.querySelector(".hero-stage");
    if (!wrap || !stage) return;
    var DESIGN_W = 1520; // must match the .hero-stage width in CSS
    var STACK_BP = 920;  // below this, CSS switches to the fluid stacked layout

    function fit() {
      if (window.innerWidth <= STACK_BP) {
        stage.style.transform = "";
        stage.style.marginLeft = "";
        wrap.style.height = "";
        return;
      }
      stage.style.transform = "none";
      var avail = wrap.clientWidth;
      var scale = Math.min(1, avail / DESIGN_W);
      var h = stage.offsetHeight; // unscaled layout height
      stage.style.transform = "scale(" + scale.toFixed(4) + ")";
      stage.style.marginLeft = Math.max(0, (avail - DESIGN_W * scale) / 2) + "px";
      wrap.style.height = (h * scale + 10) + "px"; // small buffer so the CTA's lower edge isn't clipped
    }

    window.addEventListener("resize", fit, { passive: true });
    window.addEventListener("load", fit);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(fit);
    fit();
    // re-fit shortly after, once typed ghosts / fonts have settled
    setTimeout(fit, 300);
    setTimeout(fit, 900);
  })();

  /* ---------- Reveal on scroll ---------- */
  var reveals = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en, idx) {
        if (en.isIntersecting) {
          en.target.style.transitionDelay = (Math.min(idx, 4) * 70) + "ms";
          en.target.classList.add("in");
          io.unobserve(en.target);
        }
      });
    }, { threshold: 0.14, rootMargin: "0px 0px -8% 0px" });
    reveals.forEach(function (el) { io.observe(el); });
  } else {
    reveals.forEach(function (el) { el.classList.add("in"); });
  }

  /* ---------- Count-up stats ---------- */
  function animateCount(el) {
    var target = parseFloat(el.getAttribute("data-count"));
    var suffix = el.getAttribute("data-suffix") || "";
    var decimals = (el.getAttribute("data-count").split(".")[1] || "").length;
    var dur = 1400, start = null;
    function frame(ts) {
      if (!start) start = ts;
      var p = Math.min((ts - start) / dur, 1);
      var eased = 1 - Math.pow(1 - p, 3);
      el.textContent = (target * eased).toFixed(decimals) + suffix;
      if (p < 1) requestAnimationFrame(frame);
      else el.textContent = target.toFixed(decimals) + suffix;
    }
    requestAnimationFrame(frame);
  }
  var nums = document.querySelectorAll(".num[data-count]");
  if ("IntersectionObserver" in window) {
    var io2 = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { animateCount(en.target); io2.unobserve(en.target); }
      });
    }, { threshold: 0.5 });
    nums.forEach(function (el) { io2.observe(el); });
  } else {
    nums.forEach(animateCount);
  }

  /* ---------- Minimal cards: click toggles open (for touch / pinning) ---------- */
  document.querySelectorAll(".mcard").forEach(function (c) {
    c.addEventListener("click", function () { c.classList.toggle("open"); });
  });

  /* ---------- Bullet list: click a line to expand its detail ---------- */
  document.querySelectorAll(".bl-head").forEach(function (h) {
    h.addEventListener("click", function () {
      h.parentNode.classList.toggle("open");
    });
  });

  /* ---------- Timeline steps: click to pin open (hover handled by CSS) ---------- */
  document.querySelectorAll(".tl-step").forEach(function (s) {
    s.addEventListener("click", function () { s.classList.toggle("open"); });
  });

  /* ---------- "Discover more": swap a finding's main view for detailed results ---------- */
  document.querySelectorAll(".finding-views").forEach(function (views) {
    var main = views.querySelector(".view-main");
    var detail = views.querySelector(".view-detail");
    var openBtn = views.querySelector(".discover-btn");
    var backBtn = views.querySelector(".back-btn");
    if (!main || !detail) return;

    function swap(incoming, outgoing) {
      outgoing.style.opacity = "0";
      setTimeout(function () {
        outgoing.style.display = "none";
        incoming.style.display = (incoming === detail) ? "block" : "";
        incoming.style.opacity = "0";
        requestAnimationFrame(function () {
          requestAnimationFrame(function () { incoming.style.opacity = "1"; });
        });
      }, 260);
    }

    if (openBtn) openBtn.addEventListener("click", function () { swap(detail, main); });
    if (backBtn) backBtn.addEventListener("click", function () { swap(main, detail); });
  });

  /* ---------- Pseudo-3D rotating constellation ---------- */
  function normalize(v) { var m = Math.hypot(v[0], v[1], v[2]); return [v[0] / m, v[1] / m, v[2] / m]; }

  var constel = document.querySelector(".constellation");
  if (constel) {
    var svg = constel.querySelector(".constel-lines");
    var stars = Array.prototype.slice.call(constel.querySelectorAll(".star"));
    // irregular asterism on the front hemisphere (varied radii so it doesn't read as a ring)
    var base = [
      [-0.88, 0.52, 0.16],
      [-0.12, 0.18, 0.52],
      [ 0.46, 0.80, 0.24],
      [ 0.90, -0.08, 0.20],
      [ 0.14, -0.66, 0.42],
      [-0.58, -0.34, 0.30]
    ].map(normalize);
    var edges = [];
    var SVGNS = "http://www.w3.org/2000/svg";
    var lines = edges.map(function () {
      var l = document.createElementNS(SVGNS, "line");
      l.setAttribute("class", "cline");
      svg.appendChild(l);
      return l;
    });

    /* --- background starfield (dim twinkling particles) --- */
    var canvas = constel.querySelector(".starfield");
    var ctx = canvas ? canvas.getContext("2d") : null;
    var bgStars = [];
    (function initStars() {
      var n = 110;
      for (var i = 0; i < n; i++) {
        bgStars.push({
          x: Math.random(), y: Math.random(),
          r: Math.random() * 2.6 + 2.8,
          a: Math.random() * 0.28 + 0.12
        });
      }
    })();
    var starfieldDrawn = false;
    function drawStars(w, h) {
      if (!ctx) return;
      var dpr = window.devicePixelRatio || 1;
      var needResize = canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr);
      if (!needResize && starfieldDrawn) return;
      if (needResize) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      for (var i = 0; i < bgStars.length; i++) {
        var s = bgStars[i];
        ctx.beginPath();
        ctx.arc(s.x * w, s.y * h, s.r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(78,163,207," + s.a.toFixed(3) + ")";
        ctx.fill();
      }
      starfieldDrawn = true;
    }

    var DEF_YAW = 0.2, DEF_PITCH = -0.05;
    var yaw = DEF_YAW, pitch = DEF_PITCH, tYaw = yaw, tPitch = pitch;
    var selected = null;

    function rotate(v, y, p) {
      var cy = Math.cos(y), sy = Math.sin(y);
      var x1 = v[0] * cy + v[2] * sy;
      var z1 = -v[0] * sy + v[2] * cy;
      var y1 = v[1];
      var cp = Math.cos(p), sp = Math.sin(p);
      var y2 = y1 * cp - z1 * sp;
      var z2 = y1 * sp + z1 * cp;
      return [x1, y2, z2];
    }
    function select(i) {
      selected = i;
      var v = base[i];
      tYaw = Math.atan2(-v[0], v[2]);
      tPitch = Math.atan2(v[1], Math.hypot(v[0], v[2]));
      stars.forEach(function (s) { s.classList.remove("active"); });
      stars[i].classList.add("active");
      constel.classList.add("has-active");
    }
    function deselect() {
      selected = null;
      tYaw = DEF_YAW;
      tPitch = DEF_PITCH;
      stars.forEach(function (s) { s.classList.remove("active"); });
      constel.classList.remove("has-active");
    }

    stars.forEach(function (s, i) {
      s.addEventListener("mouseenter", function () { select(i); });
      s.addEventListener("focus", function () { select(i); });
      // tap support on touch devices (no hover)
      s.addEventListener("click", function (e) {
        e.stopPropagation();
        if (selected === i) deselect(); else select(i);
      });
    });
    constel.addEventListener("mouseleave", function () { deselect(); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape" && selected !== null) deselect(); });

    function frame() {
      yaw += (tYaw - yaw) * 0.12;
      pitch += (tPitch - pitch) * 0.12;
      var w = constel.clientWidth, h = constel.clientHeight;
      drawStars(w, h);
      var cx = w / 2, cy = h / 2;
      var Rx = w * 0.44, Ry = h * 0.46, Rz = Math.min(Rx, Ry), f = Rz * 2.6;
      var proj = [];
      for (var i = 0; i < base.length; i++) {
        var r = rotate(base[i], yaw, pitch);
        var x = r[0] * Rx, y = r[1] * Ry, z = r[2] * Rz;
        var factor = f / (f - z);
        var sx = cx + x * factor, sy = cy - y * factor;
        var depth = (z + Rz) / (2 * Rz);
        proj.push({ sx: sx, sy: sy, depth: depth, z: z });

        var st = stars[i];
        st.style.left = sx + "px";
        st.style.top = sy + "px";
        var sc = 0.6 + depth * 0.7;
        if (selected === i) sc *= 1.25;
        var dot = st.querySelector(".dot");
        if (dot) dot.style.transform = "scale(" + sc.toFixed(3) + ")";
        var op = 0.4 + depth * 0.6;
        if (selected !== null && selected !== i) op *= 0.22;
        st.style.opacity = op.toFixed(3);
        st.style.zIndex = String(100 + Math.round(z));
      }
      for (var e = 0; e < edges.length; e++) {
        var a = proj[edges[e][0]], b = proj[edges[e][1]];
        var l = lines[e];
        l.setAttribute("x1", a.sx.toFixed(1));
        l.setAttribute("y1", a.sy.toFixed(1));
        l.setAttribute("x2", b.sx.toFixed(1));
        l.setAttribute("y2", b.sy.toFixed(1));
        var lo = 0.1 + Math.min(a.depth, b.depth) * 0.28;
        if (selected !== null) lo *= 0.45;
        l.style.opacity = lo.toFixed(3);
      }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
})();
