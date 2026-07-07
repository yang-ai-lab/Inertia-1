(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", function () {
    var nav = document.querySelector(".nav-links");
    var toggle = document.querySelector(".nav-toggle");

    // Mobile menu toggle
    if (toggle && nav) {
      toggle.addEventListener("click", function () {
        nav.classList.toggle("open");
      });
      nav.querySelectorAll("a").forEach(function (a) {
        a.addEventListener("click", function () {
          nav.classList.remove("open");
        });
      });
    }

    // Scrollspy: highlight nav link for the section in view
    var links = Array.prototype.slice.call(
      document.querySelectorAll('.nav-links a[data-section]')
    );
    var sections = links
      .map(function (l) { return document.getElementById(l.getAttribute("data-section")); })
      .filter(Boolean);

    function setActive(id) {
      links.forEach(function (l) {
        l.classList.toggle("active", l.getAttribute("data-section") === id);
      });
    }

    if ("IntersectionObserver" in window && sections.length) {
      var spy = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) setActive(e.target.id);
        });
      }, { rootMargin: "-45% 0px -50% 0px", threshold: 0 });
      sections.forEach(function (s) { spy.observe(s); });
    }

    // Reveal-on-scroll
    var reveals = Array.prototype.slice.call(document.querySelectorAll(".reveal"));
    if ("IntersectionObserver" in window && reveals.length) {
      var ro = new IntersectionObserver(function (entries, obs) {
        entries.forEach(function (e) {
          if (e.isIntersecting) { e.target.classList.add("in"); obs.unobserve(e.target); }
        });
      }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
      reveals.forEach(function (r) { ro.observe(r); });
    } else {
      reveals.forEach(function (r) { r.classList.add("in"); });
    }
  });
})();
