const slides = Array.from(document.querySelectorAll(".slide"));
const navButtons = Array.from(document.querySelectorAll(".nav-link"));
const progressFill = document.querySelector("#progressFill");
const slideCounter = document.querySelector("#slideCounter");
const prevBtn = document.querySelector("#prevBtn");
const nextBtn = document.querySelector("#nextBtn");
const indicators = Array.from(document.querySelectorAll(".indicator[data-region]"));
const regionMarkers = Array.from(document.querySelectorAll(".region-marker[data-region]"));

let currentSlide = 0;
let selectedRegion = "ventricles";

function setActiveRegion(region) {
  indicators.forEach((indicator) => {
    const isActive = indicator.dataset.region === region;
    indicator.classList.toggle("is-active", isActive);
    indicator.setAttribute("aria-pressed", isActive ? "true" : "false");
  });

  regionMarkers.forEach((marker) => {
    marker.classList.toggle("is-active", marker.dataset.region === region);
  });
}

function getInitialSlide() {
  const slideParam = new URLSearchParams(window.location.search).get("slide");
  const slideIndex = Number(slideParam);
  return Number.isInteger(slideIndex) ? slideIndex : 0;
}

function showSlide(index) {
  currentSlide = Math.max(0, Math.min(index, slides.length - 1));

  slides.forEach((slide, i) => {
    slide.classList.toggle("active", i === currentSlide);
    slide.setAttribute("aria-hidden", i === currentSlide ? "false" : "true");
    if (i === currentSlide && slide.id === "slide-type1") {
      slide.scrollTop = 0;
    }
  });

  navButtons.forEach((button, i) => {
    button.classList.toggle("active", i === currentSlide);
    button.setAttribute("aria-current", i === currentSlide ? "page" : "false");
  });

  progressFill.style.width = `${((currentSlide + 1) / slides.length) * 100}%`;
  slideCounter.textContent = `${currentSlide + 1} / ${slides.length}`;
  prevBtn.disabled = currentSlide === 0;
  nextBtn.disabled = currentSlide === slides.length - 1;
}

navButtons.forEach((button) => {
  button.addEventListener("click", () => {
    showSlide(Number(button.dataset.slide));
  });
});

indicators.forEach((indicator) => {
  const region = indicator.dataset.region;

  indicator.addEventListener("mouseenter", () => setActiveRegion(region));
  indicator.addEventListener("focus", () => setActiveRegion(region));
  indicator.addEventListener("mouseleave", () => setActiveRegion(selectedRegion));
  indicator.addEventListener("blur", () => setActiveRegion(selectedRegion));
  indicator.addEventListener("click", () => {
    selectedRegion = region;
    setActiveRegion(region);
  });
});

prevBtn.addEventListener("click", () => showSlide(currentSlide - 1));
nextBtn.addEventListener("click", () => showSlide(currentSlide + 1));

window.addEventListener("keydown", (event) => {
  if (event.key === "ArrowRight" || event.key === "PageDown") {
    showSlide(currentSlide + 1);
  }
  if (event.key === "ArrowLeft" || event.key === "PageUp") {
    showSlide(currentSlide - 1);
  }
  if (event.key === "Home") {
    showSlide(0);
  }
  if (event.key === "End") {
    showSlide(slides.length - 1);
  }
});

setActiveRegion(selectedRegion);
showSlide(getInitialSlide());
