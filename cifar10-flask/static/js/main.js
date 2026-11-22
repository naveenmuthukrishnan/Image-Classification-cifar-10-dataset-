// Small UX helpers for the upload page: image preview, drag highlight, loading state.
document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("upload-form");
  if (!form) return; // this script is loaded on every page; only run on the upload page

  const input = document.getElementById("image-input");
  const zone = document.getElementById("dropzone");
  const empty = document.getElementById("dz-empty");
  const preview = document.getElementById("preview");
  const fileName = document.getElementById("file-name");
  const button = document.getElementById("predict-btn");
  const label = button.querySelector(".btn-text");
  const MAX_BYTES = 5 * 1024 * 1024; // keep in sync with MAX_CONTENT_LENGTH in app.py

  function showPreview(file) {
    if (!file) return;

    if (file.size > MAX_BYTES) {
      fileName.textContent = "That file is over 5 MB. Please pick a smaller one.";
      fileName.classList.add("error");
      input.value = "";
      preview.hidden = true;
      empty.hidden = false;
      return;
    }

    fileName.classList.remove("error");
    fileName.textContent = `${file.name} · ${Math.round(file.size / 1024)} KB`;

    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      preview.hidden = false;
      empty.hidden = true;
    };
    reader.readAsDataURL(file);
  }

  input.addEventListener("change", () => showPreview(input.files[0]));

  // Highlight the zone while a file is dragged over it.
  // (We do NOT cancel "drop": the browser natively puts the file into the input.)
  ["dragenter", "dragover"].forEach((evt) =>
    zone.addEventListener(evt, () => zone.classList.add("is-over"))
  );
  ["dragleave", "drop"].forEach((evt) =>
    zone.addEventListener(evt, () => zone.classList.remove("is-over"))
  );

  form.addEventListener("submit", () => {
    button.classList.add("is-loading");
    button.disabled = true;
    label.textContent = "Thinking…";
  });

  // If the user comes back with the browser Back button, reset the button.
  window.addEventListener("pageshow", () => {
    button.classList.remove("is-loading");
    button.disabled = false;
    label.textContent = "Classify image";
  });
});
