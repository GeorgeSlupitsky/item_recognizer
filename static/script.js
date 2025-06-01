const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const resultDiv = document.getElementById("result");

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", e => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", e => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) uploadFile(file);
});

function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  resultDiv.innerHTML = "⏳ Processing...";

  fetch("/predict", {
    method: "POST",
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    let text = `✅ Category: ${data.category}\n📊 Confidence: ${(data.confidence * 100).toFixed(2)}%`;
    if (data.text) {
      text += `\n📄 Text: \n${data.text}`;
    }
    resultDiv.textContent = text;
  })
  .catch(err => {
    resultDiv.textContent = "❌ Error: " + err.message;
  });
}
