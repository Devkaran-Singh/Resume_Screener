document.querySelector("form").addEventListener("submit", async (e) => {
  e.preventDefault(); // Stop form from refreshing the page

  const form = e.target;
  const submitButton = document.getElementById("submitButton");

  // Change button state
  submitButton.disabled = true;
  const originalText = submitButton.textContent;
  submitButton.textContent = "Uploading...";

  const formData = new FormData(form);

  try {
    const response = await fetch("/upload-resume", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "Upload failed");
    }

    alert(result.message || "Resume uploaded successfully!");

    // Explicitly clear fields
    form.querySelector('input[name="name"]').value = "";
    form.querySelector('select[name="category"]').value = "";
    form.querySelector('input[name="resume"]').value = "";
  } catch (err) {
    console.error("Upload error:", err);
    alert("Failed to upload resume: " + err.message);
  } finally {
    // Restore button state
    submitButton.textContent = originalText;
    submitButton.disabled = false;
  }
});
