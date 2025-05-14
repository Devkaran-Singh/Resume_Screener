const btn = document.getElementById("job_button");
const textarea = document.getElementById("job_description_text");
const shortlistBtn = document.getElementById("shortlist-button");
const candidateList = document.getElementById("candidates");
const list_candidates = document.getElementById("list");
const shortlistInput = document.getElementById("shortlist-input");

let initialJobDescription = "";

// Load the initial job description on page load
document.addEventListener("DOMContentLoaded", async () => {
  try {
    // Fetch the job description from the backend
    const response = await fetch("/job-description"); // This should be handled by the backend
    const descText = await response.text();

    // Default description if the file is empty or not found
    const defaultJobDescription = `We are looking for an experienced Data Scientist to join our team. The ideal candidate has
strong skills in Machine Learning, Python, and SQL. Experience with deep learning frameworks
like TensorFlow or PyTorch is preferred.

Requirements:
- Proficiency in Python and SQL
- Strong experience with ML frameworks (scikit-learn, TensorFlow, PyTorch)
- Experience with data visualization tools (Matplotlib, Seaborn, Tableau)
- Knowledge of big data technologies (Spark, Hadoop) is a plus
- Excellent communication skills to present findings to stakeholders`;

    const finalDescription = descText.trim() ? descText : defaultJobDescription;

    if (textarea) {
      textarea.value = finalDescription;
      initialJobDescription = finalDescription;
    }
  } catch (error) {
    console.error("Error loading job description:", error);
  }
});

btn.addEventListener("click", async () => {
  const div1 = document.getElementById("shortlist");
  const div2 = document.getElementById("rank_skills");
  const div3 = document.getElementById("skills_set");
  div1.classList.remove("hidden");
  div2.classList.remove("hidden");
  div1.classList.add("flex", "flex-col");
  div2.classList.add("flex", "flex-col");

  try {
    const newDescription = textarea.value.trim();

    // Only make the POST request if the description has changed
    if (newDescription !== initialJobDescription) {
      const response = await fetch("/set-job-description", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ jobDescription: newDescription }),
      });

      if (!response.ok) {
        throw new Error("Failed to update job description.");
      }
    }

    // Fetch skills only after ensuring the job description was updated
    const skillsResponse = await fetch("http://localhost:3000/extract-skills");
    if (!skillsResponse.ok) {
      throw new Error("Failed to fetch skills.");
    }

    const skillsData = await skillsResponse.json();

    // Dynamically create skill rating elements
    skillsData.skills_to_rate.forEach((skill) => {
      const div4 = document.getElementById("skills");
      const skillDiv = div4.cloneNode(true);
      const sanitizedSkill = skill.toLowerCase().replace(/[^a-z]/g, "_");

      skillDiv.querySelectorAll('input[type="radio"]').forEach((radio) => {
        radio.name = `rating_${sanitizedSkill}`;
      });
      skillDiv.classList.remove("hidden");
      skillDiv.querySelector("p").textContent = `${skill}`;
      div3.appendChild(skillDiv);
    });
  } catch (error) {
    console.error("Error loading data:", error);
    alert(
      "An error occurred while updating or fetching data. Please try again."
    );
  }
});

shortlistBtn.addEventListener("click", async () => {
  const num = parseInt(shortlistInput.value);

  if (isNaN(num) || num <= 0) {
    alert("Please enter a valid number of candidates to shortlist.");
    return;
  }

  // Create an object to store skill ratings
  const skillRatings = {};

  // Iterate over each checked radio button and map the values
  document.querySelectorAll('input[type="radio"]:checked').forEach((radio) => {
    const skillName = radio.name.split("_")[1]; // Extract skill name from the radio button's name
    const ratingValue = parseInt(radio.value);

    // Map the rating (1-5) to the backend range (6-10)
    const backendRating = ratingValue + 5;

    // Store the mapped rating in the skillRatings object
    skillRatings[skillName] = backendRating;
  });

  try {
    // Prepare the request body with only skill ratings
    const requestBody = skillRatings;

    // Send the request with the "top" parameter in the query string
    const res = await fetch(`/rank-candidates?top=${num}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });

    const data = await res.json();
    console.log("Shortlist response:", data);

    const { top_candidates: candidates, visualizations } = data;

    if (!Array.isArray(candidates)) {
      throw new Error("Invalid candidate data format");
    }

    // Ensure the candidate template exists before cloning
    const c = document.getElementById("candidate-list");
    if (!c) {
      console.error("Candidate list template element not found");
      alert("Candidate template not found. Please check your HTML structure.");
      return;
    }

    // Clear previous candidates
    list_candidates.classList.remove("hidden");
    list_candidates.classList.add("flex");

    // Now clone the candidate template for each candidate
    candidates.forEach((candidate) => {
      const list = c.cloneNode(true);
      list.classList.remove("hidden");

      // Use class selectors instead of IDs
      list.querySelector(".rank").textContent = candidate.rank;
      list.querySelector(".name").textContent = candidate.name;
      list.querySelector(".category").textContent = candidate.category;
      list.querySelector(".skills").textContent =
        candidate.matching_skills.join(", ");
      list.querySelector(".s1").textContent = candidate.scores.similarity;
      list.querySelector(".s2").textContent = candidate.scores.skill_match;
      list.querySelector(".s3").textContent = candidate.scores.combined;
      list.querySelector(".s4").textContent = candidate.experience_years;
      list.querySelector(".s5").textContent = candidate.experience_skill_points;

      list_candidates.appendChild(list);
    });

    // Display the visualizations
    const scoreBreakdownImg = document.getElementById("score_breakdown");
    const topSkillsImg = document.getElementById("top_skills");
    const v = document.getElementById("visualizations");
    v.classList.remove("hidden");
    v.classList.add("flex");
    // Set the src attribute to the correct image URLs
    if (visualizations.score_breakdown) {
      scoreBreakdownImg.src = `/generated_images/${visualizations.score_breakdown}`;
    }

    if (visualizations.top_skills) {
      topSkillsImg.src = `/generated_images/${visualizations.top_skills}`;
    }
  } catch (err) {
    console.error("Error fetching shortlisted candidates:", err);
  }
});
