// Paths configuration
const express = require("express");
const multer = require("multer");
const { spawn } = require("child_process");
const fs = require("fs");
const app = express();
const path = require("path");
const port = 3000;

const UPLOAD_DIR = path.join(__dirname, "Final", "uploads");
const PYTHON_SCRIPT = path.join(__dirname, "Final/new.py");
const EXTRA_SCRIPT = path.join(__dirname, "Final/extra.py");
const JSON_PATH_Ranking = path.join(__dirname, "Final/ranking_results.json");
const CSV_PATH = path.join(__dirname, "Final/finaldataset.csv");
const SKILL_RATING = path.join(__dirname, "Final/skill_ratings.json");
const JOB_DESCRIPTION_FILE = path.join(
  __dirname,
  "Final",
  "job_description.txt"
);

app.use(express.static(path.join(__dirname, "frontend")));
app.use(
  "/generated_images",
  express.static(path.join(__dirname, "generated_images"))
);

app.get("/admin", (req, res) => {
  res.sendFile(path.join(__dirname, "frontend", "admin.html"));
});

// Serve user.html for /user route
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "frontend", "user.html"));
});

// Ensure upload directory exists
try {
  if (!fs.existsSync(UPLOAD_DIR)) {
    fs.mkdirSync(UPLOAD_DIR, { recursive: true });
    console.log(`Created upload directory: ${UPLOAD_DIR}`);
  }
} catch (err) {
  console.error(`Failed to create upload directory: ${err.message}`);
  process.exit(1);
}

// Configure file upload storage
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, UPLOAD_DIR);
  },
  filename: function (req, file, cb) {
    const sanitizedName = file.originalname.replace(/[()]/g, "_");
    cb(null, Date.now() + "-" + sanitizedName);
  },
});

const upload = multer({
  storage: storage,
  fileFilter: function (req, file, cb) {
    if (file.mimetype !== "application/pdf") {
      return cb(new Error("Only PDF files are allowed"));
    }
    cb(null, true);
  },
});

// Helper function to run Python scripts
function runPythonScript(scriptPath, args = []) {
  console.log(`Running Python script: ${scriptPath} with args:`, args);

  return new Promise((resolve, reject) => {
    const python = spawn("python", [scriptPath, ...args]);
    let stdout = "";
    let stderr = "";

    python.stdout.on("data", (data) => {
      const output = data.toString();
      stdout += output;
      console.log(`Python stdout: ${output.trim()}`);
    });

    python.stderr.on("data", (data) => {
      const output = data.toString();
      stderr += output;
      console.error(`Python stderr: ${output.trim()}`);
    });

    python.on("close", (code) => {
      console.log(`Python process exited with code ${code}`);
      if (code !== 0) {
        return reject(new Error(`Python exited with code ${code}: ${stderr}`));
      } else {
        resolve(stdout.trim());
      }
    });
  });
}

app.use(express.json()); // For raw JSON bodies

app.get("/job-description", async (req, res) => {
  try {
    const description = await fs.promises.readFile(
      JOB_DESCRIPTION_FILE,
      "utf8"
    );
    const cleanedDescription = description.trim();
    if (cleanedDescription.length === 0) {
      // Return default description if file is empty
      const defaultDescription = `We are looking for an experienced Data Scientist to join our team. The ideal candidate has
strong skills in Machine Learning, Python, and SQL. Experience with deep learning frameworks
like TensorFlow or PyTorch is preferred.

Requirements:
- Proficiency in Python and SQL
- Strong experience with ML frameworks (scikit-learn, TensorFlow, PyTorch)
- Experience with data visualization tools (Matplotlib, Seaborn, Tableau)
- Knowledge of big data technologies (Spark, Hadoop) is a plus
- Excellent communication skills to present findings to stakeholders`;
      return res.send(defaultDescription);
    }
    res.send(cleanedDescription);
  } catch (error) {
    console.error("Failed to read job description:", error.message);
    res.status(500).json({ error: "Failed to read job description" });
  }
});

// Helper function to rate skills
async function rateSkills(req, res, next) {
  const ratings = req.body;
  console.log(ratings);
  // Save skill ratings to file
  fs.writeFile(SKILL_RATING, JSON.stringify(ratings, null, 2), (err) => {
    if (err) {
      console.error("Failed to save skill ratings:", err);
      return res.status(500).json({ error: "Failed to save skill ratings" });
    }
    console.log("Skill ratings saved successfully");
    next();
  });
}

// POST endpoint for resume upload
const { execFile } = require("child_process");
app.post("/upload-resume", upload.single("resume"), async (req, res) => {
  console.log("Received resume upload request");

  if (!req.file) {
    console.error("No file uploaded or file upload failed");
    return res.status(400).json({ error: "No file uploaded" });
  }

  console.log(`File saved to: ${req.file.path}`);
  const { name, category } = req.body;
  if (!name || !category) {
    console.error(
      `Missing required fields: name=${name}, category=${category}`
    );
    return res.status(400).json({ error: "Missing name or category" });
  }

  const runPythonScript = (scriptPath, args) => {
    return new Promise((resolve, reject) => {
      execFile("python", [scriptPath, ...args], (error, stdout, stderr) => {
        if (error) {
          return reject(error);
        }
        resolve(stdout);
      });
    });
  };
  try {
    // Optional: run your Python script here
    // const output = await runPythonScript("your_script.py", [req.file.path, name, category]);
    
    console.log("Resume uploaded and processed successfully");
    return res.json({ message: "Resume uploaded successfully!" }); // ✅ Send JSON response
  } catch (err) {
    console.error("Error running Python script:", err);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// POST endpoint to rank candidates (rateSkills middleware only)
app.post("/rank-candidates", rateSkills, async (req, res) => {
  const topN = req.body.top || req.query.top || "5";
  try {
    await runPythonScript(PYTHON_SCRIPT, [
      "--top_n",
      topN,
      "--skill_ratings_json",
      SKILL_RATING,
      "--resume_csv",
      CSV_PATH,
      "--output_json",
      JSON_PATH_Ranking,
    ]);

    fs.readFile(JSON_PATH_Ranking, "utf8", (err, data) => {
      if (err)
        return res
          .status(500)
          .json({ error: "Failed to read ranking results" });
      res.json(JSON.parse(data));
    });
  } catch (error) {
    console.error("Error running ranking:", error);
    res.status(500).json({ error: "Error running ranking" });
  }
});

// GET endpoint to extract skills and return as JSON
app.get("/extract-skills", async (req, res) => {
  try {
    const result = await runPythonScript(PYTHON_SCRIPT, ["--extract_skills"]);
    const skills = JSON.parse(result);
    res.json(skills);
  } catch (error) {
    console.error("Error extracting skills:", error);
    res.status(500).json({ error: "Error extracting skills" });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error("Unhandled error:", err.message);
  if (err instanceof multer.MulterError) {
    return res.status(400).json({ error: `Upload error: ${err.message}` });
  }
  res.status(500).json({ error: err.message });
});

// Route to set job description via direct file write
app.post("/set-job-description", async (req, res) => {
  const { jobDescription } = req.body;

  if (!jobDescription || typeof jobDescription !== "string") {
    return res
      .status(400)
      .json({ error: "Invalid or missing job description" });
  }

  try {
    await fs.promises.writeFile(JOB_DESCRIPTION_FILE, jobDescription, "utf8");
    console.log("Job description updated in job_description.txt");
    res.json({ message: "Job description updated successfully" });
  } catch (error) {
    console.error("Failed to update job description:", error.message);
    res.status(500).json({
      error: "Failed to update job description",
      details: error.message,
    });
  }
});

app.listen(port, () => console.log(`Server running on port ${port}`));
