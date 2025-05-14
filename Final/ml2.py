import os
SKILL_CONFIG = os.path.join(os.path.dirname(__file__),'skills_config.json')
DATASET_PATH = os.path.join(os.path.dirname(__file__),'finaldataset.csv')
PLOT_1 = os.path.join(os.path.dirname(__file__),'combined_ranking_breakdown.png')
PLOT_2 = os.path.join(os.path.dirname(__file__),'top_skills.png')
RANKING_RESULTS = 'ranking_results.json'
JOB_DESCRIPTION = os.path.join(os.path.dirname(__file__), 'job_description.txt')
import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import io
import base64
from collections import Counter

class ResumeScreeningSystem:
    def __init__(self):
        # Load skill data from external JSON file
        with open(SKILL_CONFIG, "r") as f:
            skill_data = json.load(f)
        
        self.technical_skills = skill_data["technical_skills"]
        self.skill_variants = skill_data["skill_variants"]
        self.skill_groups = skill_data["skill_groups"]

        self.stop_words = set(stopwords.words('english')).union({
            'resume', 'cv', 'curriculum', 'vitae', 'page', 'contact', 'email', 'phone',
            'address', 'reference', 'education', 'experience', 'work', 'skill', 'skills'
        })
        
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 3),
                                   sublinear_tf=True, stop_words=list(self.stop_words))
        # For experience clustering (unsupervised)
        self.kmeans_exp = None
        self.exp_level_labels = None
        self.scaler = MinMaxScaler()
        
        self.ranked_resumes = None
        self.visualizations = {}
    
    def clean_text(self, text):
        text = re.sub(r'\b([A-Za-z]+)[+&#](\b|_)', r'\1SPECIAL', str(text).lower())
        text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+|[^\w\s.]', ' ', text)
        text = re.sub(r'\b\d+\b', 'NUM', text)
        text = re.sub(r'\b\w{1,2}\b', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def preprocess(self, text):
        tokens = word_tokenize(self.clean_text(text))
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)
        
    def extract_skills(self, text, job_skills=None):
        text_lower = self.clean_text(text)
        skills = []
        
        # Process skills (job skills first, then technical skills)
        all_skills = list(job_skills if job_skills else [])
        all_skills.extend([s for s in self.technical_skills if s not in all_skills])
        
        for skill in all_skills:
            sl = skill.lower()
            # Match both the full skill and substrings (if relevant)
            if ' ' in sl and (sl in text_lower or sl.replace('-', ' ') in text_lower):
                skills.append(skill)
            elif re.search(r'\b' + re.escape(sl) + r'\b', text_lower):
                skills.append(skill)

        # Check skill variants
        for base_skill, variants in self.skill_variants.items():
            if any(s.lower() == base_skill for s in skills):
                continue
                
            for variant in variants:
                if re.search(r'\b' + re.escape(variant) + r'\b', text_lower):
                    matching_skills = [s for s in self.technical_skills if s.lower() == base_skill]
                    if matching_skills:
                        skills.append(matching_skills[0])
                        break
        
        return list(set(skills))
    
    def extract_keywords_from_job_description(self, job_description, skills_list):
        # Preprocess job description and extract keywords (skills)
        job_tokens = self.preprocess(job_description).split()
        job_skill_counter = Counter(job_tokens)

        # Extract relevant skills mentioned in the job description
        extracted_skills = [skill for skill in skills_list if skill.lower() in job_skill_counter]
        return extracted_skills

    def get_dynamic_skill_weights(self, job_description, job_skills, skill_ratings=None):
        # Extract skills from job description
        extracted_skills = self.extract_keywords_from_job_description(job_description, self.technical_skills)
        # skill_ratings must be provided by CLI (no inline prompts)
        if skill_ratings is None:
            # If not provided, assign default rating 7 for extracted skills
            skill_ratings = {skill: 7 for skill in extracted_skills}
        # Assign weights directly
        skill_weights = {skill: rating for skill, rating in skill_ratings.items()}
        # Prepare group lookup
        group_lookup = {}
        for group, skills in self.skill_groups.items():
            for skill in skills:
                group_lookup[skill] = group
        # Find rated groups
        rated_groups = {group_lookup.get(skill) for skill in skill_ratings if skill in group_lookup}
        rated_groups.discard(None)
        # Assign weights to other skills
        for skill in self.technical_skills:
            if skill not in skill_weights:
                group = group_lookup.get(skill)
                if group in rated_groups:
                    skill_weights[skill] = 3.75
                else:
                    skill_weights[skill] = 1.5
        return skill_weights

    def process_resumes(self, resumes_df, job_description, skill_ratings=None):
        # Extract keywords (skills) from the job description
        job_skills = self.extract_keywords_from_job_description(job_description, self.technical_skills)
        # Get dynamic skill weights based on job description
        skill_weights = self.get_dynamic_skill_weights(job_description, job_skills, skill_ratings)
        # Preprocess resumes
        resumes_df['Processed'] = resumes_df['Resume'].apply(self.preprocess)
        resumes_df['Skills'] = resumes_df['Resume'].apply(lambda x: self.extract_skills(x, job_skills))
        resumes_df['Skill_Count'] = resumes_df['Skills'].apply(len)
        # TF-IDF and similarity calculation
        all_docs = resumes_df['Processed'].tolist() + [self.preprocess(job_description)]
        self.tfidf.fit(all_docs)
        tfidf_matrix = self.tfidf.transform(resumes_df['Processed'])
        job_vec = self.tfidf.transform([self.preprocess(job_description)])
        resumes_df['Similarity_Score'] = cosine_similarity(job_vec, tfidf_matrix).flatten()
        # Calculate skill match percentage dynamically based on adjusted skill weights
        total_weight = sum(skill_weights.get(s, 0) for s in job_skills)
        # Avoid division by zero
        if total_weight == 0:
            total_weight = 1
        resumes_df['Skill_Match_Percentage'] = resumes_df['Skills'].apply(
            lambda x: sum(skill_weights.get(s, 0) for s in x if s in job_skills)/total_weight * 100
        )
        resumes_df['Matching_Skills'] = resumes_df['Skills'].apply(
            lambda x: [s for s in x if s in job_skills]
        )
        resumes_df['Matching_Skill_Count'] = resumes_df['Matching_Skills'].apply(len)
        # Experience clustering (unsupervised, for skill points/level assignment)
        def extract_experience_years(text):
            text = str(text).lower()
            # Match patterns like 'X years Y months', 'X yrs Y mos', 'X+ years', etc.
            year_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:\+?\s*)?(?:years?|yrs?)', text)
            month_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:months?|mos?)', text)

            years = float(year_match.group(1)) if year_match else 0
            months = float(month_match.group(1)) if month_match else 0

            return round(years + months / 12, 2) if (years or months) else np.nan
        if 'Experience' in resumes_df.columns:
            exp_values = resumes_df['Experience'].fillna('').apply(lambda x: extract_experience_years(x) if not pd.isnull(x) else np.nan)
        else:
            exp_values = resumes_df['Resume'].apply(extract_experience_years)
        # Replace missing with median
        median_exp = exp_values[~np.isnan(exp_values)].median() if np.any(~np.isnan(exp_values)) else 0
        exp_values = exp_values.fillna(median_exp)
        resumes_df['Experience_Years'] = exp_values
        # Cluster experience into 3 levels using KMeans, assign skill points based on cluster mean experience
        experience_array = resumes_df['Experience_Years'].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        resumes_df['Experience_Skill_Points'] = kmeans.fit_predict(experience_array)
        # Map cluster labels to skill points based on mean experience per cluster
        cluster_means = resumes_df.groupby('Experience_Skill_Points')['Experience_Years'].mean()
        sorted_clusters = cluster_means.sort_values().index.tolist()
        cluster_to_points = {cluster: point+1 for point, cluster in enumerate(sorted_clusters)}
        resumes_df['Experience_Skill_Points'] = resumes_df['Experience_Skill_Points'].map(cluster_to_points)
        # If you want to keep Experience_Level, comment/uncomment below line:
        # resumes_df['Experience_Level'] = kmeans.fit_predict(experience_array)
        # Normalize experience skill points for scoring
        exp_skill_points_norm = self.scaler.fit_transform(resumes_df[['Experience_Skill_Points']].values)
        resumes_df['Experience_Skill_Score'] = exp_skill_points_norm.flatten()
        # Use only similarity, skill match, and experience skill score for combined score
        # Normalize Skill_Match_Percentage for scoring (since it's in 0-100)
        norm_cols = ['Similarity_Score', 'Skill_Match_Percentage', 'Experience_Skill_Score']
        norm_df = resumes_df[norm_cols].copy()
        norm_df['Skill_Match_Percentage'] = norm_df['Skill_Match_Percentage'] / 100.0
        scores = self.scaler.fit_transform(norm_df)
        # 40% similarity, 40% skill match, 20% experience skill score
        resumes_df['Combined_Score'] = scores[:,0]*0.2 + scores[:,1]*0.6 + scores[:,2]*0.2
        # Rank resumes by combined score
        self.ranked_resumes = resumes_df.sort_values(by='Combined_Score', ascending=False).reset_index(drop=True)

    def show_top_candidates(self, top_n=10):
        return self.ranked_resumes.head(top_n)[['Name', 'Combined_Score', 'Skill_Match_Percentage', 'Similarity_Score']]

    def print_top_candidates(self, job_skills, top_n=10):
        print(f"\n--- 🎯 TOP {top_n} CANDIDATES ---")
        for i, row in self.ranked_resumes.head(top_n).iterrows():
            print(f"\nRank {i+1}: {row['Name']}")
            print(f"Category: {row.get('Category', 'N/A')}")
            print(f"Similarity Score: {row['Similarity_Score']:.2f}")
            print(f"Skills Matched: {row['Skill_Match_Percentage']:.1f}%")
            print(f"Experience Level: {row.get('Experience_Level', 'N/A')} (Skill Points: {row.get('Experience_Skill_Points', 'N/A')})")
            print(f"Combined Score: {row['Combined_Score']:.2f}")
            matching_skills = set(row['Skills']).intersection(set(job_skills))
            print(f"Matching Skills: {', '.join(matching_skills) if matching_skills else 'None'}")

    def generate_visualizations(self, top_n=10):
        top_n = min(top_n, len(self.ranked_resumes))
        plot_images = {}

        # Helper function to save plots
        def save_plot(key):
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_images[key] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

        # Top skills bar chart (replace score_distribution)
        top_skills = []
        for skills in self.ranked_resumes.head(top_n)['Skills']:
            top_skills.extend(skills)
        skill_count = Counter(top_skills)
        top_skills_df = pd.DataFrame({
            'Skill': list(skill_count.keys()),
            'Count': list(skill_count.values())
        }).sort_values(by='Count', ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Skill', y='Count',
                    data=top_skills_df.head(min(15, len(top_skills_df))),
                    palette='Set2')
        plt.title(f'Most Common Skills Among Top {top_n} Candidates')
        plt.ylabel('Frequency')
        plt.xlabel('Skill')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_plot('top_skills')

        # Score breakdown stacked bar + line chart (replace previous score_breakdown)
        df_plot = self.ranked_resumes.head(top_n).copy()
        df_plot['Skill_Match_Percentage'] /= 100
        stacked_parts = ['Similarity_Score', 'Skill_Match_Percentage', 'Experience_Skill_Score']
        df_plot['Name_Rank'] = [f"#{i+1} {name}" for i, name in enumerate(df_plot['Name'])]

        plt.figure(figsize=(14, 8))
        bottom_vals = np.zeros(top_n)

        for part in stacked_parts:
            plt.bar(df_plot['Name_Rank'], df_plot[part], bottom=bottom_vals, label=part)
            bottom_vals += df_plot[part]

        plt.plot(df_plot['Name_Rank'], df_plot['Combined_Score'], color='black', marker='o', linewidth=2, label='Combined Score')
        plt.title('Top Candidates – Score Breakdown and Ranking')
        plt.xlabel('Candidate (Ranked)')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Score Components', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        save_plot('score_breakdown')

        with open(PLOT_1, "wb") as f:
            f.write(base64.b64decode(plot_images['score_breakdown']))

        with open(PLOT_2, "wb") as f:
            f.write(base64.b64decode(plot_images['top_skills']))




import argparse
job_description = "job_description.txt"
def get_job_description():
    return """We are looking for an experienced Data Scientist to join our team. The ideal candidate has
strong skills in Machine Learning, Python, and SQL. Experience with deep learning frameworks
like TensorFlow or PyTorch is preferred.
Requirements:
- Proficiency in Python and SQL
- Strong experience with ML frameworks (scikit-learn, TensorFlow, PyTorch)
- Experience with data visualization tools (Matplotlib, Seaborn, Tableau)
- Knowledge of big data technologies (Spark, Hadoop) is a plus
- Excellent communication skills to present findings to stakeholders
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--get_job_description", action="store_true")
    parser.add_argument("--extract_skills", action="store_true")
    parser.add_argument("--resume_csv", type=str, default=DATASET_PATH)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--skill_ratings_json", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=RANKING_RESULTS)
    parser.add_argument("--set_job_description", type=str)
    args = parser.parse_args()

    if args.get_job_description:
        if os.path.exists(job_description):
            with open(job_description, "r") as f:
                print(f.read().strip())
        else:
            job_desc_text = print(get_job_description())
        exit()
    if args.extract_skills:
        # Read job description FIRST
        if os.path.exists(JOB_DESCRIPTION):
            with open(JOB_DESCRIPTION, "r") as f:
                job_desc_text = f.read().strip()
        else:
            job_desc_text = get_job_description()
        
        screener = ResumeScreeningSystem()
        skills = screener.extract_keywords_from_job_description(job_desc_text, screener.technical_skills)
        print(json.dumps({"skills_to_rate": list(set(skills))}))
        exit()
    
    # print(job_description)
    
    if args.extract_skills:
        screener = ResumeScreeningSystem()
        skills = screener.extract_keywords_from_job_description(job_description, screener.technical_skills)
        output = {"skills_to_rate": list(set(skills))}
        print(json.dumps(output))
        exit()

    # Load resume data
    resumes_df = pd.read_csv(args.resume_csv)

    # Load skill ratings if provided
    skill_ratings = None
    if args.skill_ratings_json:
        with open(args.skill_ratings_json, "r") as f:
            skill_ratings = json.load(f)

    # Process resumes
    screener = ResumeScreeningSystem()
    screener.process_resumes(resumes_df, job_description, skill_ratings)
    screener.generate_visualizations(top_n=min(args.top_n, len(screener.ranked_resumes)))
    top_candidates = screener.show_top_candidates(top_n=min(args.top_n, len(screener.ranked_resumes)))
    job_skills = screener.extract_keywords_from_job_description(job_description, screener.technical_skills)

    # Store ranked results in output JSON
    output_data = {'top_candidates': []}
    for idx, row in screener.ranked_resumes.head(args.top_n).iterrows():
        candidate = {
            "rank": idx + 1,
            "name": row['Name'],
            "resume_id": row.get('Resume_ID', None),
            "category": row.get('Category', ''),
            "experience_years": row.get('Experience_Years', 0),
            "scores": {
                "similarity": round(float(row.get('Similarity_Score', 0)), 2),
                "skill_match": round(float(row.get('Skill_Match_Percentage', 0)), 2),
                "combined": round(float(row.get('Combined_Score', 0)), 2)
            },
            "matching_skills": list(set(row['Skills']).intersection(set(job_skills)))
        }
        # Only include experience_skill if > 0
        if row.get('Experience_Skill_Score', 0) > 0:
            candidate["scores"]["experience_skill"] = round(float(row['Experience_Skill_Score']), 2)
        # Only include experience_skill_points if > 0
        if row.get('Experience_Skill_Points', 0) > 0:
            candidate["experience_skill_points"] = row['Experience_Skill_Points']
        # Only include experience_level if > 0
        if row.get('Experience_Level', 0) > 0:
            candidate["experience_level"] = row['Experience_Level']
        output_data['top_candidates'].append(candidate)

    # Add the paths of the visualizations to the output data
    output_data['visualizations'] = {
        'score_breakdown': PLOT_1,
        'top_skills': PLOT_2
    }

    # Save the ranking results with visualizations to JSON
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2)