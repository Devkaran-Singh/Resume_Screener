import os
SKILL_CONFIG = os.path.join(os.path.dirname(__file__), 'skills_config.json')
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'finaldataset.csv')
# Use a single directory path for plots
PLOT = os.path.join(os.path.dirname(__file__), '..', 'generated_images')
RANKING_RESULTS = os.path.join(os.path.dirname(__file__), 'ranking_results.json')
JOB_DESCRIPTION = os.path.join(os.path.dirname(__file__), 'job_description.txt')
import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import base64 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import io
from collections import Counter

class ResumeScreeningSystem:
    def __init__(self):
        with open(SKILL_CONFIG, "r") as f:
            skill_data = json.load(f)

        self.job_skills = None
        self.technical_skills = skill_data["technical_skills"]
        self.skill_variants = skill_data["skill_variants"]
        self.skill_groups = skill_data["skill_groups"]
        self.contextual_skills = skill_data.get("contextual_skills", {})

        self.stop_words = set(stopwords.words('english')).union({
            'resume', 'cv', 'curriculum', 'vitae', 'page', 'contact', 'email', 'phone',
            'address', 'reference', 'education', 'experience', 'work', 'skill', 'skills'
        })
        
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 3),
                                   sublinear_tf=True, stop_words=list(self.stop_words))
        self.kmeans_exp = None
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
    
    def extract_skills(self, text):
        """Extract technical skills from text with improved multiword handling."""
        text_lower = text.lower()
        clean_text = self.clean_text(text)
        
        # POS tag the original text to preserve context
        tokens = nltk.word_tokenize(text_lower)
        pos_tags = nltk.pos_tag(tokens)
        pos_dict = {word.lower(): tag for word, tag in pos_tags}
        
        skills = set()
        
        # Process multiword skills
        multiword_skills = [s for s in self.technical_skills if ' ' in s]
        for skill in multiword_skills:
            skill_lower = skill.lower()
            # Check for the complete phrase
            pattern = r'\b' + re.escape(skill_lower).replace(r'\ ', r'[\s\-_]+') + r'\b'
            if re.search(pattern, text_lower):
                skills.add(skill)
        
        # Process single word skills
        single_word_skills = [s for s in self.technical_skills if ' ' not in s]
        clean_tokens = nltk.word_tokenize(clean_text)
        
        for token in clean_tokens:
            token_lower = token.lower()
            for skill in single_word_skills:
                skill_lower = skill.lower()
                if token_lower == skill_lower:
                    # Use POS tag to validate
                    tag = pos_dict.get(skill_lower)
                    if tag:
                        # Accept nouns, proper nouns, adjectives, verbs
                        if tag.startswith(('NN', 'JJ', 'VB')):
                            skills.add(skill)
                        # Special case for programming languages or technical terms that might be tagged differently
                        elif len(skill) <= 2 or skill in self.skill_groups.get("programming_languages", []):
                            if self._has_programming_context(skill, text_lower):
                                skills.add(skill)
                    else:
                        # If we can't find the exact token in our POS dict (due to cleaning differences)
                        # Accept the skill if it's a known technical term
                        for group_name, group_skills in self.skill_groups.items():
                            if skill in group_skills:
                                skills.add(skill)
                                break
        
        # Process skill variants with the same approach
        for skill, variants in self.skill_variants.items():
            for variant in variants:
                variant_lower = variant.lower()
                
                # Handle multiword variants
                if ' ' in variant:
                    pattern = r'\b' + re.escape(variant_lower).replace(r'\ ', r'[\s\-_]+') + r'\b'
                    if re.search(pattern, text_lower):
                        skills.add(skill)  # Add the base skill
                        break
                # Handle single word variants with POS validation
                else:
                    for token in clean_tokens:
                        if token.lower() == variant_lower:
                            tag = pos_dict.get(variant_lower)
                            if tag and tag.startswith(('NN', 'JJ', 'VB')):
                                skills.add(skill)
                                break
                            elif len(variant) <= 2 or skill in self.skill_groups.get("programming_languages", []):
                                if self._has_programming_context(variant, text_lower):
                                    skills.add(skill)
                                    break
        
        # Check for contextual skills
        for context_phrase, related_skills in self.contextual_skills.items():
            if re.search(r'\b' + re.escape(context_phrase.lower()) + r'\b', text_lower):
                for related_skill in related_skills:
                    skills.add(related_skill)
        
        return skills

    def _has_programming_context(self, letter, text):
        """Check if a single letter or short term has programming language context."""
        text_lower = text.lower()
        if len(letter) <= 2:
            context_patterns = [
                r'programming\s+(?:language\s+)?' + re.escape(letter) + r'\b',
                r'language\s+' + re.escape(letter) + r'\b',
                r'\b' + re.escape(letter) + r'\s+programming\b',
                r'\b' + re.escape(letter) + r'\s+developer\b',
                r'\bprogramming\s+in\s+' + re.escape(letter) + r'\b',
                r'\bcode\s+(?:in\s+)?' + re.escape(letter) + r'\b',
                r'\b' + re.escape(letter) + r'\s+code\b',
                r'\b' + re.escape(letter) + r'\s+library\b',
                r'\b' + re.escape(letter) + r'\s+framework\b',
                r'proficient\s+(?:in\s+)?' + re.escape(letter) + r'\b',
                r'experience\s+(?:with|in)\s+' + re.escape(letter) + r'\b'
            ]
            return any(re.search(pattern, text_lower) for pattern in context_patterns)
        else:
            tech_context_patterns = [
                r'proficient\s+(?:in|with)\s+' + re.escape(letter.lower()) + r'\b',
                r'experience\s+(?:with|in)\s+' + re.escape(letter.lower()) + r'\b',
                r'knowledge\s+of\s+' + re.escape(letter.lower()) + r'\b',
                r'skills?\s+(?:in|with)?\s+' + re.escape(letter.lower()) + r'\b',
                r'\b' + re.escape(letter.lower()) + r'\s+(?:skills?|proficiency)\b'
            ]
            return any(re.search(pattern, text_lower) for pattern in tech_context_patterns)
    
    def extract_keywords_from_job_description(self, job_description):
        # Use the extract_skills method to get skills from the job description
        self.job_skills = self.extract_skills(job_description)
        return self.job_skills

    def get_dynamic_skill_weights(self, job_description, skill_ratings=None):
        # Fixed to use self.job_skills instead of re-extracting skills
        if not self.job_skills:
            self.job_skills = self.extract_keywords_from_job_description(job_description)
            
        if skill_ratings is None:
            skill_ratings = {skill: 7 for skill in self.job_skills}
            
        skill_weights = {skill: rating for skill, rating in skill_ratings.items()}
        group_lookup = {}
        
        for group, skills in self.skill_groups.items():
            for skill in skills:
                group_lookup[skill] = group
                
        rated_groups = {group_lookup.get(skill) for skill in skill_ratings if skill in group_lookup}
        rated_groups.discard(None)
        
        for skill in self.technical_skills:
            if skill not in skill_weights:
                group = group_lookup.get(skill)
                if group in rated_groups:
                    skill_weights[skill] = 4.5
                else:
                    skill_weights[skill] = 1.5
                    
        return skill_weights

    def process_resumes(self, resumes_df, job_description, skill_ratings=None):
        # Extract job skills if not already done
        if not self.job_skills:
            self.job_skills = self.extract_keywords_from_job_description(job_description)
            
        # Get skill weights using the job skills
        skill_weights = self.get_dynamic_skill_weights(job_description, skill_ratings)
        
        resumes_df['Processed'] = resumes_df['Resume'].apply(self.preprocess)
        resumes_df['Skills'] = resumes_df['Resume'].apply(self.extract_skills)
        resumes_df['Skill_Count'] = resumes_df['Skills'].apply(len)
        
        all_docs = resumes_df['Processed'].tolist() + [self.preprocess(job_description)]
        self.tfidf.fit(all_docs)
        tfidf_matrix = self.tfidf.transform(resumes_df['Processed'])
        job_vec = self.tfidf.transform([self.preprocess(job_description)])
        resumes_df['Similarity_Score'] = cosine_similarity(job_vec, tfidf_matrix).flatten()
        
        total_weight = sum(skill_weights.get(s, 0) for s in self.job_skills)
        if total_weight == 0:
            total_weight = 1
            
        resumes_df['Skill_Match_Percentage'] = resumes_df['Skills'].apply(
            lambda x: sum(skill_weights.get(s, 0) for s in x if s in self.job_skills)/total_weight * 100
        )
        
        resumes_df['Matching_Skills'] = resumes_df['Skills'].apply(
            lambda x: [s for s in x if s in self.job_skills]
        )
        
        resumes_df['Matching_Skill_Count'] = resumes_df['Matching_Skills'].apply(len)
        
        def extract_experience_years(text):
            text = str(text).lower()
            year_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:\+?\s*)?(?:years?|yrs?)', text)
            month_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:months?|mos?)', text)

            years = float(year_match.group(1)) if year_match else 0
            months = float(month_match.group(1)) if month_match else 0

            return round(years + months / 12, 2) if (years or months) else np.nan
            
        if 'Experience' in resumes_df.columns:
            exp_values = resumes_df['Experience'].fillna('').apply(lambda x: extract_experience_years(x) if not pd.isnull(x) else np.nan)
        else:
            exp_values = resumes_df['Resume'].apply(extract_experience_years)
            
        median_exp = exp_values[~np.isnan(exp_values)].median() if np.any(~np.isnan(exp_values)) else 0
        exp_values = exp_values.fillna(median_exp)
        resumes_df['Experience_Years'] = exp_values
        
        experience_array = resumes_df['Experience_Years'].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        resumes_df['Experience_Skill_Points'] = kmeans.fit_predict(experience_array)
        
        cluster_means = resumes_df.groupby('Experience_Skill_Points')['Experience_Years'].mean()
        sorted_clusters = cluster_means.sort_values().index.tolist()
        cluster_to_points = {cluster: point+1 for point, cluster in enumerate(sorted_clusters)}
        resumes_df['Experience_Skill_Points'] = resumes_df['Experience_Skill_Points'].map(cluster_to_points)
        
        exp_skill_points_norm = self.scaler.fit_transform(resumes_df[['Experience_Skill_Points']].values)
        resumes_df['Experience_Skill_Score'] = exp_skill_points_norm.flatten()
        
        norm_cols = ['Similarity_Score', 'Skill_Match_Percentage', 'Experience_Skill_Score']
        norm_df = resumes_df[norm_cols].copy()
        norm_df['Skill_Match_Percentage'] = norm_df['Skill_Match_Percentage'] / 100.0
        scores = self.scaler.fit_transform(norm_df)
        resumes_df['Combined_Score'] = scores[:,0]*0.4 + scores[:,1]*0.4 + scores[:,2]*0.2
        
        self.ranked_resumes = resumes_df.sort_values(by='Combined_Score', ascending=False).reset_index(drop=True)

    def show_top_candidates(self, top_n=10):
        return self.ranked_resumes.head(top_n)[['Name', 'Combined_Score', 'Skill_Match_Percentage', 'Similarity_Score']]

    def print_top_candidates(self, top_n=10):
        print(f"\n--- 🎯 TOP {top_n} CANDIDATES ---")
        for i, row in self.ranked_resumes.head(top_n).iterrows():
            print(f"\nRank {i+1}: {row['Name']}")
            print(f"Category: {row.get('Category', 'N/A')}")
            print(f"Similarity Score: {row['Similarity_Score']:.2f}")
            print(f"Skills Matched: {row['Skill_Match_Percentage']:.1f}%")
            print(f"Experience Level: {row.get('Experience_Level', 'N/A')} (Skill Points: {row.get('Experience_Skill_Points', 'N/A')})")
            print(f"Combined Score: {row['Combined_Score']:.2f}")
            matching_skills = set(row['Skills']).intersection(set(self.job_skills))
            print(f"Matching Skills: {', '.join(matching_skills) if matching_skills else 'None'}")

    def generate_visualizations(self, top_n=10):
        top_n = min(top_n, len(self.ranked_resumes))
        plot_images = {}

        def save_plot(key):
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_images[key] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

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

        # Ensure the PLOT directory exists
        os.makedirs(PLOT, exist_ok=True)

        # Save the plots
        plot_files = {
            'score_breakdown': os.path.join(PLOT, 'combined_ranking_breakdown.png'),
            'top_skills': os.path.join(PLOT, 'top_skills.png')
        }

        for key, path in plot_files.items():
            with open(path, "wb") as f:
                f.write(base64.b64decode(plot_images[key]))


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Screening System CLI")
    parser.add_argument("--resume_csv", type=str, default=DATASET_PATH, help="Path to resume dataset CSV")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top candidates to output")
    parser.add_argument("--skill_ratings_json", type=str, default=None, help="JSON file with skill ratings (optional)")
    parser.add_argument("--output_json", type=str, default=RANKING_RESULTS, help="Output JSON file for ranking results")
    parser.add_argument("--extract_skills", action="store_true", help="Extract skills from the job description and save to extracted_skills.json")
    parser.add_argument("--set_job_description", type=str, help="Set a new job description and exit")
    args = parser.parse_args()

    job_description = ""
    if os.path.exists(JOB_DESCRIPTION):
        with open(JOB_DESCRIPTION, "r", encoding='utf-8') as f:
            job_description = f.read()

    if not job_description:
        job_description = """We are looking for an experienced Data Scientist to join our team. The ideal candidate has
strong skills in Machine Learning, Python, and SQL. Experience with deep learning frameworks
like TensorFlow or PyTorch is preferred.

Requirements:
- Proficiency in Python and SQL
- Strong experience with ML frameworks (scikit-learn, TensorFlow, PyTorch)
- Experience with data visualization tools (Matplotlib, Seaborn, Tableau)
- Knowledge of big data technologies (Spark, Hadoop) is a plus
- Excellent communication skills to present findings to stakeholders
"""
    
    if args.extract_skills:
        screener = ResumeScreeningSystem()
        skills = screener.extract_keywords_from_job_description(job_description)
        output = {"skills_to_rate": list(skills)}
        print(json.dumps(output))
        exit()

    resumes_df = pd.read_csv(args.resume_csv,encoding='utf-8', encoding_errors='replace')
    skill_ratings = None
    if args.skill_ratings_json:
        with open(args.skill_ratings_json, "r") as f:
            skill_ratings = json.load(f)
    screener = ResumeScreeningSystem()
    screener.process_resumes(resumes_df, job_description, skill_ratings)
    screener.generate_visualizations(top_n=min(args.top_n, len(screener.ranked_resumes)))
    top_candidates = screener.show_top_candidates(top_n=min(args.top_n, len(screener.ranked_resumes)))
    output_data = {
    'top_candidates': [],
    'visualizations': {
        'score_breakdown': 'combined_ranking_breakdown.png',
        'top_skills': 'top_skills.png'
    }
    }
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
            "matching_skills": list(set(row['Skills']).intersection(set(screener.job_skills)))
        }
        if row.get('Experience_Skill_Score', 0) > 0:
            candidate["scores"]["experience_skill"] = round(float(row['Experience_Skill_Score']), 2)
        if row.get('Experience_Skill_Points', 0) > 0:
            candidate["experience_skill_points"] = row['Experience_Skill_Points']
        if row.get('Experience_Level', 0) > 0:
            candidate["experience_level"] = row['Experience_Level']
        output_data['top_candidates'].append(candidate)
   
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2) 