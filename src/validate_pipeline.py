"""
Validates the full scoring pipeline against real PDF resumes from the
INFORMATION-TECHNOLOGY category of the Kaggle resume dataset, using the
project's own PDF extraction pipeline rather than the dataset's pre-
extracted CSV text (which was found to use a lossier extraction method
that broke section detection entirely). Resume varies across a batch;
job description stays constant.
"""

import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from parser import load_job_description, split_resume_sections, split_jd_sections, extract_text_from_pdf
from skill_extractor import build_keyword_index, build_flashtext_index, extract_esco_skills_fast, uri_to_label
from matcher import (compute_skill_gap, compute_skills_score, compute_experience_score,
                      compute_education_score, compute_composite_score)
from config import RESUME_HEADER_MAP, JD_KEYWORD_MAP


def run_pipeline_on_resume(resume_text, jd_text, jd_sections, model, kp, label_map):
    """
    Run the full scoring pipeline (skills, experience, education,
    composite) for one resume against the fixed job description, catching
    and logging errors per stage instead of crashing the whole batch if a
    single resume causes a failure.

    resume_text: full raw resume text for this specific resume.
    jd_text: full raw job description text (constant across the batch).
    jd_sections: pre-split JD sections (computed once outside the loop).
    model: SentenceTransformer used for all embedding-based comparisons.
    kp: FlashText KeywordProcessor for ESCO skill extraction.
    label_map: dict mapping ESCO conceptUri to display label.
    """
    errors = []

    try:
        resume_skill_uris = extract_esco_skills_fast(resume_text, kp)
        jd_skill_uris = extract_esco_skills_fast(jd_text, kp)
        skill_gap, matched_skills = compute_skill_gap(resume_skill_uris, jd_skill_uris, label_map, model)
        skills_score = compute_skills_score(skill_gap, matched_skills)
    except Exception as e:
        errors.append(f"Skills stage failed: {e}")
        skills_score, skill_gap, matched_skills = 0.0, [], []

    try:
        sections = split_resume_sections(resume_text, RESUME_HEADER_MAP)

        resume_experience_text = sections.get("Experience", "") + "\n" + sections.get("Projects", "")
        jd_responsibilities_text = jd_sections.get("responsibilities", "")
        experience_score, _ = compute_experience_score(resume_experience_text, jd_responsibilities_text, model)
    except Exception as e:
        errors.append(f"Experience stage failed: {e}")
        experience_score = 0.0
        sections = {}

    try:
        resume_education_text = sections.get("Education", "")
        jd_education_text = jd_sections.get("education", "")
        education_score, _ = compute_education_score(resume_education_text, jd_education_text, model)
    except Exception as e:
        errors.append(f"Education stage failed: {e}")
        education_score = 0.0

    try:
        composite_score, band = compute_composite_score(skills_score, experience_score, education_score)
    except Exception as e:
        errors.append(f"Composite stage failed: {e}")
        composite_score, band = 0.0, "Error"

    return {
        "skills_score": skills_score,
        "experience_score": experience_score,
        "education_score": education_score,
        "composite_score": composite_score,
        "band": band,
        "num_resume_skills": len(resume_skill_uris) if 'resume_skill_uris' in dir() else 0,
        "errors": errors
    }


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")

    esco_df = pd.read_csv("../data/ESCO/skills_en.csv")
    keyword_index = build_keyword_index(esco_df)
    kp = build_flashtext_index(keyword_index)
    label_map = uri_to_label(keyword_index)

    # Job description stays constant across the batch, so it's split once
    # outside the loop rather than repeated per resume.
    jd_text = load_job_description("../data/Testing/AI_Intern_job_description.txt")
    jd_sections = split_jd_sections(jd_text, JD_KEYWORD_MAP)

    resume_folder = "../data/archive/data/data/INFORMATION-TECHNOLOGY"
    resume_files = [f for f in os.listdir(resume_folder) if f.endswith(".pdf")][:15]

    results = []
    for filename in resume_files:
        filepath = os.path.join(resume_folder, filename)
        try:
            resume_text = extract_text_from_pdf(filepath)
        except Exception as e:
            print(f"{filename}: PDF extraction failed: {e}")
            continue

        result = run_pipeline_on_resume(resume_text, jd_text, jd_sections, model, kp, label_map)
        result['filename'] = filename
        results.append(result)
        print(f"{filename}: composite={result['composite_score']:.2f} band={result['band']} errors={result['errors']}")

    # Spot-check two specific resumes that showed a large score gap
    # despite similar skills sub-scores, used to diagnose the section
    # segmentation header-mismatch issue in Week 7.
    for r in results:
        if r['filename'] in ['40018190.pdf', '52246737.pdf']:
            print(f"\n--- {r['filename']} ---")
            print(f"Skills: {r['skills_score']:.2f} | Experience: {r['experience_score']:.2f} | Education: {r['education_score']:.2f}")

    results_df = pd.DataFrame(results)
    print("\n--- Summary ---")
    print(f"Total tested: {len(results_df)}")
    print(f"Resumes with errors: {(results_df['errors'].apply(len) > 0).sum()}")
    print(f"Composite score range: {results_df['composite_score'].min():.2f} - {results_df['composite_score'].max():.2f}")