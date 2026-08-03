"""
Validates the full scoring pipeline against live job postings from the
Adzuna API, rather than the static test JD used elsewhere. Reuses the
same resume throughout and scores it against each live posting returned
for a search query.

Note: this validates that the pipeline works correctly on live data, but
the live-search flow itself is not yet wired into the Streamlit app,
users currently must paste a JD manually. See Future Improvements.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from parser import extract_text_from_pdf, split_resume_sections, split_jd_sections
from skill_extractor import build_keyword_index, build_flashtext_index, extract_esco_skills_fast, uri_to_label
from matcher import (compute_skill_gap, compute_skills_score, compute_experience_score,
                      compute_education_score, compute_composite_score)
from config import RESUME_HEADER_MAP, JD_KEYWORD_MAP
from job_api import search_jobs


def run_pipeline_on_jd(resume_text, resume_sections, jd_text, model, kp, label_map):
    """
    Run the full scoring pipeline (skills, experience, education,
    composite) for one resume against one job description, catching and
    logging errors per stage instead of crashing the whole batch if a
    single posting causes a failure.

    resume_text: full raw resume text.
    resume_sections: pre-split resume sections (computed once outside the
        loop, since the resume doesn't change between jobs).
    jd_text: raw job description text for this specific posting.
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
        jd_sections = split_jd_sections(jd_text, JD_KEYWORD_MAP)
        resume_experience_text = resume_sections.get("Experience", "") + "\n" + resume_sections.get("Projects", "")
        jd_responsibilities_text = jd_sections.get("responsibilities", "")
        experience_score, _ = compute_experience_score(resume_experience_text, jd_responsibilities_text, model)
    except Exception as e:
        errors.append(f"Experience stage failed: {e}")
        experience_score = 0.0
        jd_sections = {}

    try:
        resume_education_text = resume_sections.get("Education", "")
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
        "errors": errors
    }


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")

    esco_df = pd.read_csv("../data/ESCO/skills_en.csv")
    keyword_index = build_keyword_index(esco_df)
    kp = build_flashtext_index(keyword_index)
    label_map = uri_to_label(keyword_index)

    # Resume and its sections only need to be computed once, since they
    # don't change across the batch of job postings.
    resume_text = extract_text_from_pdf("../data/Testing/OtiohKonan_Resume_AI_June2026.pdf")
    resume_sections = split_resume_sections(resume_text, RESUME_HEADER_MAP)

    results_data = search_jobs("AI engineer")
    jobs = results_data.get("results", [])

    results = []
    for job in jobs:
        title = job.get("title", "unknown")
        jd_text = job.get("description", "")

        if not jd_text:
            print(f"{title}: no description, skipping")
            continue

        result = run_pipeline_on_jd(resume_text, resume_sections, jd_text, model, kp, label_map)
        result['title'] = title
        results.append(result)
        print(f"{title}: composite={result['composite_score']:.2f} band={result['band']} errors={result['errors']}")

    results_df = pd.DataFrame(results)
    print("\n--- Summary ---")
    print(f"Total tested: {len(results_df)}")
    print(f"Jobs with errors: {(results_df['errors'].apply(len) > 0).sum()}")
    print(f"Composite score range: {results_df['composite_score'].min():.2f} - {results_df['composite_score'].max():.2f}")

    # Inspect raw description lengths: revealed Adzuna's free-tier API
    # truncates every description to ~500 characters, explaining the low
    # scores above (see report for full discussion).
    for i, job in enumerate(jobs):
        desc = job.get("description", "")
        print(f"[{i}] {job.get('title')} — {len(desc)} chars")
        print(desc[:300])
        print("---")