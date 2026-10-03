"""
Streamlit interface for the AI Career Navigator. Ties together every
module built in the project (skill extraction, semantic matching,
composite scoring, gap report and interview question generation, and
course recommendations) into a single interactive app: the user uploads
a resume PDF and pastes a job description, and receives a full match
report on one page.
"""

import os

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer

from parser import extract_text_from_pdf, split_resume_sections, split_jd_sections
from skill_extractor import build_keyword_index, build_flashtext_index, extract_esco_skills_fast, uri_to_label
from matcher import (compute_skill_gap, compute_skills_score, compute_experience_score,
                      compute_education_score, compute_composite_score, generate_gap_report,
                      generate_interview_questions, load_llm)
from course_recommender import build_course_index, recommend_courses
from config import RESUME_HEADER_MAP, JD_KEYWORD_MAP

from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")


# The functions below are cached so expensive setup (loading the sentence
# transformer, loading Phi-3 Mini, building the ~13,000-skill ESCO index,
# building the course index) only happens once per app session, rather
# than being repeated on every button click.

@st.cache_resource
def load_model():
    #Load and cache the sentence-transformer model used for all embedding-based matching.
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_secret(key):
    
    # Retrieve a secret/API key, checking Streamlit Cloud's secrets store
    # first (used when deployed), falling back to environment variables
    # (used for local development via .env).
    
    try:
        if key in st.secrets:
            return st.secrets[key]
    except FileNotFoundError:
        pass
    return os.getenv(key)


@st.cache_resource
def get_llm():
    return load_llm(get_secret("GROQ_API_KEY"))


@st.cache_data
def load_esco_index():
    """Load and cache the ESCO skills dataset as a keyword index for skill extraction."""
    df = pd.read_csv("../data/ESCO/skills_en.csv")
    keyword_index = build_keyword_index(df)
    return keyword_index


@st.cache_data
def load_course_index():
    """Load and cache the course dataset used for course recommendations."""
    df = pd.read_csv("../data/Online_Courses.csv")
    return build_course_index(df)


st.title("AI Career Navigator")

resume_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
jd_text = st.text_area("Paste the job description")

if st.button("Analyze") and resume_file and jd_text:
    with st.spinner("Analyzing..."):
        # Load all cached resources for this run.
        model = load_model()
        llm = get_llm()
        keyword_index = load_esco_index()
        kp = build_flashtext_index(keyword_index)
        label_map = uri_to_label(keyword_index)
        course_index = load_course_index()

        # pdfplumber can read directly from Streamlit's uploaded file
        # object, no need to save it to disk first.
        resume_text = extract_text_from_pdf(resume_file)

        # Skill extraction + semantic matching -> skills sub-score.
        resume_skill_uris = extract_esco_skills_fast(resume_text, kp)
        jd_skill_uris = extract_esco_skills_fast(jd_text, kp)
        skill_gap, matched_skills = compute_skill_gap(resume_skill_uris, jd_skill_uris, label_map, model)
        skills_score = compute_skills_score(skill_gap, matched_skills)

        # Section segmentation, needed for the experience and education
        # sub-scores, which compare specific resume/JD sections rather
        # than the full documents.
        sections = split_resume_sections(resume_text, RESUME_HEADER_MAP)
        jd_sections = split_jd_sections(jd_text, JD_KEYWORD_MAP)

        resume_experience_text = sections.get("Experience", "") + "\n" + sections.get("Projects", "")
        jd_responsibilities_text = jd_sections.get("responsibilities", "")
        experience_score, _ = compute_experience_score(resume_experience_text, jd_responsibilities_text, model)

        resume_education_text = sections.get("Education", "")
        jd_education_text = jd_sections.get("education", "")
        education_score, _ = compute_education_score(resume_education_text, jd_education_text, model)

        composite_score, band = compute_composite_score(skills_score, experience_score, education_score)

        # Phi-3-generated gap report and interview questions, plus course
        # recommendations for the identified skill gaps.
        gap_report = generate_gap_report(skill_gap, llm)
        interview_questions = generate_interview_questions(matched_skills, skill_gap, llm)
        recommendations = recommend_courses(skill_gap, course_index, model)

    # Render results.
    st.header(f"Match Score: {composite_score:.0%} ({band})")
    st.write(f"Skills: {skills_score:.0%} | Experience: {experience_score:.0%} | Education: {education_score:.0%}")

    st.subheader("Gap Analysis")
    st.write(gap_report)

    st.subheader("Interview Prep")
    st.write(interview_questions)

    st.subheader("Recommended Courses")
    for gap_skill, courses in recommendations.items():
        st.write(f"**{gap_skill}**")
        for title, url, score in courses:
            st.write(f"- [{title}]({url})")