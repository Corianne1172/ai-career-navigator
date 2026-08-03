import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def build_course_index(df):
    """
    Build a list of courses with their title, URL, and a clean list of
    associated skills, filtered to only rows that actually have skill
    data (many rows in this multi-platform dataset don't).

    df: courses dataframe (loaded from Online_Courses.csv), expected to
        have Title, URL, and Skills columns.
    """
    course_index = []
    df_with_skills = df[df['Skills'].notna()]
    for _, row in df_with_skills.iterrows():
        skills = [s.strip() for s in row['Skills'].split(',') if s.strip()]
        course_index.append({
            'title': row['Title'],
            'url': row['URL'],
            'skills': skills
        })
    return course_index


def recommend_courses(skill_gap, course_index, model, top_n=2, min_score=0.5):
    """
    Recommend courses for each skill gap by comparing the gap skill against
    every course's combined skill list via cosine similarity, returning
    the top matches per gap skill above a minimum confidence threshold.

    A minimum score filter is applied because some gap skills come from
    noisy ESCO extraction (short or generic labels causing false-positive
    matches), which otherwise produces confidently wrong course
    recommendations for skills that were never genuinely required.

    skill_gap: list of (skill, similarity) tuples from compute_skill_gap.
    course_index: list of course entries from build_course_index.
    model: SentenceTransformer used to embed gap skills and course skills.
    top_n: number of course recommendations to return per gap skill.
    min_score: minimum similarity for a course to be recommended; below
        this, a gap skill gets no recommendations rather than a weak one.
    """
    gap_names = [name for name, _ in skill_gap]
    course_texts = [", ".join(course['skills']) for course in course_index]

    gap_embeddings = model.encode(gap_names)
    course_embeddings = model.encode(course_texts)

    similarity_matrix = cos_sim(gap_embeddings, course_embeddings)

    recommendations = {}
    for i, gap_skill in enumerate(gap_names):
        # Get the top_n highest-scoring courses for this gap skill, then
        # filter out any that fall below min_score.
        top_indices = similarity_matrix[i].argsort(descending=True)[:top_n]
        matches = [
            (course_index[idx]['title'], course_index[idx]['url'], similarity_matrix[i][idx].item())
            for idx in top_indices
            if similarity_matrix[i][idx].item() >= min_score
        ]
        if matches:
            recommendations[gap_skill] = matches
    return recommendations


if __name__ == "__main__":
    # End-to-end smoke test: extract skill gaps from the sample resume/JD,
    # then recommend courses for each gap.
    from parser import extract_text_from_pdf, load_job_description
    from skill_extractor import build_keyword_index, build_flashtext_index, extract_esco_skills_fast, uri_to_label
    from matcher import compute_skill_gap

    model = SentenceTransformer("all-MiniLM-L6-v2")
    resume_text = extract_text_from_pdf("../data/Testing/OtiohKonan_Resume_AI_June2026.pdf")
    job_description_text = load_job_description("../data/Testing/AI_Intern_job_description.txt")

    esco_df = pd.read_csv("../data/ESCO/skills_en.csv")
    keyword_index = build_keyword_index(esco_df)
    kp = build_flashtext_index(keyword_index)

    resume_skill_uris = extract_esco_skills_fast(resume_text, kp)
    jd_skill_uris = extract_esco_skills_fast(job_description_text, kp)
    label_map = uri_to_label(keyword_index)

    skill_gap, matched_skills = compute_skill_gap(resume_skill_uris, jd_skill_uris, label_map, model)

    course_df = pd.read_csv("../data/Online_Courses.csv")
    course_index = build_course_index(course_df)

    recommendations = recommend_courses(skill_gap, course_index, model)
    for gap_skill, courses in recommendations.items():
        print(f"\nGap skill: {gap_skill}")
        for title, url, score in courses:
            print(f"  - {title} (score: {score:.3f}) - {url}")