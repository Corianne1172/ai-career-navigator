# import chromadb
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from parser import extract_text_from_pdf, load_job_description, split_resume_sections, split_jd_sections
from skill_extractor import build_keyword_index, build_flashtext_index, uri_to_label, extract_esco_skills_fast
from llama_cpp import Llama
from config import RESUME_HEADER_MAP, JD_KEYWORD_MAP


def compute_skill_gap(resume_skill_uris, jd_skill_uris, uri_to_label, model, threshold=0.65):
    """
    Compare extracted resume and JD skills semantically to find which JD
    skills are matched by the resume and which are missing, using sentence
    embeddings rather than exact string matching so differently-worded but
    equivalent skills (e.g. "Python" vs "Python (computer programming)")
    still count as a match.

    resume_skill_uris: ESCO skill URIs extracted from the resume.
    jd_skill_uris: ESCO skill URIs extracted from the job description.
    uri_to_label: dict mapping a skill's conceptUri to its display label.
    model: SentenceTransformer used to embed skill labels.
    threshold: minimum cosine similarity for a JD skill to count as matched
        rather than a gap.
    """
    resume_skills = [uri_to_label[uri] for uri in resume_skill_uris if uri in uri_to_label]
    jd_skills = [uri_to_label[uri] for uri in jd_skill_uris if uri in uri_to_label]

    # Guard against either side having no extractable skills at all, which
    # would otherwise break the embedding/similarity step below.
    if not resume_skills or not jd_skills:
        return [], []

    resume_embeddings = model.encode(resume_skills)
    jd_embeddings = model.encode(jd_skills)
    skill_gap = []
    matched_skills = []

    # Compare every JD skill against every resume skill at once (instead of
    # looping pairwise) and take each JD skill's single best match.
    similarity_matrix = cos_sim(jd_embeddings, resume_embeddings)
    max_result = similarity_matrix.max(dim=1)
    max_similarities = max_result.values
    max_indices = max_result.indices

    for i, jd_skill in enumerate(jd_skills):
        if max_similarities[i] < threshold:
            skill_gap.append((jd_skill, max_similarities[i].item()))
        else:
            matched_skills.append((jd_skill, resume_skills[max_indices[i]], max_similarities[i].item()))
    return skill_gap, matched_skills


def compute_skills_score(skill_gap, matched_skills):
    """
    Compute the skills sub-score as a weighted average of every JD skill's
    best-match similarity (from compute_skill_gap), so skills close to the
    threshold still contribute partial credit instead of a hard 0 or 1.

    skill_gap: list of (skill, similarity) tuples for unmatched JD skills.
    matched_skills: list of (jd_skill, resume_skill, similarity) tuples for
        matched JD skills.
    """
    if not skill_gap and not matched_skills:
        return 0.0
    skills_score = sum(score for _, score in skill_gap) + sum(score for _, _, score in matched_skills)
    skills_score /= (len(skill_gap) + len(matched_skills))
    return skills_score


def compute_experience_score(resume_experience_text, jd_responsibilities_text, model, threshold=0.5):
    """
    Compute the experience sub-score by comparing resume experience/project
    bullet points against JD responsibility lines at the line level (rather
    than embedding each section as one whole-document blob), since
    line-level comparison captures more specific overlap than a single
    diluted document-level embedding would.

    resume_experience_text: resume Experience + Projects section text.
    jd_responsibilities_text: JD responsibilities section text.
    model: SentenceTransformer used to embed each line.
    threshold: currently unused in the score itself; kept for a possible
        future matched/gap breakdown similar to compute_skill_gap.
    """
    resume_lines = [line.strip() for line in resume_experience_text.splitlines() if line.strip()]
    jd_lines = [line.strip() for line in jd_responsibilities_text.splitlines() if line.strip()]

    if not resume_lines or not jd_lines:
        return 0.0, []

    resume_embeddings = model.encode(resume_lines)
    jd_embeddings = model.encode(jd_lines)

    similarity_matrix = cos_sim(jd_embeddings, resume_embeddings)
    max_similarities = similarity_matrix.max(dim=1).values

    experience_score = max_similarities.sum().item() / len(jd_lines)
    return experience_score, list(zip(jd_lines, max_similarities.tolist()))


def extract_gpa(resume_education_text, default_scale=4.0):
    """
    Extract a GPA value from resume education text and normalize it to a
    0-1 scale. Detects an explicit scale if stated (e.g. "GPA: 8.5/10"),
    falling back to default_scale (4.0) if the resume only states the GPA
    number without a scale.

    resume_education_text: resume Education section text.
    default_scale: GPA scale to assume when none is stated in the text.
    """
    match = re.search(r"GPA:\s*([\d.]+)\s*(?:/\s*([\d.]+))?", resume_education_text)
    if not match:
        return None
    gpa = float(match.group(1))
    scale = float(match.group(2)) if match.group(2) else default_scale
    return min(gpa / scale, 1.0)


def compute_education_score(resume_education_text, jd_education_text, model, gpa_bonus_weight=0.1):
    """
    Compute the education sub-score by comparing resume education content
    against JD education/qualification requirements at a fine-grained
    level, splitting comma-separated lines (e.g. a coursework list) into
    individual units so each course is compared separately rather than as
    one diluted blob. Adds a capped bonus for a reported GPA, since GPA is
    a numeric signal that can't be captured through text similarity alone.

    resume_education_text: resume Education section text.
    jd_education_text: JD education/qualifications section text.
    model: SentenceTransformer used to embed each line/chunk.
    gpa_bonus_weight: maximum bonus added to the score for a reported GPA
        (e.g. 0.1 means a perfect GPA can add up to +0.1).
    """
    resume_lines = []
    for line in resume_education_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            # Split comma-separated lines (e.g. coursework lists) into
            # individual comparable units instead of one long line.
            resume_lines.extend([chunk.strip() for chunk in line.split(",") if chunk.strip()])
        else:
            resume_lines.append(line)

    jd_lines = [line.strip() for line in jd_education_text.splitlines() if line.strip()]

    if not resume_lines or not jd_lines:
        return 0.0, []

    resume_embeddings = model.encode(resume_lines)
    jd_embeddings = model.encode(jd_lines)

    similarity_matrix = cos_sim(jd_embeddings, resume_embeddings)
    max_similarities = similarity_matrix.max(dim=1).values

    education_score = max_similarities.sum().item() / len(jd_lines)

    # Apply a capped GPA bonus on top of the semantic similarity score.
    gpa_normalized = extract_gpa(resume_education_text)
    if gpa_normalized is not None:
        education_score += gpa_bonus_weight * gpa_normalized
        education_score = min(education_score, 1.0)

    return education_score, list(zip(jd_lines, max_similarities.tolist()))


def compute_composite_score(skills_score, experience_score, education_score,
                              weights=(0.5, 0.3, 0.2)):
    """
    Combine the three sub-scores into a single weighted composite score and
    classify it into a band, following the score thresholds identified in
    the literature review (>=80% excellent, 55-79% good, <55% needs
    improvement).

    skills_score, experience_score, education_score: the three sub-scores.
    weights: (skills_weight, experience_weight, education_weight) tuple;
        skills is weighted highest as the most directly verifiable signal,
        education lowest as a baseline qualification rather than a
        differentiator (see report for full justification).
    """
    skills_weight, experience_weight, education_weight = weights
    composite = (skills_score * skills_weight +
                 experience_score * experience_weight +
                 education_score * education_weight)

    if composite >= 0.80:
        band = "Excellent"
    elif composite >= 0.55:
        band = "Good"
    else:
        band = "Needs Improvement"

    return composite, band


def generate_gap_report(skill_gap, llm, max_gaps=10):
    """
    Generate a short, practical-advice paragraph summarizing a candidate's
    missing skills using Phi-3 Mini, rather than presenting the raw gap
    list directly.

    skill_gap: list of (skill, similarity) tuples for unmatched JD skills,
        from compute_skill_gap.
    llm: loaded Llama model instance.
    max_gaps: maximum number of gap skills to include in the prompt.
    """
    gap_names = [name for name, score in skill_gap][:max_gaps]
    gap_list_text = ", ".join(gap_names)

    prompt = f"""Based on this list of missing skills for a job application, write a short, encouraging paragraph (3-4 sentences) explaining what the candidate should focus on next. Do not repeat the list verbatim, synthesize it into practical advice.

Missing skills: {gap_list_text}

Advice:"""

    response = llm.create_completion(prompt, max_tokens=200, stop=["<|assistant|>", "<|end|>", "<|user|>"])
    return response["choices"][0]["text"].strip()


def generate_interview_questions(matched_skills, skill_gap, llm, num_questions=5):
    """
    Generate interview questions grounded in a candidate's actual matched
    and gap skills using Phi-3 Mini, mixing questions that probe depth on
    matched skills with ones testing for hidden experience on gap skills.

    Output is regex-extracted and truncated to exactly num_questions items
    rather than trusting the model's raw output, since Phi-3 doesn't
    reliably stop at the requested count or format cleanly on its own;
    this guarantees a clean, deterministic result regardless of what
    extra text the model generates beyond the requested questions.

    matched_skills: list of (jd_skill, resume_skill, similarity) tuples.
    skill_gap: list of (skill, similarity) tuples for unmatched JD skills.
    llm: loaded Llama model instance.
    num_questions: number of questions to return.
    """
    matched_names = [name for name, _, _ in matched_skills][:5]
    gap_names = [name for name, _ in skill_gap][:5]

    prompt = f"""Generate {num_questions} interview questions for a candidate based on their skills and the job they're applying for.

Skills the candidate has that match the job: {", ".join(matched_names)}
Skills the job requires that the candidate's resume doesn't show: {", ".join(gap_names)}

Write a numbered list of {num_questions} interview questions. Mix technical questions about their matched skills with questions that probe whether they have hidden experience with the gap skills. Do not include explanations, just the numbered list.

Questions:"""

    response = llm.create_completion(prompt, max_tokens=700, stop=["<|assistant|>", "<|end|>", "<|user|>"])
    text = response["choices"][0]["text"].strip()

    # Extract each numbered item and keep only the first num_questions,
    # discarding anything the model generates past that point.
    matches = re.findall(r'\d+\.\s.*?(?=\n\d+\.|\Z)', text, re.DOTALL)
    trimmed = matches[:num_questions]
    return "\n\n".join(m.strip() for m in trimmed)


if __name__ == "__main__":
    # End-to-end smoke test: run the full scoring + generation pipeline
    # against the sample resume/JD and print every stage's output.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    resume_text = extract_text_from_pdf("../data/Testing/OtiohKonan_Resume_AI_June2026.pdf")
    job_description_text = load_job_description("../data/Testing/AI_Intern_job_description.txt")

    # ESCO skill extraction
    df = pd.read_csv("../data/ESCO/skills_en.csv")
    keyword_index = build_keyword_index(df)
    kp = build_flashtext_index(keyword_index)

    resume_skill_uris = extract_esco_skills_fast(resume_text, kp)
    jd_skill_uris = extract_esco_skills_fast(job_description_text, kp)

    label_map = uri_to_label(keyword_index)
    skill_gap, matched_skills = compute_skill_gap(resume_skill_uris, jd_skill_uris, label_map, model)
    skills_score = compute_skills_score(skill_gap, matched_skills)
    print("Skills score:", skills_score)

    # Section segmentation
    sections = split_resume_sections(resume_text, RESUME_HEADER_MAP)
    jd_sections = split_jd_sections(job_description_text, JD_KEYWORD_MAP)

    # Experience score
    resume_experience_text = sections.get("Experience", "") + "\n" + sections.get("Projects", "")
    jd_responsibilities_text = jd_sections.get("responsibilities", "")

    experience_score, experience_matches = compute_experience_score(resume_experience_text, jd_responsibilities_text, model)
    print("Experience score:", experience_score)
    print("Experience matches:", experience_matches)

    # Education score
    resume_education_text = sections.get("Education", "")
    jd_education_text = jd_sections.get("education", "")

    education_score, education_matches = compute_education_score(resume_education_text, jd_education_text, model)
    print("Education score:", education_score)
    print("Education matches:", education_matches)

    composite_score, band = compute_composite_score(skills_score, experience_score, education_score)
    print("Composite score:", composite_score)
    print("Band:", band)

    # Gap report and interview questions (Phi-3 Mini)
    model_path = "/Users/otiohkonan/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct-gguf/snapshots/a64113399c2f6b8ad3e11c394733a2ddadaa7f33/Phi-3-mini-4k-instruct-q4.gguf"
    llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1, verbose=False)

    gap_report = generate_gap_report(skill_gap, llm)
    print("Gap report:", gap_report)

    interview_questions = generate_interview_questions(matched_skills, skill_gap, llm)
    print("Interview questions:", interview_questions)

    # OLD: whole-document embedding baseline (Phase 1), kept for report
    # comparison against the composite scoring approach.
    # resume_embedding = model.encode(resume_text)
    # job_description_embedding = model.encode(job_description_text)
    # cosine_similarity = cos_sim(resume_embedding, job_description_embedding)
    # print(f"Cosine Similarity between resume and job description: {cosine_similarity.item()}")

    # client = chromadb.Client()
    # job_postings_collection = client.create_collection("job_postings")
    # job_postings_collection.add(ids=["job_posting_1"],
    #                             embeddings=[job_description_embedding.tolist()],
    #                             documents=[job_description_text])

    # most_similar_job = job_postings_collection.query(
    #     query_embeddings=[resume_embedding.tolist()],
    #     n_results=1
    # )
    # print(f"Most similar job posting: {most_similar_job}")