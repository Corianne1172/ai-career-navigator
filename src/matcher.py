# import chromadb
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from parser import extract_text_from_pdf, load_job_description, split_resume_sections, split_jd_sections 
from skill_extractor import build_keyword_index, build_flashtext_index, uri_to_label, extract_esco_skills_fast
from llama_cpp import Llama

def compute_skill_gap(resume_skill_uris, jd_skill_uris, uri_to_label, model, threshold=0.65):
    resume_skills = [uri_to_label[uri] for uri in resume_skill_uris if uri in uri_to_label]
    jd_skills = [uri_to_label[uri] for uri in jd_skill_uris if uri in uri_to_label]
    resume_embeddings = model.encode(resume_skills)
    jd_embeddings = model.encode(jd_skills)
    skill_gap = []
    matched_skills = []
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
    if not skill_gap and not matched_skills:
        return 0.0
    skills_score = sum(score for _, score in skill_gap) + sum(score for _, _, score in matched_skills)
    skills_score /= (len(skill_gap) + len(matched_skills))
    return skills_score

def compute_experience_score(resume_experience_text, jd_responsibilities_text, model, threshold=0.5):
    resume_lines = [line.strip() for line in resume_experience_text.splitlines() if line.strip()]
    jd_lines = [line.strip() for line in jd_responsibilities_text.splitlines() if line.strip()]

    if not resume_lines or not jd_lines:
        return 0.0

    resume_embeddings = model.encode(resume_lines)
    jd_embeddings = model.encode(jd_lines)

    similarity_matrix = cos_sim(jd_embeddings, resume_embeddings)
    max_similarities = similarity_matrix.max(dim=1).values

    experience_score = max_similarities.sum().item() / len(jd_lines)
    return experience_score, list(zip(jd_lines, max_similarities.tolist()))

def extract_gpa(resume_education_text, scale=4.0):
    match = re.search(r"GPA:\s*([\d.]+)", resume_education_text)
    if not match:
        return None
    gpa = float(match.group(1))
    return min(gpa / scale, 1.0)

def compute_education_score(resume_education_text, jd_education_text, model, gpa_bonus_weight=0.1):
    resume_lines = []
    for line in resume_education_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
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

    gpa_normalized = extract_gpa(resume_education_text)
    if gpa_normalized is not None:
        education_score += gpa_bonus_weight * gpa_normalized
        education_score = min(education_score, 1.0)

    return education_score, list(zip(jd_lines, max_similarities.tolist()))

def compute_composite_score(skills_score, experience_score, education_score, 
                              weights=(0.5, 0.3, 0.2)):
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
    gap_names = [name for name, score in skill_gap][:max_gaps]
    gap_list_text = ", ".join(gap_names)

    prompt = f"""Based on this list of missing skills for a job application, write a short, encouraging paragraph (3-4 sentences) explaining what the candidate should focus on next. Do not repeat the list verbatim, synthesize it into practical advice.

Missing skills: {gap_list_text}

Advice:"""

    response = llm.create_completion(prompt, max_tokens=200)
    return response["choices"][0]["text"].strip()

if __name__ == "__main__":
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
    resume_headers = ["Education", "Experience", "Projects", "Skills", "Honors, Achievements & Activities"]
    sections = split_resume_sections(resume_text, resume_headers)

    jd_keywords = {
        "qualifications": ["qualification", "requirement", "must have", "you have", "who you are"],
        "responsibilities": ["responsibilit", "essential function", "what you'll do", "duties"],
        "education": ["education", "degree"]
    }
    jd_sections = split_jd_sections(job_description_text, jd_keywords)

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
    
    
    model_path = "/Users/otiohkonan/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct-gguf/snapshots/a64113399c2f6b8ad3e11c394733a2ddadaa7f33/Phi-3-mini-4k-instruct-q4.gguf"
    llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1, verbose=False)

    gap_report = generate_gap_report(skill_gap, llm)
    print("Gap report:", gap_report)
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