# import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from parser import extract_text_from_pdf, load_job_description
from skill_extractor import build_keyword_index, build_flashtext_index, uri_to_label, extract_esco_skills_fast

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
            skill_gap.append(jd_skill)
        else:
            matched_skills.append((jd_skill, resume_skills[max_indices[i]]))
    return skill_gap, matched_skills
    

if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    resume_text = extract_text_from_pdf("../data/Testing/OtiohKonan_Resume_AI_June2026.pdf")
    job_description_text = load_job_description("../data/Testing/AI_Intern_job_description.txt")

    df = pd.read_csv("../data/ESCO/skills_en.csv")
    keyword_index = build_keyword_index(df)
    kp = build_flashtext_index(keyword_index)
    
    resume_skill_uris = extract_esco_skills_fast(resume_text, kp)
    jd_skill_uris = extract_esco_skills_fast(job_description_text, kp)
    
    label_map = uri_to_label(keyword_index)
    skill_gap, matched_skills = compute_skill_gap(resume_skill_uris, jd_skill_uris, label_map, model)
    print("Skill gap:", skill_gap)
    print("Matched skills:", matched_skills)
    
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