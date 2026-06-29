import chromadb
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from parser import extract_text_from_pdf, load_job_description

if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    resume_text = extract_text_from_pdf("../data/OtiohKonan_Resume_AI_June2026.pdf")
    job_description_text = load_job_description("../data/AI_Intern_job_description.txt")

    resume_embedding = model.encode(resume_text)
    job_description_embedding = model.encode(job_description_text)
    cosine_similarity = cos_sim(resume_embedding, job_description_embedding)
    print(f"Cosine Similarity between resume and job description: {cosine_similarity.item()}")

    client = chromadb.Client()
    job_postings_collection = client.create_collection("job_postings")
    job_postings_collection.add(ids=["job_posting_1"], 
                                embeddings=[job_description_embedding.tolist()],
                                documents=[job_description_text])

    most_similar_job = job_postings_collection.query(
        query_embeddings=[resume_embedding.tolist()],
        n_results=1
    )
    print(f"Most similar job posting: {most_similar_job}")