#OLD: prompt-based approach (baseline, kept for report comparison)

# from llama_cpp import Llama

# model_path = "/Users/otiohkonan/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct-gguf/snapshots/a64113399c2f6b8ad3e11c394733a2ddadaa7f33/Phi-3-mini-4k-instruct-q4.gguf"

# llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1, verbose=False)

# def extract_skills(text, label="resume"):
#     prompt = f"""Extract the technical skills from this {label}. Output only the comma-separated list, no other text, no notes, no labels. Example: Python, SQL, Machine Learning. Do not use bullet points or dashes. 

# {label.upper()}:
# {text[:2000]}

# SKILLS:"""

#     response = llm.create_completion(prompt, max_tokens=400)
#     return response["choices"][0]["text"].strip()

# def clean_skills_output(skills_output):
#     cleaned = skills_output.split("- [Support]:")[0].strip()
#     cleaned = cleaned.split("[response]:")[0].strip()
#     if cleaned.startswith("- output:"):
#         cleaned = cleaned.replace("- output:", "", 1).strip()
#     if "," not in cleaned:
#         lines = [line.strip("- ").strip() for line in cleaned.split("\n") 
#                  if line.strip() and not line.strip().startswith("[") and not line.strip().startswith("output:")]
#         cleaned = ", ".join(lines)
#     else:
#         cleaned = cleaned.split("\n")[0]
#     return cleaned.strip()

# if __name__ == "__main__":
#     from parser import extract_text_from_pdf, load_job_description
    
#     resume_text = extract_text_from_pdf("../data/Testing/OtiohKonan_Resume_AI_June2026.pdf")
#     jd_text = load_job_description("../data/Testing/AI_Intern_job_description.txt")
    
#     print("Resume skills:")
#     print(clean_skills_output(extract_skills(resume_text, "resume")))
#     print("\nJob description skills:")
#     print(clean_skills_output(extract_skills(jd_text, "job description")))

import pandas as pd
from flashtext import KeywordProcessor
from parser import extract_text_from_pdf, load_job_description

def build_keyword_index(df):
    keyword_index = []
    for _, row in df.iterrows():
        alt_labels = row['altLabels'].split('\n') if pd.notna(row['altLabels']) else []
        labels = [row['preferredLabel']] + alt_labels
        keyword_index.append({'conceptUri': row['conceptUri'], 'labels': labels})
    return keyword_index

def build_flashtext_index(keyword_index):
    kp = KeywordProcessor()
    for entry in keyword_index:
        for label in entry['labels']:
            kp.add_keyword(label, entry['conceptUri'])
    return kp

def extract_esco_skills_fast(text, kp):
    return list(set(kp.extract_keywords(text)))

def uri_to_label(keyword_index):
    return {entry['conceptUri']: entry['labels'][0] for entry in keyword_index}

if __name__ == "__main__":
    df = pd.read_csv("../data/ESCO/skills_en.csv")
    keyword_index = build_keyword_index(df)
    kp = build_flashtext_index(keyword_index)
    
    resume_text = extract_text_from_pdf("../data/Testing/OtiohKonan_Resume_AI_June2026.pdf")
    jd_text = load_job_description("../data/Testing/AI_Intern_job_description.txt")

    resume_skills = extract_esco_skills_fast(resume_text, kp)
    jd_skills = extract_esco_skills_fast(jd_text, kp)

    label_map = uri_to_label(keyword_index)
    print("Resume skills:", [label_map[u] for u in resume_skills])
    print("JD skills:", [label_map[u] for u in jd_skills])
     