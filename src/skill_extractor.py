from llama_cpp import Llama

model_path = "/Users/otiohkonan/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct-gguf/snapshots/a64113399c2f6b8ad3e11c394733a2ddadaa7f33/Phi-3-mini-4k-instruct-q4.gguf"

llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1, verbose=False)

def extract_skills(text, label="resume"):
    prompt = f"""Extract the technical skills from this {label}. List them as a comma-separated list, nothing else.

{label.upper()}:
{text[:2000]}

SKILLS:"""

    response = llm.create_completion(prompt, max_tokens=400)
    return response["choices"][0]["text"].strip()

if __name__ == "__main__":
    from parser import extract_text_from_pdf, load_job_description
    
    resume_text = extract_text_from_pdf("../data/OtiohKonan_Resume_AI_June2026.pdf")
    jd_text = load_job_description("../data/AI_Intern_job_description.txt")
    
    print("Resume skills:")
    print(extract_skills(resume_text, "resume"))
    print("\nJob description skills:")
    print(extract_skills(jd_text, "job description"))