# OLD: prompt-based approach (baseline, kept for report comparison)

# from llama_cpp import Llama

# model_path = "/Users/otiohkonan/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct-gguf/snapshots/a64113399c2f6b8ad3e11c394733a2ddadaa7f33/Phi-3-mini-4k-instruct-q4.gguf"

# llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1, verbose=False)

# def extract_skills(text, label="resume"):
#     prompt = f"""Extract the technical skills from this {label}. Output only the comma-separated list, no other text, no notes, no labels. Example: Python, SQL, Machine Learning. Do not use bullet points or dashes.

# {label.upper()}:

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
#
#     resume_text = extract_text_from_pdf("../data/Testing/OtiohKonan_Resume_AI_June2026.pdf")
#     jd_text = load_job_description("../data/Testing/AI_Intern_job_description.txt")
#
#     print("Resume skills:")
#     print(clean_skills_output(extract_skills(resume_text, "resume")))
#     print("\nJob description skills:")
#     print(clean_skills_output(extract_skills(jd_text, "job description")))

import pandas as pd
from flashtext import KeywordProcessor
from parser import extract_text_from_pdf, load_job_description

# Manually curated fixes for known cases where a resume/JD phrase doesn't
# exactly match any ESCO label (e.g. word-form differences), but a clear
# corresponding ESCO skill exists. Maps a lowercase phrase to that skill's
# conceptUri, checked as a substring after the main FlashText pass.
SUPPLEMENTARY_SYNONYMS = {
    "debugging": "http://data.europa.eu/esco/skill/2522a6ce-3202-4ac8-9f5b-b9cb5a3a83a1"  # -> "debug software"
}


def build_keyword_index(df):
    """
    Build a list of ESCO skills, each with its conceptUri and full list of
    labels (preferredLabel plus all altLabels), used as the vocabulary for
    keyword-based skill extraction.

    df: ESCO skills dataframe (loaded from skills_en.csv), expected to
        have conceptUri, preferredLabel, and altLabels columns.
    """
    keyword_index = []
    for _, row in df.iterrows():
        alt_labels = row['altLabels'].split('\n') if pd.notna(row['altLabels']) else []
        labels = [row['preferredLabel']] + alt_labels
        keyword_index.append({'conceptUri': row['conceptUri'], 'labels': labels})
    return keyword_index


def build_flashtext_index(keyword_index):
    """
    Build a FlashText KeywordProcessor from the ESCO keyword index, mapping
    every skill label to its conceptUri for fast single-pass text scanning.

    keyword_index: list of skill entries from build_keyword_index.
    """
    kp = KeywordProcessor()
    for entry in keyword_index:
        for label in entry['labels']:
            kp.add_keyword(label, entry['conceptUri'])
    return kp


def extract_esco_skills_fast(text, kp):
    """
    Extract ESCO skill URIs mentioned in a piece of text (resume or JD),
    using FlashText for fast exact-phrase matching, then supplementing
    with a small manually curated synonym list to catch known gaps.

    text: raw resume or job description text to scan.
    kp: FlashText KeywordProcessor built by build_flashtext_index.
    """
    matched = set(kp.extract_keywords(text))

    # Check the small supplementary list for known phrasings FlashText
    # would otherwise miss due to exact-match limitations.
    text_lower = text.lower()
    for phrase, uri in SUPPLEMENTARY_SYNONYMS.items():
        if phrase in text_lower:
            matched.add(uri)

    return list(matched)


def uri_to_label(keyword_index):
    """
    Build a lookup dict from conceptUri to a skill's display label (its
    preferredLabel), used for turning extracted URIs back into readable
    skill names.

    keyword_index: list of skill entries from build_keyword_index.
    """
    return {entry['conceptUri']: entry['labels'][0] for entry in keyword_index}


if __name__ == "__main__":
    # Quick smoke test: extract skills from the sample resume/JD and print
    # the results as readable labels.
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