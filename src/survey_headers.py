import os

resume_root = "../data/archive/data/data"
categories = [d for d in os.listdir(resume_root) if os.path.isdir(os.path.join(resume_root, d))]
print(categories)
print(len(categories))

from parser import extract_text_from_pdf

for category in categories:
    category_path = os.path.join(resume_root, category)
    pdf_files = [f for f in os.listdir(category_path) if f.endswith(".pdf")][:10]

    print(f"\n=== {category} ===")
    for filename in pdf_files:
        filepath = os.path.join(category_path, filename)
        try:
            text = extract_text_from_pdf(filepath)
        except Exception as e:
            print(f"{filename}: extraction failed: {e}")
            continue

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        print(f"-- {filename} (first 15 non-empty lines) --")
        for line in lines[:15]:
            print(line)