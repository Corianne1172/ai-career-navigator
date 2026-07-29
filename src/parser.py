import pdfplumber

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        pdf_parsed = ""
        for page in pdf.pages if pdf.pages else []:
            text = page.extract_text()
            pdf_parsed += text if text else ""
    return pdf_parsed
    
    
def split_resume_sections(text, header_map):
    sections = {category: [] for category in header_map}
    current_category = None
    matched_categories_seen = set()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        matched_category = None
        for category, synonyms in header_map.items():
            if any(synonym.lower() == line.lower() for synonym in synonyms):
                matched_category = category
                break

        if matched_category:
            if matched_category not in matched_categories_seen:
                current_category = matched_category
                matched_categories_seen.add(matched_category)
            else:
                current_category = None
        elif current_category:
            sections[current_category].append(line)

    return {category: "\n".join(lines) for category, lines in sections.items()}

def load_job_description(job_description_path):
    with open(job_description_path, 'r') as file:
        job_description = file.read()
    return job_description

def split_jd_sections(text, keyword_map, max_header_words=8, stop_markers=None):
    if stop_markers is None:
        stop_markers = ["equal opportunity employer", "benefits found in job post"]

    sections = {category: [] for category in keyword_map}
    active_categories = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        line_lower = line.lower()

        if any(marker in line_lower for marker in stop_markers):
            active_categories = []
            continue

        word_count = len(line.split())
        is_short = word_count <= max_header_words

        matched_categories = []
        if is_short:
            for category, keywords in keyword_map.items():
                if any(keyword in line_lower for keyword in keywords):
                    matched_categories.append(category)

        if matched_categories:
            active_categories = matched_categories
        elif active_categories:
            for category in active_categories:
                sections[category].append(line)

    return {category: "\n".join(lines) for category, lines in sections.items()}

if __name__ == "__main__":
    print("Resume Sections:\n")
    my_resume = extract_text_from_pdf("../data/Testing/OtiohKonan_Resume_AI_June2026.pdf")
    header_map = {
        "Education": ["Education", "Education and Training"],
        "Experience": ["Experience", "Work Experience", "Professional Experience"],
        "Projects": ["Projects"],
        "Skills": ["Skills", "Summary of Skills"],
        "Honors": ["Honors, Achievements & Activities", "Activities", "Honors and Accomplishments"]
    }
    sections = split_resume_sections(my_resume, header_map)
    for header, content in sections.items():
        print(f"{header}:")
        print(content)
        print("\n")
        
    print("\n\nJob Description Sections:\n")
    job_description = load_job_description("../data/Testing/AI_Intern_job_description.txt")
    jd_keywords = {
        "qualifications": ["qualification", "requirement", "must have", "you have", "who you are"],
        "responsibilities": ["responsibilit", "essential function", "what you'll do", "duties"],
        "education": ["education", "degree"]
    }
    jd_sections = split_jd_sections(job_description, jd_keywords)
    for category, content in jd_sections.items():
        print(f"{category}:")
        print(content)
        print("\n")