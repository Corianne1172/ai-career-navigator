import pdfplumber

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        pdf_parsed = ""
        for page in pdf.pages if pdf.pages else []:
            text = page.extract_text()
            pdf_parsed += text if text else ""
    return pdf_parsed
    
    
def split_resume_sections(text, headers):
    sections = {}
    current_header = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(header.lower() == line.lower() for header in headers):
            matched_header = next((header for header in headers if header.lower() == line.lower()), None)
            if matched_header not in sections:
                current_header = matched_header
                sections[current_header] = []
            else:
                current_header = None
        elif current_header:
            sections[current_header].append(line)
    return {header: "\n".join(lines) for header, lines in sections.items()}

def load_job_description(job_description_path):
    with open(job_description_path, 'r') as file:
        job_description = file.read()
    return job_description

if __name__ == "__main__":
    my_resume = extract_text_from_pdf("../data/Testing/OtiohKonan_Resume_AI_June2026.pdf")
    headers = ["Education", "Experience", "Projects", "Skills", "Honors, Achievements & Activities"]
    sections = split_resume_sections(my_resume, headers)
    for header, content in sections.items():
        print(f"{header}:")
        print(content)
    job_description = load_job_description("../data/Testing/AI_Intern_job_description.txt")
    print(job_description)