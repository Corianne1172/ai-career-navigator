import pdfplumber

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        pdf_parsed = ""
        for page in pdf.pages if pdf.pages else []:
            text = page.extract_text()
            pdf_parsed += text if text else ""
    return pdf_parsed
    
my_resume = extract_text_from_pdf("../data/OtiohKonan_Resume_AI_June2026.pdf")
print(my_resume)

def load_job_description(job_description_path):
    with open(job_description_path, 'r') as file:
        job_description = file.read()
    return job_description

job_description = load_job_description("../data/AI_Intern_job_description.txt")
print(job_description)