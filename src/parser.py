import pdfplumber
from config import RESUME_HEADER_MAP, JD_KEYWORD_MAP


def extract_text_from_pdf(pdf_path):
    """
    Extract plain text from a PDF resume using pdfplumber, concatenating
    text from every page into a single string. Pages that fail to extract
    text (e.g. scanned image PDFs) contribute an empty string rather than
    raising an error.

    pdf_path: path to the PDF file to read.
    """
    with pdfplumber.open(pdf_path) as pdf:
        pdf_parsed = ""
        for page in pdf.pages if pdf.pages else []:
            text = page.extract_text()
            pdf_parsed += text if text else ""
    return pdf_parsed


def split_resume_sections(text, header_map):
    """
    Split resume text into labeled sections (Education, Experience, etc.)
    by scanning for lines that exactly match a known section header.
    Used to isolate specific parts of a resume for targeted sub-scoring
    later in the pipeline.

    text: raw resume text (e.g. from extract_text_from_pdf).
    header_map: dict mapping a category name (e.g. "Experience") to a
        list of accepted header synonyms (e.g. ["Experience", "Work
        Experience"]), since resumes phrase the same section differently.
    """
    sections = {category: [] for category in header_map}
    current_category = None
    matched_categories_seen = set()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Check if this line exactly matches a known header synonym.
        matched_category = None
        for category, synonyms in header_map.items():
            if any(synonym.lower() == line.lower() for synonym in synonyms):
                matched_category = category
                break

        if matched_category:
            if matched_category not in matched_categories_seen:
                # First time seeing this header: start collecting into it.
                current_category = matched_category
                matched_categories_seen.add(matched_category)
            else:
                # Repeated header: ignore, stop collecting until next new one.
                current_category = None
        elif current_category:
            # Not a header: append to whichever section is currently active.
            sections[current_category].append(line)

    return {category: "\n".join(lines) for category, lines in sections.items()}


def load_job_description(job_description_path):
    """
    Load raw job description text from a plain text file.

    job_description_path: path to the .txt file containing the JD.
    """
    with open(job_description_path, 'r') as file:
        job_description = file.read()
    return job_description


def split_jd_sections(text, keyword_map, max_header_words=8, stop_markers=None):
    """
    Split job description text into labeled sections (qualifications,
    responsibilities, education) by scanning for short lines containing
    category-relevant keywords. Unlike resumes, JD headers vary too
    widely across postings for exact matching, so keyword-based detection
    is used instead.

    text: raw job description text.
    keyword_map: dict mapping a category name (e.g. "qualifications") to
        a list of keywords that identify a header for that category
        (e.g. ["qualification", "requirement"]).
    max_header_words: maximum word count for a line to be considered a
        possible header, filters out long sentences that merely contain
        a keyword.
    stop_markers: phrases (e.g. "equal opportunity employer") that signal
        the end of relevant JD content, so trailing boilerplate isn't
        absorbed into the last matched section. Defaults to common EEO/
        benefits phrasing if not provided.
    """
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
            # Trailing boilerplate (EEO, benefits): stop collecting into
            # any section from here on.
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
            # New header line found: replace active categories. A single
            # line can match more than one category (e.g. "Education &
            # Requirements").
            active_categories = matched_categories
        elif active_categories:
            # Not a header: append to all currently active categories.
            for category in active_categories:
                sections[category].append(line)

    return {category: "\n".join(lines) for category, lines in sections.items()}


if __name__ == "__main__":
    # Quick smoke test against the sample resume/JD.
    print("Resume Sections:\n")
    my_resume = extract_text_from_pdf("../data/Testing/OtiohKonan_Resume_AI_June2026.pdf")
    sections = split_resume_sections(my_resume, RESUME_HEADER_MAP)
    for header, content in sections.items():
        print(f"{header}:")
        print(content)
        print("\n")

    print("\n\nJob Description Sections:\n")
    job_description = load_job_description("../data/Testing/AI_Intern_job_description.txt")
    jd_sections = split_jd_sections(job_description, JD_KEYWORD_MAP)
    for category, content in jd_sections.items():
        print(f"{category}:")
        print(content)
        print("\n")