"""
Shared configuration constants used across multiple files (matcher.py,
validate_pipeline.py, app.py, etc.), centralized here so they only need
to be updated in one place rather than staying in sync across several
copy-pasted definitions.
"""

# Maps a resume section category to its accepted header synonyms, used by
# split_resume_sections (parser.py). Compiled from a survey of 24 resume
# categories (~240 resumes) to cover real-world header phrasing variety
# beyond just "Education"/"Experience". "Summary" is included purely to
# prevent summary-section text from bleeding into whichever section
# precedes it; it isn't used in any sub-score.
RESUME_HEADER_MAP = {
    "Education": ["Education", "Education and Training"],
    "Experience": ["Experience", "Work Experience", "Professional Experience", "Work History"],
    "Projects": ["Projects"],
    "Skills": ["Skills", "Summary of Skills", "Core Qualifications", "Qualifications", "Technical Skills", "Highlights", "Skill Highlights", "Skills Used", "Core Strengths"],
    "Honors": ["Honors, Achievements & Activities", "Activities", "Honors and Accomplishments", "Accomplishments", "Core Accomplishments"],
    "Summary": ["Summary", "Professional Summary", "Executive Profile", "Career Overview", "Career Focus", "Executive Summary", "Profile", "Professional Profile"]
}

# Maps a job description section category to keywords that identify its
# header, used by split_jd_sections (parser.py). JD headers vary too
# widely across postings for exact matching, so this uses substring
# keyword detection on short candidate lines instead.
JD_KEYWORD_MAP = {
    "qualifications": ["qualification", "requirement", "must have", "you have", "who you are"],
    "responsibilities": ["responsibilit", "essential function", "what you'll do", "duties"],
    "education": ["education", "degree"]
}