# AI Career Navigator

An AI-powered tool that matches student resumes to job postings, identifies skill gaps, and generates tailored interview prep. Built for CS 497 (Special Projects) at Illinois Tech.

**Live demo:** https://ai-career-navigator-djx7gzrrwru9rrc833mmzt.streamlit.app

It differs from existing resume-matching tools through:
- Gap analysis identifying what's missing from a student's profile for their target roles, with concrete next steps
- Tailored interview prep questions generated from the student's specific resume and target job description
- Course recommendations tied directly to identified skill gaps
- (Stretch goal, not implemented) H-1B sponsorship likelihood prediction

## Status

Complete and deployed: resume/JD parsing and section segmentation, ESCO-based skill extraction, semantic skill gap matching, composite scoring (skills, experience, education sub-scores), LLM-generated gap analysis and interview questions (via Groq API), course recommendations, and a full Streamlit interface tying every feature together.

Validated against real PDF resumes (Kaggle resume dataset, multiple categories) and live job postings (Adzuna API). See the final report for detailed findings, including known limitations.

## Setup

```bash
git clone https://github.com/Corianne1172/ai-career-navigator
cd ai-career-navigator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

ESCO data is not tracked in this repo due to file size. Download the CSV bundle (English) from the [ESCO download page](https://esco.ec.europa.eu/en/use-esco/download) and place `skills_en.csv` in `data/ESCO/`.

Create a `.env` file in the project root with a Groq API key (used for gap analysis and interview question generation) and, optionally, Adzuna API credentials (used for live job search):
```
GROQ_API_KEY=your_groq_api_key
ADZUNA_APP_ID=your_app_id
ADZUNA_APP_KEY=your_app_key
```

A free Groq API key can be created at [console.groq.com](https://console.groq.com).

## Running the app

```bash
cd src
streamlit run app.py
```

Upload a resume PDF, paste a job description, and click Analyze to see the full match report.

## Project structure

```
ai-career-navigator/
├── src/
│   ├── app.py                  # Streamlit interface
│   ├── parser.py                # resume/JD text extraction and section segmentation
│   ├── skill_extractor.py       # ESCO-based skill extraction
│   ├── matcher.py               # semantic matching, composite scoring, gap report, interview questions
│   ├── course_recommender.py    # course recommendations for skill gaps
│   ├── job_api.py               # Adzuna live job search integration
│   ├── config.py                # shared header/keyword mappings
│   ├── validate_pipeline.py     # pipeline validation against real resume dataset
│   ├── validate_live_jobs.py    # pipeline validation against live Adzuna postings
│   └── survey_headers.py        # header-pattern survey across resume dataset categories
├── data/
│   ├── ESCO/                    # skills taxonomy (not tracked, see Setup)
│   └── Testing/                 # sample resume/JD for testing
├── requirements.txt
└── README.md
```

## Author

Otioh Konan - Illinois Institute of Technology, B.S. Artificial Intelligence
