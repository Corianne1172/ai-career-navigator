# AI Career Navigator

An AI-powered tool that matches student resumes to job postings, identifies skill gaps, and generates tailored interview prep. Built for CS 497 (Special Projects) at Illinois Tech.

It differs from existing resume-matching tools through:
- Gap analysis identifying what's missing from a student's profile for their target roles, with concrete next steps
- Tailored interview prep questions generated from the student's specific resume and target job description
- (Stretch goal) H-1B sponsorship likelihood prediction based on role type, company size, and other available signals

## Status

Phase 1 (complete): resume parsing, baseline embedding match, ChromaDB vector store, live job data via Adzuna API.

Phase 2 (in progress): skill extraction pivoted from prompt-based (Phi-3 Mini) to taxonomy-based matching using the [ESCO skills database](https://esco.ec.europa.eu/en/use-esco/download), with FlashText for fast keyword matching. Semantic matching pipeline (embeddings + composite scoring) is next.

## Setup

```bash
git clone https://github.com/Corianne1172/ai-career-navigator
cd ai-career-navigator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

ESCO data is not tracked in this repo due to file size. Download the CSV bundle (English) from the [ESCO download page](https://esco.ec.europa.eu/en/use-esco/download) and place `skills_en.csv` and `skillsHierarchy_en.csv` in `data/ESCO/`.

## Project structure
