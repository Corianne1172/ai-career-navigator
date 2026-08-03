"""
Adzuna job search API integration. Provides live job posting data for the
project, used both for standalone testing/demo purposes and for the
Phase 3 live-postings pipeline validation.

Note: Adzuna's free-tier API returns description as a fixed ~500-character
snippet, not the full job posting text. The full description is only
available at the posting's redirect_url on the original site, which is
out of scope to scrape. This is a known limitation of the data source,
documented in the project report.
"""

import requests
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")

APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")


def search_jobs(query):
    """
    Search live job postings via the Adzuna API for a given query.

    query: search term (e.g. "AI engineer"), used as Adzuna's "what"
        parameter.
    Returns the parsed JSON response (a dict with a "results" list of
    job postings) on success, or an error dict on failure.
    """
    url = f"https://api.adzuna.com/v1/api/jobs/us/search/1?app_id={APP_ID}&app_key={APP_KEY}&results_per_page=10&what={query}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return {"error": "Failed to fetch jobs"}


if __name__ == "__main__":
    # Quick smoke test: fetch and print a batch of live job postings.
    results = search_jobs("AI engineer")
    jobs = results.get("results", [])
    for job in jobs:
        print(f"Title: {job['title']}")
        print(f"Company: {job['company']['display_name']}")
        print(f"Description: {job['description'][:150]}")
        print("---")