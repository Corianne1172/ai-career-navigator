import requests
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")

APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")

def search_jobs(query):
    url = f"https://api.adzuna.com/v1/api/jobs/us/search/1?app_id={APP_ID}&app_key={APP_KEY}&results_per_page=10&what={query}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return {"error": "Failed to fetch jobs"}
    
if __name__ == "__main__":
    results = search_jobs("AI engineer")
    jobs = results.get("results", [])
    for job in jobs:
        print(f"Title: {job['title']}")
        print(f"Company: {job['company']['display_name']}")
        print(f"Description: {job['description'][:150]}")
        print("---")