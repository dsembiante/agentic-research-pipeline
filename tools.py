# tools.py
# Defines the four tools available to the research agent.
# Each function is decorated with @tool so LangChain can expose it to the agent.
# The agent reasons about which tools to use and in what order based on results.

import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from langchain.tools import tool
from ddgs import DDGS


@tool
def read_companies_from_csv(filepath: str) -> list:
    """Reads company names from a CSV file with a company_name column.
    Returns a Python list of company name strings."""
    # Load the CSV into a DataFrame and extract the company_name column as a list
    df = pd.read_csv(filepath)
    return df['company_name'].tolist()


@tool
def search_wikipedia(company_name: str) -> dict:
    """Searches Wikipedia for a company and returns a dict with:
      - found (bool): whether a good article was found
      - summary (str): first substantial paragraph
      - url (str): the Wikipedia URL used
    Returns found=False if the article is missing or too short."""

    # Build the Wikipedia URL by replacing spaces with underscores
    url = f'https://en.wikipedia.org/wiki/{company_name.replace(" ", "_")}'

    try:
        # Fetch the Wikipedia page with a descriptive User-Agent header
        response = requests.get(url, headers={'User-Agent': 'ResearchBot/1.0'}, timeout=10)

        # If the page doesn't exist or returns an error, report not found
        if response.status_code != 200:
            return {"found": False, "summary": "", "url": url}

        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Skip disambiguation pages — these list multiple meanings rather than one company
        if soup.find('div', {'id': 'disambigbox'}) or \
           'may refer to' in soup.get_text()[:500]:
            return {"found": False, "summary": "disambiguation page", "url": url}

        # Find the first paragraph with meaningful content (more than 150 characters)
        for para in soup.find_all('p'):
            text = para.get_text().strip()
            if len(text) > 150:
                return {"found": True, "summary": text, "url": url}

        # If no substantial paragraph was found, report not found
        return {"found": False, "summary": "no substantial paragraph found", "url": url}

    except Exception as e:
        # Catch network errors, timeouts, etc. and return the error message
        return {"found": False, "summary": str(e), "url": url}


@tool
def search_duckduckgo(company_name: str) -> dict:
    """Fallback search using DuckDuckGo when Wikipedia fails.
    Returns a dict with:
      - found (bool): whether results were found
      - summary (str): combined snippet from top results."""

    try:
        # Use DDGS (DuckDuckGo Search) to fetch the top 3 results for the company
        with DDGS() as ddgs:
            results = list(ddgs.text(
                f'{company_name} company overview industry founded',
                max_results=3
            ))

        # If no results were returned, report not found
        if not results:
            return {"found": False, "summary": ""}

        # Combine the body snippets from the top 3 results into one summary string
        # Truncate to 1000 characters to keep the output manageable for the agent
        combined = ' '.join(r.get('body', '') for r in results[:3])
        return {"found": True, "summary": combined[:1000]}

    except Exception as e:
        # Catch any search errors and return the error message
        return {"found": False, "summary": str(e)}


@tool
def write_company_result(company_json: str, filepath: str) -> str:
    """Appends a validated company JSON result to the output file.
    company_json must be a valid JSON string.
    Returns a confirmation message."""

    import json
    import os

    # Ensure the output directory exists before attempting to write
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    # Load existing results from the file if it already exists, otherwise start with an empty list
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            existing = json.load(f)
    else:
        existing = []

    # Parse the incoming JSON string and append it to the results list
    existing.append(json.loads(company_json))

    # Write the updated list back to the file with readable indentation
    with open(filepath, 'w') as f:
        json.dump(existing, f, indent=2)

    return f'Result written. Total results in file: {len(existing)}'
