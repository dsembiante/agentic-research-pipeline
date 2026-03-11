# Agentic Research Pipeline

An AI-powered company research automation tool. Give it a list of companies and it autonomously searches the web, extracts structured data, validates it, and produces a clean JSON report.

## How It Works

1. Reads a list of companies from a CSV file
2. For each company, a LangChain agent (powered by a local Llama 3.1 model via Ollama) searches Wikipedia, falling back to DuckDuckGo if needed
3. Extracted data is validated against a strict Pydantic schema
4. If validation fails, the agent retries with a corrective prompt
5. Results are saved to a structured JSON report with per-run logs

## Output

- `output/report.json` — structured research results for all companies
- `run_logs/` — detailed per-run logs with tool calls, timings, and validation metrics

## Extracted Fields

For each company:
- Name, summary, industry
- Founding year, headquarters
- Source type and confidence score

## Tech Stack

- **LLM**: [Ollama](https://ollama.com/) running `llama3.1` (local, no API key required)
- **Agent Framework**: LangChain
- **Validation**: Pydantic v2
- **Search**: Wikipedia API + DuckDuckGo fallback
- **Data**: Pandas (CSV input)

## Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running with the `llama3.1` model

```bash
ollama pull llama3.1
```

### Install Dependencies

```bash
python -m venv venv
source venv/Scripts/activate  # Windows (Git Bash)
# or
source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
```

### Add Your Companies

Edit `input/companies.csv` with a `company_name` column:

```csv
company_name
Microsoft
Apple
Amazon
```

### Run

```bash
python main.py
```

## Project Structure

```
ai-agent-version/
├── main.py          # Entry point and pipeline orchestration
├── agent.py         # LangChain agent setup and validation logic
├── tools.py         # Agent tools (CSV reader, Wikipedia, DuckDuckGo, output writer)
├── models.py        # Pydantic schemas for structured output
├── logger.py        # Structured run logging
├── input/
│   └── companies.csv
├── output/
│   └── report.json
└── run_logs/
```
