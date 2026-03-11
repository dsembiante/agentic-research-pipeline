# Agentic Research Pipeline

A production-aware AI agent that autonomously researches companies, validates structured output against a strict Pydantic schema, falls back intelligently across data sources, and logs full observability data for every run — all powered by a local LLM with no API key required.

## Background & Motivation

I've spent 6 years building RPA and intelligent automation solutions across financial services, federal government consulting, healthcare, and insurance — including production deployments at State Farm, Mercy Hospital, Deloitte, and Capital One. Traditional RPA bots excel at deterministic, structured workflows and remain the right tool for many enterprise automation problems. This project explores where AI agents extend that capability — specifically in scenarios involving ambiguous data, unreliable sources, or outputs that require reasoning rather than rule-following.

## How It Works

1. Reads a list of companies from a CSV file
2. For each company, a LangChain agent (powered by a local Llama 3.1 model via Ollama) searches Wikipedia first
3. If Wikipedia returns a poor result — missing article, disambiguation page, or insufficient content — the agent automatically falls back to DuckDuckGo without any hardcoded logic telling it to do so
4. Extracted data is validated against a strict Pydantic v2 schema with field-level constraints and custom validators
5. If validation fails, the agent retries with a corrective prompt that includes the specific validation errors, allowing it to self-correct
6. Every run produces a structured JSON report and a detailed observability log capturing tool call sequences, durations, retry counts, and success metrics

## Key Technical Decisions

**Why Ollama instead of the OpenAI API?**
Running the LLM locally eliminates API costs entirely, removes the need for credentials, and makes the project fully reproducible by anyone. For a portfolio project meant to demonstrate agentic architecture rather than model selection, this is the right tradeoff. Swapping to OpenAI or Anthropic requires changing one line.

**Why Pydantic v2 for output validation?**
LLM output is non-deterministic. In production systems you cannot trust that a model will return well-formed, complete data every time. Pydantic enforces the schema at runtime, catches missing or malformed fields immediately, and gives the agent specific error messages to fix rather than silently writing bad data. This mirrors how production AI pipelines handle LLM output unreliability.

**Why separate observability logging?**
In enterprise automation, auditability and debuggability are non-negotiable. The run log answers questions that are impossible to answer from the output file alone: which tool was called for each company, did any fallbacks trigger, how long did each step take, and what was the overall success rate? Building this in from the start rather than as an afterthought reflects how production systems are designed.

**Why decision-branching tools instead of a single search function?**
Giving the agent two separate tools — `search_wikipedia` and `search_duckduckgo` — and having Wikipedia return a `found` boolean forces the agent to reason about tool selection rather than follow a hardcoded sequence. This is the architectural difference between a script and an agent.

## Sample Output

### report.json
```json
[
  {
    "company_name": "Capital One",
    "summary": "Capital One Financial Corporation is an American bank holding company specializing in credit cards, auto loans, banking, and savings accounts, headquartered in McLean, Virginia...",
    "industry": "Financial Services",
    "founded_year": 1994,
    "headquarters": "McLean, Virginia, United States",
    "source_used": "wikipedia",
    "confidence_score": 0.92
  }
]
```

### run_logs/run_YYYYMMDD_HHMMSS.json
```json
{
  "run_id": "20250310_143022",
  "start_time": "2025-03-10T14:30:22",
  "total_duration_seconds": 87.4,
  "summary": {
    "total_companies": 5,
    "successful": 5,
    "failed": 0,
    "success_rate": 1.0,
    "avg_duration_per_company": 17.48
  }
}
```

## Tech Stack

- **LLM**: [Ollama](https://ollama.com/) running `llama3.1` (local, no API key required)
- **Agent Framework**: LangChain with tool-calling agent
- **Output Validation**: Pydantic v2 with custom field validators
- **Search**: Wikipedia scraping + DuckDuckGo fallback via `duckduckgo-search`
- **Observability**: Custom structured run logger with per-company tool call tracking
- **Data**: Pandas (CSV input), JSON (structured output)

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

## What I Would Add Next

- **Streamlit UI** — allow users to type a company name and see the validated result in a browser, deployable free on Streamlit Community Cloud
- **Additional data sources** — financial data APIs (e.g. Yahoo Finance) as a third tool, letting the agent decide when richer data is needed
- **Agent memory** — allow the agent to reference previously researched companies when writing consolidated summaries
- **OpenAI/Anthropic toggle** — a config flag to switch between local Ollama and a cloud API, making it easy to benchmark output quality vs cost tradeoffs
- **Automated output evaluation** — a lightweight scoring layer that rates each summary for completeness against a rubric, closing the loop on output quality

## Author

Built as part of an ongoing AI engineering portfolio exploring the evolution from traditional RPA and intelligent automation toward modern agentic AI architectures.
