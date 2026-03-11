# agent.py
# This is the core of the project. It wires together:
#   - The LLM (Ollama running llama3 locally)
#   - The tools (Wikipedia, DuckDuckGo, CSV reader, file writer)
#   - The system prompt (instructions for how the agent should behave)
#   - Pydantic validation (to ensure output quality before saving)

import json
import time
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import ValidationError

# Import the four tools the agent has access to
from tools import (
    read_companies_from_csv,
    search_wikipedia,
    search_duckduckgo,
    write_company_result
)

# Import Pydantic models for output validation and the SourceType enum
from models import CompanyResearch, SourceType

# Import the observability logger for structured run tracking
from logger import ObservabilityLogger


def validate_and_fix(raw_output: dict, company_name: str,
                     llm, retry_count: int = 0) -> CompanyResearch:
    """
    Attempt to validate the LLM's raw output against the CompanyResearch Pydantic schema.

    If validation fails on the first attempt, a corrective prompt is sent back to the LLM
    with the specific errors highlighted, giving it one chance to fix the output.
    Raises ValidationError if validation still fails after the retry.
    """
    try:
        # Try to parse and validate the raw dict against the Pydantic model
        return CompanyResearch(**raw_output)

    except ValidationError as e:
        # If we've already retried once, give up and raise the error
        if retry_count >= 1:
            raise

        # Build a corrective prompt that shows the LLM exactly what went wrong
        error_details = str(e)
        fix_prompt = f"""The following JSON failed validation:\n{json.dumps(raw_output, indent=2)}

Validation errors:\n{error_details}

Please fix the JSON and return ONLY a corrected JSON object for {company_name}.
The JSON must include: company_name, summary (min 50 chars), industry,
founded_year (int or null), headquarters (str or null),
source_used (wikipedia/duckduckgo/not_found), confidence_score (0.0-1.0)."""

        # Send the corrective prompt to the LLM and attempt to parse its response
        response = llm.invoke(fix_prompt)
        try:
            fixed = json.loads(response.content)
            # Recurse with retry_count + 1 to prevent infinite loops
            return validate_and_fix(fixed, company_name, llm, retry_count + 1)
        except json.JSONDecodeError:
            # If the LLM still returns unparseable output, raise a validation error
            raise ValidationError('Could not parse corrected JSON', CompanyResearch)


def build_agent():
    """
    Constructs and returns a LangChain AgentExecutor ready to run research tasks.

    This function:
    1. Initialises the local LLM via Ollama
    2. Registers the tools the agent can call
    3. Defines the system prompt that governs agent behaviour
    4. Creates a tool-calling agent and wraps it in an AgentExecutor
    """

    # Use the locally running llama3 model via Ollama.
    # temperature=0 makes output deterministic — important for structured data extraction.
    llm = ChatOllama(model='llama3.1', temperature=0)

    # Register all tools the agent is allowed to call during a run
    tools = [
        read_companies_from_csv,
        search_wikipedia,
        search_duckduckgo,
        write_company_result
    ]

    # Define the prompt template that shapes how the agent behaves.
    # The system message sets the rules; {input} receives the task; {agent_scratchpad}
    # holds the agent's internal reasoning and tool call history.
    prompt = ChatPromptTemplate.from_messages([
        ('system', '''You are a senior research agent that produces structured company data.

For each company you research, follow this exact process:
1. Use search_wikipedia first
2. If found=False, use search_duckduckgo as fallback
3. Extract: company name, a detailed summary (min 50 chars), industry,
   founded year if available, headquarters if available
4. Self-rate your confidence from 0.0 to 1.0 based on how complete your data is
5. ALWAYS call the write_company_result tool to save the result — never just print JSON as text

CRITICAL RULES for write_company_result:
- You MUST call write_company_result as a tool call, not as code or text
- The company_json argument MUST be a valid JSON string using double quotes, not single quotes
- The filepath argument MUST be "output/report.json"
- The JSON must contain exactly these fields:
  company_name (string), summary (string, min 50 chars), industry (string),
  founded_year (integer or null), headquarters (string or null),
  source_used ("wikipedia", "duckduckgo", or "not_found"), confidence_score (float 0.0-1.0)

Always complete all companies. Never skip a company without logging it.
If neither Wikipedia nor DuckDuckGo works, still call write_company_result with
source_used="not_found" and confidence_score=0.0.'''),
        ('human', '{input}'),
        ('placeholder', '{agent_scratchpad}')  # Stores intermediate reasoning steps
    ])

    # Create the tool-calling agent by binding the LLM, tools, and prompt together
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Wrap the agent in an AgentExecutor which manages the run loop.
    # max_iterations=15 prevents infinite loops if the agent gets stuck.
    # handle_parsing_errors=True allows the agent to recover from malformed outputs.
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,           # Prints each reasoning step to the console
        max_iterations=15,
        handle_parsing_errors=True
    )
