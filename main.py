# main.py
# Entry point for the agentic research pipeline.
# Orchestrates the full run: loads companies, runs the agent for each one,
# validates results against the Pydantic schema, logs everything, and
# writes the final structured report to output/report.json.

import json
import os
import time
from pydantic import ValidationError
from agent import build_agent, validate_and_fix
from models import CompanyResearch, ResearchReport, SourceType
from logger import ObservabilityLogger
from langchain_ollama import ChatOllama


def main():
    # --- Setup ---
    # Ensure output and logging directories exist before the run starts
    os.makedirs('output', exist_ok=True)
    os.makedirs('run_logs', exist_ok=True)
    output_path = 'output/report.json'

    # Clear any existing report from a previous run to start fresh
    if os.path.exists(output_path):
        os.remove(output_path)

    # Initialise the observability logger — this tracks everything that happens
    logger = ObservabilityLogger(log_dir='run_logs')

    # Initialise the LLM separately so it can be used for validation retries
    llm = ChatOllama(model='llama3.1', temperature=0)

    # Build the agent (LLM + tools + prompt wrapped in an AgentExecutor)
    agent = build_agent()

    # --- Load Companies ---
    import pandas as pd
    companies = pd.read_csv('input/companies.csv')['company_name'].tolist()
    print(f'\nStarting research for {len(companies)} companies...\n')

    # Lists to track validated successes and failures across the full run
    validated_results = []
    failed_companies = []

    # --- Process Each Company ---
    # Research is done one company at a time for precise per-company logging
    for company in companies:
        print(f'\n--- Researching: {company} ---')
        logger.start_company(company)  # Begin tracking this company in the log
        start = time.time()

        try:
            # Invoke the agent with a task prompt for the current company.
            # The agent will use its tools (Wikipedia, DuckDuckGo, etc.) to research
            # the company and write the result to output/report.json.
            result = agent.invoke({
                'input': f'''Research {company} and write the result to output/report.json.
                Use Wikipedia first, fall back to DuckDuckGo if needed.
                Return a complete JSON result with all required fields.'''
            })

            # Read the output file to retrieve the most recently written result.
            # The agent appends results one at a time, so the last entry is for this company.
            with open(output_path, 'r') as f:
                all_results = json.load(f)
            raw = all_results[-1]  # Most recently written result

            # --- Validate Against Pydantic Schema ---
            try:
                # validate_and_fix will attempt validation and retry once if it fails
                validated = validate_and_fix(raw, company, llm)
                logger.log_validation('passed')
                validated_results.append(validated)
                print(f'  Validation: PASSED (confidence: {validated.confidence_score})')

            except ValidationError as ve:
                # Validation failed even after retry — log and mark as failed
                logger.log_validation('failed', [str(ve)])
                logger.log_retry()
                failed_companies.append(company)
                print(f'  Validation: FAILED - {ve}')

        except Exception as e:
            # Catch any unexpected errors (network issues, agent errors, file errors, etc.)
            logger.log_validation('error', [str(e)])
            failed_companies.append(company)
            print(f'  Error processing {company}: {e}')

        # Record which source was used and finalise this company's log entry
        source = validated_results[-1].source_used if validated_results else 'unknown'
        logger.finish_company(str(source))

    # --- Build Final Report ---
    # Wrap all validated results into a ResearchReport and overwrite the output file
    # with the clean, fully validated version of the data.
    report = ResearchReport(
        companies=validated_results,
        total_processed=len(companies),
        successful=len(validated_results),
        failed=len(failed_companies),
        failed_companies=failed_companies
    )
    with open(output_path, 'w') as f:
        json.dump(report.model_dump(), f, indent=2)

    # Finalise the run log — computes summary stats and writes the JSON log file
    summary = logger.finish_run()

    # --- Print Final Summary to Console ---
    print('\n' + '='*50)
    print('RUN COMPLETE')
    print('='*50)
    print(f'Total processed:  {summary["total_companies"]}')
    print(f'Successful:       {summary["successful"]}')
    print(f'Failed:           {summary["failed"]}')
    print(f'Success rate:     {summary["success_rate"]*100:.0f}%')
    print(f'Avg time/company: {summary["avg_duration_per_company"]}s')
    print(f'\nFull report:  output/report.json')
    print(f'Run log:      run_logs/')


# Only run main() when this file is executed directly (not when imported as a module)
if __name__ == '__main__':
    main()
