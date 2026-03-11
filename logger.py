# logger.py
# Handles structured run logging for the research agent.
# Every agent run produces a JSON log file saved to the run_logs directory,
# capturing tool calls, validation results, retries, and timing per company.

import json
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional


# Represents a single tool call made by the agent during research.
# Tracks what tool was used, what it was given, how long it took, and whether it succeeded.
@dataclass
class ToolCall:
    tool_name: str
    input_summary: str
    duration_seconds: float
    result_status: str  # "success", "fallback", or "failed"


# Represents the full log for one company's research run.
# Collects all tool calls, validation outcome, retries, and total time taken.
@dataclass
class CompanyRunLog:
    company_name: str
    tool_calls: list[ToolCall] = field(default_factory=list)   # List of tool calls made
    retries: int = 0                                            # Number of retry attempts
    validation_status: str = 'pending'                         # 'pending', 'passed', or 'failed'
    validation_errors: list[str] = field(default_factory=list) # Any Pydantic validation errors
    total_duration_seconds: float = 0.0                        # Total time spent on this company
    source_used: str = 'unknown'                               # Final data source used


# Represents the top-level log for an entire agent run.
# Contains logs for all companies and a summary written at the end.
@dataclass
class RunLog:
    run_id: str                                                 # Unique ID based on timestamp
    start_time: str                                            # ISO format start timestamp
    end_time: Optional[str] = None                             # ISO format end timestamp
    total_duration_seconds: float = 0.0                        # Total run duration
    companies: list[CompanyRunLog] = field(default_factory=list)  # Per-company logs
    summary: dict = field(default_factory=dict)                # Aggregate stats for the run


class ObservabilityLogger:
    """
    Main logger class used throughout the agent run.
    Tracks each company's research process and writes a structured JSON log at the end.
    """

    def __init__(self, log_dir: str = 'run_logs'):
        # Ensure the log output directory exists
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Initialise a new RunLog with a timestamp-based ID
        self.run_log = RunLog(
            run_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_time=datetime.now().isoformat()
        )

        # Record the run start time for duration calculation
        self._start_time = datetime.now()

        # Holds the log for the company currently being processed
        self._current_company: Optional[CompanyRunLog] = None

    def start_company(self, company_name: str):
        """Begin logging for a new company. Called before research starts."""
        self._current_company = CompanyRunLog(company_name=company_name)
        self._company_start = datetime.now()

    def log_tool_call(self, tool_name: str, input_summary: str,
                      duration: float, status: str):
        """
        Record a tool call made during research.
        Called after each tool (Wikipedia, DuckDuckGo, etc.) is used.
        """
        if self._current_company:
            self._current_company.tool_calls.append(ToolCall(
                tool_name=tool_name,
                input_summary=input_summary,
                duration_seconds=round(duration, 3),
                result_status=status
            ))

    def log_validation(self, status: str, errors: list[str] = None):
        """
        Record the Pydantic validation result for the current company.
        Status should be 'passed' or 'failed'.
        """
        if self._current_company:
            self._current_company.validation_status = status
            self._current_company.validation_errors = errors or []

    def log_retry(self):
        """Increment the retry counter for the current company."""
        if self._current_company:
            self._current_company.retries += 1

    def finish_company(self, source_used: str):
        """
        Finalise the current company's log entry and append it to the run log.
        Calculates total duration and records which data source was ultimately used.
        """
        if self._current_company:
            self._current_company.total_duration_seconds = round(
                (datetime.now() - self._company_start).total_seconds(), 3
            )
            self._current_company.source_used = source_used
            self.run_log.companies.append(self._current_company)
            self._current_company = None  # Reset for the next company

    def finish_run(self):
        """
        Finalise the entire run log, compute summary statistics,
        and write the structured JSON log file to disk.
        Returns the summary dict for use in the final report.
        """
        # Record end time and total run duration
        self.run_log.end_time = datetime.now().isoformat()
        self.run_log.total_duration_seconds = round(
            (datetime.now() - self._start_time).total_seconds(), 3
        )

        # Calculate success/failure counts across all companies
        successful = sum(
            1 for c in self.run_log.companies if c.validation_status == "passed"
        )
        failed_companies = [
            c.company_name for c in self.run_log.companies if c.validation_status != "passed"
        ]

        # Build the summary block with aggregate stats
        self.run_log.summary = {
            'total_companies': len(self.run_log.companies),
            'successful': successful,
            'failed': len(failed_companies),
            'failed_companies': failed_companies,
            'success_rate': round(successful / max(len(self.run_log.companies), 1), 2),
            'avg_duration_per_company': round(
                sum(c.total_duration_seconds for c in self.run_log.companies) /
                max(len(self.run_log.companies), 1), 3
            )
        }

        # Write the full log to a timestamped JSON file
        log_path = os.path.join(
            self.log_dir, f'run_{self.run_log.run_id}.json'
        )
        with open(log_path, 'w') as f:
            json.dump(asdict(self.run_log), f, indent=2)

        print(f'\n Run log saved to: {log_path}')
        return self.run_log.summary
