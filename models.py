# models.py
# Defines Pydantic data models used to validate and structure the agent's output.
# Every result produced by the research agent is validated against these schemas
# before being saved, ensuring data quality and consistency.

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


# Enum representing the possible data sources the agent can use.
# Using an Enum ensures only valid source values are accepted.
class SourceType(str, Enum):
    WIKIPEDIA = "wikipedia"
    DUCKDUCKGO = "duckduckgo"
    NOT_FOUND = "not_found"


class CompanyResearch(BaseModel):
    """Schema for a single company's research result."""

    # The official name of the company being researched
    company_name: str = Field(..., description="Official company name")

    # A meaningful summary of at least 50 characters (enforced by min_length and validator)
    summary: str = Field(
        ...,
        description="Company summary of at least 50 characters",
        min_length=50
    )

    # The primary industry or business sector the company operates in
    industry: str = Field(..., description="Primary industry or sector")

    # Optional founding year — must be between 1700 and 2025 if provided
    founded_year: Optional[int] = Field(
        None,
        description="Year the company was founded",
        ge=1700,
        le=2025
    )

    # Optional headquarters location in "City, Country" format
    headquarters: Optional[str] = Field(
        None,
        description="City and country of headquarters"
    )

    # Which data source was used to gather the information
    source_used: SourceType = Field(
        ...,
        description="Which data source was used"
    )

    # A self-rated confidence score from the agent, between 0.0 (no confidence) and 1.0 (certain)
    confidence_score: float = Field(
        ...,
        description="Agent self-rated confidence from 0.0 to 1.0",
        ge=0.0,
        le=1.0
    )

    @field_validator('summary')
    @classmethod
    def summary_must_be_meaningful(cls, v: str) -> str:
        """
        Validates that the summary contains real content.
        Rejects placeholder text such as 'not found' or 'unknown'
        that would indicate the agent failed to retrieve useful data.
        """
        forbidden = ["not found", "no information", "unknown", "n/a"]
        if any(phrase in v.lower() for phrase in forbidden):
            raise ValueError("Summary contains placeholder text, not real content")
        return v


class ResearchReport(BaseModel):
    """The full report containing all company research results."""

    # List of individual company research results
    companies: list[CompanyResearch]

    # Total number of companies that were attempted
    total_processed: int

    # Number of companies successfully researched
    successful: int

    # Number of companies that failed during research
    failed: int

    # Names of companies that could not be researched (defaults to empty list)
    failed_companies: list[str] = []
