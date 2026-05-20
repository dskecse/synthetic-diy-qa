import json
from typing import Optional

from pydantic import BaseModel

class DIYRepairQA(BaseModel):
    question: str
    answer: str
    equipment_problem: str
    tools_required: list[str]
    steps: list[str]
    safety_info: str
    tips: str

class GenerationResult(BaseModel):
    """
    Model for tracking generation results and metadata
    """
    trace_id: str
    qa_pair: Optional[DIYRepairQA]
    raw_response: str
    is_valid: bool
    generated_at: str

class ValidationSummary(BaseModel):
    total_samples: int
    valid_samples: int
    invalid_samples: int
    validation_rate: int
    common_errors: list[str]

def validate_json_structure(json_str: str) -> tuple[bool, Optional[DIYRepairQA]]:
    """
    Ensure the output JSON follows a structure
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return False, None

    try:
        qa_pair = DIYRepairQA(**data)
        return True, qa_pair
    except Exception as e:
        return False, None
