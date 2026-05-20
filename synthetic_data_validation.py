"""
Synthetic Data Validation
Ensures generated Q&A pairs are structurally valid, and saves valid data only.
"""

import json
import os

from models import GenerationResult

def load_synthetic_dataset(filename: str = "generation_results.json"):
    filepath = os.path.join("data", filename)

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            dataset = json.load(file)
        print(f"✓ Loaded {len(dataset)} samples from {filepath}")
        return dataset
    except FileNotFoundError:
        print(f"❌ Dataset file {filepath} not found")
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in {filepath}")

    return None

def _validate_structure(item: GenerationResult) -> tuple[bool, list[str]]:
    """
    Validate only the structural correctness of a generated item
    """
    if not item.is_valid:
        return False, [] # Would be nice to have some validation errors from the generation phase

    if item.qa_pair is None:
        return False, ["No valid Q&A pair generated"]

    errors = []
    qa_pair = item.qa_pair

    # Check for empty or whitespace-only fields
    if not qa_pair.question.strip():
        errors.append("Question is empty or whitespace only")
    if not qa_pair.answer.strip():
        errors.append("Answer is empty or whitespace only")
    if not qa_pair.equipment_problem.strip():
        errors.append("Equipment problem is empty or whitespace only")
    if not qa_pair.safety_info.strip():
        errors.append("Safety info is empty or whitespace only")
    if not qa_pair.tips.strip():
        errors.append("Tips is empty or whitespace only")

    # Check for empty lists
    if not len(qa_pair.tools_required):
        errors.append("Tools required list is empty")
    if not len(qa_pair.steps):
        errors.append("Steps list is empty")

    # Check for empty or whitespace-only items in lists
    if any(not tool.strip() for tool in qa_pair.tools_required):
        errors.append("Tools required list has empty or whitespace-only items")
    if any(not step.strip() for step in qa_pair.steps):
        errors.append("Steps list has empty or whitespace-only items")

    return len(errors) == 0, errors

def validate(results: list[GenerationResult]) -> list[GenerationResult]:
    """
    Run structural validation of dataset items
    Return only valid dataset items.
    """
    valid_results = []
    all_errors = []

    for item in results:
        is_valid, errors = _validate_structure(item)

        if is_valid:
            valid_results.append(item)
        else:
            item.is_valid = False
            all_errors.extend(errors)

    if len(all_errors):
        print(f"Dataset has the following errors:\n{all_errors}")

    return valid_results # + summary (with errors)

def main():
    print("Starting synthetic dataset validation...")
    dataset = load_synthetic_dataset()

    if not dataset:
        return

    # Convert back to GenerationResult objects
    results = []
    for item in dataset:
        results.append(GenerationResult(**item))

    # Validate results
    valid_results = validate(results)

    print(f"Valid results: {len(valid_results)}/{len(dataset)}")

    # TODO: Print summary and save valid data.

if __name__ == "__main__":
    main()
