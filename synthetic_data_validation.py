"""
Synthetic Data Validation
Ensures generated Q&A pairs are structurally valid, and saves valid data only.
"""

from collections import Counter
import json
import os

from models import GenerationResult, ValidationSummary

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

def validate(results: list[GenerationResult]) -> tuple[list[GenerationResult], ValidationSummary]:
    """
    Run structural validation of dataset items
    Return only valid dataset items and errors (if any).
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

    error_counter = Counter(all_errors)
    common_errors = [error for error, count in error_counter.most_common(5)]

    # Create a summary
    summary = ValidationSummary(
        total_samples=len(results),
        valid_samples=len(valid_results),
        invalid_samples=len(results) - len(valid_results),
        validation_rate=len(valid_results) / len(results) * 100 if len(results) else 0,
        common_errors=common_errors
    )

    return valid_results, summary

def save_valid_data(valid_data: list[GenerationResult], filename: str = "structurally_valid_qa_pairs.json"):
    """
    Save structurally valid Q&A pairs to JSON file
    """
    serializable_data = []
    for item in valid_data:
        if item.qa_pair:
            qa_dict = item.qa_pair.model_dump()
            qa_dict["trace_id"] = item.trace_id
            serializable_data.append(qa_dict)

    # Create a directory (do not raise if already present)
    os.makedirs("data", exist_ok=True)

    filepath = os.path.join("data", filename)
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(serializable_data, file, indent=2, ensure_ascii=False)

    print(f"\nValid data saved to {filepath}")

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
    valid_results, summary = validate(results)

    # Print summary
    print(f"\nValidation Phase Complete:")
    print(f"Total samples: {summary.total_samples}")
    print(f"Structurally valid samples: {summary.valid_samples}")
    print(f"Structurally invalid samples: {summary.invalid_samples}")
    print(f"Structural validation rate: {summary.validation_rate:.1f}%")

    if len(summary.common_errors):
        print(f"Most common structural errors:")
        for i, error in enumerate(summary.common_errors, start=1):
            print(f"{i}. {error}")

    # Save valid data (Q&A pairs only)
    save_valid_data(valid_results)


if __name__ == "__main__":
    main()
