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
    if not item.is_valid:
        return False, [] # Would be nice to have some validation errors from the generation phase
    else:
        return True, []

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
