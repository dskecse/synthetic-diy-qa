import json
import os
import sys
import time
import uuid

import pandas as pd
from llm_judge import LLMJudge
from models import DIYRepairQA
from openai_client import get_openai_client

def load_structurally_valid_data(filename: str = "structurally_valid_qa_pairs.json"):
    filepath = os.path.join("data", filename)

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please run generation and validation phases first or provide the correct path.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing {filepath}: {str(e)}")
        sys.exit(1)

def create_failure_labeled_dataframe(qa_data: list[dict], model: str = "gpt-5.4-nano") -> pd.DataFrame:
    """
    Create a Pandas DataFrame with failure labels for all Q&A pairs
    """
    print(f"Starting failure labeling for {len(qa_data)} Q&A pairs using {model}...")

    judge = LLMJudge(model=model)
    labeled_data = []

    for i, qa_dict in enumerate(qa_data, start=1):
        print(f"Evaluating sample {i}/{len(qa_data)}...")

        # Convert dict to DIYRepairQA object
        qa_pair = DIYRepairQA(**{k: v for k, v in qa_dict.items() if k != "trace_id"})
        trace_id = qa_dict.get("trace_id", str(uuid.uuid4()))

        # Evaluate against all failure modes
        results = judge.evaluate(qa_pair, trace_id)
        labeled_data.append(results)

        # Pause to avoid rate limiting
        time.sleep(0.5)

    df = pd.DataFrame(labeled_data)

    print(f"\nFailure labeling complete!")
    print(f"Total samples: {len(df)}")
    print(f"Overall failures: {df["overall_failure"].sum()}")
    print(f"Overall success rate: {(1 - df["overall_failure"].mean()) * 100:.1f}%")

    return df

def main():
    print("Starting to conduct failure labeling...")

    # Load structurally valid data
    qa_data = load_structurally_valid_data()

    if not qa_data:
        print("No data to process. Please run generation and validation phases first or provide the correct path.")
        sys.exit(1)

    # Create failure labeled DataFrame
    failure_df = create_failure_labeled_dataframe(qa_data)

    # Save failure labeled data
    os.makedirs("data", exist_ok=True)

    failure_df.to_csv(os.path.join("data", "failure_labeled_data.csv"), index=False)
    failure_df.to_json(os.path.join("data", "failure_labeled_data.json"), orient="records", indent=2)

    print(f"\nResults saved to:")
    print("  • data/failure_labeled_data.csv")
    print("  • data/failure_labeled_data.json")

    # Print summary by failure mode
    failure_modes = [failure_mode.name for failure_mode in FailureModeDefinitions.failure_modes()]

    print(f"\nFailure Mode Breakdown:")
    print("-" * 40)
    for mode in failure_modes:
        if mode in failure_df.columns:
            rate = failure_df[mode].mean()
            count = failure_df[mode].sum()
            print(f"{mode.replace("_", " ").title():25}: {rate:6.1%} ({count:2d}/{len(failure_df)})")

if __name__ == "__main__":
    main()
