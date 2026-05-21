import json
import os
import sys

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

def main():
    print("Starting to conduct failure labeling...")

    # Load structurally valid data
    qa_data = load_structurally_valid_data()

    if not qa_data:
        print("No data to process. Please run generation and validation phases first or provide the correct path.")
        sys.exit(1)

    # TODO: Create failure labeled DataFrame
    print(qa_data[0])

    # TODO: Save failure labeled data

if __name__ == "__main__":
    main()
