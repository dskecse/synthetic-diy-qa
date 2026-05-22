# Synthetic DIY Q&A

Synthetic data generation of Home DIY/Repair QA. Uses LLM-as-a-Judge technique.

## Prerequisites

* Git
* Python 3.13.x
* [dotenvx](https://github.com/dotenvx/dotenvx)

## Setup

```sh
git clone https://github.com/dskecse/synthetic-diy-qa
cd $_
python3.13 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Specify your OpenAI-compatible (e.g. Poe, ChatAnywhere) `API_KEY` and `API_BASE_URL` in `.env` and encrypt using `dotenvx encrypt`.

## Run

### Generation phase

```python
dotenvx run -- python3 synthetic_dataset.py
```

Generates `data/generation_results.json`.

Initial stats:

* Total generated: 20
* Valid samples: 19
* Invalid samples: 1
* Success rate: 95.0%

### Structural validation phase

```python
python3 synthetic_data_validation.py
```

Requires `data/generation_results.json`.

Produces `data/structurally_valid_qa_pairs.json`.

Stats:

* Total samples: 20
* Structurally valid samples: 19
* Structurally invalid samples: 1
* Structural validation rate: 95.0%

### Failure labeling

```python
dotenvx run -- python3 failure_labeling.py
```

Requires `data/structurally_valid_qa_pairs.json`.

Creates `data/failure_labeled_data.csv` and `data/failure_labeled_data.json`.

Stats:

* Total samples: 19
* Overall failures: 4
* Overall success rate: 78.9%

Failure Mode Breakdown:
----------------------------------------
Incomplete Answer        :  10.5% ( 2/19)
Safety Violations        :   0.0% ( 0/19)
Unrealistic Tools        :   0.0% ( 0/19)
Overcomplicated Solution :   0.0% ( 0/19)
Missing Context          :   5.3% ( 1/19)
Poor Quality Tips        :  15.8% ( 3/19)
