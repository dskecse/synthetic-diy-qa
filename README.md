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

```sh
dotenvx run -- python3 synthetic_dataset.py
```

Generates `data/generation_results.json`.

Stats:

|                     | Iteration 0 | Iteration 1 | Iteration 2 | Iteration 3 | Iteration 4 |
| ------------------- | ----------: | ----------: | ----------: | ----------: | ----------: |
| **Total generated** | 20          | 20          | 20          | 20          | 20          |
| **Valid samples**   | 19          | 20          | 20          | 20          | 20          |
| **Invalid samples** | 1           | 0           | 0           | 0           | 0           |
| **Success rate**    | 95.0%       | 100.0%      | 100.0%      | 100.0%      | 100.0%      |

### Structural validation phase

```sh
python3 synthetic_data_validation.py
```

Requires `data/generation_results.json`.

Produces `data/structurally_valid_qa_pairs.json`.

Stats:

|                                  | Iteration 0 | Iteration 1 | Iteration 2 | Iteration 3 | Iteration 4 |
| -------------------------------- | ----------: | ----------: | ----------: | ----------: | ----------: |
| **Total samples**                | 20          | 20          | 20          | 20          | 20          |
| **Structurally valid samples**   | 19          | 20          | 20          | 20          | 20          |
| **Structurally invalid samples** | 1           | 0           | 0           | 0           | 0           |
| **Structural validation rate**   | 95.0%       | 100.0%      | 100.0%      | 100.0%      | 100.0%      |

### Failure labeling phase

```sh
dotenvx run -- python3 failure_labeling.py
```

Requires `data/structurally_valid_qa_pairs.json`.

Creates `data/failure_labeled_data.csv` and `data/failure_labeled_data.json`.

Stats:

|                          | Iteration 0 | Iteration 1 | Iteration 2 | Iteration 3 | Iteration 4 |
| ------------------------ | ----------: | ----------: | ----------: | ----------: | ----------: |
| **Total samples**        | 19          | 20          | 20          | 20          | 20          |
| **Overall failures**     | 4           | 4           | 5           | 5           | 0           |
| **Overall success rate** | 78.9%       | 80.0%       | 75.0%       | 75.0%       | 100.0%      |

Failure Mode Breakdown:

| Failure Mode             | Failure Rate |
| :----------------------- | -----------: |
| Incomplete Answer        |  0.0% (0/20) |
| Safety Violations        |  0.0% (0/20) |
| Unrealistic Tools        |  0.0% (0/20) |
| Overcomplicated Solution |  0.0% (0/20) |
| Missing Context          |  0.0% (0/20) |
| Poor Quality Tips        |  0.0% (0/20) |

### Failure analysis phase

```sh
python3 failure_analysis.py
```

Requires `data/failure_labeled_data.csv`.

Generates:

* `data/failure_analysis_report.json`
* `assets/failure_heatmap.png`: failure mode heatmap across all samples
* `assets/failure_rates.png`: failure rates by mode chart
* `assets/failure_correlations.png`: failure mode correlations heatmap

## Failure Analysis Report

|                          | Iteration 0 | Iteration 1 | Iteration 2 | Iteration 3 | Iteration 4 |
| ------------------------ | ----------: | ----------: | ----------: | ----------: | ----------: |
| **Total samples**        | 19          | 20          | 20          | 20          | 20          |
| **Overall failure rate** | 21.1%       | 20.0%       | 25.0%       | 25.0%       | 0.0%        |
| **Overall success rate** | 78.9%       | 80.0%       | 75.0%       | 75.0%       | 100.0%      |

Most Problematic Areas:

|                              | Iteration 0          | Iteration 1          | Iteration 2          | Iteration 3          | Iteration 4         |
| ---------------------------- | -------------------: | -------------------: | -------------------: | -------------------: | ------------------: |
| **Poor Quality Tips**        | 15.8% (3/19 samples) | 10.0% (2/20 samples) | 10.0% (2/20 samples) | 25.0% (5/20 samples) | 0.0% (0/20 samples) |
| **Incomplete Answer**        | 10.5% (2/19 samples) |  0.0% (0/20 samples) | 10.0% (2/20 samples) |  0.0% (0/20 samples) | 0.0% (0/20 samples) |
| **Missing context**          |  5.3% (1/19 samples) | 15.0% (3/20 samples) |  0.0% (0/20 samples) |  0.0% (0/20 samples) | 0.0% (0/20 samples) |
| **Overcomplicated Solution** |  0.0% (0/19 samples) |  0.0% (0/20 samples) |  5.0% (1/20 samples) |  0.0% (0/20 samples) | 0.0% (0/20 samples) |

Goal: increase overall success rate > 90% :done:
