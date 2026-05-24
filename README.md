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

* Iteration 0:
  * Total generated: 20
  * Valid samples: 19
  * Invalid samples: 1
  * Success rate: 95.0%
* Iteration 1:
  * Total generated: 20
  * Valid samples: 20
  * Invalid samples: 0
  * Success rate: 100.0%
* Iteration 2:
  * Total generated: 20
  * Valid samples: 20
  * Invalid samples: 0
  * Success rate: 100.0%

### Structural validation phase

```sh
python3 synthetic_data_validation.py
```

Requires `data/generation_results.json`.

Produces `data/structurally_valid_qa_pairs.json`.

Stats:

* Iteration 0:
  * Total samples: 20
  * Structurally valid samples: 19
  * Structurally invalid samples: 1
  * Structural validation rate: 95.0%
* Iteration 1:
  * Total samples: 20
  * Structurally valid samples: 20
  * Structurally invalid samples: 0
  * Structural validation rate: 100.0%

### Failure labeling phase

```sh
dotenvx run -- python3 failure_labeling.py
```

Requires `data/structurally_valid_qa_pairs.json`.

Creates `data/failure_labeled_data.csv` and `data/failure_labeled_data.json`.

Stats:

* Iteration 0:
  * Total samples: 19
  * Overall failures: 4
  * Overall success rate: 78.9%
* Iteration 1:
  * Total samples: 20
  * Overall failures: 4
  * Overall success rate: 80.0%

Failure Mode Breakdown:

| Failure Mode             | Failure Rate |
| :----------------------- | -----------: |
| Incomplete Answer        |  0.0% (0/20) |
| Safety Violations        |  0.0% (0/19) |
| Unrealistic Tools        |  0.0% (0/19) |
| Overcomplicated Solution |  0.0% (0/19) |
| Missing Context          | 15.3% (3/20) |
| Poor Quality Tips        | 10.0% (2/20) |

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

### Iteration 0

* Total samples: 19
* Overall Failure Rate: 21.1%
* Overall Success Rate: 78.9%

Most Problematic Areas:

1. Poor Quality Tips: 15.8% (3/19 samples)
2. Incomplete Answer: 10.5% (2/19 samples)
3. Missing Context: 5.3% (1/19 samples)

Recommendations:

1. Focus on improving 'poor quality tips' - it's the most common failure mode (15.8% failure rate)
2. 78.9% of samples have no failures - analyze these for best practices

Goals:

1. Increase overall success rate > 90%.
2. Fix the prompts to be able to generate all 20 samples.

### Iteration 1

* Total samples: 20
* Overall Failure Rate: 20.0%
* Overall Success Rate: 80.0%

Most Problematic Areas:

1. Missing Context: 15.0% (3/20 samples)
2. Poor Quality Tips: 10.0% (2/20 samples)

Recommendations:

1. Focus on improving 'missing context' - it's the most common failure mode (15.0% failure rate)
2. 80.0% of samples have no failures - analyze these for best practices

Goals:

1. Increase overall success rate > 90%.
