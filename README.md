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

Initial Stats:

* Total generated: 20
* Valid samples: 19
* Invalid samples: 1
* Success rate: 95.0%
