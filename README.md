---

title: ARIA Research Agent

emoji: 🔬

colorFrom: blue

colorTo: indigo

sdk: streamlit

sdk_version: "1.32.0"

app_file: [app.py](http://app.py)

pinned: true

---

# ARIA — AI Research Intelligence Assistant

An agentic research assistant for clinical trial coordinators and life sciences professionals.

## What it does

ARIA uses Claude tool use to autonomously search across multiple data sources and synthesize findings:

- **PubMed** — searches published scientific literature via NCBI E-utilities

- **[ClinicalTrials.gov](http://ClinicalTrials.gov)** — searches active and recruiting clinical trials via the v2 API

- **Multi-tool orchestration** — Claude decides which tools to call based on the query, and runs them in parallel

- **Transparency layer** — shows exactly what ARIA searched and how many results were found before synthesizing

## Tech stack

- Claude (claude-sonnet-4-5) with tool use

- PubMed NCBI E-utilities API (no key required)

- [ClinicalTrials.gov](http://ClinicalTrials.gov) API v2 (no key required)

- Python + Streamlit

- Parallel tool execution via `concurrent.futures`

## Example queries

- "Find recruiting trials for pediatric epilepsy"

- "What is the latest research on velarixin and are there any active trials?"

- "Adverse event reporting in Phase II trials"

- "Rare disease drug development challenges"

## Setup

```bash

pip install -r requirements.txt

# Add ANTHROPIC_API_KEY to .env

streamlit run [app.py](http://app.py)

```

## Note

For research support only. Findings should be verified with your clinical team before applying to trial decisions.

