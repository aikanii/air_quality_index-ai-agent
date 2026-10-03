<div align="center">

# 🌬️ AQI Analysis Agent

**A multi-agent air quality monitor that scrapes live data for any location and turns it into personalized health guidance.**

Built with Streamlit, the [Agno](https://github.com/agno-agi/agno) agent framework, Firecrawl, and an OpenAI model.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Agno](https://img.shields.io/badge/Agno-Agent%20Framework-6C47FF?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai&logoColor=white)
![Firecrawl](https://img.shields.io/badge/Firecrawl-Web%20Scraping-F97316?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly-3F4F66?style=flat-square&logo=plotly&logoColor=white)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Notes & Limitations](#notes--limitations)
- [License](#license)

---

## Overview

Enter a city and the app scrapes a live air-quality page, extracts the readings into a structured report, and then generates a health assessment tailored to your medical conditions and planned outdoor activity. Results are shown with an AQI gauge, pollutant charts, and a downloadable report.

---

## Features

- **Live data**: scrapes current air quality for any location using Firecrawl (preferring IQAir and AQICN pages).
- **Structured extraction**: readings are validated into an `AQIReport` Pydantic model (AQI, PM2.5, PM10, CO, temperature, humidity, wind speed, source URL, summary).
- **Personalized advice**: recommendations account for selected medical conditions (for example asthma, COPD, pregnancy) and the planned activity (for example running, cycling, sightseeing with kids).
- **Visual dashboard**: AQI gauge with the standard color bands, pollutant bar chart, and weather metrics, split across Overview, Pollutants, and Health Advice tabs.
- **WHO context**: PM2.5 and PM10 readings are compared against WHO 24-hour guideline levels.
- **Downloadable report**: export the full analysis as a Markdown file.
- **Flexible API keys**: load keys from a `.env` file or paste them into the sidebar at runtime.

---

## How It Works

Three agents run in sequence, each with a single job:

| Step | Agent | Role |
| --- | --- | --- |
| 1 | **AQI Scraper** | Uses the Firecrawl scrape tool to read a live air-quality page for the location and writes up every value it finds, plus the source URL. |
| 2 | **AQI Parser** | Converts the scraper's notes into a single structured `AQIReport`. It has no tools and only produces schema-constrained output. |
| 3 | **Health Recommendation Agent** | Takes the structured report plus your conditions and activity, and writes a Markdown report: health impact, a Safe / Caution / Avoid verdict, recommendations, best time of day, and weather correlation. |

> [!NOTE]
> Scraping and structuring are split into two agents on purpose. Combining live tool calls with strict structured output is unreliable across agno and model versions and often falls back to plain text. A scraper that returns free text, feeding a parser that only outputs the schema, is more consistent. The app also tolerates the parser returning JSON as a string or in code fences.

---

## Getting Started

### Prerequisites

- **Python** 3 with `pip`
- An **OpenAI API key**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- A **Firecrawl API key**: [firecrawl.dev](https://firecrawl.dev)

### Installation

```bash
git clone https://github.com/aikanii/air_quality_index-ai-agent.git
cd air_quality_index-ai-agent
pip install -r requirements.txt
```

> [!TIP]
> Use a virtual environment to keep dependencies isolated: `python -m venv .venv`, then activate it before running `pip install`.

### Add your API keys

Either copy the example file and fill it in:

```bash
cp env_example.txt .env
```

```env
OPENAI_API_KEY=your_openai_api_key_here
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
```

or skip this step and paste the keys into the app's sidebar. Values typed in the sidebar take priority for the running session.

> [!WARNING]
> `.env` is git-ignored. Never commit your API keys.

### Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (by default [http://localhost:8501](http://localhost:8501)).

---

## Usage

1. Enter your OpenAI and Firecrawl API keys in the sidebar, unless they were loaded from `.env`.
2. Optionally select medical conditions and a planned outdoor activity in the sidebar.
3. Type a location (a city, or "city, country"), or click one of the example queries.
4. Click **Analyze Air Quality**.
5. Review the AQI gauge, pollutant chart, and metrics, then read the health recommendations.
6. Click **Download Full Report** to save the results as a Markdown file.

---

## Configuration

| Setting | Where | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | `.env` or sidebar | OpenAI API key |
| `FIRECRAWL_API_KEY` | `.env` or sidebar | Firecrawl API key |
| Model | `build_agents()` in `app.py` | Defaults to `gpt-4o`. Change the `id` passed to `OpenAIChat` to use another OpenAI model. |
| Example locations, conditions, activities | Constants at the top of `app.py` | `EXAMPLE_QUERIES`, `MEDICAL_CONDITIONS`, and `ACTIVITIES` can be edited directly. |

---

## Project Structure

```text
air_quality_index-ai-agent/
├── app.py              # Streamlit UI, theme, and all three Agno agents
├── requirements.txt    # Python dependencies
├── env_example.txt     # Template for the .env file
└── README.md
```

---

## Notes & Limitations

- **Data quality depends on scraping.** The analysis is only as good as what Firecrawl can read from the source page at request time. The scraper is told to say so when a value is missing rather than guess silently, but the parser may estimate missing fields and mention it in the summary, so check the source URL shown in the results.
- **Not medical advice.** The recommendations are general precautionary guidance. Consult a physician for decisions involving specific health conditions.
- **Costs.** Each analysis makes several OpenAI calls and at least one Firecrawl scrape, which count against your API usage.
- **"Best time" is a heuristic.** The suggested outdoor window is based on typical pollution patterns, not a live forecast.

---

## License

No license has been specified for this project yet. Until one is added, all rights are reserved by default.
