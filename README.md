#  AQI Analysis Agent

A multi-agent air quality monitoring and health recommendation tool built with
**Streamlit**, **Agno** (AI agent framework), **Firecrawl** (web scraping) and
an OpenAI model.

## How it works

Two agents work together:

1. **AQI Analyzer** — uses Firecrawl to scrape a live air-quality page for
   your location and extracts it into structured data: overall AQI, PM2.5,
   PM10, CO, temperature, humidity, and wind speed.
2. **Health Recommendation Agent** — takes that structured data plus your
   medical conditions and planned activity, and produces a personalized
   report: health impact, activity safety verdict, concrete precautions,
   best time to go outside, and how the weather is affecting pollution.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get API keys:
   - **OpenAI**: https://platform.openai.com/api-keys
   - **Firecrawl**: https://firecrawl.dev

   You can either paste these into the app's sidebar at runtime, or copy
   `.env.example` to `.env` and fill them in / export them as environment
   variables — the sidebar fields always take priority for the running
   session.

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Using the app

1. Enter your OpenAI and Firecrawl API keys in the sidebar.
2. (Optional) Select any medical conditions and your planned outdoor
   activity in the sidebar — these personalize the recommendations.
3. Type a location (city, or "city, country") or click one of the example
   queries.
4. Click **Analyze Air Quality**.
5. Review the AQI gauge, pollutant chart, and metrics, then read the
   personalized health recommendations below.
6. Click **Download Full Report** to save everything as a Markdown file.

## Notes & limitations

- Data quality depends on what Firecrawl is able to scrape from the source
  page at the time of the request; if a metric isn't available, the
  analyzer agent will note that in its summary rather than fabricating a
  precise number.
- This tool provides general precautionary guidance, not medical advice.
  Always consult a physician for decisions involving specific health
  conditions.
- Model defaults to `gpt-4o` — change the `id` in `build_agents()` in
  `app.py` if you'd like to use a different OpenAI model.

## Project structure

```
aqi-analysis-agent/
├── app.py              # Streamlit app + both Agno agents
├── requirements.txt
├── .env.example
└── README.md
```
