"""
AQI Analysis Agent
-------------------
A multi-agent air quality monitoring and health recommendation tool built with
Streamlit, Agno (AI agent framework), Firecrawl (web scraping) and an OpenAI model.

Agents:
  1. AQI Analyzer      - scrapes a live air-quality page for the requested location
                          and extracts it into a structured AQIReport.
  2. Health Recommender - turns that structured data (+ the user's medical
                          conditions and planned activity) into personalized,
                          actionable health guidance.

Run with:
    streamlit run app.py
"""

import json
import os
from datetime import datetime
from textwrap import dedent

import plotly.graph_objects as go
import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.firecrawl import FirecrawlTools
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load OPENAI_API_KEY / FIRECRAWL_API_KEY from a .env file in the project root,
# if one exists. These act as defaults; anything typed into the sidebar
# overrides them for the running session.
load_dotenv()


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
class AQIReport(BaseModel):
    location: str = Field(..., description="The resolved location name, e.g. 'Delhi, India'")
    aqi: int = Field(..., description="Overall Air Quality Index value")
    aqi_category: str = Field(
        ..., description="AQI category, e.g. Good / Moderate / Unhealthy / Hazardous"
    )
    pm2_5: float = Field(..., description="PM2.5 concentration in µg/m³")
    pm10: float = Field(..., description="PM10 concentration in µg/m³")
    co: float = Field(..., description="Carbon monoxide level (ppb or µg/m³, as reported)")
    temperature_c: float = Field(..., description="Temperature in Celsius")
    humidity_pct: float = Field(..., description="Relative humidity in percent")
    wind_speed_kmh: float = Field(..., description="Wind speed in km/h")
    source_url: str = Field(..., description="The URL the data was scraped from")
    summary: str = Field(..., description="One or two sentence plain-language summary")


EXAMPLE_QUERIES = [
    "Delhi, India",
    "Los Angeles, USA",
    "Beijing, China",
    "London, UK",
    "Jakarta, Indonesia",
]

MEDICAL_CONDITIONS = [
    "None",
    "Asthma",
    "COPD",
    "Heart disease",
    "Pregnancy",
    "Child (under 12)",
    "Elderly (65+)",
    "Allergies / sinus issues",
]

ACTIVITIES = [
    "General outdoor errands",
    "Walking / light stroll",
    "Running / jogging",
    "Cycling",
    "Team sports / high-intensity exercise",
    "Outdoor work / manual labor",
    "Sightseeing with kids",
]

AQI_COLOR_SCALE = [
    (0, 50, "#00e400", "Good"),
    (51, 100, "#ffff00", "Moderate"),
    (101, 150, "#ff7e00", "Unhealthy for Sensitive Groups"),
    (151, 200, "#ff0000", "Unhealthy"),
    (201, 300, "#8f3f97", "Very Unhealthy"),
    (301, 500, "#7e0023", "Hazardous"),
]


def aqi_color(aqi: int) -> str:
    for lo, hi, color, _ in AQI_COLOR_SCALE:
        if lo <= aqi <= hi:
            return color
    return "#7e0023"


# --------------------------------------------------------------------------- #
# Agent builders (cached per API-key pair so we don't rebuild every rerun)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def build_agents(openai_api_key: str, firecrawl_api_key: str):
    model = OpenAIChat(id="gpt-4o", api_key=openai_api_key)

    analyzer = Agent(
        name="AQI Analyzer",
        model=model,
        tools=[FirecrawlTools(api_key=firecrawl_api_key, scrape=True, crawl=False)],
        response_model=AQIReport,
        instructions=dedent(
            """\
            You are an air-quality data analyst.
            Given a location, scrape a reliable live air-quality source
            (prefer https://www.iqair.com/<country>/<city> or
            https://aqicn.org/city/<city>, falling back to a web search style
            URL guess if needed) using the firecrawl scrape tool.

            From the scraped page, extract:
            - overall AQI value and its category
            - PM2.5 and PM10 (µg/m³)
            - CO level
            - temperature (°C), humidity (%), wind speed (km/h)

            If a value is not explicitly present, make the best reasonable
            estimate from what IS on the page and note that in the summary
            rather than inventing precise numbers. Always populate every field.
            Return ONLY the structured AQIReport object.
            """
        ),
        markdown=False,
    )

    recommender = Agent(
        name="Health Recommendation Agent",
        model=model,
        instructions=dedent(
            """\
            You are a cautious, practical public-health advisor focused on
            air-quality exposure. You are not a doctor and must not diagnose
            or prescribe medication; you give general precautionary guidance
            and always suggest consulting a physician for personal medical
            decisions when relevant.

            Given structured air-quality data, the user's medical conditions
            (if any), and their planned activity, produce a clear Markdown
            report with these sections:

            ## Health Impact Assessment
            Plain-language explanation of what today's readings mean for
            general health and specifically for the stated medical conditions.

            ## Is [activity] Safe Right Now?
            A direct verdict (Safe / Caution / Avoid) with reasoning tied to
            the specific pollutant levels.

            ## Recommendations
            Concrete, numbered precautions (masks, timing, indoor
            alternatives, hydration, medication reminders if relevant, etc).

            ## Best Time for Outdoor Activity Today
            Suggest a time window based on typical pollution patterns
            (mornings/evenings vs midday traffic peaks) and note this is a
            general heuristic, not a live forecast.

            ## Weather Correlation
            Briefly note how temperature/humidity/wind are likely affecting
            pollutant dispersion right now.

            Keep the tone calm and reassuring where warranted, but be
            direct about real risks. Keep the whole report under ~350 words.
            """
        ),
        markdown=True,
    )

    return analyzer, recommender


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="AQI Analysis Agent", page_icon="🌬️", layout="wide")

if "report" not in st.session_state:
    st.session_state.report = None
if "advice" not in st.session_state:
    st.session_state.advice = None

st.title("🌬️ AQI Analysis Agent")
st.caption(
    "Real-time air quality monitoring and personalized health recommendations, "
    "powered by Firecrawl + Agno."
)

with st.sidebar:
    st.header("🔑 API Keys")
    env_openai_key = os.getenv("OPENAI_API_KEY", "")
    env_firecrawl_key = os.getenv("FIRECRAWL_API_KEY", "")

    openai_api_key = st.text_input("OpenAI API Key", value=env_openai_key, type="password")
    firecrawl_api_key = st.text_input("Firecrawl API Key", value=env_firecrawl_key, type="password")

    if env_openai_key and env_firecrawl_key:
        st.caption("✅ Keys auto-loaded from your .env file.")
    else:
        st.caption(
            "Keys are used only for this session and are never stored. "
            "Get a Firecrawl key at firecrawl.dev and an OpenAI key at platform.openai.com. "
            "Tip: add them to a .env file to auto-fill this next time."
        )
    st.divider()
    st.header("👤 Your Profile")
    conditions = st.multiselect("Medical conditions (optional)", MEDICAL_CONDITIONS, default=["None"])
    activity = st.selectbox("Planned outdoor activity", ACTIVITIES)

st.subheader("📍 Location")
col_loc, col_ex = st.columns([3, 2])
with col_loc:
    location = st.text_input(
        "City / location", value=st.session_state.get("location_prefill", ""),
        placeholder="e.g. Delhi, India"
    )
with col_ex:
    st.caption("Try an example:")
    ex_cols = st.columns(len(EXAMPLE_QUERIES))
    for i, ex in enumerate(EXAMPLE_QUERIES):
        if ex_cols[i].button(ex, use_container_width=True):
            st.session_state.location_prefill = ex
            st.rerun()

analyze_clicked = st.button("🔍 Analyze Air Quality", type="primary", use_container_width=True)

if analyze_clicked:
    if not openai_api_key or not firecrawl_api_key:
        st.error("Please add both your OpenAI and Firecrawl API keys in the sidebar.")
    elif not location.strip():
        st.error("Please enter a location.")
    else:
        try:
            analyzer, recommender = build_agents(openai_api_key, firecrawl_api_key)

            with st.spinner("📡 Fetching and analyzing live air quality data..."):
                analysis = analyzer.run(f"Location: {location.strip()}")
                report: AQIReport = analysis.content
                st.session_state.report = report

            with st.spinner("🩺 Generating personalized health recommendations..."):
                cond_text = ", ".join(conditions) if conditions else "None specified"
                prompt = dedent(
                    f"""\
                    Air quality data (JSON):
                    {report.model_dump_json(indent=2)}

                    User medical conditions: {cond_text}
                    Planned activity: {activity}
                    """
                )
                advice = recommender.run(prompt)
                st.session_state.advice = advice.content

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
report = st.session_state.report
advice = st.session_state.advice

if report:
    st.divider()
    st.subheader(f"📊 Current Air Quality — {report.location}")

    top1, top2 = st.columns([1, 2])
    with top1:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=report.aqi,
                title={"text": f"AQI — {report.aqi_category}"},
                gauge={
                    "axis": {"range": [0, 500]},
                    "bar": {"color": aqi_color(report.aqi)},
                    "steps": [
                        {"range": [lo, hi], "color": color}
                        for lo, hi, color, _ in AQI_COLOR_SCALE
                    ],
                },
            )
        )
        fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with top2:
        m1, m2, m3 = st.columns(3)
        m1.metric("PM2.5", f"{report.pm2_5} µg/m³")
        m2.metric("PM10", f"{report.pm10} µg/m³")
        m3.metric("CO", f"{report.co}")
        m4, m5, m6 = st.columns(3)
        m4.metric("Temperature", f"{report.temperature_c} °C")
        m5.metric("Humidity", f"{report.humidity_pct}%")
        m6.metric("Wind Speed", f"{report.wind_speed_kmh} km/h")
        st.info(report.summary)

    pm_fig = go.Figure(
        data=[
            go.Bar(name="Level", x=["PM2.5", "PM10", "CO"], y=[report.pm2_5, report.pm10, report.co])
        ]
    )
    pm_fig.update_layout(height=300, title="Pollutant Levels", margin=dict(t=40, b=10))
    st.plotly_chart(pm_fig, use_container_width=True)

    st.caption(f"Source: {report.source_url}")

if advice:
    st.divider()
    st.subheader("🩺 Personalized Health Recommendations")
    st.markdown(advice)

    report_text = dedent(
        f"""\
        AQI ANALYSIS REPORT
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        Location: {report.location}
        Medical conditions considered: {", ".join(conditions) if conditions else "None"}
        Planned activity: {activity}

        --- AIR QUALITY DATA ---
        AQI: {report.aqi} ({report.aqi_category})
        PM2.5: {report.pm2_5} µg/m³
        PM10: {report.pm10} µg/m³
        CO: {report.co}
        Temperature: {report.temperature_c} °C
        Humidity: {report.humidity_pct}%
        Wind Speed: {report.wind_speed_kmh} km/h
        Source: {report.source_url}

        --- HEALTH RECOMMENDATIONS ---
        {advice}
        """
    )
    st.download_button(
        "⬇️ Download Full Report",
        data=report_text,
        file_name=f"aqi_report_{report.location.replace(', ', '_').replace(' ', '_')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

if not report:
    st.info("Enter a location above and click **Analyze Air Quality** to get started.")