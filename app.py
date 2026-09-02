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


def glass_heading(emoji: str, text: str, tag: str = "h2", hero: bool = False) -> str:
    """Render a heading with a frosted-glass circular icon badge in front of it."""
    hero_class = " hero" if hero else ""
    return (
        f'<{tag} class="glass-heading{hero_class}">'
        f'<span class="glass-icon-badge">{emoji}</span>{text}'
        f"</{tag}>"
    )


def coerce_to_report(content) -> AQIReport:
    """
    The analyzer agent is asked for structured output, but depending on the
    agno/model version, combining tool calls with schema-constrained output
    can silently fall back to a plain string instead of an AQIReport
    instance. Handle both cases here instead of crashing downstream.
    """
    if isinstance(content, AQIReport):
        return content

    if isinstance(content, dict):
        return AQIReport(**content)

    if isinstance(content, str):
        text = content.strip()
        # Strip markdown code fences if the model wrapped the JSON in them.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                "The AQI Analyzer didn't return valid structured data "
                "(got plain text instead of JSON). Try again, or try a "
                "more specific location."
            ) from e
        return AQIReport(**data)

    raise ValueError(f"Unexpected analyzer output type: {type(content)}")


# --------------------------------------------------------------------------- #
# Agent builders (cached per API-key pair so we don't rebuild every rerun)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def build_agents(openai_api_key: str, firecrawl_api_key: str):
    model = OpenAIChat(id="gpt-4o", api_key=openai_api_key)

    # Split into two agents instead of one tool-using + schema-constrained
    # agent: combining live tool calls with strict structured output is
    # unreliable across agno/model versions and often falls back to prose.
    # A scraper agent (free text) feeding a parser agent (schema-only, no
    # tools) is far more consistent.
    scraper = Agent(
        name="AQI Scraper",
        model=model,
        tools=[FirecrawlTools(api_key=firecrawl_api_key, enable_scrape=True, enable_crawl=False)],
        instructions=dedent(
            """\
            You are an air-quality data researcher.
            Given a location, scrape a reliable live air-quality source using
            the firecrawl scrape tool (prefer https://www.iqair.com/<country>/
            <city> or https://aqicn.org/city/<city>, adjusting the URL slug
            to fit the location; try an alternate source if the first fails).

            Then write a short plain-text summary listing every value you
            found on the page for: overall AQI and its category, PM2.5,
            PM10, CO, temperature, humidity, and wind speed. Include the
            exact source URL you scraped. If a value truly isn't on the
            page, say so explicitly rather than guessing silently.
            """
        ),
        markdown=False,
    )

    parser = Agent(
        name="AQI Parser",
        model=model,
        output_schema=AQIReport,
        instructions=dedent(
            """\
            You convert a researcher's raw notes about air quality into a
            single structured AQIReport JSON object. Use the exact values
            given where present. Where a value is missing, make the best
            reasonable estimate from context and mention that in the
            summary field rather than fabricating false precision. Always
            populate every field. Output only the structured object.
            """
        ),
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

    return scraper, parser, recommender


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="AQI Analysis Agent", page_icon="🌬️", layout="wide")

# --------------------------------------------------------------------------- #
# Glassmorphism theme (sage green accent) — fonts, backdrop, glass cards,
# animation, and interactive hover/focus states
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --sage: #8CA37F;
        --sage-dark: #6E8560;
        --sage-light: #B7C9AC;
        --sage-deep: #3F4F36;
        --glass-bg: rgba(255, 255, 255, 0.28);
        --glass-bg-strong: rgba(255, 255, 255, 0.48);
        --glass-border: rgba(255, 255, 255, 0.5);
        --glass-shadow: 0 8px 32px rgba(110, 133, 96, 0.16);
        --glass-shadow-hover: 0 14px 40px rgba(110, 133, 96, 0.26);
    }

    html { font-size: 18px; }
    html, body, [class*="css"] { font-family: 'Fraunces', 'Inter', serif; }
    h1, h2, h3, h4, .glass-heading { font-family: 'Fraunces', serif !important; }
    p, span, div, label, li { font-family: 'Fraunces', 'Inter', serif; }

    h1 { font-size: 3.1rem !important; font-weight: 600 !important; }
    h2 { font-size: 2.1rem !important; font-weight: 600 !important; }
    h3 { font-size: 1.65rem !important; font-weight: 600 !important; }
    p, .stMarkdown, label { font-size: 1.08rem !important; line-height: 1.6; }

    /* ---------- Backdrop: gradient ---------- */
    .stApp {
        background: linear-gradient(135deg, #EAF0E3 0%, #DDE8D6 35%, #CFE0C6 65%, #E4EDE0 100%);
        background-attachment: fixed;
    }

    /* Decorative floating orbs, isolated in their own fixed layer so they
       can't become a containing-block trap for other fixed/backdrop-filter
       elements on the page. */
    .glass-orb-layer {
        position: fixed;
        inset: 0;
        z-index: 0;
        overflow: hidden;
        pointer-events: none;
    }
    .glass-orb-layer::before, .glass-orb-layer::after {
        content: "";
        position: absolute;
        border-radius: 50%;
        filter: blur(70px);
        opacity: 0.55;
    }
    .glass-orb-layer::before {
        width: 420px; height: 420px;
        background: radial-gradient(circle, var(--sage-light) 0%, transparent 70%);
        top: -120px; left: -100px;
        animation: floatOrb 16s ease-in-out infinite;
    }
    .glass-orb-layer::after {
        width: 500px; height: 500px;
        background: radial-gradient(circle, #A9C79A 0%, transparent 70%);
        bottom: -160px; right: -140px;
        animation: floatOrb 20s ease-in-out infinite reverse;
    }
    @keyframes floatOrb {
        0%, 100% { transform: translate(0, 0) scale(1); }
        50% { transform: translate(30px, -30px) scale(1.08); }
    }

    /* Page-load entrance (one orchestrated moment, not per-card) */
    section.main > div.block-container {
        animation: fadeInUp 0.6s ease-out;
        position: relative;
        z-index: 1;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] > div {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid var(--glass-border);
    }

    /* ---------- Headings ---------- */
    h1, h2, h3 { color: var(--sage-deep) !important; }
    h1 {
        background: linear-gradient(90deg, var(--sage-deep) 0%, var(--sage-dark) 55%, var(--sage) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.01em;
    }

    /* ---------- Glassmorphic icon badges ---------- */
    .glass-heading {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin: 0.2rem 0 0.6rem 0 !important;
    }
    .glass-icon-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        width: 2.6rem;
        height: 2.6rem;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0.2) 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid var(--glass-border);
        box-shadow: 0 4px 16px rgba(110, 133, 96, 0.28), inset 0 1px 1px rgba(255,255,255,0.6);
        font-size: 1.3rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-heading:hover .glass-icon-badge {
        transform: translateY(-2px) rotate(-4deg);
        box-shadow: 0 6px 20px rgba(110, 133, 96, 0.38), inset 0 1px 1px rgba(255,255,255,0.6);
    }
    .glass-heading.hero .glass-icon-badge {
        width: 3.4rem;
        height: 3.4rem;
        font-size: 1.7rem;
    }

    /* ---------- Glass cards: containers, metrics, expanders, alerts, charts ---------- */
    div[data-testid="stMetric"],
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stExpander"],
    .stAlert,
    div[data-testid="stPlotlyChart"] {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border) !important;
        border-radius: 18px !important;
        box-shadow: var(--glass-shadow);
        padding: 0.9rem;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }

    /* Interactive lift on hover for card-like containers */
    div[data-testid="stVerticalBlockBorderWrapper"]:hover,
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: var(--glass-shadow-hover);
        border-color: var(--sage-light) !important;
    }

    div[data-testid="stMetric"] label { color: var(--sage-dark) !important; font-weight: 600; font-size: 1.05rem !important; }
    div[data-testid="stMetricValue"] {
        color: var(--sage-deep) !important;
        font-family: 'Fraunces', serif !important;
        font-size: 2.2rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricDelta"] { font-size: 1rem !important; }
    div[data-testid="stMetricDelta"] svg { transform: scale(1.15); }

    /* ---------- Text / select inputs ---------- */
    div[data-baseweb="input"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] {
        background: var(--glass-bg-strong) !important;
        border-radius: 12px !important;
        border: 1px solid var(--glass-border) !important;
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    input, textarea { color: var(--sage-deep) !important; font-size: 1.05rem !important; }
    div[data-baseweb="select"]:focus-within,
    div[data-baseweb="base-input"]:focus-within {
        border: 1px solid var(--sage) !important;
        box-shadow: 0 0 0 3px rgba(140, 163, 127, 0.25) !important;
    }

    /* Dropdown / multiselect popover menus */
    ul[data-baseweb="menu"] {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(12px);
        border-radius: 12px !important;
        border: 1px solid var(--glass-border) !important;
    }
    li[data-baseweb="menu-item"]:hover,
    li[aria-selected="true"] {
        background: rgba(140, 163, 127, 0.18) !important;
        color: var(--sage-deep) !important;
    }
    span[data-baseweb="tag"] {
        background-color: var(--sage) !important;
        border-radius: 8px !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: linear-gradient(135deg, var(--sage) 0%, var(--sage-dark) 100%);
        color: #FFFFFF;
        border: 1px solid var(--glass-border);
        border-radius: 14px;
        font-weight: 600;
        font-size: 1.08rem;
        padding: 0.7rem 1.3rem;
        box-shadow: 0 6px 20px rgba(110, 133, 96, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 10px 26px rgba(110, 133, 96, 0.45);
        color: #FFFFFF;
        border-color: var(--sage-light);
        filter: brightness(1.05);
    }
    .stButton > button:active { transform: translateY(0px) scale(0.99); }

    .stDownloadButton > button {
        background: var(--glass-bg-strong);
        color: var(--sage-dark);
        border: 1.5px solid var(--sage);
        border-radius: 14px;
        font-weight: 600;
        font-size: 1.08rem;
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        background: var(--sage);
        color: #FFFFFF;
        transform: translateY(-2px);
    }

    /* ---------- Tabs ---------- */
    div[data-baseweb="tab-list"] {
        background: var(--glass-bg);
        backdrop-filter: blur(14px);
        border-radius: 14px;
        border: 1px solid var(--glass-border);
        padding: 0.4rem;
        gap: 0.3rem;
    }
    button[data-baseweb="tab"] {
        border-radius: 10px !important;
        color: var(--sage-dark) !important;
        font-weight: 600;
        font-size: 1.1rem !important;
        padding: 0.6rem 1.1rem !important;
        transition: background 0.2s ease, color 0.2s ease;
    }
    button[data-baseweb="tab"] p { font-size: 1.1rem !important; }
    button[data-baseweb="tab"]:hover { background: rgba(140, 163, 127, 0.15); }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: var(--sage) !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"] { display: none; }

    /* ---------- Spinner ---------- */
    div[data-testid="stSpinner"] > div { border-top-color: var(--sage) !important; }
    div[data-testid="stSpinner"] p { color: var(--sage-dark); }

    /* ---------- Alerts / captions ---------- */
    .stAlert { color: var(--sage-deep) !important; }
    div[data-testid="stCaptionContainer"] { color: var(--sage-dark) !important; }

    /* ---------- Divider ---------- */
    hr { border-color: rgba(140, 163, 127, 0.35) !important; }

    /* ---------- Markdown report block ---------- */
    div[data-testid="stMarkdownContainer"] { color: var(--sage-deep); }

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: var(--sage-light);
        border-radius: 10px;
        border: 2px solid transparent;
        background-clip: content-box;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--sage); background-clip: content-box; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="glass-orb-layer"></div>', unsafe_allow_html=True)

if "report" not in st.session_state:
    st.session_state.report = None
if "advice" not in st.session_state:
    st.session_state.advice = None

with st.container(border=True):
    st.markdown(glass_heading("🌬️", "AQI Analysis Agent", "h1", hero=True), unsafe_allow_html=True)
    st.caption(
        "Real-time air quality monitoring and personalized health recommendations, "
        "powered by Firecrawl + Agno."
    )

with st.sidebar:
    st.markdown(glass_heading("🔑", "API Keys", "h3"), unsafe_allow_html=True)
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
    st.markdown(glass_heading("👤", "Your Profile", "h3"), unsafe_allow_html=True)
    conditions = st.multiselect("Medical conditions (optional)", MEDICAL_CONDITIONS, default=["None"])
    activity = st.selectbox("Planned outdoor activity", ACTIVITIES)

with st.container(border=True):
    st.markdown(glass_heading("📍", "Location", "h3"), unsafe_allow_html=True)
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
            analyzer, parser, recommender = build_agents(openai_api_key, firecrawl_api_key)

            with st.spinner("📡 Fetching live air quality data..."):
                scraped = analyzer.run(f"Location: {location.strip()}")

            with st.spinner("🧮 Structuring the data..."):
                parsed = parser.run(scraped.content)
                try:
                    report = coerce_to_report(parsed.content)
                except ValueError:
                    with st.expander("Raw scraper output (debug)"):
                        st.code(str(scraped.content))
                    raise
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

# WHO 24-hour air-quality guideline levels, used to add context to raw readings.
WHO_PM2_5 = 15.0
WHO_PM10 = 45.0

if report:
    st.divider()
    st.markdown(
        glass_heading("📊", f"Current Air Quality — {report.location}", "h2"),
        unsafe_allow_html=True,
    )

    tab_overview, tab_pollutants, tab_advice = st.tabs(
        ["🌡️ Overview", "🧪 Pollutants", "🩺 Health Advice"]
    )

    with tab_overview:
        with st.container(border=True):
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
                fig.update_layout(
                    height=280,
                    margin=dict(t=40, b=10, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#3F4F36"),
                )
                st.plotly_chart(fig, use_container_width=True)

            with top2:
                m4, m5, m6 = st.columns(3)
                m4.metric("Temperature", f"{report.temperature_c} °C")
                m5.metric("Humidity", f"{report.humidity_pct}%")
                m6.metric("Wind Speed", f"{report.wind_speed_kmh} km/h")
                st.info(report.summary)
                st.caption(f"Source: {report.source_url}")

    with tab_pollutants:
        with st.container(border=True):
            p1, p2, p3 = st.columns(3)
            p1.metric(
                "PM2.5", f"{report.pm2_5} µg/m³",
                delta=f"{report.pm2_5 - WHO_PM2_5:+.1f} vs WHO guideline",
                delta_color="inverse",
            )
            p2.metric(
                "PM10", f"{report.pm10} µg/m³",
                delta=f"{report.pm10 - WHO_PM10:+.1f} vs WHO guideline",
                delta_color="inverse",
            )
            p3.metric("CO", f"{report.co}")

            pm_fig = go.Figure(
                data=[
                    go.Bar(
                        name="Level",
                        x=["PM2.5", "PM10", "CO"],
                        y=[report.pm2_5, report.pm10, report.co],
                        marker_color="#8CA37F",
                    )
                ]
            )
            pm_fig.update_layout(
                height=300,
                title="Pollutant Levels",
                margin=dict(t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#3F4F36"),
            )
            pm_fig.update_xaxes(gridcolor="rgba(140,163,127,0.2)")
            pm_fig.update_yaxes(gridcolor="rgba(140,163,127,0.2)")
            st.plotly_chart(pm_fig, use_container_width=True)
            st.caption("Deltas compare today's reading to WHO 24-hour air-quality guideline levels.")

    with tab_advice:
        with st.container(border=True):
            if advice:
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
            else:
                st.info("Recommendations weren't generated for this run.")

if not report:
    st.info("Enter a location above and click **Analyze Air Quality** to get started.")