# 🛰️ SkyTrack — Satellite Intelligence Explorer (`cjbs_tool_v3`)

> **AI-powered satellite research platform** that autonomously gathers, structures and exports deep technical intelligence for any satellite — Basic specs, sensor data, launch cost, mission purpose, SDG alignment, user category and more.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51-red?logo=streamlit)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green)](https://langchain.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.1-orange)](https://groq.com)
[![Tavily](https://img.shields.io/badge/Search-Tavily-blueviolet)](https://tavily.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 Overview

**SkyTrack** is a research-grade Streamlit application built at the intersection of **agentic AI** and **space data science**. Given any satellite name, the platform spins up specialised AI agents that:

1. **Search the web** programmatically (Tavily)
2. **Extract structured data** with a Groq-hosted LLM (Llama-3.1-8b-instant)
3. **Persist** results in a local JSON store
4. **Export** to JSON files or push to Google Sheets

The system is designed for analysts, researchers, and space-tech teams who need reliable, sourced, structured satellite intelligence at scale.

---

## 🚀 Features

| Feature | Details |
|---|---|
| 🔍 **8 Specialised Agents** | Basic Info, Tech Specs, Launch & Cost, User Category, Mission Purpose, Advanced Tech, Frugal/Cost Innovation, Numeric Metrics |
| 🤖 **Groq LLM Backend** | Blazing-fast inference with Llama-3.1-8b-instant — no OpenAI quota limits |
| 🌐 **Tavily Web Search** | Real-time programmatic search across ESA, NASA, Wikipedia, NextSpaceFlight, SpaceNews |
| 📊 **Comprehensive Dashboard** | Two-column layout — Core Operations vs AI Insights |
| 💾 **Session Persistence** | Data saved to `satellite_data.json`, survives browser refresh |
| 📤 **Multi-format Export** | Download per-agent JSON files or bulk-upload to Google Sheets (Sheet1, Sheet2) |
| 🌍 **SDG Mapping** | Purpose agent maps each satellite to UN Sustainable Development Goals |
| 🔢 **Numeric Insights** | Numeric agent extracts quantitative mission KPIs |
| 🔄 **One-click Extract All** | Single button runs all 8 agents sequentially |
| 🗑️ **Full CRUD** | Add, delete, re-fetch per satellite or per data section |

---

## 🏗️ Architecture

```
SkyTrack/
├── app.py               # Streamlit UI — dashboard, session state, routing
├── agent_base.py        # SatelliteAgentBase — 2-step search+extract pipeline
│
├── basic.py             # Agent: Altitude, orbit class, payloads, orbital life
├── tech.py              # Agent: Satellite type, application, sensors, breakthroughs
├── cost.py              # Agent: Launch vehicle, cost, site, mass, reusability
│
├── gpt_user.py          # AI Agent: User/operator category (Military/Civil/Commercial/…)
├── gpt_purpose.py       # AI Agent: Mission purpose & SDG alignment
├── gpt_tech.py          # AI Agent: Advanced technology insights
├── gpt_frugal.py        # AI Agent: Cost-efficiency & frugal innovation analysis
├── gpt_numeric.py       # AI Agent: Quantitative mission metrics
│
├── data_manager.py      # JSON persistence layer (CRUD for satellite_data.json)
├── requirements.txt     # Python dependencies
└── .env.example         # API key template
```

### Agent Pipeline (per request)

```
User clicks "Run [Agent]"
        │
        ▼
  TavilySearch.invoke(query)   ← programmatic web search
        │
        ▼
  ChatGroq.invoke(prompt + context)   ← LLM structured extraction
        │
        ▼
  _extract_json(response)      ← regex JSON parser
        │
        ▼
  SatelliteDataManager.append_satellite_data()   ← persist to disk
        │
        ▼
  st.session_state update → UI re-renders with data
```

---

## ⚙️ Setup & Installation

### 1. Clone

```bash
git clone https://github.com/shubham21-ai/cjbs_tool_v3.git
cd cjbs_tool_v3
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key

# Optional — LangSmith tracing
LANGCHAIN_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=skytrack
```

> **Get API keys:**
> - Groq (free): https://console.groq.com
> - Tavily (free tier): https://app.tavily.com

### 4. Run

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🖥️ Usage Guide

### Adding a Satellite

1. In the **sidebar** → "➕ Add Satellites"
2. Type satellite names, one per line (e.g. `Sentinel-2A`, `Landsat 9`)
3. Click **📝 Process Satellites**
4. Select the satellite from "🔄 Current Session"

### Running Agents

- Click **Run [Agent Name]** under any section
- Or click **🚀 Extract All Satellite Data (Automated)** to run all 8 agents at once
- Agent logs are shown inline during execution

### Exporting Data

| Method | How |
|---|---|
| Per-section JSON | "Download JSON" button under each section |
| Full raw JSON | Sidebar → "📥 Download All Data" |
| Google Sheets | "Upload to Google Sheet" / "Upload AI to Sheet2" buttons (requires secrets config) |

### Programmatic API

```python
from basic import BasicInfoBot
from tech import TechAgent
from cost import CostBot
from gpt_user import UserBot
from gpt_purpose import PurposeBot

satellite = "Sentinel-2A"

basic_info  = BasicInfoBot().process_satellite(satellite)
tech_specs  = TechAgent().process_satellite(satellite)
launch_cost = CostBot().process_satellite(satellite)
user_info   = UserBot().process_satellite(satellite)
purpose     = PurposeBot().process_satellite(satellite)

print(basic_info)
# → {"altitude": "786 km", "orbital_life_years": "7 years", "launch_orbit_classification": "SSO", ...}
```

---

## 🤖 Agent Reference

| Agent Class | File | Data Extracted |
|---|---|---|
| `BasicInfoBot` | `basic.py` | Altitude, orbital life, orbit class, payload count |
| `TechAgent` | `tech.py` | Satellite type, application, sensor specs, breakthroughs |
| `CostBot` | `cost.py` | Launch cost, vehicle, date, site, mass, reusability, mission cost |
| `UserBot` | `gpt_user.py` | Operator category (1=Military … 5=Mix), description |
| `PurposeBot` | `gpt_purpose.py` | Mission purpose, SDG alignment |
| `TechBot` | `gpt_tech.py` | Advanced technology deep-dive |
| `FrugalBot` | `gpt_frugal.py` | Cost-efficiency, frugal innovation score |
| `NumericBot` | `gpt_numeric.py` | Quantitative operational metrics |

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Groq API key for Llama-3.1 inference |
| `TAVILY_API_KEY` | ✅ | Tavily Search API key |
| `LANGCHAIN_API_KEY` | ❌ | LangSmith tracing (optional) |
| `LANGSMITH_PROJECT` | ❌ | LangSmith project name |
| `GOOGLE_API_KEY` | ❌ | Unused (legacy, can be removed) |
| `SERPAPI_API_KEY` | ❌ | Unused (legacy, can be removed) |

---

## 📦 Dependencies

```
streamlit          # Web UI
langchain          # Agent framework
langchain-groq     # Groq LLM integration
langchain-tavily   # Tavily search integration
langgraph          # Agent graph execution
langchain-community
python-dotenv      # .env file loading
tavily-python      # Tavily direct client
pandas             # Data manipulation
requests           # HTTP client
beautifulsoup4     # HTML parsing
gspread            # Google Sheets API
gspread_dataframe  # DataFrame ↔ Sheets helper
google-auth        # Google credentials
tenacity           # Retry logic
```

---

## 🗺️ Roadmap

- [ ] Batch processing for entire satellite constellations
- [ ] CSV / Excel export
- [ ] Comparison dashboard (side-by-side multi-satellite)
- [ ] Automated scheduled refresh
- [ ] NORAD / Celestrak live TLE data integration
- [ ] Confidence scores per extracted field
- [ ] Streamlit Cloud one-click deployment template

---

## 🤝 Contributing

```bash
# 1. Fork & clone
git clone https://github.com/shubham21-ai/cjbs_tool_v3.git

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make changes & commit
git commit -m "feat: your feature description"

# 4. Push & open PR
git push origin feature/your-feature-name
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **[Groq](https://groq.com)** — Ultra-fast LLM inference (Llama-3.1)
- **[Tavily](https://tavily.com)** — AI-first search API
- **[Streamlit](https://streamlit.io)** — Rapid Python web apps
- **[LangChain](https://langchain.com)** — Agent orchestration framework
- **[ESA / NASA / NextSpaceFlight](https://nextspaceflight.com)** — Primary data sources

---

<div align="center">
  <b>Built with ❤️ for the space research community</b><br>
  <a href="https://github.com/shubham21-ai/cjbs_tool_v3">⭐ Star this repo if you find it useful!</a>
</div>
