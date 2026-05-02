import streamlit as st

# Set page config FIRST!
st.set_page_config(
    page_title=" SkyTrack: Satellite Info Explorer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import json
from tech import TechAgent
from basic import BasicInfoBot
from cost import CostBot
from data_manager import SatelliteDataManager
import pandas as pd
import os
from dotenv import load_dotenv
import gspread
from gspread_dataframe import set_with_dataframe
from google.oauth2.service_account import Credentials
from gpt_user import UserBot
from gpt_purpose import PurposeBot
from gpt_tech import TechBot
from gpt_frugal import FrugalBot
from gpt_numeric import NumericBot
from datetime import datetime
import time

# Load environment variables
load_dotenv()

# Initialize the data manager
@st.cache_resource
def get_data_manager():
    return SatelliteDataManager()

data_manager = get_data_manager()

# Custom CSS for a soft, light, and modern look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: #f2fbf2;
        color: #2c3e50;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #ffffff 0%, #f8fbf8 100%);
        color: #2c3e50;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        font-size: 1.8rem;
        font-weight: 700;
        border: 1px solid #e2e8f0;
        border-bottom: 4px solid #a7d8de;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .satellite-card {
        background: #ffffff;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #a7d8de;
        margin: 0.5rem 0 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        border: 1px solid #edf2f7;
    }
    .data-section {
        background: #ffffff;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03);
        margin: 0.8rem 0;
        transition: transform 0.2s ease;
    }
    .data-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        background: linear-gradient(135deg, #a7d8de 0%, #8ec9d2 100%);
        color: #1a202c;
        font-weight: 600;
        padding: 0.6rem;
        margin-bottom: 0.3rem;
        font-size: 1.05rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(167, 216, 222, 0.4);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #8ec9d2 0%, #7dbbc5 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(167, 216, 222, 0.6);
        color: #000000;
    }
    .stExpanderHeader {
        font-size: 1.05rem !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500;
        color: #2c3e50;
    }
    .stExpanderContent {
        padding-top: 0.5rem !important;
    }
    .stMarkdown code, .stCode, .stJson {
        font-family: 'Fira Code', 'Menlo', 'Monaco', monospace !important;
        font-size: 0.95rem;
        background: #f8fafc;
        color: #2c3e50;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
    }
    hr.tech-divider {
        border: none;
        border-top: 2px dashed #a7d8de;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

SHEET_ID = "1gWsnjIbK_c6oml5KytVbSQk7UF20P_0VAH6xSPm9Soc"
WORKSHEET_NAME = "Sheet1"

# Define consistent column order for Excel export
BASIC_INFO_COLUMNS = [
    'satellite_name', 'altitude', 'orbital_period', 'inclination', 'eccentricity',
    'launch_date', 'status', 'orbital_life', 'mass', 'power'
]

TECH_SPECS_COLUMNS = [
    'satellite_type', 'primary_mission', 'instruments', 'sensors', 'applications',
    'data_products', 'resolution', 'swath_width', 'frequency_bands'
]

LAUNCH_COST_COLUMNS = [
    'launch_vehicle', 'launch_site', 'launch_cost', 'development_cost',
    'total_mission_cost', 'launch_success', 'contractor', 'mission_duration'
]

GPT_COLUMNS = [
    'user_info', 'purpose_sdg', 'tech', 'frugal', 'numeric'
]

# Initialize session state with persistence
def init_session_state():
    """Initialize session state with proper data structure"""
    if 'satellite_name' not in st.session_state:
        st.session_state.satellite_name = ""
    
    if 'current_satellites' not in st.session_state:
        st.session_state.current_satellites = []
    
    if 'satellite_data' not in st.session_state:
        st.session_state.satellite_data = {}
    
    if 'gpt_data' not in st.session_state:
        st.session_state.gpt_data = {}
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = {}

# Load data from persistent storage
def load_satellite_data(satellite_name):
    """Load satellite data from persistent storage"""
    if satellite_name not in st.session_state.satellite_data:
        st.session_state.satellite_data[satellite_name] = {
            "basic_info": {},
            "technical_specs": {},
            "launch_cost_info": {}
        }
    if satellite_name not in st.session_state.gpt_data:
        st.session_state.gpt_data[satellite_name] = {
            "user_info": {},
            "purpose_sdg": {},
            "tech": {},
            "frugal": {},
            "numeric": {}
        }
    # Load from file system if not already loaded
    if satellite_name not in st.session_state.data_loaded:
        try:
            # Load basic data
            basic_data = data_manager.get_satellite_data(satellite_name, "basic_info")
            if basic_data and basic_data.get("data"):
                st.session_state.satellite_data[satellite_name]["basic_info"] = basic_data["data"]
            tech_data = data_manager.get_satellite_data(satellite_name, "technical_specs")
            if tech_data and tech_data.get("data"):
                st.session_state.satellite_data[satellite_name]["technical_specs"] = tech_data["data"]
            cost_data = data_manager.get_satellite_data(satellite_name, "launch_cost_info")
            if cost_data and cost_data.get("data"):
                st.session_state.satellite_data[satellite_name]["launch_cost_info"] = cost_data["data"]
            # Load GPT data sections
            for gpt_key in ["user_info", "purpose_sdg", "tech", "frugal", "numeric"]:
                gpt_section = data_manager.get_satellite_data(satellite_name, gpt_key)
                if gpt_section and gpt_section.get("data"):
                    st.session_state.gpt_data[satellite_name][gpt_key] = gpt_section["data"]
            st.session_state.data_loaded[satellite_name] = True
        except Exception as e:
            st.error(f"Error loading data for {satellite_name}: {str(e)}")

# Helper: Google Sheets client
@st.cache_resource
def get_gspread_client():
    try:
        creds_dict = st.secrets["google_service_account"]
        creds = Credentials.from_service_account_info(
            creds_dict, 
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"Failed to initialize Google Sheets client: {str(e)}")
        return None

def upload_to_gsheet(satellite_name, data_dict, sheet_name="Sheet1"):
    """Upload data to Google Sheets with proper column ordering"""
    try:
        client = get_gspread_client()
        if not client:
            return False
            
        sheet = client.open_by_key(SHEET_ID).worksheet(sheet_name)
        
        # Prepare row data with consistent column order
        row = {"satellite_name": satellite_name, "last_updated": datetime.now().isoformat()}
        
        if sheet_name == "Sheet1":
            # Order columns for main data
            for col in BASIC_INFO_COLUMNS + TECH_SPECS_COLUMNS + LAUNCH_COST_COLUMNS:
                row[col] = data_dict.get(col, "")
        else:
            # Order columns for GPT data
            for col in GPT_COLUMNS:
                if col in data_dict:
                    if isinstance(data_dict[col], dict):
                        for subk, subv in data_dict[col].items():
                            row[f"{col}_{subk}"] = subv
                    else:
                        row[col] = data_dict[col]
        
        df = pd.DataFrame([row])
        
        try:
            existing = pd.DataFrame(sheet.get_all_records())
            if existing.empty:
                set_with_dataframe(sheet, df)
            else:
                # Check if satellite already exists and update instead of append
                existing_index = existing[existing['satellite_name'] == satellite_name].index
                if not existing_index.empty:
                    # Update existing row
                    for col, val in row.items():
                        if col in existing.columns:
                            existing.loc[existing_index[0], col] = val
                    set_with_dataframe(sheet, existing)
                else:
                    # Append new row
                    set_with_dataframe(sheet, pd.concat([existing, df], ignore_index=True))
        except Exception as e:
            # If sheet is empty or has issues, create new
            set_with_dataframe(sheet, df)
        
        return True
    except Exception as e:
        st.error(f"Failed to upload to Google Sheets: {str(e)}")
        return False

# ── Live Reasoning Panel ─────────────────────────────────────────────────────

# ── Terminal-style Live Reasoning Panel ─────────────────────────────────────

STATUS_PREFIX = {
    "running": "...",
    "done":    "[OK]",
    "error":   "[ERR]",
    "warn":    "[WARN]",
}


class LiveReasoningPanel:
    """Appends each agent step as a line in a raw terminal-style code block."""

    def __init__(self, placeholder):
        """Pass an st.empty() placeholder."""
        self._ph = placeholder
        self._lines: list[str] = []

    def _render(self):
        self._ph.code("\n".join(self._lines), language="text")

    def __call__(self, step: dict):
        icon   = step.get("icon", "-")
        agent  = step.get("agent", "Agent")
        title  = step.get("title", "")
        detail = step.get("detail", "")
        prefix = STATUS_PREFIX.get(step.get("status", "running"), "...")
        # Build line: [OK] TechBot | LLM response received
        line = f"{prefix:<6} {agent} | {title}"
        if detail:
            line += f"\n       {' ' * len(agent)}   {detail}"
        self._lines.append(line)
        self._render()

    def clear(self):
        self._lines = []
        self._render()


# Enhanced tab rendering function with live reasoning steps
def render_tab(tab, satellite_name, data_key, bot_class, data_manager=None, session_key="satellite_data"):
    with tab:
        st.markdown("""
        <div style='background:#fff; border:1px solid #e3e7ee; box-shadow:0 2px 8px rgba(30,60,114,0.07); border-radius:8px; padding:1.2rem 1.5rem 1.2rem 1.5rem; margin-top:0;'>
        """, unsafe_allow_html=True)
        st.markdown(f"<h3 style='margin-bottom:0.7rem; color:#232526; font-weight:600; font-size:1.25rem;'>{data_key.replace('_', ' ').title()}</h3>", unsafe_allow_html=True)
        session_dict = st.session_state[session_key][satellite_name]
        if session_dict.get(data_key) and session_dict[data_key]:
            with st.expander("View Data", expanded=True):
                st.json(session_dict[data_key])
            col1, col2 = st.columns([1,1])
            with col1:
                json_str = json.dumps(session_dict[data_key], indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"{satellite_name}_{data_key}.json",
                    mime="application/json",
                    key=f"download_{session_key}_{data_key}_{satellite_name}"
                )
            with col2:
                if st.button(f"Delete {data_key.replace('_', ' ').title()}", key=f"delete_{session_key}_{data_key}_{satellite_name}"):
                    session_dict[data_key] = {}
                    if data_manager:
                        try:
                            data_manager.delete_satellite_section(satellite_name, data_key)
                        except Exception:
                            pass
                    st.success(f"{data_key.replace('_', ' ').title()} deleted.")
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("No data available. Click below to gather information.")
            if st.button(f"Run {data_key.replace('_', ' ').title()}", key=f"gather_{session_key}_{data_key}_{satellite_name}"):
                log_ph = st.empty()
                result_placeholder = st.empty()
                panel = LiveReasoningPanel(log_ph)
                try:
                    bot = bot_class()
                    result = bot.process_satellite(satellite_name, step_callback=panel)
                    if result:
                        session_dict[data_key] = result
                        if data_manager:
                            data_manager.append_satellite_data(satellite_name, data_key, result)
                        result_placeholder.success(f"✅ {data_key.replace('_', ' ').title()} gathered successfully!")
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        result_placeholder.error("Agent returned no data.")
                except Exception as e:
                    result_placeholder.error(f"Error: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)

# Initialize session state
init_session_state()

# Sidebar for satellite selection
with st.sidebar:
    st.markdown("## 🛰️ Satellite Selection")
    
    # Add satellites section
    st.markdown("### ➕ Add Satellites")
    satellite_input = st.text_area(
        "Enter Satellite Names (one per line)", 
        value="\n".join(st.session_state.current_satellites),
        height=100,
        placeholder="e.g.\nHubble Space Telescope\nISSR Sentinel-1A"
    )
    
    if st.button("📝 Process Satellites", type="primary"):
        new_satellites = [name.strip() for name in satellite_input.split('\n') if name.strip()]
        st.session_state.current_satellites = list(set(new_satellites))  # Remove duplicates
        if st.session_state.current_satellites:
            st.session_state.satellite_name = st.session_state.current_satellites[0]
        st.success(f"✅ {len(st.session_state.current_satellites)} satellites added!")
        time.sleep(1)
        st.rerun()
    
    # Current session satellites
    if st.session_state.current_satellites:
        st.markdown("### 🔄 Current Session")
        for sat in st.session_state.current_satellites:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"🛰️ {sat}", key=f"current_select_{sat}"):
                    st.session_state.satellite_name = sat
                    load_satellite_data(sat)
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"current_delete_{sat}"):
                    st.session_state.current_satellites.remove(sat)
                    if st.session_state.satellite_name == sat:
                        st.session_state.satellite_name = st.session_state.current_satellites[0] if st.session_state.current_satellites else ""
                    st.rerun()
    
    # Previously searched satellites
    existing_satellites = data_manager.get_all_satellites()
    if existing_satellites:
        st.markdown("### 📚 Previously Searched")
        for sat in existing_satellites:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"📂 {sat}", key=f"select_sat_{sat}"):
                    st.session_state.satellite_name = sat
                    load_satellite_data(sat)
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_sat_{sat}"):
                    data_manager.delete_satellite_data(sat)
                    if st.session_state.satellite_name == sat:
                        st.session_state.satellite_name = ""
                    st.rerun()
    
    # Download all data
    st.markdown("### 📥 Export Data")
    file_path = "satellite_data.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            all_satellite_data = f.read()
        st.download_button(
            label="📥 Download All Data",
            data=all_satellite_data,
            file_name="satellite_data.json",
            mime="application/json"
        )

# Main content area
st.markdown("""
<div class='main-header'>
    <span>SkyTrack: Satellite Info Explorer</span>
    <div style='font-size:1.08rem; font-weight:400; margin-top:0.2rem;'>Satellite analytics and AI-driven insights</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.satellite_name:
    satellite_name = st.session_state.satellite_name
    load_satellite_data(satellite_name)
    st.markdown(f"""
    <div class='satellite-card'>
        <h2 style='margin-bottom:0.2rem; font-weight:600; color:#232526;'>{satellite_name}</h2>
        <div style='font-size:1rem; color:#555;'>Active Analysis Session &nbsp;|&nbsp; Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
    """, unsafe_allow_html=True)
    # Master Extract All Data Button
    if st.button("🚀 Extract All Satellite Data (Automated)"):
        # ── pipeline definition ────────────────────────────────────────────────
        PIPELINE = [
            ("basic_info",      BasicInfoBot, "satellite_data", "🛰️ Basic Info"),
            ("technical_specs", TechAgent,    "satellite_data", "⚙️ Technical Specs"),
            ("launch_cost_info",CostBot,       "satellite_data", "💰 Launch & Cost"),
            ("user_info",       UserBot,       "gpt_data",       "👤 User Info"),
            ("purpose_sdg",     PurposeBot,    "gpt_data",       "🌍 Purpose & SDG"),
            ("tech",            TechBot,       "gpt_data",       "🔬 Advanced Tech"),
            ("frugal",          FrugalBot,     "gpt_data",       "💡 Frugal Insights"),
            ("numeric",         NumericBot,    "gpt_data",       "📊 Numeric Insights"),
        ]
        total = len(PIPELINE)

        # ── outer progress bar ─────────────────────────────────────────────────
        overall_bar = st.progress(0, text="Preparing agents…")
        agent_label = st.empty()
        log_ph      = st.empty()   # single placeholder reused across agents
        result_msg  = st.empty()

        all_ok = True
        for idx, (data_key, bot_class, sess_key, label) in enumerate(PIPELINE):
            frac = idx / total
            overall_bar.progress(frac, text=f"Running agent {idx+1}/{total}: {label}")
            agent_label.markdown(
                f"<div style='text-align:center; font-size:1rem; font-weight:600; "
                f"color:#2c3e50; padding:4px 0;'>{label}</div>",
                unsafe_allow_html=True
            )
            panel = LiveReasoningPanel(log_ph)
            try:
                bot    = bot_class()
                result = bot.process_satellite(satellite_name, step_callback=panel)
                st.session_state[sess_key][satellite_name][data_key] = result
                if data_manager:
                    data_manager.append_satellite_data(satellite_name, data_key, result)
            except Exception as e:
                result_msg.error(f"❌ Error in {label}: {e}")
                all_ok = False

        overall_bar.progress(1.0, text="All agents finished!")
        agent_label.empty()
        if all_ok:
            result_msg.success("🎉 All data extracted successfully! Reloading…")
        time.sleep(1.5)
        st.rerun()

    st.markdown("### 📊 Comprehensive Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4 style='color:#3b9ca7; margin-bottom:1rem;'>Core Operations</h4>", unsafe_allow_html=True)
        render_tab(st.container(), satellite_name, "basic_info", BasicInfoBot, data_manager, session_key="satellite_data")
        render_tab(st.container(), satellite_name, "technical_specs", TechAgent, data_manager, session_key="satellite_data")
        render_tab(st.container(), satellite_name, "launch_cost_info", CostBot, data_manager, session_key="satellite_data")
        
        st.markdown("<div class='data-section'>", unsafe_allow_html=True)
        st.subheader("Combined Raw Data")
        satellite_data = st.session_state.satellite_data.get(satellite_name, {})
        if any(satellite_data.values()) and any(v for v in satellite_data.values() if v):
            with st.expander("View Combined JSON", expanded=False):
                st.json(satellite_data)
            bc1, bc2 = st.columns(2)
            with bc1:
                json_str = json.dumps(satellite_data, indent=2)
                st.download_button(label="Download JSON", data=json_str, file_name=f"{satellite_name}_core_data.json", mime="application/json")
            with bc2:
                if st.button("Upload to Google Sheet"):
                    with st.spinner("Uploading to Google Sheets..."):
                        combined_data = {}
                        for section in ["basic_info", "technical_specs", "launch_cost_info"]:
                            if satellite_data.get(section):
                                combined_data.update(satellite_data[section])
                        if upload_to_gsheet(satellite_name, combined_data):
                            st.success("Uploaded!")
                        else:
                            st.error("Upload Failed")
        else:
            st.info("No combined data available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h4 style='color:#3b9ca7; margin-bottom:1rem;'>AI Insights</h4>", unsafe_allow_html=True)
        render_tab(st.container(), satellite_name, "user_info", UserBot, session_key="gpt_data")
        render_tab(st.container(), satellite_name, "purpose_sdg", PurposeBot, session_key="gpt_data")
        render_tab(st.container(), satellite_name, "tech", TechBot, session_key="gpt_data")
        render_tab(st.container(), satellite_name, "frugal", FrugalBot, session_key="gpt_data")
        render_tab(st.container(), satellite_name, "numeric", NumericBot, session_key="gpt_data")
        
        st.markdown("<div class='data-section'>", unsafe_allow_html=True)
        st.subheader("Combined AI Insight Data")
        gpt_data = st.session_state.gpt_data.get(satellite_name, {})
        if any(gpt_data.values()) and any(v for v in gpt_data.values() if v):
            with st.expander("View Combined GPT JSON", expanded=False):
                st.json(gpt_data)
            gc1, gc2 = st.columns(2)
            with gc1:
                json_str = json.dumps(gpt_data, indent=2)
                st.download_button(label="Download AI JSON", data=json_str, file_name=f"{satellite_name}_gpt_data.json", mime="application/json")
            with gc2:
                if st.button("Upload AI to Sheet2"):
                    with st.spinner("Uploading GPT data..."):
                        if upload_to_gsheet(satellite_name, gpt_data, "Sheet2"):
                            st.success("Uploaded!")
                        else:
                            st.error("Upload Failed")
        else:
            st.info("No specific AI data available.")
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class='data-section' style='max-width: 700px; margin: 2rem auto 0 auto;'>
        <h2 style='text-align:center; margin-bottom:0.5rem; font-weight:600; color:#232526;'>Welcome to <span style="color:#3b9ca7">SkyTrack</span></h2>
        <p style='text-align:center; margin-bottom:1.1rem; color:#232526;'>
            <b>Start by adding or selecting a satellite in the sidebar.</b>
        </p>
        <ul style='font-size:1.03rem; margin-bottom:0.7rem; color:#232526;'>
            <li><b>Data Tab's :</b> Basic info, technical specs, launch & cost</li>
            <li><b>GPT Data Tab's:</b> User info, purpose, tech, cost, numeric</li>
            <li><b>Persistence:</b> Data saved across sessions</li>
            <li><b>Export:</b> Download JSON, upload to Google Sheets</li>
            <li><b>Multi-Satellite:</b> Manage multiple satellites</li>
        </ul>
        <hr class='tech-divider'>
        <details style='margin-top:0.5rem;'>
            <summary style='font-size:1rem; color:#3b9ca7; cursor:pointer;'>Column Order for Excel Export</summary>
            <div style='font-size:0.98rem; margin-top:0.3rem; color:#232526;'>
                <b>Basic:</b> {basic}<br>
                <b>Technical:</b> {tech}<br>
                <b>Launch/Cost:</b> {cost}<br>
                <b>GPT Data:</b> {gpt}
            </div>
        </details>
    </div>
    """.format(
        basic=", ".join(BASIC_INFO_COLUMNS),
        tech=", ".join(TECH_SPECS_COLUMNS),
        cost=", ".join(LAUNCH_COST_COLUMNS),
        gpt=", ".join(GPT_COLUMNS)
    ), unsafe_allow_html=True)