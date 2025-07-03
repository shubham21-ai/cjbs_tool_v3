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
import sys
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
    html, body, [class*="css"]  {
        font-family: 'Inter', 'Roboto', 'Segoe UI', Arial, sans-serif !important;
        background: #f2fbf2; /* Light bright green background */
    }
    .main-header {
        text-align: center;
        background: #f2fbf2; /* Match page background */
        color: #333333;
        padding: 1.1rem 0.5rem 1.1rem 0.5rem;
        border-radius: 10px;
        margin-bottom: 1.1rem;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        border-bottom: 2px solid #a7d8de; /* Soft blue accent */
        box-shadow: none; /* Removed shadow for seamless look */
    }
    .satellite-card {
        background: #f2fbf2;
        padding: 0.7rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #a7d8de; /* Soft blue accent */
        margin: 0.5rem 0 1rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .data-section {
        background: #fff;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        border: 1px solid #e3e7ee;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin: 0.7rem 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        border: none;
        background: #a7d8de; /* Soft light blue */
        color: #232526; /* Dark text for high contrast */
        font-weight: 700; /* Bolder text for readability */
        margin-bottom: 0.2rem;
        font-size: 1rem;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: #8ec9d2; /* Slightly darker blue for hover */
        color: #000000; /* Ensure contrast on hover too */
    }
    .stExpanderHeader {
        font-size: 1rem !important;
        font-family: 'Inter', 'Roboto', 'Segoe UI', Arial, sans-serif !important;
    }
    .stExpanderContent {
        padding-top: 0.2rem !important;
    }
    .stMarkdown code, .stCode, .stJson {
        font-family: 'Fira Mono', 'Menlo', 'Monaco', 'Consolas', monospace !important;
        font-size: 0.98rem;
        background: #f4f6fa;
        color: #232526;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 500;
        color: #232526;
        padding: 0.5rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid #a7d8de; /* Soft blue accent */
        color: #3b9ca7;
        background: #f2fbf2;
    }
    hr.tech-divider {
        border: none;
        border-top: 1.5px solid #e3e7ee;
        margin: 1.2rem 0 1.2rem 0;
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

# Helper: CaptureStdout for agent logs
class CaptureStdout:
    def __init__(self, container):
        self.container = container
        self.placeholder = container.empty()
        self.output = []
    
    def write(self, text):
        if text.strip():
            self.output.append(text)
            try:
                self.placeholder.code(''.join(self.output), language="text")
            except Exception:
                try:
                    self.placeholder = self.container.empty()
                    self.placeholder.code(''.join(self.output), language="text")
                except:
                    pass
    
    def flush(self):
        try:
            if self.output:
                self.placeholder.code(''.join(self.output), language="text")
        except:
            pass

# Enhanced tab rendering function
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
                with st.spinner(f"Gathering {data_key.replace('_', ' ').title()}..."):
                    try:
                        with st.expander("Agent Execution Log", expanded=True):
                            terminal_container = st.container()
                            status = terminal_container.empty()
                            status.info("Agent starting...")
                            stdout_capture = CaptureStdout(terminal_container)
                            old_stdout = sys.stdout
                            sys.stdout = stdout_capture
                            try:
                                bot = bot_class()
                                result = bot.process_satellite(satellite_name)
                                if result:
                                    session_dict[data_key] = result
                                    status.success("Agent finished successfully!")
                                    if data_manager and session_key == "satellite_data":
                                        data_manager.append_satellite_data(satellite_name, data_key, result)
                                    st.success(f"{data_key.replace('_', ' ').title()} gathered successfully!")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    status.error("Agent returned no data")
                                    st.error("No data was returned by the agent.")
                            except Exception as e:
                                status.error(f"Agent failed: {e}")
                                st.error(f"Error: {str(e)}")
                            finally:
                                sys.stdout = old_stdout
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
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
    st.markdown("### Data Type Selection", unsafe_allow_html=True)
    data_type = st.radio(
        "Choose Data Type:", 
        ["Data", "GPT"], 
        key="data_type_radio",
        horizontal=True
    )
    if data_type == "Data":
        tab1, tab2, tab3, tab4 = st.tabs([
            "Basic Info", 
            "Technical Specs", 
            "Launch & Cost", 
            "Combined Data"
        ])
        render_tab(tab1, satellite_name, "basic_info", BasicInfoBot, data_manager, session_key="satellite_data")
        render_tab(tab2, satellite_name, "technical_specs", TechAgent, data_manager, session_key="satellite_data")
        render_tab(tab3, satellite_name, "launch_cost_info", CostBot, data_manager, session_key="satellite_data")
        with tab4:
            st.markdown("<div class='data-section'>", unsafe_allow_html=True)
            st.subheader("Combined Raw Data")
            satellite_data = st.session_state.satellite_data.get(satellite_name, {})
            if any(satellite_data.values()) and any(v for v in satellite_data.values() if v):
                with st.expander("View Combined JSON", expanded=False):
                    st.json(satellite_data)
                col1, col2 = st.columns(2)
                with col1:
                    json_str = json.dumps(satellite_data, indent=2)
                    st.download_button(
                        label="Download Combined JSON",
                        data=json_str,
                        file_name=f"{satellite_name}_all_data.json",
                        mime="application/json"
                    )
                with col2:
                    if st.button("Upload to Google Sheet"):
                        with st.spinner("Uploading to Google Sheets..."):
                            combined_data = {}
                            for section in ["basic_info", "technical_specs", "launch_cost_info"]:
                                if satellite_data.get(section):
                                    combined_data.update(satellite_data[section])
                            if upload_to_gsheet(satellite_name, combined_data):
                                st.success("Data uploaded to Google Sheet!")
                            else:
                                st.error("Failed to upload data")
            else:
                st.info("No combined data available. Gather information from individual tabs first.")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "User Info", 
            "Purpose & SDG", 
            "Technical Analysis", 
            "Cost Analysis", 
            "Numeric Insights",
            "Combined GPT Data"
        ])
        render_tab(tab1, satellite_name, "user_info", UserBot, session_key="gpt_data")
        render_tab(tab2, satellite_name, "purpose_sdg", PurposeBot, session_key="gpt_data")
        render_tab(tab3, satellite_name, "tech", TechBot, session_key="gpt_data")
        render_tab(tab4, satellite_name, "frugal", FrugalBot, session_key="gpt_data")
        render_tab(tab5, satellite_name, "numeric", NumericBot, session_key="gpt_data")
        with tab6:
            st.markdown("<div class='data-section'>", unsafe_allow_html=True)
            st.subheader("Combined GPT Data")
            gpt_data = st.session_state.gpt_data.get(satellite_name, {})
            if any(gpt_data.values()) and any(v for v in gpt_data.values() if v):
                with st.expander("View Combined GPT JSON", expanded=False):
                    st.json(gpt_data)
                col1, col2 = st.columns(2)
                with col1:
                    json_str = json.dumps(gpt_data, indent=2)
                    st.download_button(
                        label="Download GPT Combined JSON",
                        data=json_str,
                        file_name=f"{satellite_name}_gpt_all_data.json",
                        mime="application/json"
                    )
                with col2:
                    if st.button("Upload GPT Data to Google Sheet"):
                        with st.spinner("Uploading GPT data to Google Sheets..."):
                            if upload_to_gsheet(satellite_name, gpt_data, "Sheet2"):
                                st.success("GPT Data uploaded to Google Sheet (Sheet2)!")
                            else:
                                st.error("Failed to upload GPT data")
            else:
                st.info("No combined GPT data available. Generate insights from individual tabs first.")
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