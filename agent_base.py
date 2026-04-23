"""
Modern satellite agent base using LangGraph + ChatGroq.
Uses langgraph.prebuilt.create_react_agent (the correct modern API for langchain v1.x).
"""
import os
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from tenacity import retry, stop_after_attempt, wait_exponential
from data_manager import SatelliteDataManager

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


class SatelliteAgentBase:
    """Base class for all satellite data extraction agents (LangGraph v1.x compatible)."""

    # Subclasses define as list of (field_name, description) tuples
    fields: list = []

    def __init__(self):
        self.satellite_data_manager = SatelliteDataManager()
        self._setup_agent()

    def _setup_agent(self):
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            api_key=GROQ_API_KEY,
            temperature=0.1,
            max_retries=5,
        )

        tavily_base = TavilySearch(max_results=3)
        
        @tool
        def search_web(query: str) -> str:
            """Search the web for satellite information. Input must be a simple text query."""
            return tavily_base.invoke({"query": query})

        @tool
        def complete_task(json_data: str) -> str:
            """Call this tool when you have collected all information.
            Input: a valid JSON string with all required fields. Use 'NA' for missing fields."""
            try:
                return json.dumps(json.loads(json_data), indent=2)
            except Exception:
                return json_data

        self.agent = create_react_agent(
            model=llm,
            tools=[search_web, complete_task],
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _json_schema(self) -> str:
        lines = ["{"]
        for name, desc in self.fields:
            lines.append(f'    "{name}": "<{desc}>",')
        lines.append("}")
        return "\n".join(lines)

    def _fallback_data(self) -> dict:
        return {name: "NA" for name, _ in self.fields}

    def _extract_json(self, text: str):
        m = re.search(r"```json\s*([\s\S]+?)\s*```", text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        m = re.search(r"\{[\s\S]+\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------ #
    #  Prompt builder                                                       #
    # ------------------------------------------------------------------ #
    def _build_prompt(self, satellite_name: str) -> str:
        field_names = ", ".join(name for name, _ in self.fields)
        return (
            f"Find the following information about the satellite: {satellite_name}\n\n"
            f"Required fields: {field_names}\n\n"
            "Search nextspaceflight.com first, then Wikipedia, ESA, NASA, or other reputable sources.\n\n"
            "When you have gathered all the data, call the 'complete_task' tool with a valid JSON string "
            "exactly matching this structure (use 'NA' for any field you cannot find):\n\n"
            f"{self._json_schema()}"
        )

    # ------------------------------------------------------------------ #
    #  Execution                                                            #
    # ------------------------------------------------------------------ #
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30), reraise=True)
    def _run(self, satellite_name: str) -> dict:
        prompt = self._build_prompt(satellite_name)
        result = self.agent.invoke({"messages": [("human", prompt)]})

        # Extract the last AI message content
        messages = result.get("messages", [])
        output = ""
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if content and isinstance(content, str) and len(content) > 10:
                output = content
                break

        parsed = self._extract_json(output)
        if parsed and isinstance(parsed, dict):
            return parsed

        # Also check tool call results
        for msg in messages:
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                parsed = self._extract_json(content)
                if parsed and isinstance(parsed, dict) and len(parsed) > 1:
                    return parsed

        print(f"[{self.__class__.__name__}] Could not parse JSON – using fallback.")
        return self._fallback_data()

    def process_satellite(self, satellite_name: str) -> dict:
        print(f"[{self.__class__.__name__}] Processing: {satellite_name}")
        try:
            result = self._run(satellite_name)
            if not isinstance(result, dict):
                result = self._fallback_data()
            result["satellite_name"] = satellite_name
            return result
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error: {e}")
            data = self._fallback_data()
            data["satellite_name"] = satellite_name
            data["error"] = str(e)
            return data
