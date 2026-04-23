"""
Modern satellite agent base using simple LLM Chains + ChatGroq.
Replaces brittle ReAct agents with a deterministic 2-step process:
1. Search web programmatically
2. Parse context with LLM into JSON
This prevents all infinite loops, tool calling errors, and hanging.
"""

import os
import json
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
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
    """Base class for all satellite data extraction agents."""

    fields: list = []

    def __init__(self):
        self.satellite_data_manager = SatelliteDataManager()
        self._setup_llm()

    def _setup_llm(self):
        self.llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            api_key=GROQ_API_KEY,
            temperature=0.1,
            max_retries=3,
        )
        self.tavily = TavilySearch(max_results=3)

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
            text = m.group(1)
        m = re.search(r"\{[\s\S]+\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None

    def _get_search_query(self, satellite_name: str) -> str:
        # Search for the fields this agent cares about to get targeted results
        keywords = " ".join([name.replace("_", " ") for name, _ in self.fields[:3]])
        return f"{satellite_name} satellite {keywords} details specifications"

    # Default fallback if a subclass doesn't override it
    def _build_prompt(self, satellite_name: str) -> str:
        field_names = ", ".join(name for name, _ in self.fields)
        return (
            f"Please extract the following fields for satellite: {satellite_name}\n"
            f"Required fields: {field_names}\n"
        )

    def _execute_prompt(self, satellite_name: str, context: str) -> str:
        # Get the subclass's custom prompt
        subclass_prompt = self._build_prompt(satellite_name)
        
        # We append the context to the prompt so the LLM uses it
        return (
            f"{subclass_prompt}\n\n"
            f"=== IMPORTANT: EXTRACT DATA ONLY FROM THIS SEARCH CONTEXT ===\n"
            f"{context}\n"
            f"=============================================================\n"
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def _run(self, satellite_name: str) -> dict:
        # Step 1: Programmatic Search
        search_query = self._get_search_query(satellite_name)
        try:
            search_results = self.tavily.invoke({"query": search_query})
            context_str = json.dumps(search_results, indent=2)
        except Exception as e:
            print(f"[{self.__class__.__name__}] Search failed: {e}")
            context_str = "No search results available."

        # Step 2: Extract with LLM
        prompt = self._execute_prompt(satellite_name, context_str)
        response = self.llm.invoke(prompt)
        
        # Step 3: Parse JSON
        output = getattr(response, "content", "")
        parsed = self._extract_json(output)
        if parsed and isinstance(parsed, dict):
            # Validate all fields are present
            for name, _ in self.fields:
                if name not in parsed:
                    parsed[name] = "NA"
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
