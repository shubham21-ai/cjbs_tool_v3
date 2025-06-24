from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from data_manager import SatelliteDataManager
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import time
import json
from tenacity import retry, stop_after_attempt, wait_exponential
import re

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

class PurposeBot:
    def __init__(self):
        self.satellite_data_manager = SatelliteDataManager()
        self._initialize_tools()
        self._initialize_schema()
        self._initialize_parser()
        self._initialize_agent()

    def _initialize_tools(self):
        def complete_task(input_data):
            try:
                if isinstance(input_data, str):
                    try:
                        parsed_data = json.loads(input_data)
                        return json.dumps(parsed_data, indent=2)
                    except json.JSONDecodeError:
                        return input_data
                return json.dumps(input_data, indent=2)
            except Exception as e:
                return f"Task completed with data: {input_data}"
        self.tools = [
            Tool(
                name="Satellite Data Manager",
                func=self.satellite_data_manager.get_satellite_data,
                description="Useful for getting satellite data based on the user's query.",
            ),
            Tool(
                name="Tavily Search",
                func=TavilySearchResults(max_results=5).run,
                description="Useful for getting information from the web. Returns search results with URLs and content.",
            ),
            Tool(
                name="Complete Task",
                func=complete_task,
                description="Use this tool when you have gathered all necessary information and want to provide the final structured output. Input should be the complete satellite information in JSON format."
            )
        ]

    def _initialize_schema(self):
        self.response_schema = [
            ResponseSchema(name="purpose", description="Integer representing purpose (1: Communications, 2: Earth Observation, 3: Navigation, 4: Space Science, 5: Technology Development)"),
            ResponseSchema(name="purpose_category_number", description="Integer representing purpose category number (same as purpose)"),
            ResponseSchema(name="purpose_description", description="String describing the satellite's purpose"),
            ResponseSchema(name="purpose_source_link", description="String containing the source URL or citation for purpose"),
            ResponseSchema(name="sdg_category", description="Integer representing SDG category (1: Economic, 2: Social, 3: Environmental, 4: Innovation)"),
            ResponseSchema(name="sdg_category_identification_numbers", description="Array of integers representing SDG numbers (e.g., [13, 15])"),
            ResponseSchema(name="sdg_description", description="String describing SDGs served"),
            ResponseSchema(name="sdg_source_link", description="String containing the source URL or citation for SDG classification")
        ]

    def _initialize_parser(self):
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schema)
        self.format_instructions = self.output_parser.get_format_instructions()

    def _initialize_agent(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0.1,
            max_retries=3,
            timeout=120
        )
        agent_prompt = """
You are a satellite purpose and SDG mapping research assistant. Your task is to find out the main purpose of the given satellite, classify it (1: Communications, 2: Earth Observation, 3: Navigation, 4: Space Science, 5: Technology Development), and map it to the relevant SDG category (1: Economic, 2: Social, 3: Environmental, 4: Innovation), SDG numbers, and SDG description. Always provide a source URL or citation for each field.

IMPORTANT INSTRUCTIONS:
1. Use the available tools to search for the satellite's purpose and SDG mapping. Look for official sources, government/agency/organization websites, news, or reputable databases.
2. When you have gathered sufficient information, use the "Complete Task" tool with the structured data.
3. If you cannot find specific information, indicate "NA" for that field.
4. Always include a source URL or citation for the purpose and SDG information.
5. NEVER use undefined tools or output None.

Available tools: {tool_names}

Previous conversation:
{chat_history}

Question: {input}
Thought: I need to search for information about this satellite's purpose and SDG mapping using the available tools.
{agent_scratchpad}
"""
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=10,
            max_execution_time=300,
            early_stopping_method="generate",
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

    def _create_search_query(self, satellite_name):
        return f'"{satellite_name}" satellite purpose SDG application mission objective'

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def _process_with_retry(self, satellite_name):
        search_query = self._create_search_query(satellite_name)
        prompt_text = f"""
Find out the main purpose and SDG mapping for the satellite: {satellite_name}

Required information to find:
1. Purpose (1: Communications, 2: Earth Observation, 3: Navigation, 4: Space Science, 5: Technology Development)
2. Purpose category number (same as purpose)
3. Description of the satellite's purpose
4. Source URL or citation for the purpose
5. SDG category (1: Economic, 2: Social, 3: Environmental, 4: Innovation)
6. SDG numbers (e.g., [13, 15])
7. SDG description
8. Source URL or citation for SDG mapping

Steps to follow:
1. Search for "{satellite_name} purpose" or "{satellite_name} SDG" using Tavily Search.
2. Analyze the search results for the required information.
3. When you have gathered sufficient information, use the "Complete Task" tool with the data in this exact JSON format:

{self.format_instructions}

IMPORTANT:
- If you cannot find specific information, use "NA" for that field.
- Always include a source URL or citation for the purpose and SDG information.
- Use "NA" for any information that cannot be found or verified.
"""
        try:
            response = self.agent.invoke({"input": prompt_text})
            if isinstance(response, dict):
                output = response.get("output", "")
                intermediate_steps = response.get("intermediate_steps", [])
                if len(intermediate_steps) >= 10:
                    print("⚠️  Agent reached maximum iterations (10). Processing available data...")
                    return self._extract_data_from_steps(intermediate_steps, satellite_name)
                if "```json" in output:
                    json_start = output.find("```json") + 7
                    json_end = output.find("```", json_start)
                    json_str = output[json_start:json_end].strip()
                    return json.loads(json_str)
                elif output.startswith("{"):
                    try:
                        return json.loads(output)
                    except json.JSONDecodeError:
                        pass
                try:
                    return self.output_parser.parse(output)
                except Exception:
                    if intermediate_steps:
                        return self._extract_data_from_steps(intermediate_steps, satellite_name)
                    return self._create_fallback_response("Parsing failed", satellite_name)
            return response
        except Exception as e:
            error_msg = str(e)
            print(f"Error in agent processing: {error_msg}")
            if "maximum iterations" in error_msg.lower():
                print("⚠️  Agent reached maximum iterations limit")
                return self._create_fallback_response("Max iterations reached", satellite_name)
            elif "timeout" in error_msg.lower() or "execution time" in error_msg.lower():
                print("⚠️  Agent execution timeout")
                return self._create_fallback_response("Execution timeout", satellite_name)
            else:
                return self._create_fallback_response(f"Error: {error_msg}", satellite_name)

    def _extract_data_from_steps(self, intermediate_steps, satellite_name):
        extracted_data = {
            "purpose": "NA",
            "purpose_category_number": "NA",
            "purpose_description": "NA",
            "purpose_source_link": "NA",
            "sdg_category": "NA",
            "sdg_category_identification_numbers": "NA",
            "sdg_description": "NA",
            "sdg_source_link": "NA"
        }
        try:
            for step in intermediate_steps:
                if len(step) >= 2:
                    action, observation = step[0], step[1]
                    if hasattr(observation, 'lower') and isinstance(observation, str):
                        obs_lower = observation.lower()
                        # Heuristics for purpose
                        if "communication" in obs_lower:
                            extracted_data["purpose"] = 1
                            extracted_data["purpose_category_number"] = 1
                        elif "earth observation" in obs_lower:
                            extracted_data["purpose"] = 2
                            extracted_data["purpose_category_number"] = 2
                        elif "navigation" in obs_lower:
                            extracted_data["purpose"] = 3
                            extracted_data["purpose_category_number"] = 3
                        elif "space science" in obs_lower:
                            extracted_data["purpose"] = 4
                            extracted_data["purpose_category_number"] = 4
                        elif "technology development" in obs_lower:
                            extracted_data["purpose"] = 5
                            extracted_data["purpose_category_number"] = 5
                        # Purpose description
                        if "purpose" in obs_lower or "mission" in obs_lower or "application" in obs_lower:
                            match = re.search(r'(purpose|mission|application)[:\s]+([^\.;\n]+)', obs_lower)
                            if match:
                                extracted_data["purpose_description"] = match.group(2).strip()
                            else:
                                extracted_data["purpose_description"] = observation.strip()
                        # SDG category
                        if "economic" in obs_lower:
                            extracted_data["sdg_category"] = 1
                        elif "social" in obs_lower:
                            extracted_data["sdg_category"] = 2
                        elif "environmental" in obs_lower:
                            extracted_data["sdg_category"] = 3
                        elif "innovation" in obs_lower:
                            extracted_data["sdg_category"] = 4
                        # SDG numbers
                        sdg_nums = re.findall(r'sdg\s*(\d+)', obs_lower)
                        if sdg_nums:
                            extracted_data["sdg_category_identification_numbers"] = [int(n) for n in sdg_nums]
                        # SDG description
                        if "sdg" in obs_lower:
                            match = re.search(r'sdg[s]?[:\s]+([^\.;\n]+)', obs_lower)
                            if match:
                                extracted_data["sdg_description"] = match.group(1).strip()
                        # Source links
                        url_match = re.search(r'https?://\S+', observation)
                        if url_match:
                            if extracted_data["purpose_source_link"] == "NA":
                                extracted_data["purpose_source_link"] = url_match.group(0)
                            else:
                                extracted_data["sdg_source_link"] = url_match.group(0)
            return extracted_data
        except Exception as e:
            print(f"Error extracting data from steps: {str(e)}")
            return self._create_fallback_response("Data extraction failed", satellite_name)

    def _create_fallback_response(self, error_reason, satellite_name):
        return {
            "purpose": "NA",
            "purpose_category_number": "NA",
            "purpose_description": "NA",
            "purpose_source_link": "NA",
            "sdg_category": "NA",
            "sdg_category_identification_numbers": "NA",
            "sdg_description": "NA",
            "sdg_source_link": "NA",
            "error": error_reason
        }

    def process_satellite(self, satellite_name):
        try:
            print(f"Processing satellite: {satellite_name}")
            parsed_output = self._process_with_retry(satellite_name)
            if not isinstance(parsed_output, dict):
                parsed_output = {
                    "purpose": "NA",
                    "purpose_category_number": "NA",
                    "purpose_description": "NA",
                    "purpose_source_link": "NA",
                    "sdg_category": "NA",
                    "sdg_category_identification_numbers": "NA",
                    "sdg_description": "NA",
                    "sdg_source_link": "NA"
                }
            parsed_output["satellite_name"] = satellite_name
            return parsed_output
        except Exception as e:
            print(f"Error processing satellite {satellite_name}: {str(e)}")
            if "Resource has been exhausted" in str(e):
                print("API rate limit reached. Please try again in a few minutes.")
            return {
                "satellite_name": satellite_name,
                "purpose": "NA",
                "purpose_category_number": "NA",
                "purpose_description": "NA",
                "purpose_source_link": "NA",
                "sdg_category": "NA",
                "sdg_category_identification_numbers": "NA",
                "sdg_description": "NA",
                "sdg_source_link": "NA"
            }
