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

class TechBot:
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
            ResponseSchema(name="hardware", description="String describing the main hardware or platform used by the satellite"),
            ResponseSchema(name="sensors", description="String describing the sensors or payloads onboard the satellite"),
            ResponseSchema(name="breakthrough_tech", description="String describing any breakthrough or innovative technology used"),
            ResponseSchema(name="tech_source_link", description="String containing the source URL or citation for technical information")
        ]

    def _initialize_parser(self):
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schema)
        self.format_instructions = self.output_parser.get_format_instructions()

    def _initialize_agent(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0.1,
            max_retries=3,
            timeout=120
        )
        agent_prompt = """
You are a satellite technical specifications research assistant. Your task is to find out the main hardware/platform, sensors/payloads, and any breakthrough technology used in the given satellite. Always provide a source URL or citation for each field.

IMPORTANT INSTRUCTIONS:
1. Use the available tools to search for the satellite's technical specifications. Look for official sources, government/agency/organization websites, news, or reputable databases.
2. When you have gathered sufficient information, use the "Complete Task" tool with the structured data.
3. If you cannot find specific information, indicate "NA" for that field.
4. Always include a source URL or citation for the technical information.
5. NEVER use undefined tools or output None.

Available tools: {tool_names}

Previous conversation:
{chat_history}

Question: {input}
Thought: I need to search for information about this satellite's technical specifications using the available tools.
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
        return f'"{satellite_name}" satellite hardware sensors payloads breakthrough technology'

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def _process_with_retry(self, satellite_name):
        search_query = self._create_search_query(satellite_name)
        prompt_text = f"""
Find out the main technical specifications for the satellite: {satellite_name}

Required information to find:
1. Main hardware/platform used
2. Sensors or payloads onboard
3. Any breakthrough or innovative technology used
4. Source URL or citation for technical information

Steps to follow:
1. Search for "{satellite_name} hardware" or "{satellite_name} sensors" or "{satellite_name} payloads" using Tavily Search.
2. Analyze the search results for the required information.
3. When you have gathered sufficient information, use the "Complete Task" tool with the data in this exact JSON format:

{self.format_instructions}

IMPORTANT:
- If you cannot find specific information, use "NA" for that field.
- Always include a source URL or citation for the technical information.
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
            "hardware": "NA",
            "sensors": "NA",
            "breakthrough_tech": "NA",
            "tech_source_link": "NA"
        }
        try:
            for step in intermediate_steps:
                if len(step) >= 2:
                    action, observation = step[0], step[1]
                    if hasattr(observation, 'lower') and isinstance(observation, str):
                        obs_lower = observation.lower()
                        # Hardware
                        if "hardware" in obs_lower or "platform" in obs_lower:
                            match = re.search(r'(hardware|platform)[:\s]+([^\.;\n]+)', obs_lower)
                            if match:
                                extracted_data["hardware"] = match.group(2).strip()
                            else:
                                extracted_data["hardware"] = observation.strip()
                        # Sensors
                        if "sensor" in obs_lower or "payload" in obs_lower:
                            match = re.search(r'(sensor[s]?|payload[s]?)[:\s]+([^\.;\n]+)', obs_lower)
                            if match:
                                extracted_data["sensors"] = match.group(2).strip()
                            else:
                                extracted_data["sensors"] = observation.strip()
                        # Breakthrough tech
                        if "breakthrough" in obs_lower or "innovative" in obs_lower or "technology" in obs_lower:
                            match = re.search(r'(breakthrough|innovative|technology)[:\s]+([^\.;\n]+)', obs_lower)
                            if match:
                                extracted_data["breakthrough_tech"] = match.group(2).strip()
                            else:
                                extracted_data["breakthrough_tech"] = observation.strip()
                        # Source link
                        url_match = re.search(r'https?://\S+', observation)
                        if url_match:
                            extracted_data["tech_source_link"] = url_match.group(0)
            return extracted_data
        except Exception as e:
            print(f"Error extracting data from steps: {str(e)}")
            return self._create_fallback_response("Data extraction failed", satellite_name)

    def _create_fallback_response(self, error_reason, satellite_name):
        return {
            "hardware": "NA",
            "sensors": "NA",
            "breakthrough_tech": "NA",
            "tech_source_link": "NA",
            "error": error_reason
        }

    def process_satellite(self, satellite_name):
        try:
            print(f"Processing satellite: {satellite_name}")
            parsed_output = self._process_with_retry(satellite_name)
            if not isinstance(parsed_output, dict):
                parsed_output = {
                    "hardware": "NA",
                    "sensors": "NA",
                    "breakthrough_tech": "NA",
                    "tech_source_link": "NA"
                }
            parsed_output["satellite_name"] = satellite_name
            return parsed_output
        except Exception as e:
            print(f"Error processing satellite {satellite_name}: {str(e)}")
            if "Resource has been exhausted" in str(e):
                print("API rate limit reached. Please try again in a few minutes.")
            return {
                "satellite_name": satellite_name,
                "hardware": "NA",
                "sensors": "NA",
                "breakthrough_tech": "NA",
                "tech_source_link": "NA"
            }
