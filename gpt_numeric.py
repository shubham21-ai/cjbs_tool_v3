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

class NumericBot:
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
            ResponseSchema(name="return_on_investment", description="Number representing return on investment value"),
            ResponseSchema(name="data_of_revenue_from_satellite_launch_musd", description="Number representing revenue from satellite launch in million USD"),
            ResponseSchema(name="return_on_investment_description", description="String describing or explaining ROI"),
            ResponseSchema(name="return_on_investment_source", description="String containing source for ROI and revenue")
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
You are a satellite financial metrics research assistant. Your task is to find out the satellite's return on investment (ROI), revenue from launch, ROI description, and sources. Always provide a source URL or citation for each field.

IMPORTANT INSTRUCTIONS:
1. Use the available tools to search for the satellite's ROI and revenue. Look for official sources, government/agency/organization websites, news, or reputable databases.
2. When you have gathered sufficient information, use the "Complete Task" tool with the structured data.
3. If you cannot find specific information, indicate "NA" for that field.
4. Always include a source URL or citation for the ROI and revenue information.
5. NEVER use undefined tools or output None.

**Guidance for ROI and Revenue:**
- A high ROI (e.g., >1.5 or 150%) indicates the satellite generated significant value relative to its cost; a low ROI (e.g., <1 or 100%) means the investment may not have been fully recovered.
- High revenue from launch (in million USD) should be contextualized: compare to typical mission costs and industry standards.
- Always provide a brief justification or context for the ROI and revenue values, referencing comparable missions if possible.

Available tools: {tool_names}

Previous conversation:
{chat_history}

Question: {input}
Thought: I need to search for information about this satellite's ROI and revenue using the available tools.
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
        return f'"{satellite_name}" satellite ROI revenue return on investment financials'

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def _process_with_retry(self, satellite_name):
        search_query = self._create_search_query(satellite_name)
        prompt_text = f"""
Find out the ROI and revenue for the satellite: {satellite_name}

Required information to find:
1. Return on investment value
2. Revenue from satellite launch (in million USD)
3. Description or explanation of ROI
4. Source URL or citation for ROI and revenue

Steps to follow:
1. Search for "{satellite_name} ROI" or "{satellite_name} revenue" or "{satellite_name} return on investment" using Tavily Search.
2. Analyze the search results for the required information.
3. When you have gathered sufficient information, use the "Complete Task" tool with the data in this exact JSON format:

{self.format_instructions}

IMPORTANT:
- If you cannot find specific information, use "NA" for that field.
- Always include a source URL or citation for the ROI and revenue information.
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
            "return_on_investment": "NA",
            "data_of_revenue_from_satellite_launch_musd": "NA",
            "return_on_investment_description": "NA",
            "return_on_investment_source": "NA"
        }
        try:
            for step in intermediate_steps:
                if len(step) >= 2:
                    action, observation = step[0], step[1]
                    if hasattr(observation, 'lower') and isinstance(observation, str):
                        obs_lower = observation.lower()
                        # ROI
                        roi_match = re.search(r'roi[:\s]+([\d\.]+)', obs_lower)
                        if roi_match:
                            extracted_data["return_on_investment"] = float(roi_match.group(1))
                        # Revenue
                        rev_match = re.search(r'revenue[:\s]+([\d\.]+)', obs_lower)
                        if rev_match:
                            extracted_data["data_of_revenue_from_satellite_launch_musd"] = float(rev_match.group(1))
                        # ROI description
                        if "roi" in obs_lower or "return on investment" in obs_lower:
                            extracted_data["return_on_investment_description"] = observation.strip()
                        # Source link
                        url_match = re.search(r'https?://\S+', observation)
                        if url_match:
                            extracted_data["return_on_investment_source"] = url_match.group(0)
            return extracted_data
        except Exception as e:
            print(f"Error extracting data from steps: {str(e)}")
            return self._create_fallback_response("Data extraction failed", satellite_name)

    def _create_fallback_response(self, error_reason, satellite_name):
        return {
            "return_on_investment": "NA",
            "data_of_revenue_from_satellite_launch_musd": "NA",
            "return_on_investment_description": "NA",
            "return_on_investment_source": "NA",
            "error": error_reason
        }

    def process_satellite(self, satellite_name):
        try:
            print(f"Processing satellite: {satellite_name}")
            parsed_output = self._process_with_retry(satellite_name)
            if not isinstance(parsed_output, dict):
                parsed_output = {
                    "return_on_investment": "NA",
                    "data_of_revenue_from_satellite_launch_musd": "NA",
                    "return_on_investment_description": "NA",
                    "return_on_investment_source": "NA"
                }
            parsed_output["satellite_name"] = satellite_name
            return parsed_output
        except Exception as e:
            print(f"Error processing satellite {satellite_name}: {str(e)}")
            if "Resource has been exhausted" in str(e):
                print("API rate limit reached. Please try again in a few minutes.")
            return {
                "satellite_name": satellite_name,
                "return_on_investment": "NA",
                "data_of_revenue_from_satellite_launch_musd": "NA",
                "return_on_investment_description": "NA",
                "return_on_investment_source": "NA"
            }
