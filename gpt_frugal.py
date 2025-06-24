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


class FrugalBot:
    def __init__(self):
        self.satellite_data_manager = SatelliteDataManager()
        self._initialize_tools()
        self._initialize_schema()
        self._initialize_parser()
        self._initialize_agent()

    def _initialize_tools(self):
        """Initialize the tools for the agent"""
        def complete_task(input_data):
            """Tool to complete the task and return structured output"""
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
        """Initialize the response schema"""
        self.response_schema = [
            ResponseSchema(name="frugal", description="String enum (YES/NO) indicating if the satellite is frugal"),
            ResponseSchema(name="development_cost_efficiency", description="Integer (0: No, 1: Yes) indicating development cost efficiency"),
            ResponseSchema(name="development_cost_efficiency_description", description="String explaining development cost efficiency"),
            ResponseSchema(name="development_cost_efficiency_source", description="String containing source for development cost efficiency"),
            ResponseSchema(name="operational_cost_efficiency", description="Integer (0: No, 1: Yes) indicating operational cost efficiency"),
            ResponseSchema(name="operational_cost_efficiency_description", description="String explaining operational cost efficiency"),
            ResponseSchema(name="operational_cost_efficiency_source", description="String containing source for operational cost efficiency"),
            ResponseSchema(name="labour_cost_efficiency", description="Integer (0: No, 1: Yes) indicating labour cost efficiency"),
            ResponseSchema(name="labour_cost_efficiency_description", description="String explaining labour cost efficiency"),
            ResponseSchema(name="labour_cost_efficiency_source", description="String containing source for labour cost efficiency"),
            ResponseSchema(name="frugal_innovation_design", description="Integer (0: No, 1: Yes) indicating frugal innovation design"),
            ResponseSchema(name="frugal_innovation_design_description", description="String explaining frugal innovation design"),
            ResponseSchema(name="frugal_innovation_design_source", description="String containing source for frugal innovation design")
        ]

    def _initialize_parser(self):
        """Initialize the output parser"""
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schema)
        self.format_instructions = self.output_parser.get_format_instructions()

    def _initialize_agent(self):
        """Initialize the agent"""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0.1,  # Lower temperature for more consistent output
            max_retries=3,
            timeout=120
        )
        
        # Create a custom prompt for the agent
        agent_prompt = """
You are a satellite frugality and cost-efficiency research assistant. Your task is to find out if the satellite is frugal, and provide details on development, operational, labour cost efficiency, frugal innovation design, and sources. Always provide a source URL or citation for each field.

IMPORTANT INSTRUCTIONS:
1. Use the available tools to search for the satellite's frugality and cost-efficiency. Look for official sources, government/agency/organization websites, news, or reputable databases.
2. When you have gathered sufficient information, use the "Complete Task" tool with the structured data.
3. If you cannot find specific information, indicate "NA" for that field.
4. Always include a source URL or citation for the frugality and cost-efficiency information.
5. NEVER use undefined tools or output None.

Available tools: {tool_names}

Previous conversation:
{chat_history}

Question: {input}
Thought: I need to search for information about this satellite's frugality and cost-efficiency using the available tools.
{agent_scratchpad}
"""

        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=10,  # Limit iterations to prevent infinite loops
            max_execution_time=300,  # 5 minute timeout
            early_stopping_method="generate",  # Allow early stopping
            handle_parsing_errors=True,  # Handle parsing errors gracefully
            return_intermediate_steps=True
        )

    def _create_search_query(self, satellite_name):
        """Create an effective search query"""
        return f'"{satellite_name}" satellite frugal cost efficiency innovation design'

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def _process_with_retry(self, satellite_name):
        """Process satellite information with retry logic"""
        search_query = self._create_search_query(satellite_name)
        
        prompt_text = f"""
Find out if the satellite is frugal and provide details on cost-efficiency: {satellite_name}

Required information to find:
1. Is the satellite frugal? (YES/NO)
2. Development cost efficiency (0/1) and description
3. Operational cost efficiency (0/1) and description
4. Labour cost efficiency (0/1) and description
5. Frugal innovation design (0/1) and description
6. Source URL or citation for each field

Steps to follow:
1. Search for "{satellite_name} frugal" or "{satellite_name} cost efficiency" or "{satellite_name} innovation design" using Tavily Search.
2. Analyze the search results for the required information.
3. When you have gathered sufficient information, use the "Complete Task" tool with the data in this exact JSON format:

{self.format_instructions}

IMPORTANT:
- If you cannot find specific information, use "NA" for that field.
- Always include a source URL or citation for the frugality and cost-efficiency information.
- Use "NA" for any information that cannot be found or verified.
"""
        
        try:
            response = self.agent.invoke({
                "input": prompt_text
            })
            
            # Check if response indicates agent limits were reached
            if isinstance(response, dict):
                output = response.get("output", "")
                intermediate_steps = response.get("intermediate_steps", [])
                
                # Check if agent was stopped due to limits
                if len(intermediate_steps) >= 10:
                    print("⚠️  Agent reached maximum iterations (10). Processing available data...")
                    # Try to extract any useful information from intermediate steps
                    return self._extract_data_from_steps(intermediate_steps, satellite_name)
                
                # Normal processing
                if "```json" in output:
                    # Extract JSON from markdown code block
                    json_start = output.find("```json") + 7
                    json_end = output.find("```", json_start)
                    json_str = output[json_start:json_end].strip()
                    return json.loads(json_str)
                elif output.startswith("{"):
                    # Direct JSON output
                    try:
                        return json.loads(output)
                    except json.JSONDecodeError:
                        pass
                
                # Try to parse with the structured parser
                try:
                    return self.output_parser.parse(output)
                except Exception:
                    # If parsing fails but we have intermediate steps, try to extract data
                    if intermediate_steps:
                        return self._extract_data_from_steps(intermediate_steps, satellite_name)
                    
                    # Final fallback
                    return self._create_fallback_response("Parsing failed", satellite_name)
            
            return response
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error in agent processing: {error_msg}")
            
            # Handle specific limit-related errors
            if "maximum iterations" in error_msg.lower():
                print("⚠️  Agent reached maximum iterations limit")
                return self._create_fallback_response("Max iterations reached", satellite_name)
            elif "timeout" in error_msg.lower() or "execution time" in error_msg.lower():
                print("⚠️  Agent execution timeout")
                return self._create_fallback_response("Execution timeout", satellite_name)
            else:
                return self._create_fallback_response(f"Error: {error_msg}", satellite_name)

    def _extract_data_from_steps(self, intermediate_steps, satellite_name):
        """Extract available data from intermediate steps when agent limits are reached"""
        extracted_data = {
            "frugal": "NA",
            "development_cost_efficiency": "NA",
            "development_cost_efficiency_description": "NA",
            "development_cost_efficiency_source": "NA",
            "operational_cost_efficiency": "NA",
            "operational_cost_efficiency_description": "NA",
            "operational_cost_efficiency_source": "NA",
            "labour_cost_efficiency": "NA",
            "labour_cost_efficiency_description": "NA",
            "labour_cost_efficiency_source": "NA",
            "frugal_innovation_design": "NA",
            "frugal_innovation_design_description": "NA",
            "frugal_innovation_design_source": "NA"
        }
        
        try:
            # Look through intermediate steps for any useful data
            for step in intermediate_steps:
                if len(step) >= 2:
                    action, observation = step[0], step[1]
                    if hasattr(observation, 'lower') and isinstance(observation, str):
                        obs_lower = observation.lower()
                        
                        # Frugal
                        if "frugal" in obs_lower:
                            if "yes" in obs_lower:
                                extracted_data["frugal"] = "YES"
                            elif "no" in obs_lower:
                                extracted_data["frugal"] = "NO"
                        # Development cost efficiency
                        if "development cost" in obs_lower:
                            if "efficient" in obs_lower or "low cost" in obs_lower:
                                extracted_data["development_cost_efficiency"] = 1
                                extracted_data["development_cost_efficiency_description"] = observation.strip()
                            else:
                                extracted_data["development_cost_efficiency"] = 0
                        # Operational cost efficiency
                        if "operational cost" in obs_lower:
                            if "efficient" in obs_lower or "low cost" in obs_lower:
                                extracted_data["operational_cost_efficiency"] = 1
                                extracted_data["operational_cost_efficiency_description"] = observation.strip()
                            else:
                                extracted_data["operational_cost_efficiency"] = 0
                        # Labour cost efficiency
                        if "labour cost" in obs_lower or "labor cost" in obs_lower:
                            if "efficient" in obs_lower or "low cost" in obs_lower:
                                extracted_data["labour_cost_efficiency"] = 1
                                extracted_data["labour_cost_efficiency_description"] = observation.strip()
                            else:
                                extracted_data["labour_cost_efficiency"] = 0
                        # Frugal innovation design
                        if "frugal innovation" in obs_lower or "indigenous" in obs_lower or "reuse" in obs_lower:
                            extracted_data["frugal_innovation_design"] = 1
                            extracted_data["frugal_innovation_design_description"] = observation.strip()
                        # Source links
                        url_match = re.search(r'https?://\S+', observation)
                        if url_match:
                            if extracted_data["development_cost_efficiency_source"] == "NA":
                                extracted_data["development_cost_efficiency_source"] = url_match.group(0)
                            elif extracted_data["operational_cost_efficiency_source"] == "NA":
                                extracted_data["operational_cost_efficiency_source"] = url_match.group(0)
                            elif extracted_data["labour_cost_efficiency_source"] == "NA":
                                extracted_data["labour_cost_efficiency_source"] = url_match.group(0)
                            elif extracted_data["frugal_innovation_design_source"] == "NA":
                                extracted_data["frugal_innovation_design_source"] = url_match.group(0)
            
            return extracted_data
            
        except Exception as e:
            print(f"Error extracting data from steps: {str(e)}")
            return self._create_fallback_response("Data extraction failed", satellite_name)

    def _create_fallback_response(self, error_reason, satellite_name):
        """Create a fallback response when processing fails"""
        return {
            "frugal": "NA",
            "development_cost_efficiency": "NA",
            "development_cost_efficiency_description": "NA",
            "development_cost_efficiency_source": "NA",
            "operational_cost_efficiency": "NA",
            "operational_cost_efficiency_description": "NA",
            "operational_cost_efficiency_source": "NA",
            "labour_cost_efficiency": "NA",
            "labour_cost_efficiency_description": "NA",
            "labour_cost_efficiency_source": "NA",
            "frugal_innovation_design": "NA",
            "frugal_innovation_design_description": "NA",
            "frugal_innovation_design_source": "NA",
            "error": error_reason
        }

    def process_satellite(self, satellite_name):
        """Process satellite information and return parsed output"""
        try:
            print(f"Processing satellite: {satellite_name}")
            
            # Process with retry logic
            parsed_output = self._process_with_retry(satellite_name)
            
            # Ensure parsed_output is a dictionary
            if not isinstance(parsed_output, dict):
                parsed_output = {
                    "frugal": "NA",
                    "development_cost_efficiency": "NA",
                    "development_cost_efficiency_description": "NA",
                    "development_cost_efficiency_source": "NA",
                    "operational_cost_efficiency": "NA",
                    "operational_cost_efficiency_description": "NA",
                    "operational_cost_efficiency_source": "NA",
                    "labour_cost_efficiency": "NA",
                    "labour_cost_efficiency_description": "NA",
                    "labour_cost_efficiency_source": "NA",
                    "frugal_innovation_design": "NA",
                    "frugal_innovation_design_description": "NA",
                    "frugal_innovation_design_source": "NA"
                }
            
            # Add the satellite name to the output
            parsed_output["satellite_name"] = satellite_name
            
            return parsed_output
            
        except Exception as e:
            print(f"Error processing satellite {satellite_name}: {str(e)}")
            if "Resource has been exhausted" in str(e):
                print("API rate limit reached. Please try again in a few minutes.")
            
            # Return error structure
            return {
                "satellite_name": satellite_name,
                "frugal": "NA",
                "development_cost_efficiency": "NA",
                "development_cost_efficiency_description": "NA",
                "development_cost_efficiency_source": "NA",
                "operational_cost_efficiency": "NA",
                "operational_cost_efficiency_description": "NA",
                "operational_cost_efficiency_source": "NA",
                "labour_cost_efficiency": "NA",
                "labour_cost_efficiency_description": "NA",
                "labour_cost_efficiency_source": "NA",
                "frugal_innovation_design": "NA",
                "frugal_innovation_design_description": "NA",
                "frugal_innovation_design_source": "NA"
            }