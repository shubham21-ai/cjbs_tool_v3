from langchain.agents import initialize_agent, AgentType, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from data_manager import SatelliteDataManager
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SerpAPIWrapper

import os
from dotenv import load_dotenv
import time
import json
from tenacity import retry, stop_after_attempt, wait_exponential


# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY


class BasicInfoBot:
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
                # Parse the input as JSON if it's a string
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
            ResponseSchema(name="altitude", description="Orbital altitude in kilometers"),
            ResponseSchema(name="altitude_source", description="Source URL for altitude information"),
            ResponseSchema(name="orbital_life_years", description="Orbital lifetime in years"),
            ResponseSchema(name="orbital_life_source", description="Source URL for orbital lifetime information"),
            ResponseSchema(name="launch_orbit_classification", description="Orbit classification (LEO, MEO, GEO, etc.)"),
            ResponseSchema(name="orbit_classification_source", description="Source URL for orbit classification information"),
            ResponseSchema(name="number_of_payloads", description="Number of payloads on the satellite"),
            ResponseSchema(name="payloads_source", description="Source URL for payload information")
        ]

    def _initialize_parser(self):
        """Initialize the output parser"""
        self.output_parser = StructuredOutputParser.from_response_schemas(self.response_schema)
        self.format_instructions = self.output_parser.get_format_instructions()

    def _initialize_agent(self):
        """Initialize the agent"""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=GOOGLE_API_KEY,
            temperature=0.1,  # Lower temperature for more consistent output
            max_retries=5,
            timeout=120
        )
        
        # Create a custom prompt for the agent
        agent_prompt = """
You are a satellite information research assistant. Your task is to find comprehensive information about satellites using the available tools.
First try to find the datas from the https://nextspaceflight.com/ if the data is not available then try to find data from other websites , articles , news , press releases , parliament report and other resources as well.

IMPORTANT INSTRUCTIONS:
1. Use the available tools to search for satellite information , try to find the data from Next Spaceflight , if your are not able to get data from their then try to find other websites , articles , news , press releases , parliament report and other resources as well.
2. When you have gathered sufficient information, use the "Complete Task" tool with the structured data
3. NEVER try to use "None" or any undefined tools
4. If you cannot find specific information, indicate "Not available" for that field
5. Always include source URLs when available

Available tools: {tool_names}

Previous conversation:
{chat_history}

Question: {input}
Thought: I need to search for information about this satellite using the available tools.
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
        return f'"{satellite_name}" satellite orbital altitude orbit classification payload launch specifications'

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def _process_with_retry(self, satellite_name):
        """Process satellite information with retry logic"""
        search_query = self._create_search_query(satellite_name)
        
        prompt_text = f"""
Find comprehensive information about the satellite: {satellite_name}
First try to find the datas from the https://nextspaceflight.com/ if the data is not available then try to find data from other websites , articles , news , press releases , parliament report and other resources as well.

Required information to find:
1. Orbital altitude in kilometers (perigee/apogee or average)
2. Orbital lifetime in years (operational or design life)  
3. Orbit classification (LEO, MEO, GEO, etc.)
4. Number of payloads

Steps to follow:
1. First, search for "{satellite_name}" using Tavily Search
2. Analyze the search results for the required information
3. If needed, perform additional searches for specific details
4. When you have gathered sufficient information, use the "Complete Task" tool with the data in this exact JSON format:

{self.format_instructions}

IMPORTANT: 
- You have a maximum of 10 actions. Use them efficiently.
- If you cannot find specific information, use "NA" for that field
- If you're running out of iterations, prioritize the most critical information and use the Complete Task tool
- Always include source URLs when data is available, otherwise use "NA"

Remember: Use "NA" for any information that cannot be found or verified.
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
            "altitude": "NA",
            "altitude_source": "NA",
            "orbital_life_years": "NA",
            "orbital_life_source": "NA",
            "launch_orbit_classification": "NA",
            "orbit_classification_source": "NA",
            "number_of_payloads": "NA",
            "payloads_source": "NA"
        }
        
        try:
            # Look through intermediate steps for any useful data
            for step in intermediate_steps:
                if len(step) >= 2:
                    action, observation = step[0], step[1]
                    if hasattr(observation, 'lower') and isinstance(observation, str):
                        obs_lower = observation.lower()
                        
                        # Try to extract altitude information
                        if 'altitude' in obs_lower or 'km' in obs_lower:
                            if extracted_data['altitude'] == "NA":
                                extracted_data['altitude'] = "Partial"
                        
                        # Try to extract orbit classification
                        if any(orbit_type in obs_lower for orbit_type in ['leo', 'meo', 'geo', 'orbit']):
                            if extracted_data['launch_orbit_classification'] == "NA":
                                extracted_data['launch_orbit_classification'] = "Partial"
            
            print(f"⚠️  Extracted partial data from {len(intermediate_steps)} completed steps")
            
        except Exception as e:
            print(f"Error extracting data from steps: {str(e)}")
        
        return extracted_data

    def _create_fallback_response(self, error_reason, satellite_name):
        """Create a fallback response when agent limits are reached"""
        return {
            "altitude": "NA",
            "altitude_source": "NA",
            "orbital_life_years": "NA",
            "orbital_life_source": "NA",
            "launch_orbit_classification": "NA",
            "orbit_classification_source": "NA",
            "number_of_payloads": "NA",
            "payloads_source": "NA"
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
                    "altitude": "NA",
                    "altitude_source": "NA",
                    "orbital_life_years": "NA",
                    "orbital_life_source": "NA",
                    "launch_orbit_classification": "NA", 
                    "orbit_classification_source": "NA",
                    "number_of_payloads": "NA",
                    "payloads_source": "NA"
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
                "altitude": "NA",
                "altitude_source": "NA",
                "orbital_life_years": "NA",
                "orbital_life_source": "NA",
                "launch_orbit_classification": "NA",
                "orbit_classification_source": "NA", 
                "number_of_payloads": "NA",
                "payloads_source": "NA"
            }