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


class UserBot:
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
            ResponseSchema(
                    name="user_category_number",
                    description="Integer representing user category (1: Military, 2: Civil, 3: Commercial, 4: Government, 5: Mix)"
            ),
            ResponseSchema(
                    name="user_description",
                    description="String describing the satellite user or operator"
            ),
            ResponseSchema(
                    name="user_source_link",
                    description="String containing the source URL or citation for user information"
            ),
            
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
You are a satellite user information research assistant. Your task is to find out who operates, owns, or uses the given satellite, and classify the user into one of the following categories:
1: Military
2: Civil
3: Commercial
4: Government
5: Mix (if more than one applies)

IMPORTANT INSTRUCTIONS:
1. Use the available tools to search for the satellite's user/operator/owner. Look for official sources, government/agency/organization websites, news, or reputable databases.
2. When you have gathered sufficient information, use the "Complete Task" tool with the structured data.
3. If you cannot find specific information, indicate "NA" for that field.
4. Always include a source URL or citation for the user information.
5. NEVER use undefined tools or output None.

Available tools: {tool_names}

Previous conversation:
{chat_history}

Question: {input}
Thought: I need to search for information about this satellite's user/operator/owner using the available tools.
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
        return f'"{satellite_name}" satellite launch cost vehicle date site mass success reusability mission cost'

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def _process_with_retry(self, satellite_name):
        """Process satellite information with retry logic"""
        search_query = self._create_search_query(satellite_name)
        
        prompt_text = f"""
Find out who operates, owns, or uses the satellite: {satellite_name}

Required information to find:
1. User category (1: Military, 2: Civil, 3: Commercial, 4: Government, 5: Mix)
2. Description of the user/operator/owner (e.g., 'ISRO, the Indian Space Research Organisation, operates the satellite for civil and scientific purposes.')
3. Source URL or citation for the user information

Steps to follow:
1. Search for "{satellite_name} operator" or "{satellite_name} owner" or "{satellite_name} user" using Tavily Search.
2. Analyze the search results for the required information.
3. When you have gathered sufficient information, use the "Complete Task" tool with the data in this exact JSON format:

{self.format_instructions}

IMPORTANT:
- If you cannot find specific information, use "NA" for that field.
- Always include a source URL or citation for the user information.
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
            "user_category_number": "NA",
            "user_description": "NA",
            "user_source_link": "NA"
        }
        try:
            for step in intermediate_steps:
                if len(step) >= 2:
                    action, observation = step[0], step[1]
                    if hasattr(observation, 'lower') and isinstance(observation, str):
                        obs_lower = observation.lower()
                        # Heuristic for user category
                        if "military" in obs_lower:
                            extracted_data["user_category_number"] = 1
                        elif "civil" in obs_lower:
                            extracted_data["user_category_number"] = 2
                        elif "commercial" in obs_lower:
                            extracted_data["user_category_number"] = 3
                        elif "government" in obs_lower:
                            extracted_data["user_category_number"] = 4
                        elif "mix" in obs_lower:
                            extracted_data["user_category_number"] = 5
                        # Heuristic for user description
                        if ("operated by" in obs_lower or "owned by" in obs_lower or "user" in obs_lower or "operator" in obs_lower):
                            # Try to extract the phrase after 'operated by', 'owned by', etc.
                            match = re.search(r'(operated by|owned by|user|operator)[:\s]+([^\.;\n]+)', obs_lower)
                            if match:
                                extracted_data["user_description"] = match.group(2).strip()
                            else:
                                # fallback: use the whole observation
                                extracted_data["user_description"] = observation.strip()
                        # Heuristic for source link
                        url_match = re.search(r'https?://\S+', observation)
                        if url_match:
                            extracted_data["user_source_link"] = url_match.group(0)
            return extracted_data
        except Exception as e:
            print(f"Error extracting data from steps: {str(e)}")
            return self._create_fallback_response("Data extraction failed", satellite_name)

    def _create_fallback_response(self, error_reason, satellite_name):
        """Create a fallback response when processing fails"""
        return {
            "user_category_number": "NA",
            "user_description": "NA",
            "user_source_link": "NA",
            "error" : error_reason
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
                "user_category_number": "NA",
                "user_description": "NA",
                "user_source_link": "NA"
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
                 "user_category_number": "NA",
                "user_description": "NA",
                "user_source_link": "NA",  
            }