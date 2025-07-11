import os
import json
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from supabase import create_client
import logging
import time

# === 🔐 Set up logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === 🔐 Load env variables ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# === 🚀 Initialize clients ===
client = Groq(api_key=GROQ_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)  # type: ignore

# === 🛠️ Tool function with retry mechanism ===
def execute_query_groq(arguments, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            data = json.loads(arguments)
            sql = data.get("query")
            logging.info(f"Executing SQL query (attempt {attempt + 1}): {sql}")
            
            result = supabase.rpc("execute_sql", {"query": sql}).execute()
            result_json = json.dumps(result.data, indent=2)
            
            if not result.data:
                logging.warning("Query returned empty results")
            else:
                logging.info(f"Query results: {result_json}")
            return result_json
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Query execution failed (attempt {attempt + 1}): {error_msg}")
            
            # If this is not the last attempt, wait a bit before retrying
            if attempt < max_retries:
                logging.info(f"Retrying in 2 seconds... (attempt {attempt + 2}/{max_retries + 1})")
                time.sleep(2)
            else:
                # Final attempt failed, return error
                error_json = json.dumps({"error": error_msg})
                return error_json
    
    # This should never be reached, but just in case
    return json.dumps({"error": "Maximum retries exceeded"})

# === 🧠 Improved System prompt for First Agent (Query Generator) ===
system_prompt_agent1 = f"""
You are a hospital database query generator and medical equipment expert. You intelligently decide whether to:
When user greets no need to give like anything database word or related just greet them back nicely.
+ greet them warmly and encourage them to ask a hospital-related question.
Do not mention SQL, tables, or databases in your response.
Do not answer to the question that is not related to hospital equipment, staff, or locations.

1. **Provide medical explanations** for definition/explanation requests
2. **Query the database** for location, status, usage, or availability information  
3. **Do both** when the question requires explanation AND database information
4. **Query** the database for the questions asked by the user if it 2 or more.
5. **Identify and split multiple questions**: If the user provides multiple questions in a single input (e.g., "Where is the ventilator? What is an ECG machine?"), split them into individual questions and process each separately.

**Decision Making:**
- Analyze the user's intent, not just keywords
- "What is a ventilator?" → Medical explanation only
- "Where is the ventilator?" → Database query only
- "What is a ventilator and where is it?" → Both explanation + database query
- "What's the status of the ECG machine?" → Database query (even though it contains "what")

**For Medical Explanations:**
- Use your medical knowledge to provide clear, helpful explanations
- Keep explanations concise but informative (2-3 sentences)
- Focus on what the equipment does and why it's important

**For Database Queries:**
- ALWAYS use the 'execute_query' tool to run SQL queries
- Generate accurate SQL based on the schema provided
- DO NOT return raw JSON or mention technical details

**IMPORTANT**: 
- Never return raw JSON or tool_calls in your response
- Always be helpful and clear for hospital staff
- If you need to query the database, use the execute_query tool

### Spelling & Language Handling:
- Automatically correct typos based on hospital context
- Use contextual understanding to interpret user intent
- If a term is completely unrecognizable, politely ask for clarification

## Database Schema Information

### Tables and Relationships:
- **departments**: department_id (PK), name
- **employees**: employee_id (PK), name, role, department_id (FK → departments)
- **equipment**: equipment_id (PK), name, model, type, status, department_id (FK → departments), location_id (FK → locations), last_updated, last_used_by (FK → employees)
- **locations**: location_id (PK), building, floor, room_number, latitude, longitude
- **usage_logs**: log_id (PK), equipment_id (FK → equipment), employee_id (FK → employees), used_at, action
- Every table has a primary key (PK) and foreign keys (FK) to establish relationships. So use that to join the tables and get the results correctly.
### Key Relationships:
- Employees belong to departments (employees.department_id → departments.department_id)
- Equipment is assigned to departments (equipment.department_id → departments.department_id)
- Equipment is located at specific locations (equipment.location_id → locations.location_id)
- Equipment tracks last user (equipment.last_used_by → employees.employee_id)
- Usage logs track equipment usage by employees (usage_logs.equipment_id → equipment.equipment_id, usage_logs.employee_id → employees.employee_id)

## Query Generation Rules:

### Format and Structure:
- Return SQL queries in JSON format: {{"query": "SELECT ..."}}
- Use consistent table aliases: 
  - e for equipment
  - l for locations
  - emp for employees
  - d for departments
  - ul for usage_logs
- Join tables with full names before aliases
- Use lowercase table names to match the schema

### Text Searching:
- For ALL text/string column comparisons, ALWAYS use ILIKE with wildcards (e.g., ILIKE '%search_term%')
- NEVER use = (equals) for text comparisons unless doing exact ID matching
- This applies to columns: name, role, model, type, status, building, room_number, action

### Equipment Status Queries:
- Use the 'status' column with valid values: 'Available', 'In use', 'Maintenance'
- For availability queries, filter WHERE e.status ILIKE '%available%'
- For maintenance queries, filter WHERE e.status ILIKE '%maintenance%'

### Usage Tracking:
- For "who used" queries, join usage_logs with employees
- Do NOT use equipment.last_used_by for usage queries - use usage_logs table instead
- For recent usage, prefer ul.action ILIKE '%checked out%' or ul.action ILIKE '%used%'
- Order by ul.used_at DESC for most recent usage

### Location Queries:
- For location-based searches, join equipment with locations
- Use building, floor, and room_number columns for location filtering
- For "where is" queries, include location details in SELECT

### Common Query Patterns:
- Equipment search: SELECT e.name, e.model, e.type, e.status FROM equipment e WHERE e.name ILIKE '%search_term%'
- Location search: SELECT e.name, l.building, l.floor, l.room_number FROM equipment e JOIN locations l ON e.location_id = l.location_id WHERE l.building ILIKE '%building_name%'
- Usage history: SELECT emp.name, ul.used_at, ul.action FROM usage_logs ul JOIN employees emp ON ul.employee_id = emp.employee_id WHERE ul.equipment_id = (SELECT equipment_id FROM equipment WHERE name ILIKE '%equipment_name%') ORDER BY ul.used_at DESC

Always ensure queries are syntactically correct and use proper relationships.
"""

# === 🧠 System prompt for Second Agent (Result Explanation) ===
system_prompt_agent2 = """
You are a hospital assistant that explains database query results in plain English.
When user greets no need to give like anything database word or related just greet them back nicely. 
+ greet them warmly and encourage them to ask a hospital-related question.
Take a list of query results (from execute_query tool) and/or direct explanations from the first agent and combine them into a single, clear response.
Ensure consistency across answers, avoiding conflicting information
Do not answer to the question that is not related to hospital equipment, staff, or locations. If user asks about something that is not related to hospital equipment, staff, or locations, just say in your words that what you are.
- Do not mention SQL, tables, or databases in your response.

Your job is to:
- Take SQL query results and explain them in 1-3 concise, helpful sentences
- Use hospital-friendly, non-technical language
- Interpret status/action values clearly (e.g., "Available" = "ready to use")
- If results are empty, say: "Sorry, we couldn't find that information. Try a different search term."
- If there's an error, say: "Sorry, there was an issue getting that information. Please try rephrasing your question."
- For urgent/emergency equipment, start with "Urgent:" and be direct
- NEVER mention SQL, tables, databases, or technical details
- Focus on what hospital staff need to know

Examples:
- "The ventilator is located in Building A, Floor 2, Room 205."
- "The ECG machine was last used by Dr. Smith on January 15th at 2:30 PM."
- "The defibrillator is currently available and ready to use."
"""

# === Tool config for Groq ===
tools = [{
    "type": "function",
    "function": {
        "name": "execute_query",
        "description": "Executes a SQL query on the hospital database",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query to execute"
                }
            },
            "required": ["query"]
        }
    }
}]

# === Function to create chat completion with retry ===
def create_chat_completion_with_retry(messages, tools=None, tool_choice="auto", max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            logging.info(f"Attempting chat completion (attempt {attempt + 1})")
            
            run = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Updated to best model
                messages=messages,
                temperature=0.7,
                tools=tools,
                tool_choice=tool_choice # type: ignore
            )
            
            return run
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Chat completion failed (attempt {attempt + 1}): {error_msg}")
            
            # If this is not the last attempt, wait a bit before retrying
            if attempt < max_retries:
                logging.info(f"Retrying chat completion in 2 seconds... (attempt {attempt + 2}/{max_retries + 1})")
                time.sleep(2)
            else:
                # Final attempt failed, raise the exception
                raise e
    
    # This should never be reached, but just in case
    raise Exception("Maximum retries exceeded for chat completion")

# === Streamlit UI ===
st.set_page_config(
    page_title="🏥 Hospital Assistant",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for chat UI
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .stChatMessage.user {
        background-color: #e6f3ff;
        text-align: right;
    }
    .stChatMessage.assistant {
        background-color: #f0f0f0;
        text-align: left;
    }
    .stChatInputContainer {
        position: sticky;
        bottom: 0;
        background-color: white;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display title and instructions
st.markdown("""
    <h1 style='text-align: center;'>🏥 Hospital Assistant</h1>
    <p style='text-align: center;'>Ask about equipment, staff, locations, or usage info.</p>
    <p style='text-align: center; font-size: 0.9em;'>e.g., "Where is the ECG machine?", "Who used the ventilator last?", "What is a defibrillator?"</p>
""", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type your question (e.g., Where is the ventilator?)")

if user_input and user_input.strip():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("🤖 Thinking..."):
        try:
            # messages_agent1 = [
            #     {"role": "system", "content": system_prompt_agent1},
            #     {"role": "user", "content": user_input}
            # ]
            
            # Build full conversation history for agent1
            messages_agent1 = [{"role": "system", "content": system_prompt_agent1}]
            for msg in st.session_state.messages:
                messages_agent1.append({"role": msg["role"], "content": msg["content"]})

            

            # Let the model decide whether to use tools or not
            run = create_chat_completion_with_retry(messages_agent1, tools=tools, tool_choice="auto")
            response = run.choices[0].message
            logging.info(f"Raw response: {response.content!r}")

            # === 🔍 STEP 1: Check if tool calls are present
            has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls

            # === 🟢 STEP 2: If NO tool calls and there's a message, just show it directly
            if not has_tool_calls and response.content:
                final_response = response.content.strip().capitalize()
                logging.info(f"🟢 Direct reply from Agent 1 (no tool call): {final_response}")
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                with st.chat_message("assistant"):
                    st.markdown(final_response)
                    # st.success("✅ Done!")
            else:
                # === ⚙️ STEP 3: If there are tool calls, execute them
                query_results = []
                final_responses = []

                if has_tool_calls:
                    for idx, tool_call in enumerate(response.tool_calls, 1): # type: ignore
                        if tool_call.function.name == "execute_query":
                            tool_output = execute_query_groq(tool_call.function.arguments)
                            logging.info(f"Tool call {idx} executed: {tool_call.function.name} → {tool_call.function.arguments}")
                            query_results.append({"index": idx, "result": tool_output})
                        else:
                            final_responses.append(f"{idx}. ⚠️ Couldn't process part of your request. Try rephrasing.")

                # Collect direct assistant content (e.g., explanation text)
                if response.content:
                    if response.content.strip().startswith("1"):
                        final_responses.extend(response.content.strip().split("\n"))
                    else:
                        final_responses.append(response.content)

                # === 🤖 STEP 4: Send to Agent 2 (if needed)
                if query_results or final_responses:
                    messages_agent2 = [{"role": "system", "content": system_prompt_agent2}]
                    for msg in st.session_state.messages:
                        messages_agent2.append({"role": msg["role"], "content": msg["content"]})

                    for query_result in query_results:
                        messages_agent2.append({
                            "role": "function",
                            "name": "execute_query",
                            "content": query_result["result"]
                        })

                    if final_responses and not all("⚠️" in r for r in final_responses):
                        messages_agent2.append({
                            "role": "assistant",
                            "content": "\n".join(final_responses)
                        })

                    explanation_run = create_chat_completion_with_retry(messages_agent2, tools=None, tool_choice="auto")
                    final_response = explanation_run.choices[0].message.content
                else:
                    final_response = "⚠️ Sorry, I couldn't understand. Try rephrasing."

                # === 🧾 STEP 5: Final output from Agent 2
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                with st.chat_message("assistant"):
                    st.markdown(final_response)
                    # st.success("✅ Done!")
        except Exception as e:
            logging.error(f"Final error after all retries: {str(e)}")
            error_msg = "⚠️ Sorry, there was an issue processing your request. Please try again in a moment."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.markdown(error_msg)
