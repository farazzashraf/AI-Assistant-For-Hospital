import os
import json
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from supabase import create_client
import logging
import time
import tempfile
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import threading
import queue

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

# === 🎵 Text-to-Speech using Groq ===
def groq_text_to_speech(text, voice="Judy-PlayAI", model="playai-tts"):
    """Convert text to speech using Groq TTS API"""
    try:
        # Clean text for better speech output
        clean_text = text.replace('*', '').replace('_', '').replace('#', '')
        clean_text = ''.join(char for char in clean_text if char.isalnum() or char.isspace() or char in '.,!?-')
        
        # Remove emojis and special characters
        clean_text = ''.join(char for char in clean_text if ord(char) < 128)
        
        # Limit text length to avoid API limits
        if len(clean_text) > 1000:
            clean_text = clean_text[:1000] + "..."
        
        logging.info(f"Generating speech for: {clean_text[:50]}...")
        
        # Create speech using Groq API
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=clean_text,
            response_format="wav"
        )
        
        audio_bytes = response.read()  # ✅ This reads the binary audio
        return audio_bytes

        
    except Exception as e:
        logging.error(f"Groq TTS failed: {str(e)}")
        return None

# === 🎙️ Voice-to-Text using Groq Whisper ===
def transcribe_audio(audio_bytes):
    """Transcribe audio using Groq Whisper model"""
    try:
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        # Transcribe using Groq Whisper
        with open(tmp_file_path, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                response_format="text"
            )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return str(transcription).strip()
    
    except Exception as e:
        logging.error(f"Voice transcription failed: {str(e)}")
        return None

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

# === 🎨 Streamlit UI Setup ===
st.set_page_config(
    page_title="🏥 Hospital Voice Assistant",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
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
    .voice-controls {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 10px;
        margin: 10px 0;
    }
    .voice-status {
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .recording {
        background-color: #ff4444;
        color: white;
        animation: pulse 1s infinite;
    }
    .ready {
        background-color: #44ff44;
        color: black;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = True
if "selected_voice" not in st.session_state:
    st.session_state.selected_voice = "Judy-PlayAI"
# if "input" not in st.session_state:
#     st.session_state.input_type = None


# Display title and instructions
st.markdown("""
    <h1 style='text-align: center;'>🏥 Hospital Voice Assistant</h1>
    <p style='text-align: center;'>Ask about equipment, staff, locations, or usage info using voice or text.</p>
    <p style='text-align: center; font-size: 0.9em;'>🎤 Click the microphone to speak | 💬 Type your message</p>
""", unsafe_allow_html=True)

# Voice controls
# col1, col2, col3 = st.columns([1, 2, 1])

with st.sidebar:
    # Voice toggle with clear indication
    voice_enabled = st.checkbox("🔊 Enable Voice Response (Assistant will speak back)", value=st.session_state.voice_enabled)
    st.session_state.voice_enabled = voice_enabled
    
    if voice_enabled:
        # Voice selection
        voice_options = {
            "Judy-PlayAI": "🎭 Judy-PlayAI (Neutral)",
            "Basil-PlayAI": "Basil-PlayAI", 
            "Celeste-PlayAI": "Celeste-PlayAI",
            "Chip-PlayAI": "Chip-PlayAI",
            "Mitch-PlayAI": "Mitch-PlayAI",
            "Jennifer-PlayAI": "Jennifer-PlayAI"
        }
        
        selected_voice = st.selectbox(
            "🎤 Choose Voice:",
            options=list(voice_options.keys()),
            format_func=lambda x: voice_options[x],
            index=list(voice_options.keys()).index(st.session_state.selected_voice)
        )
        st.session_state.selected_voice = selected_voice
        
        st.success(f"✅ Voice response is ON - Using {voice_options[selected_voice]}")
    else:
        st.info("ℹ️ Voice response is OFF - Text only")

# === 🧹 Clean audio input after processing ===
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to play audio response
def play_audio_response(text):
    """Generate and play audio response using Groq TTS"""
    if st.session_state.voice_enabled:
        with st.spinner("🔊 Generating voice response..."):
            try:
                audio_content = groq_text_to_speech(text, voice=st.session_state.selected_voice)
                if audio_content:
                    # Convert to base64 for HTML audio player
                    audio_b64 = base64.b64encode(audio_content).decode()
                    audio_html = f"""
                    <audio controls autoplay style="width: 100%;">
                        <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                    """
                    st.markdown(audio_html, unsafe_allow_html=True)
                else:
                    st.warning("🔇 Could not generate voice response")
            except Exception as e:
                logging.error(f"Audio playback failed: {str(e)}")
                st.warning("🔇 Voice response unavailable")
                
input_container = st.container()
with input_container:
    col1, col2 = st.columns([1, 10])
    with col1:
        st.write("")  # for vertical alignment
        st.write("")  # for vertical alignment
        audio_bytes = audio_recorder(
            text="",  # no text
            recording_color="#ff4444",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=2.0,
            sample_rate=16000,
            # key="audio_recorder"
        )

# Process voice input
# if audio_bytes and len(audio_bytes) > 0:
if audio_bytes and len(audio_bytes) > 0 and audio_bytes != st.session_state.last_audio:
    # Store current audio to prevent reprocessing
    st.session_state.last_audio = audio_bytes
    st.markdown("### 🎧 Processing Voice...")
    
    with st.spinner("🎙️ Converting speech to text..."):
        transcribed_text = transcribe_audio(audio_bytes)
    
    if transcribed_text:
        st.success(f"🎯 Heard: \"{transcribed_text}\"")
        user_input = transcribed_text
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(f"🎤 {user_input}")
        
        # Process the transcribed input
        with st.spinner("🤖 Thinking..."):
            try:
                # Build full conversation history for agent1
                messages_agent1 = [{"role": "system", "content": system_prompt_agent1}]
                for msg in st.session_state.messages:
                    messages_agent1.append({"role": msg["role"], "content": msg["content"]})
                messages_agent1.append({"role": "user", "content": user_input})

                # Let the model decide whether to use tools or not
                run = create_chat_completion_with_retry(messages_agent1, tools=tools, tool_choice="auto")
                response = run.choices[0].message
                logging.info(f"Raw response: {response.content!r}")

                # Check if tool calls are present
                has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls

                # If NO tool calls and there's a message, just show it directly
                if not has_tool_calls and response.content:
                    final_response = response.content.strip()
                    logging.info(f"🟢 Direct reply from Agent 1 (no tool call): {final_response}")
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    with st.chat_message("assistant"):
                        st.markdown(final_response)
                    
                    # Play audio response
                    play_audio_response(final_response)
                            
                else:
                    # If there are tool calls, execute them
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

                    # Collect direct assistant content
                    if response.content:
                        if response.content.strip().startswith("1"):
                            final_responses.extend(response.content.strip().split("\n"))
                        else:
                            final_responses.append(response.content)

                    # Send to Agent 2 (if needed)
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

                    # Final output from Agent 2
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    with st.chat_message("assistant"):
                        st.markdown(final_response)
                    
                    # Play audio response
                    play_audio_response(final_response)
                            
            except Exception as e:
                logging.error(f"Final error after all retries: {str(e)}")
                error_msg = "⚠️ Sorry, there was an issue processing your request. Please try again in a moment."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
                
                # Play error audio response
                play_audio_response(error_msg)
    
    else:
        st.error("❌ Could not transcribe audio. Please try again.")

# Text input (still available as fallback)
st.caption("### 💬 Or Type Your Message")
user_input = st.chat_input("Type your question (e.g., Where is the ventilator?)")

if user_input and user_input.strip():
    # Prevent processing if we just handled voice
    if "last_text" not in st.session_state:
        st.session_state.last_text = ""
        
    if user_input != st.session_state.last_text:
        st.session_state.last_text = user_input
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("🤖 Thinking..."):
            try:
                # Build full conversation history for agent1
                messages_agent1 = [{"role": "system", "content": system_prompt_agent1}]
                for msg in st.session_state.messages:
                    messages_agent1.append({"role": msg["role"], "content": msg["content"]})
                messages_agent1.append({"role": "user", "content": user_input})  # Add current input

                # Let the model decide whether to use tools or not
                run = create_chat_completion_with_retry(messages_agent1, tools=tools, tool_choice="auto")
                response = run.choices[0].message
                logging.info(f"Raw response: {response.content!r}")

                # Check if tool calls are present
                has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls

                # If NO tool calls and there's a message, just show it directly
                if not has_tool_calls and response.content:
                    final_response = response.content.strip()
                    logging.info(f"🟢 Direct reply from Agent 1 (no tool call): {final_response}")
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    with st.chat_message("assistant"):
                        st.markdown(final_response)
                    
                    # Play audio response
                    play_audio_response(final_response)
                            
                else:
                    # If there are tool calls, execute them
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

                    # Collect direct assistant content
                    if response.content:
                        if response.content.strip().startswith("1"):
                            final_responses.extend(response.content.strip().split("\n"))
                        else:
                            final_responses.append(response.content)

                    # Send to Agent 2 (if needed)
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

                    # Final output from Agent 2
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    with st.chat_message("assistant"):
                        st.markdown(final_response)
                    
                    # Play audio response
                    play_audio_response(final_response)
                            
            except Exception as e:
                logging.error(f"Final error after all retries: {str(e)}")
                error_msg = "⚠️ Sorry, there was an issue processing your request. Please try again in a moment."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
                
                # Play error audio response
                play_audio_response(error_msg)

# Footer
with st.sidebar:
    st.markdown("---")
    st.markdown("🎤 **Voice Commands**: Speak naturally about hospital equipment, staff, or locations")
    st.markdown("🔊 **Voice Response**: Toggle on/off to hear responses spoken aloud with selectable voices")
    st.markdown("💬 **Text Input**: Type your questions as an alternative to voice commands")
    st.markdown("🎭 **Voice Options**: Choose from 6 different AI voices for personalized experience")